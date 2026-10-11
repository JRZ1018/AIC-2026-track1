"""Verbatim reviewed retry-CONT ownership/GO lifecycle; generic command supplied by joint runner."""
from __future__ import annotations
import os,signal,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def require(value, message):
    if not value: raise ValueError(message)

def parse_stat(text):
    fields = text.rsplit(') ', 1)[1].split(); require(len(fields) >= 20, 'short proc stat')
    return dict(pid=int(text.split(' (', 1)[0]), state=fields[0], ppid=int(fields[1]),
                pgrp=int(fields[2]), session=int(fields[3]), start_ticks=fields[19])

def read_pid_stat(pid, proc=Path('/proc')):
    try: raw = (Path(proc)/str(pid)/'stat').read_text(encoding='ascii')
    except (FileNotFoundError, ProcessLookupError): return None
    require(len(raw) <= 8192, 'oversize proc stat'); value = parse_stat(raw)
    require(value['pid'] == pid, 'proc PID mismatch'); return value

def snapshot(proc=Path('/proc')):
    names = [p for p in Path(proc).iterdir() if p.name.isdigit()]; require(len(names) <= 16384, 'bounded proc inventory exceeded')
    values = {}
    for name in names:
        value = read_pid_stat(int(name.name), proc)
        if value is not None: values[value['pid']] = value
    return values

def own_identity(value, pid, watched):
    require(pid not in {w['pid'] for w in watched} and value['pid'] == value['pgrp'] == value['session'] == pid,
            'only the newly created trainer session may be managed')
    return {k: value[k] for k in ('pid', 'pgrp', 'session', 'start_ticks')}

def owned_group(values, owner, watched):
    require(owner['pid'] not in {w['pid'] for w in watched} and owner['pid'] == owner['pgrp'] == owner['session'], 'cannot signal prior watched sessions')
    head = values.get(owner['pid'])
    if head is not None: require(own_identity(head, owner['pid'], watched) == owner, 'new trainer PID reused; refuse signals')
    members = [v for v in values.values() if v['pgrp'] == owner['pgrp'] or v['session'] == owner['session']]
    require(all(v['pgrp'] == owner['pgrp'] and v['session'] == owner['session'] and int(v['start_ticks']) >= int(owner['start_ticks']) for v in members), 'ambiguous owned trainer group')
    return members

def capture_owner(child, watched, *, proc=Path('/proc'), sleep=time.sleep, monotonic=time.monotonic):
    end = monotonic()+5
    while True:
        value = read_pid_stat(child.pid, proc)
        if value and value['pgrp'] == child.pid and value['session'] == child.pid: return own_identity(value, child.pid, watched)
        require(child.poll() is None and monotonic() < end, 'new owned trainer identity unavailable'); sleep(.05)

def cleanup(child, owner, watched, *, scan=snapshot, kill=getattr(os, 'killpg', None), sleep=time.sleep):
    require(owner is not None, 'unverified new trainer: preserve canonical lease for root inspection')
    for sig, count in [(signal.SIGINT, 40), (signal.SIGTERM, 20), (getattr(signal, 'SIGKILL', 9), 20)]:
        if not owned_group(scan(), owner, watched): break
        require(kill is not None, 'POSIX session signalling required'); kill(owner['pid'], sig)
        for _ in range(count):
            child.poll()
            if not owned_group(scan(), owner, watched): break
            sleep(.5)
    require(not owned_group(scan(), owner, watched), 'new trainer group still active; canonical lease must remain')
    return child.wait(timeout=5)

def close_go(child):
    """EOF closes an ungranted gate without ever entering the original trainer."""
    stream = getattr(child, 'stdin', None)
    if stream is not None and not stream.closed:
        stream.close()

def start_with_go(command, cwd, log, env, watched, on_start, on_ready, on_created, on_unconfirmed,
                  *, popen=subprocess.Popen, capture=capture_owner, stop=cleanup,
                  group_scan=snapshot):
    """Create one owned session; only a successfully attached lease grants GO.

    Injected callbacks exist solely for CPU lifecycle tests. Production passes
    the exact original command and preserves its parent PID through os.exec.
    """
    gate_command = (str(command[0]), str(HERE/'launch_gate.py'), '--', *command)
    child = popen(gate_command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                  stdin=subprocess.PIPE, env=env, start_new_session=True)
    owner = None; GO_attempted = False
    try:
        on_created(child, None) # Record every actual Popen before capture can fail.
        owner = capture(child, watched)
        on_created(child, owner)
        on_start(child.pid)
        on_ready(child, owner)
        require(child.poll() is None, 'owned gate exited before GO')
        require(child.stdin is not None, 'owned GO pipe missing')
        GO_attempted = True # Conservative even if write/flush fails.
        child.stdin.write(b'G'); child.stdin.flush(); close_go(child)
        return child, owner
    except BaseException as start_error:
        try:
            close_go(child)
            if owner is not None:
                stop(child, owner, watched)
            else:
                # No GO was sent. Reap the EOF-only gate; only a successful
                # complete inventory can prove the unattached session closed.
                child.wait(timeout=5)
                require(not any(v['pgrp'] == child.pid or v['session'] == child.pid
                                for v in group_scan().values()),
                        'unverified gate session remains; original trainer never granted GO')
        except BaseException as closure_error:
            try:
                on_unconfirmed(child, owner, GO_attempted, start_error, closure_error)
            finally:
                os._exit(70) # Also fail closed if the evidence callback itself fails.
        raise
