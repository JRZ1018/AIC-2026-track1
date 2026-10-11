"""Explicit P4 follow-up queue; approved configs only, fixed package cutoff."""
import argparse,json,subprocess,sys,time,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aictrack1.engine.training_queue import QueueItem,run_serial_queue
END=datetime(2026,9,27,14,0,tzinfo=ZoneInfo('Asia/Shanghai'))

def launch(command,cwd,log,attach):
    if datetime.now(ZoneInfo('Asia/Shanghai'))>=END:raise TimeoutError('P4 training cutoff reached')
    proc=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    attach(proc.pid)
    while proc.poll() is None:
        if datetime.now(ZoneInfo('Asia/Shanghai'))>=END:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=30)
            return proc.returncode
        time.sleep(5)
    return proc.returncode

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',action='append',required=True);p.add_argument('--queue',required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    if a.resume and len(a.config)!=1:raise ValueError('resume requires one explicit config')
    if not a.queue.startswith('20260927-p4-') or '/' in a.queue or '\\' in a.queue:raise ValueError('invalid P4 queue identity')
    if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('requires 3GiB free')
    items=[]
    for name in a.config:
        path=(ROOT/name).resolve()
        if path.parent!=ROOT/'configs/experiment' or not path.name.startswith('stage2_p4_'):raise ValueError('requires a preregistered P4 config')
        ex=yaml.safe_load(path.read_text(encoding='utf-8'))['experiment']
        if ex.get('execution_enabled') is not True:raise RuntimeError('gate has not enabled this config')
        out=(path.parent/ex['output_dir']).resolve()
        if not out.is_relative_to(ROOT/'artifacts/experiments'):raise ValueError('unsafe output')
        trainer='scripts/train_frozen_head.py' if '-h1-' in ex['run_id'] else 'scripts/train_peft.py'
        items.append(QueueItem(tag=ex['run_id'],config=path.relative_to(ROOT).as_posix(),output_dir=out.relative_to(ROOT).as_posix(),purpose='P4 explicit gated follow-up',trainer_entrypoint=trainer,resume_checkpoint=str(a.resume) if a.resume else None))
    return run_serial_queue(root=ROOT,python=Path(sys.executable),logs=ROOT/'artifacts/queues'/a.queue,queue_name=a.queue,items=items,launcher=launch)

if __name__=='__main__':raise SystemExit(main())
