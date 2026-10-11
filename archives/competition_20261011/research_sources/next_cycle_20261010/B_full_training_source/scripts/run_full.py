"""One confirmed official FULL20 arm, existing canonical queue and exact-last resume."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,uuid,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p1_contract import REMOTE_ROOT,sha,verify_files,verify_remote_assets,read
from full_route_runtime import RUNTIME_ENV,bind_budget
from aictrack1.engine.training_queue import QueueItem,run_serial_queue,read_training_lock
from owned_launch import start_with_go,cleanup,owned_group,snapshot

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    if args.smoke and (args.resume or args.stop_after_epoch is not None):parser.error('smoke is separate from candidate resume/pause')
    if args.stop_after_epoch is not None and not 1<=args.stop_after_epoch<20:parser.error('pause1..19 retains original cosine20')
    if os.name!='posix' or ROOT.resolve()!=Path(REMOTE_ROOT).resolve():raise RuntimeError('root-owned exact dedicated full training root required')
    os.environ.update(RUNTIME_ENV) # Before any evidence or trainer Torch import.
    verify_files();payload=verify_remote_assets();budget=bind_budget(ROOT);arm=payload['selected_arm'];attempt=uuid.uuid4().hex
    from capacity_contract import resource_ok
    usage=shutil.disk_usage(ROOT);resource_ok(budget,usage.free,usage.total)
    out=f'artifacts/smoke/{arm}_{attempt}' if args.smoke else f'artifacts/experiments/FULL_{arm}'
    item=QueueItem(arm,f'configs/experiment/p1_full_{arm}.yaml',out,'one positively confirmed fresh official FULL20 fixed final',
        trainer_entrypoint='scripts/train_n1_peft.py',resume_checkpoint=f'{out}/last.pt' if args.resume else None)
    def launch(command,cwd,handle,on_start):
        usage=shutil.disk_usage(ROOT);resource_ok(budget,usage.free,usage.total)
        record=read_training_lock(payload['canonical_lock'])
        if not record or record.queue_pid!=os.getpid():raise RuntimeError('full launcher lease missing')
        if args.smoke:command=tuple(command)+('--p1-smoke',)
        env=os.environ.copy();env.update(RUNTIME_ENV);env.update(P1_SOURCE_SHA=sha(ROOT/'source_manifest.json'),P1_LEASE_TOKEN=record.token,
            P1_DEADLINE=str(budget['deadline_unix']),P1_INITIAL_FREE=str(budget['adjusted_initial_free_bytes']),P1_DISK_MAX=str(budget['disk_max_bytes']),
            P1_MIN_FREE=str(budget['min_free_bytes']),P1_ATTEMPT_ID=attempt,P1_SMOKE_DIR=out,P1_ARM=arm,P1_RESUME=str(int(args.resume)),P1_SMOKE=str(int(args.smoke)))
        if args.stop_after_epoch is None:env.pop('P1_STOP_AFTER_EPOCH',None)
        else:env['P1_STOP_AFTER_EPOCH']=str(args.stop_after_epoch)
        folder=ROOT/f'artifacts/launches/{arm}_{attempt}';folder.mkdir(parents=True,exist_ok=False);child=None;owner=None
        def write(path,v):
            with path.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
        def created(c,v):
            nonlocal child,owner
            child=c;owner=v
        def unconfirmed(c,v,go,error,closure):write(folder/'HALT.json',dict(status='closure_unconfirmed_lease_retained',pid=c.pid,owner=v,GO_attempted=go,error=repr(error),closure_error=repr(closure),lease_token=record.token,test_access=False))
        def ready(c,v):write(folder/'START.json',dict(owner=v,command=list(command),source_manifest_sha256=env['P1_SOURCE_SHA'],config_sha256=sha(ROOT/item.config),lease_token=record.token,test_access=False))
        child,owner=start_with_go(command,cwd,handle,env,[],on_start,ready,created,unconfirmed)
        try:
            code=child.wait(timeout=max(.1,budget['deadline_unix']-time.time()))
            end=time.monotonic()+20
            while owned_group(snapshot(),owner,[]) and time.monotonic()<end:child.poll();time.sleep(.2)
            if owned_group(snapshot(),owner,[]):raise RuntimeError('B FULL exact owned trainer group still alive')
            write(folder/'EXIT.json',dict(owner=owner,return_code=code,group_closed=True,test_access=False));return code
        except BaseException as failure:
            try:cleanup(child,owner,[])
            except BaseException as closure:
                try:unconfirmed(child,owner,True,failure,closure)
                finally:os._exit(70)
            raise
    code=run_serial_queue(root=ROOT,python=Path(sys.executable),logs=ROOT/f'artifacts/queues/FULL_{arm}',queue_name=f'20261010-B-full-{arm}',items=[item],lock_path=Path(payload['canonical_lock']),launcher=launch)
    result=dict(schema='aictrack1.fresh_route.full.launch_result.v1',arm=arm,selected_route=payload['selected_route'],return_code=code,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),config_sha256=sha(ROOT/item.config),fresh_confirmation_receipt_sha256=payload['fresh_confirmation_receipt_sha256'],
        full_budget_sha256=payload['full_budget_sha256'],synthetic_smoke=args.smoke,resumed=args.resume,test_access=False)
    path=ROOT/f'artifacts/queues/full_attempt_{attempt}.json'
    with path.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    if args.stop_after_epoch is not None:
        screen=ROOT/out/'p1_screen_receipt.json'
        if screen.exists():
            v=read(screen)
            if v.get('attempt_id')==attempt and v.get('epoch')==args.stop_after_epoch and v.get('config_sha256')==sha(ROOT/item.config) and v.get('last_sha256')==sha(ROOT/out/'last.pt'):
                print(json.dumps(dict(status='planned_pause_verified_not_complete',arm=arm,epoch=args.stop_after_epoch)));return 0
    print(json.dumps(result));return code
if __name__=='__main__':raise SystemExit(main())
