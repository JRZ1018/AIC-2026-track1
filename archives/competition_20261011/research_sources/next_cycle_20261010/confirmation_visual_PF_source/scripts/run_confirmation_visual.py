"""Fresh J5 then one admitted PF30 under canonical queue; continuous20, exact-last resume."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p1_contract import ARMS,REMOTE_ROOT,assets,sha,verify_files,verify_remote_assets,require
from fresh_route_runtime import RUNTIME_ENV,budget,resource_check
from aictrack1.engine.training_queue import QueueItem,read_training_lock,run_serial_queue
from owned_launch import start_with_go,cleanup,owned_group,snapshot

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group();m.add_argument('--preflight',action='store_true');m.add_argument('--smoke',choices=ARMS)
    p.add_argument('--resume-arms',nargs='*',default=[],choices=ARMS);a=p.parse_args()
    if (a.preflight or a.smoke) and a.resume_arms:p.error('preflight/smoke cannot resume candidates')
    require(os.name=='posix' and ROOT.resolve()==REMOTE_ROOT.resolve(),'root-only dedicated deployed joint source')
    verify_files();payload=verify_remote_assets();b=resource_check(ROOT)
    require(set(a.resume_arms).issubset(payload['arms']) and (a.smoke is None or a.smoke in payload['arms']),'only bound selected candidate may run/resume/smoke')
    if a.preflight:
        from mainline_h1_contract import load_new_fold_head
        from p1_contract import MAPPING_SHA
        head,state=load_new_fold_head(payload['assets']['h1']['path'],feature_dim=512,num_classes=750,expected_clip_weight_sha256=payload['assets']['clip']['sha256'],expected_class_mapping_sha256=MAPPING_SHA,expected_split_sha256=payload['assets']['split']['sha256'])
        write(ROOT/f'artifacts/preflight/{uuid.uuid4().hex}.json',dict(complete=True,source_manifest_sha256=sha(ROOT/'source_manifest.json'),actual_h1_epoch=state['epoch'],h1_model_tensors=len(head.state_dict()),budget=b,test_access=False,GPU_started=False))
        return 0
    attempt=uuid.uuid4().hex;order=[a.smoke] if a.smoke else payload["arms"];items=[]
    for arm in order:
        out=f'artifacts/smoke/{arm}_{attempt}' if a.smoke else f'artifacts/experiments/CONFIRM_{arm}'
        items.append(QueueItem(tag=arm,config=f'configs/experiment/p1_{arm}.yaml',output_dir=out,purpose='new610111 independent confirmation J5 versus one PF30 candidate fixedlast20',trainer_entrypoint='scripts/train_n1_peft.py',resume_checkpoint=f'{out}/last.pt' if arm in a.resume_arms else None))
    def launch(command,cwd,log,on_start):
        arm=Path(command[3]).stem.removeprefix('p1_');record=read_training_lock(payload['canonical_lock']);require(record and record.queue_pid==os.getpid(),'exact canonical controller lease')
        env=os.environ.copy();env.update(RUNTIME_ENV);env.update(P1_SOURCE_SHA=sha(ROOT/'source_manifest.json'),P1_LEASE_TOKEN=record.token,P1_DEADLINE=str(b['deadline_unix']),P1_INITIAL_FREE=str(b['adjusted_initial_free_bytes']),P1_DISK_MAX=str(b['disk_max_bytes']),P1_MIN_FREE=str(b['min_free_bytes']),P1_ATTEMPT_ID=uuid.uuid4().hex,P1_ARM=arm,P1_RESUME=str(int('--resume' in command)),P1_SMOKE=str(int(bool(a.smoke))))
        env.pop('P1_STOP_AFTER_EPOCH',None);env.pop('P1_SMOKE_DIR',None)
        if a.smoke:command=tuple(command)+('--p1-smoke',);env['P1_SMOKE_DIR']=f'artifacts/smoke/{arm}_{attempt}'
        folder=ROOT/f'artifacts/launches/{arm}_{env["P1_ATTEMPT_ID"]}';folder.mkdir(parents=True,exist_ok=False);owner=None;child=None
        def created(c,v):
            nonlocal child,owner
            child=c;owner=v
        def unconfirmed(c,v,go,start_error,closure_error):
            write(folder/'HALT.json',dict(status='closure_unconfirmed_lease_retained',pid=c.pid,owner=v,GO_attempted=go,start_error=repr(start_error),closure_error=repr(closure_error),lease_token=record.token,test_access=False))
        def ready(c,v):write(folder/'START.json',dict(owner=v,command=list(command),source_manifest_sha256=env['P1_SOURCE_SHA'],config_sha256=sha(ROOT/command[3]),lease_token=record.token,test_access=False))
        child,owner=start_with_go(command,cwd,log,env,[],on_start,ready,created,unconfirmed)
        try:
            code=child.wait(timeout=max(.1,b['deadline_unix']-time.time()))
            end=time.monotonic()+20
            while owned_group(snapshot(),owner,[]) and time.monotonic()<end:child.poll();time.sleep(.2)
            require(not owned_group(snapshot(),owner,[]),'owned trainer session still active')
            write(folder/'EXIT.json',dict(owner=owner,return_code=code,group_closed=True,test_access=False));return code
        except BaseException as failure:
            try:
                cleanup(child,owner,[])
                require(not owned_group(snapshot(),owner,[]),'owned cleanup not closed')
            except BaseException as closure_error:
                try:unconfirmed(child,owner,True,failure,closure_error)
                finally:os._exit(70)
            raise
    return run_serial_queue(root=ROOT,python=Path(sys.executable),logs=ROOT/f'artifacts/queues/{attempt}',queue_name='20261010-independent-confirmation-J5-plus-selected',items=items,lock_path=Path(payload['canonical_lock']),launcher=launch)
if __name__=='__main__':raise SystemExit(main())
