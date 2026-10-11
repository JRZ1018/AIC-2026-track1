"""Root-owned H1100; unchanged canonical queue and reviewed ownership/GO."""
from __future__ import annotations
import argparse,json,os,sys,time,uuid
from pathlib import Path
for k,v in {'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}.items():os.environ[k]=v
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from confirmation_contract import verify_source,deployment,sha,read_json,validate_budget,require,resource_check
from aictrack1.engine.training_queue import QueueItem,run_serial_queue,read_training_lock
from owned_launch import start_with_go,cleanup,owned_group,snapshot

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    require(not(a.resume and a.preflight),'preflight cannot resume');verify_source();d,_=deployment()
    b=validate_budget(read_json(ROOT/'inputs/budget.json'),time.time())
    if a.preflight:
        from cpu_feature_preflight import cpu_audit
        result=cpu_audit(d['assets']['manifest']['path'],d['feature_cache'],d['assets']['class_mapping']['path']);print(json.dumps(result));return 0
    out='artifacts/experiments/confirmation_H1'
    item=QueueItem(tag='H1',config='configs/experiment/confirmation_H1.yaml',output_dir=out,purpose='fresh unused-query-fold train-only H1100 fixedlast',trainer_entrypoint='scripts/train_frozen_head.py',resume_checkpoint=out+'/last.pt' if a.resume else None)
    def launch(command,cwd,log,on_start):
        lease=read_training_lock(d['canonical_lock']);require(lease and lease.queue_pid==os.getpid(),'exact canonical H1 controller lease required')
        attempt=uuid.uuid4().hex;env=os.environ.copy();env.update(P1_SOURCE_SHA=sha(ROOT/'source_manifest.json'),P1_LEASE_TOKEN=lease.token,P1_DEADLINE=str(b['deadline_unix']),P1_INITIAL_FREE=str(b['adjusted_initial_free_bytes']),P1_DISK_MAX=str(b['disk_max_bytes']),P1_MIN_FREE=str(b['min_free_bytes']),P1_ATTEMPT_ID=attempt,P1_RESUME=str(int(a.resume)))
        folder=ROOT/f'artifacts/launches/{attempt}';folder.mkdir(parents=True,exist_ok=False);child=None;owner=None
        def created(c,v):
            nonlocal child,owner
            child=c;owner=v
        def halt(c,v,go,start_error,closure_error):write(folder/'HALT.json',dict(status='closure_unconfirmed_lease_retained',pid=c.pid,owner=v,GO_attempted=go,start_error=repr(start_error),closure_error=repr(closure_error),lease_token=lease.token))
        def ready(c,v):write(folder/'START.json',dict(owner=v,command=list(command),source_manifest_sha256=env['P1_SOURCE_SHA'],config_sha256=sha(ROOT/command[3]),lease_token=lease.token))
        child,owner=start_with_go(command,cwd,log,env,[],on_start,ready,created,halt)
        try:
            code=child.wait(timeout=max(.1,b['deadline_unix']-time.time()));end=time.monotonic()+20
            while owned_group(snapshot(),owner,[]) and time.monotonic()<end:child.poll();time.sleep(.2)
            require(not owned_group(snapshot(),owner,[]),'owned H1 session still active')
            write(folder/'EXIT.json',dict(owner=owner,return_code=code,group_closed=True));return code
        except BaseException as failure:
            try:cleanup(child,owner,[]);require(not owned_group(snapshot(),owner,[]),'owned cleanup not closed')
            except BaseException as closure_error:
                try:halt(child,owner,True,failure,closure_error)
                finally:os._exit(70)
            raise
    return run_serial_queue(root=ROOT,python=Path(sys.executable),logs=ROOT/f'artifacts/queues/{uuid.uuid4().hex}',queue_name='20261010-new-confirmation-H1',items=[item],lock_path=Path(d['canonical_lock']),launcher=launch)
if __name__=='__main__':raise SystemExit(main())
