"""Root-owned fresh H1 controller using the unmodified canonical serial queue."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
for k,v in {'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}.items():os.environ[k]=v
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from confirmation_contract import verify_source,deployment,sha
from aictrack1.engine.training_queue import QueueItem,run_serial_queue,read_training_lock
def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--initial-free-bytes',type=int,required=True);p.add_argument('--disk-max-bytes',type=int,default=20*1024**3);p.add_argument('--min-free-bytes',type=int,default=3*1024**3)
    a=p.parse_args()
    if os.name=='nt':raise RuntimeError('root-owned remote Linux controller only')
    verify_source();d,_=deployment()
    if time.time()>=a.deadline_unix:raise RuntimeError('H1 deadline expired')
    out='artifacts/experiments/confirmation_H1'
    item=QueueItem(tag='H1',config='configs/experiment/confirmation_H1.yaml',output_dir=out,
        purpose='fresh new-fold train-only H1100 fixedlast; immutable official CLIP336 features',trainer_entrypoint='scripts/train_frozen_head.py',resume_checkpoint=out+'/last.pt' if a.resume else None)
    def launch(command,cwd,handle,on_start):
        lease=read_training_lock(d['canonical_lock'])
        if not lease or lease.queue_pid!=os.getpid():raise RuntimeError('H1 controller canonical lease absent')
        env=os.environ.copy();env.update(P1_SOURCE_SHA=sha(ROOT/'source_manifest.json'),P1_LEASE_TOKEN=lease.token,
            P1_DEADLINE=str(a.deadline_unix),P1_INITIAL_FREE=str(a.initial_free_bytes),P1_DISK_MAX=str(a.disk_max_bytes),P1_MIN_FREE=str(a.min_free_bytes),P1_ATTEMPT_ID=uuid.uuid4().hex)
        child=subprocess.Popen(command,cwd=cwd,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True);on_start(child.pid)
        def alive():
            try:os.killpg(child.pid,0);return True
            except ProcessLookupError:return False
        try:code=child.wait(timeout=max(1,a.deadline_unix-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:pass
            if alive():os.killpg(child.pid,signal.SIGKILL)
            code=child.wait(timeout=20)
        for _ in range(50):
            if not alive():return code
            time.sleep(.1)
        raise RuntimeError('H1 child/processgroup termination ambiguous; original queue keeps lock')
    return run_serial_queue(root=ROOT,python=Path(sys.executable),logs=ROOT/'artifacts/queues/confirmation_H1',queue_name='20261009-confirmation-H1',items=[item],lock_path=Path(d['canonical_lock']),launcher=launch)
if __name__=='__main__':raise SystemExit(main())
