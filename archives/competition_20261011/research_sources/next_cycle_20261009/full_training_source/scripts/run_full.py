"""One selected FULL arm under the existing canonical serial queue and exact-last resume."""
import argparse,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p1_contract import sha,verify_files,verify_remote_assets,selection,verify_selection_evidence
from aictrack1.engine.training_queue import QueueItem,run_serial_queue,read_training_lock

def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--selection-sha256',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--initial-free-bytes',type=int,required=True)
    p.add_argument('--disk-max-bytes',type=int,default=20*1024**3);p.add_argument('--min-free-bytes',type=int,default=3*1024**3)
    args=p.parse_args()
    if args.smoke and args.resume:p.error('synthetic full smoke cannot resume a candidate')
    if os.name=='nt':raise RuntimeError('root-owned remote Linux queue required')
    # Before any Torch import in evidence validation or child process.
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    verify_files();payload=verify_remote_assets();choice=selection(args.selection,args.selection_sha256)
    verify_selection_evidence(choice)
    if args.initial_free_bytes!=choice['original_initial_free_bytes']:raise RuntimeError('must retain original cumulative disk baseline')
    if args.min_free_bytes<3*1024**3 or time.time()>=args.deadline_unix:raise RuntimeError('invalid full resource boundary')
    arm=choice['selected_arm'];out=f'artifacts/smoke/p1_full_{arm}' if args.smoke else f'artifacts/experiments/P1_full_{arm}'
    item=QueueItem(arm,f'configs/experiment/p1_full_{arm}.yaml',out,'single selected fresh official FULL20 fixed final',
        trainer_entrypoint='scripts/train_n1_peft.py',resume_checkpoint=f'{out}/last.pt' if args.resume else None)
    attempt=uuid.uuid4().hex
    def launch(command,cwd,handle,on_start):
        record=read_training_lock(payload['canonical_lock'])
        if not record or record.queue_pid!=os.getpid():raise RuntimeError('full launcher canonical lease missing')
        if args.smoke:command=tuple(command)+('--p1-smoke',)
        env=os.environ.copy();env.update(P1_SOURCE_SHA=sha(ROOT/'source_manifest.json'),P1_LEASE_TOKEN=record.token,
            P1_DEADLINE=str(args.deadline_unix),P1_INITIAL_FREE=str(args.initial_free_bytes),P1_DISK_MAX=str(args.disk_max_bytes),
            P1_MIN_FREE=str(args.min_free_bytes),P1_ATTEMPT_ID=attempt,P1_SELECTION_PATH=str(args.selection.resolve()),
            P1_SELECTION_SHA=args.selection_sha256,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env.pop('P1_STOP_AFTER_EPOCH',None)
        child=subprocess.Popen(command,cwd=cwd,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        on_start(child.pid)
        def group_alive():
            try:os.killpg(child.pid,0);return True
            except ProcessLookupError:return False
        try:code=child.wait(timeout=max(1,args.deadline_unix-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:pass
            if group_alive():os.killpg(child.pid,signal.SIGKILL)
            code=child.wait(timeout=20)
        for _ in range(50):
            if not group_alive():return code
            time.sleep(.1)
        raise RuntimeError('full trainer process group exit ambiguous; canonical lock retained')
    code=run_serial_queue(root=ROOT,python=Path(sys.executable),logs=ROOT/f'artifacts/queues/P1_full_{arm}',
        queue_name=f'20261009-P1-full-{arm}',items=[item],lock_path=Path(payload['canonical_lock']),launcher=launch)
    report=dict(schema='aictrack1.p1.full_launch_result.v1',arm=arm,return_code=code,selection_sha256=args.selection_sha256,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),synthetic_smoke=args.smoke,resumed=args.resume,
        original_initial_free_bytes=args.initial_free_bytes,runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    receipt=ROOT/f'artifacts/queues/full_attempt_{attempt}.json';receipt.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return code
if __name__=='__main__':raise SystemExit(main())
