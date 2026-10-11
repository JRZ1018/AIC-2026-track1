"""One arm under the canonical TrainingLease. Root owns all actual GPU invocation."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import signal
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from p1_contract import ARMS,assets,sha,verify_files,verify_remote_assets
from aictrack1.engine.training_queue import QueueItem,run_serial_queue

def main():
    p=argparse.ArgumentParser();p.add_argument("--arm",required=True,choices=ARMS)
    p.add_argument("--resume",action="store_true");p.add_argument("--stop-after-epoch",type=int)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--deadline-unix",type=float,required=True)
    p.add_argument("--initial-free-bytes",type=int,required=True)
    p.add_argument("--disk-max-bytes",type=int,default=20*1024**3)
    p.add_argument("--min-free-bytes",type=int,default=3*1024**3)
    args=p.parse_args()
    if args.smoke and (args.resume or args.stop_after_epoch is not None):p.error("smoke cannot be mixed with candidate resume/pause")
    verify_files();payload=verify_remote_assets()
    if args.stop_after_epoch is not None and not 1<=args.stop_after_epoch<20:
        p.error("pause boundary must be 1..19, scheduler remains20")
    if time.time()>=args.deadline_unix: raise RuntimeError("P1 deadline expired")
    out=f"artifacts/smoke/p1_{args.arm}" if args.smoke else f"artifacts/experiments/P1_{args.arm}"
    item=QueueItem(tag=args.arm,config=f"configs/experiment/p1_{args.arm}.yaml",output_dir=out,
        purpose="full-visual + task head, original20 cosine",trainer_entrypoint="scripts/train_n1_peft.py",
        resume_checkpoint=f"{out}/last.pt" if args.resume else None)
    attempt_id=uuid.uuid4().hex
    def launch(command,cwd,handle,on_start):
        if args.smoke: command=tuple(command)+("--p1-smoke",)
        from aictrack1.engine.training_queue import read_training_lock
        record=read_training_lock(payload["canonical_lock"])
        if not record or record.queue_pid!=os.getpid():raise RuntimeError("launcher lease missing")
        env=os.environ.copy();env.update(P1_SOURCE_SHA=sha(ROOT/"source_manifest.json"),
            P1_LEASE_TOKEN=record.token,P1_DEADLINE=str(args.deadline_unix),
            P1_INITIAL_FREE=str(args.initial_free_bytes),P1_DISK_MAX=str(args.disk_max_bytes),
            P1_MIN_FREE=str(args.min_free_bytes),P1_ATTEMPT_ID=attempt_id)
        if args.stop_after_epoch is None:env.pop("P1_STOP_AFTER_EPOCH",None)
        else:env["P1_STOP_AFTER_EPOCH"]=str(args.stop_after_epoch)
        if os.name=="nt": raise RuntimeError("remote Linux P1 controller required")
        child=subprocess.Popen(command,cwd=cwd,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        on_start(child.pid)
        def group_alive():
            try: os.killpg(child.pid,0); return True
            except ProcessLookupError: return False
        try:
            result=child.wait(timeout=max(1,args.deadline_unix-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: pass
            if group_alive(): os.killpg(child.pid,signal.SIGKILL)
            result=child.wait(timeout=20)
        for _ in range(50):
            if not group_alive(): return result
            time.sleep(.1)
        raise RuntimeError("trainer exited but its process group remains; canonical lock retained")
    code=run_serial_queue(root=ROOT,python=Path(sys.executable),logs=ROOT/f"artifacts/queues/P1_{args.arm}",
        queue_name=f"20261009-P1-{args.arm}",items=[item],lock_path=Path(payload["canonical_lock"]),launcher=launch)
    screen=ROOT/out/"p1_screen_receipt.json"
    if args.stop_after_epoch is not None and screen.exists():
        receipt=json.loads(screen.read_text())
        if (receipt.get("attempt_id")==attempt_id and receipt.get("epoch")==args.stop_after_epoch
                and receipt.get("config_sha256")==sha(ROOT/item.config)
                and receipt.get("last_sha256")==sha(ROOT/out/"last.pt")):
            print(json.dumps(dict(status="planned_pause_verified_not_complete",arm=args.arm,epoch=receipt["epoch"])))
            return 0
    return code

if __name__=="__main__":raise SystemExit(main())
