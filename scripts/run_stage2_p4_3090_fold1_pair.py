"""P4 matched single-variable GCE screening; no test access or inference."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aictrack1.engine.training_queue import QueueItem,run_serial_queue

def main():
    if not sys.platform.startswith('linux'):
        raise RuntimeError('P4 queue is preregistered for the remote Linux 3090 only')
    import shutil
    if shutil.disk_usage(ROOT).free < 3*1024**3:
        raise RuntimeError('P4 requires at least 3 GiB free before launch')
    items=[QueueItem(tag=tag,config=f'configs/experiment/stage2_p4_gce_fold1_3090_{tag}.yaml',output_dir=f'artifacts/experiments/20260927-stage2-p4-fold1-3090-gce-{tag.replace("_","-")}',purpose='Same-host fold1 GCE q07 control versus q05 only-variable candidate') for tag in ('q07_control','q05_candidate')]
    return run_serial_queue(root=ROOT,python=Path(sys.executable),logs=ROOT/'artifacts/queues/20260927-p4-3090-fold1-pair',queue_name='20260927-p4-3090-fold1-pair',items=items)

if __name__=='__main__':
    raise SystemExit(main())
