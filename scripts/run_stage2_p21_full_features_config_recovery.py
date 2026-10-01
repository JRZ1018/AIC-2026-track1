"""Recover absent fold1 configuration within the fresh batch's original deadline."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from run_stage2_p21_full20_4090 import ROOT,EXPECTED_ROOT,LOCK,CACHE,sha,probe_running_trainers
from aictrack1.engine.training_queue import TrainingLease
from aictrack1.data.feature_cache import load_feature_cache
ORIGINAL=ROOT/'artifacts/queues/20261001-p21-full-official336-features-fresh'
QUEUE=ROOT/'artifacts/queues/20261001-p21-full-official336-features-config-recovery'

def original_deadline(receipt,log,now):
    if (receipt['hard_seconds']!=1800 or receipt['deadline_unix']!=receipt['started_unix']+1800
            or 'FileNotFoundError' not in log or 'configs/data/stage2_fold1_336.yaml' not in log
            or '[Errno 2]' not in log or now>=receipt['deadline_unix']):
        raise RuntimeError('missing-config recovery evidence or original budget invalid')
    return receipt['deadline_unix']

def main():
    if sys.platform!='linux' or ROOT!=EXPECTED_ROOT or (ROOT/'DataSetStage2/test').exists():raise RuntimeError('train-only root required')
    if LOCK.exists() or probe_running_trainers() or QUEUE.exists() or (ROOT/CACHE).exists():raise RuntimeError('existing work; no implicit resume')
    original=json.loads((ORIGINAL/'source_receipt.json').read_text())
    deadline=original_deadline(original,(ORIGINAL/'features.log').read_text(),time.time())
    frozen=json.loads((ROOT/'p21_full_source_config_recovery_sha256.json').read_text())
    if any(frozen.get(p)!=h for p,h in original['frozen_sha256'].items()) or any(sha(ROOT/p)!=h for p,h in frozen.items()):raise RuntimeError('source drift')
    free_start=shutil.disk_usage(ROOT).free
    if free_start<9*1024**3:raise RuntimeError('insufficient disk')
    QUEUE.mkdir(parents=True,exist_ok=False)
    (QUEUE/'source_receipt.json').write_text(json.dumps({'reason':'missing full fold1 configuration; no images used by failed attempt',
        'original_receipt_sha256':sha(ORIGINAL/'source_receipt.json'),'started_unix':time.time(),
        'deadline_unix':deadline,'hard_seconds_original':1800,'max_added_bytes':1024**3,'frozen_sha256':frozen},indent=2)+'\n')
    lease=TrainingLease.acquire(LOCK,owner=QUEUE.name,config='configs/data/stage2_fold1_336.yaml',output_dir=str(ROOT/CACHE),command=tuple(sys.argv))
    try:
        with (QUEUE/'features.log').open('w') as log:
            child=subprocess.Popen([sys.executable,'scripts/extract_clip_features.py','--config','configs/data/stage2_fold1_336.yaml',
                '--source','train','--manifest','artifacts/audits/stage2/train/samples.csv','--split','artifacts/audits/stage2/train/split.csv',
                '--model-config','configs/model/openai_clip_vit_b32.yaml','--visual-resolution','336','--device','cuda','--feature-dtype','float16',
                '--output',CACHE,'--batch-size','64','--workers','16','--shard-size','4096'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                lease.attach_trainer(child.pid)
                while child.poll() is None:
                    free=shutil.disk_usage(ROOT).free
                    if time.time()>=deadline or free<5*1024**3 or free_start-free>1024**3:raise RuntimeError('original fresh deadline/resource cutoff')
                    time.sleep(5)
                if child.returncode:raise RuntimeError('feature generation failed; preserve logs/cache')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=20)
        cache=load_feature_cache(ROOT/CACHE)
        if len(cache.sample_ids)!=148695 or len(set(cache.sample_ids))!=148695 or cache.metadata.source!='train' or cache.metadata.peft_checkpoint_sha256 is not None:raise RuntimeError('full cache invalid')
        del cache
        all_frozen={**frozen,**{p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/CACHE).rglob('*') if p.is_file()}}
        with (ROOT/'p21_full_input_sha256.json').open('x') as handle:handle.write(json.dumps(all_frozen,indent=2)+'\n')
        (QUEUE/'result.json').write_text(json.dumps({'status':'complete','record_count':148695,'complete_sha256':sha(ROOT/CACHE/'complete.json'),
            'input_receipt_sha256':sha(ROOT/'p21_full_input_sha256.json')},indent=2)+'\n')
    finally:lease.release()

if __name__=='__main__':main()
