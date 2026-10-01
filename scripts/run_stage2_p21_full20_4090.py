"""Fresh all-148695 H1 then single-model P21 full20; two matched gates required."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aictrack1.engine.training_queue import QueueItem, probe_running_trainers, run_serial_queue
from aictrack1.data.feature_cache import load_feature_cache
from run_stage2_p21_fold2_4090_20ep import sha

EXPECTED_ROOT = Path('/root/autodl-tmp/AICtrack1_p21_full20_20261001')
LOCK = Path('/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json')
QUEUE = ROOT / 'artifacts/queues/20261001-p21-full20-4090'
HARD_SECONDS = 5 * 3600
MAX_ADDED = 4 * 1024**3
CACHE = 'artifacts/features/stage2/train_official336_p21_20261001'
H1 = 'artifacts/experiments/20261001-stage2-p21-h1-336-full-seed20260921-4090'
PEFT = 'artifacts/experiments/20261001-stage2-p21-sce-ac-336-full20-seed20260921-4090'
H1_CONFIG = 'configs/experiment/stage2_p21_h1_336_full_4090.yaml'
PEFT_CONFIG = 'configs/experiment/stage2_p21_sce_ac_336_full20_4090.yaml'
GATES = {
 'artifacts/reports/20261001-p21-fold2-20-vs14-matched-gate.json': 'c630d2f8b54040d477eb5e1dac2c6f8a427e691bf1957b9a5ba819f95a6f9873',
 'artifacts/reports/20261001-p21-fold3-20-vs14-matched-gate.json': '1f6c4d6515394bb2b2f20625848edada66bbd8898028c8254ebb847acfd7be74',
}


def compare_recipe(full, fold):
    expected = dict(fold)
    for key in ('expected_initial_top1', 'initial_top1_tolerance'):
        expected.pop(key, None)
    expected.update(run_id=Path(PEFT).name, output_dir='../../'+PEFT,
        training_scope='full_train_fixed_epochs', expected_train_count=148695,
        dataset_config='../data/stage2_fold1_336.yaml', split='../../artifacts/audits/stage2/train/split.csv',
        head_checkpoint='../../'+H1+'/final.pt')
    if full != expected:
        raise ValueError('full recipe differs from confirmed fold20 beyond scope/identity/new H1')


def completed_output(relative, epochs, peft):
    run = ROOT / relative
    m = json.loads((run/'metrics.json').read_text())
    if (m['training_scope'] != 'full_train_fixed_epochs' or m['train_count'] != 148695
            or m['validation_count'] != 0 or m['selected_checkpoint'] != 'final.pt'
            or [r['epoch'] for r in m['history']] != list(range(1, epochs+1))):
        raise RuntimeError('full run scope/count/final epochs incomplete')
    if peft and any(r['confidence_weighting']['next_effective_quality']['decision'] not in {'L0','L1'} for r in m['history']):
        raise RuntimeError('full PEFT AC non-L2 gate failed')
    for role, metric in (('final','selected_checkpoint_sha256'), ('last','last_checkpoint_sha256')):
        if sha(run/(role+'.pt')) != m[metric]:
            raise RuntimeError('full checkpoint SHA mismatch')
    state = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    required = {'optimizer','scheduler','scaler','epoch','history','python_rng_state','numpy_rng_state','torch_rng_state','cuda_rng_state_all'}
    required |= {'trainable_state','confidence_weighting'} if peft else {'model'}
    if not required.issubset(state) or state['epoch'] != epochs or len(state['history']) != epochs:
        raise RuntimeError('full last not safely resumable')
    del state
    final = torch.load(run/'final.pt', map_location='cpu', weights_only=True)
    if peft and (final['training_scope'] != 'full_train_fixed_epochs'
                 or final['head_initializer_sha256'] != sha(ROOT/H1/'final.pt')):
        raise RuntimeError('PEFT did not use this new full H1')
    return {'final_sha256':m['selected_checkpoint_sha256'], 'last_sha256':m['last_checkpoint_sha256'],
            'complete_epochs':epochs, 'train_count':148695, 'environment_hash':m['environment_hash']}


def preflight():
    if sys.platform != 'linux' or ROOT != EXPECTED_ROOT or (ROOT/'DataSetStage2/test').exists():
        raise RuntimeError('exact full train-only root required')
    if LOCK.exists() or probe_running_trainers() or QUEUE.exists() or any((ROOT/p).exists() for p in (H1,PEFT)):
        raise RuntimeError('existing lock/trainer/queue/output; no implicit restart')
    if not torch.cuda.is_available() or 'RTX 4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('approved RTX4090 unavailable')
    free = shutil.disk_usage(ROOT).free
    if free < 9*1024**3 or torch.cuda.mem_get_info(0)[0] < 20*1024**3:
        raise RuntimeError('full resource headroom insufficient')
    frozen = json.loads((ROOT/'p21_full_input_sha256.json').read_text())
    if any(sha(ROOT/p) != h for p,h in frozen.items()):
        raise RuntimeError('full frozen source/input drift')
    for path,h in GATES.items():
        if sha(ROOT/path) != h or json.loads((ROOT/path).read_text())['gate_passed'] is not True:
            raise RuntimeError('both matched gates required')
    compare_recipe(yaml.safe_load((ROOT/PEFT_CONFIG).read_text())['experiment'],
                   yaml.safe_load((ROOT/'configs/experiment/stage2_p21_fold2_sce_ac_20ep_4090_screen.yaml').read_text())['experiment'])
    head = yaml.safe_load((ROOT/H1_CONFIG).read_text())['experiment']
    old_head = yaml.safe_load((ROOT/'configs/experiment/stage2_p5_sce_h1_336_full_a10.yaml').read_text())['experiment']
    if head != {**old_head, 'run_id':Path(H1).name, 'output_dir':'../../'+H1, 'feature_cache':'../../'+CACHE}:
        raise RuntimeError('fresh full H1 recipe differs from established full100 recipe')
    cache = load_feature_cache(ROOT/CACHE)
    md = cache.metadata
    if (md.source != 'train' or md.dataset_stage != 'rematch' or md.expected_record_count != 148695
          or len(cache.sample_ids) != 148695 or len(set(cache.sample_ids)) != 148695
          or md.peft_checkpoint_sha256 is not None or md.feature_dim != 512
          or md.preprocessing.get('center_crop') != 336 or md.preprocessing.get('resize') != 336
          or md.clip_weight_sha256 != sha(ROOT/'artifacts/weights/ViT-B-32.pt')
          or md.manifest_sha256 != sha(ROOT/'artifacts/audits/stage2/train/samples.csv')
          or md.split_sha256 != sha(ROOT/'artifacts/audits/stage2/train/split.csv')
          or md.dataset_config_sha256 != sha(ROOT/'configs/data/stage2_fold1_336.yaml')):
        raise RuntimeError('full official336 cache provenance/count mismatch')
    del cache
    return {'scope':'Stage2 full148695 fresh100 H1 then20 SCE+AC; no test',
            'frozen_sha256':frozen, 'disk_free':free, 'hard_seconds':HARD_SECONDS, 'max_added_bytes':MAX_ADDED}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',action='store_true'); args=parser.parse_args()
    receipt=preflight()
    if not args.run:
        print(json.dumps({k:v for k,v in receipt.items() if k != 'frozen_sha256'})); return 0
    deadline=time.time()+HARD_SECONDS
    QUEUE.mkdir(parents=True,exist_ok=False)
    receipt.update(started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), deadline_unix=deadline)
    (QUEUE/'source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    checks={}
    def launch(command,cwd,log,on_start):
        if any(sha(ROOT/p) != h for p,h in receipt['frozen_sha256'].items()):
            raise RuntimeError('full frozen input changed')
        if command[1] == 'scripts/train_peft.py':
            checks['h1']=completed_output(H1,100,False)
            (QUEUE/'new_h1_receipt.json').write_text(json.dumps(checks['h1'],indent=2)+'\n')
        child=subprocess.Popen(list(command),cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            on_start(child.pid)
            while child.poll() is None:
                free=shutil.disk_usage(ROOT).free
                if time.time() >= deadline or free < 5*1024**3 or receipt['disk_free']-free > MAX_ADDED:
                    log.write('P21 full hard/resource cutoff; keep complete last\n');log.flush();return 124
                time.sleep(5)
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=30)
    code=run_serial_queue(root=ROOT,python=Path(sys.executable),logs=QUEUE,queue_name=QUEUE.name,
        lock_path=LOCK,items=[QueueItem('h1',H1_CONFIG,H1,'fresh full100 H1','scripts/train_frozen_head.py'),
                             QueueItem('peft',PEFT_CONFIG,PEFT,'official CLIP plus new full H1,20epochs')],launcher=launch)
    if code == 0:
        checks['h1']=completed_output(H1,100,False);checks['peft']=completed_output(PEFT,20,True)
    (QUEUE/'queue_result.json').write_text(json.dumps({'exit_code':code,'checks':checks,
        'finished_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()},indent=2)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
