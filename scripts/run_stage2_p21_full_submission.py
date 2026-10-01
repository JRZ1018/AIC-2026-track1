"""One frozen full148695 P21 final inference, only after both gates/full20."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aictrack1.engine.training_queue import TrainingLease, probe_running_trainers
import run_stage2_p21_full20_4090 as full

QUEUE = ROOT / 'artifacts/queues/20261002-p21-full20-final-submission'
OUTPUT = ROOT / 'submissions/20261002_p21_sce_ac_full20_fourview_balanced_4090'
TEST = ROOT / 'artifacts/audits/stage2/test/samples.csv'
SELECTION = ROOT / 'artifacts/reports/20260928-p7-balanced-prior-fold2-fold3.json'
TEST_SHA = '576ff0f69e0a104ca5a0af7f190628074b572102c6bc8daf8cb653fb08054201'
SELECTION_SHA = '8122c9c96fb80823db9d19edb403d8d147d26a32c43b7b48fe7cca9db311d703'
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
HARD_SECONDS = 3600
MAX_ADDED = 1024**3


def check_metrics(metrics, epochs, peft):
    if (metrics.get('training_scope') != 'full_train_fixed_epochs'
            or metrics.get('train_count') != 148695 or metrics.get('validation_count') != 0
            or metrics.get('selected_checkpoint') != 'final.pt'
            or [r['epoch'] for r in metrics['history']] != list(range(1, epochs + 1))):
        raise ValueError('full scope/count/fixed final/complete epochs required')
    for row in metrics['history']:
        loss = row['train_loss']
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise ValueError('nonfinite or invalid complete-epoch loss')
        if peft and row['confidence_weighting']['next_effective_quality']['decision'] not in {'L0', 'L1'}:
            raise ValueError('AC non-L2 required for every epoch')


def validate_training():
    if sys.platform != 'linux' or ROOT != full.EXPECTED_ROOT:
        raise RuntimeError('exact isolated full root required')
    if full.LOCK.exists() or probe_running_trainers():
        raise RuntimeError('training has not naturally released shared resources')
    receipt = json.loads((full.QUEUE / 'source_receipt.json').read_text())
    result = json.loads((full.QUEUE / 'queue_result.json').read_text())
    if result['exit_code'] != 0 or not {'h1', 'peft'}.issubset(result['checks']):
        raise RuntimeError('complete successful full queue receipt required')
    for path, digest in receipt['frozen_sha256'].items():
        if full.sha(ROOT / path) != digest:
            raise RuntimeError(f'frozen training source/input changed: {path}')
    for path, digest in full.GATES.items():
        if full.sha(ROOT / path) != digest or json.loads((ROOT / path).read_text())['gate_passed'] is not True:
            raise RuntimeError('both original matched gates must pass')
    checks = {}
    for name, path, epochs, peft in [('h1', full.H1, 100, False), ('peft', full.PEFT, 20, True)]:
        check_metrics(json.loads((ROOT / path / 'metrics.json').read_text()), epochs, peft)
        checks[name] = full.completed_output(path, epochs, peft)
        if checks[name] != result['checks'][name]:
            raise RuntimeError('full completion receipt changed')
    checkpoint = torch.load(ROOT / full.PEFT / 'final.pt', map_location='cpu', weights_only=True)
    if (checkpoint.get('competition_stage') != 'stage2' or checkpoint.get('num_classes') != 750
            or checkpoint.get('clip_weight_sha256') != CLIP_SHA
            or checkpoint.get('config_sha256') != full.sha(ROOT / full.PEFT_CONFIG)
            or checkpoint.get('head_initializer_sha256') != checks['h1']['final_sha256']):
        raise RuntimeError('full final official-weight/config/fresh-H1 provenance failed')
    del checkpoint
    return checks


def command_for(checkpoint_sha):
    return [sys.executable, 'scripts/infer_and_build_submission.py',
        '--experiment-config', str(ROOT / full.PEFT_CONFIG),
        '--checkpoint', str(ROOT / full.PEFT / 'final.pt'),
        '--test-manifest', str(TEST), '--output-dir', str(OUTPUT),
        '--batch-size', '64', '--workers', '16', '--four-view-scale384',
        '--expected-checkpoint-sha256', checkpoint_sha,
        '--balanced-prior-uniform-soft', '--balanced-prior-temperature-multiplier', '0.5',
        '--balanced-prior-bias-strength', '0.75', '--balanced-prior-selection-report', str(SELECTION),
        '--expected-balanced-prior-report-sha256', SELECTION_SHA]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    checks = validate_training()
    if QUEUE.exists() or OUTPUT.exists():
        raise RuntimeError('no duplicate inference or overwrite of existing output')
    if full.sha(TEST) != TEST_SHA or full.sha(SELECTION) != SELECTION_SHA:
        raise RuntimeError('official manifest/frozen P7 inference choice differs')
    if not args.run:
        print(json.dumps({'status': 'full_model_preflight_passed', 'checks': checks})); return 0
    if not (ROOT / 'DataSetStage2/test').is_dir():
        raise RuntimeError('official test is not staged')
    compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True)
    if any(line.strip().isdigit() and int(line.strip()) != os.getpid()
           for line in compute.stdout.splitlines()):
        raise RuntimeError('another GPU process exists; do not overlap inference')
    if not torch.cuda.is_available() or 'RTX 4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('approved RTX4090 required')
    if torch.cuda.mem_get_info(0)[0] < 8 * 1024**3:
        raise RuntimeError('GPU has another allocation; no overlapping inference')
    free_start = shutil.disk_usage(ROOT).free
    if free_start < 7 * 1024**3:
        raise RuntimeError('insufficient inference disk headroom')
    command = command_for(checks['peft']['final_sha256'])
    lease = TrainingLease.acquire(full.LOCK, owner=QUEUE.name, config=str(ROOT / full.PEFT_CONFIG),
                                  output_dir=str(OUTPUT), command=tuple(command))
    try:
        QUEUE.mkdir(parents=True, exist_ok=False)
        frozen = {p: full.sha(ROOT / p) for p in [full.PEFT_CONFIG,
            'scripts/infer_and_build_submission.py', 'src/aictrack1/evaluation/balanced_prior.py',
            'scripts/run_stage2_p21_full_submission.py', 'scripts/audit_stage2_p21_full_submission.py']}
        receipt = {'scope': 'one full148695 final single-checkpoint test inference; no tuning/online submission',
            'training_checks': checks, 'matched_gates': full.GATES, 'frozen_sha256': frozen,
            'full_queue_result_sha256': full.sha(full.QUEUE / 'queue_result.json'),
            'test_manifest_sha256': TEST_SHA, 'p7_selection_sha256': SELECTION_SHA,
            'hard_seconds': HARD_SECONDS, 'max_added_bytes': MAX_ADDED, 'started_unix': time.time()}
        (QUEUE / 'source_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        deadline = time.monotonic() + HARD_SECONDS
        with (QUEUE / 'inference.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                while child.poll() is None:
                    free = shutil.disk_usage(ROOT).free
                    if time.monotonic() >= deadline or free < 5 * 1024**3 or free_start - free > MAX_ADDED:
                        raise RuntimeError('inference hard/resource cutoff; preserve partial output')
                    time.sleep(5)
                if child.returncode != 0:
                    raise RuntimeError(f'inference failed with exit code {child.returncode}; no implicit rerun')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=20)
        with (QUEUE / 'audit.log').open('w') as log:
            subprocess.run([sys.executable, 'scripts/audit_stage2_p21_full_submission.py'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=max(1, deadline - time.monotonic()))
        (QUEUE / 'result.json').write_text(json.dumps({'status': 'complete',
            'checkpoint_sha256': checks['peft']['final_sha256'],
            'csv_sha256': full.sha(OUTPUT / 'pred_results.csv'),
            'zip_sha256': full.sha(OUTPUT / 'pred_results.zip'),
            'manifest_sha256': full.sha(OUTPUT / 'submission_manifest.json')}, indent=2) + '\n')
        return 0
    finally:
        lease.release()


if __name__ == '__main__':
    raise SystemExit(main())
