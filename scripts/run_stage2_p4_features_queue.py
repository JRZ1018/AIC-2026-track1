"""Gated, atomic Stage2 train-only feature extraction for the P4 full H1."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aictrack1.engine.training_queue import TrainingLease
from aictrack1.data.manifest import sha256_file
from run_stage2_p4_followup_queue import launch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=('candidate', 'fallback'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = ROOT / f'configs/experiment/stage2_p4_{args.route}_h1_336_full.yaml'
    experiment = yaml.safe_load(config.read_text(encoding='utf-8'))['experiment']
    if experiment.get('execution_enabled') is not True:
        raise RuntimeError('full route has not passed its preregistered gate')
    data = (config.parent / experiment['dataset_config']).resolve()
    output = (config.parent / experiment['feature_cache']).resolve()
    expected_output = ROOT / 'artifacts/features/stage2/train_official336_p4_20260927'
    if output != expected_output or experiment['expected_train_count'] != 148695:
        raise ValueError('unexpected cache identity or Stage2 count')
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError('requires 3GiB free')
    manifest = ROOT / 'artifacts/audits/stage2/train/samples.csv'
    split = ROOT / 'artifacts/audits/stage2/train/split.csv'
    model_config = ROOT / 'configs/model/openai_clip_vit_b32.yaml'
    model = yaml.safe_load(model_config.read_text(encoding='utf-8'))['model']
    if (output / 'complete.json').exists():
        metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
        expected = {
            'source': 'train', 'dataset_stage': 'stage2',
            'dataset_config_sha256': sha256_file(data),
            'manifest_sha256': sha256_file(manifest), 'split_sha256': sha256_file(split),
            'clip_weight_sha256': model['expected_weight_sha256'],
            'expected_record_count': 148695, 'peft_checkpoint_sha256': None,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('existing cache provenance mismatch; no overwrite')
        if metadata['preprocessing'].get('center_crop') != 336:
            raise ValueError('existing cache has a different resolution')
        print('Matching complete cache exists; H1 loader will verify shard hashes.')
        return 0
    if output.exists() and not args.resume:
        raise FileExistsError('partial cache requires explicit --resume')
    queue = ROOT / f'artifacts/queues/20260927-p4-{args.route}-features'
    lease = TrainingLease.acquire(
        ROOT / 'artifacts/queues/training.lock.json',
        owner=queue.name, config=str(config), output_dir=str(output), command=sys.argv,
    )
    try:
        queue.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, 'scripts/extract_clip_features.py', '--config', str(data),
            '--source', 'train', '--manifest', str(manifest), '--split', str(split),
            '--model-config', str(model_config), '--visual-resolution', '336',
            '--device', 'cuda', '--feature-dtype', 'float16', '--output', str(output),
            '--batch-size', '64', '--workers', '4', '--shard-size', '4096',
        ]
        if args.resume:
            command.append('--resume')
        with (queue / 'features.log').open('a', encoding='utf-8') as log:
            code = launch(command, ROOT, log, lease.attach_trainer)
        if code:
            raise RuntimeError(f'feature extraction failed: {code}')
        stats = json.loads((output / 'extraction_stats.json').read_text(encoding='utf-8'))
        if stats['record_count'] != 148695 or not (output / 'complete.json').is_file():
            raise RuntimeError('incomplete feature extraction')
    finally:
        lease.release()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
