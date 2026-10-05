"""CPU-only audit of transferred full20 assets and an existing final package.

Does not decode images, run inference, alter predictions, or deserialize pickle
without weights_only=True. Paths refer to an immutable private transfer bundle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aictrack1.data.class_mapping import ClassMapping
from aictrack1.submission.package import (
    SubmissionSpec, submission_keys_from_relative_paths,
    validate_submission_csv, validate_submission_zip,
)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_tensors(value) -> None:
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), 'nonfinite model/optimizer tensor')
    elif isinstance(value, dict):
        for item in value.values():
            finite_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tensors(item)


def audit(assets: Path, receipts: Path, official: Path, inference_sources: Path) -> dict:
    torch.set_num_threads(4)
    transfer = json.loads((assets / 'transfer_sha256.json').read_text())
    for name, item in transfer.items():
        path = (assets / name).resolve()
        require(path.is_relative_to(assets.resolve()), 'transfer path escapes bundle')
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'asset bytes/SHA mismatch: ' + name)
    for name, digest in json.loads((receipts / 'transfer_sha256.json').read_text()).items():
        require(sha(receipts / name) == digest, 'package receipt SHA mismatch: ' + name)
    freeze = json.loads((receipts / 'inference_freeze.json').read_text())
    full_source = assets / 'mainline_sources_sha256.json'
    require(sha(full_source) == freeze['full_source_sha256'], 'full source SHA mismatch')
    for name, digest in json.loads(full_source.read_text())['files'].items():
        require(sha(assets / name) == digest, 'frozen source mismatch: ' + name)
    require(sha(receipts / 'inference_sources_sha256.json') == freeze['source_sha256'],
            'inference source catalog mismatch')
    for name, digest in json.loads((receipts / 'inference_sources_sha256.json').read_text())['files'].items():
        require(sha(inference_sources / name) == digest, 'inference bytes mismatch: ' + name)

    visual = assets / 'artifacts/experiments/mainline20'
    head_path = assets / 'artifacts/experiments/mainline-h1/last.pt'
    require(sha(visual / 'final.pt') == freeze['final_sha256'], 'final SHA mismatch')
    require(sha(visual / 'last.pt') == freeze['last_sha256'], 'last SHA mismatch')
    require(sha(head_path) == freeze['head_sha256'], 'fresh H1 SHA mismatch')
    final = torch.load(visual / 'final.pt', map_location='cpu', weights_only=True)
    last = torch.load(visual / 'last.pt', map_location='cpu', weights_only=True)
    head = torch.load(head_path, map_location='cpu', weights_only=True)
    metrics = json.loads((visual / 'metrics.json').read_text())
    require(last['epoch'] == 20 and metrics['history'] == last['history']
            and [r['epoch'] for r in last['history']] == list(range(1, 21)), 'incomplete20')
    require(metrics['train_count'] == 148695 and metrics['validation_count'] == 0
            and metrics['early_stop'] is None, 'not complete full official train')
    require(final['selection_rule'] == 'fixed final epoch 20; no validation selection'
            and final['training_scope'] == 'full_train_fixed_epochs'
            and last['scheduler']['T_max'] == 20, 'wrong terminal selection/cosine')
    require(last['config_sha256'] == freeze['config_sha256']
            and sha(assets / 'configs/experiment/mainline20.yaml') == freeze['config_sha256'],
            'config differs')
    required = {'optimizer', 'scheduler', 'scaler', 'confidence_weighting', 'python_rng_state',
                'numpy_rng_state', 'torch_rng_state', 'cuda_rng_state_all', 'trainable_state'}
    require(required.issubset(last), 'incomplete visual recovery state')
    require(set(final['trainable_state']) == set(last['trainable_state'])
            and all(torch.equal(t, last['trainable_state'][n])
                    for n, t in final['trainable_state'].items()), 'final/last differ')
    for row in last['history']:
        require(row['confidence_weighting']['next_effective_quality']['decision'] != 'L2',
                'AC L2 in training')
    require(head['epoch'] == 100 and head['row_binding']['counts'] ==
            {'train': 148695, 'validation': 0}, 'not fresh full H1100')
    require(required.difference({'confidence_weighting', 'trainable_state'}).issubset(head),
            'incomplete H1 recovery')
    require(final['head_initializer_sha256'] == freeze['head_sha256'], 'different frozen head')
    for state in (final, last, head):
        require(state['competition_stage'] == 'stage2' and state['num_classes'] == 750
                and state['clip_weight_sha256'] == freeze['official_clip_sha256'], 'provenance')
    for value in (final['trainable_state'], last['optimizer'], head['model'], head['optimizer']):
        finite_tensors(value)

    test = official / 'artifacts/audits/stage2/test/samples.csv'
    mapping = ClassMapping.from_csv(official / 'artifacts/audits/stage2/metadata/classes.csv')
    require(sha(test) == freeze['test_manifest_expected_sha256'] and len(mapping) == 750,
            'official manifest/mapping differs')
    with test.open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    require(len(records) == 37444 and all(r['source_split'] == 'test' for r in records),
            'official test index differs')
    package = receipts / 'submissions/final'
    manifest = json.loads((package / 'submission_manifest.json').read_text())
    spec = SubmissionSpec(**manifest['spec'])
    names = submission_keys_from_relative_paths([r['relative_path'] for r in records], spec=spec)
    allowed = [entry.official_class_id for entry in mapping.entries]
    rows = validate_submission_csv(package / 'pred_results.csv', names, allowed, spec=spec)
    validate_submission_zip(package / 'pred_results.zip', names, allowed, spec=spec)
    require([row[0] for row in rows] == list(names), 'filename order differs')
    with zipfile.ZipFile(package / 'pred_results.zip') as archive:
        require(archive.namelist() == ['pred_results.csv']
                and archive.read('pred_results.csv') == (package / 'pred_results.csv').read_bytes(),
                'ZIP CSV differs')
    require(manifest['class_mapping_sha256'] == mapping.digest()
            == final['class_mapping_sha256'], 'mapping SHA differs')
    run = manifest['run_metadata']
    require(run['checkpoint_sha256'] == freeze['final_sha256']
            and run['config_hash'] == freeze['config_sha256'] and manifest['record_count'] == 37444,
            'package does not bind full final')
    require(sha(inference_sources / 'scripts/infer_and_build_submission.py')
            == run['inference_source_sha256']
            and sha(inference_sources / 'src/aictrack1/evaluation/balanced_prior.py')
            == run['balanced_prior_source_sha256'], 'actual inference code differs')
    require(sha(official / 'artifacts/weights/ViT-B-32.pt') == freeze['official_clip_sha256'],
            'local official CLIP differs')
    require(run['balanced_prior']['temperature_multiplier'] == .5
            and run['balanced_prior']['bias_strength'] == .75
            and run['balanced_prior']['fit']['iterations'] == 60
            and run['four_view_resize'] == 384, 'frozen inference differs')
    require(sha(package / 'pred_results.csv') == manifest['csv_sha256']
            and sha(package / 'pred_results.zip') == manifest['zip_sha256'], 'package bytes differ')
    return dict(status='PASS', record_count=37444, class_mapping_count=750, epochs=20,
                h1_epochs=100, train_count=148695, validation_count=0,
                cpu_weights_only=True, final_equals_complete_last=True, ac_no_l2=True,
                final_sha256=freeze['final_sha256'], last_sha256=freeze['last_sha256'],
                head_sha256=freeze['head_sha256'], csv_sha256=manifest['csv_sha256'],
                zip_sha256=manifest['zip_sha256'],
                manifest_sha256=sha(package / 'submission_manifest.json'),
                all_assets_sha_verified=len(transfer), online_score=None,
                limitations=['No online submission or gain guarantee',
                             'Repeated noisy train-derived validation, mixed confirmation hardware',
                             'Third confirmation seed interrupted/resumed; full model trained fresh uninterrupted'])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--receipts', type=Path, required=True)
    p.add_argument('--official-root', type=Path, required=True)
    p.add_argument('--inference-sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.assets, args.receipts, args.official_root, args.inference_sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
