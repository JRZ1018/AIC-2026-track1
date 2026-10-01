"""Separate CPU audit: complete full20 provenance plus exact CSV/ZIP contract."""
import csv
import json
from pathlib import Path
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aictrack1.data.class_mapping import ClassMapping
from aictrack1.submission.package import (SubmissionSpec, submission_keys_from_relative_paths,
    validate_submission_csv, validate_submission_zip)
import run_stage2_p21_full_submission as plan


def main():
    full = plan.full
    receipt = json.loads((plan.QUEUE / 'source_receipt.json').read_text())
    manifest = json.loads((plan.OUTPUT / 'submission_manifest.json').read_text())
    run = manifest['run_metadata']
    for path, digest in receipt['frozen_sha256'].items():
        if full.sha(ROOT / path) != digest:
            raise ValueError('inference source changed')
    if full.sha(full.QUEUE / 'queue_result.json') != receipt['full_queue_result_sha256']:
        raise ValueError('training completion receipt changed')
    for path, digest in full.GATES.items():
        if full.sha(ROOT / path) != digest:
            raise ValueError('matched gate changed')
    metrics = json.loads((ROOT / full.PEFT / 'metrics.json').read_text())
    plan.check_metrics(metrics, 20, True)
    checkpoint = torch.load(ROOT / full.PEFT / 'final.pt', map_location='cpu', weights_only=True)
    if (checkpoint['competition_stage'] != 'stage2' or checkpoint['num_classes'] != 750
            or checkpoint['training_scope'] != 'full_train_fixed_epochs'
            or run['checkpoint_sha256'] != receipt['training_checks']['peft']['final_sha256']
            or checkpoint['head_initializer_sha256'] != receipt['training_checks']['h1']['final_sha256']):
        raise ValueError('wrong full final/fresh H1')
    mapping_path = ROOT / 'artifacts/audits/stage2/metadata/classes.csv'
    mapping = ClassMapping.from_csv(mapping_path)
    spec = SubmissionSpec(**manifest['spec'])
    with plan.TEST.open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 37444 or any(r['source_split'] != 'test' for r in records):
        raise ValueError('official test manifest differs')
    filenames = submission_keys_from_relative_paths([r['relative_path'] for r in records], spec=spec)
    allowed = [entry.official_class_id for entry in mapping.entries]
    rows = validate_submission_csv(plan.OUTPUT / 'pred_results.csv', filenames, allowed, spec=spec)
    validate_submission_zip(plan.OUTPUT / 'pred_results.zip', filenames, allowed, spec=spec)
    with zipfile.ZipFile(plan.OUTPUT / 'pred_results.zip') as archive:
        if archive.namelist() != ['pred_results.csv'] or archive.read('pred_results.csv') != (plan.OUTPUT / 'pred_results.csv').read_bytes():
            raise ValueError('ZIP differs from standalone CSV')
    if len(rows) != 37444 or len(mapping) != 750 or manifest['record_count'] != 37444:
        raise ValueError('wrong record/class count')
    if mapping.digest() != manifest['class_mapping_sha256']:
        raise ValueError('mapping digest differs')
    expected = {
        'csv': (plan.OUTPUT / 'pred_results.csv', manifest['csv_sha256']),
        'zip': (plan.OUTPUT / 'pred_results.zip', manifest['zip_sha256']),
        'checkpoint': (ROOT / full.PEFT / 'final.pt', run['checkpoint_sha256']),
        'head': (ROOT / full.H1 / 'final.pt', checkpoint['head_initializer_sha256']),
        'config': (ROOT / full.PEFT_CONFIG, run['config_hash']),
        'official_clip': (ROOT / 'artifacts/weights/ViT-B-32.pt', plan.CLIP_SHA),
        'test_manifest': (plan.TEST, plan.TEST_SHA),
        'selection_report': (plan.SELECTION, plan.SELECTION_SHA),
        'train_manifest': (ROOT / 'artifacts/audits/stage2/train/samples.csv', checkpoint['train_manifest_sha256']),
        'split': (ROOT / 'artifacts/audits/stage2/train/split.csv', checkpoint['split_sha256']),
        'inference_source': (ROOT / 'scripts/infer_and_build_submission.py', run['inference_source_sha256']),
        'balanced_prior_source': (ROOT / 'src/aictrack1/evaluation/balanced_prior.py', run['balanced_prior_source_sha256']),
    }
    hashes = {name: full.sha(path) for name, (path, _) in expected.items()}
    if any(hashes[name] != digest for name, (_, digest) in expected.items()):
        raise ValueError('package or provenance SHA differs')
    selected = json.loads(plan.SELECTION.read_text())
    if (selected['selected_candidate'] != 'uniform_soft_t0.5_s0.75' or selected['gate_passed'] is not True
            or run['balanced_prior']['temperature_multiplier'] != 0.5
            or run['balanced_prior']['bias_strength'] != 0.75 or run['four_view_resize'] != 384
            or run['balanced_prior_selection']['report_sha256'] != plan.SELECTION_SHA):
        raise ValueError('frozen single-checkpoint inference protocol differs')
    summary = {'status': 'PASS', 'record_count': len(rows), 'mapping_class_count': len(mapping),
        'training_scope': 'full148695 fresh100 H1 then20 SCE+AC', 'hashes': hashes,
        'manifest_sha256': full.sha(plan.OUTPUT / 'submission_manifest.json'),
        'limitations': ['Two reused overlapping folds and one seed, not independent blind/multiseed confirmation',
                       'Online gain unknown; no online submission performed']}
    destination = ROOT / 'artifacts/reports/20261002-p21-full20-submission-audit.json'
    with destination.open('x') as stream:
        stream.write(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
