"""Bind an existing FV2 held-out logit cache to official rows; CPU only.

This script writes only beside itself. It never changes official data, models,
or another agent's outputs. It does not run image inference or fit parameters.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
TRANSFER = Path(r'D:\aicomp\stage2-server-transfer')
FOLD = TRANSFER / 'fv2_results_20261008'
FULL = TRANSFER / 'fv2_full_results_20261008'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_exact(path, payload):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT), 'output must stay in acceptance_source')
    if path.exists():
        require(path.read_bytes() == payload, f'existing output differs: {path}')
        return
    with path.open('xb') as handle:
        handle.write(payload)


def row_digest(rows):
    return hashlib.sha256(json.dumps(rows, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def main():
    import torch
    torch.set_num_threads(2)
    cache = FOLD / 'diagnostic/artifacts/diagnostics/fullvisual/fused_logits.pt'
    report_path = cache.with_name('report.json')
    manifest_path = FULL / 'official/train_samples.csv'
    mapping_path = FULL / 'official/classes.csv'
    split_path = FOLD / 'training/inputs/split.csv'
    config_path = FOLD / 'training/artifacts/experiments/fullvisual20/config.yaml'
    metrics_path = config_path.with_name('metrics.json')
    checkpoint_path = config_path.with_name('last.pt')
    h1_path = FOLD / 'training/artifacts/experiments/mainline-h1/last.pt'
    h1_receipt_path = h1_path.with_name('h1_receipt.json')
    receipt_path = FOLD / 'LOCAL_CPU_AUDIT.json'
    pair_path = FOLD / 'LOCAL_PAIR_AUDIT.json'
    report, metrics, h1_receipt = map(read_json, [report_path, metrics_path, h1_receipt_path])
    receipt, pair = map(read_json, [receipt_path, pair_path])
    cache_sha = sha(cache)
    require(cache_sha == report['fused_logits']['sha256'], 'cache SHA mismatch')
    bundle = torch.load(cache, map_location='cpu', weights_only=True, mmap=True)
    require(bundle['schema'] == 'aictrack1.stage2_train_derived_fused_logits.v1', 'unexpected cache schema')
    require(bundle['scope'] == 'Stage 2 train-derived fixed validation only; no test access', 'not a held-out cache')
    require(tuple(bundle['logits'].shape) == (29737, 750), 'wrong logit shape')
    require(bundle['logits'].dtype == torch.float16 and bool(torch.isfinite(bundle['logits']).all()), 'invalid logits')
    require(bundle['targets'].dtype == torch.int64 and tuple(bundle['targets'].shape) == (29737,), 'invalid targets')
    require(len(bundle['sample_ids']) == len(set(bundle['sample_ids'])) == 29737, 'query IDs must be unique')
    require('four float logits equally averaged' in bundle['view_protocol'], 'not the expected four-view protocol')
    require(bundle['view_protocol'] == report['four_view_protocol'], 'report/bundle view mismatch')
    require(bundle['checkpoint_epoch'] == report['checkpoint_epoch'] == 20, 'not fixed epoch20')
    require(bundle['checkpoint_sha256'] == report['checkpoint_sha256'], 'checkpoint/report mismatch')
    checkpoint_sha = sha(checkpoint_path)
    require(checkpoint_sha == bundle['checkpoint_sha256'] == metrics['last_checkpoint_sha256'], 'actual fixed-last checkpoint SHA mismatch')
    require(metrics['training_scope'] == 'fixed_split_validation', 'full-train checkpoint cannot validate a holdout')
    require(metrics['train_count'] == 118958 and metrics['validation_count'] == 29737, 'incorrect training scope counts')
    require(sha(config_path) == bundle['experiment_config_sha256'] == metrics['config_sha256'], 'config mismatch')
    require(receipt['complete'] and pair['complete'] and not receipt['test_access'] and not pair['test_access'], 'incomplete provenance receipts')
    require(receipt['arms']['fullvisual20']['last_sha256'] == checkpoint_sha, 'CPU receipt does not bind this model')
    official, splits, classes = map(read_rows, [manifest_path, split_path, mapping_path])
    require(len(official) == len(splits) == 148695, 'official/split row count mismatch')
    by_id = {row['sample_id']: row for row in official}
    assignments = {row['sample_id']: row['split'] for row in splits}
    require(len(by_id) == len(assignments) == 148695 and set(by_id) == set(assignments), 'duplicate or absent manifest/split ID')
    require(Counter(assignments.values()) == {'train': 118958, 'validation': 29737}, 'split counts mismatch')
    require(all(row['source_split'] == 'train' and row['decode_status'] == 'ok' for row in official), 'invalid official train rows')
    sorted_classes = sorted(classes, key=lambda row: int(row['internal_class_index']))
    require([int(row['internal_class_index']) for row in sorted_classes] == list(range(750)), 'class indices invalid')
    mapping_digest = hashlib.sha256('\n'.join(f"{r['internal_class_index']}\t{r['official_class_id']}\t{r.get('class_name','')}" for r in sorted_classes).encode()).hexdigest()
    manifest_sha, split_sha = sha(manifest_path), sha(split_path)
    for key, expected in [('manifest_sha256', manifest_sha), ('split_sha256', split_sha), ('class_mapping_sha256', mapping_digest)]:
        require(bundle[key] == report[key] == expected, f'{key} mismatch')
    provenance = metrics['checkpoint_provenance']
    require(provenance['train_manifest_sha256'] == manifest_sha and provenance['split_sha256'] == split_sha and provenance['class_mapping_sha256'] == mapping_digest, 'training provenance mismatch')
    query_rows = [row for row in official if assignments[row['sample_id']] == 'validation']
    require([row['sample_id'] for row in query_rows] == bundle['sample_ids'], 'cache/official query order differs')
    require([int(row['internal_class_index']) for row in query_rows] == bundle['targets'].tolist(), 'cache/official targets differ')
    train_counts = Counter(int(row['internal_class_index']) for row in official if assignments[row['sample_id']] == 'train')
    require([train_counts[i] for i in range(750)] == bundle['train_class_counts'].tolist(), 'cache train class counts differ')
    group_partition = {}
    for key in ['duplicate_group', 'near_duplicate_group']:
        partitions = {}
        for row in official:
            if row[key]:
                partitions.setdefault(row[key], set()).add(assignments[row['sample_id']])
        require(all(len(parts) == 1 for parts in partitions.values()), f'{key} train/query overlap')
        group_partition[key] = len(partitions)
    image_counts = Counter(row['sha256'] for row in official)
    require(all(row['duplicate_group'] for row in official if image_counts[row['sha256']] > 1), 'a duplicate image lacks group evidence')
    h1_sha = sha(h1_path)
    require(h1_sha == h1_receipt['checkpoint_sha256'] == metrics['head_initializer']['sha256'], 'H1 SHA mismatch')
    h1 = torch.load(h1_path, map_location='cpu', weights_only=True)
    bound = [[r['sample_id'], int(r['internal_class_index']), assignments[r['sample_id']]] for r in official]
    train_bound = [r for r in bound if r[2] == 'train']
    require(h1['training_scope'] == h1_receipt['scope'] == 'fold_train_fixed_final_epochs', 'H1 is not fold-train-only')
    require(h1['row_binding']['counts'] == {'train': 118958, 'validation': 29737}, 'H1 count mismatch')
    require(h1['row_binding']['all_rows_sha256'] == row_digest(bound), 'H1 all-row binding mismatch')
    require(h1['row_binding']['train_rows_sha256'] == row_digest(train_bound), 'H1 train-row binding mismatch')
    require(h1['row_binding']['validation_used_for_optimization_or_selection'] is False, 'H1 used query supervision')
    require(h1['epoch'] == 100 and h1_receipt['training_seed'] == 20261008, 'H1 schedule/seed mismatch')
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=['sample_id', 'target', 'group', 'relative_path', 'near_duplicate_group', 'duplicate_group'], lineterminator='\n')
    writer.writeheader()
    for row in query_rows:
        ng, dg = row['near_duplicate_group'], row['duplicate_group']
        group = ('near:' + ng) if ng else (('exact:' + dg) if dg else ('sample:' + row['sample_id']))
        writer.writerow(dict(sample_id=row['sample_id'], target=row['internal_class_index'], group=group, relative_path=row['relative_path'], near_duplicate_group=ng, duplicate_group=dg))
    row_payload = buf.getvalue().encode('utf-8')
    rows_path = OUT / 'fv2_rows.csv'
    identity = dict(
        schema='aictrack1.p0_frozen_fold_input.v1', stage=2,
        bundle_path=str(cache), bundle_sha256=cache_sha,
        checkpoint_role='last', upstream_checkpoint_role=bundle['checkpoint_role'],
        checkpoint_role_note='upstream calls the fixed final-epoch checkpoint final; actual local artifact is last.pt and SHA is identical',
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha, checkpoint_epoch=20,
        split_path=str(split_path), split_sha256=split_sha,
        manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
        class_mapping_path=str(mapping_path), class_mapping_sha256=mapping_digest, class_mapping_file_sha256=sha(mapping_path),
        view_protocol=bundle['view_protocol'], row_csv=str(rows_path), row_csv_sha256=hashlib.sha256(row_payload).hexdigest(),
        sample_ids_order_sha256=hashlib.sha256('\n'.join(bundle['sample_ids']).encode()).hexdigest(),
        source_provenance_receipt=str(receipt_path), source_receipt_sha256=sha(receipt_path),
        report_path=str(report_path), report_sha256=sha(report_path), pair_audit_sha256=sha(pair_path),
        config_path=str(config_path), experiment_config_sha256=sha(config_path), metrics_path=str(metrics_path), metrics_sha256=sha(metrics_path),
        metrics_selection_note='trainer metrics selects best.pt; this diagnostic intentionally uses independently frozen epoch20 last.pt, verified by last_checkpoint_sha256 and CPU receipt',
        h1_path=str(h1_path), h1_sha256=h1_sha, h1_receipt_sha256=sha(h1_receipt_path),
        training_validation_overlap_rows=0, train_validation_exact_group_overlap=0, train_validation_near_group_overlap=0,
        holdout_verified=True, training_scope=metrics['training_scope'], training_count=118958, validation_count=29737,
        num_classes=750, logits_dtype='float16', group_source='official manifest near_duplicate_group, then duplicate_group; sample ID only for true singleton',
        group_counts=group_partition, source_script_sha256=sha(Path(__file__)),
        test_access=False, no_training=True, no_gpu=True,
    )
    write_exact(rows_path, row_payload)
    write_exact(OUT / 'fv2_input_identity.json', (json.dumps(identity, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
    print(json.dumps(dict(complete=True, rows_csv=str(rows_path), input_identity=str(OUT/'fv2_input_identity.json'), query_count=29737, holdout_verified=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
