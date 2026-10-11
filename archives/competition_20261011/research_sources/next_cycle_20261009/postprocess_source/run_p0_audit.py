"""Bounded CPU-only audit of an identity-verified heldout four-view logit cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from p0_core import (InfeasibleGraph, SolverTimeout, decode_capacity, paired_metrics,
                     soft_bias, support_report, unique_balanced_indices, validate_scores)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_input(bundle_path, identity_path):
    identity_path = Path(identity_path).resolve()
    identity = json.loads(identity_path.read_text(encoding='utf-8-sig'))
    if identity.get('schema') != 'aictrack1.p0_frozen_fold_input.v1':
        raise ValueError('unsupported identity schema')
    if (identity.get('stage') != 2 or identity.get('holdout_verified') is not True
            or identity.get('training_validation_overlap_rows') != 0
            or identity.get('checkpoint_role') not in ('final', 'last')
            or identity.get('test_access') is not False):
        raise ValueError('input must be verified Stage2 heldout fixed final/last, not test')
    source_receipt = identity.get('source_receipt_sha256', '')
    if len(source_receipt) != 64 or not all(x in '0123456789abcdef' for x in source_receipt):
        raise ValueError('missing verified source receipt hash')
    if sha256(bundle_path) != identity['bundle_sha256']:
        raise ValueError('bundle SHA mismatch')
    payload = torch.load(bundle_path, map_location='cpu', weights_only=True)
    if (payload.get('schema') != 'aictrack1.stage2_train_derived_fused_logits.v1'
            or 'train-derived' not in str(payload.get('scope', ''))):
        raise ValueError('unsupported/nonheldout bundle')
    keys = ('checkpoint_sha256', 'split_sha256', 'manifest_sha256',
            'class_mapping_sha256', 'experiment_config_sha256', 'view_protocol')
    for key in keys:
        if identity[key] != payload.get(key):
            raise ValueError(f'identity mismatch: {key}')
    if 'four' not in str(payload.get('view_protocol', '')).lower():
        raise ValueError('fixed four-view cache required')
    scores = validate_scores(payload['logits'].float().numpy())
    targets = payload['targets'].long().numpy()
    train_counts = payload['train_class_counts'].long().numpy()
    if (targets.shape != (len(scores),) or train_counts.shape != (scores.shape[1],)
            or (targets < 0).any() or (targets >= scores.shape[1]).any()
            or (train_counts <= 0).any()):
        raise ValueError('invalid targets/training counts')
    row_path = Path(identity['row_csv'])
    if not row_path.is_absolute():
        row_path = identity_path.parent / row_path
    if sha256(row_path) != identity['row_csv_sha256']:
        raise ValueError('rows SHA mismatch')
    with row_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(scores):
        raise ValueError('ID/logit row mismatch')
    ids = [r['sample_id'] for r in rows]
    groups = [r['group'] for r in rows]
    if 'sample_ids' in payload and list(payload['sample_ids']) != ids:
        raise ValueError('bundle sample-ID order differs from verified rows')
    if len(set(ids)) != len(ids) or not all(ids) or not all(groups):
        raise ValueError('rows must have unique IDs and explicit verified groups')
    if not np.array_equal(np.asarray([int(r['target']) for r in rows]), targets):
        raise ValueError('row label order differs from logits')
    saved_argmax_mismatches = None
    if rows and 'four_view_prediction' in rows[0]:
        saved_argmax_mismatches = int(np.sum(scores.argmax(1) !=
                            np.asarray([int(r['four_view_prediction']) for r in rows])))
    if identity.get('validation_count') != len(scores) or identity.get('training_count') != int(train_counts.sum()):
        raise ValueError('training/validation count mismatch')
    return scores, targets, ids, groups, dict(identity=identity,
            identity_path=str(identity_path), identity_sha256=sha256(identity_path),
            input_dtype=str(payload['logits'].dtype), computation_dtype='float32',
            fp16_cache_argmax_mismatches_vs_saved_predictions=saved_argmax_mismatches,
            warning='FP32 conversion cannot recover precision discarded in FP16 cache')


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    scores, targets, ids, groups, provenance = load_input(args.bundle, args.identity)
    started = time.monotonic()
    output.mkdir(parents=True)
    protocols = [('natural_all_unique', np.arange(len(scores)), np.arange(scores.shape[1]),
                  'uniform target intentionally mismatches natural long-tail marginals; diagnostic only')]
    if not args.natural_only:
        subset, columns = unique_balanced_indices(targets, groups,
                                          per_class=args.subtask_per_class, seed=args.seed)
        # Spend the capacity budget on its matched-prior stress task first.
        protocols.insert(0, ('balanced_supported_unique', subset, columns,
                 'closed-set sufficiently-supported class subset; not a full 750-class metric'))
    temperatures = [0.35, 0.5, 0.75]
    strengths = [0.5, 0.75, 1.0]
    capacity_ratios = [(0.8, 1.2), (0.9, 1.1)]
    spent = {ratios: 0.0 for ratios in capacity_ratios}
    report = dict(schema='aictrack1.p0_frozen_audit.v1', test_access=False,
            provenance=provenance, command=sys.argv, environment=dict(python=platform.python_version(),
            numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
            cpu_threads=args.threads), registered_grid=dict(temperatures=temperatures, strengths=strengths,
            capacity_ratios=capacity_ratios, top_ks=[5, 20], candidate_count=12,
            capacity_seconds_per_variant_all_protocols=args.capacity_seconds), protocols=[])
    for name, indices, columns, limitation in protocols:
        query = scores[indices][:, columns].copy()
        labels = np.searchsorted(columns, targets[indices])
        query_groups = [groups[int(i)] for i in indices]
        query_ids = [ids[int(i)] for i in indices]
        raw = query.argmax(1)
        bias_cache = {}
        for temperature in temperatures:
            bias_cache[temperature] = soft_bias(query, temperature, 60)
        p7 = (query + 0.75 * bias_cache[0.5][0]).argmax(1)
        candidates = {'raw': raw}
        infos = {}
        for temperature in temperatures:
            bias, info = bias_cache[temperature]
            for strength in strengths:
                key = f'uniform_soft_t{temperature:g}_s{strength:g}'
                candidates[key] = (query + strength * bias).argmax(1)
                infos[key] = info
        failed = {}
        if not args.soft_only:
            for ratios in capacity_ratios:
                key = f'capacity_r{ratios[0]:g}_{ratios[1]:g}'
                remaining = args.capacity_seconds - spent[ratios]
                attempt_started = time.monotonic()
                try:
                    if remaining <= 0:
                        raise SolverTimeout('variant aggregate budget already exhausted')
                    candidates[key], infos[key] = decode_capacity(query, ratios, (5, 20), remaining)
                except (InfeasibleGraph, SolverTimeout) as exc:
                    failed[key] = dict(status=type(exc).__name__, message=str(exc))
                finally:
                    spent[ratios] += time.monotonic() - attempt_started
        protocol = dict(name=name, limitation=limitation,
                 columns_original_indices=columns.tolist(),
                 excluded_original_class_indices=np.setdiff1d(np.arange(scores.shape[1]), columns).tolist(),
                 support=support_report(labels, query_groups, query_ids, len(columns)),
                 status='completed', failed_candidates=failed, candidates={})
        with (output / f'{name}_queries.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source_row', 'sample_id', 'original_target', 'group'])
            writer.writerows((int(i), ids[int(i)], int(targets[int(i)]), groups[int(i)]) for i in indices)
        prediction_arrays = {}
        for key, prediction in candidates.items():
            metric = paired_metrics(prediction, p7, labels, query_groups, query,
                                    draws=args.bootstrap_draws, seed=args.seed)
            metric['delta_vs_raw_pp'] = float(((prediction == labels).mean() - (raw == labels).mean()) * 100)
            metric['decoder'] = infos.get(key, {'method': 'raw_argmax'})
            protocol['candidates'][key] = metric
            prediction_arrays[key] = columns[prediction]
        np.savez_compressed(output / f'{name}_predictions.npz', **prediction_arrays)
        report['protocols'].append(protocol)
        report['elapsed_seconds'] = time.monotonic() - started
        report['capacity_seconds_used'] = {str(k): v for k, v in spent.items()}
        save_json(output / 'report.json', report)
        print(json.dumps({'protocol': name, 'rows': len(indices), 'classes': len(columns),
                'best_observed_delta_vs_p7_pp': max(v['delta_vs_baseline_pp'] for v in protocol['candidates'].values()),
                'failed_candidates': failed}, ensure_ascii=False), flush=True)
    manifest = dict(schema='aictrack1.p0_result_manifest.v1', test_access=False,
                    files=[dict(path=p.name, bytes=p.stat().st_size, sha256=sha256(p))
                           for p in sorted(output.iterdir()) if p.is_file()])
    save_json(output / 'manifest.json', manifest)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--subtask-per-class', type=int, default=20)
    parser.add_argument('--capacity-seconds', type=float, default=3600)
    parser.add_argument('--bootstrap-draws', type=int, default=500)
    parser.add_argument('--soft-only', action='store_true')
    parser.add_argument('--natural-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or not 0 < args.capacity_seconds <= 3600 or args.bootstrap_draws < 0:
        parser.error('threads<=8, each capacity total<=3600s, bootstrap>=0 required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
