"""CPU paired P1-vs-C, frozen P7 and fixed support-only queries. No new grid."""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fresh_diag_contract import (ROOT, SPLIT_SHA, ROWS_SHA, confirmation_rows,
    load_cache, save_json, sha, validate_pair, verify_owned_sources)
sys.path.insert(0, str(ROOT / 'reference'))
from p0_core import paired_metrics, soft_bias, support_report, unique_balanced_indices


def frozen_decode(scores):
    bias, diagnostics = soft_bias(scores, temperature=0.5, iterations=60)
    return (scores + 0.75 * bias).argmax(axis=1), diagnostics


def build_queries(control, *, seed=20261009, per_class=20):
    indices, columns = unique_balanced_indices(control['targets'], control['groups'], per_class, seed)
    return {'natural_indices': np.arange(len(control['ids']), dtype=np.int64),
            'natural_columns': np.arange(control['scores'].shape[1], dtype=np.int64),
            'balanced_indices': indices, 'balanced_columns': columns}


def prepare_queries(control, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    arrays = build_queries(control)
    np.savez_compressed(output / 'query_indices.npz', **arrays)
    protocols = []
    for name, prefix in [('natural_unique_images', 'natural'), ('balanced_supported_unique', 'balanced')]:
        indices, columns = arrays[prefix + '_indices'], arrays[prefix + '_columns']
        groups = [control['groups'][int(i)] for i in indices]
        ids = [control['ids'][int(i)] for i in indices]
        labels = np.searchsorted(columns, control['targets'][indices])
        support = support_report(labels, groups, ids, len(columns))
        if prefix == 'balanced' and (len(set(groups)) != len(groups) or not np.all(np.bincount(labels, minlength=len(columns)) == 20)):
            raise ValueError('balanced query lacks independent unique group support')
        with (output / f'{name}_queries.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['source_row', 'sample_id', 'target', 'group'])
            writer.writerows((int(i), control['ids'][int(i)], int(control['targets'][i]), control['groups'][int(i)]) for i in indices)
        protocols.append(dict(name=name, array_prefix=prefix, columns=columns.tolist(),
            excluded_classes=np.setdiff1d(np.arange(control['scores'].shape[1]), columns).tolist(), support=support,
            query_csv_sha256=sha(output / f'{name}_queries.csv')))
    identity = dict(schema='aictrack1.confirmation.frozen_queries.v1', split_sha256=SPLIT_SHA,
        confirmation_rows_sha256=ROWS_SHA, control_ids_sha256=control['identity']['sample_ids_order_sha256'],
        query_indices_sha256=sha(output / 'query_indices.npz'), seed=20261009, per_class=20,
        selection_uses_predictions=False, selection_protocol_sha256=sha(ROOT / 'selection_protocol.json'),
        protocols=protocols, test_access=False)
    save_json(output / 'query_identity.json', identity)
    return identity


def load_queries(control, query_directory):
    directory = Path(query_directory)
    identity = json.loads((directory / 'query_identity.json').read_text(encoding='utf-8'))
    if (identity.get('schema') != 'aictrack1.confirmation.frozen_queries.v1'
            or identity['split_sha256'] != SPLIT_SHA
            or identity['confirmation_rows_sha256'] != ROWS_SHA
            or identity['selection_uses_predictions'] is not False
            or identity['selection_protocol_sha256'] != sha(ROOT / 'selection_protocol.json')
            or sha(directory / 'query_indices.npz') != identity['query_indices_sha256']
            or control['identity']['sample_ids_order_sha256'] != identity['control_ids_sha256']):
        raise ValueError('frozen query identity mismatch')
    with np.load(directory / 'query_indices.npz', allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    recomputed = build_queries(control)
    if set(arrays) != set(recomputed) or any(not np.array_equal(arrays[k], recomputed[k]) for k in arrays):
        raise ValueError('query selection differs from pre-registered support-only rule')
    for protocol in identity['protocols']:
        if sha(directory / (protocol['name'] + '_queries.csv')) != protocol['query_csv_sha256']:
            raise ValueError('frozen query CSV drift')
    return arrays, identity


def compare_arrays(candidate, control, queries, *, draws=500, only_protocol=None):
    results = []
    for name, prefix in [('natural_unique_images', 'natural'), ('balanced_supported_unique', 'balanced')]:
        if only_protocol is not None and name != only_protocol:
            continue
        indices, columns = queries[prefix + '_indices'], queries[prefix + '_columns']
        j_scores = candidate['scores'][indices][:, columns].copy()
        c_scores = control['scores'][indices][:, columns].copy()
        targets = np.searchsorted(columns, control['targets'][indices])
        groups = [control['groups'][int(i)] for i in indices]
        j_p7, j_diagnostics = frozen_decode(j_scores)
        c_p7, c_diagnostics = frozen_decode(c_scores)
        metrics = paired_metrics(j_p7, c_p7, targets, groups, c_scores, draws=draws, seed=20261009)
        raw = paired_metrics(j_scores.argmax(1), c_scores.argmax(1), targets, groups, c_scores, draws=draws, seed=20261009)
        control_metrics = paired_metrics(c_p7, c_p7, targets, groups, c_scores, draws=0, seed=20261009)
        class_counts = np.asarray(control.get('train_counts', np.bincount(control['targets'], minlength=control['scores'].shape[1])))
        order = sorted(range(len(class_counts)), key=lambda i: (-int(class_counts[i]), i))
        head_count, mid_count = max(1, round(len(order) * .2)), max(1, round(len(order) * .3))
        strata = {label: ('head' if pos < head_count else 'mid' if pos < head_count + mid_count else 'tail')
                  for pos, label in enumerate(order)}
        tail_diagnostics = {}
        j_recall, c_recall = metrics['per_class_recall'], control_metrics['per_class_recall']
        group_support = support_report(targets, groups, [control['ids'][int(i)] for i in indices], len(columns))['unique_groups_per_class']
        class_rows = [dict(original_class_index=int(label), stratum=strata[int(label)],
            unique_group_support=int(group_support[k]), candidate_recall=j_recall[k], control_recall=c_recall[k],
            candidate_correct=int(np.sum((j_p7==targets)&(targets==k))), control_correct=int(np.sum((c_p7==targets)&(targets==k))), query_images=int(np.sum(targets==k)),
            delta_pp=None if j_recall[k] is None else 100*int(np.sum((j_p7==targets)&(targets==k))-np.sum((c_p7==targets)&(targets==k)))/int(np.sum(targets==k)))
            for k, label in enumerate(columns)]
        for stratum in ('head', 'mid', 'tail'):
            members = [k for k, label in enumerate(columns) if strata[int(label)] == stratum and j_recall[k] is not None]
            mask = np.isin(targets, members)
            tail_diagnostics[stratum] = dict(evaluated_classes=len(members), query_images=int(mask.sum()),
                candidate_class_mean_recall_pp=None if not members else float(np.mean([j_recall[k] for k in members]) * 100),
                control_class_mean_recall_pp=None if not members else float(np.mean([c_recall[k] for k in members]) * 100),
                delta_class_mean_recall_pp=None if not members else float(np.mean([j_recall[k] - c_recall[k] for k in members]) * 100),
                delta_sample_top1_pp=None if not mask.any() else float(np.mean((j_p7[mask] == targets[mask]).astype(int) - (c_p7[mask] == targets[mask]).astype(int)) * 100))
        row = dict(name=name, support=support_report(targets, groups, [control['ids'][int(i)] for i in indices], len(columns)),
            columns_original_indices=columns.tolist(), omitted_classes=np.setdiff1d(np.arange(control['scores'].shape[1]), columns).tolist(),
            limitation=('natural long-tail prior differs from the P7 uniform target; group dependence bootstrapped'
                        if prefix == 'natural' else 'closed-set sufficiently supported subset, not a full750 metric; independent new confirmation fold'),
            p7=dict(candidate_vs_control=metrics, control=control_metrics,
                    candidate_decoder=j_diagnostics, control_decoder=c_diagnostics),
            p7_head_mid_tail=tail_diagnostics,
            p7_per_class_support_and_recall=class_rows,
            supported_large_declines=[r for r in class_rows if r['unique_group_support'] >= 20 and r['delta_pp'] is not None and r['delta_pp'] <= -10.],
            tail_rule='training counts descending; class-index ties; 20% head/30% mid/50% tail, identical to training-source stratify_classes',
            raw_argmax_candidate_vs_control=raw,
            predictions=dict(candidate_p7=columns[j_p7], control_p7=columns[c_p7],
                             candidate_raw=columns[j_scores.argmax(1)], control_raw=columns[c_scores.argmax(1)]))
        results.append(row)
    return results


def eligibility(results):
    delta=next(r for r in results if r['name']=='balanced_supported_unique')['p7']['candidate_vs_control']['delta_vs_baseline_pp']
    natural=next(r for r in results if r['name']=='natural_unique_images')['p7']['candidate_vs_control']['delta_vs_baseline_pp']
    ci=next(r for r in results if r['name']=='balanced_supported_unique')['p7']['candidate_vs_control']['group_bootstrap']['ci95_pp']
    return dict(status=('fresh_registered_direction_positive' if delta>0 else 'fresh_registered_direction_not_positive'),
        primary_delta_pp=delta,natural_p7_delta_pp=natural,primary_metric='balanced_supported_unique P7 observed-label agreement',
        registered_rule='preselected route: raw P7 direction reported; S uses this as primary, J5_GRAPH uses direct fixed graph combo versus fresh C as primary',
        natural_nonnegative_hard_gate=False,ci_lower_bound_positive_hard_gate=False,
        group_ci95_pp=ci,ci_crosses_zero=(ci is not None and ci[0]<=0<=ci[1]),
        stable_gain_claim=False,formal_test_frozen=False,low_support_classes_do_not_alone_veto=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-bundle', required=True, type=Path)
    parser.add_argument('--control-identity', required=True, type=Path)
    parser.add_argument('--candidate-bundle', type=Path)
    parser.add_argument('--candidate-identity', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--query-directory', type=Path, default=ROOT / 'inputs' / 'frozen_queries')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads != 2:
        raise ValueError('bounded CPU thread setting required')
    torch.set_num_threads(args.threads)
    control = load_cache(args.control_bundle, args.control_identity)
    if control['identity']['model_arm'] != 'C':
        raise ValueError('fresh C control required')
    source_sha = verify_owned_sources()
    if args.candidate_bundle is None or args.candidate_identity is None:
        raise ValueError('candidate cache and positive audit identity required')
    candidate = load_cache(args.candidate_bundle, args.candidate_identity)
    validate_pair(candidate, control)
    if any(data['identity']['diagnostic_source_manifest_sha256'] != source_sha for data in (candidate,control)):
        raise ValueError('both caches must originate from this exact frozen diagnostic source')
    queries, query_identity = load_queries(control, args.query_directory)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    results = compare_arrays(candidate, control, queries)
    for row in results:
        predictions = row.pop('predictions')
        np.savez_compressed(args.output / (row['name'] + '_predictions.npz'), **predictions)
    report = dict(schema='aictrack1.fresh_route.raw_paired.v1', complete=True, test_access=False, command=sys.argv,
        diagnostic_source_manifest_sha256=source_sha, selection_protocol_sha256=sha(ROOT / 'selection_protocol.json'),
        p1_arm=candidate['identity']['p1_arm'], selected_arm=candidate['identity']['model_arm'], selected_route=candidate['identity']['selected_route'],
        admission_receipt_sha256=candidate['identity']['admission_receipt_sha256'], source_h1_manifest_sha256=candidate['identity']['source_h1_manifest_sha256'],
        split_sha256=SPLIT_SHA, h1_sha256=candidate['identity']['h1_sha256'],
        training_source_manifest_sha256=candidate['identity']['training_source_manifest_sha256'],
        primary_scope=('S_P7_vs_C' if candidate['identity']['selected_route']=='S' else 'raw_J5_P7_vs_C_diagnostic_only'),
        raw_report_is_graph_confirmation_gate=False,
        frozen_postprocess_sha256=sha(ROOT/'frozen_postprocess.json'), used_test_or_online_scores=False,
        candidate=dict(candidate['provenance'],provenance=candidate['provenance']),
        control=dict(control['provenance'],provenance=control['provenance']),
        query_identity=query_identity, frozen_decoder=dict(name='P7', temperature=.5, strength=.75, iterations=60),
        registered_before_candidate_results=True, protocols=results, eligibility=eligibility(results),
        elapsed_seconds=time.monotonic() - started, environment=dict(torch=torch.__version__, numpy=np.__version__, cpu_threads=args.threads))
    save_json(args.output / 'report.json', report)
    save_json(args.output / 'result_manifest.json', dict(schema='aictrack1.fresh_route.raw_result.v1', complete=True,
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print(json.dumps(dict(complete=True, p1_arm=report['p1_arm'], eligibility=report['eligibility'],
                         report=str(args.output / 'report.json')), ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
