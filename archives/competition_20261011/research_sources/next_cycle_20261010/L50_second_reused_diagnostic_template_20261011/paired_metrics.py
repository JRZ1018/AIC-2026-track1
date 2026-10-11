"""Same registered queries and paired statistics; independent new610 confirmation, exact numerical decoder/statistics."""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
from fresh_diag_contract import ROOT,SPLIT_SHA,ROWS_SHA,confirmation_rows,sha,save_json
sys.path.insert(0,str(ROOT/'reference'))
from p0_core import paired_metrics,soft_bias,support_report,unique_balanced_indices

def frozen_decode(scores):
    bias, diagnostics = soft_bias(scores, temperature=0.5, iterations=60)
    return (scores + 0.75 * bias).argmax(axis=1), diagnostics

def build_queries(control, *, seed=20261010, per_class=20):
    indices, columns = unique_balanced_indices(control['targets'], control['groups'], per_class, seed)
    return {'natural_indices': np.arange(len(control['ids']), dtype=np.int64),
            'natural_columns': np.arange(control['scores'].shape[1], dtype=np.int64),
            'balanced_indices': indices, 'balanced_columns': columns}

def load_queries(control,query_directory):
 from fresh_diag_contract import QUERY_IDENTITY_SHA
 directory=Path(query_directory);path=directory/'query_identity.json';identity=json.loads(path.read_text(encoding='utf-8'))
 if sha(path)!=QUERY_IDENTITY_SHA or identity.get('schema')!='aictrack1.next_cycle10.pre_result_queries.v1' or identity.get('split_sha256')!=SPLIT_SHA or identity.get('selection_uses_predictions') is not False or identity.get('seed')!=20261010 or sha(directory/'query_indices.npz')!=identity['query_indices_sha256'] or hashlib.sha256(json.dumps(control['ids'],separators=(',',':'),ensure_ascii=True).encode()).hexdigest()!=identity['query_ids_order_sha256']:raise ValueError('pre-result new610 query identity/order')
 with np.load(directory/'query_indices.npz',allow_pickle=False) as stored:arrays={k:stored[k].copy() for k in stored.files}
 recomputed=build_queries(control)
 if set(arrays)!=set(recomputed) or any(not np.array_equal(arrays[k],recomputed[k]) for k in arrays):raise ValueError('predetermined support-only query selection differs')
 for label,name in [('natural','natural_unique_images'),('balanced','balanced_supported_unique')]:
  if sha(directory/(name+'_queries.csv'))!=identity[label]['csv_sha256']:raise ValueError('predetermined query CSV drift')
 return arrays,identity


def compare_arrays(candidate, control, queries, *, draws=500, only_protocol=None, control_predictions=None):
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
        if control_predictions is not None:
            original=np.asarray(control_predictions[name],dtype=np.int64)
            if original.shape!=(len(indices),) or not np.isin(original,columns).all():raise ValueError('bound graph control predictions shape/classes')
            c_p7=np.searchsorted(columns,original);c_diagnostics=dict(imported_actual_graph_P7=True,plain_J5_P7_not_used_as_primary=True)
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
                        if prefix == 'natural' else 'closed-set sufficiently supported subset, not a full750 metric; independent pre-result query; 12 insufficient classes omitted'),
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

