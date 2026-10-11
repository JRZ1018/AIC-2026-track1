"""Exactly one externally frozen candidate vs P7 on a fresh verified fold.

No grid, no best-candidate selection. Root chooses the specification BEFORE
reading this fresh fold. Identity must differ from prior exploration identity.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from p0_core import paired_metrics, support_report, unique_balanced_indices
from p0_frozen import P7_DEFAULT, apply_frozen
from run_p0_audit import load_input, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'identity', 'specification', 'exploration_identity', 'output'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--subtask-per-class', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    scores, targets, ids, groups, provenance = load_input(args.bundle, args.identity)
    exploration = json.loads(args.exploration_identity.read_text(encoding='utf-8-sig'))
    if (exploration['bundle_sha256'] == provenance['identity']['bundle_sha256']
            or exploration['split_sha256'] == provenance['identity']['split_sha256']):
        raise ValueError('fresh confirmation requires a distinct heldout split and cache')
    specification = json.loads(args.specification.read_text(encoding='utf-8-sig'))
    if specification.get('selection_frozen_before_confirmation') is not True:
        raise ValueError('caller must record prior candidate freeze')
    indices, columns = unique_balanced_indices(targets, groups, args.subtask_per_class, 20261010)
    protocols = [('balanced_supported_unique', indices, columns),
                 ('natural_all_unique', np.arange(len(scores)), np.arange(scores.shape[1]))]
    args.output.mkdir(parents=True)
    report = dict(schema='aictrack1.p0_fresh_confirmation.v1', test_access=False,
          provenance=provenance, specification=specification,
          specification_sha256=sha256(args.specification), exploration_identity_sha256=sha256(args.exploration_identity),
          no_grid_search=True, limitations='same data pool splits may overlap; report is fixed-model conditional agreement',
          protocols=[])
    for name, indices, columns in protocols:
        values = scores[indices][:, columns].copy()
        labels = np.searchsorted(columns, targets[indices])
        query_groups = [groups[int(i)] for i in indices]
        p7, _ = apply_frozen(values, P7_DEFAULT)
        prediction, decoder = apply_frozen(values, specification)
        metric = paired_metrics(prediction, p7, labels, query_groups, values, 500, 20261010)
        report['protocols'].append(dict(name=name, columns_original_indices=columns.tolist(),
            support=support_report(labels, query_groups, [ids[int(i)] for i in indices], len(columns)),
            candidate_vs_p7=metric, decoder=decoder,
            limitation='balanced task is a disclosed closed-set class subset; natural task has prior mismatch'))
        with (args.output / f'{name}_queries.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source_row', 'sample_id', 'group'])
            writer.writerows((int(i), ids[int(i)], groups[int(i)]) for i in indices)
        np.savez_compressed(args.output / f'{name}_predictions.npz', p7=columns[p7], candidate=columns[prediction])
    save_json(args.output / 'report.json', report)
    save_json(args.output / 'manifest.json', dict(test_access=False, files=[dict(path=p.name,
              sha256=sha256(p), bytes=p.stat().st_size) for p in sorted(args.output.iterdir()) if p.is_file()]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
