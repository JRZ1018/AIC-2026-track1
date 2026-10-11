"""One registered two-point temperature extension of the existing P0 queries."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from p0_core import paired_metrics, soft_bias
from run_p0_audit import load_input, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--parent-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    started = time.monotonic()
    scores, targets, ids, groups, provenance = load_input(args.bundle, args.identity)
    parent = json.loads(args.parent_report.read_text(encoding='utf-8-sig'))
    if (parent['provenance']['identity_sha256'] != provenance['identity_sha256']
            or parent['schema'] != 'aictrack1.p0_frozen_audit.v1'):
        raise ValueError('extension parent input identity differs')
    balanced = next(q for q in parent['protocols'] if q['name'] == 'balanced_supported_unique')
    best_soft = max((key for key in balanced['candidates'] if key.startswith('uniform_soft')),
                   key=lambda k: balanced['candidates'][k]['delta_vs_baseline_pp'])
    if best_soft != 'uniform_soft_t0.35_s0.75':
        raise ValueError('this registered extension only follows .35/.75 boundary result')
    args.output.mkdir(parents=True)
    result = dict(schema='aictrack1.p0_boundary_extension.v1', test_access=False,
                  parent_report_sha256=sha256(args.parent_report), provenance=provenance,
                  candidates_added=2, total_algorithm_specifications=14,
                  registered_points=[[.2, .75], [.25, .75]], default_unchanged='P7(.5,.75,60)',
                  purpose='exploration only; no freeze/test selection', protocols=[])
    for protocol in parent['protocols']:
        name = protocol['name']
        path = args.parent_report.parent / f'{name}_queries.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        indices = np.asarray([int(r['source_row']) for r in rows])
        if [r['sample_id'] for r in rows] != [ids[int(i)] for i in indices]:
            raise ValueError('parent query IDs differ')
        columns = np.asarray(protocol['columns_original_indices'])
        values = scores[indices][:, columns].copy()
        labels = np.searchsorted(columns, targets[indices])
        query_groups = [groups[int(i)] for i in indices]
        bias, _ = soft_bias(values, .5, 60)
        p7 = (values + .75 * bias).argmax(1)
        entry = dict(name=name, query_csv_sha256=sha256(path), support=protocol['support'],
                     limitation=protocol['limitation'], candidates={})
        predictions = {}
        for temperature in (.2, .25):
            key = f'uniform_soft_t{temperature:g}_s0.75'
            bias, diagnostics = soft_bias(values, temperature, 60)
            prediction = (values + .75 * bias).argmax(1)
            metric = paired_metrics(prediction, p7, labels, query_groups, values, 500, 20261009)
            metric['decoder'] = diagnostics
            entry['candidates'][key] = metric
            predictions[key] = columns[prediction]
        np.savez_compressed(args.output / f'{name}_predictions.npz', **predictions)
        result['protocols'].append(entry)
        print(json.dumps({'protocol': name, 'deltas_vs_p7_pp': {key: value['delta_vs_baseline_pp']
                           for key, value in entry['candidates'].items()}}), flush=True)
    result['elapsed_seconds'] = time.monotonic() - started
    save_json(args.output / 'report.json', result)
    save_json(args.output / 'manifest.json', dict(files=[dict(path=p.name, bytes=p.stat().st_size,
              sha256=sha256(p)) for p in sorted(args.output.iterdir()) if p.is_file()], test_access=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
