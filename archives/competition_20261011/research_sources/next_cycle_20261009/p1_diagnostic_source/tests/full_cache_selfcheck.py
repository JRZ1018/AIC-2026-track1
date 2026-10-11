"""CPU-only accepted C-vs-itself end-to-end decode/paired/query sanity check."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1_diag_contract import load_cache, save_json, sha
from compare_p1 import compare_arrays, load_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.monotonic()
    control = load_cache(args.bundle, args.identity)
    queries, identity = load_queries(control, ROOT / 'inputs' / 'frozen_queries')
    results = compare_arrays(control, control, queries)
    for row in results:
        for key in ('p7', 'raw_argmax_candidate_vs_control'):
            metric = row[key]['candidate_vs_control'] if key == 'p7' else row[key]
            if (metric['delta_vs_baseline_pp'] != 0. or metric['changed_predictions'] != 0
                    or metric['wrong_to_correct'] or metric['correct_to_wrong']
                    or metric['group_bootstrap']['ci95_pp'] != [0., 0.]):
                raise RuntimeError('C-vs-C must have exactly zero paired differences')
        row.pop('predictions')
    receipt = dict(schema='aictrack1.p1.cpu_full_cache_selfcheck.v1', complete=True,
        purpose='validation of diagnostic engineering, not a P1 result or new candidate',
        cuda_used=False, model_loaded=False, images_opened=False, training_steps=0, test_access=False,
        cache_sha256=sha(args.bundle), identity_sha256=sha(args.identity), query_identity=identity,
        protocols=results, elapsed_seconds=time.monotonic() - start)
    save_json(args.output, receipt)
    print(json.dumps(dict(complete=True, elapsed_seconds=receipt['elapsed_seconds'],
        protocols=[dict(name=r['name'], rows=r['support']['rows'], delta_pp=r['p7']['candidate_vs_control']['delta_vs_baseline_pp']) for r in results])))


if __name__ == '__main__':
    main()
