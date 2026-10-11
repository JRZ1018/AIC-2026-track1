"""Small exact optimizer, identity and independent-support verification."""
import itertools
import csv
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from p0_core import (InfeasibleGraph, capacity_bounds, decode_capacity, paired_metrics,
                     soft_bias, solve_sparse_capacity, support_report, unique_balanced_indices)
from run_p0_audit import load_input
from p0_frozen import P7_DEFAULT, apply_frozen


def brute(scores, lower, upper, k):
    edges = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    objectives = []
    for assignment in itertools.product(*edges.tolist()):
        counts = np.bincount(assignment, minlength=scores.shape[1])
        if (counts >= lower).all() and (counts <= upper).all():
            objectives.append(sum(float(scores[i, label]) for i, label in enumerate(assignment)))
    return max(objectives) if objectives else None


class TestP0(unittest.TestCase):
    def test_matches_exhaustive_all_small_graphs(self):
        rng = np.random.default_rng(810)
        for _ in range(12):
            scores = rng.normal(size=(6, 3)).astype(np.float32)
            for k in (2, 3):
                expected = brute(scores, 1, 3, k)
                if expected is None:
                    with self.assertRaises(InfeasibleGraph):
                        solve_sparse_capacity(scores, 1, 3, k, 10)
                else:
                    pred, report = solve_sparse_capacity(scores, 1, 3, k, 10)
                    self.assertAlmostEqual(report['raw_score_sum'], expected, places=5)
                    self.assertTrue(np.all(np.sum(pred[:, None] == np.argsort(-scores, kind='stable')[:, :k], axis=1) == 1))
                    self.assertLess(report['lp_decode_objective_gap'], 1e-5)

    def test_infeasible_topk_expands_and_preserves_objective(self):
        scores = np.array([[10., 9.], [10., 1.], [8., 7.], [10., 3.]], dtype=np.float32)
        pred, report = decode_capacity(scores, (1, 1), (1, 2), 10)
        self.assertEqual(report['top_k'], 2)
        self.assertEqual(report['attempts'][0]['status'], 'infeasible')
        self.assertEqual(np.bincount(pred).tolist(), [2, 2])
        self.assertAlmostEqual(report['raw_score_sum'], brute(scores, 2, 2, 2))

    def test_stable_tie_determinism(self):
        scores = np.ones((9, 3), dtype=np.float32)
        a, _ = solve_sparse_capacity(scores, 3, 3, 3, 10)
        b, _ = solve_sparse_capacity(scores, 3, 3, 3, 10)
        np.testing.assert_array_equal(a, b)

    def test_invalid_and_infeasible_bounds(self):
        for scores in (np.ones(3), np.array([[np.inf, 0.]])):
            with self.assertRaises(ValueError):
                decode_capacity(scores)
        with self.assertRaises(ValueError):
            solve_sparse_capacity(np.ones((3, 2)), 2, 2, 2)
        self.assertEqual(capacity_bounds(37444, 750, (.8, 1.2)), (39, 60))

    def test_unique_sampling_never_reuses_rows_or_groups(self):
        targets = np.repeat(np.arange(4), 5)
        groups = ['shared'] * 2 + [f'g{i}' for i in range(2, 20)]
        first, classes = unique_balanced_indices(targets, groups, 3, 42)
        again, _ = unique_balanced_indices(targets, groups, 3, 42)
        np.testing.assert_array_equal(first, again)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(first), len({groups[i] for i in first}))
        np.testing.assert_array_equal(np.bincount(targets[first]), [3] * 4)
        self.assertEqual(len(classes), 4)

    def test_duplicate_ID_support_rejected(self):
        with self.assertRaises(ValueError):
            support_report([0, 1], ['g0', 'g1'], ['a', 'a'], 2)

    def test_soft_parity_project_p7_formula(self):
        values = np.random.default_rng(3).normal(size=(8, 4)).astype(np.float32)
        actual, info = soft_bias(values, .5, 60)
        logits = torch.from_numpy(values)
        top2 = logits.topk(2, dim=1).values
        scale = max(float((top2[:, 0] - top2[:, 1]).median()) * .5, 1e-4)
        bias = torch.zeros(4)
        for _ in range(60):
            correction = -np.log(4) - torch.softmax(logits / scale + bias, 1).mean(0).clamp_min(1e-12).log()
            bias = bias + correction
            bias -= bias.mean()
            if float(correction.abs().max()) <= 1e-4:
                break
        np.testing.assert_allclose(actual, (bias * scale).numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(info['calculation_dtype'], 'float32')

    def test_frozen_interface_retains_default_and_rejects_new_grid(self):
        scores = np.random.default_rng(7).normal(size=(12, 3)).astype(np.float32)
        actual, _ = apply_frozen(scores, P7_DEFAULT)
        bias, _ = soft_bias(scores, .5, 60)
        np.testing.assert_array_equal(actual, (scores + .75*bias).argmax(1))
        with self.assertRaisesRegex(ValueError, 'unregistered soft point'):
            apply_frozen(scores, dict(P7_DEFAULT, temperature=.123))

    def test_group_bootstrap_and_flip_metrics(self):
        scores = np.array([[2, 1], [1, 2], [3, 1], [2, 3]], dtype=np.float32)
        report = paired_metrics([0, 1, 0, 1], [1, 1, 0, 0], [0, 1, 0, 1], ['a', 'a', 'b', 'c'], scores, 50)
        self.assertEqual(report['wrong_to_correct'], 2)
        self.assertEqual(report['correct_to_wrong'], 0)
        self.assertEqual(report['delta_vs_baseline_pp'], 50)
        self.assertEqual(report['group_bootstrap']['groups'], 3)

    def test_identity_hash_and_row_order_rejected(self):
        fixture_parent = Path(__file__).resolve().parent
        # Retained private fixture avoids Windows sandbox ACLs from mkdtemp(0700).
        with self.subTest('retained private identity fixture'):
            root = fixture_parent / 'identity_test_fixture'
            root.mkdir(exist_ok=True)
            root.resolve().relative_to(fixture_parent)
            rows = root / 'rows.csv'
            rows.write_text('sample_id,target,group\na,0,g0\nb,1,g1\n', encoding='utf-8')
            payload = dict(schema='aictrack1.stage2_train_derived_fused_logits.v1',
                scope='Stage2 train-derived validation', logits=torch.tensor([[1., 0.], [0., 1.]]),
                targets=torch.tensor([0, 1]), train_class_counts=torch.tensor([4, 4]),
                sample_ids=['a', 'b'], view_protocol='four fixed views',
                **{key: 'a' * 64 for key in ('checkpoint_sha256', 'split_sha256',
                'manifest_sha256', 'class_mapping_sha256', 'experiment_config_sha256')})
            bundle = root / 'logits.pt'
            torch.save(payload, bundle)
            identity = dict(schema='aictrack1.p0_frozen_fold_input.v1', stage=2,
                holdout_verified=True, training_validation_overlap_rows=0,
                checkpoint_role='last', test_access=False, source_receipt_sha256='b'*64,
                bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                training_count=8, validation_count=2, row_csv='rows.csv',
                row_csv_sha256=hashlib.sha256(rows.read_bytes()).hexdigest(),
                **{key: payload[key] for key in ('checkpoint_sha256', 'split_sha256',
                'manifest_sha256', 'class_mapping_sha256', 'experiment_config_sha256', 'view_protocol')})
            sidecar = root / 'identity.json'
            sidecar.write_text(json.dumps(identity), encoding='utf-8')
            self.assertEqual(load_input(bundle, sidecar)[0].shape, (2, 2))
            identity['test_access'] = True
            sidecar.write_text(json.dumps(identity), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'not test'):
                load_input(bundle, sidecar)
            identity['test_access'] = False
            rows.write_text('sample_id,target,group\nb,0,g0\na,1,g1\n', encoding='utf-8')
            identity['row_csv_sha256'] = hashlib.sha256(rows.read_bytes()).hexdigest()
            sidecar.write_text(json.dumps(identity), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'sample-ID order'):
                load_input(bundle, sidecar)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
