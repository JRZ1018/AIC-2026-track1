"""Fast synthetic contracts only; not an effect or real-CUDA claim."""
import copy
import io
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from aictrack1.methods.grouped_training_pool import (
    select_pool, exposure_order, aggregate_observations, verify_protocol, bound_records,
)
from aictrack1.methods.confidence import ClasswiseHistoryWeighting


def rows():
    return [dict(sample_id=f'id{i:03}', source_split='train', partition='train',
                 internal_class_index=i // 10, sha256=f'sha{i}', duplicate_group='',
                 near_duplicate_group='') for i in range(30)]


class PoolContracts(unittest.TestCase):
    def test_bound_payload_and_path(self):
        source = rows()
        positions, protocol = select_pool(source, num_classes=3)
        records = [SimpleNamespace(**r) for r in source]
        payload = dict(protocol=protocol, selected_ids=[source[i]['sample_id'] for i in positions])
        with TemporaryDirectory() as d:
            p = Path(d)/'pool.json'; p.write_text(json.dumps(payload), encoding='utf-8')
            setting = dict(enabled=True,path='pool.json',sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            selected, actual = bound_records(records,setting,root=d)
            self.assertEqual(actual, protocol)
            self.assertEqual([r.sample_id for r in selected],payload['selected_ids'])
            with self.assertRaises(ValueError): bound_records(records,dict(setting,sha256='wrong'),root=d)
            for bad in ('../pool.json',str(p.resolve())):
                with self.assertRaises(ValueError): bound_records(records,dict(setting,path=bad),root=d)
            records[0].partition='validation'
            with self.assertRaises(ValueError): bound_records(records,setting,root=d)

    def test_whole_transitive_groups(self):
        r = rows(); r[0]['sha256'] = r[10]['sha256']; r[10]['near_duplicate_group'] = r[20]['near_duplicate_group'] = 'n'
        selected, p = select_pool(r, num_classes=3)
        self.assertEqual(len({i in selected for i in (0, 10, 20)}), 1)
        self.assertTrue(all(n >= 7 for n in p['class_counts']))

    def test_full_and_deterministic(self):
        a, p = select_pool(rows(), num_classes=3, numerator=10)
        self.assertEqual(a, list(range(30)))
        self.assertEqual(select_pool(rows(), num_classes=3), select_pool(rows(), num_classes=3))
        self.assertEqual(p['selected_count'], 30)

    def test_tail_three(self):
        r = rows()[:4]
        for a in r: a['internal_class_index'] = 0
        selected, p = select_pool(r, num_classes=1)
        self.assertEqual(len(selected), 3)
        self.assertEqual(p['class_counts'], [3])

    def test_reject_validation_and_missing_class(self):
        r = rows(); r[0]['partition'] = 'validation'
        with self.assertRaises(ValueError): select_pool(r, num_classes=3)
        with self.assertRaises(ValueError): select_pool(rows(), num_classes=4)

    def test_exposures_and_rng(self):
        before = torch.get_rng_state().clone()
        a = exposure_order(21, 30, seed=20261005, epoch=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(a), 30)
        counts = torch.bincount(a, minlength=21)
        self.assertEqual(int(counts.min()), 1); self.assertEqual(int(counts.max()), 2)
        self.assertTrue(torch.equal(a, exposure_order(21, 30, seed=20261005, epoch=0)))
        self.assertFalse(torch.equal(a, exposure_order(21, 30, seed=20261005, epoch=1)))

    def test_full_order_once(self):
        self.assertTrue(torch.equal(exposure_order(30, 30, seed=5, epoch=19).sort().values, torch.arange(30)))

    def test_mean_and_modal_tie(self):
        a = aggregate_observations(torch.tensor([10, 30]), torch.tensor([30, 10, 30, 10]),
                                   torch.tensor([4., 1., 2., 3.]), torch.tensor([.8, .1, .4, .3]),
                                   torch.tensor([2, 1, 0, 1]), num_classes=3)
        self.assertTrue(torch.equal(a[1], torch.tensor([2., 3.])))
        torch.testing.assert_close(a[2], torch.tensor([.2, .6]))
        self.assertTrue(torch.equal(a[3], torch.tensor([1, 0])))

    def test_once_exact_ac(self):
        ids = torch.tensor([10, 20, 30]); cls = torch.tensor([0, 0, 1])
        observed = torch.tensor([30, 10, 20]); losses = torch.tensor([1., 3., 2.]); probs = torch.tensor([.8, .2, .4]); pred = torch.tensor([1, 1, 0])
        a = ClasswiseHistoryWeighting(ids, cls); b = ClasswiseHistoryWeighting(ids, cls)
        a.update_epoch(observed, losses, probs, pred)
        aggregated = aggregate_observations(ids, observed, losses, probs, pred, num_classes=2)
        for actual, expected in zip(aggregated[:4], (observed, losses, probs, pred)): self.assertTrue(torch.equal(actual, expected))
        b.update_epoch(*aggregated[:4])
        for k, v in a.state_dict().items():
            if isinstance(v, torch.Tensor): self.assertTrue(torch.equal(v, b.state_dict()[k]))

    def test_unknown_missing_and_nan(self):
        def run(ids, losses): return aggregate_observations(torch.tensor([10, 20]), torch.tensor(ids), torch.tensor(losses), torch.tensor([.5]*len(ids)), torch.tensor([0]*len(ids)), num_classes=2)
        for ids, losses in [([10], [1.]), ([10, 30], [1., 1.]), ([10, 20], [float('nan'), 1.])]:
            with self.assertRaises(ValueError): run(ids, losses)

    def test_safe_protocol_and_ac_resume(self):
        _, p = select_pool(rows(), num_classes=3)
        a = ClasswiseHistoryWeighting(torch.arange(3), torch.tensor([0, 1, 2]))
        a.update_epoch(torch.arange(3), torch.ones(3), torch.ones(3)*.5, torch.arange(3))
        buf = io.BytesIO(); torch.save(dict(protocol=p, ac=a.state_dict()), buf); buf.seek(0)
        state = torch.load(buf, map_location='cpu', weights_only=True)
        verify_protocol(state['protocol'], p)
        b = ClasswiseHistoryWeighting(torch.arange(3), torch.tensor([0, 1, 2])); b.load_state_dict(state['ac'])
        self.assertTrue(torch.equal(a.weights, b.weights)); self.assertEqual(b.epochs_updated, 1)
        bad = copy.deepcopy(p); bad['selected_ids_sha256'] = 'wrong'
        with self.assertRaises(ValueError): verify_protocol(state['protocol'], bad)


if __name__ == '__main__': unittest.main()
