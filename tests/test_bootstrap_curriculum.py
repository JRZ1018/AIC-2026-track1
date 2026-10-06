import io
import unittest
import torch
from aictrack1.methods.confidence import ClasswiseHistoryWeighting
from aictrack1.methods.bootstrap_curriculum import FirstEpochCurriculum


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([2, 5, 8, 11])
        self.labels = torch.tensor([0, 0, 1, 1])
        self.ac = ClasswiseHistoryWeighting(self.ids, self.labels, min_weight=.02,
                                           magnitude_mix=.75)
        self.stats = (self.ids, torch.tensor([.2, 5., .1, 3.]),
                      torch.tensor([.8, .01, .9, .05]), torch.tensor([0, 1, 1, 0]))

    def make(self):
        return FirstEpochCurriculum.from_initial_statistics(self.ac, *self.stats)[0]

    def test_no_formal_history_or_rng_change(self):
        rng = torch.random.get_rng_state().clone()
        before = self.ac.state_dict()
        self.make()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for k, v in before.items():
            got = self.ac.state_dict()[k]
            self.assertTrue(torch.equal(v, got) if isinstance(v, torch.Tensor) else v == got)

    def test_epoch_boundary(self):
        b = self.make()
        self.assertIsNotNone(b.weights_for(self.ids, epoch_index=0))
        self.assertIsNone(b.weights_for(self.ids, epoch_index=1))
        self.assertIsNone(b.weights_for(self.ids, epoch_index=19))

    def test_safe_restore(self):
        b = self.make(); buf = io.BytesIO()
        torch.save(b.state_dict(), buf); buf.seek(0)
        state = torch.load(buf, weights_only=True)
        other = FirstEpochCurriculum(self.ids, self.labels, torch.ones(4))
        other.load_state_dict(state)
        self.assertTrue(torch.equal(b.index.weights, other.index.weights))

    def test_order_invariance(self):
        b = self.make(); order = torch.tensor([3, 0, 2, 1])
        other, _ = FirstEpochCurriculum.from_initial_statistics(
            self.ac, *(x[order] for x in self.stats))
        self.assertTrue(torch.equal(b.index.weights, other.index.weights))

    def test_exact_existing_formula(self):
        b = self.make()
        reference = ClasswiseHistoryWeighting(self.ids, self.labels, min_weight=.02,
                                             magnitude_mix=.75)
        reference.update_epoch(*self.stats)
        self.assertTrue(torch.equal(b.index.weights, reference.weights))

    def test_unknown_id_in_both_paths(self):
        b = self.make()
        for epoch in (0, 1):
            with self.assertRaises(ValueError):
                b.weights_for(torch.tensor([4]), epoch_index=epoch)

    def test_duplicate_or_incomplete_observations(self):
        for ids in (torch.tensor([2, 2, 8, 11]), self.ids[:3]):
            with self.assertRaises(ValueError):
                FirstEpochCurriculum.from_initial_statistics(self.ac, ids, *self.stats[1:])

    def test_nonfinite_and_negative_stats(self):
        for loss in (torch.tensor([float('nan'), 1., 1., 1.]), -torch.ones(4)):
            with self.assertRaises(ValueError):
                FirstEpochCurriculum.from_initial_statistics(self.ac, self.ids, loss, *self.stats[2:])

    def test_used_ac_refused(self):
        self.ac.update_epoch(*self.stats)
        with self.assertRaises(ValueError): self.make()

    def test_restore_drift_refused(self):
        b = self.make(); state = b.state_dict(); state['class_ids'][0] = 1
        with self.assertRaises(ValueError): b.load_state_dict(state)

    def test_weight_bounds(self):
        for w in (torch.zeros(4), torch.full((4,), 1.1), torch.full((4,), float('inf'))):
            with self.assertRaises(ValueError): FirstEpochCurriculum(self.ids, self.labels, w)

    def test_all_ones_exact_loss_gradient(self):
        b = FirstEpochCurriculum(self.ids, self.labels, torch.ones(4))
        x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
        w = b.weights_for(self.ids, epoch_index=0)
        self.assertTrue(torch.equal(x.mean(), (x*w).sum()/w.sum()))
        a = torch.autograd.grad(x.mean(), x)[0]
        c = torch.autograd.grad((x*w).sum()/w.sum(), x)[0]
        self.assertTrue(torch.equal(a, c))


if __name__ == '__main__': unittest.main()
