"""CPU contracts only; these tests do not establish SAM training efficacy."""
import io
import unittest

import torch
from aictrack1.methods.sharpness_neighborhood import ReversibleL2Neighborhood


class SharpnessNeighborhoodTests(unittest.TestCase):
    def make(self, **kwargs):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        g = torch.tensor([3., 4.])
        return p, g, ReversibleL2Neighborhood([('p', p)], [g], **kwargs)

    def test_direction_and_exact_restore(self):
        p, g, probe = self.make()
        original = p.detach().clone()
        with probe:
            torch.testing.assert_close(p, original + .01*g)
        self.assertTrue(torch.equal(p, original))

    def test_negative_direction(self):
        p, g, probe = self.make(direction=-1)
        with probe:torch.testing.assert_close(p, torch.tensor([.97, 1.96]))

    def test_exception_restore(self):
        p, _, probe = self.make()
        original = p.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with probe:raise RuntimeError('injected')
        self.assertTrue(torch.equal(p, original))

    def test_disabled_loss_gradient_exact(self):
        p, _, probe = self.make(rho=0.)
        expected = p.square().sum()
        grad = torch.autograd.grad(expected, p)[0]
        with probe:
            actual = p.square().sum()
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(grad, torch.autograd.grad(actual, p)[0]))

    def test_zero_gradient(self):
        p, _, _ = self.make()
        original = p.detach().clone()
        with ReversibleL2Neighborhood([('p', p)], [torch.zeros_like(p)]):
            self.assertTrue(torch.equal(p, original))

    def test_global_norm_and_frozen_unchanged(self):
        p = torch.nn.Parameter(torch.ones(2))
        q = torch.nn.Parameter(torch.ones(1))
        frozen = torch.nn.Parameter(torch.tensor([9.]), requires_grad=False)
        with ReversibleL2Neighborhood([('p', p), ('q', q)],
                                     [torch.tensor([3., 0.]), torch.tensor([4.])]) as v:
            self.assertEqual(v.gradient_norm, 5.)
            self.assertEqual(float(frozen), 9.)

    def test_finite_second_gradient_single_optimizer(self):
        p, _, _ = self.make()
        opt = torch.optim.AdamW([p], lr=.001)
        g = torch.autograd.grad(p.square().sum(), p)[0]
        with ReversibleL2Neighborhood([('p', p)], [g]):
            second = torch.autograd.grad(p.square().sum(), p)[0]
        p.grad = second
        self.assertTrue(torch.isfinite(second).all())
        opt.step()
        self.assertEqual(float(opt.state[p]['step']), 1.)

    def test_checkpoint_safe_and_drift_rejected(self):
        _, _, probe = self.make()
        b = io.BytesIO();torch.save(probe.state_dict(), b);b.seek(0)
        probe.load_state_dict(torch.load(b, weights_only=True))
        state = probe.state_dict();state['rho'] = .1
        with self.assertRaises(ValueError):probe.load_state_dict(state)
        with probe:
            with self.assertRaises(RuntimeError):probe.state_dict()
            with self.assertRaises(RuntimeError):probe.__enter__()

    def test_invalid_inputs_before_mutation(self):
        p, _, _ = self.make();original = p.detach().clone()
        for g in (torch.tensor([float('nan'), 0.]), torch.zeros(3)):
            with self.assertRaises(ValueError):ReversibleL2Neighborhood([('p', p)], [g])
        with self.assertRaises(ValueError):ReversibleL2Neighborhood([('p', p), ('q', p)], [torch.ones(2)]*2)
        with self.assertRaises(ValueError):ReversibleL2Neighborhood([('p', p)], [torch.ones(2)], rho=-1.)
        p.requires_grad_(False)
        with self.assertRaises(ValueError):ReversibleL2Neighborhood([('p', p)], [torch.ones(2)])
        self.assertTrue(torch.equal(p, original))

    def test_rng_preserved(self):
        p, _, probe = self.make()
        state = torch.random.get_rng_state().clone()
        with probe:torch.autograd.grad(p.square().sum(), p)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

if __name__ == '__main__':unittest.main()
