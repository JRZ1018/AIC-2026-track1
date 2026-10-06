import io
import unittest

import torch

from aictrack1.models.residual_token_attention import ResidualTokenReadout


class ResidualAttentionTests(unittest.TestCase):
    def test_zero_start_exact_original_path(self):
        x=torch.randn(3,8);tokens=torch.randn(3,5,8)
        for mode in ('spatial','mean_channel'):
            module=ResidualTokenReadout(8,mode=mode)
            self.assertTrue(torch.equal(x,module(x,tokens)))

    def test_equal_small_parameter_count(self):
        for mode in ('spatial','mean_channel'):
            m=ResidualTokenReadout(512,mode=mode)
            self.assertEqual(sum(p.numel() for p in m.parameters()),513)

    def test_query_and_tokens_gradients_after_gain_opens(self):
        m=ResidualTokenReadout(8);m.gain.data.fill_(.2)
        tokens=torch.randn(2,4,8,requires_grad=True)
        m(torch.randn(2,8),tokens).square().sum().backward()
        for x in (tokens.grad,m.query.grad,m.gain.grad):
            self.assertTrue(bool(torch.isfinite(x).all()));self.assertGreater(float(x.abs().sum()),0)

    def test_gain_learns_at_zero_query_grad_delayed(self):
        m=ResidualTokenReadout(8)
        m(torch.ones(2,8),torch.ones(2,4,8)).sum().backward()
        self.assertGreater(float(m.gain.grad),0)
        self.assertEqual(float(m.query.grad.abs().sum()),0)

    def test_patch_permutation_equivariance(self):
        m=ResidualTokenReadout(8);m.gain.data.fill_(1)
        x=torch.randn(2,8);t=torch.randn(2,4,8)
        torch.testing.assert_close(m(x,t),m(x,t[:,[2,0,3,1]]))

    def test_bound_of_residual(self):
        m=ResidualTokenReadout(8);m.gain.data.fill_(100)
        x=torch.zeros(2,8);tokens=torch.ones(2,3,8)
        self.assertLessEqual(float(m(x,tokens).detach().abs().max()),.100001)

    def test_half_inputs_safe_and_finite(self):
        m=ResidualTokenReadout(8);m.gain.data.fill_(.2)
        x=torch.randn(2,8).half();t=torch.randn(2,4,8).half()
        y=m(x,t);self.assertEqual(y.dtype,x.dtype);self.assertTrue(bool(torch.isfinite(y).all()))

    def test_cpu_safe_restore(self):
        m=ResidualTokenReadout(8);m.gain.data.fill_(.2)
        f=io.BytesIO();torch.save(m.state_dict(),f);f.seek(0)
        other=ResidualTokenReadout(8);other.load_state_dict(torch.load(f,weights_only=True))
        x=torch.randn(2,8);t=torch.randn(2,3,8)
        self.assertTrue(torch.equal(m(x,t),other(x,t)))

    def test_invalid_shapes_and_mode(self):
        with self.assertRaises(ValueError): ResidualTokenReadout(mode='replace_backbone')
        m=ResidualTokenReadout(8)
        with self.assertRaises(ValueError): m(torch.randn(2,8),torch.randn(2,0,8))
        with self.assertRaises(ValueError): m(torch.randn(2,8),torch.randn(3,2,8))


if __name__=='__main__': unittest.main()
