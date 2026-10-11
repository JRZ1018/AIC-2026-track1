"""CPU-only tests, no images/models/data downloaded."""
import io
import random
import unittest

import torch

from aictrack1.methods.mixup_supervision import (
    PROTOCOL, batch_plan, mix_inputs, mixed_weighted_sce, verify_protocol_state,
)
from aictrack1.methods.sce import symmetric_cross_entropy


class MixupTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(27)
        self.z = torch.randn(4, 7, generator=gen, requires_grad=True)
        self.y = torch.tensor([0, 2, 4, 6])
        self.w = torch.tensor([1., .1, .5, .02])
        self.p = torch.tensor([2, 0, 3, 1])

    def test_disable_loss_and_gradient_exact(self):
        base = (symmetric_cross_entropy(self.z,self.y,alpha=.1,beta=1,log_zero=-4,reduction='none')*self.w).sum()/self.w.sum()
        actual = mixed_weighted_sce(self.z,self.y,self.w,self.p,1.)
        self.assertTrue(torch.equal(actual,base))
        self.assertTrue(torch.equal(torch.autograd.grad(actual,self.z)[0],torch.autograd.grad(base,self.z)[0]))

    def test_disable_images_same_object(self):
        self.assertIs(mix_inputs(self.z,self.p,1.),self.z)

    def test_zero_endpoint(self):
        actual = mixed_weighted_sce(self.z,self.y,self.w,self.p,0.)
        base = (symmetric_cross_entropy(self.z,self.y[self.p],alpha=.1,beta=1,log_zero=-4,reduction='none')*self.w[self.p]).sum()/self.w.sum()
        torch.testing.assert_close(actual,base)

    def test_finite_mixed_gradient(self):
        value = mixed_weighted_sce(self.z,self.y,self.w,self.p,.9)
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(torch.autograd.grad(value,self.z)[0]).all())

    def test_mass_preserved(self):
        for coefficient in [0.,.5,.9,1.]:
            torch.testing.assert_close((coefficient*self.w+(1-coefficient)*self.w[self.p]).sum(),self.w.sum())

    def test_rng_private_and_reproducible(self):
        py = random.getstate(); th = torch.random.get_rng_state().clone()
        a,p = batch_plan(64,20261005,0,3);b,q = batch_plan(64,20261005,0,3)
        self.assertEqual(a,b);self.assertTrue(torch.equal(p,q))
        self.assertEqual(random.getstate(),py);self.assertTrue(torch.equal(torch.random.get_rng_state(),th))
        self.assertTrue(.5 <= a <= 1)

    def test_protocol_safe_roundtrip(self):
        data=io.BytesIO();torch.save(dict(PROTOCOL),data);data.seek(0)
        verify_protocol_state(torch.load(data,weights_only=True))

    def test_protocol_rejects_drift(self):
        with self.assertRaises(ValueError):verify_protocol_state({**PROTOCOL,'alpha':.4})

    def test_invalid_permutation(self):
        with self.assertRaises(ValueError):mix_inputs(self.z,torch.tensor([0,0,2,3]),.9)

    def test_invalid_coefficients_and_weights(self):
        for c in [float('nan'),float('inf'),-.1,1.1]:
            with self.assertRaises(ValueError):mix_inputs(self.z,self.p,c)
        with self.assertRaises(ValueError):mixed_weighted_sce(self.z,self.y,torch.zeros(4),self.p,.9)

if __name__ == '__main__':unittest.main()
