import io
import unittest
import torch
from torch import nn
from aictrack1.models.peft import PEFTImageClassifier
from aictrack1.models.token_attention_classifier import TokenAttentionClassifier, readout_parameter_groups
from aictrack1.engine.peft import trainable_state_dict, load_trainable_state_dict


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(4, 4)
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.eye(4), requires_grad=False)
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        tokens = self.transformer(x.permute(1,0,2))
        return self.ln_post(tokens[0]) @ self.proj


class IntegrationTests(unittest.TestCase):
    def test_zero_gain_baseline_and_one_forward(self):
        v=FakeVisual();h=nn.Linear(4,3);x=torch.randn(2,3,4)
        reference=PEFTImageClassifier(v,h,normalize_features=True)(x);v.calls=0
        m=TokenAttentionClassifier(v,h,normalize_features=True,mode='spatial')
        self.assertTrue(torch.equal(reference,m(x)))
        self.assertEqual(v.calls,1)
        self.assertEqual(len(v.transformer._forward_hooks),0)

    def test_none_same_keys_and_forward(self):
        v=FakeVisual();h=nn.Linear(4,3);x=torch.randn(2,3,4)
        a=PEFTImageClassifier(v,h);b=TokenAttentionClassifier(v,h)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        self.assertTrue(torch.equal(a(x),b(x)))
        self.assertEqual(readout_parameter_groups(b),[])

    def test_rng_preserved_and_modes_share_query(self):
        v=FakeVisual();h=nn.Linear(4,3);rng=torch.random.get_rng_state()
        a=TokenAttentionClassifier(v,h,mode='spatial');b=TokenAttentionClassifier(v,h,mode='mean_channel')
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(a.readout.query,b.readout.query))

    def test_all_parameters_optimizer_and_safe_restore(self):
        v=FakeVisual();h=nn.Linear(4,3)
        for p in h.parameters():p.requires_grad_(False)
        m=TokenAttentionClassifier(v,h,mode='spatial')
        groups=[dict(params=[p for p in v.parameters() if p.requires_grad],lr=1e-6)]+readout_parameter_groups(m)
        opt=torch.optim.AdamW(groups,weight_decay=.01)
        covered=[id(p) for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(covered),len(set(covered)))
        self.assertEqual(set(covered),{id(p) for p in m.parameters() if p.requires_grad})
        m.readout.gain.data.fill_(.1)
        loss=m(torch.randn(2,3,4)).square().mean();loss.backward();opt.step()
        self.assertTrue(torch.isfinite(m.readout.query.grad).all())
        self.assertGreater(float(m.readout.query.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in h.parameters()))
        b=io.BytesIO();torch.save(dict(model=trainable_state_dict(m),optimizer=opt.state_dict()),b);b.seek(0)
        restored=torch.load(b,map_location='cpu',weights_only=True)
        load_trainable_state_dict(m,restored['model']);opt.load_state_dict(restored['optimizer'])
        self.assertEqual(len(opt.state),len(covered))
        self.assertEqual(opt.param_groups[-1]['weight_decay'],0)

    def test_exception_cleans_hook(self):
        v=FakeVisual();m=TokenAttentionClassifier(v,nn.Linear(4,3),mode='spatial')
        with self.assertRaises(RuntimeError):m(torch.randn(2,3,5))
        self.assertEqual(len(v.transformer._forward_hooks),0)


if __name__=='__main__':unittest.main()
