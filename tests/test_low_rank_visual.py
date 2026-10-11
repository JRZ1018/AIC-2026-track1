"""CPU-only LR1 contract tests, no data, GPU or third-party implementations."""
import copy
import io
import unittest

import torch
from torch import nn
from torch.nn.utils import parametrize

from aictrack1.engine.peft import trainable_state_dict, load_trainable_state_dict
from aictrack1.methods.low_rank_visual import (
    LR1, configure_low_rank_visual, expected_delta_shapes, validate_delta_state,
    validate_recipe, exploratory_gate, visual_initialization_device,
)
from aictrack1.models.peft import PEFTImageClassifier, configure_visual_peft, fold_visual_lora


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(8, 16)
        self.mlp.c_proj = nn.Linear(16, 8)
        self.ln_1 = nn.LayerNorm(8)
        self.ln_2 = nn.LayerNorm(8)

    def forward(self, x):
        z = self.ln_1(x)
        x = x + self.attn(z, z, z, need_weights=False)[0]
        return x + self.mlp.c_proj(torch.relu(self.mlp.c_fc(self.ln_2(x))))


class Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(*(Block() for _ in range(12)))

    def forward(self, x):
        return self.transformer.resblocks(x).mean(1)


def model():
    head = nn.Linear(8, 3)
    head.requires_grad_(False)
    return PEFTImageClassifier(Visual(), head)


class LowRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_disabled_exact_logits_loss_gradient_and_rng(self):
        a = model(); b = copy.deepcopy(a)
        configure_visual_peft(a.visual, 'last_n_blocks', lora_blocks=8)
        rng = torch.get_rng_state()
        configure_low_rank_visual(b.visual, None)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        x = torch.randn(2, 3, 8); y = torch.tensor([0, 2])
        za, zb = a(x), b(x)
        self.assertTrue(torch.equal(za, zb))
        la, lb = nn.functional.cross_entropy(za, y), nn.functional.cross_entropy(zb, y)
        self.assertTrue(torch.equal(la, lb)); la.backward(); lb.backward()
        self.assertTrue(all(torch.equal(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters()) if p.requires_grad))

    def test_enabled_identity_private_rng_and_exact_names(self):
        a = model(); x = torch.randn(2, 3, 8)
        configure_visual_peft(a.visual, 'frozen')
        base = copy.deepcopy(a)
        # Equal autograd policy on both sides: freezing blocks otherwise changes
        # CPU linear dispatch (sub-micro FP32 rounding, not changed weights).
        with torch.no_grad(): before = a(x).detach()
        rng = torch.get_rng_state(); configure_low_rank_visual(a.visual, LR1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        with torch.no_grad(): self.assertTrue(torch.equal(before, a(x)))
        for old, new in zip(base.visual.transformer.resblocks, a.visual.transformer.resblocks):
            for old_weight, new_weight in (
                (old.attn.in_proj_weight, new.attn.in_proj_weight),
                (old.attn.out_proj.weight, new.attn.out_proj.weight),
                (old.mlp.c_fc.weight, new.mlp.c_fc.weight),
                (old.mlp.c_proj.weight, new.mlp.c_proj.weight),
            ): self.assertTrue(torch.equal(old_weight, new_weight))
        state = trainable_state_dict(a)
        self.assertEqual(set(state), set(expected_delta_shapes(8, 16)))
        self.assertEqual(len(state), 64)
        self.assertEqual(validate_delta_state(state, width=8, mlp_width=16), 6144)

    def test_frozen_weights_and_finite_nonzero_b_gradient(self):
        a = model(); configure_low_rank_visual(a.visual, LR1)
        frozen = {n:p.detach().clone() for n,p in a.named_parameters() if not p.requires_grad}
        x = torch.randn(2, 3, 8); y = torch.tensor([0, 1]); opt = torch.optim.AdamW([p for p in a.parameters() if p.requires_grad], lr=1e-4)
        nn.functional.cross_entropy(a(x), y).backward()
        grads = {n:p.grad for n,p in a.named_parameters() if p.requires_grad}
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads.values()))
        self.assertTrue(any(g.abs().sum()>0 for n,g in grads.items() if n.endswith('lora_b')))
        # Zero B initialization makes A's first gradient zero; do not require it nonzero.
        self.assertTrue(all(torch.count_nonzero(g)==0 for n,g in grads.items() if n.endswith('lora_a')))
        opt.step()
        self.assertTrue(all(torch.equal(dict(a.named_parameters())[n],v) for n,v in frozen.items()))

    def test_safe_complete_optimizer_restore_and_fold_logits(self):
        a = model(); b = copy.deepcopy(a); configure_low_rank_visual(a.visual, LR1); configure_low_rank_visual(b.visual, LR1)
        params = [p for p in a.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,20)
        x = torch.randn(2,3,8); nn.functional.cross_entropy(a(x),torch.tensor([0,1])).backward(); opt.step(); sched.step()
        buf=io.BytesIO(); torch.save(dict(delta=trainable_state_dict(a),optimizer=opt.state_dict(),scheduler=sched.state_dict(),protocol=LR1),buf); buf.seek(0)
        saved=torch.load(buf,weights_only=True,map_location='cpu'); validate_recipe(saved['protocol']); load_trainable_state_dict(b,saved['delta'])
        opt2=torch.optim.AdamW([p for p in b.parameters() if p.requires_grad],lr=1e-4); opt2.load_state_dict(saved['optimizer'])
        self.assertEqual(len(opt2.state),64)
        for one,two in zip(opt.state.values(),opt2.state.values()):
            self.assertTrue(all(torch.equal(one[k],two[k]) for k in one))
        self.assertTrue(torch.equal(a(x),b(x)))
        with torch.no_grad(): before=b(x).detach()
        fold_visual_lora(b.visual)
        with torch.no_grad(): self.assertTrue(torch.equal(before,b(x)))
        self.assertFalse(any(parametrize.is_parametrized(m) for m in b.modules()))

    def test_drift_and_extra_state_rejected(self):
        device=torch.device('cuda')
        self.assertIs(visual_initialization_device(device,None),device)
        self.assertEqual(visual_initialization_device(device,LR1),torch.device('cpu'))
        bad=dict(LR1, rank=16)
        with self.assertRaises(ValueError):validate_recipe(bad)
        a=model();configure_low_rank_visual(a.visual,LR1);state=trainable_state_dict(a)
        state['classifier.weight']=a.classifier.weight.detach()
        with self.assertRaises(ValueError):validate_delta_state(state,width=8,mlp_width=16)

    def test_relaxed_gate_not_stable_claim(self):
        d=dict(p7=(.02,-.08,.15),raw=(0,-.09,.1),ordinary=(0,-.10,.1),tail=(0,-.25,.2))
        out=exploratory_gate(d);self.assertTrue(out['exploratory_screen_pass']);self.assertFalse(out['original_single_seed_screen_pass']);self.assertFalse(out['stable_training_gain_proven'])

    def test_protections_and_p7_strict_point(self):
        d=dict(p7=(0,-.08,.15),raw=(0,-.09,.1),ordinary=(0,-.10,.1),tail=(0,-.25,.2))
        self.assertFalse(exploratory_gate(d)['exploratory_screen_pass'])
        d['p7']=(.2,.01,.3); d['tail']=(0,-.251,.2)
        self.assertFalse(exploratory_gate(d)['exploratory_screen_pass'])

    def test_interval_and_nan_rejected(self):
        d=dict(p7=(.2,.1,.3),raw=(0,-.09,.1),ordinary=(0,-.10,.1),tail=(0,-.25,.2))
        for bad in ((float('nan'),0,1),(.2,1,0)):
            d['p7']=bad
            with self.assertRaises(ValueError):exploratory_gate(d)


if __name__ == '__main__':unittest.main()
