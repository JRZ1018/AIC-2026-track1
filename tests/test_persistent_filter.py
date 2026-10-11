import io
import unittest

import torch

from aictrack1.methods.persistent_filter import PersistentTrainingGate, gated_weighted_loss


class PersistentFilterTests(unittest.TestCase):
    def fixture(self, n=40):
        ids=torch.arange(n);labels=torch.zeros(n,dtype=torch.long)
        gate=PersistentTrainingGate(ids,labels)
        w=torch.ones(n);w[:n//2]=.05
        return gate,ids,w,torch.zeros(n),torch.ones(n)

    def test_requires_three_observations(self):
        g,i,w,a,s=self.fixture()
        for _ in range(2): self.assertEqual(g.update(i,w,a,s)['excluded'],0)
        self.assertEqual(g.update(i,w,a,s)['excluded'],4)

    def test_small_class_not_starved(self):
        for n in (1,4,5,16):
            g,i,w,a,s=self.fixture(n)
            for _ in range(5): g.update(i,w,a,s)
            self.assertTrue(bool(g.keep.all()))

    def test_high_loss_alone_not_enough(self):
        g,i,w,a,s=self.fixture();a.fill_(.5)
        for _ in range(4): g.update(i,w,a,s)
        self.assertTrue(bool(g.keep.all()))

    def test_unstable_disagreement_kept(self):
        g,i,w,a,s=self.fixture();s.fill_(.3)
        for _ in range(4): g.update(i,w,a,s)
        self.assertTrue(bool(g.keep.all()))

    def test_reentry_resets_streak(self):
        g,i,w,a,s=self.fixture()
        for _ in range(3): g.update(i,w,a,s)
        a.fill_(1);g.update(i,w,a,s)
        self.assertTrue(bool(g.keep.all()));self.assertEqual(int(g.streak.sum()),0)

    def test_wrong_unknown_or_reordered_ids_refused(self):
        g,i,w,a,s=self.fixture()
        with self.assertRaises(ValueError): g.update(i.flip(0),w,a,s)
        with self.assertRaises(ValueError): g.update(i+1,w,a,s)

    def test_nonfinite_refused(self):
        g,i,w,a,s=self.fixture();w[0]=float('nan')
        with self.assertRaises(ValueError): g.update(i,w,a,s)

    def test_mass_protection_and_ties(self):
        g,i,w,a,s=self.fixture();w.fill_(.05)
        for _ in range(3): report=g.update(i,w,a,s)
        self.assertEqual(torch.nonzero(~g.keep).flatten().tolist(),[0,1,2,3])
        row=report['classes'][0]
        self.assertGreaterEqual(row['retained_ac_mass'],.9*row['original_ac_mass']-1e-7)

    def test_safe_restore_matches_uninterrupted(self):
        g,i,w,a,s=self.fixture()
        for _ in range(2): g.update(i,w,a,s)
        f=io.BytesIO();torch.save(g.state_dict(),f);f.seek(0)
        other=PersistentTrainingGate(i,torch.zeros_like(i));other.load_state_dict(torch.load(f,weights_only=True))
        g.update(i,w,a,s);other.update(i,w,a,s)
        self.assertTrue(torch.equal(g.keep,other.keep));self.assertTrue(torch.equal(g.streak,other.streak))

    def test_invalid_restore_refused(self):
        g,i,w,a,s=self.fixture();state=g.state_dict();state['keep'].fill_(False)
        with self.assertRaises(ValueError): g.load_state_dict(state)

    def test_zero_batch_finite_and_skip_flag(self):
        losses=torch.tensor([1.,2.],requires_grad=True)
        loss,has=gated_weighted_loss(losses,torch.ones(2),torch.zeros(2,dtype=torch.bool))
        self.assertFalse(has);self.assertEqual(float(loss.detach()),0)
        loss.backward();self.assertTrue(torch.equal(losses.grad,torch.zeros(2)))

    def test_masked_loss_and_gradient(self):
        x=torch.tensor([2.,9.],requires_grad=True)
        loss,has=gated_weighted_loss(x,torch.tensor([.5,1.]),torch.tensor([True,False]))
        self.assertTrue(has);self.assertEqual(float(loss.detach()),2.)
        loss.backward();self.assertEqual(x.grad.tolist(),[1.,0.])

    def test_lookup_train_only_and_order(self):
        g=PersistentTrainingGate(torch.tensor([2,4,8]),torch.tensor([0,0,0]))
        g.keep[1]=False
        self.assertEqual(g.keep_for(torch.tensor([8,4,2])).tolist(),[True,False,True])
        for ids in (torch.tensor([3]),torch.tensor([10]),torch.tensor([-1])):
            with self.assertRaises(ValueError):g.keep_for(ids)

    def test_all_keep_exact_weighted_loss_and_grad(self):
        x=torch.tensor([2.,4.,9.],requires_grad=True);w=torch.tensor([.03,.4,.8])
        expected=(x*w).sum()/w.sum();actual,has=gated_weighted_loss(x,w,torch.ones(3,dtype=torch.bool))
        self.assertTrue(has);self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(torch.equal(torch.autograd.grad(expected,x)[0],torch.autograd.grad(actual,x)[0]))

    def test_empty_supervision_skips_adamw_all_state(self):
        p=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.AdamW([p],lr=.1,weight_decay=.1)
        before=p.detach().clone();loss,has=gated_weighted_loss(p.square(),torch.ones(1),torch.zeros(1,dtype=torch.bool))
        opt.zero_grad(set_to_none=True)
        if has:loss.backward();opt.step()
        self.assertTrue(torch.equal(before,p));self.assertEqual(len(opt.state),0)


if __name__=='__main__': unittest.main()
