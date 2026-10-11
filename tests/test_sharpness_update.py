"""SAM1 helper CPU tests; not an end-to-end trainer/CUDA acceptance."""
import copy
import io
import unittest
import torch
from aictrack1.methods.sharpness_neighborhood import ReversibleL2Neighborhood
from aictrack1.methods.sharpness_update import PROTOCOL,protocol_state,verify_protocol_state,same_batch_sam_backward


class SharpnessUpdateTests(unittest.TestCase):
    def model(self):return torch.nn.Linear(3,2)
    def loss(self,m):return m(torch.ones(4,3)).square().mean()

    def test_disabled_exact_gradient_and_no_closure(self):
        a=self.model();b=copy.deepcopy(a);x=self.loss(a);y=self.loss(b)
        x.backward()
        def forbidden():raise AssertionError('disabled extra forward')
        same_batch_sam_backward(y,forbidden,list(b.named_parameters()),torch.amp.GradScaler('cpu',enabled=False),None)
        self.assertTrue(torch.equal(x,y))
        for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p.grad,q.grad))

    def test_manual_second_gradient_and_restore(self):
        a=self.model();b=copy.deepcopy(a);original={n:p.detach().clone() for n,p in a.named_parameters()}
        g=torch.autograd.grad(self.loss(b),list(b.parameters()))
        with ReversibleL2Neighborhood(list(b.named_parameters()),g):self.loss(b).backward()
        same_batch_sam_backward(self.loss(a),lambda:self.loss(a),list(a.named_parameters()),torch.amp.GradScaler('cpu',enabled=False),PROTOCOL)
        for n,p in a.named_parameters():self.assertTrue(torch.equal(p,original[n]))
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_positive_scale_cancels_and_state_roundtrip(self):
        a=self.model();b=copy.deepcopy(a)
        scale=torch.amp.GradScaler('cpu',init_scale=8.)
        off=torch.amp.GradScaler('cpu',enabled=False)
        opt=torch.optim.AdamW(a.parameters(),lr=.001);optb=torch.optim.AdamW(b.parameters(),lr=.001)
        same_batch_sam_backward(self.loss(a),lambda:self.loss(a),list(a.named_parameters()),scale,PROTOCOL)
        same_batch_sam_backward(self.loss(b),lambda:self.loss(b),list(b.named_parameters()),off,PROTOCOL)
        scale.unscale_(opt)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        torch.nn.utils.clip_grad_norm_(a.parameters(),1.);torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
        scale.step(opt);scale.update();optb.step();sched.step()
        payload=dict(model=a.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scale.state_dict(),protocol=protocol_state(PROTOCOL))
        buff=io.BytesIO();torch.save(payload,buff);buff.seek(0);state=torch.load(buff,weights_only=True)
        c=self.model();c.load_state_dict(state['model']);optc=torch.optim.AdamW(c.parameters(),lr=.001);optc.load_state_dict(state['optimizer'])
        self.assertEqual(len(optc.state),2);verify_protocol_state(state['protocol'],PROTOCOL)
        for p,q in zip(a.parameters(),c.parameters()):self.assertTrue(torch.equal(p,q))

    def test_exception_restores(self):
        m=self.model();saved=copy.deepcopy(m.state_dict())
        def fail():raise RuntimeError('injected')
        with self.assertRaisesRegex(RuntimeError,'injected'):
            same_batch_sam_backward(self.loss(m),fail,list(m.named_parameters()),torch.amp.GradScaler('cpu',enabled=False),PROTOCOL)
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,saved[k]))

    def test_protocol_and_gradient_drift(self):
        with self.assertRaises(ValueError):protocol_state(dict(PROTOCOL,rho=.1))
        with self.assertRaises(ValueError):verify_protocol_state(None,PROTOCOL)
        m=self.model();self.loss(m).backward()
        with self.assertRaises(ValueError):same_batch_sam_backward(self.loss(m),lambda:self.loss(m),list(m.named_parameters()),torch.amp.GradScaler('cpu',enabled=False),PROTOCOL)

    def test_amp_first_overflow_no_perturb_or_optimizer_step(self):
        p=torch.nn.Parameter(torch.ones(2));original=p.detach().clone()
        scaler=torch.amp.GradScaler('cpu',init_scale=1e30)
        opt=torch.optim.AdamW([p],lr=.001)
        def forbidden():raise AssertionError('overflow must not perturb/forward')
        result=same_batch_sam_backward(p.sum()*1e10,forbidden,[('p',p)],scaler,PROTOCOL)
        self.assertTrue(result['first_overflow']);self.assertFalse(result['second_forward'])
        scaler.unscale_(opt);scaler.step(opt);scaler.update()
        self.assertTrue(torch.equal(p,original));self.assertEqual(len(opt.state),0)
        self.assertLess(scaler.get_scale(),1e30)

    def test_non_amp_nonfinite_first_rejected(self):
        p=torch.nn.Parameter(torch.ones(2));original=p.detach().clone()
        with self.assertRaises(FloatingPointError):
            same_batch_sam_backward(p.sum()*float('inf'),lambda:p.sum(),[('p',p)],torch.amp.GradScaler('cpu',enabled=False),PROTOCOL)
        self.assertTrue(torch.equal(p,original))

if __name__=='__main__':unittest.main()
