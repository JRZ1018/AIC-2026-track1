import io
import math
import unittest
import torch
from aictrack1.methods.view_consistency import view_jensen_shannon as js


class ViewConsistencyTests(unittest.TestCase):
    def test_same(self):
        a=torch.randn(7,750)
        self.assertLess(float(js(a,a).abs()),2e-7)
    def test_symmetric(self):
        a=torch.randn(4,5);b=torch.randn(4,5)
        self.assertTrue(torch.equal(js(a,b),js(b,a)))
    def test_bounds(self):
        a=torch.tensor([[10000.,-10000.],[-10000.,10000.]])
        v=js(a,-a,reduction='none')
        self.assertTrue(torch.isfinite(v).all())
        self.assertTrue(torch.all(v>=-2e-7) and torch.all(v<=math.log(2)+2e-7))
    def test_two_gradients(self):
        a=torch.tensor([[2.,0.,-1.]],requires_grad=True)
        b=torch.tensor([[-1.,2.,0.]],requires_grad=True)
        js(a,b).backward()
        for x in (a,b):self.assertTrue(torch.isfinite(x.grad).all() and x.grad.abs().sum()>0)
    def test_shift_invariant(self):
        a=torch.randn(4,5);b=torch.randn(4,5)
        self.assertTrue(torch.allclose(js(a,b),js(a+17,b-14),atol=2e-7))
    def test_half_and_reduction(self):
        a=torch.randn(4,750).half();b=torch.randn(4,750).half()
        v=js(a,b,reduction='none')
        self.assertEqual(v.dtype,torch.float32)
        self.assertTrue(torch.equal(v.mean(),js(a,b)) and torch.equal(v.sum(),js(a,b,reduction='sum')))
    def test_dense_definition(self):
        a=torch.randn(4,5);b=torch.randn(4,5);p=a.softmax(-1);q=b.softmax(-1);m=(p+q)/2
        expected=.5*((p*(p.log()-m.log())).sum(-1)+(q*(q.log()-m.log())).sum(-1))
        self.assertTrue(torch.allclose(js(a,b,reduction='none'),expected,atol=2e-7))
    def test_safe_restore(self):
        a=torch.randn(3,5);b=torch.randn(3,5);buf=io.BytesIO();torch.save({'a':a,'b':b},buf);buf.seek(0)
        s=torch.load(buf,weights_only=True);self.assertTrue(torch.equal(js(a,b),js(s['a'],s['b'])))
    def test_invalid(self):
        for a,b in [(torch.randn(2,3),torch.randn(3,3)),(torch.zeros(0,3),torch.zeros(0,3)),(torch.ones(2,1),torch.ones(2,1)),(torch.tensor([[float('nan'),0.]]),torch.zeros(1,2))]:
            with self.assertRaises(ValueError):js(a,b)
        with self.assertRaises(ValueError):js(torch.ones(1,2),torch.ones(1,2),reduction='median')

if __name__=='__main__':unittest.main()
