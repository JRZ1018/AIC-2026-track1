import io
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from aictrack1.data.contracts import TransformConfig
from aictrack1.data.indexed_images import IndexedImage, IndexedImageDataset
from aictrack1.data.transforms import build_split_transforms
from aictrack1.methods.current_view_consistency import (
    PROTOCOL, add_current_view_loss, attach_center_view, protocol_state,
    verify_protocol_state,
)


class CurrentViewTests(unittest.TestCase):
    def test_disabled_identity_and_gradient(self):
        logits=torch.randn(2,3,requires_grad=True)
        original=logits.square().mean()
        result,js=add_current_view_loss(original,logits,None,None)
        self.assertIs(result,original);self.assertIsNone(js)
        self.assertTrue(torch.equal(torch.autograd.grad(result,logits,retain_graph=True)[0],torch.autograd.grad(original,logits)[0]))

    def test_enabled_both_branches(self):
        a=torch.tensor([[3.,0.,-1.]],requires_grad=True)
        b=torch.tensor([[0.,3.,-1.]],requires_grad=True)
        loss,js=add_current_view_loss(a.square().mean(),a,b,PROTOCOL)
        self.assertGreater(float(js.detach()),0)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())
        self.assertGreater(float(b.grad.abs().sum()),0)

    def test_protocol_safe_roundtrip_and_reject(self):
        buf=io.BytesIO();torch.save({'vc1':protocol_state(PROTOCOL)},buf);buf.seek(0)
        verify_protocol_state(torch.load(buf,weights_only=True)['vc1'],PROTOCOL)
        verify_protocol_state(None,None)
        for key,value in [('coefficient',2.),('coefficient',True),('teacher',True),('temporal_updates',1),('both_branches_gradient',False)]:
            with self.assertRaises(ValueError):protocol_state({**PROTOCOL,key:value})
        with self.assertRaises(ValueError):verify_protocol_state(None,PROTOCOL)
        with self.assertRaises(ValueError):add_current_view_loss(torch.tensor(1.),torch.randn(2,3),torch.randn(2,3),None)

    def make_dataset(self, root, partition='train'):
        path=root/'synthetic.png'
        image=np.random.default_rng(123).integers(0,256,(35,51,3),dtype=np.uint8)
        Image.fromarray(image).save(path)
        row=IndexedImage(sample_index=7,sample_id='synthetic',relative_path=path.name,absolute_path=path,source='train',partition=partition,internal_class_index=2,official_class_id='003')
        transforms=build_split_transforms(TransformConfig(image_size=16,resize_size=16,train_crop_scale_min=.8,train_horizontal_flip_probability=.5))
        return IndexedImageDataset([row],transforms['train'],stochastic_seed=20261005),transforms['validation']

    def test_pair_exact_original_rng_and_center(self):
        with tempfile.TemporaryDirectory() as folder:
            primary,center=self.make_dataset(Path(folder))
            paired=attach_center_view(primary,center,PROTOCOL)
            for epoch in (0,1,19):
                primary.set_epoch(epoch);expected=primary[0]
                paired.set_epoch(epoch)
                py=random.getstate();npstate=np.random.get_state();th=torch.get_rng_state().clone()
                actual=paired[0]
                self.assertTrue(torch.equal(actual['image'],expected['image']))
                self.assertEqual(actual['sample_id'],expected['sample_id'])
                self.assertTrue(torch.equal(actual['consistency_center_image'],paired.center[0]['image']))
                self.assertEqual(random.getstate(),py)
                self.assertTrue(np.array_equal(npstate[1],np.random.get_state()[1]))
                self.assertTrue(torch.equal(th,torch.get_rng_state()))

    def test_no_validation_or_test_pairs_and_disabled_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            primary,center=self.make_dataset(Path(folder),partition='validation')
            self.assertIs(attach_center_view(primary,center,None),primary)
            with self.assertRaises(ValueError):attach_center_view(primary,center,PROTOCOL)


if __name__=='__main__':unittest.main()
