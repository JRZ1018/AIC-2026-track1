"""CPU toy corruption tests; no official model, data, CUDA or server use."""
import copy
import io
import json
import unittest
from pathlib import Path

import torch
from torch import nn
import accept_p1_state as acceptance
from aictrack1.models.heads import LinearClassifier


class JointRestore(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261009)
        self.visual=nn.Linear(4,4)
        self.head=LinearClassifier(4,3)
        self.original={name:p.detach().clone() for name,p in self.head.named_parameters()}
        state={f'visual.{name}':p.detach().clone()+.01 for name,p in self.visual.named_parameters()}
        state.update({f'classifier.{name}':p.detach().clone()+(.2 if name.endswith('weight') else .3) for name,p in self.head.named_parameters()})
        self.checkpoint=dict(mode='full_visual',epoch=20,trainable_state=state)

    def restore(self):
        return acceptance.restore_joint_state(self.visual,self.head,self.checkpoint,expected_visual_tensors=2)

    def test_new_head_overwrites_h1_initializer(self):
        buffer=io.BytesIO();torch.save(self.checkpoint,buffer);buffer.seek(0)
        self.checkpoint=torch.load(buffer,map_location='cpu',weights_only=True)
        model,evidence=self.restore()
        self.assertEqual(evidence['trainable_tensors'],4)
        self.assertTrue(evidence['head_matches_checkpoint'])
        for name,parameter in model.classifier.named_parameters():
            self.assertTrue(torch.equal(parameter,self.checkpoint['trainable_state']['classifier.'+name]))
            self.assertFalse(torch.equal(parameter,self.original[name]))
        with torch.inference_mode():logits=model(torch.tensor([[1.,2.,3.,4.]]))
        self.assertEqual(tuple(logits.shape),(1,3));self.assertTrue(torch.isfinite(logits).all())

    def test_visual_only_export_rejected(self):
        self.checkpoint['trainable_state']={k:v for k,v in self.checkpoint['trainable_state'].items() if k.startswith('visual.')}
        with self.assertRaisesRegex(ValueError,'classifier state'):self.restore()

    def test_missing_head_bias_rejected(self):
        del self.checkpoint['trainable_state']['classifier.linear.bias']
        with self.assertRaisesRegex(ValueError,'classifier state'):self.restore()

    def test_wrong_shape_rejected(self):
        self.checkpoint['trainable_state']['classifier.linear.weight']=torch.ones(3,5)
        with self.assertRaisesRegex(ValueError,'shape mismatch'):self.restore()

    def test_nonfinite_head_rejected(self):
        self.checkpoint['trainable_state']['classifier.linear.bias'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'non-finite'):self.restore()

    def test_partial_epoch_cannot_be_final(self):
        self.checkpoint['epoch']=8
        with self.assertRaisesRegex(ValueError,'epoch20'):self.restore()

    def test_two_optimizer_groups_restore_and_damaged_group_rejected(self):
        model,_=self.restore()
        optimizer=torch.optim.AdamW([dict(params=model.visual.parameters(),lr=1.75e-6),dict(params=model.classifier.parameters(),lr=5e-6)],weight_decay=.01)
        loss=model(torch.ones(2,4)).sum();loss.backward();optimizer.step()
        saved=optimizer.state_dict()
        restored=acceptance.restore_joint_optimizer(model,saved)
        self.assertEqual([len(g['params']) for g in restored.param_groups],[2,2])
        damaged=copy.deepcopy(saved);damaged['param_groups']=damaged['param_groups'][:1]
        with self.assertRaisesRegex(ValueError,'group order'):acceptance.restore_joint_optimizer(model,damaged)

    def test_real_export_requires_complete_154_including_head(self):
        with self.assertRaisesRegex(ValueError,'152 visual'):acceptance.export_from_fixed_last(self.checkpoint)
        state=dict(self.checkpoint['trainable_state'])
        state.update({f'visual.synthetic{i}':torch.tensor([0.]) for i in range(150)})
        self.checkpoint['trainable_state']=state
        exported=acceptance.export_from_fixed_last(self.checkpoint)
        self.assertEqual(len(exported),154)
        self.assertTrue(torch.equal(exported['classifier.linear.weight'],state['classifier.linear.weight']))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(JointRestore))
    report=dict(complete=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                scope='CPU synthetic toy only; not actual P1 model acceptance',torch_version=torch.__version__,
                source_sha256=acceptance.sha(acceptance.__file__),test_sha256=acceptance.sha(__file__),
                engine_peft_source_sha256=acceptance.sha(acceptance.P1/'src/aictrack1/engine/peft.py'),
                models_peft_source_sha256=acceptance.sha(acceptance.P1/'src/aictrack1/models/peft.py'),
                no_official_images=True,no_gpu=True,no_server_access=True)
    output=acceptance.OUT/('p1_restore_toy_tests_'+acceptance.sha(__file__)[:16]+'.json')
    with output.open('x',encoding='utf-8') as handle:json.dump(report,handle,indent=2,sort_keys=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
