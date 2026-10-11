"""Meaningful CPU contracts on a tiny visual encoder, original losses and full-size AFA/head."""
import copy,io,random,unittest,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
torch.set_num_threads(4)
from aictrack1.models.afa_compatible import AFACompatibleClassifier,RECIPE,FeatureDecomposition,channel_decomposition,afa_parameter_groups
from aictrack1.models.token_attention_classifier import TokenAttentionClassifier
from aictrack1.engine.peft import trainable_state_dict,load_trainable_state_dict
from train_n1_peft import _base_loss_per_sample,_weighted_base_loss
class TinyVisual(nn.Module):
 def __init__(self):
  super().__init__();self.patch=nn.Linear(8,768);self.transformer=nn.Identity();self.ln_post=nn.LayerNorm(768);self.proj=nn.Parameter(torch.randn(768,512)/30);self.calls=0
 def forward(self,images):
  self.calls+=1;x=self.patch(images).unsqueeze(0).expand(101,-1,-1);x=self.transformer(x)
  return self.ln_post(x[0])@self.proj
class Tests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(18);self.visual=TinyVisual();self.head=nn.Linear(512,750);self.images=torch.randn(4,8)
 def model(self,coefficient=.25):
  m=AFACompatibleClassifier(copy.deepcopy(self.visual),copy.deepcopy(self.head),normalize_features=True,recipe=dict(RECIPE,auxiliary_coefficient=coefficient));m.set_afa_training_context(10,torch.ones(750,dtype=torch.long));return m
 def test_01_private_initialization_and_parameters(self):
  before=torch.random.get_rng_state();m=self.model();self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
  self.assertEqual(len(list(m.afa.parameters())),11);self.assertEqual(sum(p.numel() for p in m.afa.parameters()),63613)
  self.assertTrue(torch.equal(m.afa.value.weight,torch.eye(100)));self.assertTrue(torch.equal(m.afa.value.bias,torch.zeros(100)))
 def test_02_warmup_loss_gradient_RNG_exact(self):
  m=self.model();m.set_afa_training_context(0,torch.ones(750,dtype=torch.long));base=TokenAttentionClassifier(copy.deepcopy(self.visual),copy.deepcopy(self.head),normalize_features=True,mode='none')
  y=torch.tensor([0,1,2,3]);before=torch.random.get_rng_state();a=m(self.images);after=torch.random.get_rng_state();b=base(self.images)
  self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(before,after));self.assertEqual(m.visual.calls,1);self.assertIsNone(m.afa_cache)
  aux,d=m.auxiliary_loss(y,a,torch.ones(4));self.assertEqual(float(aux),0.);self.assertFalse(d['active']);self.assertEqual(int(m.afa_observed_support.sum()),4)
  weights=torch.tensor([1.,.5,.02,.8]);loss_a=_weighted_base_loss(_base_loss_per_sample(a,y,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.1,sce_beta=1.,sce_log_zero=-4.),weights);loss_b=_weighted_base_loss(_base_loss_per_sample(b,y,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.1,sce_beta=1.,sce_log_zero=-4.),weights)
  self.assertTrue(torch.equal(loss_a,loss_b));loss_a.backward();loss_b.backward()
  for name,p in base.named_parameters():self.assertTrue(torch.equal(p.grad,dict(m.named_parameters())[name].grad),name)
  self.assertTrue(all(p.grad is None for p in m.afa.parameters()))
 def test_03_zero_coefficient_direct_original_forward(self):
  m=self.model(0.);base=TokenAttentionClassifier(copy.deepcopy(self.visual),copy.deepcopy(self.head),normalize_features=True,mode='none');state=m.afa_generator.get_state()
  a=m(self.images);b=base(self.images);self.assertTrue(torch.equal(a,b));self.assertIsNone(m.afa_cache)
  y=a.detach().argmax(1);aux,d=m.auxiliary_loss(y,a,torch.ones(4));self.assertFalse(d['active']);self.assertTrue(torch.equal(state,m.afa_generator.get_state()))
  (F.cross_entropy(a,y)+aux).backward();F.cross_entropy(b,y).backward()
  for name,p in base.named_parameters():self.assertTrue(torch.equal(p.grad,dict(m.named_parameters())[name].grad),name)
  self.assertTrue(all(p.grad is None for p in m.afa.parameters()))
 def test_04_mask_and_class_variation_equation(self):
  m=FeatureDecomposition();x=torch.randn(2,100,512);c,v,value,mask=m(x)
  self.assertTrue(torch.allclose(c+v,value,atol=1e-6));self.assertTrue(torch.allclose(mask.sum(1),torch.ones(2,100),atol=1e-6))
  self.assertTrue(bool(torch.isfinite(mask).all()));self.assertTrue(bool((mask>=0).all()))
 def test_05_samehead_and_only_one_visual_forward(self):
  m=self.model();logits=m(self.images);y=logits.detach().argmax(1);before=torch.random.get_rng_state();aux,d=m.auxiliary_loss(y,logits,torch.ones(4))
  self.assertEqual(m.visual.calls,1);self.assertEqual(d['generated_count'],2);self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
  self.assertIs(m.classifier.weight,dict(m.named_parameters())['classifier.weight'])
  (F.cross_entropy(logits,y)+aux).backward()
  self.assertTrue(bool(torch.isfinite(m.classifier.weight.grad).all()));self.assertGreater(float(m.classifier.weight.grad.norm()),0)
  for name,p in m.afa.named_parameters():self.assertIsNotNone(p.grad,name);self.assertTrue(bool(torch.isfinite(p.grad).all()),name)
 def test_06_no_false_eligible_or_out_of_batch_classes(self):
  m=self.model();logits=m(self.images);y=logits.argmax(1)
  aux,d=m.auxiliary_loss(y,logits,torch.zeros(4));self.assertEqual(d['generated_count'],0);self.assertEqual(int(m.afa_generated_support.sum()),0)
  m(self.images);aux,d=m.auxiliary_loss(y,logits,torch.ones(4));support=m.afa_generated_support
  self.assertTrue(set(torch.where(support>0)[0].tolist()).issubset(set(y.tolist())))
 def test_07_eval_generation_disabled_and_initial_identity(self):
  m=self.model().eval();base=TokenAttentionClassifier(copy.deepcopy(self.visual),copy.deepcopy(self.head),normalize_features=True,mode='none').eval();a=m(self.images)
  self.assertTrue(torch.equal(a,base(self.images)));self.assertIsNone(m.afa_cache);aux,d=m.auxiliary_loss(a.argmax(1),a,torch.ones(4));self.assertFalse(d['active'])
 def test_08_anchor_identity(self):
  c,v,x=map(lambda _:torch.randn(3,512,100),range(3));x=c+v;g=torch.randn(3,512);i=torch.tensor([0,1]);j=torch.tensor([2,0])
  generated=g[i]+.1*((c[i]+v[j]).mean(-1)-x[i].mean(-1));equivalent=g[i]+.1*(v[j].mean(-1)-v[i].mean(-1))
  self.assertTrue(torch.allclose(generated,equivalent,atol=1e-6))
 def test_09_optimizer_and_disk_nextstep_exact(self):
  m=self.model();opt=torch.optim.AdamW([dict(params=list(m.visual.parameters()),lr=1.75e-6),dict(params=list(m.classifier.parameters()),lr=5e-6),*afa_parameter_groups(m)],weight_decay=.01)
  scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
  def step():
   opt.zero_grad();z=m(self.images);y=z.detach().argmax(1);a,_=m.auxiliary_loss(y,z,torch.ones(4));(F.cross_entropy(z,y)+a).backward();opt.step();scheduler.step()
  step();stream=io.BytesIO();torch.save(dict(model=trainable_state_dict(m),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),method=m.method_state_dict(),rng=torch.random.get_rng_state()),stream)
  step();expected=trainable_state_dict(m);expected_method=m.method_state_dict()
  stream.seek(0);s=torch.load(stream,weights_only=True,map_location='cpu');load_trainable_state_dict(m,s['model']);opt.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);m.load_method_state_dict(s['method']);torch.random.set_rng_state(s['rng']);step()
  for k,v in expected.items():self.assertTrue(torch.equal(v,trainable_state_dict(m)[k]),k)
  self.assertTrue(torch.equal(expected_method['generator_state'],m.method_state_dict()['generator_state']))
 def test_10_restore_recipe_frequency_RNG_corruption_rejected(self):
  m=self.model();s=m.method_state_dict()
  for field,value in [('recipe',{}),('frequency_counts',torch.zeros(750,dtype=torch.long)),('generator_state',torch.tensor([])),('generation_disabled_at_inference',False)]:
   bad=copy.deepcopy(s);bad[field]=value
   with self.assertRaises((ValueError,RuntimeError)):self.model().load_method_state_dict(bad)
 def test_11_actual_template_config_closed(self):
  from p1_contract import assets,expected_config,validate_config
  import yaml
  p=assets(ROOT);raw=yaml.safe_load((ROOT/'configs/experiment/p1_AFA.yaml').read_text())['experiment'];validate_config(raw,'AFA',p)
  baseline=yaml.safe_load((ROOT/'inputs/baseline_J5.yaml').read_text())['experiment'];expected=copy.deepcopy(baseline);expected.update(run_id=raw['run_id'],output_dir=raw['output_dir'],afa_compatible=raw['afa_compatible'])
  self.assertEqual(raw,expected)
if __name__=='__main__':unittest.main()
