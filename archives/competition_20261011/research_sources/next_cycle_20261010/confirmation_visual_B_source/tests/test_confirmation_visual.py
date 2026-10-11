"""Synthetic CPU fixtures and actual static import closure; no SSH/GPU/images or real H1."""
from __future__ import annotations
import ast,copy,hashlib,importlib,inspect,json,os,random,subprocess,sys,tempfile,unittest
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
import yaml
import p1_contract as c
from bind_confirmation_visual import validate_budget,validate_h1_payload
from aictrack1.engine.peft import trainable_state_dict,load_trainable_state_dict
from aictrack1.models.peft import PEFTImageClassifier
from aictrack1.models.heads import build_classifier_head
from train_n1_peft import _base_loss_per_sample

class Contracts(unittest.TestCase):
 def admission(self,arm='B25'):
  from B_admission_contract import SCIENTIFIC_TEMPLATE_SHA,MATH_SOURCE_SHA,MATH_PROTOCOL_SHA,PRIMARY_SCOPE,SOFT_COEFFICIENTS
  return dict(schema='aictrack1.next_cycle10.B_admission.v1',root_authorized=True,complete=True,selected_arm=arm,soft_coefficient=SOFT_COEFFICIENTS.get(arm,.25),exploration_balanced_delta_pp=.1,query_definition_source_sha256=c.QUERY_SOURCE_SHA,new_H1_checkpoint_sha256=None,exploration_paired_sha256='1'*64,root_risk_review_sha256='2'*64,exploration_training_source_sha256='5'*64,scientific_template_source_sha256=SCIENTIFIC_TEMPLATE_SHA,math_source_sha256=MATH_SOURCE_SHA,math_protocol_sha256=MATH_PROTOCOL_SHA,primary_scope=PRIMARY_SCOPE,final_model='student_only',test_access=False)
 def test_unbound_cannot_train(self):
  self.assertEqual(c.verify_files(require_bound=False)['status'],'unbound_template')
  with self.assertRaises(ValueError):c.verify_files()
 def test_actual_admission_only(self):
  for arm in ('B25','B50'):self.assertEqual(c.admission(self.admission(arm))['selected_arm'],arm)
  for field,value in [('selected_arm','A3'),('exploration_balanced_delta_pp',0.),('new_H1_checkpoint_sha256','3'*64),('query_definition_source_sha256','4'*64),('root_authorized',False)]:
   v=self.admission();v[field]=value
   with self.assertRaises(ValueError):c.admission(v)
 def fake_assets(self,arm):
  return dict(arms=['J5',arm],assets={key:dict(path='/synthetic_not_opened/'+key) for key in ['manifest','split','label_conflicts','h1']})
 def test_control_candidate_zero_diff_except_soft_identity(self):
  for arm in ('B25','B50'):
   p=self.fake_assets(arm);base=c.expected_config('J5',p);other=c.expected_config(arm,p);self.assertEqual(base['training']['seed'],20261010);self.assertEqual(other['training'],base['training']);self.assertEqual(other['base_loss'],base['base_loss']);self.assertEqual(base['base_loss'],dict(kind='sce',alpha=.1,beta=1.,log_zero=-4.));self.assertEqual(base['josrc_soft']['soft_coefficient'],0.);self.assertEqual(other['josrc_soft']['soft_coefficient'],c.ARMS[arm])
   for v in (base,other):v['josrc_soft']['soft_coefficient']=0.;v['run_id']='same';v['output_dir']='same'
   self.assertEqual(base,other)
 def test_unselected_config_rejected(self):
  p=self.fake_assets('B25')
  with self.assertRaises(ValueError):c.expected_config('B50',p)
  cfg=c.expected_config('B25',p);cfg['training']['seed']=20261008
  with self.assertRaises(ValueError):c.validate_config(cfg,'B25',p)
 def test_original_B_numeric_source_closure_bytes(self):
  original=json.loads((ROOT/'inputs/exploration_B_source_manifest.json').read_text(encoding='utf-8'));self.assertEqual(c.sha(ROOT/'inputs/exploration_B_source_manifest.json'),'8d9c229f645e20a306b11a69bc57a6f4c7141d0c90783bd7d616cb02e580a276');self.assertEqual(c.sha(ROOT/'scripts/train_n1_peft.py'),'ca246a18949f670b822053bae9c0d43ad84302365c3dee0cb7afc15f3a885a98');self.assertEqual(c.sha(ROOT/'scripts/train_peft.py'),'0758b989a1bb33ba851e03f9a852cf35b7ad30847ccbfb62cf528f8dc015a765')
  for rel,digest in original['files'].items():
   if rel.startswith('src/') or rel in ('scripts/owned_launch.py','scripts/train_n1_peft.py','scripts/train_peft.py','scripts/p1_smoke.py','scripts/mainline_h1_contract.py','scripts/bc1_runtime.py','scripts/profile_josrc100.py','inputs/soft_math_protocol.json'):
    self.assertEqual(c.sha(ROOT/rel),digest,rel)
 def test_real_static_N1_import_closure_and_fixed_reads(self):
  seen=set();todo=[ROOT/'scripts/train_n1_peft.py'];names=[]
  while todo:
   path=todo.pop();self.assertTrue(path.is_file());path=path.resolve()
   if path in seen:continue
   seen.add(path);tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
   for node in ast.walk(tree):
    if isinstance(node,ast.ImportFrom) and node.module and node.level==0:
     name=node.module;module=importlib.import_module(name);names.append(name)
     for alias in node.names:
      if alias.name!='*':self.assertTrue(hasattr(module,alias.name),name+'.'+alias.name)
     target=Path(getattr(module,'__file__',''))
     if target.is_file() and target.resolve().is_relative_to(ROOT):todo.append(target)
    elif isinstance(node,ast.Import):
     for alias in node.names:
      candidates=[ROOT/'scripts'/Path(alias.name.replace('.', '/')+'.py'),ROOT/'src'/Path(alias.name.replace('.', '/')+'.py'),ROOT/'src'/Path(alias.name.replace('.', '/'))/'__init__.py']
      local=next((p for p in candidates if p.is_file()),None)
      if local is not None:
       module=importlib.import_module(alias.name);target=Path(getattr(module,'__file__',''));self.assertTrue(target.resolve().is_relative_to(ROOT));todo.append(target)
      # Deferred external OpenAI clip import is checked on the actual server by preflight/CUDA smoke; local host has no clip installed.
  self.assertGreaterEqual(len(seen),35);self.assertTrue((ROOT/'scripts/train_peft.py').is_file());self.assertTrue((ROOT/'src/aictrack1/methods/confidence_observation.py').is_file())
 def budget(self):
  return dict(schema='aictrack1.next_cycle10.confirmation_budget.v1',root_authorized=True,phase_start_unix=1.,deadline_unix=2.,original_initial_free_bytes=13013950464,capacity_added_bytes=32212254720,adjusted_initial_free_bytes=45226205184,actual_disk_total_bytes=128849018880,disk_max_bytes=21474836480,min_free_bytes=3221225472,atomic_reserve_bytes=2147483648,capacity_evidence=dict(path=str(Path(tempfile.gettempdir()).resolve()/'synthetic_not_opened'),sha256='1'*64))
 def test_shared_cumulative_budget_no_reset(self):
  validate_budget(self.budget())
  for k,v in [('adjusted_initial_free_bytes',50000000000),('disk_max_bytes',32212254721),('atomic_reserve_bytes',0),('min_free_bytes',0),('deadline_unix',.5)]:
   b=self.budget();b[k]=v
   with self.assertRaises(ValueError):validate_budget(b)
 def test_new_frozen_metadata_no_query_reuse(self):
  d=json.loads((ROOT/'inputs/confirmation_assets.json').read_text(encoding='utf-8'));self.assertEqual(d['old0c9_query_connected_group_overlap'],0);self.assertEqual(d['used57189_query_connected_group_overlap'],0);self.assertEqual(d['train_query_connected_group_overlap'],0);self.assertEqual(d['counts'],dict(train=118958,validation=29737));self.assertEqual(c.sha(ROOT/'inputs/confirmation_split.csv'),c.SPLIT_SHA);self.assertEqual(c.sha(ROOT/'inputs/query_definition_source_manifest.json'),c.QUERY_SOURCE_SHA)
 def test_sce_actual_original_numeric_unchanged_all_arms(self):
  logits=torch.tensor([[1.,-.4,.2],[.1,.6,-.2]]);targets=torch.tensor([0,2]);out=[]
  for arm in ('J5','B25','B50'):
   cfg=c.expected_config(arm,self.fake_assets('B25' if arm=='J5' else arm));self.assertEqual(cfg['base_loss']['alpha'],.1);out.append(_base_loss_per_sample(logits,targets,kind='sce',q=.7,label_smoothing=0.,sce_alpha=cfg['base_loss']['alpha'],sce_beta=1.,sce_log_zero=-4.))
  self.assertTrue(all(torch.equal(x,out[0]) for x in out))
 def test_B_admission_raw_A_nan_or_wrong_math_forbidden(self):
  for key,value in [('selected_arm','A5'),('primary_scope','raw_P7_vs_J5'),('exploration_balanced_delta_pp',float('nan')),('math_source_sha256','6'*64),('math_protocol_sha256','7'*64),('soft_coefficient',.5),('final_model','EMA'),('test_access',True)]:
   a=self.admission();a[key]=value
   with self.assertRaises(ValueError):c.admission(a)
 def test_control_coefficient0_hard_object_grad_no_extra_forward(self):
  from aictrack1.methods.josrc_soft import recipe,SoftTeacher,combine_loss
  y=torch.tensor([0,2]);a=torch.randn(2,3,requires_grad=True);b=a.detach().clone().requires_grad_();hard_a=_base_loss_per_sample(a,y,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.1,sce_beta=1.,sce_log_zero=-4.).mean();hard_b=_base_loss_per_sample(b,y,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.1,sce_beta=1.,sce_log_zero=-4.).mean();mixed,detail=combine_loss(hard_b,b,None,None,y,recipe(0),9);self.assertIs(mixed,hard_b);self.assertTrue(torch.equal(torch.autograd.grad(hard_a,a)[0],torch.autograd.grad(mixed,b)[0]));m=torch.nn.Linear(3,3);teacher=SoftTeacher(m,recipe(0));self.assertEqual(teacher.weak_views_before_strong(torch.randn(2,3),9),(None,None))
 def test_real_model_optimizer_scheduler_CPU_disk_restore(self):
  torch.manual_seed(20261010)
  def model():return PEFTImageClassifier(torch.nn.Linear(4,512),build_classifier_head('linear',512,750),normalize_features=True)
  m=model();opt=torch.optim.AdamW([dict(params=list(m.visual.parameters()),lr=1.75e-6),dict(params=list(m.classifier.parameters()),lr=5e-6)],weight_decay=.01);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20);x=torch.randn(3,4);y=torch.tensor([0,7,749])
  def step():
   opt.zero_grad();loss=_base_loss_per_sample(m(x),y,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.3,sce_beta=1.,sce_log_zero=-4.).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();sch.step();return float(loss.detach())
  step();state=dict(trainable_state=trainable_state_dict(m),optimizer=opt.state_dict(),scheduler=sch.state_dict(),torch_rng_state=torch.get_rng_state().clone())
  with tempfile.TemporaryDirectory(prefix='confirmation_CPU_') as directory:
   path=Path(directory)/'resume.pt';torch.save(state,path);step();expected=trainable_state_dict(m);m=model();opt=torch.optim.AdamW([dict(params=list(m.visual.parameters()),lr=1.75e-6),dict(params=list(m.classifier.parameters()),lr=5e-6)],weight_decay=.01);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20);loaded=torch.load(path,map_location='cpu',weights_only=True);load_trainable_state_dict(m,loaded['trainable_state']);opt.load_state_dict(loaded['optimizer']);sch.load_state_dict(loaded['scheduler']);torch.set_rng_state(loaded['torch_rng_state']);step();actual=trainable_state_dict(m);self.assertEqual(set(expected),set(actual));self.assertTrue(all(torch.equal(v,actual[k]) for k,v in expected.items()));self.assertEqual(sch.last_epoch,2);self.assertEqual(len(opt.state),4)
 def test_real_CLI_help_import_entries(self):
  for name in ['bind_confirmation_visual.py','run_confirmation_visual.py','train_n1_peft.py']:
   p=subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/name),'--help'],capture_output=True,text=True,timeout=60);self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
