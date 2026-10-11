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
 def admission(self,arm='A3'):
  return dict(schema='aictrack1.next_cycle10.h1_admission.v1',root_authorized=True,complete=True,selected_arm=arm,exploration_balanced_delta_pp=.1,query_definition_source_sha256=c.QUERY_SOURCE_SHA,new_H1_checkpoint_sha256=None,exploration_paired_sha256='1'*64,root_risk_review_sha256='2'*64)
 def test_unbound_cannot_train(self):
  self.assertEqual(c.verify_files(require_bound=False)['status'],'unbound_template')
  with self.assertRaises(ValueError):c.verify_files()
 def test_actual_admission_only(self):
  for arm in ('A3','A5'):self.assertEqual(c.admission(self.admission(arm))['selected_arm'],arm)
  for field,value in [('selected_arm','J5'),('exploration_balanced_delta_pp',0.),('new_H1_checkpoint_sha256','3'*64),('query_definition_source_sha256','4'*64),('root_authorized',False)]:
   v=self.admission();v[field]=value
   with self.assertRaises(ValueError):c.admission(v)
 def fake_assets(self,arm):
  return dict(arms=['J5',arm],assets={key:dict(path='/synthetic_not_opened/'+key) for key in ['manifest','split','label_conflicts','h1']})
 def test_control_candidate_zero_diff_except_alpha_identity(self):
  for arm in ('A3','A5'):
   p=self.fake_assets(arm);base=c.expected_config('J5',p);other=c.expected_config(arm,p);self.assertEqual(base['training']['seed'],20261010);self.assertEqual(other['training'],base['training']);self.assertEqual(other['base_loss']['alpha'],c.ARMS[arm]);self.assertEqual(base['base_loss']['alpha'],.1)
   for v in (base,other):v['base_loss']['alpha']=0;v['run_id']='same';v['output_dir']='same'
   self.assertEqual(base,other)
 def test_unselected_config_rejected(self):
  p=self.fake_assets('A3')
  with self.assertRaises(ValueError):c.expected_config('A5',p)
  cfg=c.expected_config('A3',p);cfg['training']['seed']=20261008
  with self.assertRaises(ValueError):c.validate_config(cfg,'A3',p)
 def test_original_numeric_source_closure_bytes(self):
  original=json.loads((ROOT/'inputs/exploration_R3_source_manifest.json').read_text(encoding='utf-8'));self.assertEqual(c.sha(ROOT/'scripts/train_n1_peft.py'),'83db63f370d4fe06bf02d90b160a1fc8ba57c39a7df90ef1683be1d0fa14b36f');self.assertEqual(c.sha(ROOT/'scripts/train_peft.py'),'0758b989a1bb33ba851e03f9a852cf35b7ad30847ccbfb62cf528f8dc015a765')
  for rel,digest in original['files'].items():
   if rel.startswith('src/') or rel in ('scripts/owned_launch.py','scripts/train_n1_peft.py','scripts/train_peft.py','scripts/p1_smoke.py','scripts/mainline_h1_contract.py','scripts/bc1_runtime.py'):
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
  for k,v in [('adjusted_initial_free_bytes',50000000000),('disk_max_bytes',21474836481),('atomic_reserve_bytes',0),('min_free_bytes',0),('deadline_unix',.5)]:
   b=self.budget();b[k]=v
   with self.assertRaises(ValueError):validate_budget(b)
 def test_new_frozen_metadata_no_query_reuse(self):
  d=json.loads((ROOT/'inputs/confirmation_assets.json').read_text(encoding='utf-8'));self.assertEqual(d['old0c9_query_connected_group_overlap'],0);self.assertEqual(d['used57189_query_connected_group_overlap'],0);self.assertEqual(d['train_query_connected_group_overlap'],0);self.assertEqual(d['counts'],dict(train=118958,validation=29737));self.assertEqual(c.sha(ROOT/'inputs/confirmation_split.csv'),c.SPLIT_SHA);self.assertEqual(c.sha(ROOT/'inputs/query_definition_source_manifest.json'),c.QUERY_SOURCE_SHA)
 def test_sce_actual_original_numeric(self):
  logits=torch.tensor([[1.,-.4,.2],[.1,.6,-.2]]);targets=torch.tensor([0,2]);out=[]
  for arm in ('J5','A3','A5'):out.append(_base_loss_per_sample(logits,targets,kind='sce',q=.7,label_smoothing=0.,sce_alpha=c.ARMS[arm],sce_beta=1.,sce_log_zero=-4.))
  ce=torch.nn.functional.cross_entropy(logits,targets,reduction='none');self.assertTrue(torch.allclose(out[1]-out[0],.2*ce,atol=1e-6));self.assertTrue(torch.allclose(out[2]-out[0],.4*ce,atol=1e-6))
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
