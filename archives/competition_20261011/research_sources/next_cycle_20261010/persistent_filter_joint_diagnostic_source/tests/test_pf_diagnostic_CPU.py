"""PF-only CPU restoration and provenance fixtures; no real model/checkpoint/images."""
import copy,hashlib,json,os,sys,unittest
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];TRAIN=ROOT.parent/'persistent_filter_joint_source'
sys.path[:0]=[str(ROOT),str(TRAIN/'scripts'),str(TRAIN/'src')]
import torch
import fresh_diag_contract as c
from phase_guard import normalize_owner,validate_actual_launch,BUDGET_SHA
from aictrack1.methods.persistent_filter import PersistentTrainingGate,ActualSupervisionLedger

class PFContracts(unittest.TestCase):
 def fixture(self):
  ix=torch.arange(100);ys=torch.arange(100)//50;gate=PersistentTrainingGate(ix,ys,max_ac_weight=.30);history=[]
  weights=torch.ones(100);updated=torch.ones(100);updated[:8]=.2;updated[50:58]=.2
  for epoch in range(1,21):
   ledger=ActualSupervisionLedger(gate,weights);ledger.observe(ix,ys);actual=ledger.finish()
   nxt=gate.update(ix,updated,torch.zeros(100),torch.ones(100))
   history.append(dict(epoch=epoch,gradient_diagnostics=dict(amp_successful_optimizer_updates=2,amp_skipped_optimizer_updates=0),persistent_filter=dict(actual=actual,next=nxt,max_ac_weight=.3,supervised_samples=actual['supervised'],skipped_optimizer_batches=0,observations=100)))
   weights=updated.clone()
  state=dict(fresh_route_arm='PF30',epoch=20,confidence_weighting=dict(weights=updated.clone(),epochs_updated=20),persistent_filter_state=gate.state_dict(),history=history)
  recipe=dict(persistent_filter=dict(enabled=True,protocol='persistent-supervision-gate-v2',max_ac_weight=.30))
  return state,ix,ys,recipe
 def verify(self,state,ix,ys,recipe):return c.validate_pf_supervision_state(state,ix,ys,recipe,num_classes=2)
 def test_successful_vs_all_masked_batch_accounting(self):
  s,ix,ys,r=self.fixture();s['history'][9]['gradient_diagnostics']['amp_successful_optimizer_updates']=1
  with self.assertRaises(ValueError):self.verify(s,ix,ys,r)
  s,ix,ys,r=self.fixture();s['optimizer']={'state':{0:{'step':torch.tensor(39.)}}}
  with self.assertRaises(ValueError):self.verify(s,ix,ys,r)
 def test_true_applied_distinct_from_terminal_next(self):
  s,ix,ys,r=self.fixture();result=self.verify(s,ix,ys,r)
  self.assertEqual(result['applied_epochs_checked'],20);self.assertEqual(result['per_epoch'][2]['applied']['excluded'],0);self.assertGreater(result['per_epoch'][2]['next']['excluded'],0);self.assertEqual(result['per_epoch'][3]['applied']['excluded'],result['per_epoch'][2]['next']['excluded'])
 def test_gate_AC_weight_drift(self):
  s,ix,ys,r=self.fixture();s['confidence_weighting']['weights'][0]=.25
  with self.assertRaises(ValueError):self.verify(s,ix,ys,r)
 def test_threshold_or_B_state_recipe_rejected(self):
  for change in (lambda s,r:r['persistent_filter'].update(max_ac_weight=.5),lambda s,r:s['persistent_filter_state'].update(protocol='persistent-supervision-gate-v1'),lambda s,r:s.update(fresh_route_arm='B25')):
   s,ix,ys,r=self.fixture();change(s,r)
   with self.assertRaises(ValueError):self.verify(s,ix,ys,r)
 def test_actual_next_substitution_rejected(self):
  s,ix,ys,r=self.fixture();s['history'][2]['persistent_filter']['actual']['excluded']=s['history'][2]['persistent_filter']['next']['excluded']
  with self.assertRaises(ValueError):self.verify(s,ix,ys,r)
 def test_mask_source_and_incomplete_coverage_rejected(self):
  for key,value in [('mask_source_completed_epoch',20),('mask_role','next_epoch_mask'),('observed',99)]:
   s,ix,ys,r=self.fixture();s['history'][7]['persistent_filter']['actual'][key]=value
   with self.assertRaises(ValueError):self.verify(s,ix,ys,r)
 def test_wrong_prior_applied_mass_rejected(self):
  s,ix,ys,r=self.fixture();s['history'][4]['persistent_filter']['actual']['classes'][0]['original_AC_mass']+=.1
  with self.assertRaises(ValueError):self.verify(s,ix,ys,r)
 def test_terminal_next_drift_rejected(self):
  s,ix,ys,r=self.fixture();s['persistent_filter_state']['keep'].fill_(True)
  with self.assertRaises(ValueError):self.verify(s,ix,ys,r)
 def test_class_cap_and_support_rejected(self):
  s,ix,ys,r=self.fixture();s['persistent_filter_state']['keep'][:6]=False
  with self.assertRaises(ValueError):self.verify(s,ix,ys,r)
 def test_birth_types_and_live_source_launch(self):
  controller=dict(pid=31,pgrp=31,session=31,start_ticks='123',cwd=c.TRAIN_ROOT.as_posix())
  launch=dict(controller=controller,command=['/venv/bin/python','-B','scripts/run_supervision_joint.py'],source_manifest_sha256='a'*64,test_access=False,planned_arms=['PF30'],epochs_each=20,planned_pause=False)
  self.assertEqual(validate_actual_launch(launch,'a'*64),controller)
  self.assertEqual(normalize_owner(dict(controller,start_ticks=123))['start_ticks'],'123')
  for changed in (dict(launch,test_access=True),dict(launch,planned_arms=['PF50']),dict(launch,command=launch['command']+['--smoke','PF30']),dict(launch,source_manifest_sha256='b'*64)):
   with self.assertRaises(ValueError):validate_actual_launch(changed,'a'*64)
  for bad in (True,None,0,'-1'):
   with self.assertRaises(ValueError):normalize_owner(dict(controller,start_ticks=bad))
 def test_actual_producer_and_truthful_future_bindings(self):
  digest=c.sha(TRAIN/'source_manifest.json');a,m=c.verify_training_binding(TRAIN,digest,binding_only=True)
  self.assertEqual(m['arms'],['PF30']);self.assertEqual(c.BINDING['status'],'unbound_template');self.assertIsNone(c.TRAINING_SOURCE_SHA);self.assertIsNone(c.BINDING['actual_launch_sha256']);self.assertEqual(c.BINDING['future_checkpoint_sha256'],{'PF30':None});self.assertEqual(c.sha(ROOT/'inputs/budget.json'),BUDGET_SHA)
  from p1_contract import verify_files,assets,validate_config
  import yaml
  verify_files(TRAIN);raw=yaml.safe_load((TRAIN/'configs/experiment/p1_PF30.yaml').read_text())['experiment'];validate_config(raw,'PF30',assets(TRAIN));self.assertEqual(raw['base_loss']['alpha'],.1)
 def test_joint_head_policy(self):
  state={'visual.a':torch.zeros(2),'visual.b':torch.zeros(3),'classifier.linear.weight':torch.zeros(750,512),'classifier.linear.bias':torch.zeros(750)}
  self.assertEqual(c.validate_state(state,arm='PF30',visual_count=2,visual_numel=5)['head_tensors'],2)
  del state['classifier.linear.bias']
  with self.assertRaises(ValueError):c.validate_state(state,arm='PF30',visual_count=2,visual_numel=5)
 def test_frozen_numerical_and_query_bytes(self):
  parent=json.loads((ROOT/'inputs/parent_alpha_diagnostic_source_manifest.json').read_text())
  for rel,digest in parent['files'].items():
   if rel in ('paired_metrics.py','view_features.py','diagnostic_resources.py','fixed_graph_protocol.json','frozen_postprocess.json') or rel.startswith(('reference/','inputs/frozen_queries/','inputs/legacy_control/')) or rel in ('inputs/control_import.json','inputs/confirmation_rows.csv'):
    self.assertEqual(c.sha(ROOT/rel),digest,rel)
  self.assertEqual(c.verify_owned_sources(),c.sha(ROOT/'source_manifest.json'))
if __name__=='__main__':unittest.main()
