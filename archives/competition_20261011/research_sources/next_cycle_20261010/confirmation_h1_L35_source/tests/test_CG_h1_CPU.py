"""New CG-only H1 contracts; original numeric engine is reused, no candidate model."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import confirmation_contract as c
import CG_admission_contract as a
from cg_fixture import make_admission,budget,write

class H1Contracts(unittest.TestCase):
 def test_unbound_and_future_pointers(self):
  with self.assertRaises(ValueError):c.verify_source()
  self.assertEqual(c.verify_source(False)['status'],'unbound_template')
  f=c.read_json(ROOT/'inputs/future_binding.json');self.assertIsNone(f['admission_sha256']);self.assertIsNone(f['budget_sha256']);self.assertIsNone(f['new_H1_checkpoint_sha256']);self.assertEqual(f['visual_checkpoint_sha256'],{'J5':None,'CG25':None})
 def test_original_H1_query_feature_engine_bytes(self):
  parent=c.read_json(ROOT/'inputs/parent_generic_template_manifest.json')
  for rel in ['scripts/train_frozen_head.py','scripts/cpu_feature_preflight.py','scripts/prepare_confirmation_assets.py','scripts/run_confirmation_h1.py','scripts/owned_launch.py','scripts/launch_gate.py','src/aictrack1/engine/frozen_head.py','src/aictrack1/engine/training_queue.py','inputs/feature_identity.json','inputs/confirmation_rows.csv','inputs/confirmation_split.csv','inputs/query_identity.json']:
   self.assertEqual(c.sha(ROOT/rel),parent['files'][rel],rel)
  self.assertEqual(len(c.read_json(ROOT/'inputs/feature_identity.json')['shards']),37);self.assertEqual(c.HEAD_CONFIG['seed'],20261008);self.assertEqual(c.HEAD_CONFIG['epochs'],100)
 def test_original_new_query_positive_metadata(self):
  p=c.normalized_assets();self.assertEqual(p['counts'],{'train':118958,'validation':29737});self.assertEqual(p['old0c9_query_connected_group_overlap'],0);self.assertEqual(p['used57189_query_connected_group_overlap'],0);self.assertEqual(p['train_query_connected_group_overlap'],0)
 def test_actual_synthetic_ref_contract_no_extra_natural_CI_gate(self):
  with tempfile.TemporaryDirectory(prefix='CG_H1_contract_') as td:
   value,_,_=make_admission(td);self.assertEqual(c.validate_admission(value)['selected_arm'],'CG25')
 def test_foreign_raw_negative_future_identity_rejected(self):
  with tempfile.TemporaryDirectory(prefix='CG_H1_contract_') as td:
   value,_,_=make_admission(td)
   for key,bad in [('schema','aictrack1.next_cycle10.B_admission.v1'),('selected_arm','A3'),('selected_arm','CG50'),('primary_scope','raw_CG_P7'),('exploration_balanced_delta_pp',0.),('exploration_balanced_delta_pp',float('nan')),('exploration_training_template_sha256','b'*64),('exploration_training_source_sha256','f'*64),('center_guard_fraction',.5),('new_H1_checkpoint_sha256','a'*64),('query_definition_source_sha256','9'*64),('test_access',True)]:
    x=copy.deepcopy(value);x[key]=bad
    with self.assertRaises(ValueError,msg=key):c.validate_admission(x)
 def test_actual_reference_and_risk_link_drift(self):
  with tempfile.TemporaryDirectory(prefix='CG_H1_contract_') as td:
   value,_,review=make_admission(td);review['admit_independent_confirmation']=False;digest=write(Path(value['root_risk_review']['path']),review);value['root_risk_review_sha256']=digest;value['root_risk_review']['sha256']=digest
   with self.assertRaises(ValueError):c.validate_admission(value)
   value,_,_=make_admission(td);path=Path(value['exploration_paired']['path']);path.write_bytes(path.read_bytes()+b'\n')
   with self.assertRaises(ValueError):c.validate_admission(value)
 def test_actual_shared_future_budget_no_reset_or_lost_tail(self):
  with tempfile.TemporaryDirectory(prefix='CG_budget_') as td:
   original=budget(td);self.assertEqual(c.validate_budget(original,15)['disk_max_bytes'],30*1024**3)
   for key,bad in [('adjusted_initial_free_bytes',45226205185),('deadline_unix',None),('phase_start_unix',float('nan')),('disk_max_bytes',31*1024**3),('atomic_reserve_bytes',0),('final_gpu_reserve_hours',0),('min_free_bytes',0),('capacity_evidence',{'path':'relative','sha256':'1'*64})]:
    x=copy.deepcopy(original);x[key]=bad
    with self.assertRaises(ValueError,msg=key):c.validate_budget(x)
   with self.assertRaises(ValueError):c.validate_budget(original,20)
 def test_class_byte_and_logical_mapping(self):
  report=c.audit_class_mapping(ROOT/'test_fixtures/classes.csv');self.assertNotEqual(report['file_sha256'],report['logical_mapping_sha256']);self.assertEqual(report['num_classes'],750)
if __name__=='__main__':unittest.main(verbosity=2)
