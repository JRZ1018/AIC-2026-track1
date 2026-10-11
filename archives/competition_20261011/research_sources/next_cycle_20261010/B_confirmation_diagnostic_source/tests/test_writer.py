"""Real temporary evidence files, exact FULL contract, synthetic models only."""
import copy,json,sys,tempfile,unittest,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TRAIN=ROOT.parent/'confirmation_visual_B_source';sys.path[:0]=[str(ROOT),str(TRAIN/'src'),str(TRAIN/'scripts')]
import fresh_diag_contract as c
from write_confirmation import make_confirmation
from reference.B_full_route_contract import verify_confirmation,validate_confirmation

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');return dict(path=str(path.resolve()),sha256=c.sha(path))
def fixture(folder):
 train=folder/'train';(train/'src/aictrack1/methods').mkdir(parents=True);(train/'inputs').mkdir()
 shutil.copyfile(TRAIN/'src/aictrack1/methods/josrc_soft.py',train/'src/aictrack1/methods/josrc_soft.py');shutil.copyfile(ROOT/'inputs/soft_math_protocol.json',train/'inputs/soft_math_protocol.json')
 files={p.relative_to(train).as_posix():c.sha(p) for p in train.rglob('*') if p.is_file()}
 source=dict(schema='aictrack1.next_cycle10.confirmation.visual.source.v1',status='bound',template_manifest_sha256=c.TEMPLATE_SHA,arms=['J5','B25'],selected_candidate='B25',training_seed=20261010,h1_checkpoint_sha256='a'*64,source_h1_manifest_sha256='b'*64,query_definition_source_sha256=c.QUERY_SOURCE_SHA,training_scope='fixed_split_validation',fixed_epochs=20,checkpoint_role='last',files=files)
 refs={'training_source':write(train/'source_manifest.json',source)};diag=c.verify_owned_sources()
 report=dict(schema='aictrack1.next_cycle10.B_confirmation.paired.v1',complete=True,independent_confirmation=True,selected_arm='B25',primary_scope=c.PRIMARY_SCOPE,primary_delta_pp=.1,natural_delta_pp=-.03,split_sha256=c.SPLIT_SHA,query_identity_sha256=c.QUERY_IDENTITY_SHA,training_source_manifest_sha256=refs['training_source']['sha256'],diagnostic_source_manifest_sha256=diag,raw_P7_separately_reported=True,test_access=False,used_test_or_online_scores=False,new_H1_checkpoint_sha256='a'*64,new_H1_source_sha256='b'*64,control_last_sha256='c'*64,candidate_last_sha256='d'*64,CI_crosses_zero=True)
 refs['paired_report']=write(folder/'paired.json',report)
 refs['root_review']=write(folder/'risk.json',dict(root_authorized=True,decision='allow_single_B_full_training',selected_arm='B25',paired_report_sha256=refs['paired_report']['sha256']))
 common=dict(complete=True,epoch=20,source_manifest_sha256=refs['training_source']['sha256'],h1_checkpoint_sha256='a'*64,split_sha256=c.SPLIT_SHA,train_count=118958,validation_count=29737,training_seed=20261010,actual_runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','actual_torch_threads':4},student_model_CPU_restored=True,josrc_soft_state_complete=True,final_model='student_only',math_source_sha256=c.MATH_SOURCE_SHA,math_protocol_sha256=c.MATH_PROTOCOL_SHA,diagnostic_source_manifest_sha256=diag,raw_student_only=True,teacher_used_for_inference=False,inference_optimizer_steps=0)
 refs['control_audit']=write(folder/'control.json',dict(common,arm='J5',last_sha256='c'*64,auxiliary_soft_coefficient=0.,teacher_active_epochs_checked=0))
 refs['candidate_audit']=write(folder/'candidate.json',dict(common,arm='B25',last_sha256='d'*64,auxiliary_soft_coefficient=.25,teacher_active_epochs_checked=15))
 return refs

def mutate(refs,key,values):
 item=refs[key];p=Path(item['path']);v=json.loads(p.read_text());v.update(values);refs[key]=write(p,v)

class Writer(unittest.TestCase):
 def test_positive_primary_negative_natural_and_crossing_CI_allowed(self):
  with tempfile.TemporaryDirectory() as d:
   refs=fixture(Path(d));v=make_confirmation(refs);self.assertEqual(v['schema'],'aictrack1.next_cycle10.B_confirmation.v1');self.assertLess(v['natural_delta_pp'],0);verify_confirmation(v)
 def test_no_positive_primary_rejected(self):
  for delta in [0.,-.1]:
   with tempfile.TemporaryDirectory() as d:
    refs=fixture(Path(d));mutate(refs,'paired_report',{'primary_delta_pp':delta});mutate(refs,'root_review',{'paired_report_sha256':refs['paired_report']['sha256']})
    with self.assertRaises(ValueError):make_confirmation(refs)
 def test_immutable_reference_drift_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   refs=fixture(Path(d));Path(refs['paired_report']['path']).write_text('{}')
   with self.assertRaises(ValueError):make_confirmation(refs)
 def test_root_review_different_report_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   refs=fixture(Path(d));mutate(refs,'root_review',{'paired_report_sha256':'1'*64})
   with self.assertRaises(ValueError):make_confirmation(refs)
 def test_old_fold_or_H1_or_teacher_inference_rejected(self):
  for key,value in [('training_seed',20261008),('h1_checkpoint_sha256','e88a1cbbaa8d9ebdbf1c7cf2707ac434cf440fac8bd5b4a5dc57893ba0fe15be'),('teacher_used_for_inference',True),('actual_runtime',{'actual_torch_threads':1})]:
   with tempfile.TemporaryDirectory() as d:
    refs=fixture(Path(d));mutate(refs,'control_audit',{key:value})
    with self.assertRaises(ValueError):make_confirmation(refs)
 def test_missing_teacher_or_nonzero_control_rejected(self):
  for values in [{'josrc_soft_state_complete':False},{'auxiliary_soft_coefficient':.25},{'teacher_active_epochs_checked':15}]:
   with tempfile.TemporaryDirectory() as d:
    refs=fixture(Path(d));mutate(refs,'control_audit',values)
    with self.assertRaises(ValueError):make_confirmation(refs)
 def test_wrong_template_manifest_header_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   refs=fixture(Path(d));mutate(refs,'training_source',{'template_manifest_sha256':'0'*64});mutate(refs,'paired_report',{'training_source_manifest_sha256':refs['training_source']['sha256']});mutate(refs,'root_review',{'paired_report_sha256':refs['paired_report']['sha256']})
   for key in ['control_audit','candidate_audit']:mutate(refs,key,{'source_manifest_sha256':refs['training_source']['sha256']})
   with self.assertRaises(ValueError):make_confirmation(refs)
 def test_missing_complete20_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   refs=fixture(Path(d));mutate(refs,'candidate_audit',{'epoch':19})
   with self.assertRaises(ValueError):make_confirmation(refs)
if __name__=='__main__':unittest.main(verbosity=2)
