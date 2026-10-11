"""Meaningful B delivery CPU contract/teacher/student/GO tests; no real model/images."""
from pathlib import Path
import ast,copy,hashlib,shutil,importlib.util,io,json,math,os,sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];FINAL=ROOT.parent/'B_final_inference_source';PREP=ROOT.parent/'B_delivery_preparation_r1'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(FINAL/'scripts')]
import torch
import full_route_contract as c
import B_soft_checkpoint_audit as soft
from aictrack1.methods.josrc_soft import recipe
spec=importlib.util.spec_from_file_location('B_delivery_final_contract',FINAL/'scripts/final_contract.py');fc=importlib.util.module_from_spec(spec);spec.loader.exec_module(fc)
import launch_gate
from mainline_guard import tensor_storage_bytes

def write(p,v):p.write_text(json.dumps(v,sort_keys=True),encoding='utf-8');return {'path':str(p.resolve()),'sha256':c.sha(p)}
def proof(root,arm='B25'):
 v=dict(schema='aictrack1.next_cycle10.B_confirmation.v1',status='complete_positive_independent_B_confirmation',selected_arm=arm,selected_route=arm+'_GRAPH',primary_scope=c.PRIMARY_SCOPE,split_sha256=c.SPLIT_SHA,query_definition_source_sha256=c.QUERY_SOURCE_SHA,query_identity_sha256=c.QUERY_IDENTITY_SHA,training_seed=20261010,checkpoint_epoch=20,checkpoint_role='last',selection_rule='fixed_final20',train_count=118958,natural_count=29737,balanced_count=14760,test_access=False,used_test_or_online_scores=False,root_risk_reviewed=True,final_model='student_only',math_source_sha256=c.MATH_SOURCE_SHA,math_protocol_sha256=c.MATH_PROTOCOL_SHA,soft_coefficient=c.ARMS[arm],primary_delta_pp=.17,natural_delta_pp=-.01,new_H1_checkpoint_sha256='a'*64,new_H1_source_sha256='b'*64,control_last_sha256='c'*64,candidate_last_sha256='d'*64,diagnostic_source_manifest_sha256='e'*64,evidence={})
 train=dict(schema='aictrack1.next_cycle10.confirmation.visual.source.v1',status='bound',template_manifest_sha256=c.CONFIRMATION_TEMPLATE_SHA,arms=['J5',arm],training_seed=20261010,h1_checkpoint_sha256=v['new_H1_checkpoint_sha256'],source_h1_manifest_sha256=v['new_H1_source_sha256'],query_definition_source_sha256=c.QUERY_SOURCE_SHA)
 train.update(training_scope='fixed_split_validation',fixed_epochs=20,checkpoint_role='last',files={})
 for rel in ['src/aictrack1/methods/josrc_soft.py','inputs/soft_math_protocol.json']:
  p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p);train['files'][rel]=c.sha(p)
 e=v['evidence'];e['training_source']=write(root/'train.json',train);v['training_source_manifest_sha256']=e['training_source']['sha256']
 report={k:v[k] for k in ['selected_arm','primary_scope','primary_delta_pp','natural_delta_pp','split_sha256','query_identity_sha256','training_source_manifest_sha256','diagnostic_source_manifest_sha256']};report['raw_P7_separately_reported']=True;e['paired_report']=write(root/'paired.json',report)
 e['root_review']=write(root/'review.json',dict(root_authorized=True,decision='allow_single_B_full_training',selected_arm=arm,paired_report_sha256=e['paired_report']['sha256']))
 for key,a,checkpoint in [('control_audit','J5',v['control_last_sha256']),('candidate_audit',arm,v['candidate_last_sha256'])]:
  audit=dict(complete=True,arm=a,epoch=20,last_sha256=checkpoint,source_manifest_sha256=v['training_source_manifest_sha256'],h1_checkpoint_sha256=v['new_H1_checkpoint_sha256'],split_sha256=c.SPLIT_SHA,train_count=118958,validation_count=29737,training_seed=20261010,actual_runtime=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',actual_torch_threads=4),student_model_CPU_restored=True,josrc_soft_state_complete=key=='candidate_audit',final_model='student_only',math_source_sha256=c.MATH_SOURCE_SHA,math_protocol_sha256=c.MATH_PROTOCOL_SHA)
  e[key]=write(root/(key+'.json'),audit)
 return v

def teacher_fixture(arm='B25'):
 counts=torch.full((750,),198,dtype=torch.int64);counts[:195]+=1;total=int(counts.sum());assert total==148695;batches=(total+63)//64;r=recipe(c.ARMS[arm]);raw={'visual.a':torch.zeros(2),'visual.b':torch.zeros(3),'classifier.linear.weight':torch.zeros(750,512),'classifier.linear.bias':torch.zeros(750)};hist=[]
 for ep in range(1,21):
  detail=dict(recipe=r,final_model='student_only',soft_active=ep>5,additional_no_grad_forwards_per_batch=2 if ep>5 else 0,hard_branch_dropped_samples=0,soft_only_additive=True,EMA_successful_updates=ep*batches-2,EMA_skipped_updates=2,hard_seen=(counts*ep).tolist(),soft_mass=(counts.double()*max(0,ep-5)*.01).tolist(),teacher_transitions_total=max(0,ep-5)*total,active_teacher_observations=max(0,ep-5)*total)
  hist.append(dict(epoch=ep,josrc_soft=detail,gradient_diagnostics=dict(amp_successful_optimizer_updates=batches-(2 if ep==1 else 0),amp_skipped_optimizer_updates=2 if ep==1 else 0)))
 success=20*batches-2
 state=dict(fresh_route_arm=arm,trainable_state=raw,history=hist,josrc_math_protocol_sha256=c.sha(ROOT/'inputs/soft_math_protocol.json'),josrc_teacher=dict(schema=r['protocol'],recipe=r,final_model='student_only',successful_updates=success,skipped_updates=2,ema=dict(decay=.999,updates=success,shadow={k:v.clone() for k,v in raw.items()})),josrc_support=dict(num_classes=750,hard_branch_dropped_samples=0,soft_only_additive=True,hard_seen=counts*20,soft_mass=counts.double()*15*.01,teacher_transitions=torch.diag(counts*15),active_teacher_observations=total*15),optimizer=dict(state={i:dict(step=torch.tensor(float(success))) for i in range(len(raw))}))
 return state,dict(josrc_soft=r),counts

class BDelivery(unittest.TestCase):
 def test_actual_file_positive_refs_and_negative_natural_are_allowed(self):
  for arm in c.ARMS:
   with tempfile.TemporaryDirectory(dir=PREP) as d:self.assertEqual(c.verify_confirmation(proof(Path(d),arm))['selected_arm'],arm)
 def test_old_exploration_or_alpha_or_naked_J5_receipts_rejected(self):
  with tempfile.TemporaryDirectory(dir=PREP) as d:
   v=proof(Path(d))
   for field,value in [('schema','aictrack1.fresh_route.confirmation.v1'),('primary_scope','J5_graph_P7_vs_C'),('split_sha256','57189'),('training_seed',20261008),('checkpoint_epoch',19),('primary_delta_pp',0),('used_test_or_online_scores',True),('new_H1_checkpoint_sha256',c.FULL_H1_SHA)]:
    q=copy.deepcopy(v);q[field]=value
    with self.assertRaises(ValueError):c.verify_confirmation(q)
 def test_real_source_or_rootreview_hash_mutation_rejected(self):
  with tempfile.TemporaryDirectory(dir=PREP) as d:
   v=proof(Path(d));Path(v['evidence']['root_review']['path']).write_text('{}')
   with self.assertRaises(ValueError):c.verify_confirmation(v)
 def test_b_full_recipe_one_candidate_preserves_scientific_parameters(self):
  assets=json.loads((ROOT/'inputs/legal_full_asset_reference.json').read_text())
  for arm in c.ARMS:
   p={**assets,'selected_arm':arm};raw=c.expected_config(arm,p);self.assertEqual(raw['training']['seed'],20261010);self.assertEqual(raw['training']['epochs'],20);self.assertEqual(raw['base_loss'],dict(kind='sce',alpha=.1,beta=1.,log_zero=-4.));self.assertEqual(raw['training_scope'],'full_train_fixed_epochs');self.assertEqual(raw['josrc_soft'],recipe(c.ARMS[arm]));self.assertEqual(raw['expected_train_count'],148695)
   bad=copy.deepcopy(raw);bad['training']['head_learning_rate']=0
   with self.assertRaises(ValueError):c.validate_config(bad,arm,p)
 def test_full_148695_teacher_support_and_AMP_counts(self):
  for arm in c.ARMS:
   v,r,n=teacher_fixture(arm);a=soft.validate_soft_state(v,r,n);self.assertTrue(a['raw_student_only']);self.assertFalse(a['teacher_used_for_inference']);self.assertEqual(a['EMA_successful_updates'],46478)
 def test_teacher_warmup_or_head_or_migration_corruption_rejected(self):
  for change in ['warmup','head','migration','steps']:
   v,r,n=teacher_fixture()
   if change=='warmup':v['history'][0]['josrc_soft']['teacher_transitions_total']=148695
   elif change=='head':v['josrc_teacher']['ema']['shadow'].pop('classifier.linear.bias')
   elif change=='migration':v['josrc_support']['teacher_transitions'][0,0]-=1
   else:v['optimizer']['state'][0]['step']-=1
   with self.assertRaises(ValueError):soft.validate_soft_state(v,r,n)
 def test_final_b_routes_restore_joint_student_tensors_not_teacher(self):
  for arm in c.ARMS:
   v,r,n=teacher_fixture(arm);export=dict(trainable_state={k:t.clone() for k,t in v['trainable_state'].items()});size=sum(t.numel() for t in v['trainable_state'].values())
   fc.validate_joint_tensors(v,export,torch,route=arm+'_GRAPH',expected_visual=2,expected_numel=size)
   export['trainable_state']['classifier.linear.bias'][0]+=1
   with self.assertRaises(ValueError):fc.validate_joint_tensors(v,export,torch,route=arm+'_GRAPH',expected_visual=2,expected_numel=size)
   with self.assertRaises(ValueError):fc.validate_joint_tensors(v,export,torch,route='J5_GRAPH',expected_visual=2,expected_numel=size)
 def test_FULL_final_metadata_raw_student_only_and_teacher_rejected(self):
  for arm in c.ARMS:
   last,r,n=teacher_fixture(arm);raw=last['trainable_state'];spec=r['josrc_soft'];cfg='1'*64;finalsha='2'*64;lastsha='3'*64;head='4'*64;source='5'*64
   meta=dict(config_sha256=cfg,visual_input_resolution=336,position_interpolation='bicubic_7x7_to_10x10',training_scope='full_train_fixed_epochs',fresh_route_full_arm=arm,selected_route=arm+'_GRAPH',fresh_route_full_source_manifest_sha256=source,classifier_head_source='joint_fixed_last_state',classifier_head_restored_equal=False,final_model='student_only',josrc_recipe=spec,josrc_math_protocol_sha256=c.sha(ROOT/'inputs/soft_math_protocol.json'),competition_stage='stage2',clip_weight_sha256=fc.CLIP_SHA,train_manifest_sha256=fc.MANIFEST_SHA,class_mapping_sha256=fc.MAPPING_SHA)
   last.update(meta,epoch=20,initial_validation=None,best_state=None,mode='full_visual',scheduler={},scaler={},python_rng_state=(),numpy_rng_state=(),torch_rng_state=torch.zeros(3),cuda_rng_state_all=[],confidence_weighting={})
   final={**meta,'trainable_state':raw,'head_initializer_sha256':head,'selection_rule':'fixed final epoch 20; no validation selection','normalize_features':True,'mode':'full_visual'}
   metrics={**meta,'train_count':148695,'validation_count':0,'early_stop':None,'initial_validation':None,'best_top1':None,'selected_checkpoint':'final.pt','selection_rule':'fixed final epoch 20; no validation selection','selected_checkpoint_sha256':finalsha,'final_checkpoint_sha256':finalsha,'last_checkpoint_sha256':lastsha,'head_initializer':{'sha256':head},'history':last['history'],'trainable_total_parameters':88233966,'base_loss':dict(kind='sce',alpha=.1,beta=1.,log_zero=-4.)}
   with patch.object(fc,'FULL_SOURCE_SHA',source):
    self.assertTrue(fc.validate_metrics(metrics,last,final,config_sha=cfg,final_sha=finalsha,last_sha=lastsha,head_sha=head,arm=arm))
    final['josrc_teacher']=last['josrc_teacher']
    with self.assertRaises(ValueError):fc.validate_metrics(metrics,last,final,config_sha=cfg,final_sha=finalsha,last_sha=lastsha,head_sha=head,arm=arm)
 def test_atomic_reserve_counts_auxiliary_teacher_and_support(self):
  v,r,n=teacher_fixture();student=tensor_storage_bytes(v['trainable_state']);total=tensor_storage_bytes(v);self.assertGreater(total,2*student)
 def test_FULL_GO_exact_single_arm_resume_and_EOF(self):
  for arm in c.ARMS:
   root=ROOT.resolve();env=dict(P1_ARM=arm,P1_RESUME='1',P1_SMOKE='0');cmd=[sys.executable,'scripts/train_n1_peft.py','--config',f'configs/experiment/p1_full_{arm}.yaml','--resume',str(root/f'artifacts/experiments/FULL_{arm}/last.pt')];called=[]
   def fake_exec(*args):called.append(args);raise SystemExit(0)
   with patch.object(launch_gate,'os',SimpleNamespace(name='posix',environ=os.environ,execvpe=fake_exec)),patch.object(launch_gate,'ROOT',root),patch.object(launch_gate,'REMOTE_ROOT',root),patch.object(launch_gate.Path,'cwd',return_value=root),patch.object(launch_gate,'assets',return_value={'selected_arm':arm}),patch.dict(os.environ,env),patch.object(sys,'argv',['gate','--',*cmd]),patch.object(sys,'stdin',type('S',(),{'buffer':io.BytesIO(b'G')})()):
    with self.assertRaises(SystemExit):launch_gate.main()
   self.assertEqual(called[0][1],cmd)
   with patch.object(launch_gate,'os',SimpleNamespace(name='posix',environ=os.environ,execvpe=fake_exec)),patch.object(launch_gate,'ROOT',root),patch.object(launch_gate,'REMOTE_ROOT',root),patch.object(launch_gate.Path,'cwd',return_value=root),patch.object(launch_gate,'assets',return_value={'selected_arm':arm}),patch.dict(os.environ,env),patch.object(sys,'argv',['gate','--',*cmd]),patch.object(sys,'stdin',type('S',(),{'buffer':io.BytesIO(b'')})()):
    self.assertEqual(launch_gate.main(),73)
 def test_scientific_math_views_graph_P7_exact_and_future_inputs_null(self):
  sci=ROOT.parent/'josrc_soft_r2_source';old=ROOT.parent.parent/'next_cycle_20261009/fresh_route_retry_final_inference_r1_source'
  self.assertEqual(c.sha(ROOT/'src/aictrack1/methods/josrc_soft.py'),c.MATH_SOURCE_SHA);self.assertEqual(c.sha(FINAL/'src/aictrack1/methods/josrc_soft.py'),c.MATH_SOURCE_SHA)
  for rel in ['scripts/view_features.py','scripts/p3_graph.py','scripts/fixed_final_graph.py','src/aictrack1/evaluation/four_view.py','src/aictrack1/evaluation/balanced_prior.py','src/aictrack1/data/transforms.py','inputs/p7_selection.json','inputs/fixed_graph_protocol.json']:self.assertEqual(c.sha(FINAL/rel),c.sha(old/rel),rel)
  for root in [ROOT,FINAL]:
   future=json.loads((root/'inputs/FUTURE_INPUTS.json').read_text());self.assertTrue(all(v is None or v is False for v in future.values()))
if __name__=='__main__':torch.set_num_threads(4);unittest.main(verbosity=2)
