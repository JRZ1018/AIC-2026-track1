"""Synthetic metadata/hash joins; no real models/images/GPU/server."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import append_diagnostic_contract as a
import write_append_confirmation as w
import fresh_diag_contract as c
import numpy as np

class Budget(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.train=self.root/'training';self.train.mkdir()
  p=patch.object(a,'TRAINING_ROOT',self.train);p.start();self.addCleanup(p.stop)
  cfg={arm:self.put(self.train/'configs/experiment'/('p1_'+arm+'.yaml'),{'fixture':arm}) for arm in ('C','J5')}
  src=self.put(self.train/'source_manifest.json',dict(schema='aictrack1.fresh_route.training.source.v1',status='bound',selected_route='J5_GRAPH',selected_arm='J5',original_phase_start_unix=a.ORIGINAL_START,original_deadline_unix=a.ORIGINAL_DEADLINE,original_data_free_baseline_bytes=a.ORIGINAL_BASELINE,disk_max_bytes=a.DISK_MAX,min_free_bytes=a.MIN_FREE,files={'configs/experiment/p1_'+arm+'.yaml':item['sha256'] for arm,item in cfg.items()}))
  self.source=src['sha256'];old=self.put(self.train/'artifacts/queues/fresh_route_budget.json',dict(schema='aictrack1.fresh_route.budget.v1',phase_start_unix=a.ORIGINAL_START,deadline_unix=a.ORIGINAL_DEADLINE,initial_free_bytes=a.ORIGINAL_BASELINE,disk_max_bytes=a.DISK_MAX,min_free_bytes=a.MIN_FREE))
  origins={arm:dict(path=str((self.train/'artifacts/experiments'/('FRESH_'+arm)/'last.pt').resolve()),epoch=20,sha256=('c' if arm=='C' else 'd')*64) for arm in ('C','J5')}
  births=[dict(role=role,pid=pid,pgrp=pid,session=pid,start_ticks=ticks) for role,pid,ticks in [('C_controller',236091,'308497119'),('C_trainer',236098,'308497585'),('J5_controller',246091,'309497119'),('J5_trainer',246098,'309497585')]]
  self.b=dict(schema=a.SCHEMA,status='root_authorized_append_only_budget',root_authorized=True,selected_route='J5_GRAPH',selected_arm='J5',test_access=False,new_fresh_training=False,scope=a.SCOPES,training_root=str(self.train.resolve()),source_manifest_sha256=self.source,original_phase_start_unix=a.ORIGINAL_START,original_deadline_unix=a.ORIGINAL_DEADLINE,initial_free_bytes=a.ORIGINAL_BASELINE,disk_max_bytes=a.DISK_MAX,min_free_bytes=a.MIN_FREE,phase_start_unix=a.APPEND_START,deadline_unix=a.APPEND_MAX_DEADLINE,original_budget=old,training_source=src,configs=cfg,checkpoints=origins,watched=births)
  self.path=self.root/'append.json';self.sign()
 def put(self,path,value):
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n');return dict(path=str(path.resolve()),sha256=a.sha(path))
 def sign(self):self.ref=self.put(self.path,self.b)
 def context(self,**kw):return a.validate_append_budget(self.path,self.ref['sha256'],expected_training_source_sha=self.source,**kw)
 def test_valid_shared_window_baseline_unchanged(self):
  ctx=self.context(live=True,now=a.APPEND_START+1);self.assertEqual(ctx['original_system_baseline_bytes'],a.ORIGINAL_BASELINE);self.assertFalse(ctx['training_budget_reset']);self.assertFalse(ctx['graph_ledger_reset'])
 def test_live_time_cannot_precede_or_outlive_shared_stage(self):
  for now in (a.APPEND_START-1,a.APPEND_MAX_DEADLINE):
   with self.subTest(now=now),self.assertRaises(ValueError):self.context(live=True,now=now)
 def test_no_second_budget_other_route_test_or_baseline_reset(self):
  original=copy.deepcopy(self.b)
  for key,value in [('phase_start_unix',a.APPEND_START+1),('deadline_unix',a.APPEND_MAX_DEADLINE+1),('initial_free_bytes',a.ORIGINAL_BASELINE+1),('selected_route','S'),('test_access',True),('scope',a.SCOPES+['new_fresh_training'])]:
   self.b=copy.deepcopy(original);self.b[key]=value;self.sign()
   with self.subTest(key=key),self.assertRaises(ValueError):self.context()
 def test_original_budget_hash_drift_fails(self):
  Path(self.b['original_budget']['path']).write_text('{}')
  with self.assertRaises(ValueError):self.context()
 def test_resigned_changed_original_deadline_fails(self):
  old=a.read(self.b['original_budget']['path']);old['deadline_unix']+=1;self.b['original_budget']=self.put(Path(self.b['original_budget']['path']),old);self.sign()
  with self.assertRaises(ValueError):self.context()
 def test_config_file_hash_drift_fails(self):
  Path(self.b['configs']['J5']['path']).write_text('{}')
  with self.assertRaises(ValueError):self.context()
 def test_missing_actual_J5_births_fails(self):
  self.b['watched']=self.b['watched'][:2];self.sign()
  with self.assertRaises(ValueError):self.context()
 def test_entry_cannot_renew_shared_deadline(self):
  ctx=self.context();a.require_deadline(ctx,a.APPEND_MAX_DEADLINE,now=a.APPEND_START+1)
  with self.assertRaises(ValueError):a.require_deadline(ctx,a.APPEND_MAX_DEADLINE+1,now=a.APPEND_START+1)
 def test_missing_continuation_evidence_fails_closed(self):
  with self.assertRaises(ValueError):w.verify_continuation_chain({},self.context(),'d'*64)
 def test_missing_actual_completion_time_fails_closed(self):
  ctx=self.context();items={key:ctx[key] for key in ('original_fresh_budget','appended_stage_budget')}
  with self.assertRaises(ValueError):w.validate_append_evidence(dict(append_stage_context=ctx),items,ctx)
 def test_two_cache_contexts_must_match(self):
  ctx=self.context()
  def cache(arm):return dict(identity=dict(model_arm=arm,checkpoint_role='last',checkpoint_epoch=20,selected_route='J5_GRAPH',append_stage_context=ctx,training_source_manifest_sha256=self.source,experiment_config_sha256=ctx['configs'][arm]['sha256']))
  j,control=cache('J5'),cache('C');a.require_pair_context(j,control,ctx);j=copy.deepcopy(j);j['identity']['append_stage_context']['deadline_unix']+=1
  with self.assertRaises(ValueError):a.require_pair_context(j,control,ctx)
 def test_actual_file_hash_no_resume_chain_and_wrong_time(self):
  ctx=self.context();member=self.root/'operator/op.py';member.parent.mkdir();member.write_text('# synthetic operator fixture\n');op=self.put(member.parent/'source_manifest.json',dict(schema='aictrack1.fresh_route.continuation.training.source.v1',files={'op.py':a.sha(member)}))
  flags=('strict_cpu_model_restored','classifier_parameters_restored_equal','optimizer_cpu_restored','scheduler_cpu_restored','scaler_cpu_restored','ac_cpu_restored','exact_ac_train_id_label_binding','python_rng_restored','numpy_rng_restored','torch_cpu_rng_restored','saved_cuda_rng_structure_valid','confidence_observation_binding','saved_state_resume_bookkeeping_valid')
  states={arm:dict(checkpoint_sha256=ctx['checkpoint_origins'][arm]['sha256'],checkpoint_epoch=20,config_sha256=ctx['configs'][arm]['sha256'],source_manifest_sha256=self.source,split_sha256=c.SPLIT_SHA,h1_sha256=c.H1_SHA,trainable_tensors=152 if arm=='C' else 154,trainable_numel=87849216 if arm=='C' else 88233966,status='complete_fixed_last20_audit',complete=True,optimizer_parameter_groups=[152] if arm=='C' else [152,2],scheduler_epoch=20,scheduler_T_max=20,ac_observation_epochs=20,cuda_rng_restored_on_gpu=False,classifier_head_source='new_fold_H1_frozen' if arm=='C' else 'joint_fixed_last_state',classifier_equals_fresh_H1=arm=='C',**{key:True for key in flags}) for arm in ('C','J5')}
  common=dict(shared_budget_sha256=ctx['appended_stage_budget']['sha256'],operator_source_manifest_sha256=op['sha256'],original_budget_sha256=ctx['original_fresh_budget']['sha256'],resume_checkpoint_sha256=None)
  audit=dict(schema='aictrack1.fresh_route.continuation.cpu_checkpoint_audit.v1',complete=True,source_manifest_sha256=self.source,original_diagnostic_source_manifest_sha256=a.UPSTREAM_DIAGNOSTIC_SHA,selected_route='J5_GRAPH',test_access=False,cuda_used=False,model_forwards=0,optimizer_steps=0,**states,**common)
  pre=self.put(self.root/'pre.json',audit);terminal=self.put(self.root/'terminal.json',audit)
  done=dict(schema='aictrack1.fresh_route.continuation.training.complete.v1',complete=True,status='no_resume_needed_both_fixed_last20',source_manifest_sha256=self.source,completed_epoch=20,completed_last_sha256=ctx['checkpoint_origins']['J5']['sha256'],test_access=False,new_fresh_training=False,cpu_checkpoint_audit=pre,terminal_cpu_checkpoint_audit=terminal,original_phase_start_unix=a.ORIGINAL_START,original_deadline_unix=a.ORIGINAL_DEADLINE,phase_start_unix=ctx['phase_start_unix'],deadline_unix=ctx['deadline_unix'],initial_free_bytes=a.ORIGINAL_BASELINE,disk_max_bytes=a.DISK_MAX,min_free_bytes=a.MIN_FREE,started_at_unix=a.APPEND_START+1,completed_at_unix=a.APPEND_START+2,actual_runtime_receipts=[],owner=None,**common)
  items=dict(continuation_operator_source=op,partial_restore_audit=pre,terminal_restore_audit=terminal,continuation_completion=self.put(self.root/'done.json',done))
  self.assertEqual(set(w.verify_continuation_chain(items,ctx,'d'*64)),set(w.CHAIN_NAMES))
  done['started_at_unix']=a.APPEND_START-1;items['continuation_completion']=self.put(self.root/'badtime.json',done)
  with self.assertRaises(ValueError):w.verify_continuation_chain(items,ctx,'d'*64)

class Pair(unittest.TestCase):
 def pair(self):
  shared=dict(split_sha256=c.SPLIT_SHA,manifest_sha256=c.MANIFEST_SHA,class_mapping_sha256=c.MAPPING_SHA,h1_sha256=c.H1_SHA,view_protocol=c.VIEW_PROTOCOL,training_source_manifest_sha256='a'*64,diagnostic_source_manifest_sha256='b'*64,admission_receipt_sha256='c'*64,source_h1_manifest_sha256=c.H1_SOURCE_SHA,selected_route='J5_GRAPH',runtime=dict(torch_num_threads=4))
  def cache(arm):return dict(identity=dict(shared,model_arm=arm,classifier_head_source='new_fold_H1_frozen' if arm=='C' else 'joint_fixed_last_state',trainable_tensors=152 if arm=='C' else 154),ids=['fixture'],groups=['fixture'],targets=np.array([0]),train_counts=np.array([1]))
  return cache('J5'),cache('C')
 def test_same_source_pair_passes(self):self.assertTrue(c.validate_pair(*self.pair()))
 def test_different_diagnostic_sources_fail(self):
  j,control=self.pair();j['identity']['diagnostic_source_manifest_sha256']='d'*64
  with self.assertRaises(ValueError):c.validate_pair(j,control)
 def test_different_training_sources_fail(self):
  j,control=self.pair();j['identity']['training_source_manifest_sha256']='d'*64
  with self.assertRaises(ValueError):c.validate_pair(j,control)
 def test_different_thread_runtime_fails(self):
  j,control=self.pair();j['identity']['runtime']=dict(torch_num_threads=64)
  with self.assertRaises(ValueError):c.validate_pair(j,control)

if __name__=='__main__':unittest.main()
