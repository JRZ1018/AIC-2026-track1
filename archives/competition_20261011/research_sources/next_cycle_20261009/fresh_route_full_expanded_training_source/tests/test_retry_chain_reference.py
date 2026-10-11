"""Reviewed retry metadata fixture replay against FULL function snapshots."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import append_diagnostic_contract as a
import retry_recovery_contract as r
import append_confirmation_reference as w
import full_route_contract as full
c=SimpleNamespace(SPLIT_SHA=full.FRESH_SPLIT_SHA,H1_SHA=full.FRESH_H1_SHA)
class RetryChainTests(unittest.TestCase):
 def setUp(self):
   self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.train=self.root/'training';self.train.mkdir()
   for module,key,value in ((a,'TRAINING_ROOT',self.train),(a,'RETRY_CONT_ROOT',self.root/'operator'),(r,'OLD_CONT',self.root/'old_cont')):
    p=patch.object(module,key,value);p.start();self.addCleanup(p.stop)
   cfg={arm:self.put(self.train/'configs/experiment'/('p1_'+arm+'.yaml'),{'fixture':arm}) for arm in ('C','J5')}
   src=self.put(self.train/'source_manifest.json',dict(schema='aictrack1.fresh_route.training.source.v1',status='bound',selected_route='J5_GRAPH',selected_arm='J5',original_phase_start_unix=a.ORIGINAL_START,original_deadline_unix=a.ORIGINAL_DEADLINE,original_data_free_baseline_bytes=a.ORIGINAL_BASELINE,disk_max_bytes=a.ORIGINAL_DISK_MAX,min_free_bytes=a.MIN_FREE,files={'configs/experiment/p1_'+arm+'.yaml':item['sha256'] for arm,item in cfg.items()}))
   self.source=src['sha256'];old=self.put(self.train/'artifacts/queues/fresh_route_budget.json',dict(schema='aictrack1.fresh_route.budget.v1',phase_start_unix=a.ORIGINAL_START,deadline_unix=a.ORIGINAL_DEADLINE,initial_free_bytes=a.ORIGINAL_BASELINE,disk_max_bytes=a.ORIGINAL_DISK_MAX,min_free_bytes=a.MIN_FREE))
   origins={arm:dict(path=str((self.train/'artifacts/experiments'/('FRESH_'+arm)/'last.pt').resolve()),epoch=20 if arm=='C' else 19,sha256=('c' if arm=='C' else 'd')*64) for arm in ('C','J5')}
   births=[dict(role=role,pid=pid,pgrp=pid,session=pid,start_ticks=ticks) for role,pid,ticks in [('C_controller',236091,'308497119'),('C_trainer',236098,'308497585'),('J5_controller',239865,'309227901'),('J5_trainer',239872,'309228342'),('failed_append_controller',243711,'309965381'),('failed_append_trainer',244056,'310043026')]]
   base=dict(status='root_authorized_append_only_budget',root_authorized=True,selected_route='J5_GRAPH',selected_arm='J5',test_access=False,new_fresh_training=False,scope=a.SCOPES,training_root=str(self.train.resolve()),source_manifest_sha256=self.source,original_phase_start_unix=a.ORIGINAL_START,original_deadline_unix=a.ORIGINAL_DEADLINE,initial_free_bytes=a.ORIGINAL_BASELINE,min_free_bytes=a.MIN_FREE,phase_start_unix=a.APPEND_START,deadline_unix=a.APPEND_MAX_DEADLINE,original_budget=old,training_source=src,configs=cfg,checkpoints=origins)
   prior=dict(base,schema='aictrack1.fresh_route.continuation.training_budget.v1',disk_max_bytes=a.ORIGINAL_DISK_MAX,watched=births[:4]);shared=self.put(r.OLD_CONT/'shared_append_budget.json',prior)
   failure=self.put(r.OLD_CONT/'attempt/FAILURE.json',dict(schema='aictrack1.fresh_route.continuation.training.failure.v1',complete=False,cleanup_error=None,shared_budget_sha256=shared['sha256'],operator_source_manifest_sha256=r.OLD_CONT_SHA,source_manifest_sha256=self.source,test_access=False))
   recovery=self.put(r.OLD_CONT/'root_actual_disk_expansion_recovery_receipt.json',dict(schema='aictrack1.fresh_route.actual_disk_expansion_recovery.v1',root_authorized=True,exact_prior_groups_closed=True,canonical_naturally_idle=True,data_capacity_increase_bytes=r.CAPACITY,original_initial_free_bytes=r.INITIAL,adjusted_initial_free_bytes=r.ADJUSTED,actual_data_total_bytes=128849018880,original_data_total_bytes=96636764160,actual_data_free_bytes=38303313920,effective_old_capacity_free_bytes=6091059200,original_shared_budget=shared,prior_failure=failure,old_source_or_budget_modified=False,test_access=False,new_GPU_started=False,actual_retained_last=dict(path=origins['J5']['path'],sha256=origins['J5']['sha256'],epoch_from_previous_strict_audit=19),incomplete_temporary=dict(valid_atomic_checkpoint=False,original_path=origins['J5']['path']+'.tmp',archived_path=origins['J5']['path']+'.tmp.synthetic_failure',bytes=123,sha256='e'*64),watched=births))
   for key,value in (('OLD_SHARED_SHA',shared['sha256']),('OLD_FAILURE_SHA',failure['sha256']),('RECOVERY_SHA',recovery['sha256'])):
    p=patch.object(r,key,value);p.start();self.addCleanup(p.stop)
   self.b=dict(base,schema=a.SCHEMA,disk_max_bytes=a.DISK_MAX,watched=births,original_shared_budget=shared,prior_failure=failure,disk_expansion_recovery=recovery,capacity_increase_bytes=r.CAPACITY,adjusted_initial_free_bytes=r.ADJUSTED,retry_after_disk_expansion=True)
   self.path=self.root/'append.json';self.sign()
 def put(self,path,value):
   path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n');return dict(path=str(path.resolve()),sha256=a.sha(path))
 def sign(self):self.ref=self.put(self.path,self.b)
 def context(self,**kw):return a.validate_append_budget(self.path,self.ref['sha256'],expected_training_source_sha=self.source,**kw)
 def test_valid_shared_window_baseline_unchanged(self):
   ctx=self.context(live=True,now=a.APPEND_START+1);self.assertEqual(ctx['original_system_baseline_bytes'],a.ORIGINAL_BASELINE);self.assertFalse(ctx['training_budget_reset']);self.assertFalse(ctx['graph_ledger_reset'])
 def test_actual_file_hash_retry_partial_chain_and_wrong_time(self):
   ctx=self.context();member=self.root/'operator/op.py';member.parent.mkdir();member.write_text('# synthetic retry operator fixture\n');op=self.put(member.parent/'source_manifest.json',dict(schema='aictrack1.fresh_route.continuation.retry_training.source.v1',files={'op.py':a.sha(member)}))
   flags=('strict_cpu_model_restored','classifier_parameters_restored_equal','optimizer_cpu_restored','scheduler_cpu_restored','scaler_cpu_restored','ac_cpu_restored','exact_ac_train_id_label_binding','python_rng_restored','numpy_rng_restored','torch_cpu_rng_restored','saved_cuda_rng_structure_valid','confidence_observation_binding','saved_state_resume_bookkeeping_valid')
   def state(arm,epoch,digest):
    complete=epoch==20
    return dict(checkpoint_sha256=digest,checkpoint_epoch=epoch,config_sha256=ctx['configs'][arm]['sha256'],source_manifest_sha256=self.source,split_sha256=c.SPLIT_SHA,h1_sha256=c.H1_SHA,trainable_tensors=152 if arm=='C' else 154,trainable_numel=87849216 if arm=='C' else 88233966,status='complete_fixed_last20_audit' if complete else 'restorable_partial_not_completed',complete=complete,optimizer_parameter_groups=[152] if arm=='C' else [152,2],scheduler_epoch=epoch,scheduler_T_max=20,ac_observation_epochs=epoch,cuda_rng_restored_on_gpu=False,classifier_head_source='new_fold_H1_frozen' if arm=='C' else 'joint_fixed_last_state',classifier_equals_fresh_H1=arm=='C',**{key:True for key in flags})
   resume=ctx['checkpoint_origins']['J5']['sha256'];common=dict(shared_budget_sha256=ctx['appended_stage_budget']['sha256'],operator_source_manifest_sha256=op['sha256'],original_budget_sha256=ctx['original_fresh_budget']['sha256'],resume_checkpoint_sha256=resume,capacity_increase_bytes=r.CAPACITY,adjusted_initial_free_bytes=r.ADJUSTED,original_data_initial_free_bytes=r.INITIAL,retry_after_disk_expansion=True,**{n:ctx[n] for n in r.REF_NAMES},**{n+'_sha256':ctx[n]['sha256'] for n in r.REF_NAMES})
   audit=dict(schema='aictrack1.fresh_route.continuation.cpu_checkpoint_audit.v1',complete=True,source_manifest_sha256=self.source,original_diagnostic_source_manifest_sha256=a.UPSTREAM_DIAGNOSTIC_SHA,selected_route='J5_GRAPH',test_access=False,cuda_used=False,model_forwards=0,optimizer_steps=0,C=state('C',20,'c'*64),J5=state('J5',19,resume),**common)
   pre=self.put(self.root/'pre.json',audit);terminal=self.put(self.root/'terminal.json',dict(audit,J5=state('J5',20,'f'*64)))
   runtime=self.put(self.root/'runtime.json',dict(schema='aictrack1.fresh_route.runtime.v1',arm='J5',source_manifest_sha256=self.source,config_sha256=ctx['configs']['J5']['sha256'],environment={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'},actual_torch_threads=4,test_access=False))
   done=dict(schema='aictrack1.fresh_route.continuation.training.complete.v1',complete=True,status='complete_exact_resume_to_fixed_last20',source_manifest_sha256=self.source,completed_epoch=20,completed_last_sha256='f'*64,test_access=False,new_fresh_training=False,cpu_checkpoint_audit=pre,terminal_cpu_checkpoint_audit=terminal,original_phase_start_unix=a.ORIGINAL_START,original_deadline_unix=a.ORIGINAL_DEADLINE,phase_start_unix=ctx['phase_start_unix'],deadline_unix=ctx['deadline_unix'],initial_free_bytes=a.ORIGINAL_BASELINE,disk_max_bytes=a.DISK_MAX,min_free_bytes=a.MIN_FREE,started_at_unix=a.APPEND_START+1,completed_at_unix=a.APPEND_START+2,actual_runtime_receipts=[runtime],owner=dict(pid=1,pgrp=1,session=1,start_ticks='1'),**common)
   items=dict(continuation_operator_source=op,partial_restore_audit=pre,terminal_restore_audit=terminal,continuation_completion=self.put(self.root/'done.json',done));self.assertEqual(set(w.verify_continuation_chain(items,ctx,'f'*64)),set(w.CHAIN_NAMES))
   for key,value in [('started_at_unix',a.APPEND_START-1),('capacity_increase_bytes',0),('prior_failure_sha256','0'*64)]:
    broken=dict(done);broken[key]=value;items['continuation_completion']=self.put(self.root/'bad.json',broken)
    with self.subTest(key=key),self.assertRaises(ValueError):w.verify_continuation_chain(items,ctx,'f'*64)
