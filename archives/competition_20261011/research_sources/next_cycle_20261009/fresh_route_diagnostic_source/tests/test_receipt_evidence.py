"""Real fixture bytes/hash joins: full typed receipt, then wrong arm/runtime/SHA negatives."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import write_fresh_route_confirmation as writer
from fresh_diag_contract import sha

class ReceiptEvidenceTests(unittest.TestCase):
 def test_full_typed_S_receipt_real_files_hashes_and_actual_runtime_corruptions(self):
  folder=ROOT/'artifacts/cpu_receipt_fixture';folder.mkdir(parents=True,exist_ok=True)
  paths={}
  def put(name,value):
   path=folder/(name+'.json');path.write_text(json.dumps(value,indent=2)+'\n');paths[name]=path;return sha(path)
  # The test patches ONLY competition identity constants to fixture byte digests;
  # real evidence hashing, schema joins and the public writer all execute unchanged.
  h1=put('h1',{'fixture':'synthetic H1, no model'});h1source=put('h1_source',{'fixture':'synthetic H1 producer source'})
  split=put('split',{'fixture':'synthetic new split metadata only'})
  hreceipt=put('h1_receipt',dict(complete=True,checkpoint_sha256=h1,source_sha256=h1source,split_sha256=split,validation_selection=False,test_access=False))
  admission=put('admission_receipt',{'fixture':'synthetic typed admission bytes'})
  configs={x:put('config_'+x,dict(fixture='synthetic config',arm=x)) for x in ('C','candidate')}
  source=put('training_source',dict(schema='aictrack1.fresh_route.training.source.v1',status='bound',selected_route='S',selected_arm='S',admission_receipt_sha256=admission,source_h1_manifest_sha256=h1source,template_manifest_sha256=writer.TRAINING_TEMPLATE_SHA,files={str(x):v for x,v in configs.items()}))
  diagnostic=put('diagnostic_source',dict(fixture='synthetic current diagnostic source'))
  runtime=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',torch_num_threads=4,torch_num_interop_threads=16)
  identities={};provenance={}
  for suffix,arm in [('C','C'),('candidate','S')]:
   checkpoint=put('checkpoint_'+suffix,dict(fixture='synthetic checkpoint bytes',arm=arm))
   raw=dict(schema='aictrack1.fresh_route.runtime.v1',arm=arm,selected_route='S',source_manifest_sha256=source,config_sha256=configs[suffix],test_access=False,environment={k:runtime[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')},actual_torch_threads=4,actual_torch_interop_threads=16)
   canonical=dict(schema='aictrack1.fresh_route.diagnostic_runtime.v1',arm=arm,source_manifest_sha256=source,config_sha256=configs[suffix],split_sha256=split,h1_sha256=h1,selected_route='S',test_access=False,runtime=runtime,upstream_receipts=[dict(path='synthetic_attempt_not_remote',sha256='a'*64,receipt=raw)])
   runtime_sha=put('runtime_'+suffix,canonical)
   audit=dict(schema='aictrack1.fresh_route.fixed_last_diagnostic_audit.v1',complete=True,status='complete_fixed_last20_audit',checkpoint_epoch=20,checkpoint_role='last',epoch=20,model_arm=arm,trainable_tensors=152,trainable_numel=87849216,numel=87849216,classifier_head_source='new_fold_H1_frozen',classifier_parameters_restored_equal=True,exact_ac_train_id_label_binding=True,model_frozen_eval=True,inference_optimizer_steps=0,test_access=False,config_sha256=configs[suffix],checkpoint_sha256=checkpoint,fixed_last_sha256=checkpoint,training_source_manifest_sha256=source,diagnostic_source_manifest_sha256=diagnostic,split_sha256=split,h1_sha256=h1,admission_receipt_sha256=admission,selected_route='S',runtime_receipt_sha256=runtime_sha)
   audit_sha=put('checkpoint_audit_'+suffix,audit)
   identity=dict(schema='aictrack1.p0_frozen_fold_input.v1',stage=2,training_scope='fixed_split_validation',training_count=118958,validation_count=29737,trainable_tensors=152 if arm!='J5' else 154,trainable_numel=87849216 if arm!='J5' else 88233966,classifier_head_source='new_fold_H1_frozen' if arm!='J5' else 'joint_fixed_last_state',model_arm=arm,selected_route='S',test_access=False,holdout_verified=True,checkpoint_role='last',checkpoint_epoch=20,source_receipt_sha256=audit_sha,checkpoint_sha256=checkpoint,experiment_config_sha256=configs[suffix],training_source_manifest_sha256=source,diagnostic_source_manifest_sha256=diagnostic,split_sha256=split,h1_sha256=h1,admission_receipt_sha256=admission,runtime=runtime,runtime_receipt_sha256=runtime_sha,sample_ids_order_sha256='e'*64)
   identity_sha=put('identity_'+suffix,identity);identities[suffix]=identity
   provenance[suffix]=dict(identity_sha256=identity_sha,checkpoint_sha256=checkpoint)
  metric=lambda delta:dict(delta_vs_baseline_pp=delta,group_bootstrap=dict(ci95_pp=[-.3,.4]))
  report=dict(schema='aictrack1.fresh_route.raw_paired.v1',complete=True,selected_route='S',selected_arm='S',p1_arm='S',test_access=False,used_test_or_online_scores=False,split_sha256=split,h1_sha256=h1,source_h1_manifest_sha256=h1source,selection_protocol_sha256=sha(ROOT/'selection_protocol.json'),frozen_postprocess_sha256=sha(ROOT/'frozen_postprocess.json'),environment=dict(cpu_threads=2),protocols=[dict(name='natural_unique_images',p7=dict(candidate_vs_control=metric(-.2))),dict(name='balanced_supported_unique',p7=dict(candidate_vs_control=metric(.1)))],eligibility=dict(primary_delta_pp=.1,natural_p7_delta_pp=-.2),training_source_manifest_sha256=source,diagnostic_source_manifest_sha256=diagnostic,admission_receipt_sha256=admission,candidate=dict(provenance['candidate'],provenance=provenance['candidate']),control=dict(provenance['C'],provenance=provenance['C']))
  report_sha=put('paired_report',report)
  put('root_review',dict(schema='aictrack1.fresh_route.root_review.v1',selected_route='S',decision='accept_fresh_route_confirmation',paired_report_sha256=report_sha,reason='synthetic fixture permits negative natural and CI crossing zero'))
  items={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in paths.items()}
  items['selection_protocol']=dict(path=str((ROOT/'selection_protocol.json').resolve()),sha256=sha(ROOT/'selection_protocol.json'))
  items['frozen_postprocess']=dict(path=str((ROOT/'frozen_postprocess.json').resolve()),sha256=sha(ROOT/'frozen_postprocess.json'))
  with patch.multiple(writer,H1_SHA=h1,H1_SOURCE_SHA=h1source,H1_RECEIPT_SHA=hreceipt,SPLIT_SHA=split):
   result=writer.build_confirmation(report,items,route='S',report_sha=report_sha,source_sha=diagnostic)
   self.assertEqual(result['status'],'complete_positive_fresh_route_confirmation');self.assertEqual(result['natural_delta_vs_C_pp'],-.2);self.assertFalse(result['stable_gain_claim'])
   broken=copy.deepcopy(items);broken['identity_candidate']['sha256']='0'*64
   with self.assertRaises(ValueError):writer.build_confirmation(report,broken,route='S',report_sha=report_sha,source_sha=diagnostic)
   # Re-signing a corrupted real runtime file still must fail semantic matching.
   corrupted=json.loads(paths['runtime_candidate'].read_text());corrupted['upstream_receipts'][0]['receipt']['actual_torch_threads']=64
   put('runtime_candidate',corrupted);broken=copy.deepcopy(items);broken['runtime_candidate']['sha256']=sha(paths['runtime_candidate'])
   with self.assertRaises(ValueError):writer.build_confirmation(report,broken,route='S',report_sha=report_sha,source_sha=diagnostic)
   paths['runtime_candidate'].write_text(json.dumps(canonical,indent=2)+'\n')
 def test_full_typed_J5_GRAPH_receipt_cpu2_joint154_directC_vs_negative_rawJ(self):
  folder=ROOT/'artifacts/cpu_receipt_fixture_graph';folder.mkdir(parents=True,exist_ok=True)
  paths={}
  def put(name,value):
   path=folder/(name+'.json');path.write_text(json.dumps(value,indent=2)+'\n');paths[name]=path;return sha(path)
  # The test patches ONLY competition identity constants to fixture byte digests;
  # real evidence hashing, schema joins and the public writer all execute unchanged.
  h1=put('h1',{'fixture':'synthetic H1, no model'});h1source=put('h1_source',{'fixture':'synthetic H1 producer source'})
  split=put('split',{'fixture':'synthetic new split metadata only'})
  hreceipt=put('h1_receipt',dict(complete=True,checkpoint_sha256=h1,source_sha256=h1source,split_sha256=split,validation_selection=False,test_access=False))
  admission=put('admission_receipt',{'fixture':'synthetic typed admission bytes'})
  configs={x:put('config_'+x,dict(fixture='synthetic config',arm=x)) for x in ('C','candidate')}
  source=put('training_source',dict(schema='aictrack1.fresh_route.training.source.v1',status='bound',selected_route='J5_GRAPH',selected_arm='J5',admission_receipt_sha256=admission,source_h1_manifest_sha256=h1source,template_manifest_sha256=writer.TRAINING_TEMPLATE_SHA,files={str(x):v for x,v in configs.items()}))
  diagnostic=put('diagnostic_source',dict(fixture='synthetic current diagnostic source'))
  runtime=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',torch_num_threads=4,torch_num_interop_threads=16)
  identities={};provenance={}
  for suffix,arm in [('C','C'),('candidate','J5')]:
   checkpoint=put('checkpoint_'+suffix,dict(fixture='synthetic checkpoint bytes',arm=arm))
   raw=dict(schema='aictrack1.fresh_route.runtime.v1',arm=arm,selected_route='J5_GRAPH',source_manifest_sha256=source,config_sha256=configs[suffix],test_access=False,environment={k:runtime[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')},actual_torch_threads=4,actual_torch_interop_threads=16)
   canonical=dict(schema='aictrack1.fresh_route.diagnostic_runtime.v1',arm=arm,source_manifest_sha256=source,config_sha256=configs[suffix],split_sha256=split,h1_sha256=h1,selected_route='J5_GRAPH',test_access=False,runtime=runtime,upstream_receipts=[dict(path='synthetic_attempt_not_remote',sha256='a'*64,receipt=raw)])
   runtime_sha=put('runtime_'+suffix,canonical)
   audit=dict(schema='aictrack1.fresh_route.fixed_last_diagnostic_audit.v1',complete=True,status='complete_fixed_last20_audit',checkpoint_epoch=20,checkpoint_role='last',epoch=20,model_arm=arm,trainable_tensors=152 if arm=='C' else 154,trainable_numel=87849216 if arm=='C' else 88233966,numel=87849216 if arm=='C' else 88233966,classifier_head_source='new_fold_H1_frozen' if arm=='C' else 'joint_fixed_last_state',classifier_parameters_restored_equal=True,exact_ac_train_id_label_binding=True,model_frozen_eval=True,inference_optimizer_steps=0,test_access=False,config_sha256=configs[suffix],checkpoint_sha256=checkpoint,fixed_last_sha256=checkpoint,training_source_manifest_sha256=source,diagnostic_source_manifest_sha256=diagnostic,split_sha256=split,h1_sha256=h1,admission_receipt_sha256=admission,selected_route='J5_GRAPH',runtime_receipt_sha256=runtime_sha)
   audit_sha=put('checkpoint_audit_'+suffix,audit)
   identity=dict(schema='aictrack1.p0_frozen_fold_input.v1',stage=2,training_scope='fixed_split_validation',training_count=118958,validation_count=29737,trainable_tensors=152 if arm!='J5' else 154,trainable_numel=87849216 if arm!='J5' else 88233966,classifier_head_source='new_fold_H1_frozen' if arm!='J5' else 'joint_fixed_last_state',model_arm=arm,selected_route='J5_GRAPH',test_access=False,holdout_verified=True,checkpoint_role='last',checkpoint_epoch=20,source_receipt_sha256=audit_sha,checkpoint_sha256=checkpoint,experiment_config_sha256=configs[suffix],training_source_manifest_sha256=source,diagnostic_source_manifest_sha256=diagnostic,split_sha256=split,h1_sha256=h1,admission_receipt_sha256=admission,runtime=runtime,runtime_receipt_sha256=runtime_sha,sample_ids_order_sha256='e'*64)
   feature_sha=put('features',{'fixture':'same checkpoint existing views'})
   if arm=='J5':identity['feature_file_sha256']=feature_sha
   identity_sha=put('identity_'+suffix,identity);identities[suffix]=identity
   provenance[suffix]=dict(identity_sha256=identity_sha,checkpoint_sha256=checkpoint)
  metric=lambda delta:dict(delta_vs_baseline_pp=delta,group_bootstrap=dict(ci95_pp=[-.3,.4]))
  report=dict(schema='aictrack1.fresh_route.graph_combo_paired.v1',complete=True,selected_route='J5_GRAPH',selected_arm='J5',p1_arm='J5',test_access=False,used_test_or_online_scores=False,split_sha256=split,h1_sha256=h1,source_h1_manifest_sha256=h1source,selection_protocol_sha256=sha(ROOT/'selection_protocol.json'),frozen_postprocess_sha256=sha(ROOT/'frozen_postprocess.json'),environment=dict(cpu_threads=2),protocols=[dict(name='natural_unique_images',p7=dict(candidate_vs_control=metric(-.2))),dict(name='balanced_supported_unique',p7=dict(candidate_vs_control=metric(.1)))],eligibility=dict(primary_delta_pp=.1,natural_p7_delta_pp=-.2),training_source_manifest_sha256=source,diagnostic_source_manifest_sha256=diagnostic,admission_receipt_sha256=admission,candidate=dict(provenance['candidate'],provenance=provenance['candidate']),control=dict(provenance['C'],provenance=provenance['C']))
  graph_source=ROOT.parent/'graph_source/source_manifest.json'
  paths['graph_source']=graph_source;paths['fixed_graph_protocol']=ROOT/'fixed_graph_protocol.json'
  budget=put('graph_budget',dict(source_manifest_sha256=writer.GRAPH_SOURCE_SHA,total_wall_seconds=7200.,spent_wall_seconds=80.))
  row=lambda name,delta:dict(name=name,p7=dict(candidate_vs_control=metric(delta)))
  report.update(graph_budget=dict(after_snapshot_sha256=budget,spent_after_seconds=80.,shared_ledger_charged=True),graph_source_sha256=writer.GRAPH_SOURCE_SHA,fixed_graph_protocol_sha256=sha(ROOT/'fixed_graph_protocol.json'),fixed_graph=dict(k=10,lambda_value=.3,iterations=3,probability_temperature=1.,margin_protection=__import__('math').log(4.)),primary_scope='J5_graph_P7_vs_C',additional_model_forwards=0,model_updates=0,features={'sha256':sha(paths['features'])},raw_j5_vs_C=[row('natural_unique_images',-.1),row('balanced_supported_unique',-.2)],graph_increment_vs_J5=[row('natural_unique_images',-.1),row('balanced_supported_unique',.3)])
  report_sha=put('paired_report',report)
  put('root_review',dict(schema='aictrack1.fresh_route.root_review.v1',selected_route='J5_GRAPH',decision='accept_fresh_route_confirmation',paired_report_sha256=report_sha,reason='synthetic fixture permits negative natural and CI crossing zero'))
  items={name:dict(path=str(path.resolve()),sha256=sha(path)) for name,path in paths.items()}
  items['selection_protocol']=dict(path=str((ROOT/'selection_protocol.json').resolve()),sha256=sha(ROOT/'selection_protocol.json'))
  items['frozen_postprocess']=dict(path=str((ROOT/'frozen_postprocess.json').resolve()),sha256=sha(ROOT/'frozen_postprocess.json'))
  with patch.multiple(writer,H1_SHA=h1,H1_SOURCE_SHA=h1source,H1_RECEIPT_SHA=hreceipt,SPLIT_SHA=split):
   result=writer.build_confirmation(report,items,route='J5_GRAPH',report_sha=report_sha,source_sha=diagnostic)
   self.assertEqual(result['status'],'complete_positive_fresh_route_confirmation');self.assertEqual(result['natural_delta_vs_C_pp'],-.2);self.assertFalse(result['stable_gain_claim'])
   broken=copy.deepcopy(items);broken['identity_candidate']['sha256']='0'*64
   with self.assertRaises(ValueError):writer.build_confirmation(report,broken,route='J5_GRAPH',report_sha=report_sha,source_sha=diagnostic)
   # Re-signing a corrupted real runtime file still must fail semantic matching.
   corrupted=json.loads(paths['runtime_candidate'].read_text());corrupted['upstream_receipts'][0]['receipt']['actual_torch_threads']=64
   put('runtime_candidate',corrupted);broken=copy.deepcopy(items);broken['runtime_candidate']['sha256']=sha(paths['runtime_candidate'])
   with self.assertRaises(ValueError):writer.build_confirmation(report,broken,route='J5_GRAPH',report_sha=report_sha,source_sha=diagnostic)
   paths['runtime_candidate'].write_text(json.dumps(canonical,indent=2)+'\n')

if __name__=='__main__':unittest.main(verbosity=2)
