"""Typed S/P7 or J5/graph/P7 fresh confirmation; actual immutable evidence only."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from fresh_diag_contract import ROOT,SPLIT_SHA,H1_SHA,H1_SOURCE_SHA,H1_RECEIPT_SHA,TRAINING_TEMPLATE_SHA,ROUTES,VISUAL_KEYS,VISUAL_NUMEL,JOINT_KEYS,JOINT_NUMEL,sha,save_json,valid_digest,verify_owned_sources
from confirm_fixed_graph_vs_C import GRAPH_SOURCE_SHA

def require(value,message):
 if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def evidence(path):
 p=Path(path).resolve();return dict(path=str(p),sha256=sha(p))
def alias_sha(report,side):
 value=report[side];direct=value.get('identity_sha256');nested=value.get('provenance',{}).get('identity_sha256')
 require(valid_digest(direct) and direct==nested,'paired identity aliases differ');return direct

def direction(report,route):
 expected='aictrack1.fresh_route.raw_paired.v1' if route=='S' else 'aictrack1.fresh_route.graph_combo_paired.v1'
 require(report.get('schema')==expected and report.get('complete') is True,'typed complete paired report')
 require(report.get('selected_route')==route and report.get('selected_arm')==ROUTES[route] and report.get('p1_arm')==ROUTES[route],'selected recipe aliases')
 require(report.get('test_access') is False and report.get('used_test_or_online_scores') is False,'train-derived observed labels only')
 require(report.get('split_sha256')==SPLIT_SHA and report.get('h1_sha256')==H1_SHA and report.get('source_h1_manifest_sha256')==H1_SOURCE_SHA,'actual new fold/H1 only')
 require(report.get('selection_protocol_sha256')==sha(ROOT/'selection_protocol.json') and report.get('frozen_postprocess_sha256')==sha(ROOT/'frozen_postprocess.json'),'pre-frozen query/P7 policy')
 require(report.get('environment',{}).get('cpu_threads')==2,'paired raw/graph/C use identical registered CPU2 decoder')
 protocols=report.get('protocols',[]);require({p.get('name') for p in protocols}=={'natural_unique_images','balanced_supported_unique'},'both frozen query protocols required')
 primary=next(p for p in protocols if p['name']=='balanced_supported_unique')['p7']['candidate_vs_control']
 natural=next(p for p in protocols if p['name']=='natural_unique_images')['p7']['candidate_vs_control']
 delta=primary['delta_vs_baseline_pp'];ndelta=natural['delta_vs_baseline_pp']
 require(isinstance(delta,(int,float)) and math.isfinite(delta) and delta>0,'fresh selected primary direction must be positive')
 require(isinstance(ndelta,(int,float)) and math.isfinite(ndelta),'natural diagnostic must be finite, never a positivity hard gate')
 require(report.get('eligibility',{}).get('primary_delta_pp')==delta and report['eligibility'].get('natural_p7_delta_pp')==ndelta,'eligibility must derive from actual primary/natural protocol')
 if route=='J5_GRAPH':
  require(report.get('graph_source_sha256')==GRAPH_SOURCE_SHA and report.get('fixed_graph_protocol_sha256')==sha(ROOT/'fixed_graph_protocol.json'),'one preselected graph point source')
  require(report.get('fixed_graph')==dict(k=10,lambda_value=.3,iterations=3,probability_temperature=1.,margin_protection=math.log(4.)),'registered graph numerical policy')
  require(report.get('primary_scope')=='J5_graph_P7_vs_C' and report.get('additional_model_forwards')==0 and report.get('model_updates')==0,'combo direct fresh C, no model changes')
  for name in ('raw_j5_vs_C','graph_increment_vs_J5'):
   require({p.get('name') for p in report.get(name,[])}=={'natural_unique_images','balanced_supported_unique'},'raw J5 and incremental graph must be separately disclosed')
 return dict(primary_delta_vs_C_pp=delta,natural_delta_vs_C_pp=ndelta,group_ci95_pp=primary['group_bootstrap']['ci95_pp'],primary_scope='S_P7_vs_C' if route=='S' else 'J5_graph_P7_vs_C')

def build_confirmation(report,items,*,route,report_sha,source_sha):
 require(route in ROUTES,'typed preselected route')
 result=direction(report,route)
 required={'checkpoint_C','config_C','identity_C','checkpoint_audit_C','runtime_C','checkpoint_candidate','config_candidate','identity_candidate','checkpoint_audit_candidate','runtime_candidate','paired_report','training_source','diagnostic_source','selection_protocol','frozen_postprocess','split','h1','h1_receipt','h1_source','admission_receipt','root_review'}
 if route=='J5_GRAPH':required|={'features','fixed_graph_protocol','graph_source','graph_budget'}
 require(required<=set(items),'complete real evidence inventory required')
 for name,item in items.items():
  require(isinstance(item.get('path'),str) and Path(item['path']).is_absolute() and valid_digest(item.get('sha256')) and sha(item['path'])==item['sha256'],'actual evidence drift '+name)
 require(items['paired_report']['sha256']==report_sha and read(items['paired_report']['path'])==report,'paired report exact SHA/content')
 bindings={'training_source':'training_source_manifest_sha256','diagnostic_source':'diagnostic_source_manifest_sha256','selection_protocol':'selection_protocol_sha256','frozen_postprocess':'frozen_postprocess_sha256','split':'split_sha256','h1':'h1_sha256','admission_receipt':'admission_receipt_sha256'}
 for name,key in bindings.items():require(items[name]['sha256']==report.get(key),'report evidence source binding '+name)
 require(items['diagnostic_source']['sha256']==source_sha and items['h1_receipt']['sha256']==H1_RECEIPT_SHA and items['h1_source']['sha256']==H1_SOURCE_SHA,'current diagnostic source / actual new H1 receipt/source')
 source=read(items['training_source']['path']);require(source.get('schema')=='aictrack1.fresh_route.training.source.v1' and source.get('status')=='bound' and source.get('selected_route')==route and source.get('selected_arm')==ROUTES[route] and source.get('template_manifest_sha256')==TRAINING_TEMPLATE_SHA,'bound reviewed05df C plus one same recipe')
 require(source.get('admission_receipt_sha256')==report['admission_receipt_sha256'] and source.get('source_h1_manifest_sha256')==H1_SOURCE_SHA,'positive admission/new H1 producer source')
 h1=read(items['h1_receipt']['path']);require(h1.get('complete') is True and h1.get('checkpoint_sha256')==H1_SHA and h1.get('source_sha256')==H1_SOURCE_SHA and h1.get('split_sha256')==SPLIT_SHA and h1.get('validation_selection') is False and h1.get('test_access') is False,'train-only fresh H1 actual producer receipt')
 identities=[]
 for suffix,side,arm in [('C','control','C'),('candidate','candidate',ROUTES[route])]:
  identity=read(items['identity_'+suffix]['path']);audit=read(items['checkpoint_audit_'+suffix]['path'])
  require(items['identity_'+suffix]['sha256']==alias_sha(report,side),'paired actual identity '+side)
  require(identity.get('model_arm')==arm and identity.get('selected_route')==route and identity.get('test_access') is False and identity.get('holdout_verified') is True and identity.get('checkpoint_role')=='last' and identity.get('checkpoint_epoch')==20,'same fresh final20 identity '+side)
  counts=(VISUAL_KEYS,VISUAL_NUMEL) if arm in ('C','S') else (JOINT_KEYS,JOINT_NUMEL)
  head='new_fold_H1_frozen' if arm in ('C','S') else 'joint_fixed_last_state'
  require(identity.get('schema')=='aictrack1.p0_frozen_fold_input.v1' and identity.get('stage')==2 and identity.get('training_scope')=='fixed_split_validation' and identity.get('training_count')==118958 and identity.get('validation_count')==29737 and identity.get('trainable_tensors')==counts[0] and identity.get('trainable_numel')==counts[1] and identity.get('classifier_head_source')==head,'complete holdout identity count/head policy '+side)
  for key,wanted in {'schema':'aictrack1.fresh_route.fixed_last_diagnostic_audit.v1','complete':True,'status':'complete_fixed_last20_audit','checkpoint_epoch':20,'checkpoint_role':'last','epoch':20,'model_arm':arm,'trainable_tensors':counts[0],'trainable_numel':counts[1],'numel':counts[1],'classifier_head_source':head,'classifier_parameters_restored_equal':True,'exact_ac_train_id_label_binding':True,'model_frozen_eval':True,'inference_optimizer_steps':0,'test_access':False}.items():require(audit.get(key)==wanted,'actual restore audit '+side+' '+key)
  require(identity.get('source_receipt_sha256')==items['checkpoint_audit_'+suffix]['sha256'] and report[side].get('checkpoint_sha256')==items['checkpoint_'+suffix]['sha256'],'checkpoint/audit/paired exact bindings')
  for name,key in [('checkpoint_'+suffix,'checkpoint_sha256'),('config_'+suffix,'experiment_config_sha256'),('training_source','training_source_manifest_sha256'),('diagnostic_source','diagnostic_source_manifest_sha256'),('split','split_sha256'),('h1','h1_sha256'),('admission_receipt','admission_receipt_sha256')]:require(identity.get(key)==items[name]['sha256'],'identity evidence '+side+' '+key)
  for key in ('checkpoint_sha256','training_source_manifest_sha256','diagnostic_source_manifest_sha256','split_sha256','h1_sha256','admission_receipt_sha256','selected_route'):require(audit.get(key)==identity.get(key),'audit identity aliases '+key)
  require(audit.get('config_sha256')==identity['experiment_config_sha256'] and audit.get('fixed_last_sha256')==identity['checkpoint_sha256'],'fixed-last config/checkpoint aliases')
  require(identity['experiment_config_sha256'] in source['files'].values(),'config owned by actual source')
  require(items['runtime_'+suffix]['sha256']==identity.get('runtime_receipt_sha256')==audit.get('runtime_receipt_sha256'),'actual runtime receipt SHA '+side)
  runtime=read(items['runtime_'+suffix]['path'])
  for key,wanted in {'schema':'aictrack1.fresh_route.diagnostic_runtime.v1','arm':arm,'source_manifest_sha256':items['training_source']['sha256'],'config_sha256':items['config_'+suffix]['sha256'],'split_sha256':SPLIT_SHA,'h1_sha256':H1_SHA,'selected_route':route,'test_access':False,'runtime':identity.get('runtime')}.items():require(runtime.get(key)==wanted,'actual runtime receipt '+side+' '+key)
  require(bool(runtime.get('upstream_receipts')),'actual immutable training attempts must be embedded')
  for attempt in runtime['upstream_receipts']:
   actual=attempt.get('receipt',{})
   require(valid_digest(attempt.get('sha256')) and actual.get('schema')=='aictrack1.fresh_route.runtime.v1' and actual.get('arm')==arm and actual.get('selected_route')==route and actual.get('source_manifest_sha256')==items['training_source']['sha256'] and actual.get('config_sha256')==items['config_'+suffix]['sha256'] and actual.get('test_access') is False,'embedded actual runtime source/config/arm')
   require(actual.get('environment')=={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'} and actual.get('actual_torch_threads')==4 and actual.get('actual_torch_interop_threads')==identity.get('runtime',{}).get('torch_num_interop_threads'),'embedded actual four-thread runtime')
  identities.append(identity)
 require(identities[0].get('runtime')==identities[1].get('runtime') and identities[0].get('runtime',{}).get('torch_num_threads')==4,'actual matched C/candidate training runtime4')
 require(identities[0].get('sample_ids_order_sha256')==identities[1].get('sample_ids_order_sha256'),'same fresh query row order')
 review=read(items['root_review']['path']);require(review.get('schema')=='aictrack1.fresh_route.root_review.v1' and review.get('selected_route')==route and review.get('decision')=='accept_fresh_route_confirmation' and review.get('paired_report_sha256')==report_sha and isinstance(review.get('reason'),str) and review['reason'].strip(),'actual root risk review of this exact report')
 if route=='J5_GRAPH':
  require(items['features']['sha256']==report.get('features',{}).get('sha256')==identities[1].get('feature_file_sha256') and items['fixed_graph_protocol']['sha256']==report.get('fixed_graph_protocol_sha256') and items['graph_source']['sha256']==GRAPH_SOURCE_SHA,'graph feature/protocol/source evidence')
  ledger=read(items['graph_budget']['path']);require(ledger.get('source_manifest_sha256')==GRAPH_SOURCE_SHA and ledger.get('total_wall_seconds')==7200. and 64.3705<=ledger.get('spent_wall_seconds',-1)<=7200.,'shared original graph budget never reset/overspent')
  require(report.get('graph_budget',{}).get('after_snapshot_sha256')==items['graph_budget']['sha256'] and report['graph_budget'].get('spent_after_seconds')==ledger['spent_wall_seconds'] and report['graph_budget'].get('shared_ledger_charged') is True,'actual charged graph snapshot must bind this report')
 value=dict(schema='aictrack1.fresh_route.confirmation.v1',status='complete_positive_fresh_route_confirmation',selected_route=route,selected_arm=ROUTES[route],checkpoint_epoch=20,checkpoint_role='last',selection_rule='fixed_final20',training_scope='fixed_split_validation',split_sha256=SPLIT_SHA,h1_sha256=H1_SHA,source_h1_manifest_sha256=H1_SOURCE_SHA,training_source_manifest_sha256=items['training_source']['sha256'],diagnostic_source_manifest_sha256=source_sha,admission_receipt_sha256=report['admission_receipt_sha256'],selected_checkpoint_sha256=items['checkpoint_candidate']['sha256'],selected_config_sha256=items['config_candidate']['sha256'],selected_training_source_manifest_sha256=items['training_source']['sha256'],selection_protocol_sha256=items['selection_protocol']['sha256'],frozen_postprocess_sha256=items['frozen_postprocess']['sha256'],evidence=items,**result,test_access=False,used_test_or_online_scores=False,natural_nonnegative_hard_gate=False,ci_lower_bound_positive_hard_gate=False,stable_gain_claim=False,formal_test_frozen=False,root_risk_reviewed=True)
 if route=='J5_GRAPH':
  balanced=lambda rows:next(p for p in rows if p['name']=='balanced_supported_unique')['p7']['candidate_vs_control']['delta_vs_baseline_pp']
  value.update(graph_parameters={'k':10,'lambda':.3,'iterations':3,'probability_temperature':1.,'protection_raw_margin':math.log(4.)},raw_J5_P7_delta_vs_C_pp=balanced(report['raw_j5_vs_C']),graph_increment_P7_delta_vs_J5_pp=balanced(report['graph_increment_vs_J5']),fixed_graph_protocol_sha256=items['fixed_graph_protocol']['sha256'],graph_source_manifest_sha256=GRAPH_SOURCE_SHA)
 return value

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paired-report',required=True,type=Path);p.add_argument('--evidence-json',required=True,type=Path);p.add_argument('--selected-route',choices=tuple(ROUTES),required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 source_sha=verify_owned_sources();value=build_confirmation(read(a.paired_report),read(a.evidence_json),route=a.selected_route,report_sha=sha(a.paired_report),source_sha=source_sha)
 a.output.parent.mkdir(parents=True,exist_ok=True);save_json(a.output,value);print(json.dumps(dict(complete=True,receipt=str(a.output),sha256=sha(a.output),primary_delta_vs_C_pp=value['primary_delta_vs_C_pp']),ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
