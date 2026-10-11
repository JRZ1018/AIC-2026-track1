"""Fixed graph+P7 on both new610 fresh raw students; paired combo versus combo, no grid."""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from fresh_diag_contract import ROOT,SPLIT_SHA,H1_SHA,H1_SOURCE_SHA,TRAIN_ROOT,TRAINING_SOURCE_SHA,BINDING,QUERY_SOURCE_SHA,QUERY_IDENTITY_SHA,PRIMARY_SCOPE,load_cache,validate_pair,verify_owned_sources,sha,save_json
from paired_metrics import load_queries,compare_arrays,frozen_decode
from phase_guard import phase_context,resource_check,read_json
from diagnostic_resources import memory_snapshot,memory_reason
sys.path.insert(0,str(ROOT/'reference'))
from p3_graph import blocked_topk,mutual_graph,smooth,p7,check

def validate_features(payload,cache,identity):
 if (payload.get('schema')!='aictrack1.stage2_train_derived_4v_features.v1'
     or payload.get('checkpoint_role')!='last' or payload.get('checkpoint_epoch')!=20
     or payload.get('additional_model_forwards')!=0 or payload.get('test_access') is not False
     or payload.get('observed_layout')!='four_calls_B'
     or payload.get('feature_protocol')!='L2_normalize(mean(L2_normalize(raw_visual_each_of_four_views))) in float32 on CPU'
     or payload.get('view_order')!=['native','native_exact_flip','scaled384_crop336','scaled_exact_flip']):raise ValueError('same-model four already-required views only')
 for field in ('checkpoint_sha256','experiment_config_sha256','split_sha256','manifest_sha256','class_mapping_sha256','view_protocol'):
  if payload.get(field)!=cache.get(field) or payload.get(field)!=identity.get(field):raise ValueError('feature/logit exact identity '+field)
 features=payload['features']
 if (not isinstance(features,torch.Tensor) or features.dtype!=torch.float32 or tuple(features.shape)!=(len(cache['sample_ids']),512)
     or not torch.isfinite(features).all() or list(payload['sample_ids'])!=list(cache['sample_ids'])
     or not torch.equal(payload['targets'],cache['targets'])):raise ValueError('feature dtype/shape/order/row binding')
 norms=torch.linalg.vector_norm(features,dim=1);nonzero=norms>0
 if not torch.allclose(norms[nonzero],torch.ones_like(norms[nonzero]),rtol=0,atol=2e-5):raise ValueError('feature normalization')
 return features.numpy()

def graph_scores(scores,features,*,deadline=None):
 neighbors,weights,top=blocked_topk(features,20,row_block=256,max_block_bytes=64*1024**2,deadline=deadline)
 transition,graph,degree=mutual_graph(neighbors,weights,10)
 values,detail=smooth(scores,transition,degree,.3,iterations=3,deadline=deadline)
 return values,dict(topk=top,graph=graph,smoothing=detail)

def run_comparison(candidate,control,queries,features,control_features,*,deadline=None,draws=500):
 raw=compare_arrays(candidate,control,queries,draws=draws);combo=[];increment=[];details=[]
 for name,prefix in [('natural_unique_images','natural'),('balanced_supported_unique','balanced')]:
  check(deadline);ix=queries[prefix+'_indices'];cols=queries[prefix+'_columns']
  values,detail=graph_scores(candidate['scores'][ix][:,cols].copy(),features[ix],deadline=deadline)
  baseline,control_detail=graph_scores(control['scores'][ix][:,cols].copy(),control_features[ix],deadline=deadline)
  control_prediction,_=frozen_decode(baseline)
  modified=dict(candidate,scores=candidate['scores'].copy());modified['scores'][np.ix_(ix,cols)]=values
  row=compare_arrays(modified,control,queries,draws=draws,only_protocol=name,control_predictions={name:cols[control_prediction]})[0]
  row.pop('raw_argmax_candidate_vs_control',None);row['predictions'].pop('candidate_raw',None);row['predictions'].pop('control_raw',None);row['primary_scope']=PRIMARY_SCOPE
  combo.append(row);increment.append(compare_arrays(modified,candidate,queries,draws=draws,only_protocol=name)[0]);details.append(dict(name=name,candidate=detail,control=control_detail))
 return raw,combo,increment,details

def load_features(folder,cache):
 if sha(folder/'fused_features.pt')!=cache['identity']['feature_file_sha256']:raise ValueError('same-checkpoint feature SHA')
 payload=torch.load(folder/'fused_features.pt',map_location='cpu',weights_only=True,mmap=True)
 logits=torch.load(folder/'fused_logits.pt',map_location='cpu',weights_only=True,mmap=True)
 return validate_features(payload,logits,cache['identity'])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=('L35',),required=True);a=p.parse_args()
 source_sha=verify_owned_sources();context=phase_context(TRAIN_ROOT,live=True);torch.set_num_threads(2)
 if a.arm!=BINDING['selected_candidate']:raise ValueError('one actually bound candidate only')
 if memory_reason(memory_snapshot(),4.) is not None:raise RuntimeError('effective host/cgroup memory below4GiB')
 cfolder=ROOT/'artifacts/diagnostics/CONFIRM_J5';folder=ROOT/f'artifacts/diagnostics/CONFIRM_{a.arm}';output=ROOT/f'artifacts/comparisons/{a.arm}'
 if output.exists():raise FileExistsError('will not overwrite a comparison')
 candidate=load_cache(folder/'fused_logits.pt',folder/'input_identity.json');control=load_cache(cfolder/'fused_logits.pt',cfolder/'input_identity.json');validate_pair(candidate,control)
 for cache in (candidate,control):
  if cache['identity']['diagnostic_source_manifest_sha256']!=source_sha:raise ValueError('both diagnostics same frozen source')
  terminal=cache['identity']['confirmation_phase_context']
  if terminal.get('controller_and_all_arm_groups_closed') is not True or terminal.get('canonical_idle') is not True or terminal.get('training_source_manifest_sha256')!=TRAINING_SOURCE_SHA or terminal['shared_phase_context']['budget_sha256']!=context['budget_sha256']:raise ValueError('actual terminal pair and shared ledger')
 sys.path[:0]=[str(TRAIN_ROOT/'src'),str(TRAIN_ROOT/'scripts')]
 from aictrack1.engine.training_queue import read_training_lock,probe_running_trainers
 if read_training_lock('/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json') or probe_running_trainers():raise RuntimeError('canonical idle required')
 features=load_features(folder,candidate);control_features=load_features(cfolder,control);queries,identity=load_queries(control,ROOT/'inputs/frozen_queries')
 deadline=time.monotonic()+max(0,context['budget']['deadline_unix']-time.time()-120);started=time.monotonic();check(deadline);output.mkdir(parents=True,exist_ok=False)
 raw,combo,increment,details=run_comparison(candidate,control,queries,features,control_features,deadline=deadline)
 for label,results in [('raw_model_P7_vs_J5_P7',raw),('graph_combo_vs_J5_GRAPH',combo),('candidate_graph_increment',increment)]:
  for row in results:np.savez_compressed(output/(label+'_'+row['name']+'_predictions.npz'),**row.pop('predictions'))
 primary=next(r for r in combo if r['name']=='balanced_supported_unique')['p7']['candidate_vs_control'];natural=next(r for r in combo if r['name']=='natural_unique_images')['p7']['candidate_vs_control']
 report=dict(schema='aictrack1.next_cycle10.L35_confirmation.paired.v1',complete=True,selected_arm=a.arm,visual_learning_rate=3.5e-6,hard_alpha=.1,primary_scope=PRIMARY_SCOPE,raw_P7_separately_reported=True,independent_confirmation=True,used_test_or_online_scores=False,test_access=False,final_model='single',diagnostic_source_manifest_sha256=source_sha,training_source_manifest_sha256=TRAINING_SOURCE_SHA,split_sha256=SPLIT_SHA,query_definition_source_sha256=QUERY_SOURCE_SHA,query_identity_sha256=QUERY_IDENTITY_SHA,new_H1_checkpoint_sha256=H1_SHA,new_H1_source_sha256=H1_SOURCE_SHA,training_seed=20261010,control_last_sha256=control['identity']['checkpoint_sha256'],candidate_last_sha256=candidate['identity']['checkpoint_sha256'],candidate=candidate['provenance'],control=control['provenance'],query_identity=identity,fixed_graph=dict(k=10,lambda_value=.3,iterations=3,probability_temperature=1.,margin_protection=math.log(4.)),frozen_P7=dict(temperature=.5,strength=.75,iterations=60),protocols=combo,raw_model_P7_vs_J5_P7=raw,candidate_graph_increment=increment,graph_details=details,primary_delta_pp=primary['delta_vs_baseline_pp'],natural_delta_pp=natural['delta_vs_baseline_pp'],direction=('positive_requires_root_risk_review' if primary['delta_vs_baseline_pp']>0 else 'not_positive'),natural_nonnegative_hard_gate=False,ci_lower_positive_hard_gate=False,stable_gain_claim=False,model_updates=0,additional_model_forwards=0,graph_wall_seconds=time.monotonic()-started,shared_phase_context=context,control_margin_reference='pregraph fresh J5 logits; decisions from this fresh J5 same-checkpoint fixed graph/P7; both graphs evaluated separately within each predetermined query')
 resource_check(context);save_json(output/'report.json',report);save_json(output/'result_manifest.json',dict(complete=True,files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(output.iterdir()) if f.is_file()},test_access=False))
 print(json.dumps(dict(complete=True,selected_arm=a.arm,report=str(output/'report.json'),primary_delta_pp=report['primary_delta_pp'],independent_confirmation=True)));return 0
if __name__=='__main__':raise SystemExit(main())
