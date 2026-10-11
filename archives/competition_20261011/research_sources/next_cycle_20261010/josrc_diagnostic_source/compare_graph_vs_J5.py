"""Fixed graph+P7 alpha candidate versus the actual old J5_GRAPH prediction, no grid."""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from fresh_diag_contract import ROOT,SPLIT_SHA,H1_SHA,TRAIN_ROOT,TRAINING_SOURCE_SHA,load_cache,validate_pair,verify_owned_sources,sha,save_json
from paired_metrics import load_queries,compare_arrays
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

def imported_graph_predictions(queries):
 registration=read_json(ROOT/'inputs/control_import.json');identity=read_json(ROOT/registration['local_identity'])
 if identity.get('model_arm')!='J5' or identity.get('checkpoint_sha256')!=registration['checkpoint_sha256'] or identity.get('diagnostic_source_manifest_sha256')!='01949375a866775cd141ded72e06afdb0b1922dd5aa60702cb7fea3b5fbc5ec1':raise ValueError('real fixed-last oldJ5 graph identity')
 graph=read_json(ROOT/registration['graph_report'],16*1024**2);manifest=read_json(ROOT/registration['graph_result_manifest'])
 if graph.get('complete') is not True or graph.get('selected_arm')!='J5' or graph.get('primary_scope')!='J5_graph_P7_vs_C' or graph.get('fixed_graph')!=dict(k=10,lambda_value=.3,iterations=3,probability_temperature=1.,margin_protection=math.log(4.)):raise ValueError('actual fixed graph producer report')
 if graph.get('candidate',{}).get('identity_sha256')!=registration['cache_identity']['sha256'] or graph.get('candidate',{}).get('checkpoint_sha256')!=identity['checkpoint_sha256']:raise ValueError('real graph belongs to this J5 checkpoint')
 if sha(ROOT/registration['graph_report'])!=manifest['files']['report.json']['sha256']:raise ValueError('actual graph report result manifest')
 predictions={};proof=[]
 for name,prefix in [('natural_unique_images','natural'),('balanced_supported_unique','balanced')]:
  filename='combo_vs_C_'+name+'_predictions.npz';path=ROOT/'inputs/legacy_control/graph'/filename
  if sha(path)!=manifest['files'][filename]['sha256']:raise ValueError('actual control graph NPZ hash')
  with np.load(path,allow_pickle=False) as stored:values=stored[registration['graph_prediction_field']].copy()
  if values.dtype!=np.int64 or values.shape!=(len(queries[prefix+'_indices']),) or not np.isin(values,queries[prefix+'_columns']).all():raise ValueError('actual graph control query order/classes')
  predictions[name]=values;proof.append(dict(protocol=name,path=str(path),sha256=sha(path),field=registration['graph_prediction_field']))
 return predictions,dict(import_registration=registration,actual_graph_report_sha256=sha(ROOT/registration['graph_report']),prediction_receipts=proof,legacy_source_identity_preserved=True,new_source_identity_not_claimed=True)

def run_comparison(candidate,control,queries,features,control_graph_predictions,*,deadline=None,draws=500):
 raw=compare_arrays(candidate,control,queries,draws=draws);combo=[];increment=[];details=[]
 for name,prefix in [('natural_unique_images','natural'),('balanced_supported_unique','balanced')]:
  check(deadline);ix=queries[prefix+'_indices'];cols=queries[prefix+'_columns']
  values,detail=graph_scores(candidate['scores'][ix][:,cols].copy(),features[ix],deadline=deadline)
  modified=dict(candidate,scores=candidate['scores'].copy());modified['scores'][np.ix_(ix,cols)]=values
  row=compare_arrays(modified,control,queries,draws=draws,only_protocol=name,control_predictions=control_graph_predictions)[0]
  row.pop('raw_argmax_candidate_vs_control',None);row['predictions'].pop('candidate_raw',None);row['predictions'].pop('control_raw',None)
  row['primary_scope']='candidate_fixed_graph_P7_vs_actual_J5_GRAPH_P7'
  combo.append(row);increment.append(compare_arrays(modified,candidate,queries,draws=draws,only_protocol=name)[0]);details.append(dict(name=name,**detail))
 return raw,combo,increment,details

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=('B25','B50'),required=True);args=parser.parse_args()
 source_sha=verify_owned_sources();context=phase_context(TRAIN_ROOT,live=True);torch.set_num_threads(2)
 if memory_reason(memory_snapshot(),4.) is not None:raise RuntimeError('effective host/cgroup memory below4GiB')
 folder=ROOT/f'artifacts/diagnostics/JOINT_{args.arm}';output=ROOT/f'artifacts/comparisons/{args.arm}'
 if output.exists():raise FileExistsError('will not overwrite comparison')
 candidate=load_cache(folder/'fused_logits.pt',folder/'input_identity.json');registration=read_json(ROOT/'inputs/control_import.json')
 if sha(registration['cache_identity']['path'])!=registration['cache_identity']['sha256'] or sha(registration['cache_bundle']['path'])!=registration['cache_bundle']['sha256']:raise ValueError('actual historical control cache identity')
 control=load_cache(registration['cache_bundle']['path'],registration['cache_identity']['path']);validate_pair(candidate,control)
 if candidate['identity']['diagnostic_source_manifest_sha256']!=source_sha or candidate['identity'].get('independent_confirmation') is not False:raise ValueError('current exploratory cache only')
 terminal=candidate['identity']['exploration_phase_context']
 if terminal.get('controller_and_all_arm_groups_closed') is not True or terminal.get('canonical_idle') is not True or terminal.get('training_source_manifest_sha256')!=TRAINING_SOURCE_SHA or terminal['shared_phase_context']['budget_sha256']!=context['budget_sha256']:raise ValueError('actual terminal shared-stage evidence')
 sys.path[:0]=[str(TRAIN_ROOT/'src'),str(TRAIN_ROOT/'scripts')]
 from aictrack1.engine.training_queue import read_training_lock,probe_running_trainers
 if read_training_lock('/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json') or probe_running_trainers():raise RuntimeError('wait for canonical idle')
 if sha(folder/'fused_features.pt')!=candidate['identity']['feature_file_sha256']:raise ValueError('feature exact SHA')
 cache=torch.load(folder/'fused_logits.pt',map_location='cpu',weights_only=True,mmap=True);payload=torch.load(folder/'fused_features.pt',map_location='cpu',weights_only=True,mmap=True)
 features=validate_features(payload,cache,candidate['identity']);queries,query_identity=load_queries(control,ROOT/'inputs/frozen_queries');baseline,import_proof=imported_graph_predictions(queries)
 deadline=time.monotonic()+max(0,context['budget']['deadline_unix']-time.time()-120);started=time.monotonic();check(deadline);output.mkdir(parents=True,exist_ok=False)
 raw,combo,increment,details=run_comparison(candidate,control,queries,features,baseline,deadline=deadline)
 for label,results in [('raw_model_P7_vs_J5_P7',raw),('graph_combo_vs_J5_GRAPH',combo),('candidate_graph_increment',increment)]:
  for row in results:np.savez_compressed(output/(label+'_'+row['name']+'_predictions.npz'),**row.pop('predictions'))
 primary=next(row for row in combo if row['name']=='balanced_supported_unique')['p7']['candidate_vs_control'];natural=next(row for row in combo if row['name']=='natural_unique_images')['p7']['candidate_vs_control']
 report=dict(schema='aictrack1.josrc_inspired.exploration_paired_graph.v1',complete=True,arm=args.arm,hard_alpha=.1,soft_coefficient=(.25 if args.arm=='B25' else .5),math_source_sha256=__import__('fresh_diag_contract').MATH_SOURCE_SHA,primary_scope='candidate_fixed_graph_P7_vs_actual_J5_GRAPH_P7',independent_confirmation=False,old_query_reused_for_exploration=True,stable_gain_claim=False,used_test_or_online_scores=False,test_access=False,diagnostic_source_manifest_sha256=source_sha,training_source_manifest_sha256=TRAINING_SOURCE_SHA,split_sha256=SPLIT_SHA,h1_sha256=H1_SHA,query_identity=query_identity,candidate=candidate['provenance'],control=control['provenance'],control_graph_import=import_proof,fixed_graph=dict(k=10,lambda_value=.3,iterations=3,probability_temperature=1.,margin_protection=math.log(4.)),frozen_P7=dict(temperature=.5,strength=.75,iterations=60),protocols=combo,raw_model_P7_vs_J5_P7=raw,candidate_graph_increment=increment,graph_details=details,primary_delta_pp=primary['delta_vs_baseline_pp'],natural_delta_pp=natural['delta_vs_baseline_pp'],direction=('exploratory_positive' if primary['delta_vs_baseline_pp']>0 else 'exploratory_not_positive'),natural_nonnegative_hard_gate=False,ci_lower_positive_hard_gate=False,model_updates=0,additional_model_forwards=0,graph_wall_seconds=time.monotonic()-started,shared_phase_context=context,control_margin_reference='pregraph raw J5 logits, kept explicit; control decisions from actual fixed graph NPZ')
 resource_check(context);save_json(output/'report.json',report);save_json(output/'result_manifest.json',dict(complete=True,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()},test_access=False))
 print(json.dumps(dict(complete=True,arm=args.arm,report=str(output/'report.json'),primary_delta_pp=report['primary_delta_pp'],exploration_only=True)));return 0
if __name__=='__main__':raise SystemExit(main())
