"""One frozen reciprocal-cosine graph point versus actual fresh C; no parameter search."""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from fresh_diag_contract import ROOT,SPLIT_SHA,H1_SHA,load_cache,validate_pair,verify_owned_sources,sha,save_json
from compare_fresh_route import load_queries,compare_arrays,eligibility
from diagnostic_resources import memory_snapshot,memory_reason
sys.path.insert(0,str(ROOT/'reference'))
from p3_graph import blocked_topk,mutual_graph,smooth,p7,check

GRAPH_ROOT=Path('/root/autodl-tmp/AICtrack1_next_cycle_graph_20261009')
GRAPH_SOURCE_SHA='5748cf07c01bd8bf3d68bf5b50065303d2c8990260b00400e5511593aaa0834c'
GRAPH_PROTOCOL_SHA='f48211db3716faaf45255dcc96e0b614bf42c1673791406bf9d7457c1a50b30d'
ORIGINAL_SPENT_SECONDS=64.3705
PHASE_DEADLINE_UNIX=1791526772.440027
FEATURE_PROTOCOL='FP32 normalize(mean(FP32 normalize(each of 4 raw visual views))); same checkpoint/order; 0 extra forwards'

def import_path(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def load_existing_budget(root=GRAPH_ROOT):
 root=Path(root).resolve()
 if root!=GRAPH_ROOT.resolve():raise ValueError('literal original shared graph root only')
 if sha(root/'source_manifest.json')!=GRAPH_SOURCE_SHA:raise ValueError('original graph source pin')
 m=json.loads((root/'source_manifest.json').read_text())
 for name,digest in m['files'].items():
  p=(root/name).resolve()
  if not p.is_relative_to(root) or sha(p)!=digest:raise ValueError('graph source drift or escape')
 path=root/'artifacts/graph_budget.json'
 if not path.is_file():raise ValueError('original graph ledger must exist; never create/reset')
 value=json.loads(path.read_text())
 spent=value.get('spent_wall_seconds')
 if (value.get('schema')!='aictrack1.p3.aggregate_budget.v1' or value.get('source_manifest_sha256')!=GRAPH_SOURCE_SHA
     or value.get('total_wall_seconds')!=7200. or not isinstance(spent,(int,float)) or not math.isfinite(spent)
     or spent<ORIGINAL_SPENT_SECONDS-1e-4):raise ValueError('original aggregate graph wall ledger binding')
 contract=import_path('p3_contract',root/'p3_contract.py')
 budget=import_path('p3_budget',root/'p3_budget.py')
 return budget.BudgetLease(path),dict(path=str(path),before_sha256=sha(path),spent_before_seconds=spent,total_wall_seconds=7200.)

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

def run_comparison(candidate,control,queries,features,*,deadline=None,draws=500):
 """Build a separate graph per registered query; all predictions before label evaluation."""
 import compare_fresh_route as cmp
 old_decode=cmp.frozen_decode
 cmp.frozen_decode=lambda scores:p7(scores,deadline=deadline)
 try:
  raw_results=compare_arrays(candidate,control,queries,draws=draws)
  combo_results=[];increments=[];details=[]
  for name,prefix in [('natural_unique_images','natural'),('balanced_supported_unique','balanced')]:
   check(deadline);ix=queries[prefix+'_indices'];cols=queries[prefix+'_columns']
   transformed,detail=graph_scores(candidate['scores'][ix][:,cols].copy(),features[ix],deadline=deadline)
   # evaluate through the identical risk/bootstrap implementation using only this query.
   # preserve global rows and class indices, without cloning any graph rows.
   graph_candidate=dict(candidate,scores=candidate['scores'].copy())
   graph_candidate['scores'][np.ix_(ix,cols)]=transformed
   one_queries={k:v for k,v in queries.items()}
   evaluated=compare_arrays(graph_candidate,control,one_queries,draws=draws,only_protocol=name)
   combo=next(r for r in evaluated if r['name']==name)
   increment=next(r for r in compare_arrays(graph_candidate,candidate,one_queries,draws=draws,only_protocol=name) if r['name']==name)
   combo_results.append(combo);increments.append(increment);details.append(dict(name=name,**detail))
  return raw_results,combo_results,increments,details
 finally:cmp.frozen_decode=old_decode

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('candidate-bundle','candidate-identity','candidate-features','control-bundle','control-identity','output'):
  parser.add_argument('--'+name,required=True,type=Path)
 parser.add_argument('--deadline-unix',required=True,type=float)
 parser.add_argument('--threads',default=2,type=int)
 args=parser.parse_args()
 if args.threads!=2 or args.deadline_unix>PHASE_DEADLINE_UNIX:raise ValueError('original CPU2/phase deadline only')
 if time.time()>=args.deadline_unix:raise TimeoutError('original fresh phase expired')
 torch.set_num_threads(2)
 source_sha=verify_owned_sources()
 if memory_reason(memory_snapshot(),4.) is not None:raise RuntimeError('effective host/cgroup free memory below4GiB')
 if args.output.exists():raise FileExistsError(args.output)
 lease,ledger=load_existing_budget()
 with lease:
  deadline=time.monotonic()+min(lease.remaining,max(0,args.deadline_unix-time.time()))
  candidate=load_cache(args.candidate_bundle,args.candidate_identity);control=load_cache(args.control_bundle,args.control_identity);validate_pair(candidate,control)
  if any(data['identity']['diagnostic_source_manifest_sha256']!=source_sha for data in (candidate,control)):raise ValueError('current exact diagnostic source required for both fresh caches')
  if candidate['identity']['selected_route']!='J5_GRAPH' or candidate['identity']['model_arm']!='J5':raise ValueError('only preselected J5_GRAPH')
  if sha(args.candidate_features)!=candidate['identity'].get('feature_file_sha256'):raise ValueError('feature positive SHA binding')
  cache=torch.load(args.candidate_bundle,map_location='cpu',weights_only=True,mmap=True);payload=torch.load(args.candidate_features,map_location='cpu',weights_only=True,mmap=True)
  features=validate_features(payload,cache,candidate['identity']);queries,query_identity=load_queries(control,ROOT/'inputs/frozen_queries')
  check(deadline);args.output.mkdir(parents=True)
  raw,combo,increment,details=run_comparison(candidate,control,queries,features,deadline=deadline)
  for label,results in [('raw_j5_vs_C',raw),('combo_vs_C',combo),('graph_increment_vs_J5',increment)]:
   for row in results:
    predictions=row.pop('predictions');np.savez_compressed(args.output/(label+'_'+row['name']+'_predictions.npz'),**predictions)
  report=dict(schema='aictrack1.fresh_route.graph_combo_paired.v1',complete=True,selected_route='J5_GRAPH',selected_arm='J5',p1_arm='J5',test_access=False,used_test_or_online_scores=False,
   split_sha256=SPLIT_SHA,h1_sha256=H1_SHA,source_h1_manifest_sha256=candidate['identity']['source_h1_manifest_sha256'],
   training_source_manifest_sha256=candidate['identity']['training_source_manifest_sha256'],diagnostic_source_manifest_sha256=source_sha,
   admission_receipt_sha256=candidate['identity']['admission_receipt_sha256'],candidate=dict(candidate['provenance'],provenance=candidate['provenance']),control=dict(control['provenance'],provenance=control['provenance']),
   features=dict(path=str(args.candidate_features.resolve()),sha256=sha(args.candidate_features)),query_identity=query_identity,
   selection_protocol_sha256=sha(ROOT/'selection_protocol.json'),frozen_postprocess_sha256=sha(ROOT/'frozen_postprocess.json'),
   fixed_graph_protocol_sha256=sha(ROOT/'fixed_graph_protocol.json'),graph_source_sha256=GRAPH_SOURCE_SHA,graph_budget=ledger,
   fixed_graph=dict(k=10,lambda_value=.3,iterations=3,probability_temperature=1.,margin_protection=math.log(4.)),environment=dict(cpu_threads=2,torch=torch.__version__,numpy=np.__version__),
   raw_j5_vs_C=raw,protocols=combo,graph_increment_vs_J5=increment,graph_details=details,eligibility=eligibility(combo),
   primary_scope='J5_graph_P7_vs_C',raw_j5_positive_is_not_admission_gate=True,additional_model_forwards=0,model_updates=0)
  save_json(args.output/'report.json',report)
  save_json(args.output/'result_manifest.json',dict(schema='aictrack1.fresh_route.graph_result.v1',complete=True,test_access=False,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(args.output.iterdir()) if p.is_file()}))
 # Snapshot only after the shared lease has charged this entire run; future runs may update original ledger.
 snapshot=args.output/'graph_budget_snapshot.json'
 snapshot.write_bytes(Path(ledger['path']).read_bytes())
 charged=json.loads(snapshot.read_text());report['graph_budget'].update(after_snapshot_sha256=sha(snapshot),spent_after_seconds=charged['spent_wall_seconds'],shared_ledger_charged=True);save_json(args.output/'report.json',report)
 manifest=json.loads((args.output/'result_manifest.json').read_text());manifest['files'][snapshot.name]=dict(bytes=snapshot.stat().st_size,sha256=sha(snapshot));manifest['files']['report.json']=dict(bytes=(args.output/'report.json').stat().st_size,sha256=sha(args.output/'report.json'));save_json(args.output/'result_manifest.json',manifest)
 print(json.dumps(dict(complete=True,report=str(args.output/'report.json'),eligibility=report['eligibility']),ensure_ascii=False))
 return 0
if __name__=='__main__':raise SystemExit(main())
