"""Root-only CPU binding after real new H1100; creates a NEW execution directory."""
from __future__ import annotations
import argparse,json,math,os,shutil,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p1_contract import REMOTE_ROOT,QUERY_SOURCE_SHA,SPLIT_SHA,CLIP_SHA,MAPPING_SHA,MANIFEST_SHA,CLASS_FILE_SHA,LOCK,sha,read_json,require,verify_files,admission,expected_config,digest

def validate_h1_payload(state,receipt,source,definition,checkpoint_sha):
 import torch
 require(receipt.get('complete') is True and receipt.get('epoch')==100 and receipt.get('checkpoint_sha256')==checkpoint_sha and receipt.get('source_sha256')==state.get('source_sha256'),'actual completed H1 provenance')
 require(receipt.get('row_binding')==state.get('row_binding')==definition['h1_row_binding'] and receipt.get('train_count')==118958 and receipt.get('validation_count')==29737 and receipt.get('split_sha256')==SPLIT_SHA,'new H1 exact positive rows')
 require(receipt.get('scope')=='fold_train_fixed_final_epochs' and receipt.get('validation_selection') is False and receipt.get('test_access') is False and receipt.get('actual_cpu_optimizer_scheduler_scaler_model_rng_restored') is True,'actual new H1 restored/train-only receipt')
 expected=dict(epochs=100,batch_size=4096,learning_rate=.01,weight_decay=0.,momentum=.9,gradient_clip_norm=None,seed=20261008,optimizer='adam',scheduler='constant',amp=False)
 require(state.get('training_scope')=='fold_train_fixed_final_epochs' and state.get('epoch')==100 and state.get('config')==expected and state.get('head')=={'kind':'linear'},'new fixed100 same H1 recipe')
 require(state.get('selection_rule')=='fixed epoch100 last; new-fold train only; no validation selection','train-only final H1 only')
 for k,v in dict(competition_stage='stage2',train_manifest_sha256=MANIFEST_SHA,split_sha256=SPLIT_SHA,class_mapping_sha256=MAPPING_SHA,clip_weight_sha256=CLIP_SHA,num_classes=750).items():require(state.get(k)==v,'actual H1 '+k)
 require(set(state.get('model',{}))=={'linear.weight','linear.bias'},'single linear head only')
 for k,shape in [('linear.weight',(750,512)),('linear.bias',(750,))]:
  v=state['model'][k];require(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and tuple(v.shape)==shape and bool(torch.isfinite(v).all()),'H1 finite master tensor '+k)
 meta=state.get('feature_cache_metadata',{});require(meta.get('normalized') is True and meta.get('split_sha256')==SPLIT_SHA and meta.get('peft_checkpoint_sha256') is None,'new normalized official feature metadata')
 require(source.get('schema')=='aictrack1.next_cycle10.CG_confirmation.h1.source.v1' and source.get('status')=='bound_h1_only' and source.get('query_definition_source_sha256')==QUERY_SOURCE_SHA,'actual new H1 source only')
 require(isinstance(receipt.get('terminal_cache_parity_top1'),(float,int)) and math.isfinite(receipt['terminal_cache_parity_top1']) and 0<=receipt['terminal_cache_parity_top1']<=1,'real initializer parity')
 return True

def validate_budget(value):
 from CG_admission_contract import validate_confirmation_budget
 return validate_confirmation_budget(value)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--h1-root',type=Path,required=True);p.add_argument('--expected-h1-source-sha256',required=True);p.add_argument('--expected-h1-checkpoint-sha256',required=True);p.add_argument('--expected-h1-template-sha256',required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--budget',type=Path,required=True);a=p.parse_args()
 source=verify_files(require_bound=False);require(source['status']=='unbound_template','template only');require(os.name=='posix' and a.output_root.resolve()==REMOTE_ROOT.resolve() and not a.output_root.exists(),'new exact dedicated remote visual ROOT')
 h1root=a.h1_root.resolve();require(h1root==Path('/root/autodl-tmp/AICtrack1_next_cycle_confirmation_h1_CG_20261010'),'dedicated actual new H1 ROOT')
 h1source_path=h1root/'source_manifest.json';require(sha(h1source_path)==a.expected_h1_source_sha256,'actual H1 source SHA');h1source=read_json(h1source_path);require(h1source.get('template_manifest_sha256')==a.expected_h1_template_sha256 and digest(a.expected_h1_template_sha256),'reviewed actual H1 template ancestry')
 for rel,d in h1source['files'].items():
  part=Path(rel);require(not part.is_absolute() and '..' not in part.parts and (h1root/part).resolve().is_relative_to(h1root) and sha(h1root/part)==d,'actual H1 source closure '+rel)
 choice=admission(read_json(a.admission));b=validate_budget(read_json(a.budget));require(sha(h1root/'inputs/budget.json')==sha(a.budget) and sha(h1root/'inputs/root_admission.json')==sha(a.admission),'H1 and both visual arms share same budget/admission bytes')
 checkpoint=h1root/'artifacts/experiments/confirmation_H1/last.pt';receipt_path=checkpoint.with_name('h1_receipt.json');require(sha(checkpoint)==a.expected_h1_checkpoint_sha256,'actual new H1 last SHA')
 import torch
 state=torch.load(checkpoint,map_location='cpu',weights_only=True);receipt=read_json(receipt_path);definition=read_json(ROOT/'inputs/confirmation_assets.json');validate_h1_payload(state,receipt,h1source,definition,a.expected_h1_checkpoint_sha256)
 from aictrack1.models.heads import load_classifier_head_checkpoint
 head,loaded=load_classifier_head_checkpoint(checkpoint,feature_dim=512,num_classes=750,expected_clip_weight_sha256=CLIP_SHA,expected_class_mapping_sha256=MAPPING_SHA,expected_split_sha256=SPLIT_SHA,expected_provenance=None,forbid_full_train_initializer=True)
 require(all(torch.equal(state['model'][k],v) for k,v in head.state_dict().items()),'real H1 model safe CPU restore parity')
 d=read_json(h1root/'inputs/h1_deployment.json');actual=dict(d['assets']);actual.update(h1={'path':str(checkpoint),'sha256':sha(checkpoint)},h1_receipt={'path':str(receipt_path),'sha256':sha(receipt_path)},h1_source={'path':str(h1source_path),'sha256':sha(h1source_path)})
 for name,item in actual.items():require(sha(item['path'])==item['sha256'],'actual asset byte SHA '+name)
 require(actual['class_mapping']['sha256']==CLASS_FILE_SHA,'classes.csv bytes versus logical mapping separate')
 from aictrack1.engine.training_queue import read_training_lock,probe_running_trainers
 require(read_training_lock(LOCK) is None and not probe_running_trainers(),'canonical idle before bind')
 a.output_root.mkdir(parents=True,exist_ok=False)
 for rel in source['files']:
  target=a.output_root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
 shutil.copyfile(ROOT/'source_manifest.json',a.output_root/'template_manifest.json');shutil.copyfile(a.admission,a.output_root/'inputs/root_admission.json');shutil.copyfile(a.budget,a.output_root/'inputs/budget.json')
 assets=dict(schema='aictrack1.next_cycle10.confirmation.visual.assets.v1',competition_stage='stage2',arms=['J5',choice['selected_arm']],selected_arm=choice['selected_arm'],selected_route=choice['selected_arm']+'_GRAPH',assets=actual,dataset_root=d['dataset_root'],canonical_lock=LOCK,train_count=118958,validation_count=29737,num_classes=750,training_seed=20261010,split_seed=20261010,h1_epoch=100,h1_row_binding=definition['h1_row_binding'],h1_validation_used_for_selection=False,query_definition_source_sha256=QUERY_SOURCE_SHA,source_h1_manifest_sha256=sha(h1source_path),h1_template_manifest_sha256=a.expected_h1_template_sha256,terminal_cache_parity_top1=receipt['terminal_cache_parity_top1'],budget_sha256=sha(a.budget),admission_sha256=sha(a.admission),runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','actual_torch_threads':4},test_access=False)
 (a.output_root/'inputs/p1_assets.json').write_text(json.dumps(assets,indent=2)+'\n',encoding='utf-8')
 for arm in assets['arms']:
  target=a.output_root/f'configs/experiment/p1_{arm}.yaml';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(yaml.safe_dump({'experiment':expected_config(arm,assets,root=a.output_root)},sort_keys=False),encoding='utf-8')
 manifest=dict(schema='aictrack1.next_cycle10.CG_confirmation.visual.source.v1',status='bound',arms=assets['arms'],selected_candidate=choice['selected_arm'],template_manifest_sha256=sha(ROOT/'source_manifest.json'),query_definition_source_sha256=QUERY_SOURCE_SHA,source_h1_manifest_sha256=sha(h1source_path),h1_checkpoint_sha256=sha(checkpoint),actual_budget_sha256=sha(a.budget),actual_admission_sha256=sha(a.admission),training_seed=20261010,training_scope='fixed_split_validation',fixed_epochs=20,checkpoint_role='last',future_checkpoint_sha256={arm:None for arm in assets['arms']},GPU_started=False,test_access=False,files={p.relative_to(a.output_root).as_posix():sha(p) for p in sorted(a.output_root.rglob('*')) if p.is_file()})
 (a.output_root/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(dict(status='bound_CPU_only',output_root=str(a.output_root),source_manifest_sha256=sha(a.output_root/'source_manifest.json'),arms=assets['arms'],h1_checkpoint_sha256=sha(checkpoint),model_CPU_restored=True,GPU_started=False)));return 0
if __name__=='__main__':raise SystemExit(main())
