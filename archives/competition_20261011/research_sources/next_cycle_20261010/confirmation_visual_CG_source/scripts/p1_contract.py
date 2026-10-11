"""Runtime-bound new610111 fold confirmation; one actual positive candidate only."""
from __future__ import annotations
import copy,hashlib,json,math
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
REMOTE_ROOT=Path('/root/autodl-tmp/AICtrack1_next_cycle_confirmation_visual_CG_20261010')
ARMS={'J5':.1,'CG25':.1}
QUERY_SOURCE_SHA='51a79ffbaf94fbb333cdab49123e55ccc3fb5cf2fe78cfdb6b5ea86da8810e50'
H1_TEMPLATE_SHA=None  # runtime receives actual reviewed PF H1 template SHA, never a future model SHA
SPLIT_SHA='610111665e1055bcda7157add15b16ade353138365e54ee87dc92316e6dfd6f1'
MAPPING_SHA='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
MANIFEST_SHA='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
CLASS_FILE_SHA='0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be'
LOCK='/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json'
def require(value,message):
 if not value:raise ValueError(message)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(value):return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)
def admission(value):
 from CG_admission_contract import validate_CG_admission
 return validate_CG_admission(value)

def assets(root=ROOT):
 root=Path(root);p=read_json(root/'inputs/p1_assets.json');definition=read_json(root/'inputs/confirmation_assets.json');choice=admission(read_json(root/'inputs/root_admission.json'))['selected_arm']
 require(p.get('schema')=='aictrack1.next_cycle10.confirmation.visual.assets.v1' and p.get('competition_stage')=='stage2' and p.get('test_access') is False,'actual new-fold assets required')
 require(p.get('selected_arm')==choice and p.get('selected_route')==choice+'_GRAPH' and p.get('arms')==['J5',choice],'control plus one selected candidate only')
 require(p.get('training_seed')==20261010 and p.get('train_count')==118958 and p.get('validation_count')==29737 and p.get('num_classes')==750 and p.get('canonical_lock')==LOCK,'new matched seed/counts/canonical required')
 require(p.get('h1_row_binding')==definition['h1_row_binding'] and p.get('h1_epoch')==100 and p.get('h1_validation_used_for_selection') is False,'new train-only H1 rows required')
 require(p.get('query_definition_source_sha256')==QUERY_SOURCE_SHA and sha(root/'inputs/query_definition_source_manifest.json')==QUERY_SOURCE_SHA,'new pre-result split source required')
 require(definition.get('old0c9_query_connected_group_overlap')==definition.get('used57189_query_connected_group_overlap')==definition.get('train_query_connected_group_overlap')==0,'all query group overlaps must be zero')
 for name,d in [('clip',CLIP_SHA),('manifest',MANIFEST_SHA),('split',SPLIT_SHA),('class_mapping',CLASS_FILE_SHA)]:require(p['assets'][name]['sha256']==d,'new official asset identity '+name)
 require(digest(p['assets']['h1']['sha256']) and p['assets']['h1']['sha256'] not in {'e88a1cbbaa8d9ebdbf1c7cf2707ac434cf440fac8bd5b4a5dc57893ba0fe15be','fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605'},'old/full H1 explicitly forbidden')
 require(p.get('source_h1_manifest_sha256')==p['assets']['h1_source']['sha256'] and digest(p['source_h1_manifest_sha256']),'actual new H1 producer required')
 return p
def expected_config(arm,p,root=ROOT):
 require(arm in ARMS and arm in p['arms'],'only registered J5/CG25 bound arms')
 raw=copy.deepcopy(yaml.safe_load((Path(root)/'inputs/baseline_J5.yaml').read_text(encoding='utf-8'))['experiment'])
 raw['base_loss']['alpha']=.1;raw['training']['seed']=20261010
 raw.pop('persistent_filter',None)
 raw['dataset_config']=f'../data/p1_{arm}_336.yaml'
 raw['run_id']=f'20261010-confirmation-{arm}-fixed20';raw['output_dir']=f'../../artifacts/experiments/CONFIRM_{arm}'
 for key,asset in [('manifest','manifest'),('split','split'),('label_conflicts','label_conflicts'),('head_checkpoint','h1')]:raw[key]=p['assets'][asset]['path']
 return raw
def validate_config(raw,arm,p):require(raw==expected_config(arm,p),'only new split/head, commonseed, dataset-config path and CG25 crop field/runid/output may differ from original J5')
def verify_files(root=ROOT,require_bound=True):
 root=Path(root).resolve();m=read_json(root/'source_manifest.json');require(m.get('schema')=='aictrack1.next_cycle10.CG_confirmation.visual.source.v1' and m.get('status') in ('unbound_template','bound'),'exact new confirmation source')
 if require_bound:require(m['status']=='bound','unbound template cannot train')
 for rel,d in m['files'].items():
  p=Path(rel);require(not p.is_absolute() and '..' not in p.parts and (root/p).resolve().is_relative_to(root),'source path escape');require(sha(root/p)==d,'source drift '+rel)
 if require_bound:
  a=assets(root);require(m['arms']==a['arms'],'manifest arms binding')
  require({x for x in m['files'] if x.startswith('configs/experiment/')}=={f'configs/experiment/p1_{arm}.yaml' for arm in a['arms']},'exactly two configs')
  for arm in a['arms']:validate_config(yaml.safe_load((root/f'configs/experiment/p1_{arm}.yaml').read_text(encoding='utf-8'))['experiment'],arm,a)
 return m
def verify_remote_assets(root=ROOT):
 p=assets(root)
 for name,item in p['assets'].items():require(sha(item['path'])==item['sha256'],'actual input drift '+name)
 h=read_json(p['assets']['h1_receipt']['path']);require(h.get('complete') is True and h.get('epoch')==100 and h.get('checkpoint_sha256')==p['assets']['h1']['sha256'] and h.get('source_sha256')==p['source_h1_manifest_sha256'] and h.get('row_binding')==p['h1_row_binding'] and h.get('scope')=='fold_train_fixed_final_epochs' and h.get('validation_selection') is False and h.get('test_access') is False and h.get('actual_cpu_optimizer_scheduler_scaler_model_rng_restored') is True,'actual new H1100 train-only CPU receipt')
 source=read_json(p['assets']['h1_source']['path']);require(source.get('schema')=='aictrack1.next_cycle10.CG_confirmation.h1.source.v1' and source.get('status')=='bound_h1_only' and source.get('query_definition_source_sha256')==QUERY_SOURCE_SHA,'actual new H1 producer source')
 require(source.get('template_manifest_sha256')==p['h1_template_manifest_sha256'] and digest(p['h1_template_manifest_sha256']),'actual reviewed H1 template ancestry')
 require(sha(Path(root)/'inputs/budget.json')==p['budget_sha256'] and sha(Path(root)/'inputs/root_admission.json')==p['admission_sha256'],'same root budget/admission')
 from fresh_route_runtime import budget
 b=budget(root);e=b['capacity_evidence'];require(sha(e['path'])==e['sha256'],'actual expansion evidence drift');v=read_json(e['path']);require(v.get('data_capacity_increase_bytes')==32212254720 and v.get('adjusted_initial_free_bytes')==45226205184 and v.get('original_initial_free_bytes')==13013950464,'same expanded cumulative ledger')
 return p
