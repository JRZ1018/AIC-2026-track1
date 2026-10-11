"""Typed old-fold route admission and exact fresh C+one recipe, never full/test training."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE_ROOT=Path('/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_20261009')
ARMS={'C':0.,'S':0.,'J5':5e-6};ROUTE_ARM={'S':'S','J5_GRAPH':'J5'}
H1_SHA='e88a1cbbaa8d9ebdbf1c7cf2707ac434cf440fac8bd5b4a5dc57893ba0fe15be'
H1_RECEIPT_SHA='06c762c1828915fd579fcc338ade3177f5eff96be742c9650e265845ce6f7d73'
H1_SOURCE_SHA='448a2b2a293cb3a7a51f7bd9b5fad7afde9f2cae196bc951485dedfa98511d2c'
SPLIT_SHA='57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7'
CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MANIFEST_SHA='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
MAPPING_FILE_SHA='0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be'
OLD_SPLIT_SHA='0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'

def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):d.update(block)
    return d.hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def valid_sha(value):return isinstance(value,str) and len(value)==64 and not(set(value)-set('0123456789abcdef'))

def validate_admission(value):
    require(value.get('schema')=='aictrack1.fresh_route.admission.v1' and value.get('status')=='selected_recipe_for_fresh_route_confirmation','typed route selection required')
    route=value.get('selected_route');require(route in ROUTE_ARM and value.get('selected_arm')==ROUTE_ARM[route],'exact route arm')
    require(value.get('test_access') is False and value.get('used_test_or_online_scores') is False and value.get('root_reviewed') is True,'root-reviewed train-only choice required')
    require(value.get('selection_split_sha256')==OLD_SPLIT_SHA and value.get('checkpoint_epoch')==20 and value.get('checkpoint_role')=='last' and value.get('training_scope')=='fixed_split_validation','old completed fixed-last20 required')
    scope='S_P7_vs_C' if route=='S' else 'J5_graph_P7_vs_C'
    require(value.get('primary_scope')==scope and math.isfinite(float(value.get('primary_delta_vs_C_pp',float('nan')))) and value['primary_delta_vs_C_pp']>0,'direct selected route vs C positive agreement required')
    if route=='S':require(value['primary_delta_vs_C_pp']>=.10,'registered S continuation floor .10pp')
    else:
        require(value.get('graph_parameters')=={'k':10,'lambda':.3,'iterations':3,'probability_temperature':1.,'protection_raw_margin':math.log(4)},'only frozen selected graph k10/lambda.3/T1/residual3/ln4')
        # Raw J5 below .10 is reported separately and is never this route gate.
        require(math.isfinite(float(value.get('raw_J5_P7_delta_vs_C_pp',float('nan')))),'raw J5 signal must be disclosed separately')
    for name in ('selected_checkpoint_sha256','selected_config_sha256','selected_training_source_manifest_sha256','selection_protocol_sha256'):
        require(valid_sha(value.get(name)),'route SHA missing '+name)
    require(value['selection_protocol_sha256']=='b60a39fc5e966b45f97a04bae30e5d2b67fe70b83f96c01c1d97799622cd2a02','frozen old-fold primary query/P7 protocol required')
    source='f473415f22fad41f3c8f3cb64164a0b777835661cb03a54723e9a80a25366ca9' if route=='S' else '2fee29fc5e5c282a331883de2b3cd4d5865c28e38a8cead89be247ec95a6c9fe'
    require(value['selected_training_source_manifest_sha256']==source,'original route training source differs')
    names=['checkpoint','config','training_source','paired_report','root_review']
    if route=='J5_GRAPH':names+=['graph_result','combo_report','graph_protocol']
    for name in names:
        item=value.get('evidence',{}).get(name,{})
        require(isinstance(item.get('path'),str) and valid_sha(item.get('sha256')),'actual evidence path/SHA missing '+name)
    for name,key in [('checkpoint','selected_checkpoint_sha256'),('config','selected_config_sha256'),('training_source','selected_training_source_manifest_sha256')]:
        require(value['evidence'][name]['sha256']==value[key],'route alias/evidence mismatch '+name)
    return value

def verify_admission_evidence(value):
    validate_admission(value)
    for name,item in value['evidence'].items():
        require(Path(item['path']).is_absolute() and sha(item['path'])==item['sha256'],'actual immutable route evidence drift '+name)
    review=json.loads(Path(value['evidence']['root_review']['path']).read_text(encoding='utf-8'))
    require(review.get('selected_route')==value['selected_route'] and review.get('decision')=='proceed_to_fresh_confirmation' and isinstance(review.get('reason'),str) and review['reason'].strip(),'positive actual root route review')
    return value

def validate_assets(p):
    require(p.get('schema')=='aictrack1.fresh_route.assets.v1' and p.get('status')=='verified_new_fold_assets','new positive verified assets required')
    require(p.get('competition_stage')=='stage2' and p.get('train_count')==118958 and p.get('validation_count')==29737 and p.get('num_classes')==750 and p.get('training_seed')==20261008,'exact fresh counts/seed')
    require(p.get('selected_route') in ROUTE_ARM and p.get('selected_arm')==ROUTE_ARM[p['selected_route']],'single selected route only')
    require(p.get('canonical_lock')=='/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json','original canonical lock required')
    expected={'h1':H1_SHA,'h1_receipt':H1_RECEIPT_SHA,'h1_source':H1_SOURCE_SHA,'split':SPLIT_SHA,'clip':CLIP_SHA,'manifest':MANIFEST_SHA,'class_mapping':MAPPING_FILE_SHA,'label_conflicts':'b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a'}
    for name,digest in expected.items():require(p.get('assets',{}).get(name,{}).get('sha256')==digest,'actual fresh input pin '+name)
    for name,item in p['assets'].items():require(isinstance(item.get('path'),str) and item['path'].startswith('/') and valid_sha(item.get('sha256')),'absolute verified input '+name)
    require(p.get('source_h1_manifest_sha256')==H1_SOURCE_SHA and p.get('h1_training_scope')=='fold_train_fixed_final_epochs' and p.get('h1_epoch')==100 and p.get('h1_training_count')==118958 and p.get('h1_validation_used_for_selection') is False,'positive train-only new H1100')
    require(p.get('h1_row_binding',{}).get('counts')=={'train':118958,'validation':29737} and p['h1_row_binding'].get('validation_used_for_optimization_or_selection') is False,'H1 positive exact row binding')
    actual=json.loads((ROOT/'inputs/h1_actual_CPU_complete.json').read_text(encoding='utf-8'))
    require(p['h1_row_binding']==actual['producer_receipt']['row_binding'],'exact accepted new H1 row hashes required')
    require(p.get('old_query_overlap')==0 and p.get('train_query_overlap')==0 and valid_sha(p.get('admission_receipt_sha256')),'new independent query/admission binding')
    require(math.isfinite(float(p.get('terminal_cache_parity_top1',float('nan')))),'actual new H1 parity')
    return p

def assets(root=ROOT):return validate_assets(json.loads((Path(root)/'inputs/p1_assets.json').read_text(encoding='utf-8')))
def verify_files(root=ROOT):
    root=Path(root).resolve();v=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    require(v.get('schema')=='aictrack1.fresh_route.training.source.v1' and v.get('status')=='bound','actual typed bound fresh source required')
    for rel,digest in v['files'].items():
        p=Path(rel);require(not p.is_absolute() and '..' not in p.parts and (root/p).resolve().is_relative_to(root),'source path escape')
        require(sha(root/p)==digest,'fresh source drift '+rel)
    p=assets(root);require(v.get('selected_route')==p['selected_route'] and v.get('selected_arm')==p['selected_arm'] and v.get('admission_receipt_sha256')==p['admission_receipt_sha256'],'source/assets typed route aliases')
    configs={f'configs/experiment/p1_{x}.yaml' for x in ('C',p['selected_arm'])}
    require({x for x in v['files'] if x.startswith('configs/experiment/')}==configs,'only C plus one candidate config allowed')
    return v
def verify_remote_assets(root=ROOT):
    p=assets(root)
    for name,item in p['assets'].items():require(sha(item['path'])==item['sha256'],'actual fresh input drift '+name)
    admission=Path(root)/'inputs/admission_receipt.json';require(sha(admission)==p['admission_receipt_sha256'],'frozen admission hash mismatch')
    verify_admission_evidence(json.loads(admission.read_text(encoding='utf-8')))
    h1=json.loads(Path(p['assets']['h1_receipt']['path']).read_text())
    require(h1.get('complete') is True and h1.get('checkpoint_sha256')==H1_SHA and h1.get('source_sha256')==H1_SOURCE_SHA and h1.get('row_binding')==p['h1_row_binding'] and h1.get('test_access') is False,'actual H1 producer receipt')
    return p

def expected_config(arm,p,base=None):
    import copy
    require(arm in {'C',p['selected_arm']},'inactive recipe forbidden')
    raw=copy.deepcopy(base or json.loads((ROOT/'inputs/base_C_config.json').read_text()))
    raw.update(run_id=f'20261009-fresh-route-{arm}-fixed20',output_dir=f'../../artifacts/experiments/FRESH_{arm}',
        manifest=p['assets']['manifest']['path'],split=p['assets']['split']['path'],label_conflicts=p['assets']['label_conflicts']['path'],head_checkpoint=p['assets']['h1']['path'])
    raw['base_loss']['alpha']=.5 if arm=='S' else .1;raw['training']['head_learning_rate']=ARMS[arm]
    return raw
def validate_config(raw,arm,p):require(raw==expected_config(arm,p),'fresh single-variable recipe/policy drift')
