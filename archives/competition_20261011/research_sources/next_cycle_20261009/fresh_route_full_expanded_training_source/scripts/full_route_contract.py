"""Single typed, positively confirmed route; legal full148695 initializer and fixed20."""
from __future__ import annotations
import copy,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE_ROOT='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_full_20261009'
ARMS={'S':0.,'J5':5e-6};ROUTES={'S':'S','J5_GRAPH':'J5'}
CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MANIFEST_SHA='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
FULL_H1_SHA='fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605'
FULL_H1_RECEIPT_SHA='0e46cf54918bc152af34f7b3edb6b84fe7a9c3d2211bc3925f8b994a1dc144cf'
FRESH_SPLIT_SHA='57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7'
FRESH_H1_SHA='e88a1cbbaa8d9ebdbf1c7cf2707ac434cf440fac8bd5b4a5dc57893ba0fe15be'
FRESH_H1_RECEIPT_SHA='06c762c1828915fd579fcc338ade3177f5eff96be742c9650e265845ce6f7d73'
FRESH_H1_SOURCE_SHA='448a2b2a293cb3a7a51f7bd9b5fad7afde9f2cae196bc951485dedfa98511d2c'
TRAINING_TEMPLATE_SHA='05df2bbd14af808847a430bb4a23ba22483774dad00e6724026ea37dd633bf88'
INITIAL_FREE_BYTES=13013950464

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def valid_digest(v):return isinstance(v,str) and len(v)==64 and not(set(v)-set('0123456789abcdef'))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def validate_confirmation(v):
    require(v.get('schema')=='aictrack1.fresh_route.confirmation.v1' and v.get('status')=='complete_positive_fresh_route_confirmation','new typed fresh positive receipt required; old naked-J gate is invalid')
    route=v.get('selected_route');require(route in ROUTES and v.get('selected_arm')==ROUTES[route],'one supported selected recipe')
    require(v.get('primary_scope')==('S_P7_vs_C' if route=='S' else 'J5_graph_P7_vs_C'),'direct selected route versus fresh C only')
    require(v.get('checkpoint_epoch')==20 and v.get('selection_rule')=='fixed_final20' and v.get('training_scope')=='fixed_split_validation','complete fresh fixed-last20 evidence')
    for key,value in [('split_sha256',FRESH_SPLIT_SHA),('h1_sha256',FRESH_H1_SHA),('source_h1_manifest_sha256',FRESH_H1_SOURCE_SHA),('test_access',False),('used_test_or_online_scores',False),('root_risk_reviewed',True)]:require(v.get(key)==value,'positive fresh confirmation '+key)
    require(math.isfinite(float(v.get('primary_delta_vs_C_pp',float('nan')))) and v['primary_delta_vs_C_pp']>0,'positive new primary direction')
    require(math.isfinite(float(v.get('natural_delta_vs_C_pp',float('nan')))),'finite natural diagnostic, no natural positivity gate')
    for name in ('training_source_manifest_sha256','diagnostic_source_manifest_sha256','admission_receipt_sha256'):require(valid_digest(v.get(name)),'actual confirmation SHA '+name)
    require(isinstance(v.get('evidence'),dict),'actual immutable confirmation evidence required')
    return v

def verify_confirmation(v):
    validate_confirmation(v)
    from fresh_confirmation_reference import build_confirmation
    e=v['evidence'];report=read(e['paired_report']['path'])
    if report.get('append_stage_context') is not None:
        require(v['selected_route']=='J5_GRAPH','retry append selects only J5_GRAPH')
        required={'original_fresh_budget','appended_stage_budget','continuation_operator_source','partial_restore_audit','terminal_restore_audit','continuation_completion','original_shared_budget','disk_expansion_recovery','prior_failure'}
        require(required<=set(e),'all nine extra actual retry evidence refs retained')
        from append_diagnostic_contract import validate_append_budget
        from append_confirmation_reference import validate_append_evidence
        context=validate_append_budget(e['appended_stage_budget']['path'],e['appended_stage_budget']['sha256'],live=False)
        validate_append_evidence(report,e,context)
    rebuilt=build_confirmation(report,e,route=v['selected_route'],report_sha=e['paired_report']['sha256'],source_sha=v['diagnostic_source_manifest_sha256'])
    for key,value in rebuilt.items():require(v.get(key)==value,'typed confirmation actual evidence/metric mismatch '+key)
    return v

def selection(path,expected_sha):
    require(valid_digest(expected_sha) and sha(path)==expected_sha,'actual typed confirmation receipt SHA mismatch')
    return validate_confirmation(read(path))
def verify_selection_evidence(v):return verify_confirmation(v)

def validate_budget(v,*,now=None,enforce_active=True):
    import time
    require(v.get('schema')=='aictrack1.fresh_route.full.budget.v1' and v.get('root_authorized') is True,'explicit new full phase budget required')
    from capacity_contract import validate_capacity
    validate_capacity(v)
    require(type(v.get('disk_max_bytes')) is int and v['disk_max_bytes']>0 and type(v.get('min_free_bytes')) is int and v['min_free_bytes']>=3*1024**3,'positive concrete disk budget and >=3GiB reserve')
    start=v.get('phase_start_unix');deadline=v.get('deadline_unix')
    require(isinstance(start,(int,float)) and isinstance(deadline,(int,float)) and math.isfinite(start) and math.isfinite(deadline) and start>0 and deadline>start,'explicit concrete full start/deadline')
    current=time.time() if now is None else now
    if enforce_active:require(start<=current<deadline,'full phase has not started or already expired; never extend on resume')
    return v

def validate_assets(p):
    require(p.get('schema')=='aictrack1.fresh_route.full.assets.v1' and p.get('status')=='pinned_legal_full_assets','positive legal full assets')
    require(p.get('competition_stage')=='stage2' and p.get('train_count')==148695 and p.get('validation_count')==0 and p.get('num_classes')==750 and p.get('training_seed')==20261008,'full official148695/0 scope only')
    require(p.get('selected_route') in ROUTES and p.get('selected_arm')==ROUTES[p['selected_route']],'single selected full route')
    require(p.get('h1_training_scope')=='full_train_fixed_epochs' and p.get('h1_epoch')==100 and p.get('h1_training_count')==148695 and p.get('h1_validation_used_for_selection') is False,'legal full100 H1 only')
    reference=read(ROOT/'inputs/legal_full_asset_reference.json')
    require(p.get('assets')==reference['assets'] and p.get('canonical_lock')==reference['canonical_lock'] and p.get('dataset_root')==reference['dataset_root'],'verified677 full asset paths/SHAs/canonical lock')
    require(p.get('h1_row_binding')==read(ROOT/'inputs/legal_full_h1_cpu_acceptance.json')['row_binding'],'actual accepted full H1 positive row binding')
    for key in ('fresh_confirmation_receipt_sha256','full_budget_sha256'):require(valid_digest(p.get(key)),'full bound input SHA '+key)
    return p

def assets(root=ROOT):return validate_assets(read(Path(root)/'inputs/full_assets.json'))
def verify_files(root=ROOT):
    root=Path(root).resolve();m=read(root/'source_manifest.json')
    require(m.get('schema')=='aictrack1.fresh_route.full.training.source.v1' and m.get('status')=='bound','full template cannot train; positive bound source required')
    for rel,digest in m['files'].items():
        p=Path(rel);require(not p.is_absolute() and '..' not in p.parts and (root/p).resolve().is_relative_to(root) and sha(root/p)==digest,'full source path/drift '+rel)
    p=assets(root)
    for key in ('selected_route','selected_arm','fresh_confirmation_receipt_sha256','full_budget_sha256'):require(m.get(key)==p.get(key),'full source/assets alias '+key)
    require({r for r in m['files'] if r.startswith('configs/experiment/')}=={f'configs/experiment/p1_full_{p["selected_arm"]}.yaml'},'exactly one selected full config')
    return m

def verify_remote_assets(root=ROOT):
    root=Path(root);p=assets(root)
    for name,item in p['assets'].items():require(sha(item['path'])==item['sha256'],'actual full asset drift '+name)
    receipt=read(p['assets']['h1_receipt']['path'])
    require(receipt.get('checkpoint_sha256')==FULL_H1_SHA and receipt.get('scope')=='full_train_fixed_epochs' and receipt.get('epoch')==100 and receipt.get('complete') is True and receipt.get('row_binding')==p['h1_row_binding'] and receipt.get('validation_selection') is False and receipt.get('test_access') is False,'actual legal full H1 producer receipt')
    for rel,key in [('inputs/fresh_confirmation.json','fresh_confirmation_receipt_sha256'),('inputs/full_budget.json','full_budget_sha256')]:require(sha(root/rel)==p[key],'actual full bound input drift '+rel)
    choice=verify_confirmation(read(root/'inputs/fresh_confirmation.json'));require(choice['selected_route']==p['selected_route'],'full recipe differs from actual confirmed route')
    # Read-only acceptance/inference may occur after the full phase ended.
    # Actual prepare, launcher bind_budget and trainer entry enforce the active deadline.
    validate_budget(read(root/'inputs/full_budget.json'),enforce_active=False)
    return p

def expected_config(arm,p,base=None):
    require(arm==p['selected_arm'],'only one confirmed full arm')
    raw=copy.deepcopy(base or read(ROOT/'inputs/base_C_config.json'))
    raw.update(run_id=f'20261009-fresh-route-full-{arm}-fixed20',output_dir=f'../../artifacts/experiments/FULL_{arm}',training_scope='full_train_fixed_epochs',expected_train_count=148695,
        manifest=p['assets']['manifest']['path'],split=p['assets']['split']['path'],label_conflicts=p['assets']['label_conflicts']['path'],head_checkpoint=p['assets']['h1']['path'])
    raw['base_loss']['alpha']=.5 if arm=='S' else .1;raw['training']['head_learning_rate']=ARMS[arm]
    return raw
def validate_config(raw,arm,p):require(raw==expected_config(arm,p),'full fixed20 single confirmed recipe drift')
