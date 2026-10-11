"""L50 independent confirmation, using established FULL provider interfaces."""
from __future__ import annotations
import copy,hashlib,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE_ROOT='/root/autodl-tmp/AICtrack1_next_cycle_L50_full_20261011'
ARMS={'L50':5e-6};ROUTES={k+'_GRAPH':k for k in ARMS}
CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MANIFEST_SHA='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
FULL_H1_SHA='fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605'
FULL_H1_RECEIPT_SHA='0e46cf54918bc152af34f7b3edb6b84fe7a9c3d2211bc3925f8b994a1dc144cf'
SPLIT_SHA='610111665e1055bcda7157add15b16ade353138365e54ee87dc92316e6dfd6f1'
QUERY_SOURCE_SHA='51a79ffbaf94fbb333cdab49123e55ccc3fb5cf2fe78cfdb6b5ea86da8810e50'
QUERY_IDENTITY_SHA='da90d50fca08e619ab577311059fad778d09960e23ddd8bcdb1b6ec6d4569663'
CONFIRMATION_TEMPLATE_SHA='12a5487243143270a0cdc2b1fe1ebfae268ab175956f1b0fd5f2a839ca97cf1d'
INITIAL_FREE_BYTES=13013950464
PRIMARY_SCOPE='candidate_fixed_graph_P7_vs_fresh_J5_GRAPH_P7'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def valid_digest(v):return isinstance(v,str) and len(v)==64 and not set(v)-set('0123456789abcdef')
def validate_confirmation(v):
 from L50_two_reused_admission import validate
 return validate(v)
def verify_confirmation(v):
 from L50_two_reused_admission import verify
 return verify(v)

def selection(path,expected_sha):require(valid_digest(expected_sha) and sha(path)==expected_sha,'actual L50 typed confirmation SHA');return validate_confirmation(read(path))
def verify_selection_evidence(v):return verify_confirmation(v)
def validate_budget(v,*,now=None,enforce_active=True):
 require(v.get('schema')=='aictrack1.next_cycle10.L50_full.budget.v1' and v.get('root_authorized') is True,'new explicit L50 FULL budget only')
 from capacity_contract import validate_capacity
 validate_capacity(v)
 require(type(v.get('disk_max_bytes')) is int and 0<v['disk_max_bytes']<=40*1024**3 and type(v.get('min_free_bytes')) is int and v['min_free_bytes']>=3*1024**3,'concrete cumulative disk/min3')
 start=v.get('phase_start_unix');end=v.get('deadline_unix');require(isinstance(start,(int,float)) and isinstance(end,(int,float)) and math.isfinite(start) and math.isfinite(end) and 0<start<end,'new actual FULL phase start/deadline')
 if enforce_active:require(start<=(time.time() if now is None else now)<end,'original immutable L50 FULL window')
 return v

def validate_assets(p):
 require(p.get('schema')=='aictrack1.fresh_route.full.assets.v1' and p.get('method_family')=='single_visual_learning_rate_point' and p.get('status')=='pinned_legal_full_assets','L50-only legal full assets')
 require(p.get('competition_stage')=='stage2' and p.get('train_count')==148695 and p.get('validation_count')==0 and p.get('num_classes')==750 and p.get('training_seed')==20261010,'fresh official full148695/validation0 seed20261010')
 require(p.get('selected_arm') in ARMS and p.get('selected_route')==p['selected_arm']+'_GRAPH','one actually confirmed L50 only')
 ref=read(ROOT/'inputs/legal_full_asset_reference.json');require(p.get('assets')==ref['assets'] and p.get('canonical_lock')==ref['canonical_lock'] and p.get('dataset_root')==ref['dataset_root'],'legal pinned full asset bytes/paths')
 require(p.get('h1_training_scope')=='full_train_fixed_epochs' and p.get('h1_epoch')==100 and p.get('h1_training_count')==148695 and p.get('h1_validation_used_for_selection') is False,'full100 official H1 only')
 require(p.get('h1_row_binding')==read(ROOT/'inputs/legal_full_h1_cpu_acceptance.json')['row_binding'],'accepted full H1 exact row binding')
 for key in ['fresh_confirmation_receipt_sha256','full_budget_sha256']:require(valid_digest(p.get(key)),'actual L50 full input SHA '+key)
 return p
def assets(root=ROOT):return validate_assets(read(Path(root)/'inputs/full_assets.json'))
def verify_files(root=ROOT):
 root=Path(root).resolve();m=read(root/'source_manifest.json');require(m.get('schema')=='aictrack1.fresh_route.full.training.source.v1' and m.get('method_family')=='single_visual_learning_rate_point' and m.get('status')=='bound','actual L50 bound full source only')
 for rel,d in m['files'].items():
  p=Path(rel);require(not p.is_absolute() and '..' not in p.parts and (root/p).resolve().is_relative_to(root) and sha(root/p)==d,'B source byte/path drift '+rel)
 p=assets(root)
 for k in ['selected_route','selected_arm','fresh_confirmation_receipt_sha256','full_budget_sha256']:require(m.get(k)==p[k],'B full source/assets alias '+k)
 require({k for k in m['files'] if k.startswith('configs/experiment/')}=={f'configs/experiment/p1_full_{p["selected_arm"]}.yaml'},'one selected full L50 config only');return m
def verify_remote_assets(root=ROOT):
 root=Path(root);p=assets(root)
 for key,item in p['assets'].items():require(sha(item['path'])==item['sha256'],'actual full asset byte drift '+key)
 receipt=read(p['assets']['h1_receipt']['path']);require(receipt.get('checkpoint_sha256')==FULL_H1_SHA and receipt.get('scope')=='full_train_fixed_epochs' and receipt.get('epoch')==100 and receipt.get('complete') is True and receipt.get('row_binding')==p['h1_row_binding'] and receipt.get('validation_selection') is False and receipt.get('test_access') is False,'actual legal full H1100')
 for rel,key in [('inputs/fresh_confirmation.json','fresh_confirmation_receipt_sha256'),('inputs/full_budget.json','full_budget_sha256')]:require(sha(root/rel)==p[key],'actual L50 full bound input '+key)
 c=verify_confirmation(read(root/'inputs/fresh_confirmation.json'));require(c['selected_arm']==p['selected_arm'],'B FULL differs from real confirmed candidate')
 validate_budget(read(root/'inputs/full_budget.json'),enforce_active=False);return p
def expected_config(arm,p,base=None):
 import yaml
 require(arm==p['selected_arm'] and arm in ARMS,'one selected L50 FULL recipe')
 raw=copy.deepcopy(base or yaml.safe_load((ROOT/'inputs/baseline_J5.yaml').read_text(encoding='utf-8'))['experiment'])
 raw.update(run_id=f'20261011-L50-full-{arm}-fixed20',output_dir=f'../../artifacts/experiments/FULL_{arm}',training_scope='full_train_fixed_epochs',expected_train_count=148695,manifest=p['assets']['manifest']['path'],split=p['assets']['split']['path'],label_conflicts=p['assets']['label_conflicts']['path'],head_checkpoint=p['assets']['h1']['path'])
 raw['training']['seed']=20261010;raw['training']['visual_learning_rate']=5e-6;raw['training']['head_learning_rate']=5e-6;return raw
def validate_config(raw,arm,p):require(raw==expected_config(arm,p),'original J5 hard/AC/336/optimizer/cos20 plus one exact L50 recipe only')
