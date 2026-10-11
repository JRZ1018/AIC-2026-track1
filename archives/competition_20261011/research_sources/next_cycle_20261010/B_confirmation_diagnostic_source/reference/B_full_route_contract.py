"""B-only independent confirmation, using established FULL provider interfaces."""
from __future__ import annotations
import copy,hashlib,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE_ROOT='/root/autodl-tmp/AICtrack1_next_cycle_B_full_20261010'
ARMS={'B25':.25,'B50':.5};ROUTES={k+'_GRAPH':k for k in ARMS}
CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MANIFEST_SHA='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
FULL_H1_SHA='fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605'
FULL_H1_RECEIPT_SHA='0e46cf54918bc152af34f7b3edb6b84fe7a9c3d2211bc3925f8b994a1dc144cf'
SPLIT_SHA='610111665e1055bcda7157add15b16ade353138365e54ee87dc92316e6dfd6f1'
QUERY_SOURCE_SHA='51a79ffbaf94fbb333cdab49123e55ccc3fb5cf2fe78cfdb6b5ea86da8810e50'
QUERY_IDENTITY_SHA='da90d50fca08e619ab577311059fad778d09960e23ddd8bcdb1b6ec6d4569663'
MATH_SOURCE_SHA='0096f4c8352e28eae4373485c45e4e6e4f54dfa750ae4b4ddeca8bd8a068df18'
MATH_PROTOCOL_SHA='3c273802ce085f2d6e629743e0b120df13bd1761d29f9e91c4ba83908f317f19'
CONFIRMATION_TEMPLATE_SHA='a7ff1626270d955ae97b30f9a62eed9f7888d030c12ecf3a461fafc3d9f430cb'
INITIAL_FREE_BYTES=13013950464
PRIMARY_SCOPE='candidate_fixed_graph_P7_vs_fresh_J5_GRAPH_P7'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def valid_digest(v):return isinstance(v,str) and len(v)==64 and not set(v)-set('0123456789abcdef')
def validate_confirmation(v):
 require(v.get('schema')=='aictrack1.next_cycle10.B_confirmation.v1' and v.get('status')=='complete_positive_independent_B_confirmation','actual independent B confirmation only; old alpha/J5/exploratory receipts invalid')
 arm=v.get('selected_arm');require(arm in ARMS and v.get('selected_route')==arm+'_GRAPH' and v.get('primary_scope')==PRIMARY_SCOPE,'registered B combo versus fresh J5 combo')
 for key,want in [('split_sha256',SPLIT_SHA),('query_definition_source_sha256',QUERY_SOURCE_SHA),('query_identity_sha256',QUERY_IDENTITY_SHA),('training_seed',20261010),('checkpoint_epoch',20),('checkpoint_role','last'),('selection_rule','fixed_final20'),('train_count',118958),('natural_count',29737),('balanced_count',14760),('test_access',False),('used_test_or_online_scores',False),('root_risk_reviewed',True),('final_model','student_only'),('math_source_sha256',MATH_SOURCE_SHA),('math_protocol_sha256',MATH_PROTOCOL_SHA)]:require(v.get(key)==want,'actual B confirmation '+key)
 require(v.get('soft_coefficient')==ARMS[arm] and math.isfinite(float(v.get('primary_delta_pp',float('nan')))) and v['primary_delta_pp']>0,'actual positive B combo required')
 require(math.isfinite(float(v.get('natural_delta_pp',float('nan')))),'finite natural diagnostic, no positivity threshold')
 for key in ['training_source_manifest_sha256','diagnostic_source_manifest_sha256','new_H1_checkpoint_sha256','new_H1_source_sha256','control_last_sha256','candidate_last_sha256']:require(valid_digest(v.get(key)),'actual B SHA '+key)
 require(v['new_H1_checkpoint_sha256'] not in ['e88a1cbbaa8d9ebdbf1c7cf2707ac434cf440fac8bd5b4a5dc57893ba0fe15be',FULL_H1_SHA],'new-query train-only H1 cannot be old/full')
 require(isinstance(v.get('evidence'),dict),'actual immutable B confirmation evidence');return v
def verify_confirmation(v):
 validate_confirmation(v);e=v['evidence'];required={'paired_report','root_review','training_source','control_audit','candidate_audit'};require(required<=set(e),'actual independent report/review/source/two audits required')
 loaded={}
 for key,item in e.items():
  require(isinstance(item,dict) and Path(item.get('path','')).is_absolute() and valid_digest(item.get('sha256')) and sha(item['path'])==item['sha256'],'actual confirmation evidence drift '+key)
  if key in required:loaded[key]=read(item['path'])
 train=loaded['training_source'];require(e['training_source']['sha256']==v['training_source_manifest_sha256'] and train.get('schema')=='aictrack1.next_cycle10.confirmation.visual.source.v1' and train.get('status')=='bound' and train.get('template_manifest_sha256')==CONFIRMATION_TEMPLATE_SHA and train.get('arms')==['J5',v['selected_arm']] and train.get('training_seed')==20261010 and train.get('h1_checkpoint_sha256')==v['new_H1_checkpoint_sha256'] and train.get('source_h1_manifest_sha256')==v['new_H1_source_sha256'] and train.get('query_definition_source_sha256')==QUERY_SOURCE_SHA,'actual reviewed new610 source/head/seed only')
 require(train.get('training_scope')=='fixed_split_validation' and train.get('fixed_epochs')==20 and train.get('checkpoint_role')=='last','actual reviewed fixed20 B source math')
 folder=Path(e['training_source']['path']).parent.resolve();files=train.get('files',{});require(isinstance(files,dict) and files.get('src/aictrack1/methods/josrc_soft.py')==MATH_SOURCE_SHA and files.get('inputs/soft_math_protocol.json')==MATH_PROTOCOL_SHA,'actual B source lists exact science files')
 for rel,d in files.items():
  part=Path(rel);require(not part.is_absolute() and '..' not in part.parts and valid_digest(d) and (folder/part).resolve().is_relative_to(folder) and sha(folder/part)==d,'actual confirmation science/source closure '+rel)
 report=loaded['paired_report'];require(report.get('selected_arm')==v['selected_arm'] and report.get('primary_scope')==PRIMARY_SCOPE and report.get('primary_delta_pp')==v['primary_delta_pp'] and report.get('natural_delta_pp')==v['natural_delta_pp'] and report.get('split_sha256')==SPLIT_SHA and report.get('query_identity_sha256')==QUERY_IDENTITY_SHA and report.get('training_source_manifest_sha256')==v['training_source_manifest_sha256'] and report.get('diagnostic_source_manifest_sha256')==v['diagnostic_source_manifest_sha256'] and report.get('raw_P7_separately_reported') is True,'actual raw-versus-combo report identities')
 review=loaded['root_review'];require(review.get('root_authorized') is True and review.get('decision')=='allow_single_B_full_training' and review.get('selected_arm')==v['selected_arm'] and review.get('paired_report_sha256')==e['paired_report']['sha256'],'actual root risk review bound to this report')
 for key,arm,checkpoint in [('control_audit','J5',v['control_last_sha256']),('candidate_audit',v['selected_arm'],v['candidate_last_sha256'])]:
  a=loaded[key];require(a.get('complete') is True and a.get('arm')==arm and a.get('epoch')==20 and a.get('last_sha256')==checkpoint and a.get('source_manifest_sha256')==v['training_source_manifest_sha256'] and a.get('h1_checkpoint_sha256')==v['new_H1_checkpoint_sha256'] and a.get('split_sha256')==SPLIT_SHA and a.get('train_count')==118958 and a.get('validation_count')==29737 and a.get('training_seed')==20261010 and a.get('actual_runtime')=={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','actual_torch_threads':4} and a.get('student_model_CPU_restored') is True,'actual matched fixed20/newH1/runtime '+key)
  if key=='candidate_audit':require(a.get('josrc_soft_state_complete') is True and a.get('final_model')=='student_only' and a.get('math_source_sha256')==MATH_SOURCE_SHA and a.get('math_protocol_sha256')==MATH_PROTOCOL_SHA,'actual complete B teacher/support restoration')
 return v
def selection(path,expected_sha):require(valid_digest(expected_sha) and sha(path)==expected_sha,'actual B typed confirmation SHA');return validate_confirmation(read(path))
def verify_selection_evidence(v):return verify_confirmation(v)
def validate_budget(v,*,now=None,enforce_active=True):
 require(v.get('schema')=='aictrack1.next_cycle10.B_full.budget.v1' and v.get('root_authorized') is True,'new explicit B FULL budget only')
 from capacity_contract import validate_capacity
 validate_capacity(v)
 require(type(v.get('disk_max_bytes')) is int and v['disk_max_bytes']>0 and type(v.get('min_free_bytes')) is int and v['min_free_bytes']>=3*1024**3,'concrete cumulative disk/min3')
 start=v.get('phase_start_unix');end=v.get('deadline_unix');require(isinstance(start,(int,float)) and isinstance(end,(int,float)) and math.isfinite(start) and math.isfinite(end) and 0<start<end,'new actual FULL phase start/deadline')
 if enforce_active:require(start<=(time.time() if now is None else now)<end,'original immutable B FULL window')
 return v

def validate_assets(p):
 require(p.get('schema')=='aictrack1.fresh_route.full.assets.v1' and p.get('method_family')=='JoSRC_inspired_B' and p.get('status')=='pinned_legal_full_assets','B-only legal full assets')
 require(p.get('competition_stage')=='stage2' and p.get('train_count')==148695 and p.get('validation_count')==0 and p.get('num_classes')==750 and p.get('training_seed')==20261010,'fresh official full148695/validation0 seed20261010')
 require(p.get('selected_arm') in ARMS and p.get('selected_route')==p['selected_arm']+'_GRAPH','one actually confirmed B only')
 ref=read(ROOT/'inputs/legal_full_asset_reference.json');require(p.get('assets')==ref['assets'] and p.get('canonical_lock')==ref['canonical_lock'] and p.get('dataset_root')==ref['dataset_root'],'legal pinned full asset bytes/paths')
 require(p.get('h1_training_scope')=='full_train_fixed_epochs' and p.get('h1_epoch')==100 and p.get('h1_training_count')==148695 and p.get('h1_validation_used_for_selection') is False,'full100 official H1 only')
 require(p.get('h1_row_binding')==read(ROOT/'inputs/legal_full_h1_cpu_acceptance.json')['row_binding'],'accepted full H1 exact row binding')
 for key in ['fresh_confirmation_receipt_sha256','full_budget_sha256']:require(valid_digest(p.get(key)),'actual B full input SHA '+key)
 return p
def assets(root=ROOT):return validate_assets(read(Path(root)/'inputs/full_assets.json'))
def verify_files(root=ROOT):
 root=Path(root).resolve();m=read(root/'source_manifest.json');require(m.get('schema')=='aictrack1.fresh_route.full.training.source.v1' and m.get('method_family')=='JoSRC_inspired_B' and m.get('status')=='bound','actual B bound full source only')
 for rel,d in m['files'].items():
  p=Path(rel);require(not p.is_absolute() and '..' not in p.parts and (root/p).resolve().is_relative_to(root) and sha(root/p)==d,'B source byte/path drift '+rel)
 p=assets(root)
 for k in ['selected_route','selected_arm','fresh_confirmation_receipt_sha256','full_budget_sha256']:require(m.get(k)==p[k],'B full source/assets alias '+k)
 require({k for k in m['files'] if k.startswith('configs/experiment/')}=={f'configs/experiment/p1_full_{p["selected_arm"]}.yaml'},'one selected full B config only');return m
def verify_remote_assets(root=ROOT):
 root=Path(root);p=assets(root)
 for key,item in p['assets'].items():require(sha(item['path'])==item['sha256'],'actual full asset byte drift '+key)
 receipt=read(p['assets']['h1_receipt']['path']);require(receipt.get('checkpoint_sha256')==FULL_H1_SHA and receipt.get('scope')=='full_train_fixed_epochs' and receipt.get('epoch')==100 and receipt.get('complete') is True and receipt.get('row_binding')==p['h1_row_binding'] and receipt.get('validation_selection') is False and receipt.get('test_access') is False,'actual legal full H1100')
 for rel,key in [('inputs/fresh_confirmation.json','fresh_confirmation_receipt_sha256'),('inputs/full_budget.json','full_budget_sha256')]:require(sha(root/rel)==p[key],'actual B full bound input '+key)
 c=verify_confirmation(read(root/'inputs/fresh_confirmation.json'));require(c['selected_arm']==p['selected_arm'],'B FULL differs from real confirmed candidate')
 validate_budget(read(root/'inputs/full_budget.json'),enforce_active=False);return p
def expected_config(arm,p,base=None):
 import yaml
 from aictrack1.methods.josrc_soft import recipe
 require(arm==p['selected_arm'] and arm in ARMS,'one selected B FULL recipe')
 raw=copy.deepcopy(base or yaml.safe_load((ROOT/'inputs/baseline_J5.yaml').read_text(encoding='utf-8'))['experiment'])
 raw.update(run_id=f'20261010-B-full-{arm}-fixed20',output_dir=f'../../artifacts/experiments/FULL_{arm}',training_scope='full_train_fixed_epochs',expected_train_count=148695,manifest=p['assets']['manifest']['path'],split=p['assets']['split']['path'],label_conflicts=p['assets']['label_conflicts']['path'],head_checkpoint=p['assets']['h1']['path'])
 raw['training']['seed']=20261010;raw['josrc_soft']=recipe(ARMS[arm]);return raw
def validate_config(raw,arm,p):require(raw==expected_config(arm,p),'original J5 hard/AC/336/optimizer/cos20 plus one exact B recipe only')
