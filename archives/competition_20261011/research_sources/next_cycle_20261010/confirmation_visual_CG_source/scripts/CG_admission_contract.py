"""Actual CG25 complete graph/P7 versus J5_GRAPH admits new610 confirmation; no FULL/test gate."""
import hashlib,json,math,re
from pathlib import Path
QUERY_SOURCE_SHA='51a79ffbaf94fbb333cdab49123e55ccc3fb5cf2fe78cfdb6b5ea86da8810e50'
EXPLORATION_SOURCE_SHA='03381421c2398e6d4d3d510a8be7bafc46e6b8a0d70515f7bf813ccf9e93e605'
EXPLORATION_TEMPLATE_SHA='7175b378a3ff67c907bea49de75cab3429f4f8aa63b072ba562f3bfe527c65d5'
PRIMARY_SCOPE='candidate_fixed_graph_P7_vs_actual_J5_GRAPH_P7'
ROOT=Path(__file__).resolve().parents[1]
AUGMENTATION=json.loads((ROOT/'inputs/CG_exploration_augmentation.json').read_text(encoding='utf-8'))
def require(value,message):
 if not value:raise ValueError(message)
def digest(v):return isinstance(v,str) and bool(re.fullmatch('[0-9a-f]{64}',v))
def evidence(ref,expected):
 require(isinstance(ref,dict) and ref.get('sha256')==expected and digest(expected) and isinstance(ref.get('path'),str) and Path(ref['path']).is_absolute(),'actual absolute evidence path/SHA required')
 p=Path(ref['path']);require(p.is_file() and p.stat().st_size<=32*1024**2,'bounded actual evidence JSON')
 with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 require(actual==expected,'actual evidence file drift')
 return json.loads(p.read_text(encoding='utf-8'))
def validate_CG_admission(a):
 require(a.get('schema')=='aictrack1.next_cycle10.CG_admission.v1' and a.get('root_authorized') is True and a.get('complete') is True,'actual CG-only root admission required')
 require(a.get('selected_arm')=='CG25' and a.get('center_guard_fraction')==.25 and a.get('hard_alpha')==.1,'one actual CG25 original hard recipe')
 delta=a.get('exploration_balanced_delta_pp');require(isinstance(delta,(float,int)) and not isinstance(delta,bool) and math.isfinite(delta) and delta>0,'positive actual CG graph combination required')
 require(a.get('primary_scope')==PRIMARY_SCOPE,'raw CG cannot stand in for CG graph versus actual J5_GRAPH')
 require(a.get('exploration_training_source_sha256')==EXPLORATION_SOURCE_SHA and a.get('exploration_training_template_sha256')==EXPLORATION_TEMPLATE_SHA and a.get('center_guard_augmentation')==AUGMENTATION,'actual reviewed CG source and exact augmentation identity required')
 for k in ('exploration_paired_sha256','root_risk_review_sha256','exploration_diagnostic_source_sha256'):require(digest(a.get(k)),'actual CG evidence digest '+k)
 require(a.get('query_definition_source_sha256')==QUERY_SOURCE_SHA and a.get('new_H1_checkpoint_sha256') is None and a.get('test_access') is False,'pre-result new query/future H1 null/no test')
 paired=evidence(a.get('exploration_paired'),a['exploration_paired_sha256']);review=evidence(a.get('root_risk_review'),a['root_risk_review_sha256'])
 require(paired.get('schema')=='aictrack1.center_guard_joint.exploration_paired_graph.v1' and paired.get('complete') is True and paired.get('arm')=='CG25' and paired.get('primary_scope')==PRIMARY_SCOPE and paired.get('primary_delta_pp')==delta,'actual combo report identity and delta')
 require(paired.get('training_source_manifest_sha256')==EXPLORATION_SOURCE_SHA and paired.get('diagnostic_source_manifest_sha256')==a['exploration_diagnostic_source_sha256'] and paired.get('center_guard_augmentation')==AUGMENTATION and paired.get('center_guard_fraction')==.25 and paired.get('alpha')==.1 and paired.get('test_access') is False,'actual CG report producer/settings')
 require(review.get('root_authorized') is True and review.get('admit_independent_confirmation') is True and review.get('exploration_paired_sha256')==a['exploration_paired_sha256'],'actual root review admits this paired report')
 return a

def validate_confirmation_budget(b,now=None):
 require(b.get('schema')=='aictrack1.next_cycle10.confirmation_budget.v1' and b.get('root_authorized') is True,'future actual root shared confirmation budget required')
 for key,want in dict(original_initial_free_bytes=13013950464,capacity_added_bytes=32212254720,adjusted_initial_free_bytes=45226205184,actual_disk_total_bytes=128849018880).items():require(b.get(key)==want,'same expanded cumulative baseline '+key)
 start,end=b.get('phase_start_unix'),b.get('deadline_unix')
 require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (start,end)) and 0<start<end,'actual positive finite shared clock')
 for key in ('disk_max_bytes','min_free_bytes','atomic_reserve_bytes'):require(isinstance(b.get(key),int) and not isinstance(b[key],bool),'actual integer resource '+key)
 require(20*1024**3<=b['disk_max_bytes']<=30*1024**3 and b['min_free_bytes']>=3*1024**3 and b['atomic_reserve_bytes']>=2*1024**3 and b.get('final_gpu_reserve_hours')==4,'explicit cumulative20..30GiB/min3/atomic2 and final4GPUh')
 ref=b.get('capacity_evidence');require(isinstance(ref,dict) and isinstance(ref.get('path'),str) and Path(ref['path']).is_absolute() and digest(ref.get('sha256')),'actual capacity evidence path/SHA object')
 if now is not None:require(start<=now<end,'actual shared budget window active')
 return b
