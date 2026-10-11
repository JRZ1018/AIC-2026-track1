"""Only actual PF30 graph-combination evidence admits new610 train-only confirmation."""
import hashlib,json,math,re
from pathlib import Path
QUERY_SOURCE_SHA='51a79ffbaf94fbb333cdab49123e55ccc3fb5cf2fe78cfdb6b5ea86da8810e50'
EXPLORATION_SOURCE_SHA='61f66abb4aae14d8a30e56ab1fe3edeb5321f9937326259abbd9e6928c51f5f4'
PF_SOURCE_SHA='18d9ad67a6cf70e2ae6b52fecf1a179ec46426380742ab6bd92509ccc8dcfa87'
PF_PROTOCOL_SHA='96ca446cafe75f89136ed617c6db284a190fe7eb21657796bb8cc4121ffbb2c3'
PRIMARY_SCOPE='candidate_fixed_graph_P7_vs_actual_J5_GRAPH_P7'
PROTOCOL='persistent-supervision-gate-v2'

def require(value,message):
 if not value:raise ValueError(message)
def digest(v):return isinstance(v,str) and bool(re.fullmatch('[0-9a-f]{64}',v))
def evidence(ref,expected):
 require(isinstance(ref,dict) and ref.get('sha256')==expected and digest(expected) and isinstance(ref.get('path'),str) and Path(ref['path']).is_absolute(),'actual absolute evidence path/SHA required')
 p=Path(ref['path']);require(p.is_file() and p.stat().st_size<=32*1024**2,'bounded actual evidence JSON')
 with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 require(actual==expected,'actual evidence file drift')
 return json.loads(p.read_text(encoding='utf-8'))
def validate_PF_admission(a):
 require(a.get('schema')=='aictrack1.next_cycle10.PF_admission.v1' and a.get('root_authorized') is True and a.get('complete') is True,'actual PF-only root admission required')
 require(a.get('selected_arm')=='PF30' and a.get('max_ac_weight')==.30 and a.get('persistent_filter_protocol')==PROTOCOL and a.get('hard_alpha')==.1,'one actual PF30 original hard recipe')
 delta=a.get('exploration_balanced_delta_pp');require(isinstance(delta,(float,int)) and not isinstance(delta,bool) and math.isfinite(delta) and delta>0,'positive actual PF graph combination required')
 require(a.get('primary_scope')==PRIMARY_SCOPE,'raw PF cannot stand in for PF graph versus actual J5_GRAPH')
 require(a.get('exploration_training_source_sha256')==EXPLORATION_SOURCE_SHA and a.get('persistent_filter_source_sha256')==PF_SOURCE_SHA and a.get('persistent_filter_protocol_sha256')==PF_PROTOCOL_SHA,'reviewed actual PF producer/math/protocol required')
 for k in ('exploration_paired_sha256','root_risk_review_sha256','exploration_diagnostic_source_sha256'):require(digest(a.get(k)),'actual PF evidence digest '+k)
 require(a.get('query_definition_source_sha256')==QUERY_SOURCE_SHA and a.get('new_H1_checkpoint_sha256') is None and a.get('test_access') is False,'pre-result new query/future H1 null/no test')
 paired=evidence(a.get('exploration_paired'),a['exploration_paired_sha256']);review=evidence(a.get('root_risk_review'),a['root_risk_review_sha256'])
 require(paired.get('schema')=='aictrack1.persistent_filter_joint.exploration_paired_graph.v1' and paired.get('complete') is True and paired.get('arm')=='PF30' and paired.get('primary_scope')==PRIMARY_SCOPE and paired.get('primary_delta_pp')==delta,'actual combo report identity and delta')
 require(paired.get('training_source_manifest_sha256')==EXPLORATION_SOURCE_SHA and paired.get('diagnostic_source_manifest_sha256')==a['exploration_diagnostic_source_sha256'] and paired.get('persistent_filter_protocol')==PROTOCOL and paired.get('max_ac_weight')==.30 and paired.get('alpha')==.1 and paired.get('test_access') is False,'actual PF report producer/settings')
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
