"""L35 actual complete combo and root review admit only independent610 confirmation."""
import hashlib,json,math,re
from pathlib import Path
QUERY_SOURCE_SHA='51a79ffbaf94fbb333cdab49123e55ccc3fb5cf2fe78cfdb6b5ea86da8810e50'
EXPLORATION_SOURCE_SHA='b2ad95ad6009516412ca1757ab64fd2b8820688da2c31b777ea3cd68d58207a1'
PRIMARY_SCOPE='candidate_fixed_graph_P7_vs_actual_J5_GRAPH_P7'
def require(x,m):
 if not x:raise ValueError(m)
def digest(x):return isinstance(x,str) and bool(re.fullmatch('[0-9a-f]{64}',x))
def evidence(ref,expected):
 require(isinstance(ref,dict) and digest(expected) and ref.get('sha256')==expected and isinstance(ref.get('path'),str) and Path(ref['path']).is_absolute(),'actual evidence reference')
 p=Path(ref['path']);require(p.is_file() and p.stat().st_size<=32*1024**2,'bounded actual evidence')
 require(hashlib.sha256(p.read_bytes()).hexdigest()==expected,'actual evidence SHA');return json.loads(p.read_text())
def validate_L35_admission(a):
 require(a.get('schema')=='aictrack1.next_cycle10.L35_admission.v1' and a.get('root_authorized') is True and a.get('complete') is True,'actual LR-specific admission')
 require(a.get('selected_arm')=='L35' and a.get('visual_learning_rate')==3.5e-6 and a.get('hard_alpha')==.1,'single LR point only')
 delta=a.get('exploration_balanced_delta_pp');require(isinstance(delta,(float,int)) and not isinstance(delta,bool) and math.isfinite(delta) and delta>0,'actual positive complete combination')
 require(a.get('primary_scope')==PRIMARY_SCOPE and a.get('exploration_training_source_sha256')==EXPLORATION_SOURCE_SHA,'actual LR source and primary scope')
 require(a.get('query_definition_source_sha256')==QUERY_SOURCE_SHA and a.get('new_H1_checkpoint_sha256') is None and a.get('test_access') is False,'new frozen query and pre-H1 null')
 for k in ('exploration_paired_sha256','root_risk_review_sha256','exploration_diagnostic_source_sha256'):require(digest(a.get(k)),'actual digest '+k)
 p=evidence(a.get('exploration_paired'),a['exploration_paired_sha256']);rv=evidence(a.get('root_risk_review'),a['root_risk_review_sha256'])
 require(p.get('schema')=='aictrack1.learning_rate.exploration_paired_graph.v1' and p.get('complete') is True and p.get('arm')=='L35' and p.get('primary_scope')==PRIMARY_SCOPE and p.get('primary_delta_pp')==delta,'actual report primary identity')
 require(p.get('training_source_manifest_sha256')==EXPLORATION_SOURCE_SHA and p.get('diagnostic_source_manifest_sha256')==a['exploration_diagnostic_source_sha256'] and p.get('visual_learning_rate')==3.5e-6 and p.get('alpha')==.1 and p.get('test_access') is False,'actual LR report recipe')
 require(rv.get('root_authorized') is True and rv.get('admit_independent_confirmation') is True and rv.get('exploration_paired_sha256')==a['exploration_paired_sha256'],'actual root risk admission');return a
def validate_confirmation_budget(b,now=None):
 require(b.get('schema')=='aictrack1.next_cycle10.confirmation_budget.v1' and b.get('root_authorized') is True,'new explicit shared budget')
 for k,x in dict(original_initial_free_bytes=13013950464,capacity_added_bytes=32212254720,adjusted_initial_free_bytes=45226205184,actual_disk_total_bytes=128849018880).items():require(b.get(k)==x,'same cumulative baseline '+k)
 start,end=b.get('phase_start_unix'),b.get('deadline_unix');require(all(isinstance(x,(int,float)) and math.isfinite(x) for x in (start,end)) and 0<start<end,'explicit actual clock')
 require(b.get('disk_max_bytes')==38654705664 and b.get('min_free_bytes',0)>=3221225472 and b.get('atomic_reserve_bytes',0)>=2147483648 and b.get('final_gpu_reserve_hours')==4,'original expanded ledger with explicit36GiB cap/min3/atomic2/final4')
 require(isinstance(b.get('capacity_evidence'),dict) and digest(b['capacity_evidence'].get('sha256')),'capacity receipt reference')
 if now is not None:require(start<=now<end,'shared budget active')
 return b
