"""Original cumulative disk baseline plus the one actual additive capacity receipt."""
from __future__ import annotations
import json,hashlib,math,time
from pathlib import Path
INITIAL=13013950464
CAPACITY=32212254720
ADJUSTED=INITIAL+CAPACITY
TOTAL=128849018880
OLD_TOTAL=96636764160
OLD_CONT='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_continuation_20261009'
RECOVERY_REF=dict(path=OLD_CONT+'/root_actual_disk_expansion_recovery_receipt.json',sha256='ba819e4b72731e810743deb1b4b4849eb070e8c664187aeaf255ab3840a78e52')
SHARED_REF=dict(path=OLD_CONT+'/shared_append_budget.json',sha256='ee50f2149b5b08be1a9dcf7abc706fc977345ece3cf455b0ef8f2ed1abf5f9ed')
FAILURE_REF=dict(path=OLD_CONT+'/attempt/FAILURE.json',sha256='b47f82f076fe9d9a284160a1e5f9a478b731a25cee19ae55e478b3284fe39a32')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def validate_capacity(v):
 require(v.get('initial_free_bytes')==INITIAL and type(v.get('initial_free_bytes')) is int,'original full cumulative baseline retained')
 for key,value in [('capacity_increase_bytes',CAPACITY),('adjusted_initial_free_bytes',ADJUSTED),('actual_data_total_bytes',TOTAL)]:
  require(type(v.get(key)) is int and v[key]==value,'actual additive capacity field '+key)
 require(v.get('disk_expansion_recovery')==RECOVERY_REF and sha(RECOVERY_REF['path'])==RECOVERY_REF['sha256'],'actual bound capacity recovery path/SHA')
 recovery=json.loads(Path(RECOVERY_REF['path']).read_text(encoding='utf-8'))
 require(recovery.get('schema')=='aictrack1.fresh_route.actual_disk_expansion_recovery.v1' and recovery.get('root_authorized') is True and recovery.get('original_data_total_bytes')==OLD_TOTAL and recovery.get('actual_data_total_bytes')==TOTAL and recovery.get('data_capacity_increase_bytes')==CAPACITY and recovery.get('original_initial_free_bytes')==INITIAL and recovery.get('adjusted_initial_free_bytes')==ADJUSTED,'actual recovery capacity/baseline metadata')
 require(recovery.get('original_shared_budget')==SHARED_REF and recovery.get('prior_failure')==FAILURE_REF and recovery.get('old_source_or_budget_modified') is False and recovery.get('test_access') is False and recovery.get('exact_prior_groups_closed') is True and recovery.get('canonical_naturally_idle') is True,'actual original shared failure/closure lineage preserved')
 return v
def resource_ok(v,free,total,*,reserve=0,now=None):
 require(type(free) is int and type(total) is int and total==TOTAL,'actual capacity changed; new root evidence required')
 require(v.get('initial_free_bytes')==INITIAL and v.get('capacity_increase_bytes')==CAPACITY and v.get('adjusted_initial_free_bytes')==ADJUSTED and v.get('actual_data_total_bytes')==TOTAL,'one adjusted original ledger required')
 require(type(reserve) is int and reserve>=0,'concrete atomic-save reserve required')
 current=time.time() if now is None else now
 require(v['phase_start_unix']<=current<v['deadline_unix'],'new immutable FULL phase exceeded')
 require(free-reserve>=v['min_free_bytes'] and ADJUSTED-free+reserve<=v['disk_max_bytes'],'FULL capacity-adjusted cumulative disk/atomic floor exceeded')
 return dict(consumed_bytes=ADJUSTED-free,atomic_reserve_bytes=reserve,original_initial_free_bytes=INITIAL,capacity_increase_bytes=CAPACITY,adjusted_initial_free_bytes=ADJUSTED)
