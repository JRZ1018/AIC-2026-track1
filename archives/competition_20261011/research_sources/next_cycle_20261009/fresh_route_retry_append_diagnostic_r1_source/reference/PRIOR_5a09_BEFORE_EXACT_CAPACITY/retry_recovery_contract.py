"""Actual one-time capacity correction, immutable failed-attempt provenance. Metadata only."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
OLD_CONT=Path('/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_continuation_20261009')
RECOVERY_SHA='ba819e4b72731e810743deb1b4b4849eb070e8c664187aeaf255ab3840a78e52'
OLD_SHARED_SHA='ee50f2149b5b08be1a9dcf7abc706fc977345ece3cf455b0ef8f2ed1abf5f9ed'
OLD_FAILURE_SHA='b47f82f076fe9d9a284160a1e5f9a478b731a25cee19ae55e478b3284fe39a32'
OLD_CONT_SHA='c5dabc472eee7dd20ba69cb57b3c2b000076eb6b4fca802f39b2861d0918957f'
CAPACITY=32212254720
INITIAL=13013950464
ADJUSTED=INITIAL+CAPACITY
OLD_DISK_MAX=8589934592
DISK_MAX=12884901888
MIN_FREE=3221225472
REF_NAMES=('original_shared_budget','disk_expansion_recovery','prior_failure')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 with Path(path).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def checked(item,path,digest,label):
 require(item==dict(path=str(path),sha256=digest) and sha(path)==digest,'actual pinned retry evidence: '+label)
 return read(path)
def validate_recovery(value):
 refs={name:value.get(name) for name in REF_NAMES}
 old=checked(refs['original_shared_budget'],OLD_CONT/'shared_append_budget.json',OLD_SHARED_SHA,'original shared budget')
 failed=checked(refs['prior_failure'],OLD_CONT/'attempt/FAILURE.json',OLD_FAILURE_SHA,'prior failed continuation')
 recovery=checked(refs['disk_expansion_recovery'],OLD_CONT/'root_actual_disk_expansion_recovery_receipt.json',RECOVERY_SHA,'root actual capacity correction')
 require(recovery.get('schema')=='aictrack1.fresh_route.actual_disk_expansion_recovery.v1' and recovery.get('root_authorized') is True and recovery.get('exact_prior_groups_closed') is True and recovery.get('canonical_naturally_idle') is True,'actual root expansion and natural closure evidence required')
 require(recovery.get('data_capacity_increase_bytes')==CAPACITY and recovery.get('original_initial_free_bytes')==INITIAL and recovery.get('adjusted_initial_free_bytes')==ADJUSTED and recovery.get('actual_data_total_bytes',0)-recovery.get('original_data_total_bytes',0)==CAPACITY and recovery.get('actual_data_free_bytes',0)>=MIN_FREE,'capacity correction must be actual additive30GiB, never a fresh baseline')
 require(recovery.get('effective_old_capacity_free_bytes')==recovery['actual_data_free_bytes']-CAPACITY,'root capacity-normalized free-space evidence')
 require(recovery.get('original_shared_budget')==refs['original_shared_budget'] and recovery.get('prior_failure')==refs['prior_failure'] and recovery.get('old_source_or_budget_modified') is False and recovery.get('test_access') is False and recovery.get('new_GPU_started') is False,'root correction must bind unchanged original failure/budget')
 require(old.get('schema')=='aictrack1.fresh_route.continuation.training_budget.v1' and old.get('initial_free_bytes')==INITIAL and old.get('disk_max_bytes')==OLD_DISK_MAX and old.get('min_free_bytes')==MIN_FREE and old.get('phase_start_unix')==value.get('phase_start_unix') and old.get('deadline_unix')==value.get('deadline_unix'),'original shared8GiB window must stay unchanged')
 for key in ('original_budget','training_source','configs','checkpoints','source_manifest_sha256','training_root','scope'):
  require(old.get(key)==value.get(key),'retry cannot change original source/config/checkpoint/scope: '+key)
 require(failed.get('schema')=='aictrack1.fresh_route.continuation.training.failure.v1' and failed.get('complete') is False and failed.get('shared_budget_sha256')==OLD_SHARED_SHA and failed.get('operator_source_manifest_sha256')==OLD_CONT_SHA and failed.get('source_manifest_sha256')==value.get('source_manifest_sha256') and failed.get('cleanup_error') is None and failed.get('test_access') is False,'actual safely closed prior failure required; failure cannot become completion')
 require(value.get('retry_after_disk_expansion') is True and value.get('capacity_increase_bytes')==CAPACITY and value.get('adjusted_initial_free_bytes')==ADJUSTED and value.get('initial_free_bytes')==INITIAL and value.get('disk_max_bytes')==DISK_MAX and value.get('min_free_bytes')==MIN_FREE,'retry capacity correction and explicit12GiB cumulative allocation required')
 retained=recovery.get('actual_retained_last',{});origin=value['checkpoints']['J5']
 require(retained.get('path')==origin['path'] and retained.get('sha256')==origin['sha256'] and retained.get('epoch_from_previous_strict_audit')==origin['epoch']==19,'retry must resume the actually retained atomic last19')
 temporary=recovery.get('incomplete_temporary',{})
 require(temporary.get('valid_atomic_checkpoint') is False and isinstance(temporary.get('archived_path'),str) and temporary['archived_path']!=temporary.get('original_path') and temporary.get('bytes',0)>0 and isinstance(temporary.get('sha256'),str) and len(temporary['sha256'])==64,'incomplete failed temporary must be retained separately, never resumed')
 require(value.get('watched')==recovery.get('watched') and isinstance(value.get('watched'),list) and len(value['watched'])==6,'retry must cover original four births and failed CONT controller/trainer')
 require({(p['pid'],p['start_ticks']) for p in value['watched']}=={(236091,'308497119'),(236098,'308497585'),(239865,'309227901'),(239872,'309228342'),(243711,'309965381'),(244056,'310043026')},'actual original and failed continuation process births required')
 return dict(refs,capacity_increase_bytes=CAPACITY,adjusted_initial_free_bytes=ADJUSTED,original_data_initial_free_bytes=INITIAL,retry_after_disk_expansion=True,actual_data_total_bytes=recovery['actual_data_total_bytes'])
def require_common(value,context):
 for name in REF_NAMES:
  require(value.get(name)==context[name] and value.get(name+'_sha256')==context[name]['sha256'],'retry completion/audit must bind actual capacity/failure/old shared provenance: '+name)
 for name in ('capacity_increase_bytes','adjusted_initial_free_bytes','original_data_initial_free_bytes'):
  require(value.get(name)==context[name],'retry audit/completion additive capacity accounting: '+name)
 require(value.get('retry_after_disk_expansion') is True,'explicit retry provenance required')
def require_disk_context(context,*,free_bytes=None,total_bytes=None):
 if free_bytes is None or total_bytes is None:
  import shutil
  usage=shutil.disk_usage(context['diagnostic_root']);free_bytes,total_bytes=usage.free,usage.total
 require(type(free_bytes) is int and type(total_bytes) is int and total_bytes>=context['actual_data_total_bytes'] and free_bytes>=MIN_FREE,'actual capacity/free-space floor required')
 consumed=INITIAL+CAPACITY-free_bytes
 require(consumed<=DISK_MAX,'original baseline plus one-time capacity correction exceeded cumulative12GiB')
 return dict(actual_free_bytes=free_bytes,actual_total_bytes=total_bytes,capacity_normalized_consumed_bytes=consumed,capacity_increase_bytes=CAPACITY,original_initial_free_bytes=INITIAL,adjusted_initial_free_bytes=ADJUSTED,disk_max_bytes=DISK_MAX)
