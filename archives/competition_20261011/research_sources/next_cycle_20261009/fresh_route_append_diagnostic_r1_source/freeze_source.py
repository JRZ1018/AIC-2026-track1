"""Freeze the exclusive append adapter after its actual local CPU acceptance."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT.parent/'fresh_route_diagnostic_source'
UPSTREAM_SHA='554d7c8f839d4ab6fee1ce427c5e2b9b73435d4766ac5347794ceecace53fbc2'
TRAINING_SHA='4fc8e1420cf9347e79f7a0f607cd49a2f28735900de52477091324c06a0b8482'
def sha(path):
 with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def main():
 target=ROOT/'source_manifest.json'
 if target.exists():raise FileExistsError('frozen adapter cannot be overwritten')
 upstream=UPSTREAM/'source_manifest.json'
 if sha(upstream)!=UPSTREAM_SHA:raise ValueError('original554d source drift')
 inventory=json.loads(upstream.read_text(encoding='utf-8'))
 for name,digest in inventory['files'].items():
  rel=Path(name)
  if rel.is_absolute() or '..' in rel.parts or sha(UPSTREAM/rel)!=digest:raise ValueError('original554d member drift: '+name)
 receipt=json.loads((ROOT/'CPU_TEST_RECEIPT.json').read_text(encoding='utf-8'))
 if receipt.get('schema')!='aictrack1.fresh_route.append_diagnostic_CPU_acceptance.v1' or receipt.get('complete') is not True or receipt.get('exit_code')!=0:raise ValueError('actual adapter CPU acceptance required')
 for name,digest in receipt['tested_sources_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('source changed since CPU acceptance: '+name)
 files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and not any(x in ('artifacts','__pycache__') for x in p.relative_to(ROOT).parts) and p.name!='source_manifest.json'}
 unchanged={name:digest for name,digest in inventory['files'].items() if name in files and files[name]==digest}
 value=dict(schema='aictrack1.fresh_route.diagnostic_source.v1',revision=2,adapter_revision='append_r1',status='frozen_local_CPU_engineering',files=files,upstream_diagnostic_source_manifest_sha256=UPSTREAM_SHA,unchanged_upstream_members=unchanged,training_template_sha256='05df2bbd14af808847a430bb4a23ba22483774dad00e6724026ea37dd633bf88',production_root='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_append_diagnostic_r1_20261009',training_root='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_20261009',actual_bound_training_source_sha256=TRAINING_SHA,actual_production_dependency_check='pending_root_actual_bound_source',actual_checkpoint_GPU_inference='not_run',selected_routes_supported=['J5_GRAPH'],append_budget_schema='aictrack1.fresh_route.continuation.training_budget.v1',append_budget_sha256=None,actual_future_last_sha256=None,scope=['fixed_last20_C_J5_4V_diagnostic','fixed_J5_graph_validation','typed_append_confirmation'],test_access=False,server_used=False,gpu_used=False,original_phase_start_unix=1791497972.440027,original_deadline_unix=1791526772.440027,append_phase_start_unix=1791526772.440027,maximum_append_deadline_unix=1791533972.440027,original_system_baseline_bytes=13013950464,training_budget_reset=False,graph_ledger_reset=False)
 with target.open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n')
 print(json.dumps(dict(source_manifest_sha256=sha(target),members=len(files),unchanged_upstream_members=len(unchanged),bytes=sum((ROOT/p).stat().st_size for p in files),upstream_source_intact=True)))
if __name__=='__main__':main()
