"""Freeze this independent source once; artifacts and bytecode never deploy."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 target=ROOT/'source_manifest.json'
 if target.exists():raise FileExistsError('frozen source must not be silently replaced')
 files={}
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or any(x in ('artifacts','__pycache__') for x in p.relative_to(ROOT).parts) or p.name=='source_manifest.json':continue
  files[p.relative_to(ROOT).as_posix()]=sha(p)
 value=dict(schema='aictrack1.fresh_route.diagnostic_source.v1',revision=1,status='frozen_local_CPU_engineering',files=files,training_template_sha256='05df2bbd14af808847a430bb4a23ba22483774dad00e6724026ea37dd633bf88',production_root='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_diagnostic_20261009',training_root='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_20261009',actual_bound_training_source_sha256=None,actual_production_dependency_check='pending_actual_bound_source',actual_checkpoint_GPU_inference='pending_root_only',selected_routes_supported=['S','J5_GRAPH'],test_access=False,server_used=False,gpu_used=False,original_phase_start_unix=1791497972.440027,original_deadline_unix=1791526772.440027,original_system_baseline_bytes=13013950464)
 target.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(dict(source_manifest_sha256=sha(target),members=len(files),bytes=sum((ROOT/p).stat().st_size for p in files))))
if __name__=='__main__':main()
