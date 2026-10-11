"""Freeze only this independent source; no runtime assets, fixtures or model weights."""
from __future__ import annotations
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=ROOT/'template_manifest.json'
    if target.exists():raise FileExistsError('immutable template already frozen')
    files={}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(k in ('artifacts','submissions','test_fixtures','__pycache__') for k in rel.parts):continue
        if p.suffix in ('.pt','.pyc') or p.name in ('source_manifest.json','template_manifest.json'):continue
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        files[rel.as_posix()]=sha(p)
    value=dict(schema='aictrack1.fresh_route.final_inference.template.v1',status='unbound_actual_typed_positive_and_completed_full20_required',files=files,source_only=True,future_checkpoint_sha256=None,
        supported_routes=['S','J5_GRAPH'],full_root='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_full_20261009',template_staging_root='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_final_template_20261009',final_root='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_final_20261009',
        reviewed_full_template_sha256=sha(ROOT/'references/UPSTREAM_FULL_TEMPLATE.json'),revision='r2_both_root_operands_resolved',
        previous_template_sha256=sha(ROOT/'references/UPSTREAM_FINAL_62367_TEMPLATE.json'),
        single_checkpoint=True,views=dict(native336=True,resize384_crop336=True,equal_four_logits=True),P7=dict(temperature=.5,strength=.75,iterations=60),
        graph_source_sha256='5748cf07c01bd8bf3d68bf5b50065303d2c8990260b00400e5511593aaa0834c',graph_policy_sha256=sha(ROOT/'inputs/fixed_graph_protocol.json'),runtime=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8'),
        graph_cpu_threads=2,original_confirmation_ledger_modified=False,optional_feature_hook_extra_forwards=0,test_access=False,gpu_access=False)
    target.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(template_manifest_sha256=sha(target),members=len(files),bytes=sum((ROOT/r).stat().st_size for r in files))))
    return 0
if __name__=='__main__':raise SystemExit(main())
