"""Local-only text template freeze; never overwrite actual bound source."""
import json
from full_route_contract import ROOT,sha,FULL_H1_SHA
def freeze():
    if (ROOT/'source_manifest.json').exists():raise ValueError('cannot rewrite an actual bound source')
    files={}
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT).as_posix()
        if not path.is_file() or rel=='template_manifest.json' or rel.startswith('artifacts/') or '__pycache__' in path.parts:continue
        if path.suffix not in {'.py','.md','.json','.yaml','.yml','.txt'}:raise ValueError('non-text full source member '+rel)
        path.read_text(encoding='utf-8');files[rel]=sha(path)
    m=dict(schema='aictrack1.fresh_route.full.training.template.v1',status='unbound_requires_actual_fresh_positive_and_explicit_full_budget',files=files,
        supported_routes=['S','J5_GRAPH'],full_h1_sha256=FULL_H1_SHA,train_count=148695,validation_count=0,fixed_epoch=20,
        original_data_free_baseline_bytes=13013950464,capacity_increase_bytes=32212254720,adjusted_initial_free_bytes=45226205184,actual_data_total_bytes=128849018880,actual_disk_expansion_recovery_sha256='ba819e4b72731e810743deb1b4b4849eb070e8c664187aeaf255ab3840a78e52',previous_template_manifest_sha256='d0c7ccbc2276b300a758bbbab992a36c4a5652fca47648bc80dccd797f5c8291',reviewed_retry_diagnostic_source_manifest_sha256='01949375a866775cd141ded72e06afdb0b1922dd5aa60702cb7fea3b5fbc5ec1',template_revision='expanded_capacity_r1',runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'},test_access=False,source_only=True)
    (ROOT/'template_manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return dict(template_manifest_sha256=sha(ROOT/'template_manifest.json'),files=len(files),bytes=sum((ROOT/r).stat().st_size for r in files))
if __name__=='__main__':print(json.dumps(freeze()))
