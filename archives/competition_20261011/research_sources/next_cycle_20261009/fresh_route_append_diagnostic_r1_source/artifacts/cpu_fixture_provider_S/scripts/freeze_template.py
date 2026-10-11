"""Local preparation only; freeze text template before independent CPU review."""
import json
from pathlib import Path
from fresh_route_contract import ROOT,sha,H1_SOURCE_SHA,H1_SHA,SPLIT_SHA

def freeze():
    if (ROOT/'source_manifest.json').exists():raise ValueError('do not alter an actual bound source')
    files={}
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT).as_posix()
        if not path.is_file() or rel=='template_manifest.json' or '__pycache__' in path.parts or rel.startswith('artifacts/'):continue
        if path.suffix not in {'.py','.md','.json','.yaml','.yml','.txt'}:raise ValueError('non-text template member '+rel)
        path.read_text(encoding='utf-8');files[rel]=sha(path)
    payload=dict(schema='aictrack1.fresh_route.training.template.v1',status='reviewed_template_not_bound_not_launched',files=files,
        source_h1_manifest_sha256=H1_SOURCE_SHA,h1_sha256=H1_SHA,split_sha256=SPLIT_SHA,
        supported_routes=['S','J5_GRAPH'],actual_admission_required=True,test_access=False,
        original_phase_start_unix=1791497972.440027,original_deadline_unix=1791526772.440027,
        original_data_free_baseline_bytes=13013950464,disk_max_bytes=8589934592,min_free_bytes=3221225472)
    (ROOT/'template_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return dict(template_manifest_sha256=sha(ROOT/'template_manifest.json'),files=len(files),bytes=sum((ROOT/r).stat().st_size for r in files))
if __name__=='__main__':print(json.dumps(freeze()))
