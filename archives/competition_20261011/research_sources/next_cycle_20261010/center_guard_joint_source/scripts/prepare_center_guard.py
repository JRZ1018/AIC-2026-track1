"""Bind the already frozen CG25 template to one explicit future resource budget."""
from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path
from p1_contract import ROOT,REMOTE_ROOT,sha,require,verify_files
from fresh_route_runtime import budget as validate_budget

def main():
    p=argparse.ArgumentParser();p.add_argument('--budget',required=True,type=Path);p.add_argument('--expected-budget-sha256',required=True);a=p.parse_args()
    m=verify_files(require_bound=False);require(m['status']=='prepared','template must be unbound')
    require(os.name=='posix' and ROOT.resolve()!=REMOTE_ROOT.resolve(),'binder is dedicated CPU-only template operation')
    require(sha(a.budget)==a.expected_budget_sha256,'actual typed budget SHA mismatch')
    actual=json.loads(a.budget.read_text(encoding='utf-8'));require(actual.get('schema')=='aictrack1.center_guard_joint.budget.v1','CG25 typed budget required')
    # Evaluate the original capacity evidence and future-clock contract without creating the execution root.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        probe=Path(td);(probe/'inputs').mkdir();shutil.copyfile(a.budget,probe/'inputs/budget.json');validate_budget(probe)
    evidence=actual['capacity_evidence'];require(sha(evidence['path'])==evidence['sha256'],'original expansion receipt SHA')
    require(not REMOTE_ROOT.exists(),'exclusive future execution root already exists')
    REMOTE_ROOT.mkdir()
    for rel in m['files']:
        target=REMOTE_ROOT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    shutil.copyfile(a.budget,REMOTE_ROOT/'inputs/budget.json')
    bound=dict(m,status='bound',template_source_manifest_sha256=sha(ROOT/'source_manifest.json'),budget_sha256=a.expected_budget_sha256,future_last_sha256=None,future_admission_sha256=None)
    bound['files']=dict(m['files']);bound['files']['inputs/budget.json']=sha(REMOTE_ROOT/'inputs/budget.json')
    with (REMOTE_ROOT/'source_manifest.json').open('x',encoding='utf-8') as f:json.dump(bound,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    verify_files(REMOTE_ROOT)
    print(json.dumps(dict(status='bound',source_manifest_sha256=sha(REMOTE_ROOT/'source_manifest.json'),execution_root=str(REMOTE_ROOT),budget_sha256=a.expected_budget_sha256,GPU_started=False,model_reads=0,image_reads=0)))
    return 0
if __name__=='__main__':raise SystemExit(main())
