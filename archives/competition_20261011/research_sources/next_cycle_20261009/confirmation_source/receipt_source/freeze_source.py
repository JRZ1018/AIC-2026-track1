"""Freeze the independent receipt writer, excluding fixtures and older test runs."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    latest=ROOT/('CPU_TESTS_'+sha(ROOT/'test_confirmation_receipt.py')[:16]+'.json')
    result=json.loads(latest.read_text())
    if result.get('passed') is not True or result.get('tests_run')!=10:raise ValueError('ten actual CPU tests must pass')
    files={}
    for p in sorted(ROOT.iterdir()):
        if not p.is_file() or p.name=='source_manifest.json' or (p.name.startswith('CPU_TESTS_') and p!=latest):continue
        if p.suffix not in {'.py','.json','.md'}:continue
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        files[p.name]=sha(p)
    full=ROOT.parents[1]/'full_training_source/source_manifest.json'
    payload=dict(schema='aictrack1.confirmation.receipt_source.v1',status='CPU_gate_tests_pass_actual_models_not_yet_available',files=files,
        latest_cpu_tests=latest.name,tests_run=10,full_source_metadata_schema_integration_sha256=sha(full) if full.is_file() else None,
        bootstrap_deployment_inventory_sha256='b41a38578160e49718d406e7c234be7f6b9cbc2b2e14c7a2c31802f4f1a7b38a',
        no_mutation_to_bootstrap=True,no_gpu=True,test_access=False,positive_receipt_not_yet_generated=True)
    raw=(json.dumps(payload,indent=2,sort_keys=True)+'\n').encode();path=ROOT/'source_manifest.json'
    if path.exists() and path.read_bytes()!=raw:raise ValueError('independent receipt source already frozen; create a new revision')
    if not path.exists():path.write_bytes(raw)
    print(json.dumps(dict(path=str(path),sha256=sha(path),files=len(files),bytes=sum((ROOT/rel).stat().st_size for rel in files))))
if __name__=='__main__':main()
