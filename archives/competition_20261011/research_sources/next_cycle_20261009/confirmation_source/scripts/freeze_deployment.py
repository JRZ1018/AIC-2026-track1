"""Freeze only ready bootstrap files, without inventing the later canonical H1 source manifest."""
import ast,json
from pathlib import Path
from confirmation_contract import ROOT,sha,read_json,write_new
def main():
    blocked={'artifacts','__pycache__','prepared_visual','test_fixtures','official_feature_cache','receipt_source'}
    test_path=ROOT/('cpu_contract_tests_'+sha(ROOT/'scripts/test_confirmation_contract.py')[:16]+'.json')
    test=read_json(test_path)
    if test.get('passed') is not True or test.get('tests_run')!=10:raise ValueError('actual latest ten CPU tests must pass')
    files={}
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or blocked.intersection(rel.parts) or path.name in {'source_manifest.json','DEPLOYMENT_FILES.json','CONFIRMATION_DELIVERY.json'}:continue
        if path.name.startswith('cpu_contract_tests') and path!=test_path:continue
        if path.suffix not in {'.py','.yaml','.json','.csv','.txt','.md'}:continue
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        files[rel.as_posix()]=sha(path)
    feature=read_json(ROOT/'inputs/feature_identity.json');assets=read_json(ROOT/'inputs/confirmation_assets.json')
    payload=dict(schema='aictrack1.confirmation.deployment_files.v1',status='ready_to_stage_then_bind_actual_remote_H1_assets',
        deployment_root='/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009',files=files,
        file_count=len(files),total_bytes=sum((ROOT/rel).stat().st_size for rel in files),
        canonical_source_manifest_not_yet_created=True,split_sha256=assets['assets']['split']['sha256'],
        upstream_training_source_manifest_sha256=read_json(ROOT/'clone_receipt.json')['source_manifest_sha256'],
        latest_cpu_test_receipt=dict(path=test_path.relative_to(ROOT).as_posix(),sha256=sha(test_path),tests_run=10,passed=True),
        excluded_directories=sorted(blocked),test_access=False,gpu_access=False,
        external_feature_assets=dict(local_root=feature['original_root'],expected_remote_root='/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009/inputs/official_feature_cache',
            metadata_sha256=feature['metadata_sha256'],complete_sha256=feature['complete_sha256'],shards=feature['shards'],
            bytes=sum(x['bytes'] for x in feature['shards']),record_count=148695,original_remote_partial_cache_must_not_be_modified=True))
    write_new(ROOT/'DEPLOYMENT_FILES.json',payload)
    print(json.dumps(dict(path=str(ROOT/'DEPLOYMENT_FILES.json'),sha256=sha(ROOT/'DEPLOYMENT_FILES.json'),files=len(files),bytes=payload['total_bytes'],cpu_tests=test_path.name)))
if __name__=='__main__':main()
