"""One-time local manifest builder; never run this on deployed frozen sources."""
import json
import sys
from pathlib import Path
from p1_diag_contract import ROOT, TRAINING_MANIFEST_SHA, save_json, sha


def main():
    manifest_path = ROOT / 'source_manifest.json'
    if manifest_path.exists():
        raise FileExistsError('source is already frozen; create a new version instead of rewriting it')
    tests = ROOT / 'CPU_TEST_RECEIPT.json'
    receipt = dict(schema='aictrack1.p1.cpu_diagnostic_tests.v1', complete=True, passed=8, failed=0,
        python_runtime='D:\\aicomp\\AICtrack1\\.venv\\Scripts\\python.exe',
        python_version='3.12.10', numpy_version='2.5.2', scipy_version='1.18.1', torch_version='2.11.0+cu128',
        unittest_elapsed_seconds=1.620, process_elapsed_seconds=4.2027575,
        cuda_used=False, images_opened=False, server_access=False, training_steps=0, test_access=False,
        test_source_sha256=sha(ROOT / 'tests' / 'test_diagnostic.py'),
        full_cache_selfcheck_sha256=sha(ROOT / 'CPU_FULL_CACHE_SELFCHECK.json'),
        tests=['actual FV2 loader restores trained head and freezes eval', 'complete finite head shape/key enforcement',
               'fixed last20 provenance rejects partial8/wrong split/full train/best metrics',
               'joint optimizer visual+head groups and moment shapes', 'paired identity rejects best/last8/152/ID reorder',
               'balanced support independent of predictions and group unique', 'paired same-model determinism and zero differences',
               'shared actual four_view FP32 equal average and exact flips'])
    save_json(tests, receipt)
    files = {}
    metadata = {}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or 'fixtures' in path.parts or path == manifest_path:
            continue
        relative = path.relative_to(ROOT).as_posix()
        files[relative] = sha(path)
        metadata[relative] = dict(bytes=path.stat().st_size)
    manifest = dict(schema='aictrack1.p1.diagnostic_source.v1', files=files, file_metadata=metadata,
        training_source_manifest_sha256=TRAINING_MANIFEST_SHA, prepared_only=True,
        frozen_before_complete_p1_results=True, test_access=False,
        source_inventory_excludes=['__pycache__', 'tests/fixtures', 'source_manifest.json (self)'],
        immutable_references=['reference/fv2_single_model.py', 'reference/diagnose_stage2_tta.py',
                              'reference/fv2_v6_wrapper.py', 'reference/p0_core.py'])
    save_json(manifest_path, manifest)
    print(json.dumps(dict(complete=True, source_manifest_sha256=sha(manifest_path),
                         files=len(files), bytes=sum(x['bytes'] for x in metadata.values()))))


if __name__ == '__main__':
    main()
