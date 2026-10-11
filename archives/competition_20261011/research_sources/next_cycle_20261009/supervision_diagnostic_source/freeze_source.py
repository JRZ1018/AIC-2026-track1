"""Freeze the independent S diagnostic after CPU acceptance and actual S source binding."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    training=ROOT.parent/'supervision_source'
    source=json.loads((training/'source_manifest.json').read_text(encoding='utf-8'))
    if source.get('schema')!='aictrack1.supervision.source.v1' or source.get('status')!='bound':
        raise ValueError('actual bound supervision source required')
    digest=sha(training/'source_manifest.json')
    contract=(ROOT/'supervision_diag_contract.py').read_text()
    if "TRAINING_MANIFEST_SHA = '"+digest+"'" not in contract:
        raise ValueError('contract must positively pin actual S source bytes before freeze')
    receipt=json.loads((ROOT/'CPU_TEST_RECEIPT.json').read_text())
    if receipt.get('complete') is not True or receipt.get('failures')!=0 or receipt.get('training_source_manifest_sha256')!=digest:
        raise ValueError('actual full CPU test acceptance with S source binding required')
    names=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and 'artifacts' not in p.relative_to(ROOT).parts
        and p.name!='source_manifest.json']
    inventory={name:sha(ROOT/name) for name in sorted(names)}
    payload=dict(schema='aictrack1.supervision.diagnostic_source.v1',revision=1,status='frozen',
        training_source_manifest_sha256=digest,training_config_sha256=sha(training/'configs/experiment/p1_S.yaml'),
        checkpoint_role='last',checkpoint_epoch=20,trainable_tensors=152,trainable_numel=87849216,
        classifier_head_source='old_fold_h1_frozen',model_arm='S',test_access=False,
        deployment_root='/root/autodl-tmp/AICtrack1_next_cycle_supervision_diagnostic_20261009',
        training_root='/root/autodl-tmp/AICtrack1_next_cycle_supervision_20261009',
        files=inventory,file_bytes={n:(ROOT/n).stat().st_size for n in inventory},
        inherited_baseline_diagnostic_source_sha256='7bb94f9192a6c02ce0928dd2e94aecb879c661f66097bb66ed625224c7d364f6',
        raw_engine_sha256=sha(ROOT/'reference/diagnose_stage2_tta.py'),
        raw_loader_sha256=sha(ROOT/'reference/fv2_single_model.py'),
        selection_protocol_sha256=sha(ROOT/'selection_protocol.json'),
        query_identity_sha256=sha(ROOT/'inputs/frozen_queries/query_identity.json'),
        supervision_registration_sha256=sha(ROOT/'supervision_registration.json'),
        remote_dependency_check='pending',production_restore='pending',gpu_inference='pending')
    path=ROOT/'source_manifest.json'
    if path.exists():raise FileExistsError('source is already frozen; revision required')
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(source_manifest_sha256=sha(path),files=len(inventory),bytes=sum(payload['file_bytes'].values()))))

if __name__=='__main__':main()
