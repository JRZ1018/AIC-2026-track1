"""Consume the root's verified remote identity receipt without accessing remote/GPU."""
import argparse,json
from pathlib import Path
from prepare_p1 import prepare
from p1_contract import ROOT,CLIP_SHA,MANIFEST_SHA,sha

def bind(receipt):
    rows={Path(row['path']).name:row for row in receipt['assets']}
    h1=receipt['h1_receipt'];prefix='/root/autodl-tmp/AICtrack1_q0_fold2_inputs_20261002'
    def asset(name):return {key:rows[name][key] for key in ('path','sha256')}
    payload=dict(schema='aictrack1.p1.assets.v1',status='verified_fold_assets',competition_stage='stage2',
        train_count=118958,validation_count=29737,num_classes=750,training_seed=20261008,split_seed=20261003,
        h1_training_count=118958,h1_training_scope=h1['scope'],h1_validation_used_for_selection=h1['validation_selection'],
        h1_epoch=h1['epoch'],terminal_cache_parity_top1=h1['terminal_cache_parity_top1'],
        h1_row_binding=h1['row_binding'],dataset_root=prefix+'/DataSetStage2',
        canonical_lock='/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json',
        root_verified_remote_receipt_epoch=receipt['epoch'],assets=dict(
            split=asset('split.csv'),h1=asset('last.pt'),h1_receipt=asset('h1_receipt.json'),
            clip=dict(path=prefix+'/artifacts/weights/ViT-B-32.pt',sha256=CLIP_SHA),
            manifest=dict(path=prefix+'/artifacts/audits/stage2/train/samples.csv',sha256=MANIFEST_SHA),
            class_mapping=dict(path=prefix+'/artifacts/audits/stage2/metadata/classes.csv',sha256='0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be'),
            label_conflicts=dict(path=prefix+'/artifacts/audits/stage2/train/label_conflicts.csv',sha256='b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a')),
        control=dict(role='reuse_completed_fullvisual20_epoch20_last',checkpoint_sha256='078bdae1654cecc076d19e8e67b2401c6a47b985d90c91f7b6ec25293002b6c9',
            checkpoint_path='/root/AICtrack1_fullvisual_confirm_v5_20261008/artifacts/experiments/fullvisual20/last.pt',
            experiment_config_sha256='58766198d816efc7655cb21079a70cde00459c71b8948e762bcfdfda048f68d8'))
    if payload['assets']['h1']['sha256']!=h1['checkpoint_sha256']:raise ValueError('H1 receipt/checkpoint differ')
    if payload['h1_row_binding']['validation_used_for_optimization_or_selection'] is not False:raise ValueError('H1 leakage')
    return payload

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('receipt',type=Path);args=parser.parse_args()
    manifest=prepare(bind(json.loads(args.receipt.read_text(encoding='utf-8'))))
    print(json.dumps(dict(status=manifest['status'],files=len(manifest['files']),source_sha256=sha(ROOT/'source_manifest.json'))))
