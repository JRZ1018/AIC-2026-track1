"""Build immutable full configs/assets; GPU admission uses a separate choice receipt."""
import json
from pathlib import Path
import yaml
from p1_contract import ROOT,ARMS,CLIP_SHA,MANIFEST_SHA,FULL_H1_SHA,sha,validate_assets

def prepare():
    prefix='/root/autodl-tmp/AICtrack1_q0_fold2_inputs_20261002'
    previous='/root/AICtrack1_fullvisual_full_v5_20261008'
    payload=dict(schema='aictrack1.p1.full_assets.v1',status='pinned_full_assets_runtime_sha_required',competition_stage='stage2',
        train_count=148695,validation_count=0,num_classes=750,training_seed=20261008,h1_training_scope='full_train_fixed_epochs',
        h1_training_count=148695,h1_epoch=100,h1_validation_used_for_selection=False,dataset_root=prefix+'/DataSetStage2',
        canonical_lock='/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json',assets=dict(
            h1=dict(path=previous+'/artifacts/experiments/mainline-h1/last.pt',sha256=FULL_H1_SHA),
            h1_receipt=dict(path=previous+'/artifacts/experiments/mainline-h1/h1_receipt.json',sha256='0e46cf54918bc152af34f7b3edb6b84fe7a9c3d2211bc3925f8b994a1dc144cf'),
            split=dict(path=previous+'/inputs/split.csv',sha256='0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'),
            clip=dict(path=prefix+'/artifacts/weights/ViT-B-32.pt',sha256=CLIP_SHA),
            manifest=dict(path=prefix+'/artifacts/audits/stage2/train/samples.csv',sha256=MANIFEST_SHA),
            class_mapping=dict(path=prefix+'/artifacts/audits/stage2/metadata/classes.csv',sha256='0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be'),
            label_conflicts=dict(path=prefix+'/artifacts/audits/stage2/train/label_conflicts.csv',sha256='b1afe084e654aeac5d431501657734c168a3cfc1f928defacac219dcf84c430a')))
    validate_assets(payload)
    (ROOT/'inputs/full_assets.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    asset=payload['assets']
    model={'model':dict(provider='OpenAI',name='ViT-B/32',upstream_commit='d05afc436d78f1c48dc0dbf8e5980a9d471f35f6',
        weight_path=asset['clip']['path'],expected_weight_sha256=CLIP_SHA,runtime_downloads_allowed=False)}
    data={'dataset':dict(competition_stage='stage2',contract_confirmed=True,expected_train_count=148695,
        expected_test_count=37444,expected_num_classes=750,stage='rematch',dataset_root=payload['dataset_root'],
        class_mapping=asset['class_mapping']['path'],class_mapping_provenance='train_directory_names',extensions=['.jpg'],
        allow_truncated_images=False,sources=dict(train=dict(root='train',label_source='directory',class_from_parent=True),
            test=dict(root='__unavailable_test__',label_source='none')),
        audit=dict(output_root='../../artifacts/audits/stage2',workers=4,near_duplicate_hamming=3,near_duplicate_max_bucket=500,
            fail_on_bad_files=True,fail_on_label_conflicts=False),split=dict(validation_fraction=.2,seed=20261003,min_train_per_class=1),
        transforms=dict(image_size=336,resize_size=336,train_crop_scale_min=.8,train_horizontal_flip_probability=.5),
        submission=dict(filename_mode='basename',official_class_id_pattern=r'\d{4}'))}
    for name,value in [('model/full_clip.yaml',model),('data/full336.yaml',data)]:
        (ROOT/'configs'/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    for arm,lr in ARMS.items():
        raw=dict(competition_stage='stage2',run_id=f'20261009-p1-full-{arm}-fixed20',execution_enabled=True,
            training_scope='full_train_fixed_epochs',expected_train_count=148695,selection_rule='fixed_final',
            confidence_observation_view='strong',dataset_config='../data/full336.yaml',model_config='../model/full_clip.yaml',
            manifest=asset['manifest']['path'],split=asset['split']['path'],label_conflicts=asset['label_conflicts']['path'],
            head_checkpoint=asset['h1']['path'],output_dir=f'../../artifacts/experiments/P1_full_{arm}',device='cuda',
            visual_input_resolution=336,position_interpolation='bicubic_7x7_to_10x10',initial_top1_tolerance=.002,
            base_loss=dict(kind='sce',alpha=.1,beta=1.,log_zero=-4.),
            runtime=dict(defer_epoch_metrics=True,reuse_validation_loader=True,record_gradient_diagnostics=True,safe_numpy_rng_checkpoint=True),
            peft=dict(mode='full_visual',lora_blocks=8),confidence_weighting=dict(enabled=True,warmup_epochs=1,momentum=.8,
                history_mix=.5,min_weight=.02,magnitude_mix=.75,stop_on_effective_quality_l2=True),
            training=dict(epochs=20,batch_size=64,workers=8,prefetch_factor=2,persistent_workers=True,seed=20261008,
                visual_learning_rate=1.75e-6,head_learning_rate=lr,weight_decay=.01,gradient_clip_norm=1.),
            token_readout=dict(mode='none',initialization_seed=20261005))
        (ROOT/f'configs/experiment/p1_full_{arm}.yaml').write_text(yaml.safe_dump({'experiment':raw},sort_keys=False),encoding='utf-8')
    files={}
    for p in sorted(ROOT.rglob('*')):
        relative=p.relative_to(ROOT)
        if (p.is_file() and '__pycache__' not in relative.parts and 'artifacts' not in relative.parts
                and p.name not in {'source_manifest.json','cpu_smoke_result.json','full_selection_receipt.json'}
                and p.suffix in {'.py','.yaml','.json','.txt','.md'}):files[relative.as_posix()]=sha(p)
    manifest=dict(schema='aictrack1.p1.full_source.v1',status='ready_requires_positive_selection',files=files,
        upstream_fold_source_sha256='2fee29fc5e5c282a331883de2b3cd4d5865c28e38a8cead89be247ec95a6c9fe',
        full_h1_strict_reference_sha256=sha(ROOT/'scripts/full_h1_reference.py'),
        immutable_recipe='official fresh CLIP + full H1100 + strong SCE/AC20 fixed final; no validation selection',
        selected_arm_source='external full_selection_receipt + positive root-reviewed fresh matched confirmation SHA; J5/J20 only; one canonical queue arm',
        runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return manifest

if __name__=='__main__':
    manifest=prepare();print(json.dumps(dict(status=manifest['status'],files=len(manifest['files']),source_sha256=sha(ROOT/'source_manifest.json'))))
