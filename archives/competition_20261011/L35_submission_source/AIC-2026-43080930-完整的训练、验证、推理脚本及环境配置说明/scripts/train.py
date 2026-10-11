"""Portable L35 reproduction: frozen features, H1100, full or heldout joint20."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portable_training.runtime import CODE, DATA, MODELS, SOURCE, SPLITS, CLIP_SHA, bootstrap, import_file, install_adapter, make_dataset, sha, write_immutable


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for name in ('full', 'heldout', 'features'):
        q = commands.add_parser(name)
        q.add_argument('--data-root', type=Path, required=True, help='Official Stage2 directory containing train/; test is inaccessible to training.')
        q.add_argument('--work-dir', type=Path, required=True, help='New work directory; immutable generated config supports exact resume.')
        q.add_argument('--model-dir', type=Path, default=MODELS)
        q.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
        q.add_argument('--workers', type=int, default=8)
        if name == 'features':
            q.add_argument('--split', choices=('full', 'first', 'second'), default='full')
            q.add_argument('--resume', action='store_true')
        else:
            q.add_argument('--head-checkpoint', type=Path, required=name == 'heldout')
            q.add_argument('--resume', type=Path)
            q.add_argument('--min-free-gib', type=float, default=3.)
            q.add_argument('--prepare-only', action='store_true', help='Check path/config/source/H1 metadata without opening images or training.')
            if name == 'heldout': q.add_argument('--split', choices=('first', 'second'), required=True)
    q = commands.add_parser('h1')
    q.add_argument('--cache', type=Path, required=True, help='Complete normalized train-only official CLIP336 cache.')
    q.add_argument('--work-dir', type=Path, required=True)
    q.add_argument('--scope', choices=('full', 'heldout'), default='full')
    q.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    q.add_argument('--resume', type=Path)
    q = commands.add_parser('check', help='CPU synthetic scientific-function and adapter check; no competition images.')
    args = p.parse_args()
    bootstrap()
    if args.command == 'check':
        from portable_training.selfcheck import run
        return run()
    if args.command == 'h1':
        from portable_training.head import train_h1
        return train_h1(args)
    import yaml
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    split_name = getattr(args, 'split', 'full')
    dataset = make_dataset(args.data_root, work, split_name)
    clip = args.model_dir.resolve() / 'openai_clip_ViT-B-32.pt'
    if sha(clip) != CLIP_SHA: raise ValueError('Only exact official OpenAI ViT-B/32 weights are accepted')
    model_document = dict(model=dict(provider='OpenAI', name='ViT-B/32',
        runtime_downloads_allowed=False, weight_path=str(clip), expected_weight_sha256=CLIP_SHA))
    model_config = work / 'runtime' / 'model.yaml'
    write_immutable(model_config, yaml.safe_dump(model_document, sort_keys=False))
    if args.command == 'features':
        engine = import_file('portable_official_clip_features', CODE / 'portable_training/extract_clip_features_original.py')
        sys.argv = [str(engine.__file__), '--config', str(dataset), '--source', 'train',
            '--manifest', str(DATA / 'train_manifest.csv'), '--split', str(SPLITS[split_name]),
            '--model-config', str(model_config), '--output', str(work / 'cache'),
            '--device', args.device, '--visual-resolution', '336', '--batch-size', '128',
            '--workers', str(args.workers), '--feature-dtype', 'float16'] + (['--resume'] if args.resume else [])
        return engine.main()
    document = yaml.safe_load((SOURCE / 'configs/experiment/p1_full_L35.yaml').read_text(encoding='utf-8'))
    raw = document['experiment']
    raw.update(dataset_config=str(dataset), model_config=str(model_config),
        manifest=str(DATA / 'train_manifest.csv'), split=str(SPLITS[split_name]),
        label_conflicts=str(DATA / 'label_conflicts.csv'), output_dir=str(work / 'checkpoints'),
        device=args.device, run_id='portable-L35-' + args.command)
    raw['training']['workers'] = args.workers
    raw['training']['persistent_workers'] = bool(args.workers)
    if args.command == 'heldout':
        raw['training_scope'] = 'fixed_split_validation'
        raw['expected_train_count'] = 118958
        raw['training']['seed'] = 20261008 if split_name == 'first' else 20261010
    head = args.head_checkpoint or args.model_dir / 'h1_full100.pt'
    raw['head_checkpoint'] = str(head.resolve())
    config = work / 'runtime' / 'p1_full_L35.yaml'
    write_immutable(config, yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    install_adapter(config, work, head, int(args.min_free_gib * 1024**3))
    engine = import_file('portable_frozen_n1_trainer', SOURCE / 'scripts/train_n1_peft.py')
    engine.ROOT = SOURCE
    # Original code already has an archive fallback; avoid requiring a git binary
    # in the submitted reproduction environment. This changes metadata only.
    engine._git_commit = lambda root: 'unknown_archive'
    if args.prepare_only:
        import json
        from aictrack1.data.class_mapping import ClassMapping
        from aictrack1.engine.provenance import CheckpointProvenance
        mapping = ClassMapping.from_csv(DATA / 'class_mapping.csv')
        provenance = CheckpointProvenance('stage2', mapping.digest(), sha(DATA / 'train_manifest.csv'), sha(SPLITS[split_name]), CLIP_SHA, 750)
        head_model, state = engine.load_classifier_head_checkpoint(head.resolve(), feature_dim=512,
            num_classes=750, expected_clip_weight_sha256=CLIP_SHA,
            expected_class_mapping_sha256=mapping.digest(), expected_split_sha256=sha(SPLITS[split_name]),
            expected_provenance=provenance, forbid_full_train_initializer=args.command == 'heldout')
        engine._model_settings(model_config)
        engine._runtime_settings(raw)
        if sha(SOURCE / 'scripts/train_peft.py') != '0758b989a1bb33ba851e03f9a852cf35b7ad30847ccbfb62cf528f8dc015a765':
            raise ValueError('Archived baseline source drift')
        print(json.dumps(dict(complete=True, prepared_config=str(config), head_tensors=len(head_model.state_dict()),
            head_epoch=state['epoch'], training_scope=raw['training_scope'], competition_image_reads=0,
            model_parameter_updates=0, scope='metadata and dependency preflight only; dataset and full training not executed')))
        return 0
    sys.argv = [str(engine.__file__), '--config', str(config)] + (['--resume', str(args.resume.resolve())] if args.resume else [])
    return engine.main()


if __name__ == '__main__':
    raise SystemExit(main())
