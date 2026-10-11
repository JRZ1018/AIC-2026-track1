"""Root-operated P1 GPU diagnostic; engineering/tests are performed on CPU only.

The immutable FV2 loader and final-role diagnostic engine are reused verbatim.
Only this outer contract changes: complete joint head, terminal last20, and P1
source/asset identities. No best selection, test images, or optimizer updates.
"""
from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import importlib.util
import importlib
import inspect
import hashlib
import json
import signal
import shutil
import os
import subprocess
import sys
import time
from pathlib import Path

from p1_diag_contract import (ARMS, HEAD_KEYS, H1_SHA, JOINT_KEYS, JOINT_NUMEL,
    MAPPING_SHA, ROOT, SPLIT_SHA, TRAINING_MANIFEST_SHA, VIEW_PROTOCOL,
    save_json, sha, validate_checkpoint_metadata, validate_joint_state,
    validate_optimizer_state, verify_owned_sources)
from diagnostic_resources import memory_snapshot, memory_reason
from view_features import FourViewFeatureCollector, FEATURE_PROTOCOL

TRAIN_ROOT = None
CURRENT_ARM = None
OUTPUT = None
SOURCE_SHA = None
SAVE_FEATURES = False
FEATURE_COLLECTOR = None


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def engine_argv(config, checkpoint, digest, lock, output, batch_size, workers, seconds):
    return [str(ROOT / 'run_p1_diagnostic.py'), '--experiment-config', str(config),
        '--checkpoint', str(checkpoint), '--expected-checkpoint-sha256', digest,
        '--checkpoint-role', 'final', '--shared-lock-path', str(lock), '--output-dir', str(output),
        '--batch-size', str(batch_size), '--workers', str(workers),
        '--four-view-scale384', '--save-fused-logits', '--hard-seconds', str(seconds)]


def dependency_probe(training_root):
    """Real frozen dependencies/parser only; never calls GPU/image/checkpoint entry."""
    training_root = Path(training_root).resolve()
    if sha(training_root / 'source_manifest.json') != TRAINING_MANIFEST_SHA:
        raise ValueError('dependency probe requires frozen 2fee training source')
    sys.path[:0] = [str(training_root / 'scripts'), str(training_root / 'src')]
    from p1_contract import verify_files, assets, validate_config, verify_remote_assets
    from mainline_h1_contract import load_new_fold_head
    from aictrack1.engine.training_queue import TrainingLease, read_training_lock, probe_running_trainers
    verify_files(training_root)
    declarations = assets(training_root)
    imports = {}
    # Check every static import in the actual production wrapper against real modules.
    for node in ast.walk(ast.parse(Path(__file__).read_text(encoding='utf-8'))):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = importlib.import_module(node.module)
            for alias in node.names:
                if not hasattr(module, alias.name):
                    raise ImportError(node.module + '.' + alias.name)
            imports[node.module] = getattr(module, '__file__', 'built-in')
        elif isinstance(node, ast.Import):
            for alias in node.names:
                importlib.import_module(alias.name)
    for module_name in ('p1_contract', 'mainline_h1_contract', 'aictrack1.engine.training_queue'):
        path = Path(importlib.import_module(module_name).__file__).resolve()
        if not path.is_relative_to(training_root):
            raise ValueError('dependency resolved outside frozen training package: ' + module_name)
    import yaml
    for arm in ARMS:
        config = training_root / 'configs' / 'experiment' / f'p1_{arm}.yaml'
        raw = yaml.safe_load(config.read_text(encoding='utf-8'))['experiment']
        inspect.signature(validate_config).bind(raw, arm, declarations)
        validate_config(raw, arm, declarations)
    inspect.signature(verify_remote_assets).bind(training_root)
    inspect.signature(load_new_fold_head).bind(Path('/not_opened/head.pt'), feature_dim=512, num_classes=750,
        expected_clip_weight_sha256='not_opened', expected_class_mapping_sha256=MAPPING_SHA,
        expected_split_sha256=SPLIT_SHA, expected_provenance=None)
    inspect.signature(TrainingLease.acquire).bind(Path('/not_opened/queues/training.lock.json'),
        owner='dependency_probe_only', config='not_opened', output_dir='not_opened', command=('not_run',))
    inspect.signature(read_training_lock).bind(Path('/not_opened/queues/training.lock.json'))
    inspect.signature(probe_running_trainers).bind()
    loader = import_file('aictrack1.models.p1_dependency_fv2_loader', ROOT / 'reference' / 'fv2_single_model.py')
    inspect.signature(loader.load_diagnostic_model).bind(config_path=Path('/not_opened/config.yaml'),
        checkpoint_path=Path('/not_opened/last.pt'), class_mapping_digest=MAPPING_SHA, num_classes=750,
        split_sha256=SPLIT_SHA, device='cpu', git_commit='dependency_probe_only', checkpoint_role='final')
    inspect.signature(loader.configure_visual_peft).bind('not_constructed_visual', 'full_visual',
        lora_rank=4, lora_alpha=8., lora_blocks=8)
    inspect.signature(loader.enable_interpolated_visual_resolution).bind('not_constructed_visual', target_resolution=336)
    inspect.signature(loader.PEFTImageClassifier).bind('not_constructed_visual', 'not_constructed_head', normalize_features=True)
    inspect.signature(loader.configure_classifier_trainability_from_state).bind('not_constructed_head', {})
    inspect.signature(loader.load_trainable_state_dict).bind('not_constructed_model', {})
    from aictrack1.models.heads import load_classifier_head_checkpoint
    inspect.signature(load_classifier_head_checkpoint).bind(Path('/not_opened/head.pt'), feature_dim=512,
        num_classes=750, expected_clip_weight_sha256='not_opened', expected_class_mapping_sha256=MAPPING_SHA,
        expected_split_sha256=SPLIT_SHA, expected_provenance=None, forbid_full_train_initializer=True)
    engine_path = ROOT / 'reference' / 'diagnose_stage2_tta.py'
    engine = import_file('p1_dependency_fourview_engine', engine_path)
    inspect.signature(engine.main).bind()
    inspect.signature(engine.validate_checkpoint_role).bind(Path('/not_opened/last.pt'), 'not_opened', {}, 'final')
    inspect.signature(engine.validate_final_train_only_payload).bind({}, {}, {}, manifest_sha256='not_opened',
        split_sha256=SPLIT_SHA, class_mapping_sha256=MAPPING_SHA, num_classes=750, train_count=118958,
        validation_count=29737, expected_total_count=148695)
    inspect.signature(engine.PairedViewDataset).bind([], 'not_constructed_transform', 'not_constructed_transform',
        required_partition='validation', allow_truncated_images=True)
    inspect.signature(engine.fixed_four_view_logits).bind('not_constructed_model', 'not_constructed_image', 'not_constructed_image')
    inspect.signature(engine.resolve_four_view_resize).bind(336, 'scale384')
    # Execute only the vendor main's parser-construction prefix, stopping before parse_args.
    # Vendor SHA is fixed; no config, CUDA check, lock or image read is in this prefix.
    tree = ast.parse(engine_path.read_text(encoding='utf-8'))
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main').body
    prefix = []
    for node in body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'parse_args'):
            break
        prefix.append(node)
    namespace = {'argparse': argparse, 'Path': Path}
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(engine_path), 'exec'), namespace)
    command = engine_argv(training_root / 'configs/experiment/p1_J5.yaml',
        training_root / 'artifacts/experiments/P1_J5/last.pt', '0' * 64,
        Path(declarations['canonical_lock']), ROOT / 'artifacts/diagnostics/not_run', 64, 4, 1800)
    parsed = namespace['parser'].parse_args(command[1:])
    if (parsed.checkpoint_role != 'final' or not parsed.four_view_scale384 or not parsed.save_fused_logits
            or parsed.max_samples is not None or parsed.six_view_scale448):
        raise ValueError('actual vendor CLI differs from frozen production argv')
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join([str(training_root / 'src'), str(training_root / 'scripts')])
    child = subprocess.run([sys.executable, str(engine_path), '--help'], cwd=ROOT, env=environment,
                           capture_output=True, text=True, timeout=30)
    if child.returncode or '--checkpoint-role {best,final}' not in child.stdout:
        raise RuntimeError('actual vendor subprocess --help failed: ' + child.stderr)
    return dict(schema='aictrack1.p1.cpu_entry_dependency_check.v1', complete=True,
        frozen_training_source_sha256=TRAINING_MANIFEST_SHA, static_imports=imports,
        real_vendor_cli_help_passed=True, real_vendor_production_argv_parsed=True,
        imports_and_signatures_verified=True, remote_assets_opened=False,
        independent_host_memory_snapshot=memory_snapshot(),
        model_loaded=False, checkpoint_opened=False, gpu_probed=False, images_opened=False,
        training_steps=0, test_access=False)


def fold_rows(assets):
    with Path(assets['assets']['manifest']['path']).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    with Path(assets['assets']['split']['path']).open(encoding='utf-8-sig', newline='') as handle:
        split_rows = list(csv.DictReader(handle))
    split = {r['sample_id']: r['split'] for r in split_rows}
    if (len(split) != len(split_rows) or len(rows) != 148695
            or len({r['sample_id'] for r in rows}) != len(rows)
            or set(split) != {r['sample_id'] for r in rows}):
        raise ValueError('complete one-to-one official manifest/split required')
    partitions = {'train': [], 'validation': []}
    for index, row in enumerate(rows):
        if row.get('source_split') != 'train' or row.get('decode_status') != 'ok':
            raise ValueError('diagnostic manifest must contain only audited official train')
        partition = split[row['sample_id']]
        if partition not in partitions:
            raise ValueError('unexpected fold partition')
        partitions[partition].append((index, row))
    if [len(partitions[k]) for k in ('train', 'validation')] != [118958, 29737]:
        raise ValueError('matching P1 fold counts required')
    for column in ('duplicate_group', 'near_duplicate_group'):
        train = {r[column] for _, r in partitions['train'] if r[column]}
        val = {r[column] for _, r in partitions['validation'] if r[column]}
        if train & val:
            raise ValueError('train/validation group overlap: ' + column)
    return partitions


def load_model(**kwargs):
    global FEATURE_COLLECTOR
    import torch
    import yaml
    from mainline_h1_contract import load_new_fold_head
    from p1_contract import assets, validate_config

    config = Path(kwargs['config_path']).resolve()
    checkpoint = Path(kwargs['checkpoint_path']).resolve()
    expected_config = TRAIN_ROOT / 'configs' / 'experiment' / f'p1_{CURRENT_ARM}.yaml'
    expected_checkpoint = TRAIN_ROOT / 'artifacts' / 'experiments' / f'P1_{CURRENT_ARM}' / 'last.pt'
    if config != expected_config or checkpoint != expected_checkpoint or kwargs['checkpoint_role'] != 'final':
        raise ValueError('P1 exact arm fixed-last path required')
    experiment = yaml.safe_load(config.read_text(encoding='utf-8'))['experiment']
    registered = assets(TRAIN_ROOT)
    validate_config(experiment, CURRENT_ARM, registered)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    metrics = json.loads((checkpoint.parent / 'metrics.json').read_text(encoding='utf-8'))
    audit = validate_checkpoint_metadata(state, metrics, experiment, arm=CURRENT_ARM,
                                        config_sha=sha(config), last_sha=sha(checkpoint))
    if sha(experiment['head_checkpoint']) != H1_SHA:
        raise ValueError('H1 initializer identity mismatch')
    loader = import_file('aictrack1.models.p1_fv2_audited_loader', ROOT / 'reference' / 'fv2_single_model.py')
    loader.load_classifier_head_checkpoint = load_new_fold_head
    device = kwargs['device']
    fitted = loader.load_diagnostic_model(**dict(kwargs, device=torch.device('cpu')))
    params = dict(fitted.model.named_parameters())
    names = list(state['trainable_state'])
    visual_names = [n for n in params if n.startswith('visual.')]
    audit.update(validate_joint_state(state['trainable_state'], visual_names=visual_names))
    if set(names) != set(params) or len(names) != JOINT_KEYS or audit['trainable_numel'] != JOINT_NUMEL:
        raise ValueError('joint state must restore all 154 model parameters exactly')
    for name in names:
        if not torch.equal(params[name].detach(), state['trainable_state'][name]):
            raise ValueError('restored model differs from final tensor: ' + name)
    if any(p.requires_grad for p in fitted.model.parameters()) or fitted.model.training:
        raise ValueError('diagnostic model must be frozen eval after joint restoration')
    audit.update(validate_optimizer_state(state, names, visual_names=visual_names))
    fold = fold_rows(registered)
    confidence = state['confidence_weighting']
    if (not torch.equal(confidence['sample_indices'], torch.tensor([i for i, _ in fold['train']]))
            or not torch.equal(confidence['class_ids'], torch.tensor([int(r['internal_class_index']) for _, r in fold['train']]))
            or not bool(torch.isfinite(confidence['weights']).all())
            or not bool(torch.all(confidence['seen'] == 1))):
        raise ValueError('AC exact global train ID/label binding or complete coverage failed')
    audit.update(schema='aictrack1.p1.fixed_last_diagnostic_audit.v1', p1_arm=CURRENT_ARM,
                 classifier_head_source='joint_fixed_last_state',
                 classifier_parameters_restored_equal=True, model_frozen_eval=True,
                 training_source_manifest_sha256=TRAINING_MANIFEST_SHA,
                 diagnostic_source_manifest_sha256=SOURCE_SHA,
                 exact_ac_train_id_label_binding=True, train_validation_group_overlap=0,
                 model_initialization_official_clip_sha256=registered['assets']['clip']['sha256'],
                 checkpoint_path=str(checkpoint), config_path=str(config),
                 inference_optimizer_steps=0, historical_control_cache_reused=True)
    save_json(OUTPUT / 'p1_checkpoint_audit.json', audit)
    if SAVE_FEATURES:
        # Observe the already required visual forwards; no added model invocation.
        FEATURE_COLLECTOR = FourViewFeatureCollector(fitted.model.visual, layout='four_calls_B')
    fitted.model.to(device)
    return dataclasses.replace(fitted, device=device)


def emit_identity(output, registered):
    import numpy as np
    import torch

    bundle_path = output / 'fused_logits.pt'
    cache = torch.load(bundle_path, map_location='cpu', weights_only=True, mmap=True)
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    audit_path = output / 'p1_checkpoint_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if (cache.get('view_protocol') != VIEW_PROTOCOL or cache.get('checkpoint_epoch') != 20
            or tuple(cache['logits'].shape) != (29737, 750) or report.get('smoke_only')
            or cache['logits'].dtype != torch.float16 or not bool(torch.isfinite(cache['logits']).all())
            or cache.get('checkpoint_sha256') != audit['fixed_last_sha256']):
        raise ValueError('only complete frozen four-view last20 cache can receive an identity')
    fold = fold_rows(registered)
    validation = [row for _, row in fold['validation']]
    expected_ids = [r['sample_id'] for r in validation]
    if (list(cache['sample_ids']) != expected_ids
            or not np.array_equal(cache['targets'].numpy(), [int(r['internal_class_index']) for r in validation])):
        raise ValueError('exported logits do not match exact manifest validation order')
    expected_counts = torch.bincount(torch.tensor([int(r['internal_class_index']) for _, r in fold['train']]), minlength=750)
    if not torch.equal(cache['train_class_counts'], expected_counts):
        raise ValueError('exported logits training class counts mismatch')
    with (output / 'paired_predictions.csv').open(encoding='utf-8', newline='') as handle:
        paired = list(csv.DictReader(handle))
    if (len(paired) != len(expected_ids) or [r['sample_id'] for r in paired] != expected_ids
            or not all('four_view_prediction' in r for r in paired)):
        raise ValueError('original FP32 four-view prediction receipt is missing/misaligned')
    row_path = output / 'rows.csv'
    with row_path.open('x', encoding='utf-8', newline='') as handle:
        fields = ['sample_id', 'target', 'group', 'relative_path', 'near_duplicate_group',
                  'duplicate_group', 'four_view_prediction']
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row, prediction in zip(validation, paired, strict=True):
            near, exact = row['near_duplicate_group'], row['duplicate_group']
            group = ('near:' + near) if near else ('exact:' + exact) if exact else ('sample:' + row['sample_id'])
            writer.writerow(dict(sample_id=row['sample_id'], target=row['internal_class_index'], group=group,
                                 relative_path=row['relative_path'], near_duplicate_group=near,
                                 duplicate_group=exact, four_view_prediction=prediction['four_view_prediction']))
    identity = dict(schema='aictrack1.p0_frozen_fold_input.v1', stage=2, holdout_verified=True,
        test_access=False, training_validation_overlap_rows=0, train_validation_exact_group_overlap=0,
        train_validation_near_group_overlap=0, checkpoint_role='last', upstream_checkpoint_role='final',
        checkpoint_epoch=20, checkpoint_path=audit['checkpoint_path'], checkpoint_sha256=cache['checkpoint_sha256'],
        experiment_config_sha256=cache['experiment_config_sha256'], split_sha256=cache['split_sha256'],
        manifest_sha256=cache['manifest_sha256'], class_mapping_sha256=cache['class_mapping_sha256'],
        view_protocol=cache['view_protocol'], h1_sha256=H1_SHA, training_count=118958,
        validation_count=29737, num_classes=750, training_scope='fixed_split_validation',
        bundle_path='fused_logits.pt', bundle_sha256=sha(bundle_path), logits_dtype='float16',
        row_csv='rows.csv', row_csv_sha256=sha(row_path),
        sample_ids_order_sha256=hashlib.sha256('\n'.join(expected_ids).encode('utf-8')).hexdigest(),
        source_provenance_receipt='p1_checkpoint_audit.json', source_receipt_sha256=sha(audit_path),
        report_sha256=sha(output / 'report.json'), p1_arm=CURRENT_ARM,
        joint_trainable_tensors=JOINT_KEYS, joint_trainable_numel=JOINT_NUMEL,
        classifier_head_source='joint_fixed_last_state',
        training_source_manifest_sha256=TRAINING_MANIFEST_SHA,
        diagnostic_source_manifest_sha256=SOURCE_SHA,
        inference_optimizer_steps=0, training_seed=20261008,
        group_source='official manifest near_duplicate_group, then duplicate_group; sample ID only for true singleton')
    identity['save_features'] = SAVE_FEATURES
    if SAVE_FEATURES:
        if FEATURE_COLLECTOR is None:
            raise ValueError('requested feature collector was not attached')
        features = FEATURE_COLLECTOR.finish(len(expected_ids))
        feature_path = output / 'fused_features.pt'
        feature_payload = dict(schema='aictrack1.stage2_train_derived_4v_features.v1',
            scope='Stage 2 train-derived fixed validation only; no test access',
            features=features, sample_ids=expected_ids, targets=cache['targets'],
            feature_protocol=FEATURE_PROTOCOL, observed_layout='four_calls_B',
            view_order=['native', 'native_exact_flip', 'scaled384_crop336', 'scaled_exact_flip'],
            checkpoint_sha256=cache['checkpoint_sha256'], experiment_config_sha256=cache['experiment_config_sha256'],
            split_sha256=cache['split_sha256'], manifest_sha256=cache['manifest_sha256'],
            class_mapping_sha256=cache['class_mapping_sha256'], view_protocol=VIEW_PROTOCOL,
            checkpoint_role='last', checkpoint_epoch=20, additional_model_forwards=0,
            observed_visual_calls=FEATURE_COLLECTOR.calls, test_access=False)
        temporary = feature_path.with_suffix('.pt.tmp')
        torch.save(feature_payload, temporary)
        temporary.replace(feature_path)
        checked = torch.load(feature_path, weights_only=True, map_location='cpu')
        if not torch.equal(checked['features'], features) or checked['sample_ids'] != expected_ids:
            raise ValueError('saved feature cache failed safe-load parity')
        identity.update(feature_file='fused_features.pt', feature_file_sha256=sha(feature_path),
            feature_dim=512, feature_dtype='float32', feature_construction_dtype='float32',
            feature_protocol=FEATURE_PROTOCOL, additional_model_forwards=0)
    save_json(output / 'input_identity.json', identity)
    save_json(output / 'result_manifest.json', dict(schema='aictrack1.p1.diagnostic_result.v1',
        p1_arm=CURRENT_ARM, complete=True, test_access=False,
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()}))


def main():
    global TRAIN_ROOT, CURRENT_ARM, OUTPUT, SOURCE_SHA, SAVE_FEATURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(ARMS))
    parser.add_argument('--expected-last-sha256')
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--hard-seconds', type=int, default=1800)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--attempt-name', choices=('primary', 'retry1'), default='primary')
    parser.add_argument('--min-free-gib', type=float, default=2.)
    parser.add_argument('--min-host-free-gib', type=float, default=4.)
    parser.add_argument('--dependency-check', action='store_true', help='CPU imports/signatures/vendor CLI only; never inference')
    parser.add_argument('--save-features', action='store_true', help='observe existing 4V visual forwards, no extra forward')
    parser.add_argument('--run', action='store_true', help='explicit root-operated CUDA inference')
    args = parser.parse_args()
    if args.dependency_check:
        if args.run:
            raise ValueError('dependency-check cannot run inference')
        result = dependency_probe(args.training_root)
        result['diagnostic_source_manifest_sha256'] = verify_owned_sources() if (ROOT / 'source_manifest.json').exists() else None
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.arm is None or args.expected_last_sha256 is None or args.deadline_unix is None:
        raise ValueError('GPU entry requires arm, expected last SHA, and deadline')
    if not args.run or not 60 <= args.hard_seconds <= 3600:
        raise ValueError('explicit --run and a bounded 60..3600 second budget required')
    if args.batch_size != 64 or args.workers not in (0, 4):
        raise ValueError('frozen inference batch64, workers0/4 required')
    TRAIN_ROOT = args.training_root.resolve()
    SAVE_FEATURES = args.save_features
    CURRENT_ARM = args.arm
    suffix = '' if args.attempt_name == 'primary' else '_retry1'
    OUTPUT = ROOT / 'artifacts' / 'diagnostics' / (f'P1_{CURRENT_ARM}' + suffix)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError('will not overwrite a diagnostic result: ' + str(OUTPUT))
    SOURCE_SHA = verify_owned_sources()
    if args.min_free_gib < 2. or shutil.disk_usage(ROOT).free < args.min_free_gib * 2**30:
        raise RuntimeError('at least 2GiB free and the declared disk guard are required')
    if sha(TRAIN_ROOT / 'source_manifest.json') != TRAINING_MANIFEST_SHA:
        raise ValueError('registered active P1 training manifest mismatch')
    sys.path[:0] = [str(TRAIN_ROOT / 'scripts'), str(TRAIN_ROOT / 'src')]
    from p1_contract import assets, verify_files, verify_remote_assets
    verify_files(TRAIN_ROOT)
    registered = verify_remote_assets(TRAIN_ROOT)
    if registered != json.loads((ROOT / 'inputs' / 'p1_assets.json').read_text(encoding='utf-8')):
        raise ValueError('diagnostic and training asset declarations differ')
    if memory_reason(memory_snapshot(), args.min_host_free_gib) is not None:
        raise RuntimeError('effective host/cgroup memory is below the independent diagnostic floor')
    from aictrack1.engine.training_queue import read_training_lock, probe_running_trainers
    lock = Path(registered['canonical_lock'])
    if read_training_lock(lock) or probe_running_trainers():
        raise RuntimeError('canonical GPU/train process lease is busy; diagnostic must wait')
    checkpoint = TRAIN_ROOT / 'artifacts' / 'experiments' / f'P1_{CURRENT_ARM}' / 'last.pt'
    if sha(checkpoint) != args.expected_last_sha256.lower():
        raise ValueError('root-selected terminal checkpoint SHA mismatch')
    seconds = min(args.hard_seconds, int(args.deadline_unix - time.time() - 120))
    if seconds < 60:
        raise TimeoutError('insufficient phase time for a complete diagnostic')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    engine = import_file('p1_audited_fourview_engine', ROOT / 'reference' / 'diagnose_stage2_tta.py')
    engine.ROOT = ROOT
    engine.load_diagnostic_model = load_model
    sys.argv = engine_argv(TRAIN_ROOT / 'configs' / 'experiment' / f'p1_{CURRENT_ARM}.yaml',
        checkpoint, args.expected_last_sha256, lock, OUTPUT, args.batch_size, args.workers, seconds)
    def cutoff(*_):
        raise TimeoutError('P1 diagnostic phase wall-clock ceiling reached')
    # Root runs this on Linux. The engine also checks the ceiling at batch boundaries.
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, cutoff)
        signal.alarm(seconds)
    try:
        result = engine.main()
        emit_identity(OUTPUT, assets(TRAIN_ROOT))
        print(json.dumps(dict(complete=True, p1_arm=CURRENT_ARM,
                             identity=str(OUTPUT / 'input_identity.json'), test_access=False)), flush=True)
        return result
    finally:
        if FEATURE_COLLECTOR is not None:
            FEATURE_COLLECTOR.close()
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


if __name__ == '__main__':
    raise SystemExit(main())
