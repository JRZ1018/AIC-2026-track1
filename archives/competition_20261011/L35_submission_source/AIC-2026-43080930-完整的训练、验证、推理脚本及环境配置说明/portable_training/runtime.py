"""Path, asset and resource adaptation for the frozen L35 trainer.

The archived train_n1_peft.py is loaded byte-for-byte. Only its deployment
contracts (remote lease/root/budget) are provided by this explicit local adapter.
Losses, transforms, AC, optimizer, AMP, schedule, RNG and checkpoint loop remain
the original functions. Portable run receipts identify new reproduction runs;
they do not impersonate the original competition run.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
PACKAGE = CODE.parent
LIBRARY = PACKAGE / 'AIC-2026-43080930-可复现的代码与运行说明文档'
MODELS = PACKAGE / 'AIC-2026-43080930-可执行模型文件或容器化复现环境'
DATA = CODE / 'data' / 'metadata'
SOURCE = LIBRARY / 'archived' / 'full_source'
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MANIFEST_SHA = '1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
TRAINER_SHA = '7ef3cb850fc91c876bbb5a17e80e13f8f4fb3504b45a29b4f67f3e977d5f1066'
CONFIRMATION_SHA = '15fd22f3c36b65dc90250692538ca0e2c3fe8a581f36be660fad05e72a061546'
SPLITS = {'full': DATA / 'full_split.csv', 'first': DATA / 'first_query_split.csv',
          'second': DATA / 'second_query_split.csv'}
HEAD_CONFIG = dict(epochs=100, batch_size=4096, learning_rate=.01,
    weight_decay=0., momentum=.9, gradient_clip_norm=None, seed=20261008,
    optimizer='adam', scheduler='constant', amp=False)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def bootstrap():
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # The actual L35 run records unknown_archive; file hashes bind its source.
    for value in (LIBRARY / 'vendor', LIBRARY / 'src', SOURCE / 'scripts', CODE / 'scripts'):
        sys.path.insert(0, str(value))
    import torch
    torch.set_num_threads(4)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_immutable(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ValueError(f'Reproduction configuration changed; use a new work directory: {path}')
    else:
        path.write_text(text, encoding='utf-8')


def make_dataset(data_root, output, split):
    import yaml
    value = yaml.safe_load((SOURCE / 'configs/data/p1_336.yaml').read_text(encoding='utf-8'))
    value['dataset']['dataset_root'] = str(Path(data_root).resolve())
    value['dataset']['class_mapping'] = str(DATA / 'class_mapping.csv')
    value['dataset']['audit']['output_root'] = str(Path(output).resolve() / 'audit')
    value['dataset']['sources']['test']['root'] = '__unavailable_test__'
    target = Path(output).resolve() / 'runtime' / 'data.yaml'
    write_immutable(target, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
    return target


def check_disk(work, minimum, payload=None):
    reserve = 0
    if payload is not None:
        import torch
        seen = set()
        def visit(item):
            nonlocal reserve
            if isinstance(item, torch.Tensor):
                storage = item.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes(), str(item.device))
                if key not in seen:
                    seen.add(key)
                    reserve += storage.nbytes()
            elif isinstance(item, dict):
                for child in item.values(): visit(child)
            elif isinstance(item, (tuple, list)):
                for child in item: visit(child)
        visit(payload)
        reserve += 64 * 1024**2
    free = shutil.disk_usage(work).free
    if free < minimum + reserve:
        raise RuntimeError(f'Insufficient free space for atomic checkpoint: free={free}, reserve={reserve}, minimum={minimum}')


def install_adapter(config, work, head_path, minimum_free_bytes):
    import yaml
    import torch
    raw = yaml.safe_load(Path(config).read_text(encoding='utf-8'))['experiment']
    head_path = Path(head_path).resolve()
    gate = SOURCE / 'inputs/fresh_confirmation.json'
    if sha(SOURCE / 'scripts/train_n1_peft.py') != TRAINER_SHA or sha(gate) != CONFIRMATION_SHA:
        raise ValueError('Exact archived L35 trainer and typed confirmation are required')
    historical = json.loads(gate.read_text(encoding='utf-8'))
    if (historical.get('schema') != 'aictrack1.next_cycle10.L35_confirmation.v1' or
            historical.get('status') != 'complete_positive_independent_L35_confirmation' or
            historical.get('selected_arm') != 'L35' or historical.get('selected_route') != 'L35_GRAPH' or
            historical.get('visual_learning_rate') != 3.5e-6 or historical.get('test_access') is not False):
        raise ValueError('Archived L35 confirmation has another route or evidence type')
    if (raw['training']['visual_learning_rate'] != 3.5e-6 or raw['training']['head_learning_rate'] != 5e-6 or
            raw['training']['epochs'] != 20 or raw['training']['batch_size'] != 64 or
            raw['training']['weight_decay'] != .01 or raw['training']['gradient_clip_norm'] != 1. or
            raw['selection_rule'] != 'fixed_final' or raw['peft']['mode'] != 'full_visual'):
        raise ValueError('Portable configuration differs from the fixed L35 joint20 recipe')
    binding = dict(schema='aictrack1.portable.L35.training.v1',
        scope=raw['training_scope'], source_trainer_sha256=sha(SOURCE / 'scripts/train_n1_peft.py'),
        adapter_sha256=sha(Path(__file__)), config_sha256=sha(config),
        head_sha256=sha(head_path), manifest_sha256=sha(DATA / 'train_manifest.csv'),
        split_sha256=sha(raw['split']), competition_stage='stage2', test_access=False,
        original_cloud_run_replayed=False, minimum_free_bytes=minimum_free_bytes)
    identity = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    write_immutable(Path(work) / 'portable_binding.json', json.dumps(binding, indent=2, sort_keys=True) + '\n')
    os.environ['P1_SOURCE_SHA'] = identity
    os.environ['P1_ATTEMPT_ID'] = 'portable-L35-reproduction'
    os.environ.pop('P1_STOP_AFTER_EPOCH', None)
    contract = types.ModuleType('p1_contract')
    contract.assets = lambda: dict(selected_route='L35_GRAPH',
        fresh_confirmation_receipt_sha256=sha(gate), full_budget_sha256=identity)
    sys.modules['p1_contract'] = contract
    guard = types.ModuleType('mainline_guard')
    guard.sha = sha
    guard.entry = lambda root, phase, resume, config=None: check_disk(work, minimum_free_bytes)
    guard.reserve_save = lambda payload: check_disk(work, minimum_free_bytes, payload)
    receipt_path = head_path.parent / 'h1_receipt.json'
    def receipt():
        if raw['training_scope'] == 'full_train_fixed_epochs':
            return dict(full_train_only=True, validation_count=0)
        if not receipt_path.is_file():
            raise ValueError('Heldout initialization needs h1_receipt.json beside its train-only H1')
        result = json.loads(receipt_path.read_text(encoding='utf-8'))
        if result.get('checkpoint_sha256') != sha(head_path) or result.get('validation_selection') is not False:
            raise ValueError('Train-only H1 receipt/checkpoint binding mismatch')
        return result
    guard.h1_receipt = receipt
    sys.modules['mainline_guard'] = guard
    head_contract = types.ModuleType('mainline_h1_contract')
    def load_head(path, **kwargs):
        from aictrack1.models.heads import load_classifier_head_checkpoint
        if Path(path).resolve() != head_path:
            raise ValueError('Head path changed after source binding')
        head, state = load_classifier_head_checkpoint(path, **kwargs)
        if state.get('epoch') != 100 or state.get('head') != {'kind': 'linear'}:
            raise ValueError('L35 requires fixed100 original linear H1')
        if state.get('config') != HEAD_CONFIG or state.get('best_model') is not None:
            raise ValueError('H1 must retain the fixed Adam/constant100 configuration without best selection')
        if [r.get('epoch') for r in state.get('history', [])] != list(range(1, 101)):
            raise ValueError('H1 complete fixed100 history is required')
        required = {'optimizer', 'scheduler', 'scaler', 'python_rng_state', 'numpy_rng_state',
                    'torch_rng_state', 'cuda_rng_state_all', 'row_binding'}
        if not required.issubset(state) or state['scheduler'].get('last_epoch') != 100:
            raise ValueError('H1 requires its complete optimizer/scheduler/scaler/RNG state')
        meta = state['feature_cache_metadata']
        if meta.get('source') != 'train' or meta.get('normalized') is not True or meta.get('peft_checkpoint_sha256') is not None:
            raise ValueError('H1 must come from normalized official frozen-CLIP train features')
        expected_scope = raw['training_scope']
        if state.get('training_scope') != expected_scope or meta.get('training_scope') != expected_scope:
            raise ValueError('H1 scope differs from new training scope; full H1 forbidden for heldout')
        expected_count = 148695 if expected_scope == 'full_train_fixed_epochs' else 118958
        if meta.get('training_record_count') != expected_count:
            raise ValueError('H1 training count differs from official scope')
        expected_validation = 0 if expected_scope == 'full_train_fixed_epochs' else 29737
        if state['row_binding'].get('counts') != {'train': expected_count, 'validation': expected_validation}:
            raise ValueError('H1 positive row binding differs from official scope')
        if any(not torch.isfinite(t).all() for t in state['model'].values()):
            raise ValueError('H1 nonfinite tensors')
        return head, state
    head_contract.load_new_fold_head = load_head
    sys.modules['mainline_h1_contract'] = head_contract
    return binding
