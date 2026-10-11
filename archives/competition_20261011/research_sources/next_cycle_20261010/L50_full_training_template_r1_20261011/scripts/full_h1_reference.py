"""New-fold H1 final selection: positive row binding, never an old/full-train head."""
from __future__ import annotations

import hashlib
import json

SCOPE = 'full_train_fixed_epochs'
SPLIT_SHA = '0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'
MANIFEST_SHA = '1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA = '811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
HEAD_CONFIG = dict(epochs=100, batch_size=4096, learning_rate=.01, weight_decay=0.,
                   momentum=.9, gradient_clip_norm=None, seed=20261008,
                   optimizer='adam', scheduler='constant', amp=False)


def digest_rows(rows):
    return hashlib.sha256(json.dumps(rows, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def bind_rows(sample_ids, labels, partitions, records):
    """All rows and their order must match the official indexed manifest selection."""
    if not (len(sample_ids) == len(labels) == len(partitions) == len(records)):
        raise ValueError('feature/official record count differs')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('duplicate feature sample IDs')
    rows = []
    train_positions = []
    counts = {'train': 0, 'validation': 0}
    for position, (sid, target, part, record) in enumerate(zip(sample_ids, labels, partitions, records)):
        if (part not in counts or record.source != 'train' or sid != record.sample_id
                or int(target) != record.internal_class_index or part != record.partition):
            raise ValueError('official feature ID/label/partition/source mismatch')
        counts[part] += 1
        rows.append([sid, int(target), part])
        train_positions.append(position)
    if counts != {'train': 118958, 'validation': 29737}:
        raise ValueError('historical official cache partition mismatch')
    return train_positions, {
        'schema': 'aictrack1.mainline.h1.rows.v1', 'counts': {'train': 148695, 'validation': 0},
        'historical_cache_partition_counts': counts,
        'all_rows_sha256': digest_rows(rows),
        'train_rows_sha256': digest_rows([rows[i] for i in train_positions]),
        'validation_used_for_optimization_or_selection': False,
        'historical_validation_rows_now_part_of_full_official_train': True,
        'historical_split_used_only_for_cache_provenance': True,
    }


def validate_cache_metadata(metadata):
    # The real data API calls the Stage2 transform/data stage "rematch".
    # Competition provenance remains the independently verified "stage2" field.
    expected = dict(source='train', dataset_stage='rematch', manifest_sha256=MANIFEST_SHA,
                    split_sha256=SPLIT_SHA, class_mapping_sha256=MAPPING_SHA,
                    clip_weight_sha256=CLIP_SHA, feature_dim=512, normalized=True,
                    expected_record_count=148695, peft_checkpoint_sha256=None)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError('new H1 feature metadata mismatch: ' + key)
    pre = metadata.get('preprocessing', {})
    if pre.get('resize') != 336 or pre.get('center_crop') != 336:
        raise ValueError('H1 fixed336 features required')


def validate_final(state, expected_binding):
    if (state.get('training_scope') != SCOPE or state.get('epoch') != 100
            or state.get('config') != HEAD_CONFIG or state.get('head') != {'kind': 'linear'}
            or state.get('selection_rule') != 'fixed epoch100 last; full official train148695; no validation selection'
            or state.get('row_binding') != expected_binding):
        raise ValueError('new H1 final scope/config/selection/row binding mismatch')
    metadata = state.get('feature_cache_metadata', {})
    validate_cache_metadata(metadata)
    if metadata.get('training_scope') != SCOPE or metadata.get('training_record_count') != 148695:
        raise ValueError('H1 train-only scope/count mismatch')
    history = state.get('history')
    if not isinstance(history, list) or [r.get('epoch') for r in history] != list(range(1, 101)):
        raise ValueError('H1 incomplete full100 history')
    if any(set(row) != {'epoch', 'train_loss', 'learning_rate'} for row in history):
        raise ValueError('H1 validation/best-selection history forbidden')
    required = {'optimizer', 'scheduler', 'scaler', 'python_rng_state', 'numpy_rng_state',
                'torch_rng_state', 'cuda_rng_state_all', 'model', 'loss_config',
                'class_weights', 'best_model', 'best_top1'}
    if not required.issubset(state) or state['best_model'] is not None:
        raise ValueError('H1 complete recovery state required; no selected best')
    if (state.get('competition_stage') != 'stage2' or state.get('split_sha256') != SPLIT_SHA
            or state.get('train_manifest_sha256') != MANIFEST_SHA
            or state.get('class_mapping_sha256') != MAPPING_SHA
            or state.get('clip_weight_sha256') != CLIP_SHA):
        raise ValueError('H1 positive provenance mismatch')


def load_new_fold_head(path, **kwargs):
    """Overlay used only by the independently frozen mainline source copy."""
    import torch
    from aictrack1.models.heads import load_classifier_head_checkpoint
    head, state = load_classifier_head_checkpoint(path, **kwargs)
    binding = state.get('row_binding')
    if not isinstance(binding, dict) or binding.get('counts') != {'train': 148695, 'validation': 0}:
        raise ValueError('new H1 exact row binding required')
    validate_final(state, binding)
    if set(state['model']) != {'linear.weight', 'linear.bias'}:
        raise ValueError('single original linear H1 only')
    if any(not torch.isfinite(t).all() for t in state['model'].values()):
        raise ValueError('nonfinite H1')
    return head, state
