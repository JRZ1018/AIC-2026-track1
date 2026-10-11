"""Strict CPU-only contracts for S-on-C152 fixed-last diagnostic and cache identities."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
TRAINING_MANIFEST_SHA = 'f473415f22fad41f3c8f3cb64164a0b777835661cb03a54723e9a80a25366ca9'
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
H1_SHA = '45266a98329a59f708b12f394775640e032e50be8c44077a1037048f28c88753'
SPLIT_SHA = '0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'
MANIFEST_SHA = '1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA = '811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
CONTROL_LAST_SHA = '078bdae1654cecc076d19e8e67b2401c6a47b985d90c91f7b6ec25293002b6c9'
CONTROL_CACHE_SHA = '4813ad6727624f3072ff6e5d28d8ecaa9b4a9e919a5e0b489946dfeea08180b9'
CONTROL_CONFIG_SHA = '58766198d816efc7655cb21079a70cde00459c71b8948e762bcfdfda048f68d8'
VIEW_PROTOCOL = ('native336 center and exact flip plus resize_short_side384_center_crop336 '
                 'and exact flip; one checkpoint; four float logits equally averaged')
HEAD_KEYS = {'classifier.linear.weight', 'classifier.linear.bias'}
VISUAL_KEYS = 152
VISUAL_NUMEL = 87849216
ARMS = {'S': 0.0}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_owned_sources(root=ROOT):
    root = Path(root).resolve()
    path = root / 'source_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'aictrack1.supervision.diagnostic_source.v1':
        raise ValueError('diagnostic source manifest schema mismatch')
    if manifest.get('training_source_manifest_sha256') != TRAINING_MANIFEST_SHA or len(TRAINING_MANIFEST_SHA) != 64:
        raise ValueError('diagnostic must bind a positively frozen S source')
    for name, digest in manifest['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not (root / rel).resolve().is_relative_to(root):
            raise ValueError('source manifest path escape')
        if sha(root / rel) != digest:
            raise ValueError('diagnostic source drift: ' + name)
    return sha(path)


def validate_visual_state(state, *, visual_names=None, visual_count=VISUAL_KEYS, visual_numel=VISUAL_NUMEL):
    if not isinstance(state, dict) or not state:
        raise ValueError('missing trainable_state')
    names = set(state)
    if len(names) != visual_count or any(not n.startswith('visual.') for n in names):
        raise ValueError('S must contain every visual tensor and no head/readout tensor')
    if visual_names is not None and names != set(visual_names):
        raise ValueError('S does not restore every visual parameter')
    if any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32
           or not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('S visual state must contain finite FP32 master tensors')
    if sum(v.numel() for v in state.values()) != visual_numel:
        raise ValueError('S visual numel mismatch')
    return dict(trainable_tensors=len(names), trainable_numel=sum(v.numel() for v in state.values()),
                visual_tensors=len(names), head_tensors=0)


def validate_frozen_head(model, h1_state):
    expected = h1_state.get('model', {})
    if set(expected) != {'linear.weight', 'linear.bias'}:
        raise ValueError('H1 must be an exact single linear head')
    restored = model.classifier.state_dict()
    if set(restored) != set(expected) or any(restored[k].dtype != expected[k].dtype
            or not torch.equal(restored[k].detach().cpu(), expected[k].cpu()) for k in expected):
        raise ValueError('S frozen classifier differs from the original H1 tensors')
    if any(p.requires_grad for p in model.classifier.parameters()):
        raise ValueError('S head must remain frozen')
    return True


def validate_checkpoint_metadata(state, metrics, experiment, *, arm, config_sha, last_sha):
    if arm not in ARMS:
        raise ValueError('diagnostic only accepts S-on-C alpha0.5')
    expected = dict(competition_stage='stage2', training_scope='fixed_split_validation',
                    mode='full_visual', visual_input_resolution=336,
                    position_interpolation='bicubic_7x7_to_10x10', epoch=20,
                    config_sha256=config_sha, train_manifest_sha256=MANIFEST_SHA,
                    split_sha256=SPLIT_SHA, class_mapping_sha256=MAPPING_SHA,
                    clip_weight_sha256=CLIP_SHA, num_classes=750,
                    supervision_source_manifest_sha256=TRAINING_MANIFEST_SHA,
                    supervision_arm='S', classifier_head_source='original_fold_H1_frozen',
                    classifier_head_restored_equal=True)
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError('S-on-C152 fixed-last metadata mismatch: ' + key)
    if (metrics.get('last_checkpoint_sha256') != last_sha
            or metrics.get('training_scope') != 'fixed_split_validation'
            or metrics.get('train_count') != 118958 or metrics.get('validation_count') != 29737
            or metrics.get('head_initializer', {}).get('sha256') != H1_SHA
            or metrics.get('base_loss') != {'kind':'sce','alpha':.5,'beta':1.,'log_zero':-4.}
            or metrics.get('trainable_total_parameters') != VISUAL_NUMEL
            or metrics.get('early_stop') is not None):
        raise ValueError('incomplete or mismatched terminal metrics')
    history = state.get('history')
    if (not isinstance(history, list) or history != metrics.get('history')
            or [r.get('epoch') for r in history] != list(range(1, 21))
            or any('validation' not in r for r in history)):
        raise ValueError('complete identical twenty-epoch history required')
    train = experiment.get('training', {})
    if (experiment.get('peft') != {'mode': 'full_visual', 'lora_blocks': 8}
            or experiment.get('selection_rule') != 'fixed_final'
            or experiment.get('token_readout', {}).get('mode') != 'none'
            or train.get('epochs') != 20 or train.get('seed') != 20261008
            or train.get('visual_learning_rate') != 1.75e-6
            or train.get('head_learning_rate') != 0.0
            or experiment.get('base_loss') != {'kind': 'sce', 'alpha': .5, 'beta': 1., 'log_zero': -4.}
            or state.get('token_readout') != experiment.get('token_readout')):
        raise ValueError('P1 fixed algorithm/configuration mismatch')
    if state.get('head_initializer_sha256', H1_SHA) != H1_SHA:
        raise ValueError('wrong head initializer')
    for forbidden in ('persistent_filter_state', 'bootstrap_curriculum_state', 'trainable_ema', 'elr_state'):
        if forbidden in state:
            raise ValueError('unexpected non-P1 algorithm state: ' + forbidden)
    return dict(checkpoint_role='last', checkpoint_epoch=20, epoch=20, fixed_last_sha256=last_sha, checkpoint_sha256=last_sha,
                config_sha256=config_sha, split_sha256=SPLIT_SHA, h1_sha256=H1_SHA,
                train_count=118958, validation_count=29737, training_seed=20261008,
                h1_initializer_sha256=H1_SHA, test_access=False, history_epochs=20)


def validate_optimizer_state(state, names, *, visual_names):
    optimizer = state.get('optimizer', {})
    groups = optimizer.get('param_groups', [])
    ordered_visual = [n for n in names if n in set(visual_names)]
    ordered_head = [n for n in names if n in HEAD_KEYS]
    if len(groups) != 1 or [len(g['params']) for g in groups] != [len(ordered_visual)] or ordered_head:
        raise ValueError('S optimizer must contain exactly one complete visual-only group')
    ids = [i for g in groups for i in g['params']]
    if len(set(ids)) != len(ids) or len(optimizer.get('state', {})) != len(names):
        raise ValueError('optimizer state count/IDs mismatch')
    for name, pid in zip(ordered_visual + ordered_head, ids, strict=True):
        entry = optimizer['state'].get(pid, {})
        for field in ('exp_avg', 'exp_avg_sq'):
            value = entry.get(field)
            if (not isinstance(value, torch.Tensor) or value.shape != state['trainable_state'][name].shape
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('optimizer shape/nonfinite mismatch: ' + name + '/' + field)
        step = entry.get('step')
        if isinstance(step, torch.Tensor):
            step = float(step)
        if not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0:
            raise ValueError('missing positive optimizer step: ' + name)
    scheduler = state.get('scheduler', {})
    if scheduler.get('T_max') != 20 or scheduler.get('last_epoch') != 20:
        raise ValueError('terminal cosine schedule must complete the uncompressed twenty epochs')
    scale = state.get('scaler', {}).get('scale')
    if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('missing finite positive AMP scaler')
    confidence = state.get('confidence_weighting', {})
    if confidence.get('epochs_updated') != 20:
        raise ValueError('formal AC must have completed twenty observation epochs')
    return dict(optimizer_parameter_groups=[len(ordered_visual)], optimizer_moments_finite=True,
                scheduler_epoch=20, ac_observation_epochs=20)


def validate_candidate_audit(identity, audit):
    expected = dict(schema='aictrack1.supervision.fixed_last_diagnostic_audit.v1', complete=True,
        status='complete_fixed_last20_audit', model_arm='S', checkpoint_role='last', checkpoint_epoch=20,
        fixed_last_sha256=identity.get('checkpoint_sha256'), trainable_tensors=VISUAL_KEYS,
        trainable_numel=VISUAL_NUMEL, classifier_parameters_restored_equal=True,
        classifier_head_source='old_fold_h1_frozen', exact_ac_train_id_label_binding=True,
        train_count=118958, validation_count=29737, sce_alpha=.5,
        model_frozen_eval=True, inference_optimizer_steps=0, test_access=False,
        training_source_manifest_sha256=TRAINING_MANIFEST_SHA,
        diagnostic_source_manifest_sha256=identity.get('diagnostic_source_manifest_sha256'))
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ValueError('S candidate audit incomplete/mismatched: ' + key)
    for field in ('config_sha256','split_sha256','h1_sha256'):
        value = identity['experiment_config_sha256'] if field == 'config_sha256' else identity.get(field)
        if audit.get(field) != value:
            raise ValueError('S identity/audit field mismatch: ' + field)
    if (identity.get('trainable_tensors') != VISUAL_KEYS or identity.get('trainable_numel') != VISUAL_NUMEL
            or identity.get('classifier_head_source') != 'old_fold_h1_frozen'
            or identity.get('training_source_manifest_sha256') != TRAINING_MANIFEST_SHA):
        raise ValueError('S identity must retain visual-only and frozen-head provenance')
    return True


def load_cache(bundle_path, identity_path):
    """No images, model construction, GPU, or unsafe pickle loading."""
    identity_path = Path(identity_path).resolve()
    identity = json.loads(identity_path.read_text(encoding='utf-8-sig'))
    if (identity.get('schema') != 'aictrack1.p0_frozen_fold_input.v1'
            or identity.get('stage') != 2 or identity.get('holdout_verified') is not True
            or identity.get('checkpoint_role') not in ('last', 'final')
            or identity.get('checkpoint_epoch') != 20
            or identity.get('training_scope') != 'fixed_split_validation'
            or identity.get('training_validation_overlap_rows') != 0
            or identity.get('test_access') is not False):
        raise ValueError('cache requires verified Stage2 fixed-last20 holdout provenance')
    digest = identity.get('source_receipt_sha256', '')
    if len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
        raise ValueError('missing source audit receipt SHA')
    receipt_path = Path(identity['source_provenance_receipt'])
    if not receipt_path.is_absolute():
        receipt_path = identity_path.parent / receipt_path
    if sha(receipt_path) != digest:
        raise ValueError('source audit receipt SHA mismatch')
    if identity.get('model_arm') == 'S':
        audit = json.loads(receipt_path.read_text(encoding='utf-8'))
        validate_candidate_audit(identity, audit)
    elif identity.get('bundle_sha256') != CONTROL_CACHE_SHA:
        raise ValueError('only S and the pinned historical fixed-last C cache are allowed')
    if sha(bundle_path) != identity.get('bundle_sha256'):
        raise ValueError('cache SHA mismatch')
    cache = torch.load(bundle_path, map_location='cpu', weights_only=True, mmap=True)
    if (cache.get('schema') != 'aictrack1.stage2_train_derived_fused_logits.v1'
            or 'train-derived' not in cache.get('scope', '')
            or cache.get('checkpoint_role') != 'final' or cache.get('checkpoint_epoch') != 20):
        raise ValueError('cache is not the train-derived schema')
    for field in ('checkpoint_sha256', 'split_sha256', 'manifest_sha256', 'class_mapping_sha256',
                  'experiment_config_sha256', 'view_protocol'):
        if cache.get(field) != identity.get(field):
            raise ValueError('cache/identity mismatch: ' + field)
    if identity.get('view_protocol') != VIEW_PROTOCOL:
        raise ValueError('requires the frozen equally averaged four-view336/384 protocol')
    scores = cache['logits'].float().numpy()
    targets = cache['targets'].long().numpy()
    counts = cache['train_class_counts'].long().numpy()
    if (cache['logits'].dtype != torch.float16 or cache['targets'].dtype != torch.int64 or cache['train_class_counts'].dtype != torch.int64 or scores.shape != (29737, 750) or targets.shape != (29737,) or counts.shape != (750,)
            or not np.isfinite(scores).all() or np.any(counts <= 0)
            or int(counts.sum()) != 118958 or np.any(targets < 0) or np.any(targets >= 750)
            or identity.get('training_count') != 118958 or identity.get('validation_count') != 29737):
        raise ValueError('invalid full folded cache counts/shapes/scores')
    row_path = Path(identity['row_csv'])
    if not row_path.is_absolute():
        row_path = identity_path.parent / row_path
    if sha(row_path) != identity['row_csv_sha256']:
        raise ValueError('row CSV SHA mismatch')
    with row_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    ids = [r['sample_id'] for r in rows]
    groups = [r['group'] for r in rows]
    if (len(rows) != len(scores) or len(set(ids)) != len(ids) or not all(ids) or not all(groups)
            or list(cache.get('sample_ids', [])) != ids
            or not np.array_equal(targets, [int(r['target']) for r in rows])):
        raise ValueError('exact unique ID/target/group order binding failed')
    if hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest() != identity.get('sample_ids_order_sha256'):
        raise ValueError('verified sample ID order digest mismatch')
    rounding_mismatches = None
    if 'four_view_prediction' in rows[0]:
        rounding_mismatches = int(np.sum(scores.argmax(1) != [int(r['four_view_prediction']) for r in rows]))
    return dict(scores=scores, targets=targets, train_counts=counts, ids=ids, groups=groups,
                identity=identity, provenance=dict(identity_path=str(identity_path),
                identity_sha256=sha(identity_path), bundle_sha256=sha(bundle_path),
                cached_logits_dtype=str(cache['logits'].dtype), decode_dtype='float32',
                fp16_argmax_mismatches_vs_original_fp32=rounding_mismatches,
                precision_note='FP32 conversion cannot restore discarded FP16 precision'))


def validate_pair(candidate, control):
    c, j = control['identity'], candidate['identity']
    if (c.get('checkpoint_sha256') != CONTROL_LAST_SHA or c.get('bundle_sha256') != CONTROL_CACHE_SHA
            or c.get('experiment_config_sha256') != CONTROL_CONFIG_SHA):
        raise ValueError('control must be the accepted FV2 fullvisual20 fixed-last cache')
    for key, expected in (('split_sha256', SPLIT_SHA), ('manifest_sha256', MANIFEST_SHA),
                          ('class_mapping_sha256', MAPPING_SHA), ('h1_sha256', H1_SHA),
                          ('view_protocol', VIEW_PROTOCOL)):
        if c.get(key) != expected or j.get(key) != expected:
            raise ValueError('paired input mismatch: ' + key)
    if (j.get('model_arm') != 'S' or j.get('trainable_tensors') != VISUAL_KEYS
            or j.get('trainable_numel') != VISUAL_NUMEL
            or j.get('classifier_head_source') != 'old_fold_h1_frozen'
            or j.get('training_source_manifest_sha256') != TRAINING_MANIFEST_SHA):
        raise ValueError('candidate must be the audited S visual152 frozen-H1 fixed-last20 checkpoint')
    if (candidate['ids'] != control['ids'] or candidate['groups'] != control['groups']
            or not np.array_equal(candidate['targets'], control['targets'])
            or not np.array_equal(candidate['train_counts'], control['train_counts'])):
        raise ValueError('paired images/groups/labels/training prior must be identical in order')
    return True


def validate_ac_state(confidence, sample_indices, class_ids, raw):
    from aictrack1.methods.confidence import ClasswiseHistoryWeighting
    restored = ClasswiseHistoryWeighting(torch.as_tensor(sample_indices), torch.as_tensor(class_ids),
        **{k: raw[k] for k in ('momentum','history_mix','min_weight','magnitude_mix','tail_class_fraction',
        'tail_min_mean_weight','min_effective_mass_per_class') if k in raw})
    restored.load_state_dict(confidence)
    if any(isinstance(v,torch.Tensor) and v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in confidence.values()):
        raise ValueError('AC contains nonfinite state')
    if restored.epochs_updated != 20 or not bool(torch.all(restored.seen == 1)):
        raise ValueError('AC needs complete twenty-epoch coverage')
    return restored
