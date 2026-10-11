"""Meaningful CPU failure tests; no official model, images, CUDA or optimizer step."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT.parent / 'training_source'
sys.path[:0] = [str(ROOT), str(TRAINING / 'src'), str(TRAINING / 'scripts')]
from p1_diag_contract import (CLIP_SHA, HEAD_KEYS, H1_SHA, MANIFEST_SHA, MAPPING_SHA,
    SPLIT_SHA, CONTROL_CACHE_SHA, CONTROL_LAST_SHA, CONTROL_CONFIG_SHA, VIEW_PROTOCOL,
    TRAINING_MANIFEST_SHA, validate_checkpoint_metadata, validate_joint_state,
    validate_optimizer_state, validate_pair, sha)
from compare_p1 import build_queries, compare_arrays, eligibility


def tiny_state():
    return {'visual.weight': torch.eye(2), 'classifier.linear.weight': torch.arange(6).float().reshape(3, 2),
            'classifier.linear.bias': torch.tensor([1., 2., 3.])}


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.linear(x)


class DiagnosticTests(unittest.TestCase):
    def test_complete_head_state_and_fail_closed(self):
        state = tiny_state()
        validate_joint_state(state, visual_names=['visual.weight'], num_classes=3, feature_dim=2,
                             visual_count=1, visual_numel=4)
        for mutate in (lambda x: x.pop('classifier.linear.bias'),
                       lambda x: x.__setitem__('classifier.linear.bias', torch.zeros(4)),
                       lambda x: x.__setitem__('classifier.linear.bias', torch.tensor([1., float('nan'), 3.])),
                       lambda x: x.__setitem__('extra.weight', torch.zeros(1))):
            corrupt = copy.deepcopy(state)
            mutate(corrupt)
            with self.assertRaises(ValueError):
                validate_joint_state(corrupt, visual_names=['visual.weight'], num_classes=3, feature_dim=2,
                                     visual_count=1, visual_numel=4)

    def test_actual_fv2_loader_restores_joint_head_instead_of_initializer(self):
        # Ordinary inherited-ACL fixtures retained privately for audit; no tempfile chmod or recursive deletion.
        fixture = ROOT / 'tests' / 'fixtures'
        fixture.mkdir(exist_ok=True)
        config = fixture / 'joint.yaml'
        checkpoint_path = fixture / 'last.pt'
        clip_path = fixture / 'toy_clip.pt'
        clip_path.write_bytes(b'unit-test synthetic official-loader replacement')
        document = dict(experiment=dict(run_id='cpu_joint_test', model_config='model.yaml', head_checkpoint='head.pt',
            peft={'mode': 'full_visual', 'lora_blocks': 8}, training={'epochs': 20}, visual_input_resolution=336,
            position_interpolation='bicubic_7x7_to_10x10', num_classes=3))
        config.write_text(yaml.safe_dump(document), encoding='utf-8')
        (fixture / 'model.yaml').write_text(yaml.safe_dump(dict(model=dict(provider='OpenAI', name='ViT-B/32',
            weight_path=str(clip_path), expected_weight_sha256=sha(clip_path), runtime_downloads_allowed=False))), encoding='utf-8')
        state = dict(config_sha256=sha(config), mode='full_visual', visual_input_resolution=336,
            position_interpolation='bicubic_7x7_to_10x10', class_mapping_sha256='toy_mapping', num_classes=3,
            clip_weight_sha256=sha(clip_path), trainable_state=tiny_state())
        path = ROOT / 'reference' / 'fv2_single_model.py'
        spec = importlib.util.spec_from_file_location('aictrack1.models.p1_cpu_actual_fv2_loader', path)
        loader = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loader
        spec.loader.exec_module(loader)
        def clip_factory(*args):
            return SimpleNamespace(visual=nn.Linear(2, 2, bias=False)), None, SimpleNamespace(weight_sha256=sha(clip_path), feature_dim=2)
        def configure(visual, *args, **kwargs):
            for p in visual.parameters():
                p.requires_grad_(True)
        def head_factory(*args, **kwargs):
            return Head(), {'feature_cache_metadata': {'normalized': True}}
        kwargs = dict(config_path=config, checkpoint_path=checkpoint_path, class_mapping_digest='toy_mapping',
                      num_classes=3, split_sha256='toy_split', device=torch.device('cpu'),
                      git_commit='synthetic_cpu', checkpoint_role='final')
        with patch.object(loader, 'load_clip_vit_b32', clip_factory), \
             patch.object(loader, 'configure_visual_peft', configure), \
             patch.object(loader, 'enable_interpolated_visual_resolution', lambda *a, **k: None), \
             patch.object(loader, 'load_classifier_head_checkpoint', head_factory):
            torch.save(state, checkpoint_path)
            fitted = loader.load_diagnostic_model(**kwargs)
            self.assertTrue(torch.equal(fitted.model.classifier.linear.weight, state['trainable_state']['classifier.linear.weight']))
            self.assertTrue(torch.equal(fitted.model.classifier.linear.bias, state['trainable_state']['classifier.linear.bias']))
            self.assertFalse(fitted.model.training)
            self.assertTrue(all(not p.requires_grad for p in fitted.model.parameters()))
            x = torch.tensor([[1., 0.]])
            self.assertTrue(torch.equal(fitted.model(x), torch.tensor([[1., 4., 7.]])))
            for bad in ('missing_bias', 'extra_key'):
                broken = copy.deepcopy(state)
                if bad == 'missing_bias':
                    broken['trainable_state'].pop('classifier.linear.bias')
                else:
                    broken['trainable_state']['classifier.extra.weight'] = torch.zeros(1)
                torch.save(broken, checkpoint_path)
                with self.assertRaises(ValueError):
                    loader.load_diagnostic_model(**kwargs)
            torch.save(state, checkpoint_path)

    def test_fixed_last_metadata_rejects_epoch8_wrong_split_and_best_metrics(self):
        config_sha, last_sha = '1' * 64, '2' * 64
        history = [dict(epoch=i, validation={'all': {'top1': .5}}) for i in range(1, 21)]
        state = dict(competition_stage='stage2', training_scope='fixed_split_validation', mode='full_visual',
            visual_input_resolution=336, position_interpolation='bicubic_7x7_to_10x10', epoch=20,
            config_sha256=config_sha, train_manifest_sha256=MANIFEST_SHA, split_sha256=SPLIT_SHA,
            class_mapping_sha256=MAPPING_SHA, clip_weight_sha256=CLIP_SHA, num_classes=750,
            history=history, token_readout={'mode': 'none'})
        metrics = dict(last_checkpoint_sha256=last_sha, training_scope='fixed_split_validation',
                       train_count=118958, validation_count=29737, early_stop=None, history=history)
        experiment = dict(peft={'mode': 'full_visual', 'lora_blocks': 8}, selection_rule='fixed_final',
            token_readout={'mode': 'none'}, training=dict(epochs=20, seed=20261008,
            visual_learning_rate=1.75e-6, head_learning_rate=5e-6))
        validate_checkpoint_metadata(state, metrics, experiment, arm='J5', config_sha=config_sha, last_sha=last_sha)
        for key, value in [('epoch', 8), ('split_sha256', 'wrong'), ('mode', 'last_n_blocks'),
                           ('training_scope', 'full_train_fixed_epochs')]:
            corrupt = copy.deepcopy(state)
            corrupt[key] = value
            with self.assertRaises(ValueError):
                validate_checkpoint_metadata(corrupt, metrics, experiment, arm='J5', config_sha=config_sha, last_sha=last_sha)
        bad_metrics = dict(metrics, last_checkpoint_sha256='best_digest')
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(state, bad_metrics, experiment, arm='J5', config_sha=config_sha, last_sha=last_sha)

    def test_joint_optimizer_both_groups_and_moments(self):
        tensors = tiny_state()
        opt = dict(param_groups=[{'params': [0]}, {'params': [1, 2]}], state={})
        for pid, name in enumerate(tensors):
            opt['state'][pid] = dict(step=torch.tensor(4.), exp_avg=torch.zeros_like(tensors[name]), exp_avg_sq=torch.ones_like(tensors[name]))
        state = dict(trainable_state=tensors, optimizer=opt, scheduler={'T_max': 20, 'last_epoch': 20},
                     scaler={'scale': 128.}, confidence_weighting={'epochs_updated': 20})
        validate_optimizer_state(state, list(tensors), visual_names=['visual.weight'])
        bad = copy.deepcopy(state)
        bad['optimizer']['param_groups'] = [{'params': [0, 1, 2]}]
        with self.assertRaises(ValueError):
            validate_optimizer_state(bad, list(tensors), visual_names=['visual.weight'])
        bad = copy.deepcopy(state)
        bad['optimizer']['state'][2]['exp_avg'] = torch.zeros(4)
        with self.assertRaises(ValueError):
            validate_optimizer_state(bad, list(tensors), visual_names=['visual.weight'])

    def test_pair_rejects_best_last8_head152_and_row_reordering(self):
        identity = dict(checkpoint_sha256=CONTROL_LAST_SHA, bundle_sha256=CONTROL_CACHE_SHA,
            experiment_config_sha256=CONTROL_CONFIG_SHA, split_sha256=SPLIT_SHA,
            manifest_sha256=MANIFEST_SHA, class_mapping_sha256=MAPPING_SHA, h1_sha256=H1_SHA,
            view_protocol=VIEW_PROTOCOL)
        control = dict(identity=identity, ids=['a', 'b'], groups=['ga', 'gb'], targets=np.array([0, 1]), train_counts=np.array([2, 3]))
        joint_identity = dict(identity, checkpoint_sha256='new_last', bundle_sha256='new_cache',
            p1_arm='J5', joint_trainable_tensors=154, joint_trainable_numel=88233966,
            classifier_head_source='joint_fixed_last_state', training_source_manifest_sha256=TRAINING_MANIFEST_SHA)
        candidate = dict(control, identity=joint_identity)
        self.assertTrue(validate_pair(candidate, control))
        for wrong in ['best_digest', 'last8_digest']:
            bad = dict(control, identity=dict(identity, checkpoint_sha256=wrong))
            with self.assertRaises(ValueError):
                validate_pair(candidate, bad)
        bad = dict(candidate, identity=dict(joint_identity, joint_trainable_tensors=152))
        with self.assertRaises(ValueError):
            validate_pair(bad, control)
        with self.assertRaises(ValueError):
            validate_pair(dict(candidate, ids=['b', 'a']), control)

    def test_shared_four_view_float32_average_and_exact_flip(self):
        from aictrack1.evaluation.four_view import fixed_four_view_logits
        seen = []
        def model(images):
            seen.append(images.clone())
            return torch.stack([images[:, 0, 0, 0], images[:, 0, 0, -1]], dim=1).half()
        native = torch.tensor([[[[1., 2., 3.]]]])
        scaled = torch.tensor([[[[4., 5., 6.]]]])
        center, two, four = fixed_four_view_logits(model, native, scaled)
        self.assertEqual(len(seen), 4)
        self.assertTrue(torch.equal(seen[1], torch.flip(native, dims=(-1,))))
        self.assertTrue(torch.equal(seen[3], torch.flip(scaled, dims=(-1,))))
        self.assertEqual(four.dtype, torch.float32)
        self.assertTrue(torch.equal(center, torch.tensor([[1., 3.]])))
        self.assertTrue(torch.equal(two, torch.tensor([[2., 2.]])))
        self.assertTrue(torch.equal(four, torch.tensor([[3.5, 3.5]])))

    def test_balanced_queries_support_only_and_group_unique(self):
        control = dict(scores=np.zeros((8, 3), np.float32), targets=np.array([0, 0, 0, 1, 1, 1, 2, 2]),
                       ids=[f'i{i}' for i in range(8)], groups=['a', 'a', 'b', 'c', 'd', 'e', 'f', 'g'])
        first = build_queries(control, per_class=2)
        altered = dict(control, scores=np.arange(24).reshape(8, 3).astype(np.float32))
        second = build_queries(altered, per_class=2)
        self.assertTrue(all(np.array_equal(first[k], second[k]) for k in first))
        ids = first['balanced_indices']
        self.assertEqual(len(ids), len({control['groups'][i] for i in ids}))
        self.assertTrue(np.all(np.bincount(control['targets'][ids], minlength=3) == 2))

    def test_paired_decode_identical_control_zero_delta_and_deterministic(self):
        scores = np.array([[3, 1, 0], [1, 3, 0], [0, 1, 3], [3, 2, 1], [1, 3, 2], [2, 1, 3]], np.float32)
        control = dict(scores=scores, targets=np.array([0, 1, 2, 0, 1, 2]),
                       ids=[f'i{i}' for i in range(6)], groups=[f'g{i}' for i in range(6)])
        queries = build_queries(control, per_class=2)
        rows = compare_arrays(control, control, queries, draws=50)
        again = compare_arrays(control, control, queries, draws=50)
        for first, second in zip(rows, again, strict=True):
            self.assertEqual(first['p7']['candidate_vs_control'], second['p7']['candidate_vs_control'])
            self.assertEqual(first['p7']['candidate_vs_control']['delta_vs_baseline_pp'], 0.)
            self.assertEqual(first['p7']['candidate_vs_control']['group_bootstrap']['ci95_pp'], [0., 0.])
            self.assertEqual(first['p7']['candidate_vs_control']['changed_predictions'], 0)
        self.assertEqual(eligibility(rows)['status'], 'registered_continuation_threshold_not_met')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
