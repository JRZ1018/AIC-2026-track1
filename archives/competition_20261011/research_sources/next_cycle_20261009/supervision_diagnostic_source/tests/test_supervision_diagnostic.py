"""Meaningful CPU failure tests; no official model, images, CUDA or optimizer step."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT.parent / 'supervision_source'
sys.path[:0] = [str(ROOT), str(TRAINING / 'src'), str(TRAINING / 'scripts')]
from supervision_diag_contract import (CLIP_SHA, HEAD_KEYS, H1_SHA, MANIFEST_SHA, MAPPING_SHA,
    SPLIT_SHA, CONTROL_CACHE_SHA, CONTROL_LAST_SHA, CONTROL_CONFIG_SHA, VIEW_PROTOCOL,
    TRAINING_MANIFEST_SHA, validate_checkpoint_metadata, validate_visual_state,
    validate_optimizer_state, validate_pair, sha, validate_frozen_head, validate_ac_state, VISUAL_NUMEL)
from compare_supervision import build_queries, compare_arrays, eligibility


def tiny_state():
    return {'visual.weight': torch.eye(2)}


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
    def test_features_hook_four_B_two_2B_order_zero_extra_forward_unchanged_logits(self):
        from view_features import FourViewFeatureCollector
        from aictrack1.models.peft import PEFTImageClassifier
        from aictrack1.evaluation.four_view import fixed_four_view_logits
        from torch.nn.functional import normalize
        class Visual(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, images):
                self.calls += 1
                return torch.stack([images[:, 0, 0, 0] + 2., images[:, 0, 0, -1] + 3., images.mean((1, 2, 3)) + 4.], dim=1).half()
        class Classifier(nn.Module):
            def forward(self, features):
                return torch.stack([features[:, 0] * 2. + features[:, 1], features[:, 2] - features[:, 0]], dim=1)
        visual = Visual()
        model = PEFTImageClassifier(visual, Classifier(), normalize_features=True).eval()
        native = torch.tensor([[[[1., 2., 3.]]], [[[4., 5., 6.]]]])
        scaled = torch.tensor([[[[7., 8., 9.]]], [[[10., 11., 12.]]]])
        baseline = fixed_four_view_logits(model, native, scaled)
        before = visual.calls
        collector = FourViewFeatureCollector(visual, feature_dim=3, layout='four_calls_B')
        observed = fixed_four_view_logits(model, native, scaled)
        features = collector.finish(2)
        self.assertEqual(visual.calls - before, 4)
        self.assertEqual(collector.calls, 4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline, observed, strict=True)))
        expected = []
        for images in [native, torch.flip(native, (-1,)), scaled, torch.flip(scaled, (-1,))]:
            raw = torch.stack([images[:, 0, 0, 0] + 2., images[:, 0, 0, -1] + 3., images.mean((1, 2, 3)) + 4.], dim=1).half().float()
            expected.append(normalize(raw, dim=-1))
        expected = normalize(torch.stack(expected).mean(0), dim=-1)
        self.assertTrue(torch.equal(features, expected))
        collector.close()
        before = visual.calls
        two = FourViewFeatureCollector(visual, feature_dim=3, layout='two_calls_2B')
        native_pair = model(torch.cat([native, torch.flip(native, (-1,))], dim=0))
        scaled_pair = model(torch.cat([scaled, torch.flip(scaled, (-1,))], dim=0))
        first, flipped = native_pair.chunk(2)
        resized, resized_flip = scaled_pair.chunk(2)
        combined = (first + flipped + resized + resized_flip) / 4
        self.assertEqual(visual.calls - before, 2)
        self.assertTrue(torch.equal(two.finish(2), expected))
        self.assertTrue(torch.equal(combined, baseline[2]))
        two.close()
        partial = FourViewFeatureCollector(visual, feature_dim=3)
        model(native)
        with self.assertRaises(ValueError):
            partial.finish(2)
        partial.close()

    def test_actual_production_entry_dependency_check_frozen_training_and_vendor_cli(self):
        # No mocked modules/APIs: a new process imports the actual frozen 2fee package.
        result = subprocess.run([sys.executable, str(ROOT / 'run_supervision_diagnostic.py'),
            '--training-root', str(TRAINING), '--dependency-check'], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report['complete'])
        self.assertTrue(report['imports_and_signatures_verified'])
        self.assertTrue(report['real_vendor_cli_help_passed'])
        self.assertTrue(report['real_vendor_production_argv_parsed'])
        self.assertEqual(report['frozen_training_source_sha256'], TRAINING_MANIFEST_SHA)
        self.assertFalse(report['gpu_probed'])
        self.assertFalse(report['images_opened'])
        self.assertFalse(report['checkpoint_opened'])
        self.assertFalse(report['remote_assets_opened'])
        self.assertTrue(Path(report['static_imports']['p1_contract']).resolve().is_relative_to(TRAINING.resolve()))

    def test_independent_host_and_cgroup_memory_floor_without_gpu(self):
        from diagnostic_resources import linux_memory_snapshot, memory_snapshot, memory_reason
        fixture = ROOT / 'artifacts' / 'cpu_fixtures' / 'cgroup'
        (fixture / 'proc').mkdir(parents=True, exist_ok=True)
        (fixture / 'sys/fs/cgroup').mkdir(parents=True, exist_ok=True)
        (fixture / 'proc/meminfo').write_text('MemTotal: 16777216 kB\nMemAvailable: 8388608 kB\n', encoding='ascii')
        (fixture / 'sys/fs/cgroup/memory.max').write_text(str(3 * 2**30), encoding='ascii')
        (fixture / 'sys/fs/cgroup/memory.current').write_text(str(2**30), encoding='ascii')
        snapshot = linux_memory_snapshot(fixture)
        self.assertEqual(snapshot['effective_available_bytes'], 2 * 2**30)
        self.assertIsNotNone(memory_reason(snapshot))
        (fixture / 'sys/fs/cgroup/memory.max').write_text('max', encoding='ascii')
        self.assertIsNone(memory_reason(linux_memory_snapshot(fixture)))
        live = memory_snapshot()  # Actual Windows/Linux host read, no GPU enumeration.
        self.assertIsInstance(live['effective_available_bytes'], int)
        self.assertFalse(live['gpu_probed'])

    def test_visual_only_state_and_fail_closed(self):
        state = tiny_state()
        validate_visual_state(state, visual_names=['visual.weight'], visual_count=1, visual_numel=4)
        for mutate in (lambda x:x.__setitem__('classifier.linear.bias',torch.zeros(3)),
                       lambda x:x.__setitem__('visual.other',torch.zeros(1)),
                       lambda x:x.__setitem__('visual.weight',torch.full((2,2),float('nan'))),
                       lambda x:x.__setitem__('visual.weight',torch.eye(2).half()),
                       lambda x:x.__setitem__('visual.weight',torch.zeros(5))):
            broken=copy.deepcopy(state);mutate(broken)
            with self.assertRaises(ValueError):
                validate_visual_state(broken,visual_names=['visual.weight'],visual_count=1,visual_numel=4)

    def test_exact_frozen_h1_head_rejects_changed_or_trainable(self):
        model=SimpleNamespace(classifier=Head())
        original={'model':copy.deepcopy(model.classifier.state_dict())}
        self.assertTrue(validate_frozen_head(model,original))
        with self.assertRaises(ValueError):
            validate_frozen_head(model,{'model':{k:v.half() for k,v in original['model'].items()}})
        with torch.no_grad():model.classifier.linear.bias.add_(.001)
        with self.assertRaises(ValueError):validate_frozen_head(model,original)
        model.classifier.load_state_dict(original['model'])
        model.classifier.linear.weight.requires_grad_(True)
        with self.assertRaises(ValueError):validate_frozen_head(model,original)

    def test_actual_fv2_loader_restores_visual_keeps_original_frozen_head(self):
        # Ordinary inherited-ACL fixtures retained privately for audit; no tempfile chmod or recursive deletion.
        fixture = ROOT / 'artifacts' / 'cpu_fixtures' / 'loader'
        fixture.mkdir(parents=True, exist_ok=True)
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
            self.assertTrue(torch.equal(fitted.model.visual.weight, state['trainable_state']['visual.weight']))
            self.assertTrue(torch.equal(fitted.model.classifier.linear.weight, torch.zeros(3,2)))
            self.assertTrue(torch.equal(fitted.model.classifier.linear.bias, torch.zeros(3)))
            validate_frozen_head(fitted.model, {'model':Head().state_dict()})
            self.assertFalse(fitted.model.training)
            self.assertTrue(all(not p.requires_grad for p in fitted.model.parameters()))
            x = torch.tensor([[1., 0.]])
            self.assertTrue(torch.equal(fitted.model(x), torch.zeros(1,3)))
            for bad in ('missing_visual', 'extra_key'):
                broken = copy.deepcopy(state)
                if bad == 'missing_visual':
                    broken['trainable_state'].pop('visual.weight')
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
            history=history, token_readout={'mode': 'none'}, supervision_source_manifest_sha256=TRAINING_MANIFEST_SHA,
            supervision_arm='S', classifier_head_source='original_fold_H1_frozen', classifier_head_restored_equal=True)
        metrics = dict(last_checkpoint_sha256=last_sha, training_scope='fixed_split_validation',
                       train_count=118958, validation_count=29737, early_stop=None, history=history,
            head_initializer={'sha256':H1_SHA}, base_loss={'kind':'sce','alpha':.5,'beta':1.,'log_zero':-4.}, trainable_total_parameters=VISUAL_NUMEL)
        experiment = dict(peft={'mode': 'full_visual', 'lora_blocks': 8}, selection_rule='fixed_final',
            token_readout={'mode': 'none'}, training=dict(epochs=20, seed=20261008,
            visual_learning_rate=1.75e-6, head_learning_rate=0.),base_loss={'kind':'sce','alpha':.5,'beta':1.,'log_zero':-4.})
        validate_checkpoint_metadata(state, metrics, experiment, arm='S', config_sha=config_sha, last_sha=last_sha)
        for key, value in [('epoch', 8), ('split_sha256', 'wrong'), ('mode', 'last_n_blocks'),
                           ('training_scope', 'full_train_fixed_epochs'),('supervision_arm','J5'),('classifier_head_restored_equal',False),
                           ('supervision_source_manifest_sha256','wrong')]:
            corrupt = copy.deepcopy(state)
            corrupt[key] = value
            with self.assertRaises(ValueError):
                validate_checkpoint_metadata(corrupt, metrics, experiment, arm='S', config_sha=config_sha, last_sha=last_sha)
        bad_metrics = dict(metrics, last_checkpoint_sha256='best_digest')
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(state, bad_metrics, experiment, arm='S', config_sha=config_sha, last_sha=last_sha)

    def test_visual_only_optimizer_moments_and_complete_schedule(self):
        tensors = tiny_state()
        opt = dict(param_groups=[{'params': [0]}], state={})
        for pid, name in enumerate(tensors):
            opt['state'][pid] = dict(step=torch.tensor(4.), exp_avg=torch.zeros_like(tensors[name]), exp_avg_sq=torch.ones_like(tensors[name]))
        state = dict(trainable_state=tensors, optimizer=opt, scheduler={'T_max': 20, 'last_epoch': 20},
                     scaler={'scale': 128.}, confidence_weighting={'epochs_updated': 20})
        validate_optimizer_state(state, list(tensors), visual_names=['visual.weight'])
        bad = copy.deepcopy(state)
        bad['optimizer']['param_groups'] = [{'params':[0]},{'params':[1,2]}]
        with self.assertRaises(ValueError):
            validate_optimizer_state(bad, list(tensors), visual_names=['visual.weight'])
        bad = copy.deepcopy(state)
        bad['optimizer']['state'][0]['exp_avg'] = torch.zeros(4)
        with self.assertRaises(ValueError):
            validate_optimizer_state(bad, list(tensors), visual_names=['visual.weight'])

    def test_pair_rejects_best_last8_joint154_fresh_h1_and_row_reordering(self):
        identity = dict(checkpoint_sha256=CONTROL_LAST_SHA, bundle_sha256=CONTROL_CACHE_SHA,
            experiment_config_sha256=CONTROL_CONFIG_SHA, split_sha256=SPLIT_SHA,
            manifest_sha256=MANIFEST_SHA, class_mapping_sha256=MAPPING_SHA, h1_sha256=H1_SHA,
            view_protocol=VIEW_PROTOCOL)
        control = dict(identity=identity, ids=['a', 'b'], groups=['ga', 'gb'], targets=np.array([0, 1]), train_counts=np.array([2, 3]))
        joint_identity = dict(identity, checkpoint_sha256='new_last', bundle_sha256='new_cache',
            model_arm='S', trainable_tensors=152, trainable_numel=87849216,
            classifier_head_source='old_fold_h1_frozen', training_source_manifest_sha256=TRAINING_MANIFEST_SHA)
        candidate = dict(control, identity=joint_identity)
        self.assertTrue(validate_pair(candidate, control))
        for wrong in ['best_digest', 'last8_digest']:
            bad = dict(control, identity=dict(identity, checkpoint_sha256=wrong))
            with self.assertRaises(ValueError):
                validate_pair(candidate, bad)
        bad = dict(candidate, identity=dict(joint_identity, trainable_tensors=154))
        with self.assertRaises(ValueError):
            validate_pair(bad, control)
        with self.assertRaises(ValueError):
            validate_pair(dict(candidate, ids=['b', 'a']), control)
        with self.assertRaises(ValueError):
            validate_pair(dict(candidate, identity=dict(joint_identity, h1_sha256='new_fresh_h1')), control)

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


class AdditionalContractTests(unittest.TestCase):
    def test_supported_decline_exact_two_of_twenty_boundary_is_reported(self):
        import compare_supervision
        targets=np.repeat(np.arange(2),20)
        control=dict(scores=np.zeros((40,2),np.float32),targets=targets,
            ids=[f'i{i}' for i in range(40)],groups=[f'g{i}' for i in range(40)],train_counts=np.array([100,100]))
        c=targets.copy();c[13:20]=1
        s=targets.copy();s[11:20]=1
        arrays=dict(natural_indices=np.arange(40),natural_columns=np.arange(2),
            balanced_indices=np.arange(40),balanced_columns=np.arange(2))
        with patch.object(compare_supervision,'frozen_decode',side_effect=[(s,{}),(c,{}),(s,{}),(c,{})]):
            reports=compare_arrays(control,control,arrays,draws=0)
        for report in reports:
            declines=report['supported_large_declines']
            self.assertEqual(len(declines),1)
            self.assertEqual(declines[0]['delta_pp'],-10.)
            self.assertEqual((declines[0]['candidate_correct'],declines[0]['control_correct']),(11,13))

    def test_s_candidate_audit_rejects_joint_old_source_or_false_restoration(self):
        from supervision_diag_contract import validate_candidate_audit
        identity=dict(checkpoint_sha256='2'*64, experiment_config_sha256='3'*64, split_sha256=SPLIT_SHA,
            h1_sha256=H1_SHA, diagnostic_source_manifest_sha256='4'*64, trainable_tensors=152,
            trainable_numel=87849216, classifier_head_source='old_fold_h1_frozen',
            training_source_manifest_sha256=TRAINING_MANIFEST_SHA)
        audit=dict(schema='aictrack1.supervision.fixed_last_diagnostic_audit.v1', complete=True,
            status='complete_fixed_last20_audit', model_arm='S', checkpoint_role='last', checkpoint_epoch=20,
            fixed_last_sha256=identity['checkpoint_sha256'],trainable_tensors=152, trainable_numel=87849216,
            classifier_parameters_restored_equal=True,classifier_head_source='old_fold_h1_frozen',
            exact_ac_train_id_label_binding=True,model_frozen_eval=True,inference_optimizer_steps=0,
            test_access=False,training_source_manifest_sha256=TRAINING_MANIFEST_SHA,
            diagnostic_source_manifest_sha256='4'*64,config_sha256='3'*64,split_sha256=SPLIT_SHA,
            h1_sha256=H1_SHA,train_count=118958,validation_count=29737,sce_alpha=.5)
        self.assertTrue(validate_candidate_audit(identity,audit))
        for key,value in [('schema','aictrack1.p1.fixed_last_diagnostic_audit.v1'),('trainable_tensors',154),
                ('classifier_parameters_restored_equal',False),('training_source_manifest_sha256','2fee-old'),
                ('exact_ac_train_id_label_binding',False),('sce_alpha',.1),('config_sha256','other')]:
            with self.assertRaises(ValueError):validate_candidate_audit(identity,dict(audit,**{key:value}))

    def test_ac_actual_class_exact_ids_labels_hyperparams_and_full_coverage(self):
        from aictrack1.methods.confidence import ClasswiseHistoryWeighting
        raw=dict(momentum=.8,history_mix=.5,min_weight=.02,magnitude_mix=.75)
        original=ClasswiseHistoryWeighting(torch.tensor([2,5]),torch.tensor([0,1]),**raw)
        state=original.state_dict();state['epochs_updated']=20;state['seen'].fill_(1)
        validate_ac_state(state,[2,5],[0,1],raw)
        cases=[dict(state,sample_indices=torch.tensor([5,2])),dict(state,class_ids=torch.tensor([1,0])),
            dict(state,momentum=.7),dict(state,epochs_updated=19),dict(state,weights=torch.tensor([float('nan'),1.]))]
        broken=copy.deepcopy(state);broken['seen'][0]=0;cases.append(broken)
        for case in cases:
            with self.assertRaises(ValueError):validate_ac_state(case,[2,5],[0,1],raw)

    def test_registered_thresholds_no_natural_or_ci_lower_hard_gate(self):
        def row(name,delta):return {'name':name,'p7':{'candidate_vs_control':{'delta_vs_baseline_pp':delta,'group_bootstrap':{'ci95_pp':[-.5,.8]}}}}
        result=eligibility([row('natural_unique_images',-.9),row('balanced_supported_unique',.1)])
        self.assertEqual(result['status'],'requires_fresh_same_protocol_positive_confirmation')
        self.assertEqual(result['natural_p7_delta_pp'],-.9)
        result=eligibility([row('natural_unique_images',-.9),row('balanced_supported_unique',.2)])
        self.assertEqual(result['status'],'exploratory_continuation_threshold_met')
        self.assertFalse(result['stable_gain_claim'])

    def test_orchestration_exact_s_last20_cli_and_no_background(self):
        from diagnose_and_pair import commands
        args=SimpleNamespace(training_root=Path('/root/autodl-tmp/AICtrack1_next_cycle_supervision_20261009'),
            expected_last_sha256='2'*64,deadline_unix=123.,hard_seconds=1800,workers=4,min_free_gib=3.,
            save_features=True,control_bundle=Path('/pinned/C.pt'),control_identity=Path('/pinned/C.json'))
        diagnostic,compare=commands(args)
        self.assertIn('--save-features',diagnostic);self.assertEqual(diagnostic[diagnostic.index('--arm')+1],'S')
        self.assertEqual(compare[compare.index('--threads')+1],'4')
        self.assertIn('P1_S',compare[compare.index('--candidate-bundle')+1])
        self.assertNotIn('--prepare-queries',compare)
        for entry in ('diagnose_and_pair.py','compare_supervision.py'):
            result=subprocess.run([sys.executable,str(ROOT/entry),'--help'],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
