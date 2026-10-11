"""Validate a fresh heldout L35 model on frozen train-derived queries only.

The released all-train final model is deliberately rejected here. Natural and
balanced queries build separate fixed graphs, as in the recorded validation.
The original second query was an independent L35 confirmation before the full
run. Both queries are now disclosed, so a new reproduction is a replay and
does not establish a new independent confirmation or a competition test score.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import json
import hashlib
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portable_training.runtime import CODE, DATA, MODELS, SPLITS, bootstrap, make_dataset, sha

ORIGINAL_PAIRED_CACHE_SHA = {
    ('first', 'natural'): '453969670ecac5a5a8a621730f2d3d9be02dd474ee07eec62be349f2e4c83347',
    ('first', 'balanced'): '48238882913f90b983736ab16b6c7bb28f9410e4b8325e1dcd68245dda0576c4',
    ('second', 'natural'): '9601fca7be8961f3a67fc010b29b36d0a586c6164fd0440781cae0ffe59c6a6d',
    ('second', 'balanced'): '21f538436ef81bb931a5c7b17ac08429bd38b12f84461c624469657d017cc212',
}


def _validate_checkpoint(state, split, metrics):
    if state.get('training_scope') != 'fixed_split_validation':
        raise ValueError('Validation refuses all-train/full checkpoints; train a fresh heldout model first')
    if state.get('split_sha256') != sha(SPLITS[split]):
        raise ValueError('Checkpoint belongs to another heldout split')
    if state.get('competition_stage') != 'stage2' or state.get('num_classes') != 750:
        raise ValueError('Official Stage2/750 provenance required')
    if (state.get('fresh_route_full_arm') != 'L35' or state.get('selected_route') != 'L35_GRAPH' or
            metrics.get('fresh_route_full_arm') != 'L35' or metrics.get('selected_route') != 'L35_GRAPH'):
        raise ValueError('Heldout reproduction must retain the exact L35/L35_GRAPH binding')
    tensors = state.get('trainable_state', {})
    if (len(tensors) != 154 or sum(t.numel() for t in tensors.values()) != 88233966 or
            state.get('classifier_head_source') != 'joint_fixed_last_state'):
        raise ValueError('Exact joint154/88233966 L35 trainables required')
    if ([r.get('epoch') for r in metrics.get('history', [])] != list(range(1, 21)) or
            metrics.get('training_scope') != 'fixed_split_validation' or
            metrics.get('config_sha256') != state.get('config_sha256') or
            metrics.get('split_sha256') != state.get('split_sha256') or
            metrics.get('selected_checkpoint') != 'final.pt' or
            metrics.get('selection_rule') != 'fixed final epoch 20; no validation selection' or
            state.get('selection_rule') != 'fixed final epoch 20; no validation selection'):
        raise ValueError('Fixed20 complete checkpoint required')


def _paired(pred, control, targets, groups, columns, train_counts):
    import numpy as np
    correct, baseline = pred == targets, control == targets
    delta = correct.astype(np.int64) - baseline.astype(np.int64)
    _, codes = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    count = int(codes.max()) + 1
    group_n = np.bincount(codes, minlength=count)
    group_delta = np.bincount(codes, weights=delta, minlength=count)
    rng = np.random.default_rng(20261009)
    boot = []
    for _ in range(500):
        selected = rng.integers(0, count, size=count)
        boot.append(float(group_delta[selected].sum() / group_n[selected].sum()) * 100)
    order = sorted(range(750), key=lambda i: (-int(train_counts[i]), i))
    strata = {c: 'head' if i < 150 else 'mid' if i < 375 else 'tail' for i, c in enumerate(order)}
    classes = []
    for c in columns:
        mask = targets == c
        classes.append(dict(class_index=int(c), stratum=strata[int(c)], support=int(mask.sum()),
            candidate_correct=int(correct[mask].sum()), control_correct=int(baseline[mask].sum()),
            delta_correct=int(delta[mask].sum())))
    return dict(candidate_top1_pp=float(correct.mean() * 100), control_top1_pp=float(baseline.mean() * 100),
        delta_pp=float(delta.mean() * 100), net_correct=int(delta.sum()),
        wrong_to_correct=int((~baseline & correct).sum()), correct_to_wrong=int((baseline & ~correct).sum()),
        groups=count, grouped_bootstrap=dict(draws=500, seed=20261009, ci95_pp=np.quantile(boot, [.025, .975]).tolist(),
            limitation='Conditional fixed-model observed-label agreement; not training-seed variance'),
        strata={s: dict(net_correct=sum(r['delta_correct'] for r in classes if r['stratum'] == s),
            support=sum(r['support'] for r in classes if r['stratum'] == s)) for s in ('head', 'mid', 'tail')},
        per_class=classes, newly_zero_recall=[r['class_index'] for r in classes if r['candidate_correct'] == 0 and r['control_correct'] > 0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Fresh fixed20 heldout checkpoint, never the released FULL final.pt.')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--split', choices=('first', 'second'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=MODELS)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--hard-seconds', type=int, default=3600)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('Will not overwrite a validation output')
    if args.workers < 0 or args.batch_size < 1 or args.hard_seconds < 1: p.error('Invalid runtime limits')
    bootstrap()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import release_common as common
    from aictrack1.data.contracts import load_dataset_config
    from aictrack1.data.indexed_images import read_indexed_images
    from aictrack1.data.transforms import build_split_transforms
    from aictrack1.evaluation.four_view import PairedViewDataset, fixed_four_view_logits
    from aictrack1.evaluation.classification import classification_diagnostics
    from view_features import FourViewFeatureCollector
    from p3_graph import blocked_topk, mutual_graph, smooth, p7
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    metrics_path = args.checkpoint.parent / 'metrics.json'
    if not metrics_path.is_file(): raise ValueError('Heldout final checkpoint needs its real terminal metrics.json')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if metrics.get('selected_checkpoint_sha256') != sha(args.checkpoint) or metrics.get('selected_checkpoint') != args.checkpoint.name:
        raise ValueError('Checkpoint differs from fixed terminal export metrics')
    _validate_checkpoint(state, args.split, metrics)
    config_path = args.checkpoint.parent / 'config.yaml'
    if not config_path.is_file() or sha(config_path) != state.get('config_sha256'):
        raise ValueError('Heldout final checkpoint needs its exact recorded training config.yaml')
    import yaml
    recorded_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))['experiment']
    training = recorded_config['training']
    if (training.get('visual_learning_rate') != 3.5e-6 or training.get('head_learning_rate') != 5e-6 or
            training.get('epochs') != 20 or training.get('batch_size') != 64 or
            training.get('weight_decay') != .01 or training.get('gradient_clip_norm') != 1. or
            recorded_config.get('selection_rule') != 'fixed_final'):
        raise ValueError('Recorded training recipe differs from L35 joint20')
    args.output.mkdir(parents=True)
    dataset_path = make_dataset(args.data_root, args.output, args.split)
    cfg = load_dataset_config(dataset_path)
    records = read_indexed_images(cfg, source='train', manifest_path=DATA / 'train_manifest.csv', split_path=SPLITS[args.split])
    train_records = [r for r in records if r.partition == 'train']
    validation = {r.sample_id: r for r in records if r.partition == 'validation'}
    query_root = CODE / 'data' / 'validation' / args.split
    with (query_root / 'natural_unique_images_queries.csv').open(encoding='utf-8-sig', newline='') as stream:
        natural = list(csv.DictReader(stream))
    if len(validation) != 29737 or len(natural) != 29737 or len(train_records) != 118958:
        raise ValueError('Official heldout counts differ')
    if {r['sample_id'] for r in natural} != set(validation): raise ValueError('Frozen natural query IDs differ from heldout split')
    ordered = [validation[r['sample_id']] for r in natural]
    if any(r.internal_class_index != int(q['target']) or int(q['source_row']) != i for i, (r, q) in enumerate(zip(ordered, natural))):
        raise ValueError('Frozen local query row/label binding differs')
    control_identity = json.loads((query_root / 'J5_control_identity.json').read_text(encoding='utf-8'))
    original_candidate_identity = json.loads((query_root / 'L35_control_identity.json').read_text(encoding='utf-8'))
    # Recorded diagnostic identities hash newline-joined IDs without a final newline.
    order_sha = hashlib.sha256('\n'.join(r.sample_id for r in ordered).encode()).hexdigest()
    if control_identity.get('sample_ids_order_sha256') != order_sha or control_identity.get('split_sha256') != sha(SPLITS[args.split]):
        raise ValueError('Cached J5 control does not bind this exact heldout query row order')
    if control_identity.get('model_arm') != 'J5' or control_identity.get('test_access') is not False:
        raise ValueError('Historical control must be the recorded train-only J5 model')
    model, _ = common.load_model(args.checkpoint, args.model_dir, args.device)
    transforms = build_split_transforms(cfg.transforms)
    scaled = build_split_transforms(dataclasses.replace(cfg.transforms, resize_size=384))['validation']
    paired = PairedViewDataset(ordered, transforms['validation'], scaled, required_partition='validation',
        allow_truncated_images=cfg.allow_truncated_images)
    loader = DataLoader(paired, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=args.device == 'cuda', persistent_workers=args.workers > 0)
    observer = FourViewFeatureCollector(model.visual)
    logits = []
    deadline = time.monotonic() + args.hard_seconds
    try:
        with torch.inference_mode():
            for batch in loader:
                if time.monotonic() >= deadline: raise TimeoutError('Validation budget exhausted; partial result is not complete')
                with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.device == 'cuda'):
                    _, _, fused = fixed_four_view_logits(model, batch['image'].to(args.device), batch['scaled_image'].to(args.device))
                # Match the recorded diagnostic cache: FP16 storage, FP32 decode.
                logits.append(fused.detach().cpu().half())
        features = observer.finish(29737)
    finally:
        observer.close()
    scores = torch.cat(logits).float().numpy()
    targets = np.array([r.internal_class_index for r in ordered], dtype=np.int64)
    groups = [r['group'] for r in natural]
    train_counts = np.bincount([r.internal_class_index for r in train_records], minlength=750)
    with np.load(query_root / 'query_indices.npz', allow_pickle=False) as stored:
        queries = {k: stored[k].copy() for k in stored.files}
    reports = {}
    torch.set_num_threads(2)
    for name, expected_n, expected_c in (('natural', 29737, 750), ('balanced', 14760, 738)):
        ix, columns = queries[name + '_indices'], queries[name + '_columns']
        if ix.shape != (expected_n,) or columns.shape != (expected_c,): raise ValueError('Frozen query shape differs')
        values = scores[ix][:, columns].copy()
        embeddings = features[ix].numpy()
        neighbors, weights, _ = blocked_topk(embeddings, 20, row_block=256, max_block_bytes=64 * 1024**2, deadline=deadline)
        transition, _, degree = mutual_graph(neighbors, weights, 10)
        changed, graph = smooth(values, transition, degree, .3, iterations=3, deadline=deadline)
        combo_local, decoder = p7(changed, deadline=deadline)
        raw_p7_local, _ = p7(values, deadline=deadline)
        pred, raw_pred = columns[combo_local], columns[raw_p7_local]
        paired_cache = query_root / f'L35_{name}_predictions.npz'
        if sha(paired_cache) != ORIGINAL_PAIRED_CACHE_SHA[(args.split, name)]:
            raise ValueError('Original L35-vs-J5 paired cache bytes differ')
        with np.load(paired_cache, allow_pickle=False) as cached:
            control = cached['control_p7'].copy()  # Original L35-vs-J5 artifact; J5_GRAPH is the control.
            if 'targets' in cached and not np.array_equal(cached['targets'], targets[ix]):
                raise ValueError('Cached control target order differs')
        if control.shape != (expected_n,) or not np.isin(control, columns).all(): raise ValueError('J5 cache shape/classes differ')
        reports[name] = dict(samples=expected_n, classes=expected_c,
            raw_argmax=classification_diagnostics(torch.from_numpy(columns[values.argmax(1)]), torch.from_numpy(targets[ix]), num_classes=750),
            raw_P7=classification_diagnostics(torch.from_numpy(raw_pred), torch.from_numpy(targets[ix]), num_classes=750),
            combo_vs_cached_J5_GRAPH=_paired(pred, control, targets[ix], [groups[int(i)] for i in ix], columns, train_counts),
            graph=graph, decoder=decoder, control_prediction_file_sha256=sha(query_root / f'L35_{name}_predictions.npz'))
        np.savez_compressed(args.output / f'{name}_predictions.npz', candidate_p7=pred, control_p7=control, raw_p7=raw_pred, targets=targets[ix])
    torch.save(dict(logits=torch.from_numpy(scores), features=features,
        sample_ids=[r.sample_id for r in ordered], targets=torch.from_numpy(targets),
        checkpoint_sha256=sha(args.checkpoint), split_sha256=sha(SPLITS[args.split]),
        test_access=False), args.output / 'validation_cache.pt')
    common.write_json(args.output / 'report.json', dict(complete=True, schema='aictrack1.portable.L35.validation_reproduction.v1',
        checkpoint_sha256=sha(args.checkpoint), split_sha256=sha(SPLITS[args.split]),
        device=args.device, raw_model_forward_calls=4 * len(loader), additional_feature_forwards=0,
        model_parameter_updates=0, test_access=False, queries_already_disclosed=True,
        independent_validation=False, validation_metric='Observed-label agreement on official train-derived heldout queries',
        historical_original_independent_confirmation=original_candidate_identity.get('independent_confirmation') is True,
        historical_candidate_checkpoint_sha256=original_candidate_identity.get('checkpoint_sha256'),
        historical_control_checkpoint_sha256=control_identity.get('checkpoint_sha256'),
        historical_control_identity_sha256=sha(query_root / 'J5_control_identity.json'),
        cache_control='Recorded same-query J5_GRAPH control_p7 from the original L35-vs-J5 artifact; no control retraining/forward',
        statistics_scope='Top1, per-class, HMT, newly-zero recall and grouped-bootstrap CI; original full diagnostic reports retained separately',
        cached_logits_dtype='float16', decoder_logits_dtype='float32',
        cross_hardware_bitwise_identity_claim=False, protocols=reports))
    print(json.dumps(dict(complete=True, report=str(args.output / 'report.json'), test_access=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
