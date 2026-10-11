"""Fixed100 H1 using the exact archived FrozenHeadTrainer.train_epoch."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from portable_training.runtime import DATA, HEAD_CONFIG, MANIFEST_SHA, CLIP_SHA, SPLITS, sha, write_immutable


def _rows(cache):
    # Verify actual ID, label and order, not just a declared metadata hash.
    with (DATA / 'train_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        official = list(csv.DictReader(stream))
    if len(official) != 148695 or len(cache.sample_ids) != len(official):
        raise ValueError('H1 requires all 148695 official train cache rows')
    for sid, label, row in zip(cache.sample_ids, cache.labels.tolist(), official):
        if row['source_split'] != 'train' or sid != row['sample_id'] or int(label) != int(row['internal_class_index']):
            raise ValueError('H1 feature ID-label-order differs from official train manifest')
    return [[sid, int(label), part] for sid, label, part in zip(cache.sample_ids, cache.labels.tolist(), cache.partitions)]


def train_h1(args):
    import numpy as np
    import torch
    from aictrack1.data.feature_cache import load_feature_cache
    from aictrack1.engine.frozen_head import FrozenHeadConfig, FrozenHeadTrainer
    from aictrack1.engine.provenance import CheckpointProvenance
    from aictrack1.models.heads import build_classifier_head
    cache = load_feature_cache(args.cache)
    meta = cache.metadata
    if meta.source != 'train' or not meta.normalized or meta.peft_checkpoint_sha256 is not None or meta.feature_dim != 512:
        raise ValueError('H1 requires normalized original OpenAI CLIP train-only features')
    if meta.manifest_sha256 != MANIFEST_SHA or meta.expected_record_count != 148695:
        raise ValueError('H1 cache does not bind official full train manifest')
    from aictrack1.data.class_mapping import ClassMapping
    if meta.clip_weight_sha256 != CLIP_SHA or meta.class_mapping_sha256 != ClassMapping.from_csv(DATA / 'class_mapping.csv').digest():
        raise ValueError('H1 cache backbone/class mapping differs from official Stage2')
    if meta.split_sha256 not in {sha(p) for p in SPLITS.values()}:
        raise ValueError('H1 cache must use one of the supplied frozen splits')
    if meta.preprocessing.get('resize') != 336 or meta.preprocessing.get('center_crop') != 336:
        raise ValueError('L35 H1 requires frozen 336 features')
    rows = _rows(cache)
    if set(cache.partitions) != {'train', 'validation'}:
        raise ValueError('Cache partition vocabulary differs from frozen splits')
    full = args.scope == 'full'
    train = list(range(len(rows))) if full else [i for i, row in enumerate(rows) if row[2] == 'train']
    query = [] if full else [i for i, row in enumerate(rows) if row[2] == 'validation']
    if len(train) != (148695 if full else 118958) or len(query) != (0 if full else 29737):
        raise ValueError('H1 split counts differ from official full or heldout scope')
    x, y = cache.features[train].float(), cache.labels[train]
    if int(torch.bincount(y, minlength=750).min()) < 1:
        raise ValueError('Every class must have training support')
    binding = dict(schema='aictrack1.mainline.h1.rows.v1',
        counts=dict(train=len(train), validation=len(query)),
        all_rows_sha256=hashlib.sha256(json.dumps(rows, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest(),
        train_rows_sha256=hashlib.sha256(json.dumps([rows[i] for i in train], separators=(',', ':'), ensure_ascii=True).encode()).hexdigest(),
        validation_used_for_optimization_or_selection=False)
    scope = 'full_train_fixed_epochs' if full else 'fixed_split_validation'
    if full:
        binding.update(historical_cache_partition_counts=dict(train=sum(p == 'train' for p in cache.partitions),
            validation=sum(p == 'validation' for p in cache.partitions)),
            historical_validation_rows_now_part_of_full_official_train=True,
            historical_split_used_only_for_cache_provenance=True)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    identity = dict(schema='aictrack1.portable.H1.v1', config=HEAD_CONFIG, scope=scope,
        cache_metadata=meta.__dict__, row_binding=binding,
        test_access=False, original_cloud_run_replayed=False)
    write_immutable(work / 'portable_h1_binding.json', json.dumps(identity, indent=2, sort_keys=True) + '\n')
    random.seed(20261008)
    np.random.seed(20261008)
    torch.manual_seed(20261008)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(20261008)
    provenance = CheckpointProvenance('stage2', meta.class_mapping_sha256,
        meta.manifest_sha256, meta.split_sha256, meta.clip_weight_sha256, 750)
    trainer = FrozenHeadTrainer(build_classifier_head('linear', 512, 750),
        FrozenHeadConfig(**HEAD_CONFIG), device=args.device, num_classes=750, provenance=provenance)
    output = work / 'last.pt'
    if output.exists() and args.resume is None:
        raise FileExistsError('H1 output exists; use exact --resume or a new work directory')
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        if state.get('row_binding') != binding or state.get('training_scope') != scope:
            raise ValueError('H1 resume scope or exact row binding differs')
        trainer.load_state_dict(state)
    while trainer.epoch < 100:
        loss = trainer.train_epoch(x, y)
        if not math.isfinite(loss): raise RuntimeError('Nonfinite H1 loss')
        trainer.history.append(dict(epoch=trainer.epoch, train_loss=loss,
            learning_rate=float(trainer.optimizer.param_groups[0]['lr'])))
        state = trainer.state_dict()
        state.update(training_scope=scope, head=dict(kind='linear'), row_binding=binding,
            selection_rule=('fixed epoch100 last; full official train148695; no validation selection' if full
                else 'fixed epoch100 last; train-only initialization; no validation selection'),
            feature_cache_metadata=meta.__dict__ | dict(training_scope=scope, training_record_count=len(train)),
            git_commit='portable_archive_reproduction', fresh_feature_extraction=True,
            portable_h1_binding_sha256=sha(work / 'portable_h1_binding.json'))
        state.setdefault('cuda_rng_state_all', [])
        temp = output.with_suffix('.pt.tmp')
        torch.save(state, temp)
        temp.replace(output)
        print(json.dumps(dict(phase='H1', epoch=trainer.epoch, train_loss=loss)), flush=True)
    # Query is only evaluated after fixed100 fitting; never for optimizer/selection.
    parity = None if full else trainer.evaluate(cache.features[query].float(), cache.labels[query])['top1']
    receipt = dict(schema='aictrack1.portable.H1.receipt.v1', complete=True,
        checkpoint_sha256=sha(output), epoch=100, training_seed=20261008,
        row_binding=binding, scope=scope, training_count=len(train), validation_count=len(query),
        terminal_cache_parity_top1=parity, validation_selection=False, test_access=False,
        portable_h1_binding_sha256=sha(work / 'portable_h1_binding.json'))
    (work / 'h1_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(complete=True, checkpoint=str(output), sha256=receipt['checkpoint_sha256'])), flush=True)
    return 0
