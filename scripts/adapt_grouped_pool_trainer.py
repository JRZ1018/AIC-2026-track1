"""Fail-closed mechanical adaptation of the registered original strong trainer.

Both new arms use this same original implementation. No writes at import.
"""
import ast
import hashlib

BASE_SHA = 'c189491570efc8036c34d6312994db1f4d135d269ba1ba4ef368fb5a2a6083ee'


def adapt_source(source_bytes):
    if hashlib.sha256(source_bytes).hexdigest() != BASE_SHA:
        raise ValueError('unregistered immutable base trainer')
    source = source_bytes.decode('utf-8')

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f'absent or ambiguous trainer hook: {old[:80]}')
        source = source.replace(old, new, 1)

    replace('    transforms = build_split_transforms(dataset_config.transforms)\n', '''    from aictrack1.methods.grouped_training_pool import (
        bound_records, MatchedExposureDataset, aggregate_observations, verify_protocol,
    )
    pool_setting = raw.get("training_pool")
    full_train_records = train_records
    if pool_setting is not None:
        if (training_scope != "fixed_split_validation"
                or not raw.get("confidence_weighting", {}).get("enabled", False)
                or raw.get("confidence_observation_view", "strong") != "strong"
                or raw.get("bootstrap_curriculum") is not None
                or raw.get("input_mix") is not None or raw.get("view_consistency") is not None
                or raw.get("sharpness_update") is not None
                or raw.get("persistent_filter", {"enabled": False}) != {"enabled": False}
                or raw.get("prefix_learning_rate_scale") is not None
                or raw.get("token_readout", {}).get("mode") != "none"
                or raw.get("trainable_ema", {}).get("enabled", False)
                or raw.get("base_loss", {}).get("kind") != "sce"
                or training.get("sampling", {"mode": "natural"}) != {"mode": "natural"}):
            raise ValueError("training pool is isolated original strong SCE/AC")
    train_records, pool_protocol = bound_records(train_records, pool_setting, root=ROOT)
    transforms = build_split_transforms(dataset_config.transforms)
''')
    replace('        training_class_counts,\n        head_fraction=0.2,', '''        torch.bincount(torch.tensor([int(r.internal_class_index) for r in full_train_records]), minlength=num_classes),
        head_fraction=0.2,''')
    replace('    if epoch == 0 and training_scope == "fixed_split_validation":', '''    if args.resume is not None:
        verify_protocol(state.get("training_pool_protocol"), pool_protocol)

    if epoch == 0 and training_scope == "fixed_split_validation":''')
    replace('        loader = _loader(\n            train_dataset,', '''        epoch_dataset = (train_dataset if pool_protocol is None else MatchedExposureDataset(
            train_dataset, exposures=pool_protocol["full_count"], seed=seed, epoch=epoch))
        pool_optimizer_steps = 0
        pool_attempted_steps = 0
        pool_amp_overflow_steps = 0
        loader = _loader(
            epoch_dataset,''')
    replace('            shuffle=True,\n            seed=seed + epoch,', '            shuffle=pool_protocol is None,\n            seed=seed + epoch,')
    replace('                scaler.step(optimizer)\n                scaler.update()\n', '''                pool_scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                pool_step_ok = optimizer_step_completed(scaler_enabled=scaler.is_enabled(),
                    before=pool_scale_before, after=scaler.get_scale())
                pool_attempted_steps += 1
                pool_optimizer_steps += int(pool_step_ok)
                pool_amp_overflow_steps += int(not pool_step_ok)
''')
    replace('            next_confidence = confidence.update_epoch(\n', '''            if pool_protocol is not None:
                if len(completed_indices) != pool_protocol["full_count"]:
                    raise ValueError("matched exposure count drift")
                (completed_indices, completed_losses, completed_observed_probabilities,
                 completed_predictions, pool_id_exposures) = aggregate_observations(
                    confidence.sample_indices, completed_indices, completed_losses,
                    completed_observed_probabilities, completed_predictions, num_classes=num_classes)
            next_confidence = confidence.update_epoch(
''')
    replace('        if gate is not None:\n            row["persistent_filter"]', '''        if pool_protocol is not None:
            row["training_pool"] = dict(protocol=pool_protocol,
                physical_exposures=count, unique_ac_observations=len(train_records),
                exposure_min=int(pool_id_exposures.min()), exposure_max=int(pool_id_exposures.max()),
                attempted_optimizer_steps=pool_attempted_steps, optimizer_steps=pool_optimizer_steps,
                amp_overflow_steps=pool_amp_overflow_steps, tail_definition="original full fold train")
        if gate is not None:
            row["persistent_filter"]''')
    replace('        checkpoint["token_readout"] = raw["token_readout"]', '''        checkpoint["training_pool_protocol"] = pool_protocol
        checkpoint["token_readout"] = raw["token_readout"]''')
    replace('    _save_atomic(selected_checkpoint, output / selected_name)', '''    selected_checkpoint["training_pool_protocol"] = pool_protocol
    _save_atomic(selected_checkpoint, output / selected_name)''')
    ast.parse(source)
    return source
