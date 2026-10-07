"""Fail-closed SAM1 hooks into a single registered immutable base; no IO."""
import ast
import hashlib

BASE_SHA = 'c189491570efc8036c34d6312994db1f4d135d269ba1ba4ef368fb5a2a6083ee'


def adapt_source(data):
    if hashlib.sha256(data).hexdigest() != BASE_SHA:
        raise ValueError('unregistered immutable SAM1 base')
    source = data.decode('utf-8')
    def insert(anchor, value):
        nonlocal source
        if source.count(anchor) != 1:raise ValueError('ambiguous SAM1 hook')
        source = source.replace(anchor, value+anchor, 1)
    insert('    observation = observation_protocol(', '''    from aictrack1.methods.sharpness_update import (
        PROTOCOL, same_batch_sam_backward, protocol_state, verify_protocol_state,
    )
    sam1_setting = raw.get("sharpness_update")
    protocol_state(sam1_setting)
    if sam1_setting is not None:
        if (training_scope != "fixed_split_validation"
                or raw.get("confidence_observation_view", "strong") != "strong"
                or raw.get("bootstrap_curriculum") is not None
                or raw.get("persistent_filter", {"enabled": False}) != {"enabled": False}
                or raw.get("prefix_learning_rate_scale") is not None
                or raw.get("token_readout", {}).get("mode") != "none"
                or raw.get("trainable_ema", {}).get("enabled", False)
                or raw.get("base_loss", {}).get("kind") != "sce"
                or raw.get("view_consistency") is not None
                or raw.get("input_mix") is not None):
            raise ValueError("SAM1 isolated original strong SCE+AC same-batch two-step gradient")
''')
    insert('    if epoch == 0 and training_scope == "fixed_split_validation":', '''    if args.resume is not None:
        verify_protocol_state(state.get("sharpness_protocol"), sam1_setting)
''')
    insert('        for batch in loader:\n', '''        sam1_batches = 0
        sam1_forward_samples = 0
        sam1_first_overflow_samples = 0
        sam1_second_overflow_batches = 0
        sam1_loss_sum = torch.zeros((), dtype=torch.float64, device=device) if sam1_setting is not None else None
''')
    anchor='                scaler.scale(loss).backward()\n'
    if source.count(anchor)!=1:raise ValueError('SAM1 backward anchor drift')
    replacement='''            if sam1_setting is None:
                scaler.scale(loss).backward()
            else:
                def sam1_closure():
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        second_logits = model(images)
                        second_loss_logits = second_logits if logit_adjustment is None else second_logits + logit_adjustment.to(device=device, dtype=second_logits.dtype)
                        second_per_sample = _base_loss_per_sample(second_loss_logits, labels,
                            kind=base_loss_kind, q=base_loss_q, label_smoothing=label_smoothing,
                            sce_alpha=sce_alpha, sce_beta=sce_beta, sce_log_zero=sce_log_zero,
                            nce_alpha=nce_alpha, nce_beta=nce_beta, nce_log_zero=nce_log_zero,
                            elr_loss=elr_loss, sample_indices=batch["sample_index"])
                        return _weighted_base_loss(second_per_sample, batch_weights)
                sam1_result = same_batch_sam_backward(loss, sam1_closure,
                    [(n,p) for n,p in model.named_parameters() if p.requires_grad], scaler, sam1_setting)
                if sam1_result["second_forward"]:
                    sam1_loss_sum += sam1_result["perturbed_loss"].to(dtype=torch.float64) * len(labels)
                    sam1_forward_samples += len(labels)
                    sam1_batches += 1
                else:
                    sam1_first_overflow_samples += len(labels)
                sam1_second_overflow_batches += int(sam1_result["second_gradient_overflow"])
'''
    # The immutable trainer nests updates under PF1's has_supervision branch.
    # SAM1 forbids enabling PF1, but must preserve that exact disabled structure.
    replacement=''.join('    '+line if line.strip() else line for line in replacement.splitlines(keepends=True))
    source=source.replace(anchor,replacement,1)
    insert('        row["bootstrap_curriculum"] = ', '''        row["sharpness_update"] = {
            "protocol": protocol_state(sam1_setting),
            "perturbed_sce_mean": float(sam1_loss_sum / sam1_forward_samples) if sam1_forward_samples else None,
            "perturbed_forward_samples": sam1_forward_samples,
            "first_amp_overflow_samples": sam1_first_overflow_samples,
            "second_amp_overflow_batches": sam1_second_overflow_batches,
            "formal_ac_observations": count,
            "two_backward_batches": sam1_batches,
            "parameters_at_base_when_checkpointed": True,
        }
''')
    insert('        checkpoint["token_readout"] = raw["token_readout"]', '''        checkpoint["sharpness_protocol"] = protocol_state(sam1_setting)
''')
    insert('    _save_atomic(selected_checkpoint, output / selected_name)', '''    selected_checkpoint["sharpness_protocol"] = protocol_state(sam1_setting)
''')
    ast.parse(source)
    return source
