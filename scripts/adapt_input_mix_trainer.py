"""Fail-closed adaptation of one immutable base; no writes or training at import."""
import ast
import hashlib

BASE_SHA = 'c189491570efc8036c34d6312994db1f4d135d269ba1ba4ef368fb5a2a6083ee'


def adapt_source(data):
    if hashlib.sha256(data).hexdigest() != BASE_SHA:
        raise ValueError('unregistered immutable base')
    source = data.decode('utf-8')
    def insert(anchor, value):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError('ambiguous or missing MX1 hook')
        source = source.replace(anchor, value+anchor, 1)
    insert('    observation = observation_protocol(', '''    from aictrack1.methods.mixup_supervision import (
        PROTOCOL, batch_plan, mix_inputs, add_current_mix_loss, protocol_state, verify_protocol_state,
    )
    mx1_setting = raw.get("input_mix")
    verify_protocol_state(mx1_setting, mx1_setting)
    if mx1_setting is not None:
        if (training_scope != "fixed_split_validation"
                or raw.get("confidence_observation_view", "strong") != "strong"
                or raw.get("bootstrap_curriculum") is not None
                or raw.get("persistent_filter", {"enabled": False}) != {"enabled": False}
                or raw.get("prefix_learning_rate_scale") is not None
                or raw.get("token_readout", {}).get("mode") != "none"
                or raw.get("trainable_ema", {}).get("enabled", False)
                or raw.get("base_loss", {}).get("kind") != "sce"
                or raw.get("view_consistency") is not None):
            raise ValueError("MX1 is isolated original strong-AC last8 SCE plus mixed supervision")
''')
    insert('    if epoch == 0 and training_scope == "fixed_split_validation":', '''    if args.resume is not None:
        verify_protocol_state(state.get("input_mix_protocol"), mx1_setting)
''')
    insert('        for batch in loader:\n', '''        mx1_step = 0
        mx1_loss_sum = torch.zeros((), dtype=torch.float64, device=device) if mx1_setting is not None else None
''')
    insert('            if not torch.isfinite(loss):\n', '''            if mx1_setting is not None:
                mx1_coefficient, mx1_order = batch_plan(len(labels), seed, epoch, mx1_step)
                mx1_images = mix_inputs(images, mx1_order, mx1_coefficient)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    mx1_logits = model(mx1_images)
                mx1_weights = torch.ones_like(per_sample_loss) if batch_weights is None else batch_weights
                loss, mx1_loss = add_current_mix_loss(loss, mx1_logits, labels, mx1_weights,
                    mx1_order, mx1_coefficient, mx1_setting)
                mx1_loss_sum += mx1_loss.detach().to(dtype=torch.float64) * len(labels)
                mx1_step += 1
''')
    insert('        row["bootstrap_curriculum"] = ', '''        row["input_mix"] = {
            "protocol": protocol_state(mx1_setting),
            "mixed_sce_mean": float(mx1_loss_sum / count) if mx1_loss_sum is not None else None,
            "mixed_forward_samples": count if mx1_setting is not None else 0,
            "formal_ac_observations": count,
            "mixed_batches": mx1_step,
        }
''')
    insert('        checkpoint["token_readout"] = raw["token_readout"]', '''        checkpoint["input_mix_protocol"] = protocol_state(mx1_setting)
''')
    insert('    _save_atomic(selected_checkpoint, output / selected_name)', '''    selected_checkpoint["input_mix_protocol"] = protocol_state(mx1_setting)
''')
    ast.parse(source)
    return source
