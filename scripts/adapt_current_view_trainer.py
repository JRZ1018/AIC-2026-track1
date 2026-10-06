"""Mechanical, fail-closed VC1 adaptation of one audited immutable trainer.

No training or file writes at import. The eventual independent scope builder
must freeze the returned source and this adapter alongside the original SHA.
"""
import ast
import hashlib

BASE_SHA = "c189491570efc8036c34d6312994db1f4d135d269ba1ba4ef368fb5a2a6083ee"


def adapt_source(source_bytes: bytes) -> str:
    if hashlib.sha256(source_bytes).hexdigest() != BASE_SHA:
        raise ValueError("Unregistered trainer source; do not adapt by guessing")
    source = source_bytes.decode("utf-8")

    def insert_before(anchor: str, code: str) -> None:
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError("Trainer hook is absent or ambiguous")
        source = source.replace(anchor, code + anchor, 1)

    insert_before('    observation = observation_protocol(', '''    from aictrack1.methods.current_view_consistency import (
        attach_center_view, add_current_view_loss, protocol_state, verify_protocol_state,
    )
    vc1_setting = raw.get("view_consistency")
    if vc1_setting is not None:
        if (training_scope != "fixed_split_validation"
                or raw.get("confidence_observation_view", "strong") != "strong"
                or raw.get("bootstrap_curriculum") is not None
                or raw.get("persistent_filter", {"enabled": False}) != {"enabled": False}
                or raw.get("prefix_learning_rate_scale") is not None
                or raw.get("token_readout", {}).get("mode") != "none"
                or raw.get("trainable_ema", {}).get("enabled", False)
                or raw.get("base_loss", {}).get("kind") != "sce"):
            raise ValueError("VC1 is an isolated strong-AC/CLS/last8 SCE candidate")
    train_dataset = attach_center_view(train_dataset, transforms["validation"], vc1_setting)
''')
    insert_before('    if epoch == 0 and training_scope == "fixed_split_validation":', '''    if args.resume is not None:
        verify_protocol_state(state.get("view_consistency_protocol"), vc1_setting)
''')
    insert_before('        for batch in loader:\n', '''        vc1_js_sum = torch.zeros((), dtype=torch.float64, device=device) if vc1_setting is not None else None
''')
    insert_before('            if not torch.isfinite(loss):\n', '''            if vc1_setting is not None:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    center_logits = model(batch["consistency_center_image"].to(device, non_blocking=True))
                loss, vc1_js = add_current_view_loss(loss, logits, center_logits, vc1_setting)
                vc1_js_sum += vc1_js.detach().to(dtype=torch.float64) * len(labels)
''')
    insert_before('        row["bootstrap_curriculum"] = ', '''        row["view_consistency"] = {
            "protocol": protocol_state(vc1_setting),
            "js_mean": float(vc1_js_sum / count) if vc1_js_sum is not None else None,
            "center_forward_samples": count if vc1_setting is not None else 0,
            "formal_ac_observations": count,
        }
''')
    insert_before('        checkpoint["token_readout"] = raw["token_readout"]', '''        checkpoint["view_consistency_protocol"] = protocol_state(vc1_setting)
''')
    insert_before('    _save_atomic(selected_checkpoint, output / selected_name)', '''    selected_checkpoint["view_consistency_protocol"] = protocol_state(vc1_setting)
''')
    ast.parse(source)
    return source
