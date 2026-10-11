"""Exact frozen-original trainer adapter; no change to AC, crops or loss."""
import ast
import hashlib

BASE_SHA = 'c189491570efc8036c34d6312994db1f4d135d269ba1ba4ef368fb5a2a6083ee'

def adapt(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != BASE_SHA:
        raise ValueError('unregistered original trainer')
    text = source.decode('utf-8')
    def once(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('trainer hook drift')
        text = text.replace(old, new)
    once('from aictrack1.models.peft import PEFTImageClassifier, configure_visual_peft',
         'from aictrack1.models.peft import PEFTImageClassifier, configure_visual_peft\nfrom aictrack1.methods.low_rank_visual import LR1, configure_low_rank_visual, validate_recipe, validate_delta_state, visual_initialization_device')
    once('load_clip_vit_b32(weight_path, expected_weight_sha256, device)',
         'load_clip_vit_b32(weight_path, expected_weight_sha256, visual_initialization_device(device, raw.get("low_rank_visual")))')
    old='''    summary = configure_visual_peft(
        visual,
        str(peft["mode"]),
        lora_rank=int(peft.get("lora_rank", 4)),
        lora_alpha=float(peft.get("lora_alpha", 8.0)),
        lora_blocks=int(peft.get("lora_blocks", 1)),
    )'''
    once(old, '''    low_rank = raw.get("low_rank_visual")
    if low_rank is not None:
        validate_recipe(low_rank)
        if peft != {"mode": LR1["mode"], "lora_blocks": 8, "lora_rank": 8, "lora_alpha": 16.0}:
            raise ValueError("LR1 PEFT protocol drift")
        if float(raw["training"]["visual_learning_rate"]) != LR1["learning_rate"]:
            raise ValueError("LR1 learning rate drift")
        if any(raw.get(k) is not None for k in ("bootstrap_curriculum", "persistent_filter", "prefix_learning_rate_scale", "sharpness_update", "mixup_supervision", "view_consistency")):
            raise ValueError("LR1 cannot stack other candidate mechanisms")
        summary = configure_low_rank_visual(visual, low_rank)
    else:
''' + '\n'.join('    '+line for line in old.splitlines()))
    once('        checkpoint["token_readout"] = raw["token_readout"]',
         '        if low_rank is not None:\n            validate_delta_state(checkpoint["trainable_state"])\n            checkpoint["low_rank_visual_protocol"] = validate_recipe(low_rank)\n        checkpoint["token_readout"] = raw["token_readout"]')
    once('    _save_atomic(selected_checkpoint, output / selected_name)',
         '    if low_rank is not None:\n        validate_delta_state(selected_checkpoint["trainable_state"])\n        selected_checkpoint["low_rank_visual_protocol"] = validate_recipe(low_rank)\n    _save_atomic(selected_checkpoint, output / selected_name)')
    ast.parse(text)
    return text.encode('utf-8')
