"""P1 deployment identities. No data, checkpoint, download or GPU side effects."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"C": 0.0, "J5": 5e-06}
CLIP_SHA = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
MANIFEST_SHA = "1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d"
MAPPING_SHA = "811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879"

def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            d.update(block)
    return d.hexdigest()

def validate_assets(payload):
    if payload.get("schema") != "aictrack1.p1.assets.v1" or payload.get("status") != "verified_fold_assets":
        raise ValueError("root must bind verified fold assets before launch")
    if (payload.get("competition_stage") != "stage2" or payload.get("train_count") != 118958
            or payload.get("validation_count") != 29737 or payload.get("num_classes") != 750):
        raise ValueError("P1 exact Stage2 fold counts required")
    if payload.get("h1_training_scope") == "full_train_fixed_epochs":
        raise ValueError("full H1 forbidden in a heldout experiment")
    if (payload.get("h1_training_count") != payload["train_count"]
            or payload.get("h1_validation_used_for_selection") is not False
            or payload.get("h1_epoch") != 100):
        raise ValueError("positive train-only H1100 declaration required")
    if not math.isfinite(float(payload.get("terminal_cache_parity_top1", float("nan")))):
        raise ValueError("verified H1 terminal feature parity required")
    for name in ("manifest", "split", "class_mapping", "clip", "h1", "label_conflicts"):
        item = payload.get("assets", {}).get(name, {})
        path, digest = item.get("path"), item.get("sha256")
        if not isinstance(path, str) or not path.startswith("/") or "PLACEHOLDER" in path:
            raise ValueError("exact absolute remote asset path required: " + name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("exact lowercase SHA required: " + name)
    if payload["assets"]["clip"]["sha256"] != CLIP_SHA or payload["assets"]["manifest"]["sha256"] != MANIFEST_SHA:
        raise ValueError("official CLIP/train identity differs")
    for name in ("dataset_root", "canonical_lock"):
        if not isinstance(payload.get(name), str) or not payload[name].startswith("/"):
            raise ValueError("absolute remote path required: " + name)
    if payload["canonical_lock"] != "/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json":
        raise ValueError("must use existing repository-wide canonical training lock")
    return payload

def assets(root=ROOT):
    return validate_assets(json.loads((Path(root)/"inputs/p1_assets.json").read_text(encoding="utf-8")))

def verify_files(root=ROOT):
    root = Path(root).resolve()
    path = root / "source_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "aictrack1.p1.source.v1":
        raise ValueError("P1 source schema mismatch")
    for rel, expected in manifest["files"].items():
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or not (root/p).resolve().is_relative_to(root):
            raise ValueError("source path escape")
        if sha(root/p) != expected:
            raise ValueError("P1 source drift: " + rel)
    return manifest

def verify_remote_assets(root=ROOT):
    payload = assets(root)
    for name, item in payload["assets"].items():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("P1 input drift: " + name)
    return payload

def validate_config(raw, arm, payload):
    if arm not in ARMS or raw.get("training_scope") != "fixed_split_validation":
        raise ValueError("unknown arm or non-heldout scope")
    train = raw["training"]
    expected = dict(epochs=20, batch_size=64, workers=8, prefetch_factor=2,
                    persistent_workers=True, seed=payload["training_seed"],
                    visual_learning_rate=1.75e-6, head_learning_rate=ARMS[arm],
                    weight_decay=.01, gradient_clip_norm=1.)
    if train != expected:
        raise ValueError("P1 training policy drift")
    if (raw.get("peft") != {"mode":"full_visual", "lora_blocks":8}
            or raw.get("base_loss") != {"kind":"sce", "alpha":.1, "beta":1., "log_zero":-4.}
            or raw.get("selection_rule") != "fixed_final"
            or raw.get("confidence_observation_view") != "strong"):
        raise ValueError("P1 algorithm drift")
    for setting, name in (("manifest","manifest"),("split","split"),
                          ("head_checkpoint","h1"),("label_conflicts","label_conflicts")):
        if raw.get(setting) != payload["assets"][name]["path"]:
            raise ValueError("configuration asset mismatch: " + setting)

