"""P1 fold H1 loader; prevents reuse of the full-train FV2 head."""
from pathlib import Path
import torch
from p1_contract import ROOT, assets, sha

def load_new_fold_head(path, **kwargs):
    from aictrack1.models.heads import load_classifier_head_checkpoint
    payload=assets()
    if str(Path(path).resolve()) != str(Path(payload["assets"]["h1"]["path"]).resolve()):
        raise ValueError("unregistered H1 path")
    if sha(path) != payload["assets"]["h1"]["sha256"]:
        raise ValueError("H1 identity drift")
    kwargs["forbid_full_train_initializer"] = True
    head,state=load_classifier_head_checkpoint(path,**kwargs)
    meta=state["feature_cache_metadata"]
    if (state.get("training_scope") == "full_train_fixed_epochs" or meta.get("training_scope") == "full_train_fixed_epochs"
            or state.get("epoch") != payload["h1_epoch"] or state.get("head") != {"kind":"linear"}
            or meta.get("normalized") is not True):
        raise ValueError("train-only fixed100 normalized linear H1 required")
    binding=state.get("row_binding",{})
    if binding != payload["h1_row_binding"]:raise ValueError("fresh exact H1 row binding mismatch")
    if binding.get("counts") != {"train":payload["train_count"],"validation":payload["validation_count"]}:
        raise ValueError("H1 positive fold row binding required")
    if binding.get("validation_used_for_optimization_or_selection") is not False:
        raise ValueError("H1 validation selection/optimization forbidden")
    if set(state["model"]) != {"linear.weight","linear.bias"} or any(not torch.isfinite(x).all() for x in state["model"].values()):
        raise ValueError("finite single linear H1 required")
    return head,state

