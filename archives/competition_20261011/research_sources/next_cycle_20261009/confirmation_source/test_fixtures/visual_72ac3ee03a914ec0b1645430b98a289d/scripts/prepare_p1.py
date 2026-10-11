"""Bind verified assets, write C/J5/J20 configs, then freeze deployable source."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from p1_contract import ROOT, ARMS, assets, sha, validate_assets

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")

def prepare(payload=None, root=ROOT):
    root = Path(root)
    enabled = payload is not None
    if enabled:
        validate_assets(payload)
        write_json(root/"inputs/p1_assets.json", payload)
    values = payload or dict(training_seed=20261008, dataset_root="/PLACEHOLDER_DATASET_ROOT",
        split_seed=20261003, assets={name:dict(path="/PLACEHOLDER_"+name.upper(),sha256="")
        for name in ("manifest","split","class_mapping","clip","h1","label_conflicts")})
    asset = values["assets"]
    model = {"model":dict(provider="OpenAI",name="ViT-B/32",
        upstream_commit="d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
        weight_path=asset["clip"]["path"],expected_weight_sha256=asset["clip"]["sha256"],
        runtime_downloads_allowed=False)}
    data = {"dataset":dict(competition_stage="stage2", contract_confirmed=True,
        expected_train_count=148695,expected_test_count=37444,expected_num_classes=750,
        stage="rematch",dataset_root=values["dataset_root"],
        class_mapping=asset["class_mapping"]["path"],class_mapping_provenance="train_directory_names",
        extensions=[".jpg"],allow_truncated_images=False,
        sources=dict(train=dict(root="train",label_source="directory",class_from_parent=True),
                     test=dict(root="__unavailable_test__",label_source="none")),
        audit=dict(output_root="../../artifacts/audits/stage2",workers=4,near_duplicate_hamming=3,
            near_duplicate_max_bucket=500,fail_on_bad_files=True,fail_on_label_conflicts=False),
        split=dict(validation_fraction=.2,seed=values.get("split_seed",20261003),min_train_per_class=1),
        transforms=dict(image_size=336,resize_size=336,train_crop_scale_min=.8,train_horizontal_flip_probability=.5),
        submission=dict(filename_mode="basename",official_class_id_pattern=r"\d{4}"))}
    for name,value in (("model/p1_clip.yaml",model),("data/p1_336.yaml",data)):
        (root/"configs"/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding="utf-8")
    for arm,lr in ARMS.items():
        raw = dict(competition_stage="stage2",run_id=f"20261009-p1-{arm}-fullvisual20",
            execution_enabled=enabled,training_scope="fixed_split_validation",selection_rule="fixed_final",
            confidence_observation_view="strong",dataset_config="../data/p1_336.yaml",model_config="../model/p1_clip.yaml",
            manifest=asset["manifest"]["path"],split=asset["split"]["path"],
            label_conflicts=asset["label_conflicts"]["path"],head_checkpoint=asset["h1"]["path"],
            output_dir=f"../../artifacts/experiments/P1_{arm}",device="cuda",visual_input_resolution=336,
            position_interpolation="bicubic_7x7_to_10x10",initial_top1_tolerance=.002,
            base_loss=dict(kind="sce",alpha=.1,beta=1.,log_zero=-4.),
            runtime=dict(defer_epoch_metrics=True,reuse_validation_loader=True,
                record_gradient_diagnostics=True,safe_numpy_rng_checkpoint=True),
            peft=dict(mode="full_visual",lora_blocks=8),
            confidence_weighting=dict(enabled=True,warmup_epochs=1,momentum=.8,history_mix=.5,
                min_weight=.02,magnitude_mix=.75,stop_on_effective_quality_l2=True),
            training=dict(epochs=20,batch_size=64,workers=8,prefetch_factor=2,persistent_workers=True,
                seed=values["training_seed"],visual_learning_rate=1.75e-6,head_learning_rate=lr,
                weight_decay=.01,gradient_clip_norm=1.),
            token_readout=dict(mode="none",initialization_seed=20261005))
        (root/f"configs/experiment/p1_{arm}.yaml").write_text(yaml.safe_dump({"experiment":raw},sort_keys=False),encoding="utf-8")
    files = {}
    for p in sorted(root.rglob("*")):
        if (p.is_file() and "__pycache__" not in p.parts and "artifacts" not in p.relative_to(root).parts
                and p.name not in {"source_manifest.json","cpu_smoke_result.json"}
                and p.suffix in {".py",".yaml",".json",".txt",".md"}):
            files[p.relative_to(root).as_posix()] = sha(p)
    manifest = dict(schema="aictrack1.p1.source.v1", status="bound" if enabled else "unbound_design_only",
        files=files, upstream="FV2 verified full source; no binary/data copied",
        mechanical_changes=["fold asset/lease guard","train-only H1 binding",
            "n1 explicit resume whitelist","optional pause with original T_max20",
            "fixed-final export instead of best selection; epoch loop unchanged"])
    write_json(root/"source_manifest.json",manifest)
    return manifest

if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--assets",type=Path)
    args=parser.parse_args()
    payload=json.loads(args.assets.read_text(encoding="utf-8")) if args.assets else None
    result=prepare(payload)
    print(json.dumps(dict(status=result["status"],files=len(result["files"]),source_sha256=sha(ROOT/"source_manifest.json"))))

