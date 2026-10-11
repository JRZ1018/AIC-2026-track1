"""Run one final test inference and atomically build the strict submission package."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aictrack1.data.class_mapping import ClassMapping
from aictrack1.data.contracts import (
    ensure_output_path_is_safe,
    load_dataset_config,
    validate_experiment_runtime_contract,
)
from aictrack1.data.indexed_images import IndexedImageDataset, read_indexed_images
from aictrack1.data.transforms import build_split_transforms
from aictrack1.engine.peft import (
    configure_classifier_trainability_from_state,
    load_trainable_state_dict,
)
from aictrack1.engine.training_queue import TrainingLease
from aictrack1.engine.provenance import (
    CheckpointProvenance,
    validate_checkpoint_provenance,
)
from aictrack1.evaluation.four_view import (
    PairedViewDataset,
    fixed_four_view_logits,
    resolve_four_view_resize,
)
from aictrack1.evaluation.balanced_prior import fit_uniform_soft_bias
from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
from aictrack1.models.clip_resolution import enable_interpolated_visual_resolution
from aictrack1.models.heads import load_classifier_head_checkpoint
from aictrack1.models.peft import (
    PEFTImageClassifier,
    configure_visual_peft,
    fold_visual_lora,
)


SUPPORTED_FINAL_PEFT_MODES = frozenset(
    {"full_visual"}
)
from aictrack1.submission import (
    SubmissionSpec,
    build_submission_package,
    submission_keys_from_relative_paths,
)
from final_contract import (
    FULL_SOURCE_SHA, FULL_ROOT, LOCK, P7_SHA, TEST_SHA, VIEW,
    require, read, sha, verify_source, runtime, validate_dataset_override,
    read_test_rows, resource_check, write_new,
)
from view_features import FourViewFeatureCollector, FEATURE_PROTOCOL


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256_variants(path: Path) -> set[str]:
    """Return hashes differing only by platform text line endings."""

    raw = path.read_bytes()
    text = raw.decode("utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        hashlib.sha256(normalized.replace("\n", "\r\n").encode("utf-8")).hexdigest(),
    }


def _model_settings(path: Path) -> tuple[Path, str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = document.get("model", document)
    if raw.get("provider") != "OpenAI" or raw.get("name") != "ViT-B/32":
        raise ValueError("model config must declare OpenAI ViT-B/32")
    if raw.get("runtime_downloads_allowed") is not False:
        raise ValueError("runtime downloads must be disabled")
    return _resolve(path.parent, str(raw["weight_path"])), str(raw["expected_weight_sha256"])


def _validate_resolution_contract(experiment, dataset_config, checkpoint) -> int:
    resolution = int(experiment.get("visual_input_resolution", 224))
    if resolution != dataset_config.transforms.image_size:
        raise ValueError("inference resolution differs from dataset transform contract")
    if int(checkpoint.get("visual_input_resolution", 224)) != resolution:
        raise ValueError("checkpoint and inference visual resolutions differ")
    interpolation = experiment.get("position_interpolation")
    if resolution in (320, 336):
        if interpolation != "bicubic_7x7_to_10x10":
            raise ValueError("336 inference requires the preregistered position interpolation")
        if checkpoint.get("position_interpolation") != interpolation:
            raise ValueError("checkpoint and inference position interpolation differ")
    elif resolution != 224 or interpolation is not None:
        raise ValueError("unsupported final inference resolution contract")
    return resolution


def _fixed_horizontal_flip_logits(model, images: torch.Tensor) -> torch.Tensor:
    """The preregistered single-checkpoint two-view inference protocol."""

    center_logits = model(images).float()
    flipped_logits = model(torch.flip(images, dims=(-1,))).float()
    return (center_logits + flipped_logits) / 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument('--inference-dataset-config', required=True, type=Path)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--cpu-audit', required=True, type=Path)
    parser.add_argument('--cpu-audit-sha256', required=True)
    parser.add_argument('--resource-policy', required=True, type=Path)
    parser.add_argument('--resource-policy-sha256', required=True)
    parser.add_argument('--save-features', action='store_true')
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--source-git-commit",
        help="verified commit of the source workspace when the inference host has no Git metadata",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--balanced-prior-uniform-soft",
        action="store_true",
        help="apply the frozen uniform soft-mass inference prior after all logits are collected",
    )
    parser.add_argument("--balanced-prior-temperature-multiplier", type=float)
    parser.add_argument("--balanced-prior-bias-strength", type=float)
    parser.add_argument("--balanced-prior-selection-report", type=Path)
    parser.add_argument("--expected-balanced-prior-report-sha256")
    view_protocol = parser.add_mutually_exclusive_group()
    view_protocol.add_argument(
        "--tta-horizontal-flip",
        action="store_true",
        help="fixed equal-logit average of native and horizontal-flip views from one checkpoint",
    )
    view_protocol.add_argument(
        "--tta-four-view-scale384",
        "--four-view-scale384",
        dest="tta_four_view_scale384",
        action="store_true",
        help=(
            "fixed equal-logit average of native336, its exact flip, "
            "resize_short_side384/center_crop336 and its exact flip from one checkpoint"
        ),
    )
    args = parser.parse_args()
    verify_source(expected=args.source_sha256)
    runtime_value = runtime(torch)
    require(sha(args.cpu_audit)==args.cpu_audit_sha256, 'actual FULL CPU audit SHA differs')
    frozen = read(args.cpu_audit)
    require(frozen.get('schema')=='aictrack1.fresh_route.full_final_cpu_audit.v1' and frozen.get('complete') is True
        and frozen.get('full_training_source_sha256')==FULL_SOURCE_SHA
        and frozen.get('inference_source_sha256')==args.source_sha256
        and frozen.get('selected_route') in ('L50_GRAPH',)
        and frozen.get('trainable_tensors')==(152 if frozen['selected_route']=='S' else 154)
        and frozen.get('trainable_numel')==(87849216 if frozen['selected_route']=='S' else 88233966)
        and frozen.get('final_last_all_tensors_equal') is True and frozen.get('classifier_parameters_restored_equal') is True
        and frozen.get('test_access') is False and frozen.get('train_count')==148695 and frozen.get('validation_count')==0,
        'complete actual typed FULL152/154 CPU recovery is required before test access')
    require(sha(args.resource_policy)==args.resource_policy_sha256, 'resource policy SHA differs')
    policy = read(args.resource_policy)
    resource_check(ROOT, policy, reserve=1024**3)
    require(args.tta_four_view_scale384 and args.balanced_prior_uniform_soft
        and args.balanced_prior_temperature_multiplier==.5 and args.balanced_prior_bias_strength==.75
        and args.expected_balanced_prior_report_sha256==P7_SHA, 'only fixed4V/P7.5/.75/60 is allowed')
    require(args.expected_checkpoint_sha256==frozen['final_sha256']
        and args.checkpoint.resolve()==Path(frozen['evidence']['final']['path']).resolve(), 'actual audited final checkpoint differs')
    require(args.experiment_config.resolve()==Path(frozen['evidence']['config']['path']).resolve(), 'actual audited original config differs')
    for name,item in frozen['evidence'].items():
        require(sha(item['path'])==item['sha256'], 'actual FULL CPU evidence changed: '+name)
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    balanced_values = (
        args.balanced_prior_temperature_multiplier,
        args.balanced_prior_bias_strength,
    )
    balanced_report_values = (
        args.balanced_prior_selection_report,
        args.expected_balanced_prior_report_sha256,
    )
    if args.balanced_prior_uniform_soft:
        if any(value is None or value <= 0 for value in balanced_values):
            raise ValueError("balanced prior requires positive temperature and bias strength")
        if any(value is None for value in balanced_report_values):
            raise ValueError("balanced prior requires its frozen train-derived selection report")
    elif any(value is not None for value in balanced_values):
        raise ValueError("balanced prior values require --balanced-prior-uniform-soft")
    elif any(value is not None for value in balanced_report_values):
        raise ValueError("balanced prior report requires --balanced-prior-uniform-soft")
    git_commit = (
        args.source_git_commit.lower()
        if args.source_git_commit is not None
        else subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ValueError("source Git commit must be a 40-character SHA-1")

    config_path = args.experiment_config.resolve()
    config_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config_document.get("experiment", config_document)
    if experiment.get("execution_enabled", True) is not True:
        raise RuntimeError("this experiment configuration is not approved for inference")
    base = config_path.parent
    original_dataset_path = _resolve(base, str(experiment["dataset_config"]))
    dataset_path = args.inference_dataset_config.resolve()
    validate_dataset_override(yaml.safe_load(original_dataset_path.read_text(encoding='utf-8')),
        yaml.safe_load(dataset_path.read_text(encoding='utf-8')))
    model_path = _resolve(base, str(experiment["model_config"]))
    head_checkpoint_setting = experiment.get(
        "head_checkpoint", experiment.get("a1_head_checkpoint")
    )
    if head_checkpoint_setting is None:
        raise ValueError("experiment must define head_checkpoint or legacy a1_head_checkpoint")
    if "head_checkpoint" in experiment and "a1_head_checkpoint" in experiment:
        raise ValueError("experiment must not define both head checkpoint fields")
    head_path = _resolve(base, str(head_checkpoint_setting))
    split_path = _resolve(base, str(experiment["split"]))
    checkpoint_path = args.checkpoint.resolve()
    test_manifest = args.test_manifest.resolve()
    dataset_config = load_dataset_config(dataset_path)
    competition_stage = validate_experiment_runtime_contract(
        experiment, dataset_config
    )
    output = ensure_output_path_is_safe(dataset_config, args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"submission output directory is not empty: {output}")
    lease = TrainingLease.acquire(
        LOCK,
        owner="stage2-final-single-checkpoint-inference",
        config=str(config_path),
        output_dir=str(output),
        command=tuple(sys.argv),
    )
    atexit.register(lease.release)
    # The canonical lease exists before the first official test manifest/image read.
    official_test_rows = read_test_rows(test_manifest)

    balanced_selection_metadata = None
    if args.balanced_prior_uniform_soft:
        selection_report = args.balanced_prior_selection_report.resolve()
        selection_report_sha256 = _sha256(selection_report)
        if selection_report_sha256 != args.expected_balanced_prior_report_sha256.lower():
            raise ValueError("balanced-prior selection report SHA-256 mismatch")
        selection = json.loads(selection_report.read_text(encoding="utf-8"))
        expected_name = (
            f"uniform_soft_t{args.balanced_prior_temperature_multiplier:g}_"
            f"s{args.balanced_prior_bias_strength:g}"
        )
        if (
            selection.get("scope")
            != "Stage 2 train-derived balanced proxy only; no test access"
            or selection.get("gate_passed") is not True
            or selection.get("selected_candidate") != expected_name
        ):
            raise ValueError("balanced-prior parameters do not match the frozen train-derived gate")
        balanced_selection_metadata = {
            "report_sha256": selection_report_sha256,
            "selected_candidate": expected_name,
            "gate": selection["gate"],
        }

    checkpoint_sha256 = _sha256(checkpoint_path)
    if args.tta_four_view_scale384 and not args.expected_checkpoint_sha256:
        raise ValueError("four-view inference requires a frozen expected checkpoint SHA-256")
    if args.expected_checkpoint_sha256 and checkpoint_sha256 != args.expected_checkpoint_sha256.lower():
        raise ValueError("checkpoint SHA-256 differs from the frozen expected checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config_sha256 = _sha256(config_path)
    if checkpoint.get("config_sha256") not in _text_sha256_variants(config_path):
        raise ValueError(
            "checkpoint and experiment configuration hashes differ beyond LF/CRLF normalization"
        )
    peft = experiment["peft"]
    if checkpoint.get("mode") != peft.get("mode"):
        raise ValueError("checkpoint and experiment PEFT modes differ")
    if peft.get("mode") not in SUPPORTED_FINAL_PEFT_MODES:
        raise ValueError("unsupported final PEFT mode")
    visual_resolution = _validate_resolution_contract(experiment, dataset_config, checkpoint)

    mapping = ClassMapping.from_csv(dataset_config.class_mapping)
    test_records = read_indexed_images(
        dataset_config,
        source="test",
        manifest_path=test_manifest,
    )
    validate_experiment_runtime_contract(
        experiment,
        dataset_config,
        test_count=len(test_records),
        num_classes=len(mapping),
    )
    submission_spec = SubmissionSpec(
        include_header=False,
        filename_mode=dataset_config.submission.filename_mode,
        official_class_id_pattern=dataset_config.submission.official_class_id_pattern,
    )
    filenames = submission_keys_from_relative_paths(
        [record.relative_path for record in test_records],
        spec=submission_spec,
    )

    weight_path, expected_weight_sha256 = _model_settings(model_path)
    provenance = None
    if competition_stage is not None:
        train_manifest = _resolve(base, str(experiment["manifest"]))
        provenance = CheckpointProvenance(
            competition_stage=competition_stage,
            class_mapping_sha256=mapping.digest(),
            train_manifest_sha256=_sha256(train_manifest),
            split_sha256=_sha256(split_path),
            clip_weight_sha256=expected_weight_sha256,
            num_classes=len(mapping),
        )
        validate_checkpoint_provenance(
            checkpoint,
            provenance,
            context="inference checkpoint",
        )
    elif checkpoint.get("class_mapping_sha256") != mapping.digest():
        raise ValueError("checkpoint and class mapping hashes differ")

    device = torch.device(str(experiment.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    clip_model, _, clip_metadata = load_clip_vit_b32(weight_path, expected_weight_sha256, device)
    if checkpoint.get("clip_weight_sha256") != clip_metadata.weight_sha256:
        raise ValueError("checkpoint and official CLIP weight hashes differ")
    visual = clip_model.visual.float()
    del clip_model
    if visual_resolution in (320, 336):
        enable_interpolated_visual_resolution(visual, target_resolution=visual_resolution)
    configure_visual_peft(
        visual,
        str(peft["mode"]),
        lora_rank=int(peft.get("lora_rank", 4)),
        lora_alpha=float(peft.get("lora_alpha", 8.0)),
        lora_blocks=int(peft.get("lora_blocks", 1)),
    )
    head, head_checkpoint = load_classifier_head_checkpoint(
        head_path,
        feature_dim=clip_metadata.feature_dim,
        num_classes=len(mapping),
        expected_clip_weight_sha256=clip_metadata.weight_sha256,
        expected_class_mapping_sha256=mapping.digest(),
        expected_split_sha256=_sha256(split_path),
        expected_provenance=provenance,
    )
    expected_head_sha256 = checkpoint.get("head_initializer_sha256")
    if expected_head_sha256 is not None and expected_head_sha256 != _sha256(head_path):
        raise ValueError("checkpoint and classifier head initializer hashes differ")
    normalized_features = head_checkpoint["feature_cache_metadata"].get("normalized")
    if not isinstance(normalized_features, bool):
        raise ValueError("head checkpoint must declare whether cached features were normalized")
    checkpoint_normalized = checkpoint.get("normalize_features")
    if checkpoint_normalized is not None and checkpoint_normalized != normalized_features:
        raise ValueError("checkpoint and classifier head feature normalization differ")
    configure_classifier_trainability_from_state(head, checkpoint["trainable_state"])
    model = PEFTImageClassifier(
        visual,
        head,
        normalize_features=normalized_features,
    ).to(device)
    load_trainable_state_dict(model, checkpoint["trainable_state"])
    require(len(checkpoint['trainable_state'])==frozen['trainable_tensors'] and sum(v.numel() for v in checkpoint['trainable_state'].values())==frozen['trainable_numel'],
        'final export omitted joint classifier tensors')
    for name,parameter in model.named_parameters():
        if name in checkpoint['trainable_state']:
            require(torch.equal(parameter.detach().cpu(), checkpoint['trainable_state'][name].cpu()), 'inference restore differs: '+name)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    lora_folded = str(peft["mode"]).startswith("lora_")
    if lora_folded:
        fold_visual_lora(model.visual)
    model.eval()
    graph_route=frozen['selected_route']=='L50_GRAPH'
    feature_collector = FourViewFeatureCollector(model.visual) if (args.save_features or graph_route) else None
    features=None;graph_metadata=None

    transforms = build_split_transforms(dataset_config.transforms)
    four_view_resize = resolve_four_view_resize(
        visual_resolution, "scale384" if args.tta_four_view_scale384 else None
    )
    if four_view_resize is None:
        test_dataset = IndexedImageDataset(
            test_records,
            transforms["validation"],
            allow_truncated_images=dataset_config.allow_truncated_images,
        )
    else:
        # Test records are unlabeled, so no partition guard applies here.  Image
        # loading still matches the two-view path exactly, which keeps the
        # additional views the only difference from the frozen 61.67 protocol.
        test_dataset = PairedViewDataset(
            test_records,
            transforms["validation"],
            build_split_transforms(
                replace(dataset_config.transforms, resize_size=four_view_resize)
            )["validation"],
            required_partition=None,
            allow_truncated_images=dataset_config.allow_truncated_images,
        )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    predicted_indices: list[int] = []
    fused_logits_chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            resource_check(ROOT, policy)
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if four_view_resize is not None:
                    scaled = batch["scaled_image"].to(device, non_blocking=True)
                    logits = fixed_four_view_logits(model, images, scaled)[2]
                elif args.tta_horizontal_flip:
                    logits = _fixed_horizontal_flip_logits(model, images)
                else:
                    logits = model(images)
            if args.balanced_prior_uniform_soft:
                fused_logits_chunks.append(logits.detach().to(device="cpu", dtype=torch.float16))
            else:
                predicted_indices.extend(int(value) for value in logits.argmax(dim=1).cpu())
    balanced_prior_metadata = None
    if args.balanced_prior_uniform_soft:
        stored_logits = torch.cat(fused_logits_chunks)
        require(torch.isfinite(stored_logits).all().item(), 'nonfinite four-view logits')
        fused_logits = stored_logits.to(device=device, dtype=torch.float32)
        if fused_logits.shape != (len(filenames), len(mapping)):
            raise RuntimeError("test logits have an unexpected shape")
        if graph_route:
            from fixed_final_graph import apply_fixed_graph
            features=feature_collector.finish(len(filenames))
            graph_policy=read(ROOT/'artifacts/final_graph_budget.json')
            require(graph_policy['source_manifest_sha256']==args.source_sha256 and graph_policy['checkpoint_sha256']==checkpoint_sha256 and 0<graph_policy['total_wall_seconds']<=7200.,'explicit independent final graph budget required')
            graph_started=time.monotonic()
            graph_deadline=graph_started+min(graph_policy['total_wall_seconds'],max(0,policy['deadline_unix']-time.time()))
            try:
                prediction_array,transformed,graph_metadata=apply_fixed_graph(stored_logits.float().numpy(),features.numpy(),deadline=graph_deadline)
                predicted_indices=prediction_array.tolist();balanced_prior_fit=graph_metadata['p7']
                write_new(ROOT/'artifacts/final_graph_result.json',dict(complete=True,status='fixed_graph_complete',spent_wall_seconds=time.monotonic()-graph_started,total_wall_seconds=graph_policy['total_wall_seconds'],checkpoint_sha256=checkpoint_sha256,details=graph_metadata,confirmation_ledger_touched=False))
            except BaseException as error:
                write_new(ROOT/'artifacts/final_graph_result.json',dict(complete=False,status='fixed_graph_failed',spent_wall_seconds=time.monotonic()-graph_started,total_wall_seconds=graph_policy['total_wall_seconds'],reason=repr(error),confirmation_ledger_touched=False));raise
        else:
            bias,balanced_prior_fit=fit_uniform_soft_bias(fused_logits,temperature_multiplier=.5,iterations=60)
            adjusted_logits=fused_logits+.75*bias
            predicted_indices=[int(value) for value in adjusted_logits.argmax(dim=1).cpu()]
        counts = torch.bincount(
            torch.tensor(predicted_indices), minlength=len(mapping)
        )
        balanced_prior_metadata = {
            "schema": "aictrack1.uniform_soft_balanced_prior.v1",
            "temperature_multiplier": float(args.balanced_prior_temperature_multiplier),
            "bias_strength": float(args.balanced_prior_bias_strength),
            "fit": balanced_prior_fit,
            "prediction_count_min": int(counts.min().item()),
            "prediction_count_median": float(counts.float().median().item()),
            "prediction_count_max": int(counts.max().item()),
            "selection_evidence": "Stage2 train-derived balanced proxy only",
        }
    if len(predicted_indices) != len(filenames):
        raise RuntimeError("test inference output count does not match manifest")
    inference_seconds = time.perf_counter() - started
    predictions = dict(zip(filenames, predicted_indices, strict=True))
    command = subprocess.list2cmdline(sys.argv)
    package = build_submission_package(
        predictions,
        filenames,
        mapping,
        output,
        run_metadata={
            "run_id": str(experiment["run_id"]),
            "checkpoint_sha256": checkpoint_sha256,
            "config_hash": config_sha256,
            "data_hash": _sha256(test_manifest),
            "git_commit": git_commit,
            "inference_command": command,
            "official_clip_weight_sha256": clip_metadata.weight_sha256,
            "class_mapping_sha256": mapping.digest(),
            "test_manifest_sha256": _sha256(test_manifest),
            "test_record_count": len(test_records),
            "inference_seconds": inference_seconds,
            "device": str(device),
            "lora_folded_for_single_model_inference": lora_folded,
            "visual_input_resolution": visual_resolution,
            "position_interpolation": experiment.get("position_interpolation"),
            "four_view_resize": four_view_resize,
            "inference_view_protocol": (
                "native_center_plus_exact_horizontal_flip_equal_mean_logits_single_checkpoint"
                if args.tta_horizontal_flip
                else (
                    f"native{visual_resolution}_center_and_exact_flip_plus_resize_short_side"
                    f"{four_view_resize}_center_crop{visual_resolution}_and_exact_flip_"
                    "equal_four_logits_single_checkpoint"
                    if four_view_resize is not None
                    else "native_center_single_view"
                )
            ),
            "balanced_prior": balanced_prior_metadata,
            "balanced_prior_selection": balanced_selection_metadata,
            "inference_source_sha256": _sha256(Path(__file__)),
            "balanced_prior_source_sha256": _sha256(
                ROOT / "src" / "aictrack1" / "evaluation" / "balanced_prior.py"
            ),
            'full_training_source_sha256': FULL_SOURCE_SHA,
            'inference_source_manifest_sha256': args.source_sha256,
            'full_cpu_audit_sha256': args.cpu_audit_sha256,
            'last_checkpoint_sha256': frozen['last_sha256'],
            'metrics_sha256': frozen['metrics_sha256'],
            'classifier_head_source': frozen['classifier_head_source'],
            'trainable_tensors_restored': frozen['trainable_tensors'],
            'selected_route': frozen['selected_route'],
            'fixed_graph': graph_metadata,
            'inference_dataset_config_sha256': sha(dataset_path),
            'runtime': runtime_value,
            'test_used_for_training_or_parameter_updates': False,
            'frozen_postprocess': dict(temperature_multiplier=.5,bias_strength=.75,iterations=60),
        },
        spec=submission_spec,
    )
    resource_check(ROOT, policy, reserve=256*1024**2)
    # No labels are exported. These are the same observed forwards used above.
    artifacts = {}
    logits_path = output/'fused_logits.pt'
    with logits_path.open('xb') as handle:
        torch.save(dict(schema='aictrack1.full_final.test_fused_logits.v1', logits=stored_logits,
            sample_ids=[r['sample_id'] for r in official_test_rows], filenames=filenames,
            checkpoint_sha256=checkpoint_sha256, full_cpu_audit_sha256=args.cpu_audit_sha256,
            inference_source_manifest_sha256=args.source_sha256, test_manifest_sha256=TEST_SHA,
            class_mapping_sha256=mapping.digest(), view_protocol=VIEW, targets=None,
            model_parameter_updates=False, test_access=True), handle)
    artifacts['fused_logits.pt']=sha(logits_path)
    if feature_collector is not None:
        features=feature_collector.finish(len(filenames)) if features is None else features;feature_collector.close()
        path=output/'four_view_features.pt'
        with path.open('xb') as handle:
            torch.save(dict(schema='aictrack1.full_final.test_four_view_features.v1', features=features,
                sample_ids=[r['sample_id'] for r in official_test_rows], filenames=filenames,
                checkpoint_sha256=checkpoint_sha256, view_protocol=VIEW, feature_protocol=FEATURE_PROTOCOL,
                visual_forward_calls=feature_collector.calls, extra_forward_calls=0, targets=None,
                test_manifest_sha256=TEST_SHA, model_parameter_updates=False),handle)
        artifacts['four_view_features.pt']=sha(path)
    write_new(output/'inference_artifacts.json', dict(schema='aictrack1.full_final.inference_artifacts.v1',
        complete=True,files=artifacts,checkpoint_sha256=checkpoint_sha256,source_manifest_sha256=args.source_sha256,
        cpu_audit_sha256=args.cpu_audit_sha256,test_record_count=len(filenames),test_access=True,
        frozen_postprocess=('fixed k10/lambda.3 graph then P7' if graph_route else 'P7 t=.5 s=.75 max60'),selected_route=frozen['selected_route'],feature_export_enabled=(args.save_features or graph_route),extra_forward_calls=0))
    lease.release()
    atexit.unregister(lease.release)
    print(
        json.dumps(
            {
                "csv_path": str(package.csv_path),
                "zip_path": str(package.zip_path),
                "manifest_path": str(package.manifest_path),
                "record_count": package.record_count,
                "csv_sha256": package.csv_sha256,
                "zip_sha256": package.zip_sha256,
                "inference_seconds": inference_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
