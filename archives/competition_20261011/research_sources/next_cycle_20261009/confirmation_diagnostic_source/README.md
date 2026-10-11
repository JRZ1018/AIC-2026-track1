# Independent fresh-confirmation diagnostic

This is a separate, CPU-tested adapter for new split `57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7`. The old diagnostic source `7bb94f9192a6c02ce0928dd2e94aecb879c661f66097bb66ed625224c7d364f6` stays unchanged. This directory was prepared locally without server access, GPU access, training, image loading, or test access.

The real filenames are **README.md** and **DEPLOYMENT_CONTRACT.json**. There is no DEPLOYMENT.md.

## Inputs and strict scope

The confirmation bootstrap is `/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009`. After its actual train-only H1100 completes, the already reviewed bootstrap builder writes a new independent visual source at `prepared_visual`. The required training source manifest and H1 SHA are actual outputs of that step. They are deliberately not replaced by old `2fee`, old `45266...H1`, or placeholders.

The diagnostic accepts only that source's new C and **one previously selected J5 or J20**. All training/source files, both configs, the source selection receipt, assets, H1 receipt, and exact connected row binding are checked. C restores 152 visual tensors / 87,849,216 scalars and its head must equal the fresh fold H1. J restores 154 tensors / 88,233,966 scalars including the jointly trained head. Both require complete fixed-last20 metadata, optimizer moment shapes/finite values, scheduler epoch20, AMP scaler, actual AC restoration with exact global train indices/labels, full coverage and finite AC tensors. No best checkpoint or full-trained holdout model is accepted.

The original FV2 four-view engine and strict single-model loader are verbatim references. Four FP32 logits are averaged at native336, exact flip, resize384/crop336, exact flip; the engine emits FP16 fused logits. FP16-to-FP32 decode cannot reconstruct discarded precision. Original FP32 argmax is retained in rows; the paired report counts differences. Each arm uses its actual training runtime receipt: OMP/MKL/torch4, CUBLAS `:4096:8`, actual interop count, source/config/split/H1 SHA. C and J actual runtime objects must match.

## Queries and decision rule

The support-only queries were frozen before fresh C/J predictions existed, using the actual new `confirmation_rows.csv`. The natural query is all 29,737 unique images / 750 classes / 27,686 connected groups. The balanced query is 14,760 unique images and groups / 738 sufficiently supported classes, 20 groups per class. The omitted 12 classes are reported. Selection uses seed20261009, targets and connected groups only. All old query connected groups were excluded by the upstream split builder; this source binds its exact fresh row SHA. Query class membership can differ from the old fold while following the same independent-support rule.

P7 stays temperature0.5 / strength0.75 / 60 iterations. Each protocol independently computes its prediction-matrix bias; there is no scan. The preselected J confirms direction when its fresh balanced P7 primary delta is positive, provenance and runtime match, and root reviews the complete diagnostics. Natural750, per-class support/recalls, head/mid/tail, supported declines and high-margin flips remain visible. Neither natural>=0 nor bootstrap lower bound>0 is a new hard gate. A CI crossing zero does not establish a stable gain. No low-support class alone vetoes. No result in this prepared source authorizes or freezes formal test inference.

## Root deployment and commands

Upload exactly `source_manifest.json` plus every file it inventories, preserving bytes, into `/root/autodl-tmp/AICtrack1_next_cycle_confirm_diagnostic_20261009`. Do not upload synthetic `tests/fixtures`, test binding, pycache or artifacts. Verify every source SHA and the manifest's raw SHA, then run the CPU dependency check against the **actual newly prepared** visual source. This preflight imports real modules and validates public signatures plus the actual vendor production argv and subprocess `--help`. It does not open a checkpoint, model, images or remote assets. The bootstrap runtime source manifest SHA `448a2b2a293cb3a7a51f7bd9b5fad7afde9f2cae196bc951485dedfa98511d2c` is a current upstream receipt, not the future prepared_visual SHA.

```bash
DIAG=/root/autodl-tmp/AICtrack1_next_cycle_confirm_diagnostic_20261009
TRAIN=/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009/prepared_visual
# These are verified actual runtime values from root's receipt, not literal placeholder arguments.
python "$DIAG/run_confirmation_diagnostic.py" --training-root "$TRAIN" \
  --expected-training-source-sha256 "$ACTUAL_PREPARED_SOURCE_SHA" --dependency-check
```

After H1, C and chosen J have finished, root audits their epoch20 SHA and runtime ledger. Only after all trainers exit and the shared lease is empty, invoke C then chosen J serially. The wrapper establishes OMP/MKL4/CUBLAS before importing torch and uses actual torch4. The original engine acquires the canonical shared GPU lease `/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json`; a lock or running trainer defers the operation. Each invocation needs a declared phase deadline, 1,800-second default ceiling (range60..3600), batch64, workers0/4, at least2GiB disk and4GiB effective host/cgroup availability. A named retry1 is the only alternate output; outputs are never overwritten. A phase's residual time reserves120 seconds. Root must keep the total diagnostic phase/retry within its authorized ceiling.

```bash
python "$DIAG/run_confirmation_diagnostic.py" --training-root "$TRAIN" \
  --expected-training-source-sha256 "$ACTUAL_PREPARED_SOURCE_SHA" \
  --arm C --expected-last-sha256 "$ACTUAL_C_LAST20_SHA" \
  --deadline-unix "$PHASE_DEADLINE_UNIX" --hard-seconds 1800 --workers 4 --save-features --run
python "$DIAG/run_confirmation_diagnostic.py" --training-root "$TRAIN" \
  --expected-training-source-sha256 "$ACTUAL_PREPARED_SOURCE_SHA" \
  --arm "$PRESELECTED_J_ARM" --expected-last-sha256 "$ACTUAL_J_LAST20_SHA" \
  --deadline-unix "$PHASE_DEADLINE_UNIX" --hard-seconds 1800 --workers 4 --save-features --run
python "$DIAG/compare_confirmation.py" \
  --control-bundle "$DIAG/artifacts/diagnostics/P1_C/fused_logits.pt" \
  --control-identity "$DIAG/artifacts/diagnostics/P1_C/input_identity.json" \
  --candidate-bundle "$DIAG/artifacts/diagnostics/P1_${PRESELECTED_J_ARM}/fused_logits.pt" \
  --candidate-identity "$DIAG/artifacts/diagnostics/P1_${PRESELECTED_J_ARM}/input_identity.json" \
  --output "$DIAG/artifacts/paired_confirmation" --threads 4
```

`--save-features` is optional and defaults off. It attaches a visual forward hook to the already required four B-sized calls; it never adds a forward, model or optimizer. It saves FP32 `[29737,512]` normalized mean of four individually normalized raw embeddings, exact sample IDs/order/targets/checkpoint/config/split/view identity, and zero additional forwards. `feature_file` and SHA join input identity and result manifest. CPU fixtures verify both the actual four_calls_B order and equivalent two_calls_2B first-B original/last-B flip layout, two-scale order, sample order, exact logits equality and zero extra forward count.

## Outputs and integration contract

Each arm writes `fused_logits.pt`, original `paired_predictions.csv`, `rows.csv`, `runtime_receipt.json`, `p1_checkpoint_audit.json`, `input_identity.json`, `result_manifest.json`, and optional `fused_features.pt` under `artifacts/diagnostics/P1_C|P1_J5|P1_J20` (retry adds `_retry1`). Identity stays `aictrack1.p0_frozen_fold_input.v1`, adds `model_arm`, fresh H1/source/selection/runtime bindings, and positive holdout provenance.

The audit schema is `aictrack1.confirmation.fixed_last_diagnostic_audit.v1`: `complete=true`, `status=complete_fixed_last20_audit`, `checkpoint_role=last`, `checkpoint_epoch=epoch=20`, identical `fixed_last_sha256` and `checkpoint_sha256`, `config_sha256`, `split_sha256`, `h1_sha256`, source SHAs, `model_arm`, `trainable_tensors`, identical `trainable_numel`/`numel`, head source `fresh_fold_h1_frozen` or `joint_fixed_last_state`, exact classifier restoration, full AC binding and zero inference optimizer steps.

Paired `report.json` schema is `aictrack1.confirmation.fixed_last_paired.v1`; `p1_arm=selected_arm`, `complete=true`, `test_access=false`, `used_test_or_online_scores=false`, fresh split/H1/training/diagnostic/protocol/postprocess SHAs. Stable decision keys are `eligibility.primary_delta_pp`, `eligibility.natural_p7_delta_pp`, `eligibility.group_ci95_pp`, `eligibility.ci_crosses_zero`; each named protocol has `p7.candidate_vs_control.delta_vs_baseline_pp` and `group_bootstrap.ci95_pp`. Candidate/control direct provenance aliases and their nested `.provenance` are identical: `identity_sha256`, `bundle_sha256`, `checkpoint_sha256`, `checkpoint_audit_sha256`. The receipt writer/full gate must reject conflicting aliases. The full confirmation receipt is produced by the separately frozen acceptance writer after root risk review; this report does not independently approve a full training or test step.

## Local validation and remaining runtime work

11 CPU tests cover actual new-source generation and real entry imports/CLI, C/J strict loader behavior, 152/154 policies, moment/scheduler checks, AC restoration, oldsplit/best/H1 rejection, exact fresh queries, runtime/row pair mismatches, deterministic P7 selfcomparison, and feature-hook parity. The runtime integration uses the actual upstream visual generator with synthetic asset locations; it proves interfaces, not a trained H1 or actual model result. Actual fresh H1, prepared source, C/J checkpoints and remote source entry check remain runtime work for root. The audit does not claim GPU inference has run.
