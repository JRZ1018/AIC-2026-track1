This private source prepares S-on-C152 fixed-last20 diagnostics and the fixed P7 paired report. It never trains, selects a checkpoint, searches a decoder, reads test images, connects to a server or submits results. Root alone deploys and explicitly runs the production entry.

The training question is SCE alpha 0.1 to 0.5 on the accepted C recipe. The original legal fold H1 remains frozen, all 152 visual tensors remain trainable during training, and diagnostics restore that exact visual state on CPU. The training/source/config identities are pinned in this package before deployment.

`reference/fv2_single_model.py`, `reference/diagnose_stage2_tta.py`, `reference/p0_core.py`, `diagnostic_resources.py` and `view_features.py` are byte-identical copies from frozen diagnostic 7bb. New wrappers and contracts are independent. The raw engine acquires the existing canonical TrainingLease; this package does not recreate, remove or bypass the lock.

Production paths:

- training: `/root/autodl-tmp/AICtrack1_next_cycle_supervision_20261009`
- diagnostic: `/root/autodl-tmp/AICtrack1_next_cycle_supervision_diagnostic_20261009`
- checkpoint: training `artifacts/experiments/P1_S/last.pt`
- diagnostic output: diagnostic `artifacts/diagnostics/P1_S`
- paired output: diagnostic `artifacts/paired/P1_S`
- canonical lease: `/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json`

No output directory is overwritten. A partial attempt is preserved. The independent runner permits only a separately named `retry1` attempt; that attempt's CPU comparison needs an explicit new output path. The combined entry always uses the primary attempt and stops if it fails.

CPU integration precheck (opens source/config declarations, imports actual training APIs and parses the actual vendor CLI; does not load checkpoints, touch images, enumerate CUDA or acquire a lease):

```bash
/root/autodl-tmp/venvs/aictrack1/bin/python /root/autodl-tmp/AICtrack1_next_cycle_supervision_diagnostic_20261009/diagnose_and_pair.py \
  --training-root /root/autodl-tmp/AICtrack1_next_cycle_supervision_20261009 \
  --dependency-check
```

Root must verify the actual frozen source manifests, full20 terminal metrics, actual last SHA, completion of the training controller, canonical idle and resource headroom before invoking the run. The command receives the existing phase deadline; it does not create or reset a deadline. Set the launch environment to OMP4/MKL4/CUBLAS4096:8. The S trainer also records real source/config/runtime receipts, which diagnostics require and bind by SHA.

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
/root/autodl-tmp/venvs/aictrack1/bin/python /root/autodl-tmp/AICtrack1_next_cycle_supervision_diagnostic_20261009/diagnose_and_pair.py \
  --training-root /root/autodl-tmp/AICtrack1_next_cycle_supervision_20261009 \
  --expected-last-sha256 ACTUAL_ROOT_VERIFIED_S_LAST20_SHA \
  --deadline-unix EXISTING_ROOT_PHASE_DEADLINE \
  --hard-seconds 1800 --workers 4 --min-free-gib 3 --save-features \
  --control-bundle ACTUAL_EXISTING_C4813_CACHE_ABSOLUTE_PATH \
  --control-identity /root/autodl-tmp/AICtrack1_next_cycle_supervision_diagnostic_20261009/inputs/control/fv2_input_identity.json \
  --run
```

The SHA, deadline and C bundle placeholders must come from actual root evidence. No remote C bundle path is guessed. The included portable C identity changes only the row CSV and provenance-audit path to relative byte-identical copies. Its origin identity is 87a74d13c61cc006867b15fa5ee2b0aff47ed9f9a1cae0f096e2dba1d191e5e9. Checkpoint078b, cache4813, config5876, H145266, split0c902, manifest1e912 and mapping811b remain unchanged. Historical Windows provenance paths remain observations; decoding opens only the explicit bundle plus the relative row/audit files. No old best or control-last8 cache is accepted.

The diagnostic defaults to 1800 seconds and permits a root-specified ceiling of 60–3600. The raw engine has a wall alarm and batch-boundary checks; the combined entry requires enough existing phase time for the full diagnostic plus120 seconds and separately bounds CPU pairing by the remaining deadline. Root must retain normal child/lease supervision when launching it detached. No idle wait, detached controller or autonomous phase advancement is implemented here. The independent memory helper requires effective host/cgroup available memory of at least4GiB; the free-disk floor cannot be reduced below3GiB. There are zero inference optimizer steps.

The CPU model audit positively requires S source metadata, the full20 matching metrics/history, original strong-view AC, all152 visual names/87849216 parameters, one visual-only optimizer group with finite complete moments, cosine last_epoch/T_max20, positive scaler, and full AC restoration with exact official train indices, labels, hyperparameters and seen coverage. It compares restored visual tensors and the unchanged H1 head bit-for-bit, then asserts frozen eval before moving that single model to the requested inference device. A previous joint154 audit cannot satisfy this contract.

The four views remain native336, exact horizontal flip, resize-short-side384/center-crop336 and exact flip; their FP32 logits are equally averaged. The exported FP16 cache retains exact validation IDs/order/targets/class counts. Optional `--save-features` attaches a readonly visual forward hook to these existing four calls. Features use FP32 `normalize(mean(normalize(each raw visual embedding)))`, have the same order/checkpoint/config/split and incur zero extra forwards. Hook fixtures cover four-B and two-2B layouts, order, split and strict logits parity. Production explicitly uses four-B.

Expected primary diagnostic inventory: `supervision_checkpoint_audit.json`, `report.json`, `paired_predictions.csv`, `fused_logits.pt`, `rows.csv`, `input_identity.json`, and `result_manifest.json`. With features, `fused_features.pt` is added to identity and result inventory. Audit schema is `aictrack1.supervision.fixed_last_diagnostic_audit.v1`; identity remains `aictrack1.p0_frozen_fold_input.v1` with model_arm/p1_arm S, visual152 fields and `classifier_head_source=old_fold_h1_frozen`. Feature schema remains `aictrack1.stage2_train_derived_4v_features.v1`. Raw cache schema remains `aictrack1.stage2_train_derived_fused_logits.v1`.

The CPU pair report schema is `aictrack1.supervision.fixed_last_paired.v1`. The relevant fields are `candidate.identity_sha256`, `control.identity_sha256`, `diagnostic_source_manifest_sha256`, `training_source_manifest_sha256`, `selection_protocol_sha256`, `supervision_registration_sha256`, `protocols` and `eligibility`. Both protocols retain the original frozen support-only selection: all29737 natural validation IDs once; balanced737 classes×20 independent groups =14740, with13 omissions explicitly reported. This is the old exploration fold, not fresh confirmation. Each protocol computes its own P7 adjustment on its own query matrix. No decoder grid is opened.

The byte-identical b60 `selection_protocol.json` originally lists J5/J20. It supplies the unchanged baseline query/decoder specification; its old ranking does not select S. `supervision_registration.json` independently records the root-selected single S axis and unchanged continuation thresholds. `eligibility.primary_delta_pp` is balanced P7 observed-label Top1 gain; >=.20pp is exploration continuation, >=.10pp requires fresh positive matched confirmation, below .10pp does not meet the registered rule. Natural750 agreement, macro, group-bootstrap500/seed20261009 CI, head/mid/tail, per-class support, large supported declines and high-margin reversals are fully reported. Neither natural>=0 nor positive CI lower bound is added as a hard gate. One/two-image class flips alone do not veto. No stable leaderboard or71-point claim is made by these outputs.

To complete pairing locally after a verified cache return, use the same compare entry with explicit bundle/identity paths and a new output directory:

```powershell
& 'D:\aicomp\AICtrack1\.venv\Scripts\python.exe' 'D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\supervision_diagnostic_source\compare_supervision.py' --control-bundle 'D:\aicomp\stage2-server-transfer\fv2_results_20261008\diagnostic\artifacts\diagnostics\fullvisual\fused_logits.pt' --control-identity 'D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\supervision_diagnostic_source\inputs\control\fv2_input_identity.json' --candidate-bundle 'RETURNED_S\fused_logits.pt' --candidate-identity 'RETURNED_S\input_identity.json' --output 'NEW_OUTPUT' --threads 4
```

CPU tests use small synthetic tensors; the real dependency test imports the actual frozen supervision package and runs the actual vendor `--help` subprocess. They do not load official weights, images or the future S checkpoint. Local command:

```powershell
& 'D:\aicomp\AICtrack1\.venv\Scripts\python.exe' 'D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\supervision_diagnostic_source\tests\test_supervision_diagnostic.py'
```

Production restoration, runtime receipts, source paths, inference, finite real outputs and remote dependencies remain pending until root runs them. CPU engineering acceptance is not a server/GPU result.
