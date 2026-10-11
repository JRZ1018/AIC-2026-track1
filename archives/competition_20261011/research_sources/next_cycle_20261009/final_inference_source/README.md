# Independent FULL joint-head final inference

This package is only an engineering preparation. No future FULL checkpoint has
been trained, selected, hashed or restored by the local preparation. No server,
GPU or actual test image was accessed. The root controls future execution.

Deploy the exact `source_manifest.json` whitelist to the previously nonexistent
`/root/AICtrack1_next_cycle_final_inference_20261009`. Exclude `test_fixtures`, the
permission-denied `tests/tmp*` fixtures, `__pycache__`, runtime `artifacts` and
`submissions`. Do not modify any existing677 training source or earlier package.
The84-file FULL source at `/root/AICtrack1_next_cycle_full_20261009` must remain
`677578ba379652d8b0277f8fe3055a2dba4dbe914f76f321ff6161f2c5838262`.

## Algorithm and precision

There is exactly one full_visual OpenAI CLIP ViT-B/32 model. Strict loading
restores152 visual tensors and `classifier.linear.weight` plus `.bias` from the
actual learned final checkpoint. H1100 is used only to construct/verify the
original head; it cannot replace the final learned head. All learned tensors
must be identical in actual `last.pt` and `final.pt`. Parameters are frozen only
after restoration. No optimizer or gradient update occurs on test images.

The core FV2 `fixed_four_view_logits`, balanced-prior and submission modules are
copied byte-for-byte from accepted677 modules. Four existing forwards use
native336 center/flip and resize384-center336/flip, equal float32 logit mean,
float16 CPU storage then float32 fixed P7 global adjustment. This preserves the
already accepted FV2 precision path. P7 is always temperature .5, strength .75,
max60 iterations, with the unchanged train-derived report SHA
`8122c9c96fb80823db9d19edb403d8d147d26a32c43b7b48fe7cca9db311d703`.
The engine has no test-based candidate search, TTA weight search, class labels,
hard quota, graph refinement or updated model state.

Optional `--save-features` only observes those same four visual forwards with the
existing tested hook. It preserves logits exactly, exports FP32 unit-normalized
mean unit embeddings, and makes zero extra forwards. It does not activate P3 or
any new postprocessing; a later validation-frozen P3 result needs its own root
admission. Both unlabeled logits and optional embeddings bind row IDs/filenames
and the actual checkpoint. Default output remains the frozen P7 submission.

## Admission and execution

Use the deployed Python environment; the launcher sets OMP4, MKL4 and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before any Torch import. Actual Torch threads
must equal4. CPU verification records actual interop threads without inventing
a training runtime receipt: unchanged677 `mainline_guard.entry` requires actual
Torch4, and the successful actual full launcher proves that entry passed.

Run a dependency check, which checks all real677 source hashes and imports the
real `train_n1_peft.py --help`, without loading a model or reading test data:

```sh
python -B scripts/audit_full_checkpoint.py --dependency-check
```

After real FULL training ends, root supplies the positively confirmed selection
receipt and successful actual `artifacts/queues/full_attempt_<uuid>.json`. The
audit verifies these actual files and all underlying fresh confirmation
evidence. It derives the checkpoint hashes from real completed `metrics.json`,
`last.pt` and `final.pt`; no guessed checkpoint hash is admissible.

```sh
python -B scripts/audit_full_checkpoint.py --arm J5 \
  --source-sha256 "$INFERENCE_SOURCE_SHA" \
  --selection /actual/full_selection_receipt.json \
  --selection-sha256 "$ACTUAL_SELECTION_SHA" \
  --launch-receipt /root/AICtrack1_next_cycle_full_20261009/artifacts/queues/full_attempt_ACTUAL.json \
  --output artifacts/full_final_cpu_audit.json
```

Use J20 only when it is the actual positively confirmed chosen arm. The CPU
audit restores the official backbone, joint154 parameters, all AdamW moments,
two original LR groups, cosine20 scheduler, scaler, AC20 for all148695 global
manifest indices and their labels, and CPU RNG states. It makes one synthesized
336 image forward and reads no actual image bytes. CUDA RNG remains serialized.
Both original full H1 and its positive actual row-binding receipt are verified.
Expected memory is approximately3GiB; root must check actual host RAM. A missing
or incomplete FULL checkpoint stops admission before test access.

The launcher below uses the actual audit SHA and a genuinely verified source
Git commit if available (per-file source SHA remains authoritative). Retain the
same system-disk cumulative initial-free baseline as the successful FULL run;
do not reset the budget between training and inference. Root supplies its
actual future deadline. Default disk budget20GiB, reserve3GiB and available RAM
floor2GiB can be tightened for actual capacity. No new model copies are made on
the server; inference outputs need roughly140MiB with features, under256MiB
including logs, manifests and atomic CSV/ZIP files.

```sh
python -B scripts/run_final.py \
  --source-sha256 "$INFERENCE_SOURCE_SHA" \
  --cpu-audit artifacts/full_final_cpu_audit.json \
  --cpu-audit-sha256 "$ACTUAL_AUDIT_SHA" \
  --source-git-commit "$VERIFIED_GIT_COMMIT" \
  --deadline-unix "$ACTUAL_DEADLINE" \
  --initial-free-bytes "$ORIGINAL_FULL_SYSTEM_DISK_BASELINE" \
  --save-features
```

The child acquires the existing canonical lease
`/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json` before any test
manifest/image read, guards disk/RAM/deadline per batch and verifies its source,
audit and asset hashes again. The controller guards the owned process identity,
lease token and resources every2seconds. It kills only its own process group on
failure; it never clears a retained shared lock or retries automatically. Paths
are exclusive/no-overwrite. Never run two copies or a queue trainer concurrently.

The independent dataset overlay retains the original transforms, class mapping,
counts and all other fields; it explicitly opens the pinned official directory
`/root/autodl-tmp/AICtrack1/DataSetStage2/test`. The full training config stays
byte-identical and test-unavailable. Actual test metadata is pinned SHA
`576ff0f69e0a104ca5a0af7f190628074b572102c6bc8daf8cb653fb08054201`,
37,444 unlabeled flat JPEGs. No image copies or new data symlinks are required.

## Outputs and independent local acceptance

`submissions/final/pred_results.csv` has37,444 rows, no header, official filename
order and original4-digit official IDs. `pred_results.zip` has only that CSV at
its root, with decompressed bytes exactly equal to the CSV. The submission
manifest binds actual model/config/test/mapping/P7/source/runtime hashes. The
source also saves the observed FP16 fused logits and optional FP32 features.

Only after these validations succeed, `artifacts/return_manifest.json` provides
an exact `files` mapping: each key is the local download-relative path; each
value has the real absolute remote path, SHA and byte size. It contains full
final/last/metrics/config/selection/launcher/H1 recovery assets, official CLIP,
class mapping and train/split/test metadata, inference freeze/CPU receipts,
submission/logits/features and both exact source inventories. No images are
included. Fetch the return manifest itself and `artifacts/controller_result.json`
separately and verify the manifest SHA recorded by the successful controller.
Reuse previously verified same-SHA CLIP/H1 assets locally where possible; last
is roughly1GiB and final roughly337MiB, so download storage must be checked.

Root's downloader places files according to the manifest keys beneath one new
local return directory, refusing overwrite and checking each SHA. Then use:

```powershell
python -B scripts/accept_return.py --return-root 'D:\actual_new_return' `
  --manifest 'D:\actual_new_return\return_manifest.json' `
  --expected-manifest-sha256 ACTUAL_SHA --cpu-model-recovery `
  --output 'D:\actual_new_return\local_acceptance.json'
```

This verifies all downloaded bytes and both exact source inventories, rechecks
CSV/ZIP and the real model identity, reconstructs one CPU model with the complete
joint154 head and optimizer/scheduler/scaler/AC state and makes one synthesized
image forward. It never executes the old hardcoded delivery controller. A
submission is ready for user upload only after this actual local acceptance.

## Local engineering evidence

`tests/test_final_contract.py` has13 CPU tests: learned head overridesH1;
partial/mismatched head rejected; final/last corruption rejected; partial19,
metrics hash drift, validation selection or old152 rejected; real AdamW groups,
cosine20 and missing moments; four existing calls with/without featurehook give
bitwise identical logits; fixed P7 is deterministic; labeled/unsafe/duplicate
test metadata rejected; dataset override cannot change transforms; exact CSV
order, labels, root ZIP and no-overwrite; downloaded asset corruption; actual
inference/audit/controller/local-acceptance entry imports and CLI help.

`CPU_VERIFICATION.json` records the actual run and real677 dependency check. This
is synthetic engineering validation, not an actual future FULL recovery or
37,444-image GPU test inference. Runtime full checkpoint and server resource
acceptance remain explicitly outstanding.
