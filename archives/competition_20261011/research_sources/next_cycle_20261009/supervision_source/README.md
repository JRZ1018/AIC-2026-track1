# P1 S-on-C: one supervised-loss experiment

This independent text source implements docs/240 S: SCE alpha **0.1 -> 0.5**,
beta=1 and log_zero=-4 unchanged. It uses full visual152/87,849,216 trainable
parameters and the original fold H1 frozen classifier. It starts fresh from
official OpenAI CLIP and that train-only H1; it never continues C/J final20.
No checkpoint, image, feature shard or credentials are included.

The canonical entry is `scripts/run_supervision.py`, not an invented trainer.
It calls the existing `run_serial_queue` and existing canonical lease at
`/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json`. The actual trainer
remains `scripts/train_n1_peft.py`, recognized by the process probe. Root alone
deploys or invokes CUDA. All actual trainer children receive OMP4/MKL4 and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python/Torch starts; the trainer verifies
actual Torch threads4 and writes `artifacts/runtime_receipts/S_<attempt>.json`.
Literal deployment target is `/root/autodl-tmp/AICtrack1_next_cycle_supervision_20261009`;
comparison resolves both target and actual root to support the mount symlink.

## Exact assets and control

`inputs/p1_assets.json` binds the actual official train manifest, official CLIP,
old split0c902..., old train-only fixed100 H145266..., H1 receipt314a..., class
mapping and label-conflicts. Launch hashes every actual remote input. The H1
loader additionally checks normalized linear head, epoch100, positive118958/29737
row binding and no validation optimization/selection. Full-train H1 is refused.
`inputs/control_semantic_equivalence.json` records actual FV2 control source/config,
checkpoint078b... fixed-last20, and the common policy comparison. C's historical
best export is never substituted for its fixed-last20. P1 parent_C JSON is a
contract template; it is not represented as a completed new training run.

S and real old C share officialCLIP/H1/split/seed20261008,336/b64/crop.8/hflip.5,
strong-view AC(.8/.5/.02/.75), visualLR1.75e-6, headLR0, WD.01, clipnorm1,
CosineAnnealingLR T_max20 and one scheduler step per epoch. The original controller
explicitly sets OMP4. A complete original C actual-MKL/CUBLAS runtime ledger was
not found; this old-fold comparison remains exploratory. Any future C/S fresh
confirmation must have matched actual runtime ledgers. Adding `fixed_final` only
changes the new selected export; evaluation uses last20 on both sides.

## Root-only launch and resume

After the current H1 or other GPU phase has completely closed and the canonical
lock is free, root binds an actual S phase start and deadline (at most six hours).
Use the CUDA Python `/root/autodl-tmp/venvs/aictrack1/bin/python` from the exact
deployment root:

```
python scripts/run_supervision.py --smoke --phase-start-unix ACTUAL_START --deadline-unix ORIGINAL_S_DEADLINE
python scripts/run_supervision.py --phase-start-unix ACTUAL_START --deadline-unix ORIGINAL_S_DEADLINE
python scripts/run_supervision.py --resume --phase-start-unix ACTUAL_START --deadline-unix ORIGINAL_S_DEADLINE
```

`ACTUAL_START` and `ORIGINAL_S_DEADLINE` must be real numbers, not these placeholders.
The source itself contains no pre-expired deadline or automatic GPU launch.
The first launch freezes these values in `artifacts/queues/supervision_budget.json`;
resume must preserve them. Original data-disk free baseline is **13013950464**,
cumulative growth max **6GiB**, minimum free **3GiB** including atomic-save reserve.
The atomic reserve counts deduplicated tensor storage plus64MiB while the old last
exists. This budget includes cumulative other phase growth on the same data disk;
it is not reset to current stat. Do not delete prior evidence to make room.

CUDA smoke is synthetic2 images, follows alpha.5 and actual legal CLIP/H1; it
checks152 visual tensors, frozen head, gradients, normal AMP overflow backoff and
a real disk optimizer/scheduler/scaler/RNG/model next-step resume within existing
tolerances. Its receipt is engineering evidence, not candidate accuracy. Every
smoke has a separate UUID directory and no fake training completion.

Formal output is `artifacts/experiments/P1_S`. It uses original20 cosine. Optional
`--stop-after-epoch 1..19` is an explicit pause, never an8-epoch schedule or final
candidate. Resume requires this exact unchanged config and this output's last.pt,
with source SHA, S arm and frozen-H1 metadata verified before restoring the real
model/AdamW/scheduler/scaler/AC/RNG. Frozen head equality and152 exact count are
asserted before training, after resume and every epoch before checkpoint save.
Validation is observed-label agreement on the legal heldout fold. Early AC-L2 or
resource failure preserves partial evidence and does not export a completed model.

## Gradient evidence and downstream diagnosis

Every epoch retains original preclip norm mean/max/clip fraction/nonfinite count,
plus finite coefficients `min(1,max_norm/(norm+1e-6))` min/p10/p25/p50/p75/p90/max.
Nonfinite norms are excluded from coefficient quantiles and separately counted.
Actual GradScaler scale decrease is recorded as an AMP skipped optimizer update;
successful updates are counted separately. This is observation only: no changes
to forward, loss/AC, gradient clipping, step or original once-per-epoch scheduler.
Public scale reads add small synchronization overhead, which will be reported.

Each real `last.pt` adds `supervision_source_manifest_sha256`, `supervision_arm=S`,
`classifier_head_source=original_fold_H1_frozen`, `classifier_head_restored_equal=true`.
The trainable state contains152 FP32 visual tensors and **no classifier keys**;
head recovery must reload H145266... then strictly restore visual last20. Metrics
retain actual count/base_loss/head_initializer/config/split/last SHA and full
epoch history. This source supplies training only. A separate S152 diagnostic
uses the FV2 generic four-view engine, fixed legal b60... unique queries and P7
(.5/.75/60); the frozen7bb joint154 guard is not relaxed. No test inference/full
training/selection receipt is authorized by this source's mere completion.

## Local CPU verification

```
D:\aicomp\python312\python.exe tests/test_cpu_supervision.py
```

CPU fixtures test actual SCE/weighted-loss gradients, model/head/AdamW/Cosine20/
scaler/AC/RNG disk resume, real queue with injected CPU launcher, frozen-head152
meta shape, clip coefficient/nonfinite semantics, AMP observation, unchanged
epoch scheduler, budget continuity, exact alpha-only config, source binding and
real CLI parsing. They do not run CLIP forward, CUDA, images or server operations.
`cpu_smoke_result.json` is written by the actual run and excluded from immutable
source members to avoid self-referential manifest hashes. Root must still run
actual remote asset checks and CUDA recovery smoke before formal20.
