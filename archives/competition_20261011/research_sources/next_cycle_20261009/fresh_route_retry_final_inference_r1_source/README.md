# Expanded-capacity typed FULL final inference and CPU acceptance

This independent revision preserves the reviewed b517 final implementation and
its two-sided resolved-root guards. It binds the actual expanded-capacity FULL
producer template 578eb15ce02421e16e3476b063c406fdf0322d8cc63233e740c29d9f51ab4d71.
The original b517 manifest and d0c7 FULL manifest are retained under references;
those prior source directories are unchanged.

This is an unbound engineering template. No actual fresh-positive receipt,
completed typed FULL checkpoint, server model recovery or test inference has
been performed here. It cannot launch from its template manifest. Root owns
future preparation and execution. Existing27e,677,05df,b049 and554d are unchanged.

One confirmed route is allowed. `S` restores all152 FP32 visual tensors from the
fixed FULL20 checkpoint and preserves the legal full H1100 head
fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605.
`J5_GRAPH` restores all154 tensors, including the complete learned classifier
weight and bias. Strict restoration, optimizer152/154, cosine20, scaler, AC20
for all148695 train IDs/labels and CPU RNG are checked before inference. Both
last20 and final20 must contain exactly equal learned tensors; best is refused.

Both routes use unchanged FV2 native336/flip plus384-center336/flip, four FP32
logits equally averaged, FP16 CPU cache storage then FP32 decoder input. S uses
fixed P7 temperature.5/strength.75/max60. J5_GRAPH observes the same four visual
calls, exports normalized mean unit features in FP32 with zero extra forwards,
then executes only the preselected reciprocal positive cosine k10/lambda.3,
T1/three original-P residual rounds/ln4 and zero-degree exact-logit protection,
followed once by P7. The copied scientific graph core is exactly5748; no grid,
labels, extra anchors, model updates or test-based selection exists.

The graph decoder uses CPU2, matching its validation protocol. An independent
final graph budget is explicitly recorded (at most7200s and the root's final
deadline) and charged in a success/failure receipt. It never opens or resets
the original graph confirmation ledger. Raw-margin protection preserves raw
scores; global P7 can still alter a protected row's final prediction.

## Actual preparation and CPU recovery

The actual FULL producer must be the independent typed full source at
`/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_full_20261009`, with one config
`p1_full_S.yaml` or `p1_full_J5.yaml` and output `FULL_S` or `FULL_J5`.
Its actual contract verifies the typed positive fresh confirmation and all
root review/evidence hashes. Windows observation paths cannot stand in for
actual server evidence. No future checkpoint digest is prefilled.
The reviewed producer template is pinned by the exact reference manifest
`references/UPSTREAM_FULL_TEMPLATE.json` (578eb15ce02421e16e3476b063c406fdf0322d8cc63233e740c29d9f51ab4d71).
A changed producer template requires an explicit new inference revision.

After FULL training actually completes, root binds a new final source:
Stage this unbound template into the separate literal staging root
`/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_retry_final_template_r1_20261009`
using `template_manifest.json`. Keep the final execution root absent, then run
the following binder from the template staging root; preparation creates the
new execution root and copies only the frozen files.

```sh
python -B scripts/prepare_final_source.py --template-sha256 ACTUAL_TEMPLATE_SHA \
  --full-source-sha256 ACTUAL_FULL_SOURCE_SHA \
  --confirmation /actual/fresh_confirmation.json --confirmation-sha256 ACTUAL_SHA \
  --launch-receipt /actual/full_attempt.json --launch-receipt-sha256 ACTUAL_SHA
```

Preparation hashes real metrics,last,final,config and the successful non-smoke
launcher. It creates the previously nonexistent literal root
`/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_final_20261009` and a bound
source manifest. This structural binding still requires actual CPU recovery:

```sh
python -B scripts/audit_full_checkpoint.py --arm S --source-sha256 ACTUAL_BOUND_SHA \
  --selection /actual/fresh_confirmation.json --selection-sha256 ACTUAL_SHA \
  --launch-receipt /actual/full_attempt.json --output artifacts/full_final_cpu_audit.json
```

Use J5 only for the confirmed graph route. CPU recovery hides CUDA and makes
one synthesized336 forward, opening no image bytes. Source, official CLIP,
mapping, legal full H1, all learned tensors and training runtime receipts are
verified. Actual OMP4/MKL4/CUBLAS=:4096:8 and torch4 are required. Available RAM
must cover the approximately3GiB recovery; synthetic fixtures do not prove
actual full-model recovery or GPU throughput.

## One root-operated official test pass

```sh
python -B scripts/run_final.py --source-sha256 ACTUAL_BOUND_SHA \
  --cpu-audit artifacts/full_final_cpu_audit.json --cpu-audit-sha256 ACTUAL_SHA \
  --source-git-commit ACTUAL_VERIFIED_COMMIT --deadline-unix ACTUAL_FINAL_DEADLINE \
  --initial-free-bytes ORIGINAL_CUMULATIVE_DATA_FREE_BASELINE \
  --disk-max-bytes ROOT_AUTHORIZED_CUMULATIVE_DISK_MAX \
  --final-graph-budget-seconds 7200
```

The initial-free CLI remains the original full-run baseline 13013950464 bytes.
The controller copies only the actual FULL budget's bound capacity fields and
receipt ba819e4b72731e810743deb1b4b4849eb070e8c664187aeaf255ab3840a78e52
into an exclusive final policy, whose phase starts at the actual final launch.
The one additive 32212254720-byte expansion gives adjusted baseline45226205184;
usage is adjusted baseline minus current free, plus any explicit atomic reserve.
Disk total must equal128849018880 bytes exactly. Further capacity changes require
new root evidence; observed current free never becomes a replacement baseline.
Original shared/failure lineage is verified by the immutable recovery receipt,
which is also included in the return inventory. Root supplies a concrete final
deadline/disk/RAM policy. FULL's training deadline is not reused as the final
inference deadline; read-only actual FULL acceptance still checks its completed
provenance rather than trying to re-admit its expired training phase. The existing canonical lease is held during the whole
single inference/package operation. Output is exclusive, no retries or lock
deletion occur. Only37,444 official unlabeled flat JPEGs at
`/root/autodl-tmp/AICtrack1/DataSetStage2/test` are used, with pinned manifest
576ff0f69e0a104ca5a0af7f190628074b572102c6bc8daf8cb653fb08054201
and exact official750 mapping. Overlay changes only the official test root;
training transforms, source config and classes remain byte bound.

CSV has no header, preserves official manifest filename order and four-digit
official IDs. ZIP contains only root `pred_results.csv`, byte equal to CSV.
Actual checkpoint/source/config/mapping/test/P7/graph/runtime hashes are retained.
Graph features are automatically captured for J5; optional S features grant no
permission to apply graph postprocessing.

## Return and independent local acceptance

`artifacts/return_manifest.json` maps every download-relative key to actual
remote path/SHA/bytes: learned last/final, full/source/config/metrics, legal H1,
official backbone and train/split/test metadata, confirmation/budget/runtime,
both exact implementation inventories and inference/CSV/ZIP/logits/features.
It contains no image bytes. Root downloads into one new directory and verifies
all hashes. Then run the independently supplied template acceptance tool:

```sh
python -B scripts/accept_return.py --return-root ACTUAL_NEW_LOCAL_RETURN \
  --manifest ACTUAL_RETURN_MANIFEST --expected-manifest-sha256 ACTUAL_SHA \
  --cpu-model-recovery --output ACTUAL_NEW_LOCAL_ACCEPTANCE_JSON
```

This rechecks all returned bytes and implementations, CSV/ZIP, and independently
restores one CPU model/head/optimizer/scheduler/scaler/full AC/RNG with synthesized
forward. A submission is ready only after actual remote inference and local
acceptance. Engineering tests are not an actual submission acceptance.

## CPU engineering scope

Tests cover both frozen152 and learned154 restoration, missing/corrupt heads,
equal last/final tensors, incomplete/wrong full scope and metric hashes, real
AdamW/scaler/cosine20 recovery, unchanged four-view logits with feature hook,
fixed graph parity with5748 and raw-margin/isolation protection, expired budget
or wrong graph point, unsafe/labeled test metadata, exact CSV/ZIP mapping/order,
no overwrite and downloaded corruption, all actual CLI imports. No official
model, checkpoint, image, server, connector or CUDA is used by local fixtures.
The owned-session cleanup fixture also covers a transient `/proc` permission
failure without allowing a foreign process group to receive a signal.

The revision's bounded CPU verification receipt is `CPU_VERIFICATION.json`;
its new metadata fixtures check original-plus-capacity cumulative accounting,
exact total, missing recovery, atomic reserve and final deadline. Provider CPU
verification uses the actual 578 template and its three CLI imports. No future
FULL source/checkpoint SHA or real-model restoration is claimed by these tests.
