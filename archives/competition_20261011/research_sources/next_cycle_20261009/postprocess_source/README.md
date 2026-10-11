# Private P0 frozen-score implementation

Scope: Stage2 **heldout** final/last four-view logits only. CPU only; no images,
model construction, optimizer, CUDA, remote access or test-label lookup.
Original implementation, based on this project's P7 formula and the standard
sparse transportation LP; no third-party research code copied.

Runtime verified locally: `D:\aicomp\AICtrack1\.venv\Scripts\python.exe`, Python
3.12.10, NumPy 2.5.2, SciPy 1.18.1, Torch 2.11.0+cu128. Torch only supplies CPU
FP32 P7 arithmetic and `weights_only=True` cache loading. SciPy HiGHS dual simplex
supplies the sparse transport optimizer; no package installation is needed.
Other runtimes must record actual dependency versions; source manifest records
these versions as tested, not as a claim about the server.

## Commands

```powershell
$py='D:\aicomp\AICtrack1\.venv\Scripts\python.exe'
$src='D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\postprocess_source'
Set-Location $src
& $py -m unittest -v test_p0
& $py "$src\run_p0_audit.py" --bundle '<verified fold fused_logits.pt>' --identity '<input_identity.json>' --output '<new output directory>' --soft-only --threads 4
```

Omit `--soft-only` for both capacity variants. Each capacity candidate has a
single 3600-second CPU solver budget **across both query protocols and all top-k
attempts**; two variants total at most 7200 seconds plus bounded soft/metric
work. `--capacity-seconds 120` is useful for a smoke profile. `--natural-only`
disables the independent balanced subtask (diagnostic only). Output paths must
not already exist; source/cache files are never overwritten. Run on Linux by
using the same three Python files and local bundle/identity/rows paths.

The preregistered 12 algorithm specifications are raw, 9 soft settings from
temperatures [.35,.5,.75] x strengths [.5,.75,1], and two raw-score capacity
intervals [.8N/C,1.2N/C] and [.9N/C,1.1N/C], floor lower / ceil upper. P7 .5/.75/60
is one of the 9 soft candidates. No automatic extension or test-based tuning.

The optimizer retains each row's stable top-5 score edges; **only infeasibility**
allows expansion to top-20. Timeout/numerical/identity failures do not produce
a certified assignment. With fewer than 20 classes the graph size is capped
at C. The algorithm minimizes total raw-score regret subject to one class per
row and per-class capacities. The network LP has integer capacities and a
totally unimodular incidence formulation. Returned predictions are checked
for integrality, row/edge validity, capacities, and decoded-vs-LP objective gap.
The report explicitly limits optimality to the supplied graph; top-5 feasible
is not necessarily the top-20 optimum, and neither proves dense N x C optimality.
It never constructs an N x N repeated-class-slot Hungarian matrix.

## Input schema

The existing `.pt` bundle must use schema
`aictrack1.stage2_train_derived_fused_logits.v1`, train-derived scope, floating
finite logits [N,C], targets [N], train_class_counts [C], checkpoint/config/split/
manifest/mapping hashes and the fixed **four-view** protocol. If it contains
sample_ids they must exactly match sidecar row order. Convert FP16 cache to
FP32 for common calculation; this cannot restore discarded FP32 precision.

An independently audited identity sidecar is required:

```json
{
  "schema": "aictrack1.p0_frozen_fold_input.v1",
  "stage": 2,
  "holdout_verified": true,
  "training_validation_overlap_rows": 0,
  "checkpoint_role": "last",
  "test_access": false,
  "source_receipt_sha256": "<actual independent provenance receipt SHA256>",
  "bundle_sha256": "<actual SHA256>",
  "checkpoint_sha256": "<same as bundle>",
  "experiment_config_sha256": "<same as bundle>",
  "split_sha256": "<same as bundle>",
  "manifest_sha256": "<same as bundle>",
  "class_mapping_sha256": "<same as bundle>",
  "view_protocol": "<exact bundle four-view text>",
  "training_count": 118958,
  "validation_count": 29737,
  "row_csv": "rows.csv",
  "row_csv_sha256": "<actual SHA256>"
}
```

`rows.csv`: `sample_id,target,group` in exact bundle order; optional
`four_view_prediction` records saved FP32 argmax for quantization sensitivity.
Group must be independently joined from official train near-duplicate group;
verified singleton uses `sample:<sample_id>`. Do not fabricate every row as
singleton when duplicate metadata exists. The sidecar records prior independent
holdout/source audit; its boolean is not a replacement for that audit.

## Query protocols and interpretation

1. `natural_all_unique`: all distinct cached IDs, full C columns, uniform target.
   Report actual per-class image/group support and natural marginal imbalance.
   This is a diagnostic of uniform-prior sensitivity, not a realistic balanced
   test proxy; a loss here cannot alone reject all balanced methods.
2. `balanced_supported_unique`: classes with >=20 distinct available groups,
   deterministic seed20261009 sample of 20 per retained class without replacement,
   at most one image from each group even across conflicting labels. Only those
   class columns remain, so this is a disclosed **closed-set subtask**, not a full
   750-class score. Omitted classes and all query IDs are saved. It cannot prove
   behavior for rare classes, which remain visible in protocol 1.

Each candidate reports observed-label Top1/macro, deltas against P7 and raw,
correct-to-wrong/wrong-to-correct, raw-logit regret, class predictions/recall,
fixed raw top-decile-margin damage, and paired group bootstrap500. These are
conditional fixed-model/noisy-label intervals, not training-seed confidence or
hidden test accuracy. Model/crop/precision/cache identity is common to controls.
No algorithm uses query labels except predeclared subtask construction and
metrics. No automatic claim of stable gain or freeze decision is made.

Result `report.json`, query CSVs, candidate predictions NPZ, and `manifest.json`
are private, SHA bound and reusable for independent review. Source manifests
list all private Python/tests/docs files, actual dependencies and test command.
