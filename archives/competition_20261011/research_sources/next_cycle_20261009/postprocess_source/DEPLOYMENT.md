# P0 execution and handoff, 2026-10-09

Implementation is private under `artifacts/next_cycle_20261009/postprocess_source`.
No root method was changed; no new parameters have been frozen for test. Existing
P7(.5,.75,60) stays default.

## Exact input and completed commands

CPU runtime:
`D:\aicomp\AICtrack1\.venv\Scripts\python.exe` (Python3.12.10,
numpy2.5.2/scipy1.18.1/torch2.11.0+cu128); CUDA was not used.

Verified cache:
`D:\aicomp\stage2-server-transfer\fv2_results_20261008\diagnostic\artifacts\diagnostics\fullvisual\fused_logits.pt`

SHA256 `4813ad6727624f3072ff6e5d28d8ecaa9b4a9e919a5e0b489946dfeea08180b9`;
FP16[29737,750]. This is **heldout fold final epoch20 last**
`078bdae1654cecc076d19e8e67b2401c6a47b985d90c91f7b6ec25293002b6c9`, not
metrics.selected_checkpoint best and not the full-train model.

Independent input sidecar:
`D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\acceptance_source\fv2_input_identity.json`

ID/group rows:
`D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\acceptance_source\fv2_rows.csv`

Schema and loader guards are in README. Literature audit checked actual fixed
last, H1 train-row binding, official manifest/split/labels/order/counts, and zero
train/validation exact/near duplicate group overlap. There is no original saved
FP32 argmax CSV locally: quantization mismatches remain **unknown**, not zero.

```powershell
$py='D:\aicomp\AICtrack1\.venv\Scripts\python.exe'
$src='D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\postprocess_source'
$cache='D:\aicomp\stage2-server-transfer\fv2_results_20261008\diagnostic\artifacts\diagnostics\fullvisual\fused_logits.pt'
$identity='D:\aicomp\AICtrack1\artifacts\next_cycle_20261009\acceptance_source\fv2_input_identity.json'
Set-Location $src
& $py -m unittest -v test_p0
& $py run_p0_audit.py --bundle $cache --identity $identity --output '<new output directory>' --threads 4
& $py run_p0_extension.py --bundle $cache --identity $identity --parent-report '<P0 audit output>\report.json' --output '<new extension output directory>'
```

Completed private result folders:

- `../p0_fv2_soft_audit_v1`: 9 soft points, raw, unique query protocols.
- `../p0_fv2_capacity_audit_v1`: all12 registered specifications. Soft calculations
  repeated only to complete this common table; this is **not independent evidence**.
- `../p0_fv2_boundary_extension_v1`: exactly two added points (.2,.75),(.25,.75),
  original query IDs/groups unchanged, parent report hash bound. Total14 specifications.

Each folder has JSON report, query-ID CSVs, predictions NPZ and file SHA manifest.
Metrics are percentage points/observed-label agreement, not hidden-test accuracy.

## Actual findings

Balanced unique stress:737 classes x20=14,740 rows, each a distinct image and group,
13 excluded classes disclosed. Raw74.36906%, P7 approximately74.59294%. Best soft
.35/.75 gains20 correct /+.135685pp against P7, group500 CI[-.020353,+.278155]pp.
Extensions .2/.75:+.067843pp CI[-.149254,+.257802], .25/.75:+.122117pp
CI[-.074627,+.298507]. The original boundary point stays best; no further expansion.

Capacity [.8,1.2]:−.217096pp vsP7, [.9,1.1]:−.169607pp. Top5 feasible and integral,
counts16–24 and18–23; ~.72s each. The solutions optimize raw scores **within the
top5 graph**, not the dense matrix or all conceivable capacity choices.

All750 natural unique:29,737 IDs/29,632 groups, uniform prior mismatch disclosed.
Raw beatsP7+.161415pp. Best-balanced soft .35/.75 is−.188318pp vsP7; extensions
−.531325/−.383361pp. Capacity top5 graphs infeasible; top20 then feasible, ~5.3s
each, −.100884/−.568316pp vsP7. This protocol is diagnostic, not a balanced test
estimate. The two protocols disagree; current signals are small/conditional.

Decision: P7 unchanged. New soft point is only a candidate for a **fresh matched
confirmation**; capacity variants have no improvement over P7 here. Do not add
these gains to P1 or promise71. No test inference/parameter selection occurred.

## Integration with P1 and fresh confirmation

P1 should save its fixed final/last four-view logits using the exact same ordering,
view averaging and precision as its control. The new split needs a new independently
verified input sidecar, including H1 fitted on that split's training rows only.
A full148695-trained model must never be used for heldout validation.

Freeze **one** candidate JSON before looking at the fresh cache:

```json
{
  "schema": "aictrack1.p0_frozen_decoder.v1",
  "type": "soft",
  "temperature": 0.35,
  "strength": 0.75,
  "iterations": 60,
  "selection_frozen_before_confirmation": true,
  "selection_report_sha256": "<actual prior P0 report SHA>",
  "status": "candidate for confirmation; not deployed to test"
}
```

The example is **not a newly frozen deployment decision**. Root may choose it for
confirmation or retainP7. `run_p0_confirmation.py` compares only this point and
P7, group bootstrap500 seed20261010, with no rescan:

```powershell
& $py run_p0_confirmation.py --bundle '<fresh fold final four-view cache>' --identity '<fresh verified sidecar>' --specification '<one preregistered candidate JSON>' --exploration-identity $identity --output '<new confirmation folder>'
```

It rejects identical exploration split/cache. It reports both unique protocols,
fixed-model conditional interval, class support, changes and margin damage;
different splits can still overlap in this official data pool, so do not call
the whole fresh fold statistically independent of past work without auditing IDs.

Callable interface for P1 combination / final inference after root freezes the
actual decision:

```python
from p0_frozen import P7_DEFAULT, apply_frozen
predictions, decoder_report = apply_frozen(four_view_logits_numpy_fp32, P7_DEFAULT)
```

The matrix must be one checkpoint's equally averaged4V scores. This API reads
no labels/images, builds no model and updates no parameters. Recompute frozen
soft bias on that inference matrix (as existingP7 does); freeze hyperparameters,
not a class bias learned from validation. Final mapping/CSV/ZIP validation and
singletest authorization stay with root's existing submission pipeline.

Combination table is at most {C,acceptedP1}x{P7,one selectedP0}; no model-specific
new grid. Never average C andP1 logits or checkpoints. Candidate gain remains
exploratory unless fresh confirmation supports it. If confirmation≤−.25pp or
highmargin/rare-class damage is severe, retain P7/current model.

## Verification and limitations

Ten lightweight tests pass:24 small sparse graphs compared with exhaustive
assignment optimum; top-k infeasibility fallback; integer/edge/capacity/objective
checks; repeated tie determinism; invalid scores/bounds; no-replacement/group
uniqueness; duplicate-ID rejection; projectP7 formula parity; grouped flip metrics;
test/nonholdout and row-order rejection; frozen interface rejects extra grid.

Windows mkdtemp0700 created an inaccessible empty fixture on first two attempts;
test now uses a retained private synthetic fixture with ordinary inherited ACL.
No unsafe delete/permission override was used. Source manifest lists actual source
files explicitly, not these empty test directories or bytecode. Fresh confirmation
runner is prepared but has **not** seen a freshfold, nor was final test deployment run.
