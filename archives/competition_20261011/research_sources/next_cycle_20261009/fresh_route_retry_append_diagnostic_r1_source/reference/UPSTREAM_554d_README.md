# Fresh typed route diagnostic, revision 1

This independent source provides new-fold C versus **one previously chosen route**: S/P7 or J5/fixed graph/P7. It does not select another route, run training, prepare full-data recipes, access test, or submit online. The root operator performs deployment and execution. No server/GPU/real fresh checkpoint was used during local engineering acceptance.

Production diagnostic root: `/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_diagnostic_20261009`.
Production training root: `/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_20261009`.
The actual bound training source SHA is generated **after** the actual typed admission. The reviewed template is `05df2bbd14af808847a430bb4a23ba22483774dad00e6724026ea37dd633bf88`; this source verifies its static members against `inputs/training_template_manifest.json`. An arbitrary template cannot be substituted.

## Fixed identities and restoration

- Fresh split `57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7`; 118958 train / 29737 validation, 750 classes.
- Fresh train-only H1 `e88a1cbbaa8d9ebdbf1c7cf2707ac434cf440fac8bd5b4a5dc57893ba0fe15be`, producer source `448a2b2a293cb3a7a51f7bd9b5fad7afde9f2cae196bc951485dedfa98511d2c`, receipt `06c762c1828915fd579fcc338ade3177f5eff96be742c9650e265845ce6f7d73`.
- C and S restore exactly visual152 / 87849216 FP32 master parameters and keep the actual new H1 head bitwise equal, frozen. S changes only SCE alpha .1→.5. J5 restores visual152 plus complete classifier weight/bias, joint154 /88233966. Its training `classifier_head_restored_equal=false` denotes a changed head; the diagnostic separately verifies exact restoration of its trained head.
- Only fixed epoch20 `last.pt`; actual terminal metrics, all20 history entries, source/config SHA, complete optimizer moments, scheduler/scaler and complete AC global train IDs/labels must agree. `best.pt` and old C078/cache4813 are rejected.
- The real last producer does not save `head_initializer_sha256`; initializer binding uses actual config, terminal metrics and H1 bytes. The selected `final.pt` field is not imposed on `last.pt`.
- The training contract is imported as its real `fresh_route_contract`; the local diagnostic contract is named `fresh_diag_contract` to avoid a module cache collision.
- Matching OMP4/MKL4/CUBLAS4096:8 and actual torch threads4 plus equal interop are recorded from immutable training attempts; diagnostic inference has zero optimizer updates.

## Resources and execution

Keep original phase start `1791497972.440027`, deadline **`1791526772.440027` (2026-10-09 14:19:32.440027 +08)** and original system data-disk free baseline **13013950464**. The overall cumulative allocation ceiling is8GiB, min actual disk free3GiB, min effective host/cgroup free4GiB. This source never resets the baseline/deadline. Parent controllers must preserve their own exact PID supervision and deadline.

The raw four-view engine acquires the existing canonical lease `/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json`. It refuses a live training/diagnostic lease. CPU comparison/graph work does not acquire a GPU lease. A root operator must check actual previous phase closure before GPU entry; this package neither clears locks nor kills other processes.

Set `PY` to the already accepted server interpreter, `TRAIN` to the literal training root, `DIAG` to the literal diagnostic root, `SOURCE_SHA` to the actual newly bound training manifest SHA, and `LAST_SHA` to the actual arm last20 SHA. The following are operator commands, **not local acceptance commands**.

```bash
"$PY" "$DIAG/run_fresh_route_diagnostic.py" --training-root "$TRAIN" --expected-training-source-sha256 "$SOURCE_SHA" --dependency-check
"$PY" "$DIAG/run_fresh_route_diagnostic.py" --training-root "$TRAIN" --expected-training-source-sha256 "$SOURCE_SHA" --arm C --expected-last-sha256 "$C_LAST_SHA" --deadline-unix 1791526772.440027 --hard-seconds 1800 --workers 4 --run
```

Run exactly the selected candidate after C completes. Use `--save-features` for J5_GRAPH; it attaches a read-only visual hook to the **same four already-required forward calls**, with no extra forward. Features are FP32 normalize(mean(normalize4 raw embeddings)) with exact sample-ID order/checkpoint/config binding.

```bash
"$PY" "$DIAG/run_fresh_route_diagnostic.py" --training-root "$TRAIN" --expected-training-source-sha256 "$SOURCE_SHA" --arm "$SELECTED_ARM" --expected-last-sha256 "$CANDIDATE_LAST_SHA" --deadline-unix 1791526772.440027 --hard-seconds 1800 --workers 4 --save-features --run
```

Outputs are exclusive `artifacts/diagnostics/FRESH_C` and `FRESH_S` or `FRESH_J5`. Each has `fused_logits.pt`, `rows.csv` (actual original FP32 argmax included), `input_identity.json`, complete restore `p1_checkpoint_audit.json`, canonical `runtime_receipt.json`, report and SHA/byte result inventory; optionally `fused_features.pt`. `--attempt-name retry1` creates a separate receipt path, never overwrites primary.

## CPU raw report

`selection_protocol.json` is registered for the typed routes before fresh predictions. The copied 2e0 support-only query arrays/CSV remain byte-identical; only the registration reference in `query_identity.json` changed. `inputs/query_origin_receipt.json` records original2e0 SHA and unchanged membership. Natural covers29737 unique images and750 columns. Balanced supports **738×20=14760** independent groups, omitting12 low-support classes; it is not a full750 score. Old exploration supported737×20 and is not substituted here.

```bash
"$PY" "$DIAG/compare_fresh_route.py" --control-bundle "$DIAG/artifacts/diagnostics/FRESH_C/fused_logits.pt" --control-identity "$DIAG/artifacts/diagnostics/FRESH_C/input_identity.json" --candidate-bundle "$DIAG/artifacts/diagnostics/FRESH_$SELECTED_ARM/fused_logits.pt" --candidate-identity "$DIAG/artifacts/diagnostics/FRESH_$SELECTED_ARM/input_identity.json" --threads 2 --output "$DIAG/artifacts/paired/raw_$SELECTED_ARM"
```

Both sides use fixed P7 temperature.5/strength.75/max60 and CPU2. Report per-class actual correct counts, 20/30/50 head/mid/tail, supported ≥10pp decline, high-margin flips, group bootstrap500/seed20261009, natural and raw results. FP16 cache conversion cannot recover discarded original logit precision; FP32 argmax mismatch counts are disclosed.

For S, this raw P7 direct-C report is the primary. For J5_GRAPH it is a **separate diagnostic**, not the graph confirmation gate.

## Single fixed graph direct-C confirmation

This route requires actual candidate features and only k10/lambda.3/T1/3 original-P residual iterations/ln4 protection. Cosine top20 is blocked at64MiB, top10 reciprocal strictly-positive edges are symmetric then row-normalized. Query graphs are built separately for natural and balanced; labels never enter graph construction/smoothing. Protected/zero-degree/unchanged rows preserve exact original logits. P7 is applied once after graph scores.

The existing5748 shared ledger must already exist at `/root/autodl-tmp/AICtrack1_next_cycle_graph_20261009/artifacts/graph_budget.json`, source pin `5748cf07c01bd8bf3d68bf5b50065303d2c8990260b00400e5511593aaa0834c`, spent at least64.3705 /total7200. This source imports the original `BudgetLease`, charges validation plus computations, takes its existing exclusive lock, and never initializes/resets that ledger. Future attempts share the same remaining budget. Output is exclusive; after charging, a small ledger snapshot is retained.

```bash
"$PY" "$DIAG/confirm_fixed_graph_vs_C.py" --candidate-bundle "$DIAG/artifacts/diagnostics/FRESH_J5/fused_logits.pt" --candidate-identity "$DIAG/artifacts/diagnostics/FRESH_J5/input_identity.json" --candidate-features "$DIAG/artifacts/diagnostics/FRESH_J5/fused_features.pt" --control-bundle "$DIAG/artifacts/diagnostics/FRESH_C/fused_logits.pt" --control-identity "$DIAG/artifacts/diagnostics/FRESH_C/input_identity.json" --threads 2 --deadline-unix 1791526772.440027 --output "$DIAG/artifacts/paired/fixed_graph_J5_vs_C"
```

This reports direct **combo versus fresh C**, raw J5 versus C and graph increment versus J5 independently. A negative or below.10 raw J5 signal is disclosed and does not veto this typed graph route. No new grid, model, training or extra forward runs.

## Typed confirmation receipt

`write_fresh_route_confirmation.py` emits schema `aictrack1.fresh_route.confirmation.v1`, status `complete_positive_fresh_route_confirmation`. Primary must be >0 on the new balanced protocol. Natural≥0 and CI lower>0 are **not hard gates**; CI crossing zero leaves `stable_gain_claim=false`. Root must review class/collapse/high-margin/source/recovery risks in the actual report. This typed receipt is not a677/12cc naked-J receipt and does not prepare or launch full/final work.

Pass JSON `--evidence-json` with real absolute `{path,sha256}` entries:

`checkpoint_C`, `config_C`, `identity_C`, `checkpoint_audit_C`, `runtime_C`, `checkpoint_candidate`, `config_candidate`, `identity_candidate`, `checkpoint_audit_candidate`, `runtime_candidate`, `paired_report`, `training_source`, `diagnostic_source`, `selection_protocol`, `frozen_postprocess`, `split`, `h1`, `h1_receipt`, `h1_source`, `admission_receipt`, `root_review`.

J5_GRAPH additionally needs `features`, `fixed_graph_protocol`, `graph_source`, `graph_budget` (the after-charge snapshot). `diagnostic_source` and `training_source` are their actual manifest files; both identities/audits/runtimes are bound to them, their config/last SHA and the selected route. Direct/nested paired identity SHA aliases must agree. The root review JSON must have schema `aictrack1.fresh_route.root_review.v1`, selected_route, decision `accept_fresh_route_confirmation`, exact `paired_report_sha256` and nonempty `reason`.

```bash
"$PY" "$DIAG/write_fresh_route_confirmation.py" --selected-route "$SELECTED_ROUTE" --paired-report "$ACTUAL_PRIMARY_REPORT" --evidence-json "$ACTUAL_EVIDENCE_JSON" --output "$DIAG/artifacts/confirmation/typed_confirmation.json"
```

## Local acceptance boundaries

Local tests use synthetic CPU declarations around private copies of the actual05df producer, actual vendor CLI/API imports, toy models, actual support-only query rows, graph/scientific fixtures and real small fixture hash joins. They do not claim a genuine production admission or actual new fold checkpoint exists. Synthetic bindings are explicitly rejected by production `--run`.

```powershell
.venv\Scripts\python.exe artifacts\next_cycle_20261009\fresh_route_diagnostic_source\tests\build_cpu_binding.py
.venv\Scripts\python.exe -m unittest discover -s artifacts\next_cycle_20261009\fresh_route_diagnostic_source\tests -p test*.py -v
```

Only the explicitly listed source-manifest members deploy; `artifacts/` fixtures and caches do not. Genuine bound-source dependency-check, C/candidate checkpoint restore, GPU inference and fresh scores remain pending until root performs those stages against their actual produced files.
