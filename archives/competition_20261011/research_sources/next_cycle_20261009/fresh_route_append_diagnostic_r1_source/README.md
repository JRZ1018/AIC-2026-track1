# Fresh J5_GRAPH append diagnostic, revision 1

This exclusive adapter keeps original554d and bound4fc8 unchanged. It only restores actual C/J5 fixed-last20, emits frozen four-view logits (J5 features from the same four forwards), compares the fixed graph+P7 route, and writes typed confirmation with the real continuation chain. Local CPU acceptance is engineering evidence; no real weights, images, server or GPU are opened by the tests.

Production adapter root: `/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_append_diagnostic_r1_20261009`.
Training root: `/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_20261009`.
Bound training SHA: `4fc8e1420cf9347e79f7a0f607cd49a2f28735900de52477091324c06a0b8482`.
Unchanged upstream diagnostic SHA: `554d7c8f839d4ab6fee1ce427c5e2b9b73435d4766ac5347794ceecace53fbc2`.

## One shared appended stage

The original start1791497972.440027/deadline1791526772.440027/baseline13013950464 and original budget bytes remain unchanged. One root-published `aictrack1.fresh_route.continuation.training_budget.v1` must start at1791526772.440027 and end no later than1791533972.440027 (14:19:32 to16:19:32 +08). Exact J5 resume if needed, both frozen4V diagnostics, and fixed graph validation all consume this same7200-second window. There is no new per-arm stage or baseline reset. Cumulative allocation max8GiB, actual free floor3GiB, effective host/cgroup floor4GiB remain.

The budget binds real C20 and J5 partial-or20 origin metadata, source, configs, exact original C births plus actual J5 controller/trainer births. The continuation operator must publish its frozen source, pre-resume CPU audit, terminal CPU audit and COMPLETE receipt. GPU diagnostic admission revalidates all four actual files and the original/current trainer groups closed; it never clears locks or kills prior processes. If J5 was already20, the same operator emits a no-resume chain with zero additional trainer.

## Root deployment and CLI

Deploy this new source exclusively; never overwrite original554d or its production root. Define PY as the accepted server interpreter, TRAIN/DIAG as the roots above, SOURCE_SHA as the bound4fc8 SHA, APPEND_BUDGET as the actual published budget path, APPEND_SHA as its actual SHA, DEADLINE as its declared deadline, and CHAIN_JSON as a real evidence mapping of `continuation_operator_source`, `partial_restore_audit`, `terminal_restore_audit`, `continuation_completion`, each an absolute path and SHA. None of these future values is fabricated in the template.

```bash
"$PY" "$DIAG/run_fresh_route_diagnostic.py" --training-root "$TRAIN" --expected-training-source-sha256 "$SOURCE_SHA" --dependency-check
"$PY" "$DIAG/run_fresh_route_diagnostic.py" --training-root "$TRAIN" --expected-training-source-sha256 "$SOURCE_SHA" --append-budget "$APPEND_BUDGET" --expected-append-budget-sha256 "$APPEND_SHA" --continuation-evidence "$CHAIN_JSON" --arm C --expected-last-sha256 "$C_LAST_SHA" --deadline-unix "$DEADLINE" --hard-seconds 1800 --workers 4 --run
"$PY" "$DIAG/run_fresh_route_diagnostic.py" --training-root "$TRAIN" --expected-training-source-sha256 "$SOURCE_SHA" --append-budget "$APPEND_BUDGET" --expected-append-budget-sha256 "$APPEND_SHA" --continuation-evidence "$CHAIN_JSON" --arm J5 --expected-last-sha256 "$J5_LAST_SHA" --deadline-unix "$DEADLINE" --hard-seconds 1800 --workers 4 --save-features --run
```

Both arms must use this same frozen diagnostic source SHA; old554d caches cannot substitute. Each entry requires exact current lastSHA, full20 history, actual152/154 strict restoration, real H1/e88a initializer, AC global train IDs/labels and optimizer/scheduler/runtime4 evidence. It uses the canonical GPU lease, frozen eval, batch64 and zero optimizer steps. Each hard-seconds ceiling is subordinate to the remaining shared deadline; it is not a renewed7200-second budget. Outputs are exclusive `artifacts/diagnostics/FRESH_C` and `FRESH_J5`; retry1 is separate and retains failures.

## Paired and fixed graph comparison

```bash
"$PY" "$DIAG/compare_fresh_route.py" --control-bundle "$DIAG/artifacts/diagnostics/FRESH_C/fused_logits.pt" --control-identity "$DIAG/artifacts/diagnostics/FRESH_C/input_identity.json" --candidate-bundle "$DIAG/artifacts/diagnostics/FRESH_J5/fused_logits.pt" --candidate-identity "$DIAG/artifacts/diagnostics/FRESH_J5/input_identity.json" --threads 2 --output "$DIAG/artifacts/paired/raw_J5"
"$PY" "$DIAG/confirm_fixed_graph_vs_C.py" --candidate-bundle "$DIAG/artifacts/diagnostics/FRESH_J5/fused_logits.pt" --candidate-identity "$DIAG/artifacts/diagnostics/FRESH_J5/input_identity.json" --candidate-features "$DIAG/artifacts/diagnostics/FRESH_J5/fused_features.pt" --control-bundle "$DIAG/artifacts/diagnostics/FRESH_C/fused_logits.pt" --control-identity "$DIAG/artifacts/diagnostics/FRESH_C/input_identity.json" --threads 2 --append-budget "$APPEND_BUDGET" --expected-append-budget-sha256 "$APPEND_SHA" --deadline-unix "$DEADLINE" --output "$DIAG/artifacts/paired/fixed_graph_J5_vs_C"
```

Primary is direct fixed graph+P7 versus freshC. RawJ5 versusC and graph increment remain separate reports. Fixed graph k10/lambda.3/3iterations/T1/ln4 high-margin protection; P7 temperature.5/strength.75/max60. Natural29737 and balanced738x20=14760 retain the original support-only query bytes and group-bootstrap CI. Do not run another grid. Original5748 graph shared ledger7200seconds, already-spent floor64.3705, CPU2 and its exclusive lease remain; this appended wall-clock window never resets that graph ledger. All computation must finish before the same appended deadline.

## Typed confirmation

`write_append_confirmation.py` verifies the additional budget/continuation chain, then calls the unchanged canonical `build_confirmation` and emits `aictrack1.fresh_route.confirmation.v1`. Positive primary delta>0 plus actual root risk review is required; natural>=0 or CI lower>0 are not added hard gates. The receipt is not a raw-J677/12cc admission.

Use the canonical21 evidence items: checkpoint_C, config_C, identity_C, checkpoint_audit_C, runtime_C, checkpoint_candidate, config_candidate, identity_candidate, checkpoint_audit_candidate, runtime_candidate, paired_report, training_source, diagnostic_source, selection_protocol, frozen_postprocess, split, h1, h1_receipt, h1_source, admission_receipt, root_review. Add graph4: features, fixed_graph_protocol, graph_source, graph_budget (the charged snapshot). Add append6: original_fresh_budget, appended_stage_budget, continuation_operator_source, partial_restore_audit, terminal_restore_audit, continuation_completion. Every entry is a real absolute path and SHA. Root review must bind paired report, both budgetSHAs and continuation sourceSHA and explicitly set `acknowledges_completion_after_original_deadline=true`.

```bash
"$PY" "$DIAG/write_append_confirmation.py" --paired-report "$DIAG/artifacts/paired/fixed_graph_J5_vs_C/report.json" --evidence-json "$EVIDENCE_JSON" --append-budget "$APPEND_BUDGET" --expected-append-budget-sha256 "$APPEND_SHA" --output "$CONFIRMATION_JSON"
```

No full/test work is launched here. Genuine frozen-source deployment, actual bound-source dependency check, actual checkpoints and actual runtime/budget/root-review receipts remain root-only operations after local acceptance. Historical copied554d CPU receipts are retained under `reference/UPSTREAM_554d_*`; they are not this adapter's acceptance.
