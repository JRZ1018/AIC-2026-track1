# Fresh C plus one typed route

This directory is an immutable source template. It is not bound for training and has no permission to run against test images. Root alone prepares and launches an actual source bundle after choosing one route from real old-fold evidence. It never modifies an existing bundle.

Supported routes are `S` (C152 frozen new H1, SCE alpha .1 versus .5) and `J5_GRAPH` (C152 versus J5 joint154, head LR 5e-6; both alpha .1). J5_GRAPH admission measures the actual frozen k10/lambda.3 graph plus fixed P7 directly against C. The old naked J5 +.06784pp is disclosed separately and cannot stand in for this graph route. Graph processing is done by the separate frozen graph engine, never inside training.

Both fresh arms start from official CLIP and the actual new-fold H1100 checkpoint e88a1c..., trained on only the new 118958 train rows. The split is 57189c... with 29737 previously unused query rows, all 750 classes, and zero old/new query connected-group intersection. The actual H1 source448a..., receipt06c762..., exact positive row hashes, and actual CPU restore acceptance are copied under inputs and bound by the template. No full-train H1 is accepted. Training seed20261008, raw336 augmentation, AC, AdamW, maxnorm1 and cosine20 match the current recipe. C/S use152 trainable FP32 visual tensors,87849216parameters, one optimizer group and head source `new_fold_H1_frozen`. J5 uses154 trainables,88233966parameters, two groups and `joint_fixed_last_state`; the actual diagnostic must reload and verify that head from the fixed last.

The sole launch root is `/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_20261009`, compared after resolving the literal mount path. Canonical lock remains `/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json`. All launch paths enforce actual assets/source/config SHA, OMP4/MKL4/CUBLAS=:4096:8 before Torch import, actual torch threads4, and original phase start1791497972.440027/deadline1791526772.440027 (14:19:32). They retain initial free baseline13013950464bytes, maximum aggregate growth8GiB and minimum3GiB, including atomic save peak plus64MiB. The deadline and baseline are never restarted. A source declaration is not an actual runtime receipt.

## Prepare and inspect, only after real route selection

Root supplies an actual `aictrack1.fresh_route.admission.v1` receipt with status `selected_recipe_for_fresh_route_confirmation`, selected_route/arm, old split0c902..., fixed-last20, direct primary scope `S_P7_vs_C` or `J5_graph_P7_vs_C`, positive primary_delta_vs_C_pp, test_access=false, used_test_or_online_scores=false and root_reviewed=true. S retains the registered .10pp old-fold continuation floor. Evidence includes absolute actual path/SHA entries checkpoint, config, training_source, paired_report and root_review. Graph route also requires graph_result, combo_report and graph_protocol, the exact frozen graph parameters k10/lambda.3/residual3/T1/rawln4, and separate raw_J5_P7_delta_vs_C_pp. Root review decision is `proceed_to_fresh_confirmation`, with selected_route and a nonempty reason. Every referenced file is actually hashed before preparation and again before launch; this tool does not invent evidence or select a route.

```
python scripts/prepare_fresh_route.py --target NEW_ABSOLUTE_BUNDLE --admission ACTUAL_ADMISSION_JSON --admission-sha256 ACTUAL_SHA
```

The new bundle has `source_manifest.json` schema `aictrack1.fresh_route.training.source.v1`, status bound, actual dynamic SHA, and ONLY configs/experiment/p1_C.yaml plus p1_S.yaml OR p1_J5.yaml. It copies the actual admission as inputs/admission_receipt.json; files and source aliases bind selected_route/arm, admission_receipt_sha256, source_h1_manifest_sha256 and the actual new H1 row binding. Its assets schema is `aictrack1.fresh_route.assets.v1`. Templates have template_manifest.json instead of source_manifest.json and are deliberately rejected by training.

## Root-only GPU command, after source and real asset review

Run from the exact new bound root using the existing CUDA Python `/root/autodl-tmp/venvs/aictrack1/bin/python`, with environment OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 set before launch. For owned detached execution use the existing root operator launch pattern, redirect one unique outer log, and record the returned controller PID. No new server connector is included.

```
python scripts/run_fresh_route_arm.py --arm C --smoke
python scripts/run_fresh_route_arm.py --arm S --smoke
python scripts/run_fresh_route_pair.py
python scripts/run_fresh_route_pair.py --resume-arms C S
```

Replace S with J5 only when selected_route is J5_GRAPH. The pair always runs C then the single selected arm under the original canonical serial queue; it does not run all three. Smoke uses a UUID directory and does not produce a candidate checkpoint. Exact `last.pt` resume retains model/head, optimizer groups/moments, cosine20 scheduler, scaler, AC, observation protocol, RNG, immutable config and source. Optional `--stop-after-epoch 1..19` retains cosine20 and returns a pause receipt, not completion. No best checkpoint is used for the paired decision. Per-arm runtime receipts live at artifacts/runtime_receipts/ARM_ATTEMPT.json, schema `aictrack1.fresh_route.runtime.v1`, including arm, selected_route, actual threads, environment, config/sourceSHA and PID.

Fixed last fields are fresh_route_source_manifest_sha256, fresh_route_arm, selected_route, classifier_head_source and classifier_head_restored_equal. Logs retain coefficient quantiles, finite/nonfinite norm counts and AMP successful/skipped updates separately; original once-per-epoch scheduler logic is unchanged. A fresh positive paired diagnostic remains necessary before any full training. This source neither grants full/test access nor uses the old 677/12cc bare-J admission.

## Local CPU verification

`D:\aicomp\python312\python.exe tests/cpu_contract_tests.py` runs actual loss/AC/optimizer/scheduler/scaler/RNG resume interfaces on toy CPU models, exact recipe/admission/budget fixtures, immutable new-bundle preparation and real CLI help. It does not read images, checkpoints or server credentials, open a network connection, import a connector, or call CUDA. Tests do not claim CLIP throughput or real fresh validation improvement. Results are saved in the independent parent operator_tools directory, outside the frozen template.
