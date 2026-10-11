# PF2 joint154 persistent supervision eligibility

Fresh official OpenAI CLIP + legal e88a fold H1100, disclosed split57189, all118958 train observations, seed20261008. Original J5 alpha.1/AC/336/fullvisual154/88,233,966 parameters; only PF2 gate is added. PF30 max_ac_weight=.30 is the only executable arm. PF50 .50 is pre-registered in inputs/pf2_protocol.json but has no runnable config. No B teacher or extra weak-view forward; no test access.

Actual root-issued budget inputs/budget.json is copied byte-for-byte from actual_PF2_budget_r1.json SHAab5edf6db05351732364a8ec6263dc1114d7dcf45ceee341d128a7f7f19a0049. Phase06:47:07-09:47:07 Asia/Shanghai, <=10800sec shared train+frozen-validation. Original initial13,013,950,464 plus actual32,212,254,720=45,226,205,184, total128,849,018,880, cumulative30GiB/min3GiB/atomic2GiB, final4GPUh retained. inputs/previous_A_budget.json is historical unchanged, not active. No future checkpoint/launch SHA is invented.

Dedicated ROOT /root/autodl-tmp/AICtrack1_next_cycle_persistent_filter_joint_20261010. Before Python set OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8.

    python -B scripts/run_supervision_joint.py --preflight
    python -B scripts/run_supervision_joint.py --smoke PF30
    python -B scripts/run_supervision_joint.py

Only PF30 executes continuous20/cos20/fixedlast via original canonical queue/GO. Inspected partial resume uses --resume-arms PF30. No plannedpause and no automatic waiter. Original p1_smoke verifies model154/nextstep; gate math/actual mask accounting is covered by CPU tests, not inferred from synthetic CUDA model smoke.

Eligibility: AC weight<=.30, EMA label agreement<=.20 and prediction stability>=.80 for3 consecutive observations. Next-epoch mask, reentry allowed. Per class <=10% rows excluded, >=16 retained, <=10% original AC mass removed. Weighted loss renormalizes by retained batch AC mass; .30 does not mean remove30% data. All official train rows still supply detached original strong AC observations. All-zero supervision batch skips optimizer and AdamW decay as original gate.

Epoch history persistent_filter.actual reports actual mask SHA/source completed epoch, observation/supervision/exclusion counts and original/removed/retained AC mass globally and per class, from immutable CPU mask/weight snapshots before later AC/gate updates. persistent_filter.next is explicitly next-epoch mask. AMP skipped updates are separate; AC mass is neither optimizer-update magnitude nor true-noise precision. Strict gate-v2 restore binds threshold/IDs/streak/mask/epoch/current AC weights and10%mass; old v1 state is rejected. Pure module .10 is reserved only for old-v1 numerical parity CPU fixtures; no .10 runnable config exists.

Same-checkpoint frozen4V/features, fixed graph(k10/lambda.3/3round/T1/ln4) and P7(.5,.75,60) compare directly vs actual matched J5_GRAPH; raw separate. This source does not select winners, produce admission, confirm on new610 or run FULL/test. PF2 diagnostic/typed receipt must be adapted to real PF2 state and cannot borrow B positive gates. Old PF1 was96/frozenhead with nearzero negative result; PF2 gains/71 remain unproven.
