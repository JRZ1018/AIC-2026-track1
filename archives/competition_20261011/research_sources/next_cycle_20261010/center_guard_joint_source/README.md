# CG25 single-variable exploration, CPU-prepared only

Status: UNBOUND. No server deployment, budget clock, GPU training, last checkpoint or admission.

Foundation: frozen e94 joint154 producer and exact baseline J5 config a65f60; original scientific source 4fc8. This source runs only CG25 and preserves official CLIP, legal e88a oldfold H1, split57189, alpha .1/beta1/logzero-4, joint152visual+2head, original AC strong observation, 336, seed20261008, OMP4/MKL4/CUBLAS4096:8, AdamW/Cos20 and fixedlast20. No PF/EMA/AFA is enabled.

The only scientific variable is center_guard_fraction=.25 in training transforms. Draw original scale/left/top/flip RNG with unchanged bounds and draw count. Deterministically remap the drawn coordinate to the integer interval making the image center lie in the central50% of the square crop. None/0 follows the original pixel path exactly. Keep crop size and flip; no letterbox, no crop .95, no object detector and no extra view/forward. Tiny integer geometry uses centered feasible coordinates. This is geometric center protection, not evidence of object preservation. Off-center subjects and reduced translation diversity are risks.

AC observes the same candidate strong logits with unchanged observer math. Crop changes can change confidence histories, weights and all later optimization; no claim of identical AC trajectory for enabled CG25.

Existing control can be reused only after root checks original J5 last456301, split/order/query/initial officialCLIP/e88a, original config, actual runtime, fixed20 and fixedgraph/P7 identities. Parent data/config are stored byte-for-byte. Candidate trainable state plus augmentation identity are saved in last/final/metrics and required before exactlast resume.

CPU preparation: python scripts/run_center_guard_joint.py --check-source; python tests/test_center_guard_CPU.py.
Root must supply a new typed budget with real finite start/deadline <=3h and the original expanded capacity ledger (30GiB/min3/atomic2/finalreserve4). No old clock is reset. On the dedicated deployed TEMPLATE only, run python scripts/prepare_center_guard.py --budget ACTUAL_JSON --expected-budget-sha256 ACTUAL_SHA. It creates the new exclusive execution root /root/autodl-tmp/AICtrack1_next_cycle_center_guard_joint_20261010 and a bound manifest, without model or image reads.
Then actual root CPU asset preflight, canonical CUDA smoke CG25 (never candidate initialization), all smoke groups closed/canonicalidle, and one original canonical runner launch. Existing commands: python scripts/run_center_guard_joint.py --preflight; --smoke CG25; or no args for continuous20. Only exact own last may resume with --resume-arms CG25. The original run_serial_queue, owned GO lifecycle, trainer loss/optimizer/AC loop and safe atomic-save guards are retained.

No test images or labels are accessed. No diagnostic/FULL/final/writer/terminal waiter is built here. Future budget/last/admission are null. Expected 20epoch training cost ~1.8-2.2h from actual J5/PF wallclock, conditional on server throughput; this is not package ETA or a gain promise. Root may decline activation based on EMA/AFA priorities.

The prior 30case in-memory synthetic design proof was a real tool output, not a saved receipt. Only new test runs produce author receipts.
