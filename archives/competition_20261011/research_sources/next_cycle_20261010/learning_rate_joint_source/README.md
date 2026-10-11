# Joint J5 alpha exploration

Independent source from the actual 4fc8 J5 fold. Only SCE alpha .3/.5, run ID and output directory differ. The unchanged producer is scripts/train_n1_peft.py; its original optimizer, cosine20, AC, augmentations, AMP, RNG and exact-last restore remain unchanged. This is exploratory reuse of disclosed split57189, not an unused confirmation fold. New official CLIP plus legal fold H1100 e88a initializes each arm; neither full H1 nor parentfinal initializes a candidate.

Dedicated ROOT: /root/autodl-tmp/AICtrack1_next_cycle_supervision_joint_r3_20261010. Canonical lock is the original global training.lock.json. Fixed root budget inputs/budget.json is 2026-10-09 18:20 to 2026-10-10 02:20 Asia/Shanghai, 28800 seconds; it does not reuse or extend the old 14:19 gate. The disk ledger preserves original13,013,950,464 plus actual capacity32,212,254,720, adjusted45,226,205,184 minus real free; cumulative max20GiB, minimum3GiB, save reserve max(actual payload+64MiB,2GiB). Final GPU4h is retained in the overall plan. No test images are trained or inferred by this source.

Run from the deployed exact ROOT with existing CUDA Python, setting OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python/Torch import:

    python -B scripts/run_supervision_joint.py --preflight
    python -B scripts/run_supervision_joint.py --smoke A3
    python -B scripts/run_supervision_joint.py --smoke A5
    python -B scripts/run_supervision_joint.py

Resume an inspected partial arm only with --resume-arms A3 and/or A5. The canonical queue skips existing complete arms; partial output requires exact last.pt and unchanged source/config. No plannedpause/stop19 is exposed. A3 then A5 each finish real20 or the queue stops on failure. The original trainables are154/88,233,966 and head LR5e-6. All four evaluation views/P7/graph happen in separate frozen validation later; graph/P7 are not swept per model.

The safe GO lifecycle functions are copied verbatim from reviewed retry CONT: every created Popen is registered before identity capture, on_start attaches the actual canonical trainer before GO, command exec preserves PID/session/parent, and closure-unconfirmed os._exit(70) cannot traverse queue finally and release a lease. Synthetic smoke deletes only its own resume.pt after actual next-step restore, retaining the real report/SHA; it does not produce a candidate. This source has no SSH helper and no automatic leaderboard submission.

R3 supplies the unchanged real baseline scripts/train_peft.py SHA0758b989 required by train_n1:491 during full initialization. The trainer and every src member remain byte-equal to actual4fc8; reviewed full import/ROOT hard-file closure is verified separately. GPU synthetic smoke alone does not verify this full initialization path. R2 real failed initialization is retained separately; new R3 uses the same original18:20–02:20 budget without restarting its clock.
