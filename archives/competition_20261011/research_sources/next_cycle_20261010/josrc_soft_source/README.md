# B: JoSRC-inspired single-student auxiliary soft repair

This template is unbound. It does not claim a complete Jo-SRC reproduction or an accuracy improvement. It uses only official Stage2 train images, prescribed OpenAI CLIP ViT-B/32, and the legal oldfold train-only H1100 (e88a). The revealed 57189 fold is exploratory; any selected B recipe still needs new independent heldout confirmation. No test images, labels or parameters participate in training.

The unchanged original J5 hard branch uses alpha=.1/beta=1/logzero=-4, original strong augmentations, natural sampler, strong-view AC, fullvisual152+head2, original LR/AdamW/Cos20/seed20261008. Additional soft coefficient .25 is first; .5 is preregistered and cannot launch implicitly. EMA decay=.999 updates on every successful optimizer step, including epochs1–5. Those epochs skip the extra weak forwards and soft term. Starting epoch6, independent RNG-isolated same-student weak eval and EMA weak eval finish before the strong gradient graph. Reliability uses the two student views; the detached EMA provides the target. The soft branch does not multiply the hard AC mask or remove samples. The raw student alone is selected at fixedlast20 and exported. There is no EMA validation/selection/export.

Numerical protocol is inputs/soft_math_protocol.json. The complete student, optimizer, Cos20, scaler, teacher/counts, support ledger and RNG are saved in last; mathematical source SHA is checked on resume. Full teacher transition matrices stay in checkpoint tensors while epoch metrics contain summaries.

## Profile first

Deploy this template exclusively to a new directory (e.g. /root/autodl-tmp/AICtrack1_next_cycle_josrc_soft_template_20261010). The unchanged e94e36 R3 files may be copied locally on the server only after per-file SHA comparison. Do not modify prior frozen source or rerun prior models. Create an actual root-authorized budget JSON with schema aictrack1.josrc_inspired.budget.v1, explicit real start/deadline, original free baseline13013950464 + observed expansion32212254720 =45226205184, actual total128849018880, explicit cumulative20..30GiB cap, min3GiB, atomic2GiB, final_gpu_reserve_hours4 and the existing actual capacity receipt. Future budget and checkpoint hashes are null in this template.

Use /root/autodl-tmp/venvs/aictrack1/bin/python and set OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, CUBLAS_WORKSPACE_CONFIG=:4096:8 before imports. Commands:

```text
python scripts/bind_josrc_source.py --mode profile --output-root /root/autodl-tmp/AICtrack1_next_cycle_josrc_soft_profile_20261010 --budget ACTUAL_ROOT_BUDGET.json
cd /root/autodl-tmp/AICtrack1_next_cycle_josrc_soft_profile_20261010
python scripts/run_josrc.py --preflight
python scripts/run_josrc.py --profile100 --arms B25
```

The actual profile opens official train-partition images only (118958/750 supports), creates a fresh prescribed CLIP+e88a student, and runs100 successful CUDA optimizer updates. It uses a virtual epoch6 to measure both additional no_grad forwards, with cold original AC warmup weights and an untrained EMA: it never claims actual five-epoch warmup or candidate performance. The 99th successful state is atomically written; the same materialized real batch is updated uninterrupted and after full disk restoration, comparing student/teacher/AdamW/scaler/RNG/support. Replay is a diagnostic update; the restored final state records100 updates. profile_receipt.json reports actual peak VRAM, throughput, honest upper-cost projection and full restore evidence; partial batch need not contain every class. Profile state cannot initialize or resume candidates. Canonical queue lock plus existing reviewed owned-child GO lifecycle controls execution. Never call the raw trainer outside the canonical controller.

## Production only after actual profile/root cost review

Bind --mode train to the distinct /root/autodl-tmp/AICtrack1_next_cycle_josrc_soft_20261010 root using a real later budget with the same cumulative baseline, not the profile state. Run --preflight then `run_josrc.py --run --arms B25`. It starts fresh official CLIP+legal H1 for20 continuous original cosine epochs and fixedlast student. --arms B50 may be explicitly selected only after root budget/cost review. --resume-arms B25 requires the same config/source and exact production JOINT_B25/last.pt after previous exact process groups have closed and canonical lock naturally releases. No automatic19th-epoch pause, waiter, query-based selection or test inference is included.

Author fixtures use tiny CPU models and one synthetic temporary image. CPU provider imports are deferred until actual production loader; local CPU validation never claims a real CLIP/CUDA profile has run.
