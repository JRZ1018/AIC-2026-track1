# Joint alpha exploration diagnostic

This independent adapter restores only actual R3 A3/A5 fixed-last20 checkpoints from source `e94e36cdea559ca8d424e9c1558e8aea37a6fcccd6af634f55e16c64a73bf524`. The unchanged numerical four-view loader/engine and graph producer are copied from the frozen historical source; their hashes are recorded in source_manifest.json. This is exploration on the already used57189 fold, not independent confirmation and never authorization for FULL or test inference.

The primary comparison is candidate fixed graph+P7 versus the previously generated actual J5_GRAPH prediction. Imported graph NPZ `candidate_p7` is used as the control decision; plain J5/P7 is reported only in the separate raw-model comparison. Every legacy identity retains its actual4fc8/019493 source hashes. The natural29737 and balanced738x20 query NPZ/CSV bytes remain unchanged. Only their protocol registration was rebound before alpha predictions. Graph k10/lambda.3/3iterations/T1/ln4 protection and P7 temperature.5/strength.75/60iterations are fixed. No grid, test access, parameter update, or extra forward is performed.

The same f6bc19 18:20-to02:20 budget covers training plus diagnostics. Original free/capacity expansion/cumulative20GiB/min3GiB/atomic2GiB remain unchanged. Historical graph budget references are provenance only and do not reset the present ledger.

## CPU-only check

Run `python -B tests/test_exploration_contracts.py` locally. Fixtures are explicitly synthetic; they exercise complete-head rejection, optimizer/scheduler, private CPU RNG restoration and corrupted-payload rejection, actual imported graph predictions, fixed query binding, graph margin protection, paired-control override, and source/import/parser entry compatibility. They do not inspect a real checkpoint or images.

On the server after exclusive deployment, use:

```
python -B run_exploration_diagnostic.py --training-root /root/autodl-tmp/AICtrack1_next_cycle_supervision_joint_r3_20261010 --expected-training-source-sha256 e94e36cdea559ca8d424e9c1558e8aea37a6fcccd6af634f55e16c64a73bf524 --dependency-check
```

This imports actual frozen modules, validates signatures and vendor parser/subprocess help only. It does not load a checkpoint/model, probe CUDA or read images. A local equivalent check passes the local R3 source path.

## Actual diagnostic execution (root only)

First establish that actual controller9475/start311656925 and all arm sessions are closed, canonical lock is naturally idle, and both arms exported full20 metrics/last/final. The adapter repeats exact source, launch SHA, closure receipts and CPU metadata/head/optimizer/AC/RNG validation before CUDA. Root must provide each actual last SHA; no future SHA is fabricated in this package.

```
python -B run_exploration_diagnostic.py --training-root /root/autodl-tmp/AICtrack1_next_cycle_supervision_joint_r3_20261010 --expected-training-source-sha256 e94e36cdea559ca8d424e9c1558e8aea37a6fcccd6af634f55e16c64a73bf524 --actual-training-launch /root/autodl-tmp/AICtrack1_next_cycle_supervision_joint_r3_20261010/root_actual_launch.json --expected-training-launch-sha256 3de925785738f2944b09b8e4b119e17d526d7e70c835b0814b355e1ca18ed5bd --arm A3 --expected-last-sha256 ACTUAL_A3_LAST_SHA --hard-seconds 1800 --workers 4 --run
```

After A3 finishes and its inference lease closes, run the same command for A5 with the actual A5 last SHA. Same-checkpoint normalized4V features are observed from the already required four visual forwards and saved automatically, with zero additional model forward.

After both diagnostics complete and canonical is idle, run comparisons serially:

```
python -B compare_graph_vs_J5.py --arm A3
python -B compare_graph_vs_J5.py --arm A5
```

Each comparison verifies actual historical cache/graph/query/source and runtime identities and emits raw-model, direct graph-combination and candidate-increment NPZs plus paired metrics, support, head/mid/tail, supported declines, zero recall and high-margin/connected-group risk/CI evidence. Root reviews results. Balanced direct delta>0 admits at most one independent-confirmation candidate; neither natural>=0 nor positive CI lower bound is an added hard gate. CI crossing0 does not establish a stable gain.

Outputs are exclusive: artifacts/diagnostics/JOINT_A3 and JOINT_A5 and artifacts/comparisons/A3 and A5. Failed output directories and logs are preserved; do not blindly rerun or replace evidence. Official Stage2 train only, legal e88a H1, official OpenAI ViT-B/32, final single-model discipline remain mandatory.
