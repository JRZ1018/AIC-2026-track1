# New unused-query train-only H1100 template

Status: CPU engineering only. No remote deployment, no GPU invocation, no new H1 checkpoint. Official Stage2 train only.

The frozen pre-result query source is 51a79ffbaf94fbb333cdab49123e55ccc3fb5cf2fe78cfdb6b5ea86da8810e50, split610111665e1055bcda7157add15b16ade353138365e54ee87dc92316e6dfd6f1. 118958 train/29737 query, both sides750 classes. No old0c9 or57189 query group overlaps. Exact metadata copies are unchanged. The official37 CLIP336 feature shards are immutable external inputs and only partitions/new split metadata are rebound after positive all148695 ID-label-index-path checks.

H1 is a new linear750x512 head, official frozen features, CE, fixed100 epochs, Adam lr.01/batch4096, seed20261008/constant scheduler/no AMP. No query metric during fitting; no best selection. The terminal query parity is only after fixed100 and is not used to alter training. Actual complete model/optimizer/scheduler/scaler/RNG restoration is performed before the H1 receipt. A future H1 SHA remains null here. Later visual control J5alpha.1 and one alpha winner must use this actual new H1 and common visual seed20261010; this template does not authorize those stages.

The original FrozenHeadTrainer and canonical training queue are exact copied source bytes. The H1 script changes only the run identity from prior train-only producer; the reviewed owned_launch GO lifecycle is exact copied. The new gate only execs train_frozen_head after actual canonical trainer attachment; unknown group closure fails closed and retains the lease. Never clear a lock or restart a living process.

CPU local/remote preflight (no fitting/image decode):

```text
python -B scripts/cpu_feature_preflight.py --manifest <actual official samples.csv> --feature-cache <actual all37 official cache> --receipt <new external receipt.json>
```

Root performs `scripts/bind_confirmation_h1.py --help` to see required actual paths. It requires an actual positive single-A3/A5 root admission and a new explicit runtime budget, and creates only the NEW exact execution root `/root/autodl-tmp/AICtrack1_next_cycle_confirmation_h1_20261010`. The template and old sources are not modified. No deadline/start or future model SHA is stored here. Admission schema `aictrack1.next_cycle10.h1_admission.v1`: root_authorized true, complete true, selected_arm A3|A5, exploration_balanced_delta_pp>0, actual exploration_paired_sha256 and root_risk_review_sha256, exact query_definition_source_sha256, new_H1_checkpoint_sha256 null. Budget schema `aictrack1.next_cycle10.confirmation_budget.v1`: root_authorized true, phase_start_unix, deadline_unix, adjusted_initial_free_bytes, disk_max_bytes, min_free_bytes>=3GiB, original_initial_free_bytes13013950464/capacity_added_bytes32212254720/adjusted_initial_free_bytes45226205184/actual_disk_total_bytes128849018880; disk_max_bytes<=20GiB. Root supplies actual clock values; copying an old clock/baseline is prohibited.

After binding and exact prior groups closed/canonical idle:

```text
python -B scripts/run_confirmation_h1.py --preflight
python -B scripts/run_confirmation_h1.py
```

Resume only with the exact last, same source/config/row binding: `--resume`. H1 fixed100 output is `artifacts/experiments/confirmation_H1/last.pt`, receipt and metrics. No fullH1/e88a/model swap, test input, extra weights, or GPU worker is created by local preparation. Real Linux deployment/GO/CUDA training remain root-owned and unexecuted.
