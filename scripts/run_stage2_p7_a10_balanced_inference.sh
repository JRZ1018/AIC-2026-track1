#!/usr/bin/env bash
set -euo pipefail

cd /srv/aictrack1/AICtrack1-current-20260927

queue=artifacts/queues/20260928-p7-a10-balanced-inference
output=submissions/20260928_p7_sce_full14_fourview_balanced_a10
checkpoint=artifacts/experiments/20260928-stage2-p5-sce-336-full-14ep-seed20260921/final.pt
head=artifacts/experiments/20260928-stage2-p5-sce-h1-336-full-seed20260921/final.pt
test_manifest=artifacts/audits/stage2/test/samples.csv
selection_report=artifacts/reports/20260928-p7-balanced-prior-fold2-fold3.json

assert_sha() {
  local actual
  actual=$(sha256sum "$1")
  actual=${actual%% *}
  if [[ "$actual" != "$2" ]]; then
    printf 'SHA-256 mismatch: %s\n' "$1" >&2
    exit 1
  fi
}

[[ ! -e "$queue" && ! -e "$output" && ! -e artifacts/queues/training.lock.json ]]
[[ -d DataSetStage2/test && -f "$test_manifest" ]]
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '^[0-9]+$'; then
  printf 'Another GPU compute process is running.\n' >&2
  exit 1
fi

assert_sha "$checkpoint" d8938abebada5556dc41a033e3f6b541a031aecffc42fe15ca4aee6f96c65e9f
assert_sha "$head" b17d08319da4a622b29a603cc2b88382cd1b79500b5aac377e4b0b47d54c2a3e
assert_sha "$test_manifest" 576ff0f69e0a104ca5a0af7f190628074b572102c6bc8daf8cb653fb08054201
assert_sha "$selection_report" 8122c9c96fb80823db9d19edb403d8d147d26a32c43b7b48fe7cca9db311d703
assert_sha artifacts/weights/ViT-B-32.pt 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
assert_sha scripts/infer_and_build_submission.py 4d6e03447da0657b88ebf2b191ee524cedca715db22d5c4d965f63adde645e57
assert_sha src/aictrack1/evaluation/balanced_prior.py 49d55dc51d5b9ee8ed2a9d1eff7feeec7aa17eae35ca88a103aeddf4dd82c10b
assert_sha src/aictrack1/evaluation/four_view.py 8f1a6f305cedbfa0a96d276d6af61b9c4413efe358ce1cfdb0639530d2ba946b
assert_sha src/aictrack1/models/heads.py abc37031d8f2de7e052db9892f6f26fd539924fe79e699f59a3a92077c368d55

mkdir -m 700 "$queue"
date -Is > "$queue/started_at.txt"
trap 'printf "%s\n" "$?" > "$queue/exit_code.txt"' EXIT

export LD_LIBRARY_PATH="/usr/local/cuda-12.8/extras/CUPTI/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
.venv/bin/python scripts/infer_and_build_submission.py \
  --experiment-config configs/experiment/stage2_p5_sce_peft_336_full_14ep_a10.yaml \
  --checkpoint "$checkpoint" \
  --test-manifest "$test_manifest" \
  --output-dir "$output" \
  --source-git-commit 1d8249b86d46594bdf30c6828d7dabb889c3a4e4 \
  --batch-size 64 \
  --workers 4 \
  --four-view-scale384 \
  --expected-checkpoint-sha256 d8938abebada5556dc41a033e3f6b541a031aecffc42fe15ca4aee6f96c65e9f \
  --balanced-prior-uniform-soft \
  --balanced-prior-temperature-multiplier 0.5 \
  --balanced-prior-bias-strength 0.75 \
  --balanced-prior-selection-report "$selection_report" \
  --expected-balanced-prior-report-sha256 8122c9c96fb80823db9d19edb403d8d147d26a32c43b7b48fe7cca9db311d703 \
  > "$queue/inference.log" 2>&1
