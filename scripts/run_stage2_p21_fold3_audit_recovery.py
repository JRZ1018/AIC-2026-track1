"""CPU-only audit recovery inside the original fold3 diagnostic deadline."""
import json
import subprocess
import sys
import time
from pathlib import Path

from run_stage2_p21_fold3_diagnostics import (
    ROOT, EXPECTED_ROOT, LOCK, QUEUE, GATE, HARD_SECONDS, diagnostic, sha,
    training_evidence, TrainingLease, probe_running_trainers,
)

RECOVERY = ROOT / "artifacts/queues/20261001-p21-fold3-audit-recovery"


def main():
    if ROOT != EXPECTED_ROOT or (ROOT / "DataSetStage2/test").exists():
        raise RuntimeError("exact train-only root required")
    if LOCK.exists() or probe_running_trainers() or RECOVERY.exists() or GATE.exists():
        raise RuntimeError("existing process, lock, recovery, or gate")
    receipt = json.loads((QUEUE / "source_receipt.json").read_text())
    remaining = receipt["started_unix"] + HARD_SECONDS - time.time()
    if remaining < 30:
        raise RuntimeError("original diagnostic budget exhausted; no reset")
    for p, digest in {**receipt["frozen_sha256"], **receipt["new_entry_sha256"]}.items():
        if sha(ROOT / p) != digest:
            raise RuntimeError("frozen diagnostic source drift: " + p)
    evidence = training_evidence(safe_load=False)
    assets = {str(n): {"report_sha256": sha(diagnostic(n) / "report.json"),
                      "predictions_sha256": sha(diagnostic(n) / "paired_predictions.csv")}
              for n in (14, 20)}
    RECOVERY.mkdir(parents=True, exist_ok=False)
    (RECOVERY / "source_receipt.json").write_text(json.dumps({
        "scope": "CPU-only missing dependency recovery; existing train-only predictions; no GPU/test",
        "original_receipt_sha256": sha(QUEUE / "source_receipt.json"),
        "original_deadline_unix": receipt["started_unix"] + HARD_SECONDS,
        "diagnostics": assets, "checks": evidence["checks"],
        "dependency_sha256": sha(ROOT / "scripts/audit_stage2_p21_schedule_pair.py"),
        "recovery_entry_sha256": sha(Path(__file__)),
    }, indent=2) + "\n")
    lease = TrainingLease.acquire(LOCK, owner=RECOVERY.name, config=str(GATE),
                                  output_dir=str(RECOVERY), command=tuple(sys.argv))
    try:
        with (RECOVERY / "audit.log").open("w") as log:
            child = subprocess.Popen([sys.executable, "scripts/audit_stage2_p21_fold3_schedule_pair.py",
                "--control-dir", str(diagnostic(14)), "--candidate-dir", str(diagnostic(20)),
                "--output", str(GATE)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            lease.attach_trainer(child.pid)
            try:
                code = child.wait(timeout=max(1, receipt["started_unix"] + HARD_SECONDS - time.time()))
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=10)
                raise RuntimeError("original diagnostic deadline reached")
        if code:
            raise RuntimeError("CPU audit recovery failed; preserve outputs")
        gate = json.loads(GATE.read_text())
        (RECOVERY / "result.json").write_text(json.dumps({"status": "complete",
            "recovered_after_missing_dependency": True, "gate_passed": gate["gate_passed"],
            "gate_sha256": sha(GATE), "diagnostics": assets}, indent=2) + "\n")
    finally:
        lease.release()


if __name__ == "__main__":
    main()
