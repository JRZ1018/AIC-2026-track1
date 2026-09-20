from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aictrack1.engine.training_queue import (
    QueueItem,
    RunningTrainer,
    TrainingAlreadyRunning,
    TrainingLease,
    TrainingLockHeld,
    TrainingProcessProbeError,
    clear_stale_training_lock,
    inspect_run_output,
    probe_pid_exists,
    probe_running_trainers,
    read_training_lock,
    run_serial_queue,
)


ROOT = Path(__file__).resolve().parents[1]
QUEUE_SCRIPTS = (
    "run_queue_0918.py",
    "run_queue_0918_night.py",
    "run_queue_0919_batch2.py",
    "run_queue_0919_batch3.py",
)


def _item(tag: str = "A", output: str = "artifacts/experiments/run-a") -> QueueItem:
    return QueueItem(
        tag=tag,
        config=f"configs/{tag.lower()}.yaml",
        output_dir=output,
        purpose=f"synthetic {tag}",
    )


def _write_complete_output(root: Path, item: QueueItem) -> None:
    output = root / item.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps({"selected_checkpoint": "best.pt"}),
        encoding="utf-8",
    )
    (output / "config.yaml").write_text("experiment: {}\n", encoding="utf-8")
    (output / "command.txt").write_text("synthetic\n", encoding="utf-8")
    (output / "best.pt").write_bytes(b"synthetic checkpoint")


def test_training_lease_is_atomic_and_records_owner_and_trainer_pid(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "training.lock.json"
    lease = TrainingLease.acquire(
        lock,
        owner="queue-a",
        config="configs/a.yaml",
        output_dir="artifacts/a",
        command=("python", "scripts/train_peft.py", "--config", "configs/a.yaml"),
        process_probe=lambda: [],
    )
    record = read_training_lock(lock)
    assert record is not None
    assert record.owner == "queue-a"
    assert record.queue_pid > 0
    assert record.trainer_pid is None
    assert record.config == "configs/a.yaml"
    assert record.output_dir == "artifacts/a"
    assert record.started_at

    with pytest.raises(TrainingLockHeld, match="already exists"):
        TrainingLease.acquire(
            lock,
            owner="queue-b",
            config="configs/b.yaml",
            output_dir="artifacts/b",
            command=("python", "scripts/train_peft.py"),
            process_probe=lambda: [],
        )

    lease.attach_trainer(4321)
    updated = read_training_lock(lock)
    assert updated is not None and updated.trainer_pid == 4321
    assert updated.token == record.token
    lease.release()
    assert not lock.exists()


def test_external_training_process_blocks_launch_and_releases_probe_lease(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "training.lock.json"
    running = [RunningTrainer(99, "python scripts/train_peft.py --config external.yaml")]
    with pytest.raises(TrainingAlreadyRunning, match="pid=99"):
        TrainingLease.acquire(
            lock,
            owner="queue-a",
            config="configs/a.yaml",
            output_dir="artifacts/a",
            command=("python", "scripts/train_peft.py"),
            process_probe=lambda: running,
        )
    assert not lock.exists()


def test_process_probe_failure_is_fail_closed(tmp_path: Path) -> None:
    lock = tmp_path / "training.lock.json"

    def failed_probe() -> list[RunningTrainer]:
        raise TrainingProcessProbeError("CIM unavailable")

    with pytest.raises(TrainingProcessProbeError, match="CIM unavailable"):
        TrainingLease.acquire(
            lock,
            owner="queue-a",
            config="configs/a.yaml",
            output_dir="artifacts/a",
            command=("python", "scripts/train_peft.py"),
            process_probe=failed_probe,
        )
    assert not lock.exists()


def test_windows_process_probe_parses_pid_and_restricts_query_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "ProcessId": 123,
                    "CommandLine": "python scripts/train_peft.py --config x.yaml",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("aictrack1.engine.training_queue.subprocess.run", fake_run)
    assert probe_running_trainers() == [
        RunningTrainer(123, "python scripts/train_peft.py --config x.yaml")
    ]
    powershell = captured[0][-1]
    assert "python.exe" in powershell
    assert "pythonw.exe" in powershell
    assert "train_peft.py" in powershell


def test_windows_process_probe_rejects_command_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aictrack1.engine.training_queue.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="access denied",
        ),
    )
    with pytest.raises(TrainingProcessProbeError, match="access denied"):
        probe_running_trainers()


def test_windows_owner_pid_probe_is_read_only_and_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout="exists\n", stderr="")

    monkeypatch.setattr("aictrack1.engine.training_queue.subprocess.run", fake_run)
    assert probe_pid_exists(321) is True
    assert "Get-Process -Id 321" in captured[0][-1]
    assert "Stop-Process" not in captured[0][-1]


def test_stale_lock_requires_exact_token_and_no_running_trainer(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "training.lock.json"
    lease = TrainingLease.acquire(
        lock,
        owner="crashed-queue",
        config="configs/a.yaml",
        output_dir="artifacts/a",
        command=("python", "scripts/train_peft.py"),
        process_probe=lambda: [],
    )
    token = lease.record.token
    with pytest.raises(TrainingLockHeld, match="token does not match"):
        clear_stale_training_lock(
            lock,
            confirmed_token="wrong-token",
            process_probe=lambda: [],
            owner_probe=lambda _pid: False,
        )
    with pytest.raises(TrainingLockHeld, match="queue owner PID"):
        clear_stale_training_lock(
            lock,
            confirmed_token=token,
            process_probe=lambda: [],
            owner_probe=lambda _pid: True,
        )
    with pytest.raises(TrainingAlreadyRunning, match="still running"):
        clear_stale_training_lock(
            lock,
            confirmed_token=token,
            process_probe=lambda: [RunningTrainer(55, "python train_peft.py")],
            owner_probe=lambda _pid: False,
        )
    removed = clear_stale_training_lock(
        lock,
        confirmed_token=token,
        process_probe=lambda: [],
        owner_probe=lambda _pid: False,
    )
    assert removed.owner == "crashed-queue"
    assert not lock.exists()


def test_output_state_distinguishes_ready_partial_and_complete(tmp_path: Path) -> None:
    item = _item()
    assert inspect_run_output(tmp_path, item.output_dir).status == "ready"
    (tmp_path / item.output_dir).mkdir(parents=True)
    partial = inspect_run_output(tmp_path, item.output_dir)
    assert partial.status == "partial"
    assert "without metrics" in partial.reason
    _write_complete_output(tmp_path, item)
    assert inspect_run_output(tmp_path, item.output_dir).status == "complete"
    assert inspect_run_output(tmp_path, "../escape").status == "partial"


def test_queue_skips_validated_completed_run_without_probe_or_launch(
    tmp_path: Path,
) -> None:
    item = _item()
    _write_complete_output(tmp_path, item)

    def forbidden_probe() -> list[RunningTrainer]:
        raise AssertionError("completed runs must not probe or acquire the lock")

    def forbidden_launcher(*_args):
        raise AssertionError("completed runs must not launch")

    result = run_serial_queue(
        root=tmp_path,
        python=Path("python"),
        logs=tmp_path / "logs",
        queue_name="synthetic-queue",
        items=[item],
        process_probe=forbidden_probe,
        launcher=forbidden_launcher,
        lock_path=tmp_path / "training.lock.json",
    )
    assert result == 0
    assert "already complete" in (tmp_path / "logs/chain.log").read_text(
        encoding="utf-8"
    )


def test_queue_halts_on_partial_output_without_launch(tmp_path: Path) -> None:
    item = _item()
    (tmp_path / item.output_dir).mkdir(parents=True)
    launched = False

    def launcher(*_args):
        nonlocal launched
        launched = True
        return 0

    result = run_serial_queue(
        root=tmp_path,
        python=Path("python"),
        logs=tmp_path / "logs",
        queue_name="synthetic-queue",
        items=[item],
        process_probe=lambda: [],
        launcher=launcher,
        lock_path=tmp_path / "training.lock.json",
    )
    assert result == 2
    assert launched is False
    assert "partial outputs require manual inspection" in (
        tmp_path / "logs/chain.log"
    ).read_text(encoding="utf-8")


def test_queue_halts_after_abnormal_exit_and_does_not_advance(
    tmp_path: Path,
) -> None:
    first = _item("A", "artifacts/experiments/run-a")
    second = _item("B", "artifacts/experiments/run-b")
    launched: list[str] = []

    def failing_launcher(command, root, _handle, on_start) -> int:
        launched.append(str(command[-1]))
        on_start(777)
        (root / first.output_dir).mkdir(parents=True)
        return 9

    lock = tmp_path / "training.lock.json"
    result = run_serial_queue(
        root=tmp_path,
        python=Path("python"),
        logs=tmp_path / "logs",
        queue_name="synthetic-queue",
        items=[first, second],
        process_probe=lambda: [],
        launcher=failing_launcher,
        lock_path=lock,
    )
    assert result == 1
    assert launched == [first.config]
    assert not (tmp_path / second.output_dir).exists()
    assert not lock.exists()


def test_queue_retains_lock_when_started_trainer_exit_is_unknown(
    tmp_path: Path,
) -> None:
    item = _item()

    def ambiguous_launcher(_command, _root, _handle, on_start) -> int:
        on_start(999)
        raise RuntimeError("lost contact with child")

    lock = tmp_path / "training.lock.json"
    result = run_serial_queue(
        root=tmp_path,
        python=Path("python"),
        logs=tmp_path / "logs",
        queue_name="synthetic-queue",
        items=[item],
        process_probe=lambda: [],
        launcher=ambiguous_launcher,
        lock_path=lock,
    )
    assert result == 1
    record = read_training_lock(lock)
    assert record is not None and record.trainer_pid == 999
    clear_stale_training_lock(
        lock,
        confirmed_token=record.token,
        process_probe=lambda: [],
        owner_probe=lambda _pid: False,
    )


def test_queue_success_requires_validated_artifacts(tmp_path: Path) -> None:
    item = _item()

    def successful_launcher(_command, root, _handle, on_start) -> int:
        on_start(888)
        _write_complete_output(root, item)
        return 0

    result = run_serial_queue(
        root=tmp_path,
        python=Path("python"),
        logs=tmp_path / "logs",
        queue_name="synthetic-queue",
        items=[item],
        process_probe=lambda: [],
        launcher=successful_launcher,
        lock_path=tmp_path / "training.lock.json",
    )
    assert result == 0
    assert "output validated" in (tmp_path / "logs/chain.log").read_text(
        encoding="utf-8"
    )


def test_all_historical_queue_entries_use_the_shared_guard() -> None:
    for name in QUEUE_SCRIPTS:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "run_serial_queue(" in source
        assert "Get-CimInstance" not in source
        assert "subprocess.run(" not in source
        assert "time.sleep(" not in source
