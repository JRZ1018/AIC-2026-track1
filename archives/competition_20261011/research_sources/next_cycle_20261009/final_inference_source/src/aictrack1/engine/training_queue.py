"""Fail-closed single-machine guard for serialized training queues."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, IO, Sequence
from zoneinfo import ZoneInfo


LOCK_SCHEMA_VERSION = 1
DEFAULT_LOCK_RELATIVE_PATH = Path("artifacts/queues/training.lock.json")
TRAINER_ENTRYPOINTS = (
    "train_peft.py",
    "train_n1_peft.py",
    "train_frozen_head.py",
    "train_final_h1_lora.py",
)


class TrainingGuardError(RuntimeError):
    """Base class for queue guard failures that must stop a queue."""


class TrainingLockHeld(TrainingGuardError):
    pass


class TrainingAlreadyRunning(TrainingGuardError):
    pass


class TrainingProcessProbeError(TrainingGuardError):
    pass


@dataclass(frozen=True)
class QueueItem:
    tag: str
    config: str
    output_dir: str
    purpose: str
    trainer_entrypoint: str = "scripts/train_peft.py"
    resume_checkpoint: str | None = None


@dataclass(frozen=True)
class RunningTrainer:
    pid: int
    command_line: str


@dataclass(frozen=True)
class RunOutputState:
    status: str
    reason: str


@dataclass(frozen=True)
class TrainingLockRecord:
    schema_version: int
    token: str
    owner: str
    queue_pid: int
    trainer_pid: int | None
    config: str
    output_dir: str
    started_at: str
    command: tuple[str, ...]


def _lock_timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read_training_lock(path: str | Path) -> TrainingLockRecord | None:
    lock_path = Path(path)
    if not lock_path.exists():
        return None
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        schema_version = int(payload["schema_version"])
        if schema_version != LOCK_SCHEMA_VERSION:
            raise ValueError(f"unsupported lock schema {schema_version}")
        return TrainingLockRecord(
            schema_version=schema_version,
            token=str(payload["token"]),
            owner=str(payload["owner"]),
            queue_pid=int(payload["queue_pid"]),
            trainer_pid=(
                None
                if payload.get("trainer_pid") is None
                else int(payload["trainer_pid"])
            ),
            config=str(payload["config"]),
            output_dir=str(payload["output_dir"]),
            started_at=str(payload["started_at"]),
            command=tuple(str(value) for value in payload["command"]),
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise TrainingLockHeld(
            f"training lock exists but is unreadable: {lock_path}; "
            "do not delete it until the machine has been checked manually"
        ) from exc


def probe_running_trainers() -> list[RunningTrainer]:
    """Return every supported trainer process on the current OS, failing closed."""

    if os.name != "nt":
        try:
            completed = subprocess.run(
                ["ps", "-eo", "pid=,args="],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - defensive OS boundary
            raise TrainingProcessProbeError(
                f"could not inspect running training processes: {exc}"
            ) from exc
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise TrainingProcessProbeError(
                f"training process probe failed with code {completed.returncode}: {message}"
            )
        trainers: list[RunningTrainer] = []
        for line in completed.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split(maxsplit=1)
            if len(fields) != 2:
                continue
            try:
                pid = int(fields[0])
            except ValueError as exc:
                raise TrainingProcessProbeError("process probe returned an invalid PID") from exc
            command_line = fields[1]
            executable = command_line.split(maxsplit=1)[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
            if not executable.startswith("python"):
                continue
            if any(entrypoint in command_line for entrypoint in TRAINER_ENTRYPOINTS):
                trainers.append(RunningTrainer(pid=pid, command_line=command_line))
        return trainers

    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and "
        "$_.CommandLine -and ("
        + " -or ".join(
            f"$_.CommandLine -like '*{entrypoint}*'"
            for entrypoint in TRAINER_ENTRYPOINTS
        )
        + ") } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive OS boundary
        raise TrainingProcessProbeError(
            f"could not inspect running training processes: {exc}"
        ) from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise TrainingProcessProbeError(
            f"training process probe failed with code {completed.returncode}: {message}"
        )
    raw = completed.stdout.strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrainingProcessProbeError("training process probe returned invalid JSON") from exc
    rows = payload if isinstance(payload, list) else [payload]
    trainers: list[RunningTrainer] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TrainingProcessProbeError("training process probe returned invalid rows")
        try:
            trainers.append(
                RunningTrainer(
                    pid=int(row["ProcessId"]),
                    command_line=str(row["CommandLine"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingProcessProbeError(
                "training process probe returned incomplete rows"
            ) from exc
    return trainers


def probe_pid_exists(pid: int) -> bool:
    """Check one Windows PID without signalling or modifying the process."""

    if pid < 1:
        raise ValueError("PID must be positive")
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:  # pragma: no cover - defensive OS boundary
            raise TrainingProcessProbeError(f"could not inspect queue owner PID: {exc}") from exc
        return True
    command = (
        f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) "
        "{ 'exists' } else { 'missing' }"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive OS boundary
        raise TrainingProcessProbeError(f"could not inspect queue owner PID: {exc}") from exc
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise TrainingProcessProbeError(
            f"queue owner PID probe failed with code {completed.returncode}: {message}"
        )
    answer = completed.stdout.strip().lower()
    if answer not in {"exists", "missing"}:
        raise TrainingProcessProbeError("queue owner PID probe returned an invalid result")
    return answer == "exists"


def _write_exclusive(path: Path, record: TrainingLockRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(asdict(record), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _replace_owned_lock(path: Path, record: TrainingLockRecord) -> None:
    current = read_training_lock(path)
    if current is None or current.token != record.token:
        raise TrainingLockHeld("training lock ownership changed unexpectedly")
    temporary = path.with_name(f".{path.name}.{record.token}.tmp")
    temporary.write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


class TrainingLease:
    def __init__(self, path: Path, record: TrainingLockRecord) -> None:
        self.path = path
        self.record = record
        self.released = False

    @classmethod
    def acquire(
        cls,
        path: str | Path,
        *,
        owner: str,
        config: str,
        output_dir: str,
        command: Sequence[str],
        process_probe: Callable[[], list[RunningTrainer]] = probe_running_trainers,
    ) -> "TrainingLease":
        lock_path = Path(path)
        record = TrainingLockRecord(
            schema_version=LOCK_SCHEMA_VERSION,
            token=uuid.uuid4().hex,
            owner=owner,
            queue_pid=os.getpid(),
            trainer_pid=None,
            config=config,
            output_dir=output_dir,
            started_at=_lock_timestamp(),
            command=tuple(str(value) for value in command),
        )
        try:
            _write_exclusive(lock_path, record)
        except FileExistsError as exc:
            existing = read_training_lock(lock_path)
            detail = "unreadable"
            if existing is not None:
                detail = (
                    f"owner={existing.owner!r}, queue_pid={existing.queue_pid}, "
                    f"trainer_pid={existing.trainer_pid}, config={existing.config!r}, "
                    f"started_at={existing.started_at}"
                )
            raise TrainingLockHeld(
                f"training lock already exists ({detail}); stale locks require explicit "
                "manual inspection and token-confirmed cleanup"
            ) from exc
        lease = cls(lock_path, record)
        try:
            running = process_probe()
            if running:
                details = "; ".join(
                    f"pid={item.pid} command={item.command_line[:160]!r}"
                    for item in running
                )
                raise TrainingAlreadyRunning(
                    f"another training process is already running: {details}"
                )
        except Exception:
            lease.release()
            raise
        return lease

    def attach_trainer(self, pid: int) -> None:
        if self.released:
            raise TrainingLockHeld("cannot update a released training lock")
        if pid < 1:
            raise ValueError("trainer PID must be positive")
        updated = TrainingLockRecord(**(asdict(self.record) | {"trainer_pid": pid}))
        _replace_owned_lock(self.path, updated)
        self.record = updated

    def release(self) -> None:
        if self.released:
            return
        current = read_training_lock(self.path)
        if current is None:
            raise TrainingLockHeld("owned training lock disappeared unexpectedly")
        if current.token != self.record.token:
            raise TrainingLockHeld("refusing to remove a training lock owned by another process")
        self.path.unlink()
        self.released = True


def clear_stale_training_lock(
    path: str | Path,
    *,
    confirmed_token: str,
    process_probe: Callable[[], list[RunningTrainer]] = probe_running_trainers,
    owner_probe: Callable[[int], bool] = probe_pid_exists,
) -> TrainingLockRecord:
    """Remove a stale lock only after exact token confirmation and a clean probe."""

    lock_path = Path(path)
    record = read_training_lock(lock_path)
    if record is None:
        raise FileNotFoundError(f"training lock does not exist: {lock_path}")
    if confirmed_token != record.token:
        raise TrainingLockHeld("stale lock confirmation token does not match")
    if owner_probe(record.queue_pid):
        raise TrainingLockHeld(
            f"refusing to clear the lock while queue owner PID {record.queue_pid} exists"
        )
    running = process_probe()
    if running:
        raise TrainingAlreadyRunning(
            "refusing to clear the lock while a training process is still running"
        )
    current = read_training_lock(lock_path)
    if current is None or current.token != record.token:
        raise TrainingLockHeld("training lock changed during stale-lock cleanup")
    lock_path.unlink()
    return record


def inspect_run_output(root: Path, output_dir: str) -> RunOutputState:
    repository_root = root.resolve()
    output = (root / output_dir).resolve()
    if not output.is_relative_to(repository_root):
        return RunOutputState("partial", "output path escapes the repository root")
    if not output.exists():
        return RunOutputState("ready", "output directory does not exist")
    if not output.is_dir():
        return RunOutputState("partial", "output path exists but is not a directory")
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        return RunOutputState("partial", "output directory exists without metrics.json")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RunOutputState("partial", f"metrics.json is unreadable: {exc}")
    selected = metrics.get("selected_checkpoint") if isinstance(metrics, dict) else None
    if not isinstance(selected, str) or not selected or Path(selected).name != selected:
        return RunOutputState("partial", "metrics.json has no safe selected_checkpoint")
    required = (output / "config.yaml", output / "command.txt", output / selected)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return RunOutputState("partial", f"completed output is missing {missing}")
    return RunOutputState("complete", "validated completion artifacts are present")


LaunchFunction = Callable[[Sequence[str], Path, IO[str], Callable[[int], None]], int]
ProcessFactory = Callable[[Sequence[str], Path, IO[str]], Any]


def launch_and_wait(
    command: Sequence[str],
    cwd: Path,
    output: IO[str],
    on_start: Callable[[int], None],
) -> int:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=output,
        stderr=subprocess.STDOUT,
    )
    on_start(process.pid)
    return process.wait()


def start_process(command: Sequence[str], cwd: Path, output: IO[str]) -> subprocess.Popen:
    return subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=output,
        stderr=subprocess.STDOUT,
    )


def _write_parallel_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def run_parallel_queue(
    *,
    root: Path,
    python: Path,
    logs: Path,
    queue_name: str,
    items: Sequence[QueueItem],
    max_concurrent_jobs: int,
    process_probe: Callable[[], list[RunningTrainer]] = probe_running_trainers,
    process_factory: ProcessFactory = start_process,
    lock_path: Path | None = None,
) -> int:
    """Run one explicitly bounded batch under the repository-wide lease.

    This is intentionally not a general scheduler.  Every item is preflighted
    before the lease is acquired, all child PIDs and exit codes are written to
    ``parallel_state.json``, and the lease is retained if child termination is
    ambiguous.  The first PID is also attached to the shared lock; stale-lock
    cleanup still probes the full system for every supported trainer.
    """

    if max_concurrent_jobs < 2:
        raise ValueError("parallel mode requires max_concurrent_jobs >= 2")
    if len(items) > max_concurrent_jobs:
        raise ValueError("parallel batch exceeds max_concurrent_jobs")
    if not items:
        raise ValueError("parallel batch must contain at least one item")
    tags = [item.tag for item in items]
    if len(set(tags)) != len(tags):
        raise ValueError("parallel item tags must be unique")
    outputs = [item.output_dir for item in items]
    if len(set(outputs)) != len(outputs):
        raise ValueError("parallel item output directories must be unique")
    if any(item.trainer_entrypoint != "scripts/train_peft.py" for item in items):
        raise ValueError("parallel queue currently supports only train_peft.py")

    logs.mkdir(parents=True, exist_ok=True)
    effective_lock = lock_path or root / DEFAULT_LOCK_RELATIVE_PATH

    def log(message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with (logs / "chain.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"{queue_name} parallel start; max_concurrent_jobs={max_concurrent_jobs}")
    ready: list[QueueItem] = []
    for item in items:
        before = inspect_run_output(root, item.output_dir)
        if before.status == "complete":
            log(f"{item.tag}: already complete, skipping ({item.purpose})")
            continue
        if before.status != "ready":
            log(
                f"{item.tag}: HALT before launch: {before.reason}; "
                "partial outputs require manual inspection or an explicit resume task"
            )
            return 2
        ready.append(item)
    if not ready:
        log(f"{queue_name} finished; all outputs were already complete")
        return 0

    commands = {
        item.tag: (str(python), "scripts/train_peft.py", "--config", item.config)
        for item in ready
    }
    lease_command = (
        str(python),
        "parallel-training-controller",
        queue_name,
        *(item.config for item in ready),
    )
    try:
        lease = TrainingLease.acquire(
            effective_lock,
            owner=queue_name,
            config=" | ".join(item.config for item in ready),
            output_dir=" | ".join(item.output_dir for item in ready),
            command=lease_command,
            process_probe=process_probe,
        )
    except TrainingGuardError as exc:
        log(f"HALT before parallel launch: {exc}")
        return 2

    state_path = logs / "parallel_state.json"
    state: dict[str, Any] = {
        "schema": "aictrack1.parallel_training_queue.v1",
        "queue_name": queue_name,
        "max_concurrent_jobs": max_concurrent_jobs,
        "started_at": _lock_timestamp(),
        "status": "launching",
        "jobs": {
            item.tag: {
                "config": item.config,
                "output_dir": item.output_dir,
                "log": str((logs / f"{item.tag}.log").relative_to(root)),
                "pid": None,
                "return_code": None,
            }
            for item in ready
        },
    }
    _write_parallel_state(state_path, state)
    processes: dict[str, Any] = {}
    handles: dict[str, IO[str]] = {}
    waits_confirmed = False
    launch_error: Exception | None = None
    release_error: Exception | None = None
    started = time.time()
    try:
        for index, item in enumerate(ready):
            handle = (logs / f"{item.tag}.log").open("w", encoding="utf-8")
            handles[item.tag] = handle
            process = process_factory(commands[item.tag], root, handle)
            processes[item.tag] = process
            if index == 0:
                lease.attach_trainer(int(process.pid))
            state["jobs"][item.tag]["pid"] = int(process.pid)
            log(f"{item.tag}: started pid={process.pid} - {item.purpose}")
            _write_parallel_state(state_path, state)
        state["status"] = "running"
        _write_parallel_state(state_path, state)
        for item in ready:
            return_code = int(processes[item.tag].wait())
            state["jobs"][item.tag]["return_code"] = return_code
            _write_parallel_state(state_path, state)
        waits_confirmed = True
    except Exception as exc:  # pragma: no cover - defensive process boundary
        launch_error = exc
    finally:
        for handle in handles.values():
            handle.close()
        if waits_confirmed or not processes:
            try:
                lease.release()
            except Exception as exc:
                release_error = exc
        else:
            release_error = TrainingLockHeld(
                "one or more parallel trainers were started but every exit was not confirmed; "
                "lock retained"
            )

    elapsed = time.time() - started
    if launch_error is not None or release_error is not None:
        state["status"] = "ambiguous"
        state["error"] = f"launch_error={launch_error!r}; release_error={release_error!r}"
        _write_parallel_state(state_path, state)
        log(
            f"STOPPED after {elapsed / 60:.1f} min; {state['error']}; "
            "manual inspection required"
        )
        return 1

    failed = False
    for item in ready:
        return_code = state["jobs"][item.tag]["return_code"]
        after = inspect_run_output(root, item.output_dir)
        state["jobs"][item.tag]["output_status"] = after.status
        state["jobs"][item.tag]["output_reason"] = after.reason
        if return_code != 0 or after.status != "complete":
            failed = True
            log(
                f"{item.tag}: failed return_code={return_code}; "
                f"output status={after.status}: {after.reason}"
            )
        else:
            log(f"{item.tag}: output validated")
    state["status"] = "failed" if failed else "complete"
    state["finished_at"] = _lock_timestamp()
    state["elapsed_seconds"] = elapsed
    _write_parallel_state(state_path, state)
    log(f"{queue_name} finished in {elapsed / 60:.1f} min; status={state['status']}")
    return 1 if failed else 0


def run_serial_queue(
    *,
    root: Path,
    python: Path,
    logs: Path,
    queue_name: str,
    items: Sequence[QueueItem],
    process_probe: Callable[[], list[RunningTrainer]] = probe_running_trainers,
    launcher: LaunchFunction = launch_and_wait,
    lock_path: Path | None = None,
) -> int:
    """Run a queue under one atomic machine-wide lease, stopping on ambiguity."""

    logs.mkdir(parents=True, exist_ok=True)
    effective_lock = lock_path or root / DEFAULT_LOCK_RELATIVE_PATH

    def log(message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with (logs / "chain.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"{queue_name} start")
    for item in items:
        if item.trainer_entrypoint not in {
            "scripts/train_peft.py",
            "scripts/train_n1_peft.py",
            "scripts/train_frozen_head.py",
        }:
            raise ValueError(f"unsupported serial trainer entrypoint {item.trainer_entrypoint!r}")
        before = inspect_run_output(root, item.output_dir)
        if before.status == "complete":
            log(f"{item.tag}: already complete, skipping ({item.purpose})")
            continue
        resume_path: Path | None = None
        if item.resume_checkpoint is not None:
            output = (root / item.output_dir).resolve()
            resume_path = (root / item.resume_checkpoint).resolve()
            if (
                item.trainer_entrypoint
                not in {"scripts/train_peft.py", "scripts/train_n1_peft.py", "scripts/train_frozen_head.py"}
                or before.status != "partial"
                or not output.is_relative_to(root.resolve())
                or resume_path != output / "last.pt"
                or not resume_path.is_file()
                or (output / "metrics.json").exists()
            ):
                log(
                    f"{item.tag}: HALT before launch: explicit resume requires "
                    "a partial supported trainer output with its exact last.pt and no metrics.json"
                )
                return 2
        elif before.status != "ready":
            log(
                f"{item.tag}: HALT before launch: {before.reason}; "
                "partial outputs require manual inspection or an explicit resume task"
            )
            return 2

        command = (str(python), item.trainer_entrypoint, "--config", item.config)
        if resume_path is not None:
            command += ("--resume", str(resume_path))
        try:
            lease = TrainingLease.acquire(
                effective_lock,
                owner=queue_name,
                config=item.config,
                output_dir=item.output_dir,
                command=command,
                process_probe=process_probe,
            )
        except TrainingGuardError as exc:
            log(f"{item.tag}: HALT before launch: {exc}")
            return 2

        log_path = logs / f"{item.tag}.log"
        log(f"{item.tag}: starting - {item.purpose}")
        started = time.time()
        return_code: int | None = None
        launch_completed = False
        launch_error: Exception | None = None
        release_error: Exception | None = None
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                return_code = launcher(command, root, handle, lease.attach_trainer)
                launch_completed = True
        except Exception as exc:  # pragma: no cover - exercised with injected launcher
            launch_error = exc
        finally:
            if launch_completed or lease.record.trainer_pid is None:
                try:
                    lease.release()
                except Exception as exc:  # keep the lock if ownership changed
                    release_error = exc
            else:
                release_error = TrainingLockHeld(
                    "trainer was started but its exit was not confirmed; lock retained"
                )
        elapsed = time.time() - started
        after = inspect_run_output(root, item.output_dir)
        if launch_error is not None or release_error is not None:
            log(
                f"{item.tag}: STOPPED after {elapsed / 60:.1f} min; "
                f"launch_error={launch_error!r}, release_error={release_error!r}; "
                "manual inspection required"
            )
            return 1
        if return_code != 0 or after.status != "complete":
            log(
                f"{item.tag}: STOPPED after {elapsed / 60:.1f} min with return code "
                f"{return_code}; output status={after.status}: {after.reason}; "
                f"halting queue. Log: {log_path}"
            )
            return 1
        log(f"{item.tag}: finished in {elapsed / 60:.1f} min; output validated")
    log(f"{queue_name} finished")
    return 0
