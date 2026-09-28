"""Lightweight process-tree and progress sampling for tracked experiments."""

from __future__ import annotations

import os
from pathlib import Path
import re
import time


def _read_proc_stat(pid: int):
    raw = Path(f"/proc/{pid}/stat").read_text()
    tail = raw[raw.rfind(")") + 2 :].split()
    return {
        "ppid": int(tail[1]),
        "cpu_ticks": int(tail[11]) + int(tail[12]),
    }


def _read_proc_memory(pid: int):
    values = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            values["rss_kib"] = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            values["hwm_kib"] = int(line.split()[1])
    return values


def process_tree_stats(root_pid: int) -> dict:
    """Sample aggregate RSS and CPU time for a process and its descendants."""
    records = {}
    children = {}
    try:
        entries = os.scandir("/proc")
    except OSError as exc:
        return {"monitor_error": str(exc)}
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                record = _read_proc_stat(pid)
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
                continue
            records[pid] = record
            children.setdefault(record["ppid"], []).append(pid)

    selected = []
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen or pid not in records:
            continue
        seen.add(pid)
        selected.append(pid)
        stack.extend(children.get(pid, ()))

    rss_kib = 0
    hwm_kib = 0
    for pid in selected:
        try:
            memory = _read_proc_memory(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        rss_kib += memory.get("rss_kib", 0)
        hwm_kib += memory.get("hwm_kib", 0)
    clock_ticks = os.sysconf("SC_CLK_TCK")
    return {
        "process_count": len(selected),
        "rss_mib": rss_kib / 1024,
        "sum_vm_hwm_mib": hwm_kib / 1024,
        "cpu_time_seconds": sum(records[pid]["cpu_ticks"] for pid in selected) / clock_ticks,
    }


def gpu_memory_for_process(gpu_rows: str, process_rows: str, pid: int) -> dict[int, int]:
    """Return sampled VRAM for one process, keyed by physical GPU index."""
    uuid_to_index = {}
    for row in gpu_rows.splitlines():
        fields = [value.strip() for value in row.split(",")]
        if len(fields) >= 2:
            try:
                uuid_to_index[fields[1]] = int(fields[0])
            except ValueError:
                continue

    memory_by_gpu = {}
    for row in process_rows.splitlines():
        fields = [value.strip() for value in row.split(",")]
        if len(fields) != 3 or fields[1] not in uuid_to_index:
            continue
        try:
            row_pid = int(fields[0])
            memory_mib = int(fields[2])
        except ValueError:
            continue
        if row_pid == pid:
            physical_gpu = uuid_to_index[fields[1]]
            memory_by_gpu[physical_gpu] = max(memory_by_gpu.get(physical_gpu, 0), memory_mib)
    return memory_by_gpu


def assert_gpu_memory_budget(free_mib: int, expected_peak_mib: int, reserve_mib: int = 1024):
    """Require the measured free memory to cover a task estimate and explicit reserve."""
    if free_mib < 0 or expected_peak_mib < 1 or reserve_mib < 0:
        raise ValueError("GPU memory values must be nonnegative and peak must be positive")
    required_mib = expected_peak_mib + reserve_mib
    if free_mib < required_mib:
        raise RuntimeError(
            f"Insufficient free GPU memory: need {required_mib} MiB, have {free_mib} MiB"
        )
    return required_mib


def progress_from_log(path: Path, tail_bytes: int = 65536, max_epochs: int | None = None):
    """Read the latest tqdm epoch or ensemble marker without loading full logs."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - tail_bytes))
            content = stream.read().decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return None

    patterns = (
        ("ensemble", re.compile(r"ensembles:\s*\d+%[^\n]*?\b(\d+)\s*/\s*(\d+)", re.IGNORECASE)),
        ("epoch", re.compile(r"Epoch\s+(\d+):\s*\d+%[^\n]*?\b(\d+)\s*/\s*(\d+)", re.IGNORECASE)),
        ("training", re.compile(r"Training:\s*\d+%[^\n]*?\b(\d+)\s*/\s*(\d+)", re.IGNORECASE)),
    )
    for stage, pattern in patterns:
        matches = pattern.findall(content)
        if matches:
            match = matches[-1]
            if stage == "epoch":
                epoch, completed, total = (int(value) for value in match)
                if max_epochs is not None:
                    return {
                        "stage": "training",
                        "key": "training_epochs",
                        "epoch": epoch,
                        "batch_completed": completed,
                        "steps_per_epoch": total,
                        "completed": epoch * total + completed,
                        "total": max_epochs * total,
                    }
                key = f"epoch_{epoch}"
            else:
                completed, total = (int(value) for value in match)
                key = stage
            if total > 0:
                return {"stage": stage, "key": key, "completed": completed, "total": total}
    return None


def progress_rate(progress: dict, previous: dict | None, now: float | None = None):
    """Estimate completed progress steps per second from the last increment."""
    now = time.monotonic() if now is None else now
    if not previous or previous["key"] != progress["key"]:
        return None
    steps = progress["completed"] - previous["completed"]
    elapsed = now - previous["monotonic"]
    if steps <= 0 or elapsed <= 0:
        return previous.get("steps_per_second")
    return steps / elapsed


def eta_seconds(progress: dict, previous: dict | None, now: float | None = None):
    """Estimate remaining time from the latest observed progress increment."""
    now = time.monotonic() if now is None else now
    if progress["completed"] >= progress["total"]:
        return 0.0
    if not previous or previous["key"] != progress["key"]:
        return None
    rate = progress_rate(progress, previous, now)
    if rate is None or rate <= 0:
        return previous.get("eta_seconds")
    return (progress["total"] - progress["completed"]) / rate
