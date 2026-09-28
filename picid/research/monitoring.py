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


def progress_from_log(path: Path, tail_bytes: int = 65536):
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
                key = f"epoch_{epoch}"
            else:
                completed, total = (int(value) for value in match)
                key = stage
            if total > 0:
                return {"stage": stage, "key": key, "completed": completed, "total": total}
    return None


def eta_seconds(progress: dict, previous: dict | None, now: float | None = None):
    """Estimate remaining time from the latest observed progress increment."""
    now = time.monotonic() if now is None else now
    if progress["completed"] >= progress["total"]:
        return 0.0
    if not previous or previous["key"] != progress["key"]:
        return None
    steps = progress["completed"] - previous["completed"]
    elapsed = now - previous["monotonic"]
    if steps <= 0 or elapsed <= 0:
        return previous.get("eta_seconds")
    seconds_per_step = elapsed / steps
    return (progress["total"] - progress["completed"]) * seconds_per_step
