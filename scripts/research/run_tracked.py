"""Run a single command with GPU admission, immutable identity and live telemetry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from picid.research.chunks import atomic_json
from picid.research.monitoring import (
    eta_seconds,
    gpu_memory_for_process,
    process_tree_stats,
    progress_from_log,
    progress_rate,
)
from picid.research.protocol import PHYSICAL_GPUS


def source_snapshot(root):
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    suffixes = {".py", ".yaml", ".yml", ".toml", ".lock", ".patch", ".json", ".md"}
    digest = hashlib.sha256()
    count = 0
    for item in sorted(Path(os.fsdecode(path)) for path in raw.split(b"\0") if path):
        if item.suffix not in suffixes:
            continue
        full = root / item
        if not full.is_file():
            continue
        digest.update(item.as_posix().encode())
        digest.update(b"\0")
        with full.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
        count += 1
    return {"sha256": digest.hexdigest(), "source_file_count": count,
            "includes_untracked_source": True}


def gpu_snapshot():
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args], text=True).strip()
    devices = query(["--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    processes = query(["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
    rows = [row.split(",") for row in devices.splitlines()]
    uuid_to_index = {row[1].strip(): int(row[0]) for row in rows}
    counts, peaks = {}, {}
    for row in processes.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 3 and fields[1] in uuid_to_index:
            index = uuid_to_index[fields[1]]
            counts[index] = counts.get(index, 0) + 1
            try:
                peaks[index] = max(peaks.get(index, 0), int(fields[2]))
            except ValueError:
                pass
    return {"gpus": devices, "processes": processes,
            "concurrent_process_counts_by_physical_gpu": counts,
            "observed_process_memory_mib_by_physical_gpu": peaks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=PHYSICAL_GPUS)
    parser.add_argument("--expected-peak-mib", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Command required")
    if args.output.exists():
        if not args.output.is_dir() or any(args.output.iterdir()):
            raise FileExistsError(args.output)
    else:
        args.output.mkdir(parents=True)
    # Record device occupancy even for CPU tasks so concurrent GPU work is visible
    # in every experiment manifest and heartbeat.
    try:
        initial = gpu_snapshot()
    except subprocess.CalledProcessError as exc:
        failure = {
            "command": command,
            "shell_command": shlex.join(command),
            "cwd": str(Path.cwd()),
            "seed": args.seed,
            "stage": args.stage,
            "status": "preflight_monitor_failed",
            "physical_gpus": [] if args.gpu is None else [args.gpu],
            "gpu_assignment": "cpu_only" if args.gpu is None else "single_physical_gpu",
            "logical_device": "cpu" if args.gpu is None else "cuda:0",
            "cuda_visible_devices": "" if args.gpu is None else str(args.gpu),
            "expected_peak_mib": args.expected_peak_mib,
            "initial_gpu_snapshot_error": str(exc),
            "exit_code": None,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
            "source_snapshot": source_snapshot(Path.cwd()),
            "output": str(args.output.resolve()),
        }
        atomic_json(args.output / "manifest.json", failure)
        raise RuntimeError("GPU occupancy preflight failed; model command was not launched") from exc
    if args.gpu is not None:
        if not args.expected_peak_mib or args.expected_peak_mib < 1:
            parser.error("GPU task requires an expected peak estimate")
        free = {int(row.split(",")[0].strip()): int(row.split(",")[2].strip())
                for row in initial["gpus"].splitlines()}
        if free[args.gpu] < args.expected_peak_mib + 1024:
            raise RuntimeError("Insufficient free GPU memory plus 1 GiB reserve")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if args.gpu is None else str(args.gpu))
    manifest = {"command": command, "shell_command": shlex.join(command), "cwd": str(Path.cwd()),
                "seed": args.seed, "stage": args.stage, "status": "running",
                "physical_gpus": [] if args.gpu is None else [args.gpu],
                "gpu_assignment": "cpu_only" if args.gpu is None else "single_physical_gpu",
                "logical_device": "cpu" if args.gpu is None else "cuda:0",
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "expected_peak_mib": args.expected_peak_mib, "initial_gpu_snapshot": initial,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
                "source_snapshot": source_snapshot(Path.cwd()),
                "output": str(args.output.resolve())}
    path = args.output / "manifest.json"
    atomic_json(path, manifest)
    started = time.monotonic()
    peaks = {}
    tracked_process_peaks = {}
    peak_process_tree_rss = 0.0
    peak_process_tree_hwm_sum = 0.0
    last_progress = None
    max_epochs = next(
        (int(value.split("=", 1)[1]) for value in command
         if value.startswith("trainer.max_epochs=")),
        None,
    )
    with (args.output / "stdout.log").open("w") as log, (args.output / "telemetry.jsonl").open("w") as telemetry:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest["pid"] = process.pid
        atomic_json(path, manifest)
        while process.poll() is None:
            event = {"elapsed_seconds": time.monotonic() - started, "pid": process.pid,
                     "log_bytes": log.tell()}
            process_stats = process_tree_stats(process.pid)
            event["process_tree"] = process_stats
            peak_process_tree_rss = max(peak_process_tree_rss, process_stats.get("rss_mib", 0.0))
            peak_process_tree_hwm_sum = max(
                peak_process_tree_hwm_sum, process_stats.get("sum_vm_hwm_mib", 0.0)
            )
            progress = progress_from_log(args.output / "stdout.log", max_epochs=max_epochs)
            if progress is not None:
                now_progress = time.monotonic()
                progress["steps_per_second"] = progress_rate(progress, last_progress, now_progress)
                progress["eta_seconds"] = eta_seconds(progress, last_progress, now_progress)
                event["stage_progress"] = progress
                if (last_progress is None or last_progress["key"] != progress["key"]
                        or progress["completed"] > last_progress["completed"]):
                    last_progress = {
                        "key": progress["key"],
                        "completed": progress["completed"],
                        "total": progress["total"],
                        "monotonic": now_progress,
                        "eta_seconds": progress["eta_seconds"],
                        "steps_per_second": progress["steps_per_second"],
                    }
                else:
                    progress["eta_seconds"] = last_progress.get("eta_seconds")
                    progress["steps_per_second"] = last_progress.get("steps_per_second")
            try:
                event.update(gpu_snapshot())
                tracked_memory = gpu_memory_for_process(
                    event["gpus"], event["processes"], process.pid
                )
                event["tracked_process_gpu_memory_mib"] = tracked_memory
                for gpu, memory in tracked_memory.items():
                    tracked_process_peaks[str(gpu)] = max(
                        tracked_process_peaks.get(str(gpu), 0), memory
                    )
                if args.gpu is not None:
                    for gpu, memory in event.get("observed_process_memory_mib_by_physical_gpu", {}).items():
                        peaks[gpu] = max(peaks.get(gpu, 0), memory)
            except subprocess.CalledProcessError as exc:
                event["gpu_diagnostic_error"] = str(exc)
            telemetry.write(json.dumps(event) + "\n")
            telemetry.flush()
            atomic_json(args.output / "heartbeat.json", event)
            time.sleep(10)
    manifest.update(exit_code=process.returncode, elapsed_seconds=time.monotonic() - started,
                    observed_peak_process_memory_mib_by_physical_gpu=peaks,
                    observed_peak_tracked_process_gpu_memory_mib_by_physical_gpu=tracked_process_peaks,
                    observed_peak_process_tree_rss_mib=peak_process_tree_rss,
                    observed_peak_process_tree_vm_hwm_sum_mib=peak_process_tree_hwm_sum,
                    status="process_success_requires_result_audit" if process.returncode == 0 else "failed")
    atomic_json(path, manifest)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
