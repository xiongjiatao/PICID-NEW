"""Extract sampled runtime, host memory, and process-specific GPU memory."""

import argparse
import json
from pathlib import Path


def _gpu_map(gpu_text: str):
    rows = {}
    for line in gpu_text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 5:
            rows[int(fields[0])] = {
                "uuid": fields[1],
                "used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
            }
    return rows


def _tracked_process_gpu_memory(process_text: str, gpu_text: str, pids: set[int]):
    uuid_to_index = {info["uuid"]: index for index, info in _gpu_map(gpu_text).items()}
    memory = {}
    for line in process_text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        pid, gpu_uuid, used_mib = int(fields[0]), fields[1], int(fields[2])
        if pid not in pids or gpu_uuid not in uuid_to_index:
            continue
        index = uuid_to_index[gpu_uuid]
        memory[index] = max(memory.get(index, 0), used_mib)
    return memory


def summarize_run_cost(manifest: dict, telemetry: list[dict]):
    tracked_pid = int(manifest["pid"])
    peak_vram = {}
    peak_host_rss = 0.0
    peak_host_hwm_sum = 0.0
    host_memory_samples = 0
    peak_gpu_util_by_gpu = {}
    last_progress = None
    for event in telemetry:
        tree = event.get("process_tree", {})
        if "rss_mib" in tree:
            host_memory_samples += 1
        peak_host_rss = max(peak_host_rss, float(tree.get("rss_mib", 0.0)))
        peak_host_hwm_sum = max(
            peak_host_hwm_sum, float(tree.get("sum_vm_hwm_mib", 0.0))
        )
        for gpu, memory in _tracked_process_gpu_memory(
            event.get("processes", ""), event.get("gpus", ""), {tracked_pid}
        ).items():
            peak_vram[gpu] = max(peak_vram.get(gpu, 0), memory)
        for gpu, values in _gpu_map(event.get("gpus", "")).items():
            peak_gpu_util_by_gpu[gpu] = max(
                peak_gpu_util_by_gpu.get(gpu, 0), values["utilization_percent"]
            )
        if "stage_progress" in event:
            last_progress = event["stage_progress"]

    return {
        "seed": manifest["seed"],
        "stage": manifest["stage"],
        "status": manifest["status"],
        "exit_code": manifest.get("exit_code"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "physical_gpus": manifest["physical_gpus"],
        "logical_device": manifest["logical_device"],
        "cuda_visible_devices": manifest["cuda_visible_devices"],
        "concurrent_process_counts_at_start": (
            manifest.get("initial_gpu_snapshot") or {}
        ).get("concurrent_process_counts_by_physical_gpu", {}),
        "sampled_peak_tracked_gpu_memory_mib": peak_vram,
        "sampled_peak_host_process_tree_rss_mib": peak_host_rss if host_memory_samples else None,
        "sum_of_process_tree_vm_hwm_mib": peak_host_hwm_sum if host_memory_samples else None,
        "host_memory_sample_count": host_memory_samples,
        "maximum_sampled_gpu_utilization_percent_by_physical_gpu": peak_gpu_util_by_gpu,
        "last_progress_sample": last_progress,
        "telemetry_samples": len(telemetry),
        "command": manifest["shell_command"],
        "output": manifest["output"],
        "source_snapshot_sha256": manifest["source_snapshot"]["sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    telemetry = [json.loads(line) for line in args.telemetry.read_text().splitlines() if line.strip()]
    result = summarize_run_cost(manifest, telemetry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
