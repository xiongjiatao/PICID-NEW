"""Run validation-only candidates, recording loss without touching test metrics."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def parse_validation_metrics(run_log):
    text = run_log.read_text(errors="replace")
    if "Skipping final test evaluation because test=false" not in text:
        raise RuntimeError("Test-disable evidence missing from runner log")
    found = re.findall(r"Output dir: (.+)", text)
    if not found:
        raise RuntimeError("Missing Hydra output directory in runner log")
    output = Path(found[-1].strip())
    metrics_path = output / "csv_logs/version_0/metrics.csv"
    if not metrics_path.exists():
        candidates = list(output.glob("**/metrics.csv"))
        if not candidates:
            raise FileNotFoundError(metrics_path)
        metrics_path = candidates[0]
    losses = []
    with metrics_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for field in ("val/loss", "val/loss_epoch"):
                value = row.get(field)
                if value not in (None, "", "NaN"):
                    losses.append(float(value))
    if not losses or any(not (value == value and abs(value) != float("inf")) for value in losses):
        raise RuntimeError("Finite validation loss missing")
    with metrics_path.open(newline="") as stream:
        headers = next(csv.reader(stream))
    if any(field.startswith("test/") for field in headers):
        with metrics_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if any(row.get(field) not in (None, "", "NaN") for field in headers if field.startswith("test/") for row in rows):
            raise RuntimeError("Test metric accessed in selection candidate")
    return {"best_val_loss": min(losses), "metrics_csv": str(metrics_path),
            "val_loss_observations": len(losses), "test_fields_populated": False,
            "test_disabled_log_verified": True}


def main():
    sys.path.insert(0, str(ROOT))
    from picid.research.chunks import atomic_json
    from picid.research.protocol import Candidate, freeze_selection

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("nc_p", "xjtu"), required=True)
    parser.add_argument("--model", choices=("lstm", "tabdpt", "tabdpt130", "tabpfn", "xgboost"), required=True)
    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument("--gpu", type=int, choices=(0, 1, 2))
    device_group.add_argument("--cpu", action="store_true")
    parser.add_argument("--expected-peak-mib", type=int)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--cache-base", type=Path,
                        help="Environment-specific cache namespace root")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--reuse", action="append", default=[],
                        help="Existing verified validation run as candidate_key=/path/to/run")
    args = parser.parse_args()
    if args.cpu and args.model != "xgboost":
        parser.error("Only the configured XGBoost baseline supports CPU-only execution")
    if args.gpu is not None and not args.expected_peak_mib:
        parser.error("GPU selection requires --expected-peak-mib")
    records = {}
    expected = set()
    grid = []
    for value in args.reuse:
        key, raw_path = value.split("=", 1)
        run_dir = Path(raw_path)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("exit_code") != 0 or manifest.get("seed") != 72:
            raise ValueError(f"Reuse run not successful seed 72: {run_dir}")
        candidate_metrics = parse_validation_metrics(run_dir / "stdout.log")
        records[key] = {"status": "success", "seed": 72, "test_enabled": False,
                        **candidate_metrics, "output": str(run_dir),
                        "physical_gpu": manifest["physical_gpus"][0]
                        if manifest["physical_gpus"] else None,
                        "logical_device": manifest["logical_device"],
                        "reused_existing_result": True}
    for window in ((1, 10, 50) if args.model == "lstm" else (1, 5, 10, 20, 50)):
        strides = (1,) if args.model == "lstm" else ({1: 1, 5: 1, 10: 5, 20: 5, 50: 50}[window],)
        learning_rates = (0.001, 0.0005, 0.0001) if args.model == "lstm" else (None,)
        for stride in strides:
            for lr in learning_rates:
                candidate = Candidate(args.dataset, args.model, window, stride, lr)
                grid.append(candidate)
                expected.add(candidate.key)
                if candidate.key in records:
                    continue
                run_name = candidate.key + (f"_{args.run_tag}" if args.run_tag else "")
                output = ROOT / "artifacts/formal/selection_runs" / run_name
                if output.exists() and (not output.is_dir() or any(output.iterdir())):
                    raise FileExistsError(output)
                tracker_python = ROOT / ".venv/bin/python"
                if not tracker_python.exists():
                    tracker_python = Path(sys.executable)
                runtime_python = args.runtime_python or tracker_python
                run = [str(tracker_python), "scripts/research/run_tracked.py", "--output", str(output),
                       "--seed", "72", "--stage", "validation_only_selection"]
                if args.gpu is not None:
                    run += ["--gpu", str(args.gpu), "--expected-peak-mib", str(args.expected_peak_mib)]
                run += ["--", str(runtime_python), "picid/run.py",
                        *candidate.overrides(seed=72, test=False), *args.override]
                if args.cache_base:
                    cache_name = "MultiSource_concepts_N-CMAPSS" if args.dataset == "nc_p" else "XJTU-SY"
                    run.append(f"paths.cache_path={args.cache_base.resolve() / cache_name}")
                if args.dataset == "xjtu":
                    run += ["paths.cache_path=/home/xiongjiatao/.aaa-newproj/.aa-alarmllm/picid-release copy/.worktrees/protocol-audit/datasets/cache/XJTU-SY",
                            "datasource.cache_dir=/home/xiongjiatao/.aaa-newproj/.aa-alarmllm/picid-release copy/.worktrees/protocol-audit/datasets/phmd_cache"]
                process = subprocess.run(run, cwd=ROOT, env=dict(os.environ, MPLCONFIGDIR="/tmp/picid-mpl"))
                manifest_path = output / "manifest.json"
                if not manifest_path.exists():
                    raise RuntimeError(f"Missing tracked manifest for {candidate.key}")
                manifest = json.loads(manifest_path.read_text())
                if process.returncode != 0 or manifest["exit_code"] != 0:
                    records[candidate.key] = {"status": "failed", "seed": 72,
                                              "exit_code": manifest.get("exit_code"),
                                              "output": str(output)}
                else:
                    metrics = parse_validation_metrics(output / "stdout.log")
                    records[candidate.key] = {"status": "success", "seed": 72,
                                              "test_enabled": False, **metrics,
                                              "output": str(output),
                                              "physical_gpu": args.gpu,
                                              "logical_device": "cpu" if args.cpu else "cuda:0"}
                atomic_json(ROOT / f"artifacts/formal/{args.dataset}_{args.model}_selection_progress.json",
                            {"updated_utc": datetime.now(timezone.utc).isoformat(),
                             "dataset": args.dataset, "model": args.model, "seed": 72,
                             "candidates": records,
                             "selection_sha256": __import__("hashlib").sha256(
                                 json.dumps(records, sort_keys=True).encode()).hexdigest()})
                if records[candidate.key]["status"] == "failed":
                    raise RuntimeError(f"Candidate failed: {candidate.key}")
    if set(records) != expected:
        raise RuntimeError(f"Selection grid incomplete: missing={sorted(expected - set(records))}")
    frozen = freeze_selection(grid, records)
    atomic_json(ROOT / f"artifacts/formal/{args.dataset}_{args.model}_frozen_seed72.json", frozen)
    atomic_json(ROOT / f"artifacts/formal/{args.dataset}_{args.model}_selection_progress.json",
                {"updated_utc": datetime.now(timezone.utc).isoformat(),
                 "dataset": args.dataset, "model": args.model, "seed": 72,
                 "candidates": records,
                 "selection_sha256": __import__("hashlib").sha256(
                     json.dumps(records, sort_keys=True).encode()).hexdigest()})


if __name__ == "__main__":
    main()
