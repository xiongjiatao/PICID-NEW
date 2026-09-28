"""Export audited per-device test metrics from MultiUnitEvaluator output."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nc_p", "xjtu"), required=True)
    parser.add_argument("--seed", type=int, choices=(72, 88, 101), required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    per_device = {}
    for key, value in metrics.items():
        match = re.fullmatch(r"(.+)_\((\d+),\s*(\d+)\)", key)
        if not match:
            continue
        metric, source, unit = match.group(1), int(match.group(2)), int(match.group(3))
        device = f"DS{source:02d}-unit{unit:02d}" if args.dataset == "nc_p" else f"bearing{source}_{unit}"
        per_device.setdefault(device, {})[metric] = float(value)
    if not per_device:
        raise ValueError("No per-device metrics found")
    if args.dataset == "nc_p" and len(per_device) != 16:
        raise ValueError(f"Expected 16 NC-P test engines, found {len(per_device)}")
    means = {key: float(value) for key, value in metrics.items()
             if key.endswith("_mean")}
    payload = {"dataset": args.dataset, "seed": args.seed,
               "source_protocol_sha256": args.config_sha256,
               "device_count": len(per_device), "aggregation": "equal_device_macro",
               "metrics": means, "per_device": per_device,
               "raw_metrics_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({"dataset": args.dataset, "seed": args.seed,
                      "device_count": len(per_device), "metrics": means}))


if __name__ == "__main__":
    main()
