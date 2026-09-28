"""Write the complete validation grid and gated final task definitions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from picid.research.protocol import candidates, SEEDS  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = [{**asdict(c), "id": c.key, "seed": 72, "stage": "selection",
              "test_enabled": False, "overrides": c.overrides(),
              "cost_estimate": None, "cost_status": "requires_measured_preflight"}
             for c in candidates()]
    finals = [{"dataset": d, "model": m, "seed": s, "stage": "final",
               "status": "blocked_on_validation_freeze_and_execution_equivalence"}
              for d in ("nc_p", "xjtu")
              for m in ("tabdpt", "tabdpt130", "tabpfn", "xgboost", "lstm") for s in SEEDS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"selection": tasks, "final": finals}, indent=2))
