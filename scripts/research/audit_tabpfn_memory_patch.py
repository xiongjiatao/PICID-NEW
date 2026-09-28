"""Check bounded-NaN predicate semantics on edge, strided and large tensors."""
import argparse
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.overlay.resolve()))
    from tabpfn.architectures.base.transformer import _bounded_has_nan

    cases = []
    tensors = [
        torch.empty(0),
        torch.tensor(1.0),
        torch.arange(2_500_002, dtype=torch.float32).reshape(3, 833_334)[:, ::2],
        torch.arange(2_500_002, dtype=torch.float32).reshape(3, 833_334),
        torch.arange(2_500_002, dtype=torch.float32).reshape(3, 833_334),
    ]
    tensors[3][0, 0] = float("nan")
    tensors[4][-1, -1] = float("nan")
    for i, tensor in enumerate(tensors):
        expected = bool(torch.isnan(tensor).any())
        actual = _bounded_has_nan(tensor, max_elements=1_048_576)
        cases.append({"case": i, "shape": list(tensor.shape), "stride": list(tensor.stride()),
                      "expected": expected, "actual": actual, "equal": expected == actual})
    result = {"status": "passed" if all(row["equal"] for row in cases) else "failed",
              "max_temporary_elements": 1_048_576, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
