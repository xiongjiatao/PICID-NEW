"""Read-only protocol inspection; writes reports only to the selected output directory.

Run with the locked PICID Python environment. This does not load data, instantiate
models, or run training. A blocked result is not a failed performance experiment.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [72, 88, 101, 666, 226688]
PAIRS = [(1, 1), (5, 1), (10, 5), (20, 5), (50, 50)]
MODELS = ["tabpfn_fit_predict", "tabdpt_fit_predict", "xgboost_fit_predict", "lstm"]
SKIP = {
    ".git",
    ".worktrees",
    ".venv",
    ".cache",
    "__pycache__",
    ".pytest_tmp",
    "artifacts",
    "research_outputs",
    ".pytest_cache",
    ".ruff_cache",
    ".model_cache",
    ".nox",
}
ROOT_SKIP = {"datasets", "checkpoints", "logs", "results"}


def snapshot(root):
    records = {}
    for folder, directories, files in os.walk(root):
        ignored = SKIP | ROOT_SKIP if Path(folder) == root else SKIP
        directories[:] = sorted(d for d in directories if d not in ignored)
        for filename in sorted(files):
            if filename in SKIP:
                continue
            path = Path(folder) / filename
            if path.is_symlink():
                value = "symlink:" + os.readlink(path)
            else:
                value = hashlib.sha256(path.read_bytes()).hexdigest()
            records[str(path.relative_to(root))] = value
    return records


def aggregation_contract(root):
    tree = ast.parse(
        (root / "picid/transforms/base_transforms/subsample.py").read_text()
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "WindowedAggregationTransform"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    parameters = {n.arg for n in init.args.args + init.args.kwonlyargs}
    return {
        "parameters": sorted(parameters),
        "aggregation_alias_supported": "aggregation" in parameters,
    }


def inspect_config(cfg, contract):
    problems = []
    for name, entry in (cfg.get("transforms") or {}).items():
        transform = entry.get("transform", {}) if isinstance(entry, dict) else {}
        if transform.get("_target_", "").endswith("WindowedAggregationTransform"):
            requested = transform.get("aggregation")
            effective = transform.get("agg", "mean")
            if (
                requested is not None
                and not contract["aggregation_alias_supported"]
                and requested != effective
            ):
                problems.append(
                    {
                        "code": "IGNORED_AGGREGATION",
                        "transform": name,
                        "requested": requested,
                        "effective": effective,
                    }
                )
    return problems


def resolve_aggregation_fields(cfg, protocol):
    """Replace legacy YAML key names with an explicit executable ``agg``.

    ``released_behavior_legacy`` reproduces the pre-fix constructor, where
    ``aggregation`` was swallowed by **kwargs and therefore defaulted to mean.
    ``paper_intent_fixed`` applies the aggregation requested by the YAML.
    """
    resolutions = []
    for name, entry in (cfg.get("transforms") or {}).items():
        transform = entry.get("transform", {}) if isinstance(entry, dict) else {}
        if not transform.get("_target_", "").endswith("WindowedAggregationTransform"):
            continue
        declared = transform.get("aggregation")
        explicit = transform.get("agg")
        if declared is not None and explicit is not None and declared != explicit:
            raise ValueError(
                f"Conflicting aggregation values in {name}: "
                f"aggregation={declared!r}, agg={explicit!r}"
            )
        requested = declared if declared is not None else explicit
        if requested is None:
            requested = "mean"
        effective = (
            requested
            if protocol == "paper_intent_fixed"
            else (explicit if explicit is not None else "mean")
        )
        if "aggregation" in transform:
            del transform["aggregation"]
        transform["agg"] = effective
        resolutions.append(
            {
                "transform": name,
                "requested": requested,
                "effective": effective,
                "protocol": protocol,
            }
        )
    return resolutions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=["paper_intent_fixed", "released_behavior_legacy"],
        default="paper_intent_fixed",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.expanduser().resolve()
    paper_specs = [
        (
            "From_paper_to_benchmark(agentic,_framework-based).pdf",
            "2605.28371",
            "workflow",
        ),
        (
            "Picid(A_Modular_Evaluation_Infrastructure_for_Reproducible_PHM_Across_Tasks_and_Domains).pdf",
            "2605.28345",
            "contracts",
        ),
        (
            "Towards Unified and Data-Efficient Prognostics and Health Management with Tabular Foundation Models.pdf",
            "2606.05481",
            "experiments",
        ),
    ]
    references = []
    for filename, arxiv_id, role in paper_specs:
        path = args.baseline.parent / filename
        references.append(
            {
                "filename": filename,
                "arxiv": arxiv_id,
                "role": role,
                "present": path.is_file(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None,
            }
        )
    (args.output / "required_references.json").write_text(
        json.dumps(references, indent=2)
    )
    baseline = snapshot(args.baseline)
    baseline_file = args.output / "baseline_sha256.json"
    if baseline_file.exists() and json.loads(baseline_file.read_text()) != baseline:
        raise RuntimeError(
            "Read-only baseline differs from the previously recorded manifest"
        )
    baseline_file.write_text(json.dumps(baseline, indent=2))
    working = snapshot(ROOT)
    (args.output / "working_sha256.json").write_text(json.dumps(working, indent=2))
    difference = {
        "added": sorted(working.keys() - baseline.keys()),
        "absent": sorted(baseline.keys() - working.keys()),
        "changed": sorted(
            k for k in baseline.keys() & working.keys() if baseline[k] != working[k]
        ),
    }
    contract = aggregation_contract(ROOT)
    report = {
        "status": "AUDIT_ONLY",
        "required_references": references,
        "protocol": args.protocol,
        "source_difference": difference,
        "seeds": SEEDS,
        "window_stride_pairs": PAIRS,
        "models": MODELS,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "aggregation_contract": contract,
        "n_cmapss_scaler_contract": {
            "tfm_appendix_a": "min-max wording",
            "tfm_detailed_appendix_c_2_3": "fixed standard scalers",
            "picid_appendix_f_3": "benchmark-defined constants, not fitted on current split",
            "checked_code": "fixed hard-coded standard means/stds in N_CMAPSSFeaturesScaler and N_CMAPSSDescriptorsScaler",
            "constant_source_population": "not specified in paper or code; retain values and report train-only-stat comparison separately",
        },
        "xjtu_target_contract": {
            "phmd_task": "rul",
            "raw_grain": "32768 vibration samples per acquisition record",
            "source_rul_direction": "decreases to zero at failure",
            "health_index": "RUL / N acquisitions; PHMD RUL trace is N-1,...,0",
            "paper_equivalence": "1 - one-based elapsed acquisitions / N; zero-based elapsed differs by 1/N",
            "inverse_metric_unit": "remaining acquisition-minutes",
        },
        "data_preflight": {},
        "experiments": [],
        "missing_n_cmapss": [],
        "review_required": [
            "N-CMAPSS fixed-scaler derivation population is undocumented even though values match NC-P train statistics closely",
            "Model checkpoint/runtime replay on the PHMD and NC-P loaders",
            "Runtime causality and cache equivalence tests",
        ],
    }
    for ds in ("01", "04", "05", "07"):
        path = data_root / "N-CMAPSS" / f"N-CMAPSS_DS{ds}.h5"
        if not path.is_file():
            report["missing_n_cmapss"].append(str(path))
    audit_artifacts = data_root.parent / "artifacts"
    scaler_audit_path = audit_artifacts / "ncmapss_scaler_audit.json"
    xjtu_audit_path = audit_artifacts / "xjtu_phmd_payload_audit.json"
    report["data_preflight"] = {
        "ncmapss_files_present": not report["missing_n_cmapss"],
        "ncmapss_scaler_audit": {
            "path": str(scaler_audit_path),
            "status": None,
        },
        "xjtu_payload_audit": {"path": str(xjtu_audit_path), "status": None},
    }
    for key, path, expected_status in (
        ("ncmapss_scaler_audit", scaler_audit_path, "FIXED_SCALERS_COMPARED_TO_NC_P_TRAIN_UNITS"),
        ("xjtu_payload_audit", xjtu_audit_path, "PASS"),
    ):
        if path.is_file():
            audit_payload = json.loads(path.read_text())
            report["data_preflight"][key]["status"] = audit_payload.get("status")
            report["data_preflight"][key]["matches_expected_status"] = (
                audit_payload.get("status") == expected_status
            )
        else:
            report["data_preflight"][key]["matches_expected_status"] = False
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    os.environ["PROJECT_ROOT"] = str(ROOT)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        for dataset in ("ncmapss", "xjtu"):
            for model in MODELS:
                exp = (
                    f"concepts_n_cmapss_multi/prognostics/{model}"
                    if dataset == "ncmapss"
                    else f"xjtu_sy/prognostics/phmd_split/combined/{model}"
                )
                overrides = [f"experiment={exp}", "seed=72", "logger=csv"]
                item = {"dataset": dataset, "model": model, "overrides": overrides}
                try:
                    cfg = compose(config_name="run", overrides=overrides)
                    selected = OmegaConf.masked_copy(
                        cfg,
                        [
                            "datasource",
                            "transforms",
                            "task_definition",
                            "model",
                            "evaluator",
                        ],
                    )
                    # Resolve relevant contracts; output paths depend on Hydra runtime and
                    # are intentionally left interpolated in the complete saved config.
                    resolved_cfg = OmegaConf.to_container(cfg, resolve=False)
                    aggregation_resolution = resolve_aggregation_fields(
                        resolved_cfg, args.protocol
                    )
                    cfg = OmegaConf.create(resolved_cfg)
                    selected = OmegaConf.masked_copy(
                        cfg,
                        [
                            "datasource",
                            "transforms",
                            "task_definition",
                            "model",
                            "evaluator",
                        ],
                    )
                    config = OmegaConf.to_container(selected, resolve=False)
                    item["aggregation_resolution"] = aggregation_resolution
                    item["problems"] = inspect_config(config, contract)
                    (args.output / f"{dataset}_{model}.yaml").write_text(
                        OmegaConf.to_yaml(cfg)
                    )
                    item["composition"] = "PASS"
                    item["target"] = (
                        "RUL cycles"
                        if dataset == "ncmapss"
                        else "normalized remaining-life HI; inverse transform returns RUL minutes"
                    )
                except Exception as exc:
                    item.update(
                        composition="FAIL", error=f"{type(exc).__name__}: {exc}"
                    )
                report["experiments"].append(item)
    report["blockers"] = []
    if report["missing_n_cmapss"]:
        report["blockers"].append("NC_P_HDF5_FILES_MISSING")
    for key in ("ncmapss_scaler_audit", "xjtu_payload_audit"):
        if not report["data_preflight"][key]["matches_expected_status"]:
            report["blockers"].append(f"{key.upper()}_MISSING_OR_FAILED")
    if any(e.get("problems") for e in report["experiments"]):
        report["blockers"].append("CONFIG_TRANSFORM_CONTRACT_FAILED")
    if any(e["composition"] != "PASS" for e in report["experiments"]):
        report["blockers"].append("HYDRA_CONFIG_COMPOSITION_FAILED")
    if report["blockers"]:
        report["status"] = "BLOCKED_PROTOCOL_OR_DATA"
    else:
        report["status"] = "READY_FOR_SEED72_MODEL_SMOKE"
    (args.output / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
