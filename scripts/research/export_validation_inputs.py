"""Materialize training/validation arrays using the canonical preprocessing cache."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from picid.data.data_objects import SplitViewPolicy  # noqa: E402
from picid.data.preprocessing.preprocessor import PreProcessor  # noqa: E402
from picid.transforms.base.transform_manager.transform_manager import ConfigTransformManager  # noqa: E402
from picid.research.chunks import atomic_json, file_digest  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("nc_p", "xjtu"), required=True)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasource-cache-dir", type=Path)
    parser.add_argument("--cache-path", type=Path,
                        help="Environment-version-specific PICID cache root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ROOT
    os.environ["PROJECT_ROOT"] = str(root)
    base = "concepts_n_cmapss_multi/prognostics" if args.dataset == "nc_p" else "xjtu_sy/prognostics/phmd_split/combined"
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="run", overrides=[f"experiment={base}/xgboost_fit_predict", "seed=72", "test=false",
            f"task_definition.seq_len={args.window}", f"task_definition.stride_train={args.stride}"])
    # PROJECT_ROOT is the code worktree; all data caches must resolve to the
    # shared, audited data root passed explicitly by the operator.
    cfg.paths.data_dir = str(args.data_root.resolve())
    if args.cache_path:
        cfg.paths.cache_path = str(args.cache_path.resolve())
    if args.datasource_cache_dir:
        if args.datasource_cache_dir.resolve() != (args.data_root / "phmd_cache").resolve():
            raise ValueError("Cache alias must resolve to the same raw payload")
        cfg.datasource.cache_dir = str(args.datasource_cache_dir.absolute())
    started = time.monotonic()
    pp = PreProcessor(datasource=hydra.utils.instantiate(cfg.datasource),
                      transforms=ConfigTransformManager(transforms_config=cfg.transforms))
    pp.pipeline(data_cache_path=cfg.paths.cache_path,
                data_library_part_path=root / "picid/data/datasources",
                transform_library_part_path=root / "picid/transforms", cache_preprocessed=True,
                use_boundary_cache=True)
    data = pp.get_processed_split_dict(view_policy=SplitViewPolicy.KEEP_UNIT_LISTS)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": args.dataset, "window": args.window, "stride": args.stride,
               "preprocessing_seconds": time.monotonic() - started, "splits": {}}
    for split in ("train", "val"):
        row = {}
        device_ids = []
        device_lengths = []
        units_by_source = data[split].get("unit", [])
        sources_by_ds = data[split].get("n_DS", [])
        for source_idx, unit_values in enumerate(units_by_source):
            ds_values = sources_by_ds[source_idx]
            if hasattr(unit_values, "to_list"):
                unit_groups = unit_values.to_list()
                ds_groups = ds_values.to_list()
                for unit_group, ds_group in zip(unit_groups, ds_groups):
                    unit_id = int(np.asarray(unit_group).reshape(-1)[0])
                    source_id = int(np.asarray(ds_group).reshape(-1)[0])
                    device_ids.append(f"DS{source_id:02d}-unit{unit_id:02d}")
                    device_lengths.append(len(unit_group))
        split_stride = int(cfg.task_definition.stride_train if split == "train"
                           else cfg.task_definition.stride)
        seq_len = int(cfg.task_definition.seq_len)
        pred_len = int(cfg.task_definition.pred_len)
        pred_offset = int(OmegaConf.select(cfg.task_definition, "pred_offset", default=0) or 0)
        padding = bool(cfg.task_definition.padding_left_flag)
        min_start = -(seq_len - 1) if padding else 0
        device_sequence_counts = []
        for length in device_lengths:
            required = seq_len + pred_len + pred_offset
            max_start = length - required + 1
            count = (len(range(min_start, max_start, split_stride))
                     if min_start + required <= length else 0)
            device_sequence_counts.append(count)
        for field in ("features", "rul"):
            values = data[split][field]
            if isinstance(values, list):
                array = np.concatenate([np.asarray(v).squeeze(0) for v in values])
                lengths = list(device_sequence_counts) if device_sequence_counts else [v.shape[1] for v in values]
            else:
                array = np.asarray(values).squeeze(0)
                lengths = list(device_sequence_counts) if device_sequence_counts else [len(array)]
            if sum(lengths) != len(array):
                raise ValueError(f"{split}/{field}: unit segments do not cover model rows")
            path = args.output / f"{split}_{field}.npy"
            np.save(path, array, allow_pickle=False)
            row[field] = {"shape": array.shape, "unit_lengths": lengths, "sha256": file_digest(path)}
        unit_values = data[split].get("unit_id")
        if unit_values is not None:
            unit_id = np.concatenate([np.asarray(v).squeeze(0) for v in unit_values])
            if len(unit_id) != row["features"]["shape"][0] or unit_id.shape[1] != 2:
                raise ValueError(f"{split}: composite unit_id does not align with feature rows")
            path = args.output / f"{split}_unit_id.npy"
            np.save(path, unit_id, allow_pickle=False)
            row["unit_id"] = {"shape": unit_id.shape, "sha256": file_digest(path),
                               "fields": ["n_DS", "unit"]}
        summary["splits"][split] = row
        summary["splits"][split]["device_ids"] = device_ids
        summary["splits"][split]["device_identity"] = "datasource ID + source-local unit ID"
        summary["splits"][split]["raw_device_timeline_lengths"] = device_lengths
        summary["splits"][split]["query_or_context_sequences_per_device"] = device_sequence_counts
        summary["splits"][split]["sequence_stride"] = split_stride
        summary["splits"][split]["sequence_length"] = seq_len
        # The fit/predict pipeline presents each source as one concatenated task.
        # Retain raw device identities separately for device-macro reporting.
        for field in ("unit", "n_DS"):
            values = data[split].get(field, [])
            pieces = []
            for value in values:
                if hasattr(value, "to_list"):
                    pieces.extend(value.to_list())
                else:
                    pieces.extend(np.asarray(value).tolist())
            (args.output / f"{split}_{field}.json").write_text(json.dumps(pieces))
    (args.output / "preprocessing.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create({"datasource": OmegaConf.to_container(cfg.datasource, resolve=True), "transforms": OmegaConf.to_container(cfg.transforms, resolve=True)})))
    atomic_json(args.output / "manifest.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
