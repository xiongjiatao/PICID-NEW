# Execution status — 2026-09-27

## Completed in this project

- Stopped the four old project GPU workers and two D_select controllers on user
  request on 2026-09-26. They exited after SIGTERM. A new, single-GPU XJTU
  reproduction was later launched on physical GPU0 only.
- Initial source snapshot was committed and pushed to `xiongjiatao/PICID-NEW`;
  `picid-release` remains read-only.
- Installed the frozen `uv.lock` in the copy's independent Python 3.12.14 /
  torch 2.9.1 environment, including the locked TabPFN fork, TabDPT and PHMD.
- Read the full text of all three required PDFs and compared their target, split,
  preprocessing, context, metric and reproduction claims against the code. The
  result is in `FULL_TEXT_CROSS_REVIEW.md`.
- Fixed the aggregation constructor so `aggregation=` is accepted as an alias
  for `agg=` and conflicting values fail fast. Fixed `HealthIndexTransform`
  documentation and tests to represent its true PHMD RUL input; configurations
  now use the explicit `rul_key` while old `runtime_key` remains supported.
- Added two deterministic config-audit modes. Both compose all eight selected
  dataset/model configurations without fitting. All select the requested splits,
  and the two aggregation modes resolve as expected.
- Installed test/lint checks for both fixes: 61 targeted tests pass and Ruff
  reports no issues. This does not constitute a PHM performance reproduction.

## Data state and current gate

- The user-provided NASA archive now has the exact official outer-archive byte
  count (15,760,443,389 bytes) and a valid outer ZIP directory. It contains a
  nested `data_set.zip`, so extraction requires one streaming staging pass; the
  extractor now supports that without unpacking the full raw archive into a
  persistent second copy. HDF member CRC and SHA256 checks passed during
  selective extraction; DS01/04/05/07 are ready in the copy's ignored `datasets`
  directory.
- The earlier NASA download remains paused with 6.77 GB in a separate `.part`
  file. It has not been resumed or used.
- `prepare_ncmapss.py --archive <path>` supports the supplied nested archive and
  never contacts NASA in that mode. It extracted only DS01, DS04, DS05, DS07;
  each HDF SHA256 is in the extraction status record.
- Paper config audits and scaler provenance comparison have completed. The
  existing XJTU source is linked read-only into the expected PHMD cache layout.
  A bounded-memory audit passed for all 9,216 CSVs / 15 bearings: each file has
  one header plus 32,768 rows, acquisition counts match the PHMD lifetime lookup,
  and the 8/3/4 split plus reversed-index RUL formula pass. The first full
  object-materialization attempt read all files but did not yield an audit
  artifact; the streaming audit replaced it. The first full seed-72 PHMD-split
  LSTM attempt processed all XJTU units and completed both feature transforms,
  but failed before model fit while serializing a boundary cache: a local lambda
  stored by `WindowedAggregationTransform` is not picklable. Time statistics
  took 7,195.99 seconds; spectral features took 1,045.47 seconds. No model
  metrics were produced. The lambda has been removed without changing first/last
  aggregation semantics, and a recovery utility is being validated against the
  fully written 65 MiB boundary `DatasetContainer`; the 3.4 GiB load/split cache
  remains available. Its run directory is
  `artifacts/picid_seed72_xjtu_lstm/runs/picid_seed72_xjtu_lstm+xjtu_sy+prognostics+phmd_split+combined+lstm/2026-09-27_11-34-53`.

## Important protocol decisions

- Primary benchmark is NC-P, not single-source N-CMAPSS or NC-DS02. Use the
  existing multi-source experiment config and 1–5/6/7–10 unit split per source.
- XJTU reproduction uses the TFM paper's PHMD split (8/3/4). PICID's
  in-domain-fold split (9/3/3) is a separate experiment.
- Fixed standard N-CMAPSS scalers are the paper-matched path. The papers define
  them as benchmark constants, not current-split-fitted values, but do not trace
  their original source population. The CPU audit excluded validation/test
  units and found close numerical agreement with NC-P training-unit statistics.
- Report normalized XJTU HI errors and inverse RUL-acquisition-minute errors
  separately. With N acquisitions, PHMD labels RUL as N-1,...,0; HI=RUL/N
  equals the complement of one-based elapsed acquisitions, while zero-based
  elapsed conventions differ by one interval. The inverse transformation uses
  per-unit full-life metadata for evaluation; it is not a model feature or an
  online estimate of total device life.
- No warning method novelty or field maintenance-event performance has been
  established. The public run-to-failure labels support only a derived warning
  task until separately validated against real maintenance events.
