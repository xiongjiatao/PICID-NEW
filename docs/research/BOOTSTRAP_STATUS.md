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
- Installed test/lint checks for protocol, cache, metric, XGBoost and test-gate
  fixes: 283 targeted tests pass and Ruff reports no issues.
- Completed one full XJTU-SY PHMD-split LSTM run for seed 72 after recovering
  the feature boundary cache. This is a single-seed pilot, not the paper's
  five-seed reproduction or a performance conclusion; other models, N-CMAPSS,
  and the remaining seeds are pending.
- Audited and corrected the XGBoost identity mismatch in this copy: the
  imported wrapper constructed sklearn GradientBoosting despite its XGBoost
  name. The corrected wrapper now uses locked XGBoost 3.1.3, preserves the
  released 1,000-round / seed-42 defaults, and keeps tree-parameter assumptions
  explicit because the paper does not list them. Ten wrapper tests pass and
  Ruff passes; lock-file validation is pending because offline `uv lock`
  attempts to fetch the pinned PHMD Git source.
- Fixed a runner protocol defect: `picid.run` ignored top-level `test=false` and
  always called `trainer.test`. A seed-72 XGBoost 1/1 candidate therefore
  exposed test metrics despite test being disabled. That attempt is marked
  ineligible for validation selection; its metrics are not used. The test gate
  now skips both final test evaluation and best-checkpoint test reruns, with
  focused regression tests. Re-run all context/stride candidates after this
  fix and select using validation only.
- The first clean validation-only 1/1 rerun correctly skipped test, but exposed a
  second selection defect: fit-predict's `val/loss` was hard-coded to 1.0, as
  documented by an existing `picid_report` test. The validation metrics from
  that attempt are excluded from selection. The copy now computes configured
  regression loss on validation/test predictions while retaining the dummy
  training loss; report-analysis tests now guard that validation loss varies.
  Focused pipeline/report tests pass after this change; the next 1/1 validation
  rerun must verify a nonconstant `val/loss` before the other four candidates
  are run.

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
  aggregation semantics. A recovery utility re-keyed the fully written boundary
  `DatasetContainer` under the new source hash; source/recovered logical pickle
  fingerprints match exactly. A CPU-only preflight then restored that boundary
  with zero transforms remaining and wrote the complete ~65 MiB preprocessed
  cache. The preflight hit a sandbox-only local-socket permission error when its
  CPU DataLoader workers started, before model fit; its exact process and worker
  group were stopped, without touching other users. The successful GPU0 retry
  loaded the preprocessed cache, trained LSTM for 27 epochs, selected checkpoint
  epoch 1 by validation loss, and produced test metrics. The 3.4 GiB load/split cache remains
  available as well. Recovery evidence is in the failed attempt's ignored
  `artifacts/.../cache_recovery_report.json`; preflight evidence is in
  `artifacts/picid_seed72_xjtu_lstm/runs/cache_preflight_2026-09-27/experiment_manifest.json`.
  The first-attempt run directory is
  `artifacts/picid_seed72_xjtu_lstm/runs/picid_seed72_xjtu_lstm+xjtu_sy+prognostics+phmd_split+combined+lstm/2026-09-27_11-34-53`.

- The seed-72 LSTM selected epoch 1 (validation loss 0.07829). Test device-macro
  normalized-HI MAE/RMSE were 0.20298 / 0.24632 (20.30% / 24.63%), and the
  device-macro PHM score was 0.23196. Window-weighted values were 0.24640 / 0.30896. The inverse
  RUL-acquisition-minute device-macro MAE/RMSE were 139.28 / 171.08; these use
  per-bearing full-life metadata and are offline benchmark metrics, not online
  deployable RUL. Scores vary substantially by test bearing, so this one seed
  cannot stand in for the paper's five-seed result. See the local retry manifest
  and `csv_logs/version_0/metrics.csv` under its run directory.
- The first fit emitted an overflow warning from the *unselected* PHMScore
  `np.where` branch; logged per-device and macro scores were finite. The metric
  now evaluates only the selected branch, preserving the score equation, and an
  extreme-error regression test confirms no overflow warning.
- The official TabPFN v2 regression weight was retrieved from the Zenodo v2
  archive; its archive MD5 and extracted file SHA-256 match the official
  metadata. The seed-72 1/1 TabPFN run reached the fit stage but OOMed on GPU0
  while building the 8-estimator cached context from 6,557×460 training rows:
  it had 5.00 GiB free and requested another 5.41 GiB. No TabPFN metrics were
  produced. The current executor builds this KV cache in one forward pass;
  a cache-free single-step path is being tested without changing the data.
- The mislabeled sklearn GradientBoosting control was stopped at the user's
  request before completion: 92/1,000 trees after 2m16s, with about 22m still
  estimated. No metrics were saved or used.

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
