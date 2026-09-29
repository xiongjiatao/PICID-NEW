# Execution status — updated 2026-09-30

## Host reconciliation — 2026-09-30 00:25 Asia/Shanghai

Host-level process and GPU inspection found no surviving PICID model or tracker
processes. Earlier `running` statements below are historical, not current status.
NC-P TabDPT v1.1 seed101 stopped without a recorded exit code after six of eight
test ensemble members; TabPFN NC-P 10/5, 20/5 retry and XJTU seeds88/101 also
lack completion evidence. These attempts are excluded from result averages.
Their manifests and partial logs are retained; the cause is not established.
The input-export preprocessing job likewise has no surviving process.

NC-P TabDPT v1.1 seed88 completed and its saved 196,153 predictions reconcile
against all 16 devices: native device-macro MAE 7.0351963, RMSE 9.5353289.
The stricter prediction audit now rejects empty/nonfinite arrays, nonintegral
IDs, invalid tolerances and nonfinite reported metrics.
Four validation audit/manifest JSON files for NC-P TabPFN 1/1 and 5/1 had
literal trailing backslash-n bytes. Original bytes are archived alongside the
files; valid JSON and audit digest links are repaired, without changing scores.
See local `artifacts/formal/results/json_serialization_repair_20260930.json`.

The latest supplied AGENTS instructions restrict new scheduling to physical
GPUs0–2. Previous GPUs1–6 assignments remain historical provenance. Host memory
is also a constraint: the completed TabDPT seed88 tracker observed about
208,441 MiB peak process-tree RSS; do not relaunch all pending tasks together.

## Superseding status update — 2026-09-30

- All five seed-72 XJTU TabPFN validation candidates completed with
  `test=false` under the low-memory, 32-query-yield path. The frozen 50/50
  seed-72 final run completed on physical GPU4 in 21,957 s and its saved
  predictions reconcile against all four bearings and six regression metrics.
  It produced device-macro normalized MAE 0.17742 and query-weighted MAE
  0.22105 over 2,261 query rows. This result is kept outside the
  paper-matched cached baseline because cached/low-memory predictions failed
  the registered equivalence tolerance and 50/50 uses 23,000 features, far
  beyond TabPFN v2's documented 500-feature intended range. Seeds 88 and 101
  are now running the same frozen alternate execution path on physical GPUs 4
  and 5. Prior public test access remains disclosed.
- The earlier statement that further XJTU TabPFN retries were not planned is
  superseded. Retries are allowed after an execution-path and resource
  preflight; prior OOM/slow attempts and test access remain part of the record.
- NC-P TabPFN's 265,359-row full-context attempt ended with a CUDA kernel
  configuration error, not a confirmed OOM. The pinned local TabPFN 2.2.1
  package declares an intended range of 10,000 rows and 500 features. A
  separate, explicit 10,000-row device-balanced temporal training-context
  policy is being validated. Seed-72 1/1 completed with validation loss
  0.0097866 after 11,937 s on physical GPU5. The 5/1 candidate completed
  validation on GPU1 with loss 0.0122329 after 33,144 s (9.21 h); its audit
  confirms `test=false` and no populated test metrics. The 10/5 candidate is
  active on GPU2; all use `test=false`. The first 20/5 attempt OOMed during fit
  on GPU1 while the 5/1
  run shared the card (15.85 GiB plus 2.11 GiB resident; a 6.47 GiB allocation
  failed with 5.72 GiB free). That attempt has no score and is retained as a
  concurrency-induced failure; a follow-up launcher rejected GPU6 before model
  start because the experiment allow-list ended at GPU5. The allow-list now
  matches the current authorized GPUs 1–6 and its regression test passes. The
  isolated retry is running on GPU6 with a 23,000 MiB peak budget and 1,024 MiB
  reserve under a tracked manifest. The 50/50
  candidate is pending and is outside TabPFN's documented 500-feature range.
- NC-P TabDPT 1.1.13 seed 72 is complete and audited. Seeds 88 and 101 remain
  active; their preprocessing is currently CPU-bound, while model inference is
  expected to use the assigned GPUs. Do not interpret short GPU-idle intervals
  as a failed run.
- The supplementary NC-P grouped device-held-out window control v2 completed
  all five folds × five seeds × five candidates. It evaluated every query row
  for all 20 development engines (1,326,795 rows per candidate); canonical
  test units were untouched. TabDPT 1.3.0's native-RUL device-macro MAE rose
  from 10.3084 ± 0.0509 at 1/1 to 10.8741, 11.2172, 11.6201, and 13.4467 for
  5/1, 10/5, 20/5, and 50/50. Source-stratified engine bootstrap intervals for
  paired MAE increases are above zero for each longer window. Preserve this
  negative result; it is development-pool evidence, not a blind test or a
  replacement for the main split. Full results are under
  `artifacts/research/ncp_device_cv_v2/`.

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
  metadata. The original full-context KV-cache mode OOMed on seed-72 1/1, and
  cache-free single-query inference was too slow. Switching only the inference
  execution mode to `fit_preprocessors` allowed all five paper-listed XJTU
  context/stride candidates to complete validation with all training rows and
  eight estimators preserved. Their configured validation MSEs were 0.10049,
  0.12052, 0.11767, 0.10240, and 0.04189 for (1,1), (5,1), (10,5), (20,5),
  and (50,50), respectively. The apparent 50/50 validation advantage is a
  severe feature-count extrapolation: 23,000 flattened columns versus TabPFN
  v2's 500-feature pretraining limit (`ignore_pretraining_limits=true`). It is
  only a validation-selected single-seed pilot, not evidence of generalization.
  Test evaluation of the selected candidate OOMed when the full 2,261-query
  trajectory was passed at once (requested 4.93 GiB with 3.20 GiB free). A
  cache-reusing test-only attempt with 32-query wrapper chunks then ran for
  30m59s without writing metrics and was stopped. These attempts touched the
  public test split and must remain disclosed. As of the superseding update
  above, all five validation-only candidates completed under the low-memory /
  yield32 path. The selected seed-72 final output and its per-bearing metric
  reconciliation are complete; seeds 88 and 101 are running. The earlier
  no-retry sentence is historical and no longer describes the active plan.
  Per-run manifests record both the successful validation candidates and failed
  test attempts under their ignored `artifacts/` directories.
- The mislabeled sklearn GradientBoosting control was stopped at the user's
  request before completion: 92/1,000 trees after 2m16s, with about 22m still
  estimated. No metrics were saved or used.
- Corrected XGBoost 3.1.3 seed-72 XJTU validation candidates all completed
  without test metrics. The lowest configured validation MSE was at context /
  train-stride (50, 50): 0.07709; the other four losses were 0.09630, 0.09254,
  0.08754, and 0.09615 for (1,1), (5,1), (10,5), and (20,5). The selected
  50/50 test run gave device-macro normalized-HI MAE 0.21063 (21.06%), RMSE
  0.26145 (26.14%), and PHM score 0.23328. The paper's Table 15 reports
  XGBoost normalized MAE 19.20±0.00%; Table 16 reports PHM score 20.14±0.00.
  The differences are not attributable yet: this is one seed, our tree grid is
  under-specified relative to the paper, and an earlier runner bug exposed the
  same test set on an unselected 1/1 attempt. Treat this result as exploratory,
  not confirmatory or an exact reproduction.

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


## GPU authorization update — 2026-09-30

The user explicitly updated scheduling authorization to physical GPUs0–6 and
allows concurrent jobs. This supersedes the 0–2 restriction in the prior host
reconciliation entry for new jobs in this session. Running jobs on GPUs0–2
continue unchanged. At the update, GPUs3–6 were empty; NC-P 20/5 and50/50
validation candidates are assigned separately there. A concurrent memory
preflight remains mandatory, and no existing user or third-party process is
interrupted.
