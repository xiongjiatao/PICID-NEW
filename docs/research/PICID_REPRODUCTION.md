# PICID reproduction and research preparation

## Scope and provenance

The expanded grouped device-validation design and blind-test boundary are
registered in
[DEVICE_VALIDATION_AND_BLIND_TEST_PREREGISTRATION.md](DEVICE_VALIDATION_AND_BLIND_TEST_PREREGISTRATION.md).
It preserves the paper's five seeds and five fit-predict window/stride pairs.
The supplementary 20-engine grouped cross-validation is explicitly a
development-pool analysis; the existing canonical NC-P and XJTU-SY test results
are exposed benchmark replays, not blind evaluations.

The three mandatory papers have been cross-reviewed in full in
[FULL_TEXT_CROSS_REVIEW.md](FULL_TEXT_CROSS_REVIEW.md). This protocol follows
the target TFM-PHM paper and treats the PICID infrastructure paper as the
framework specification.

The unmodified source snapshot is commit `d9d93ee`. The sibling `picid-release`
directory is read-only. This repository is an academic research derivative of
PICID; the original LICENSE.txt and authorship are retained. Initial import is
a local source snapshot, not a reconstruction of upstream Git history.

Targeted reproduction models: TabPFN, TabDPT, the paper-reported XGBoost baseline,
and LSTM. The imported/upstream `xgboost_fit_predict` wrapper actually
instantiated scikit-learn GradientBoosting rather than XGBoost. This research
copy now invokes the `xgboost` 3.1.3 package with 1,000 boosting rounds and the
formal run seed as `random_state`; other tree settings use the locked library
defaults. Historical exploratory XGBoost runs with seed 42 remain separately
identified. This corrects model identity but is not an exact recovery of the
paper's undisclosed tree parameter search. Table 9 supplies
five context/stride pairs, while the paper's generic tuning statement does not
specify XGBoost's tree-specific search ranges. These defaults are therefore
declared reproduction assumptions, not exact paper hyperparameters. The main task is
N-CMAPSS NC-P over DS01/04/05/07. XJTU-SY is reproduced separately with the
TFM-PHM paper's PHMD split (8/3/4); its existing leave-condition evaluation is a
separate generalization result. These are independently fitted tasks, not
zero-shot transfer between turbofans and bearings.

Paper seeds: 72, 88, 101, 666, 226688. First smoke seed: 72. Fit-predict
window/stride candidates: (1,1), (5,1), (10,5), (20,5), (50,50).
Five random seeds quantify algorithm randomness, not independent asset evidence.
This phase uses seeds 72, 88, and 101 as a fixed-configuration baseline; it is
not presented as the paper's five-seed numeric reproduction.

## Execution gates

1. Preserve the import snapshot and record file hashes. Use an isolated Git
   worktree for every subsequent source change.
2. Install the original `uv.lock`, including the paper-specific TabPFN Git fork.
   A generic PyPI TabPFN substitute does not establish exact reproduction.
3. Compose all eight model/dataset configurations and materialize two named
   aggregation policies: `paper_intent_fixed` honors YAML `last/mean`; the
   historical-behavior control reproduces the old swallowed-key mean default.
4. Test XJTU's locked PHMD RUL-to-HI equation, inverse units, fit-on-train feature
   scalers, training-only ICL contexts, label/window alignment and cache equality.
5. After complete source files pass archive/CRC checks, run seed 72 and record
   time/peak memory; freeze validation-selected configurations, then run seeds
   88 and 101. Under the latest authorization, formal GPU work may use
   physical GPUs 1–6, with an explicit CUDA_VISIBLE_DEVICES subset and the
   logical device recorded separately.

Validation-only context/stride selection depends on the top-level `test=false`
flag. The imported runner ignored that flag and still called `trainer.test`; the
copy now enforces it and has a regression test. One XGBoost 1/1 candidate ran
before the fix and emitted test metrics despite the override, so that run is
excluded from selection and recorded as protocol-invalid. A subsequent clean
validation-only run confirmed test was skipped, but found the fit-predict
Lightning module also hard-coded regression `val/loss=1.0`. The copy now passes
the configured loss to the fit-predict module and computes it on validation
predictions (the train-step placeholder remains, since these models do not
optimize weights). The earlier clean run is also excluded from candidate
selection; no test metrics were accessed in that run.

All runtime artifacts belong in ignored `artifacts/`. Record command, commit,
resolved configuration, dependency versions, data/checkpoint digests, GPU mapping,
concurrent processes, exit code, and final status. Archive validation and
configuration preflight passed. The first full XJTU seed-72 attempt ran on
physical GPU0 (`CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`), completed
`time_domain_features` (7,195.99 s) and `spectral_domain_features` (1,045.47 s),
then failed while saving the preprocessor boundary because a local lambda in
`WindowedAggregationTransform` could not be pickled. No model fit or metrics were
produced. The lambda has been removed without changing aggregation behavior.
`recover_xjtu_preprocessed_boundary.py` re-keyed the fully written boundary under
the updated source hash, and the source/recovered DatasetContainer fingerprints
match exactly. The 3.4 GiB load/split cache and 65 MiB boundary payload are
preserved. A CPU-only cache preflight restored the boundary with zero transforms
remaining and wrote a 65 MiB final preprocessed cache. It then encountered a
sandbox-only multiprocessing socket permission error during CPU DataLoader
worker setup, before model fit; only this agent-launched preflight was stopped.
The real GPU0 retry loaded the completed cache, trained the configured LSTM for
27 epochs, selected epoch 1 by validation loss, and evaluated the PHMD test
devices. Seed-72 device-macro test normalized-HI MAE/RMSE were 0.20298/0.24632
(20.30%/24.63%); inverse RUL acquisition-minute MAE/RMSE were 139.28/171.08.
This is an earlier single-seed pilot, not the paper's five-seed reproduction.
The frozen window-50, learning-rate-1e-4 run is a separate later result below.
The recovery report is in the failed run's ignored
`artifacts/.../cache_recovery_report.json`; the CPU preflight command and status
are in `artifacts/picid_seed72_xjtu_lstm/runs/cache_preflight_2026-09-27/experiment_manifest.json`.
The successful retry metrics are in
`artifacts/picid_seed72_xjtu_lstm_retry1/runs/picid_seed72_xjtu_lstm_retry1+xjtu_sy+prognostics+phmd_split+combined+lstm/2026-09-27_14-58-48/csv_logs/version_0/metrics.csv`.

## Resolved findings and remaining evidence

### XGBoost seed-72 PHMD pilot

Using the corrected XGBoost 3.1.3 adapter, context/stride was selected using
only the repaired validation `val/loss` (configured MSE). The five losses were
0.09630 (1,1), 0.09254 (5,1), 0.08754 (10,5), 0.09615 (20,5), and 0.07709
(50,50); therefore 50/50 was evaluated once on the PHMD test bearings. Its
device-macro normalized-HI MAE/RMSE were 21.06%/26.14%, and PHM score was
0.23328. For comparison, the paper's Appendix Table 15 gives XGBoost normalized
MAE 19.20±0.00%, and Table 16 gives PHM score 20.14±0.00. This is not an exact
reproduction: the paper does not expose the XGBoost tree search grid, the
this historical pilot used pinned library defaults with 1,000 rounds and seed
42, and the PHMD test set was accidentally evaluated on one unselected 1/1 run
before the test gate fix.
No test values were used to choose 50/50, but the test is not fully blind, so
these metrics are exploratory pending a clean external/independent evaluation.

### TabPFN seed-72 PHMD validation pilot

The official TabPFN v2 regressor checkpoint was verified by archive MD5 and
checkpoint SHA-256. The paper-listed context/stride candidates were then run
validation-only with the locked TabPFN 2.2.1 package, eight estimators, all
training rows, and `fit_mode=fit_preprocessors`; `test=false` was confirmed by
the runner log and zero populated test fields. The selection metric is the
configured normalized window-weighted MSE. Device-macro validation metrics
were:

| Context / stride | Validation MSE | Normalized-HI MAE | Normalized-HI RMSE | PHM score |
|---|---:|---:|---:|---:|
| (1,1) | 0.10049 | 26.55% | 31.16% | 0.20396 |
| (5,1) | 0.12052 | 29.31% | 33.99% | 0.17041 |
| (10,5) | 0.11767 | 28.60% | 33.11% | 0.18834 |
| (20,5) | 0.10240 | 25.85% | 30.76% | 0.23136 |
| (50,50) | **0.04189** | **18.00%** | **20.39%** | **0.25869** |

Validation therefore selects 50/50 for this one split/seed. Do not interpret
the unusually large gap as reliable evidence: 50/50 creates 23,000 flattened
features, 46 times TabPFN v2's expected 500-feature range; this alone does not invalidate
the experiment. The wrapper's
`ignore_pretraining_limits=true` permits the run but does not make this
feature-count extrapolation in-distribution. This is a single-seed
validation-only pilot, not a five-seed reproduction.

The selected full test attempt OOMed while predicting all 2,261 test queries at
once: it requested another 4.93 GiB with only 3.20 GiB free. It completed
validation but produced no test metrics. A follow-up test-only run reused the
serialized fitted model and enabled the existing PICID query-yield wrapper at
32 queries per call; after 30m59s without test metrics it was stopped. The
initial full-batch attempt had already accessed the test input, and the XGBoost
pilot had also evaluated this PHMD test split, so no subsequent result on this
split can be described as blind confirmation. There is no TabPFN test result
to compare with the XGBoost/LSTM pilots, and execution-equivalent retries remain permitted after memory and throughput
preflight. Test-input forward OOM alone is not label leakage; audit whether
test data influenced fitting or selection and whether query batching changes
predictions. The failed and validation-only runs have per-run manifests under their
ignored `artifacts/` experiment directories.

- The transform now accepts both aggregation parameter names and rejects
  disagreement. The audit writes explicit `agg` values. The old default-mean
  execution remains a named comparison, not the primary paper-intent protocol.
- The copied paper runner does not list the NC-P multi-source configuration;
  the reproduction audit now uses `concepts_n_cmapss_multi/prognostics/*`.
- The supplied XJTU payload contains 9,216 acquisition CSVs across all 15
  bearings. A bounded-memory audit found 32,769 physical lines in every file:
  one header plus 32,768 vibration rows, matching the locked PHMD reader's
  `pandas.read_csv` and 32,768-row RUL period. Each bearing's acquisition count
  matches its lifetime-table entry, and PHMD's reversed index formula yields
  `N-1, ..., 0` RUL at the acquisitions. The fixed 8/3/4 PHMD-paper split and
  all 15 membership assignments pass; see the ignored local report
  `artifacts/xjtu_phmd_payload_audit.json`.
- The transform now names the input `rul_key` (keeping `runtime_key` as a
  compatibility alias) and computes `HI=RUL/N`. For XJTU, RUL begins at `N-1`
  because failure is the final indexed acquisition. This equals
  `1 - elapsed/N` with one-based elapsed acquisition time; against zero-based
  elapsed indices it differs by one acquisition interval. Inverse metrics are
  RUL acquisition-minutes; normalized HI and inverse-minute errors will be
  reported separately.
- The TFM-PHM paper uses XJTU PHMD split 8/3/4; the PICID infrastructure paper's
  in-domain fold-1 is 9/3/3. Their results must remain separate.
- N-CMAPSS fixed standard scalers are benchmark constants, not fitted per run.
  Their original derivation population is undocumented. The main reproduction
  retains them. A streaming comparison on 15,922,221 NC-P training rows found
  maximum fixed-mean difference 0.0522 train standard deviations and fixed/train
  standard-deviation ratios 0.9832–1.0085. This supports consistency with the
  train distribution but does not identify the constants' original source
  population. TFM Appendix A.3 says min-max, but detailed Appendix C.2.3, PICID
  Appendix F.3, and code say fixed standard scaling.
- The historical `domain_shift` fold table always tests condition 3. It is one
  held-condition scenario with five split variants, not five independent domains.

## Warning study preregistration (design only)

N-CMAPSS warning labels use remaining flight cycles. XJTU warning labels derive
from the PHMD remaining-acquisition RUL sequence, separately from normalized HI.
Candidate horizons are 5%, 10%, 20% of training-device median lifetime, computed
per dataset/split in native units. Do not estimate horizons from test lifetimes.
Dataset terminal time is an operational endpoint proxy, not proof of a field
maintenance event or a common physical failure criterion.

Reserve device-disjoint selection/calibration/evaluation roles before fitting
probability mappings. Report device-balanced Brier/log loss, AUPRC, empirical
FPR, detection rate, missed events, repeated alarms, and lead-time distribution.
Calibrate thresholds on calibration devices only; matched test-FPR curves are
descriptive oracle curves, not deployable threshold guarantees. Bootstrap whole
devices; label insufficient sample support explicitly. Do not claim 95%-confidence
FPR <= 5% from correlated timestamps. New method design follows replication and
the literature matrix; no BA-TCT/DH implementation is presupposed.


## Next-stage protocol wiring (2026-09-28)

Work is isolated in branch `research/formal-baselines` and worktree
`.worktrees/formal-baselines`; the source snapshot remains unchanged. The full
seed-72 validation candidate grid and gated final task grid are in
`formal_tasks.json`; the per-model progress and frozen files record which
candidates have since completed.
The independent TabDPT 1.3.0 environment has a recorded lock, and its pinned
weight digests, cross-checked against the official model file page, are stored
in `patches/tabdpt-weights.json`. Per-model selection progress, frozen
seed-72 configurations, final-run manifests and audited result tables record
the experiments completed after this task registry was written.

On a fixed 2,049-row by 18-feature NC-P slice with 512 queries, the old TabDPT
model produced identical predictions at batch 32, then exceeded the configured
1e-4 tolerance at larger batches; batch 512 OOMed. The newer model, using its native subsampling with a 2,048-row context, was stable
through the tested 512 batch size on that slice. TabPFN was stable at batch 32
and differed at 128 and above. These short checks only guide execution settings.
TabDPT 1.3 subsequently completed full NC-P validation selection and its three
selected-configuration test runs; the seed-72 full validation estimate was
about 45 seconds. TabDPT 1.1.13 full validation is still running at the status
reported below. TabPFN's full-context failure remains unresolved and has no
test score. Detailed logs and traces are under the local ignored
`artifacts/formal/`.

The old-worktree result archive has 349 files (4,562,643,595 bytes) with matching
source/target SHA-256 digests. The original worktree remains in place until
cache restoration is demonstrated from the new path. XJTU cache reuse requires
retaining the prior worktree path strings in `paths.cache_path` and
`datasource.cache_dir`; although both point to the same shared directory, PICID
includes those strings in cache identity. A new-path run that omitted these
legacy path values reread the 9,216 source files (about 355 seconds) before any
metrics; it was stopped before model fitting. The seed-72 LSTM selection and
final test runs then used the verified original cache identity.


## N-CMAPSS engine-boundary correction

The audit of the existing transformed cache found row-aligned `unit` and `n_DS`
segments for each source (DS01, DS04, DS05, DS07), but the old window transform
could flatten a source segment before sliding windows. The validation-only XJTU
LSTM sweep was resumed after the correction and all nine seed-72 candidates now
have validation records; its selected configuration is frozen separately. The
new NC-P transform aggregates each ragged engine independently, retains the
engine axis for `TimeseriesTabularizer`, and then creates histories inside each
engine. A synthetic two-engine test checks both aggregate values and the
resulting history rows. A cached NC-P audit checks all 40 source/unit/split
groups, row counts, and the source-plus-unit identity. Any previous NC-P model
outputs produced before this correction are protocol-invalid for non-unit
windows and must be excluded.

The first NC-P LSTM validation attempt then exposed a second interface mismatch:
`RULContextBatchDataset` required datasource `unit_ids` metadata, while NC-P's
composite `(source, unit)` IDs are built as aligned row data. The dataset now
sequences those IDs with the same ragged windows and emits the last ID at each
query. A two-engine regression test verifies alignment. The failed pre-fit
attempt has no validation or test score and is excluded. The corrected LSTM
validation grid is running with `test=false` on physical GPU 0.


## Runtime-specific cache identities

The old TabDPT 1.1 environment uses Awkward 2.9.0; the isolated TabDPT 1.3
environment uses Awkward 2.14.0 and `awkward-cpp` 57. Sharing a cache root
across those environments caused an Awkward pickle schema load failure before
any baseline fit. Candidate runs now use separate roots under
`datasets/cache/formal-py312-awkw29/` and
`datasets/cache/formal-py312-awkw214/`. The raw dataset files remain shared.

An early parallel TabDPT 1.1.13 seed-72 attempt omitted candidate-specific
`experiment_group` values. The window-5 and window-10 launches landed in the
same second-level Hydra output directory and shared `model_cache_dir`; window 5
then failed with a 90-versus-180 feature-count mismatch, and window 10 was
excluded and stopped at six of eight ensembles. Neither attempt accessed test
metrics or enters validation selection. Candidate runners now assign a unique
Hydra output group to every window/stride configuration; corrected window-5 and
window-10 runs are in progress.

## NC-P unit-isolated validation integration

With `seq_len=5`, stride 1, and tests disabled, the real preprocessing path now
emits 67,709 validation queries with 90 features. Their source/device lengths
are DS01-unit06 13,947, DS04-unit06 19,784, DS05-unit06 16,233, and DS07-unit06
17,745; the sum is exactly 67,709. The 20 train engines remain separately
identified. A synthetic boundary regression test exercises both the ragged
window aggregator and `TimeseriesTabularizer`. The quick integration outputs
and identity manifest are under ignored `artifacts/formal/inputs/nc_p_unitwise_w5_s1_final/`.

## Updated model execution evidence

- TabDPT 1.3, context 2,048, seed 72, eight ensembles and inference batch 512
  selected on a 2,049-row by 18-feature NC-P slice. The configuration is
  numerically stable across tested batches up to 512 with a 1e-4 tolerance.
  The full-data window-1 validation candidate ran test-free and reported
  `val/loss=0.0086336`, normalized MAE 0.0704 and RMSE 0.0929. Peak observed
  process memory was about 1.4 GiB; validation inference itself took about
  45 seconds. A subsequent aggregate-only seed-72 test produced normalized MAE/RMSE
  0.0707/0.0984, but it lacked engine IDs and is retained as exploratory only.
  The per-engine `MultiUnitEvaluator` protocol is now wired. The three selected
  fixed-configuration seeds have been rerun under that protocol; the corrected
  results are reported below.
- TabPFN v2.2.1, eight estimators, passed replay and batch-32 equivalence on
  16,384 training rows and one query, with 1.61 GiB peak allocated and 7.36
  seconds for the replay query. At 265,359 NC-P training rows the one-query
  full-context preflight ran for about 1,197 seconds and ended with
  `CUDA error: invalid configuration argument`; observed peak allocation was
  not captured, and the largest sampled GPU usage was 11.3 GiB. This failure
  is not an OOM determination. Separately, the pinned local TabPFN 2.2.1
  `InferenceConfig` declares 10,000 rows and 500 features as intended ranges;
  `ignore_pretraining_limits=true` bypasses the guard but does not subsample.
  This matches the [official TabPFN inference configuration](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/inference_config.py).
  Full-context NC-P TabPFN is therefore not treated as a supported core
  baseline. An explicitly separate validation-only variant now caps the fit
  context at 10,000 rows by assigning near-equal quotas to each training
  engine and selecting rows evenly across its chronological order. Row-aligned
  `unit_id=(source, local unit)` metadata is used only during fitting; validation
  and test query sets are unchanged. The implementation is unit-tested. The
  seed-72 1/1 validation completed with `val/loss=0.0097866` after 11,937 s
  on physical GPU 5; the tracked log confirms `test=false` and the validation
  loss passed the selection parser. The 5/1 candidate remains active on GPU1
  after more than eight hours, and 10/5 is active on GPU2. A first 20/5 try on
  GPU1 OOMed during cached fitting while 5/1 shared that card: the new process
  held 15.85 GiB, the older process 2.11 GiB, and a 6.47 GiB allocation failed
  with 5.72 GiB free. It produced no validation score and is retained as a
  concurrency-induced failure; retry it on an isolated card. The 50/50
  candidate has not started. Window-50 has 900 flattened features, outside the
  documented 500-feature intended range, and must be labelled extrapolative.
- The earlier XJTU LSTM validation-only search has all nine seed-72 losses.
  It selects window 50, train batch 512, learning rate 1e-4 with minimum
  normalized validation loss 0.047744. The frozen configuration and result
  digest are in `xjtu_lstm_frozen_seed72.json`; seed-72 final device-macro
  metrics are recorded below. The public-test history remains disclosed above.


## NC-P device-macro evaluation contract

The selected fit/predict dataset now constructs `unit_id=(n_DS, source-local unit)`
after unit-wise aggregation, tabularizes one current-point ID per query with the
same sequence stride, and passes those IDs through the evaluator. The experiment
uses `MultiUnitEvaluator`, so `*_mean` metrics average engine metrics instead of
all timestamps together. `CustomEvaluatorLightningModule.process_outputs`
removes the singleton fit-predict task axis from IDs before evaluator updates.
Regression tests cover composite IDs and the metric macro. Final seed-72 results
were rerun with this corrected reporting protocol before seeds 88 and 101 were
launched.

## NC-P device-macro evaluator wiring

NC-P fit/predict rows now carry a two-column `unit_id=(n_DS, unit)` vector
through the same per-unit window sequence as the model input. The dataset
collates that vector with each query row and `MultiUnitEvaluator` reports each
engine's metrics plus an equal-engine `*_mean`. The adapter removes only the
singleton fit-predict task dimension; it verifies the remaining ID row count
against predictions. Synthetic tests exercise ID generation, task-axis removal,
per-engine aggregation, temporal windows, and the evaluator's macro arithmetic.
The real window-5 integration exports 67,709 aligned validation IDs. The prior
aggregate-only NC-P seed-72 test is retained as exploratory and is excluded from
final device-macro summaries; it must be repeated with the new evaluator before
being considered for the three-seed table.


For the formal three-seed runs, `subset_seed=72` stays fixed because the
registered NC-P protocol uses `subset_ratio=1.0`; this avoids invalidating the
preprocessed cache without changing the selected rows. Model RNG remains the
reported run seed (72, 88 or 101).

## NC-P grouped device-held-out window control

The supplementary v2 development-pool control is complete for all five
pre-registered seeds (72, 88, 101, 666, 226688), all five device folds, and all
five window/stride candidates: 125 candidate evaluations. In each fold, one
engine from each of DS01/04/05/07 is held out and every chronological query row
from those four engines is evaluated; the other 16 of the 20 DS/source-unit
training engines fit the model. This covers all 20 engines out-of-fold and
1,326,795 query rows per candidate across the five seeds. The canonical held-out
units 7–10 were not used. These folds are development evidence, not the
preregistered test or a blind evaluation.

Every candidate used TabDPT 1.3.0, 2,048 context rows, eight ensembles, and
prediction batches of 512. Results below are device-macro native-RUL metrics;
the parenthesized SD is across the five model seeds. Cost is mean candidate
wall time per fold/seed under the recorded co-scheduled GPU execution, so it
describes this workload rather than an isolated latency benchmark.

| Window / stride | MAE | RMSE | NASA score | Mean seconds | Query rows/s |
|---|---:|---:|---:|---:|---:|
| 1 / 1 | 10.3084 (0.0509) | 13.0865 (0.0548) | 2.7367 (0.0336) | 40.94 | 1,341 |
| 5 / 1 | 10.8741 (0.0505) | 13.7207 (0.0480) | 3.1761 (0.0216) | 40.02 | 1,370 |
| 10 / 5 | 11.2172 (0.0554) | 14.4166 (0.0770) | 3.3758 (0.0415) | 39.95 | 1,377 |
| 20 / 5 | 11.6201 (0.1125) | 14.8810 (0.0945) | 3.6229 (0.0590) | 44.20 | 1,238 |
| 50 / 50 | 13.4467 (0.1188) | 17.0140 (0.1229) | 5.3760 (0.1165) | 49.60 | 1,103 |

The same-stride 5/1 control is worse than 1/1 by 0.5657 native-RUL MAE. For
10/5, 20/5, and 50/50, paired source-stratified bootstrap intervals over the
20 engines for MAE increases versus 1/1 are respectively 0.4844–1.3159,
0.7667–1.8077, and 2.2021–4.0207 (10,000 engine resamples; seeds are first
averaged per engine). These are descriptive device-level intervals, not a
seed-level significance test. In this control, longer and more sparsely
sampled windows do not improve accuracy; 50/50 also has the lowest measured
throughput. Because stride changes for the last three candidates, those
contrasts do not isolate window length. The complete predictions, manifests,
hashes, seed variation, paired intervals, and cost breakdown are in
`artifacts/research/ncp_device_cv_v2/summary.json` and its fold directories.
Candidate time includes model initialization, preparation, prediction, and
serialization; simultaneous runs are recorded in each manifest.

## Device-macro final results

After adding the composite engine ID and `MultiUnitEvaluator`, NC-P TabDPT 1.3
was reevaluated with the frozen window-1 configuration for seeds 72, 88 and
101. The normalized equal-engine macro MAE is 0.07206 (seed sample SD 0.00070),
RMSE is 0.09712 (SD 0.00093), and PHM score is 0.41839 (SD 0.00101). Across
the 16 held-out engines, the source-stratified 95% bootstrap interval is
0.06494–0.07856 for MAE and 0.08921–0.10441 for RMSE. These intervals resample
engines within each of the four source groups and average the three seed scores
per engine; seed SD is reported separately. The intervals describe variation
over this small benchmark split and do not imply a population-level guarantee.
The per-seed, per-engine, per-source results and input digests are in
`artifacts/formal/results/nc_p_tabdpt130_three_seed_summary.json` and the three
adjacent `nc_p_tabdpt130_device_metrics_seed*.json` reports. The earlier
aggregate-only seed-72 test remains exploratory and is excluded.
The equal-engine NASA score, recomputed from saved raw-RUL predictions using
PICID's `NASAScoreMetric` formula, is 1.39440 (seed SD 0.01628); this preserves
the N-CMAPSS paper's asymmetric score alongside the secondary PHM score.
Its source-stratified engine bootstrap 95% interval is 1.138–1.656.
Per-device NASA values and seed aggregation are in the adjacent
`nc_p_tabdpt130_nasa_*` reports.
The seed-averaged normalized MAE varies by NC-P source: DS01 0.06376, DS04
0.07999, DS05 0.05866 and DS07 0.08584 (four engines per source). This is a
descriptive source contrast on the fixed test split, not an independently
replicated source-generalization estimate.

Critical-phase errors use fixed horizons at 5%, 10% and 20% of the median
maximum RUL among the 20 NC-P training engines (units 1--5 in DS01/04/05/07).
The raw training HDF5 targets give a median of 84.0 target units, so the frozen
thresholds are 4.2, 8.4 and 16.8. No validation or test device contributes to
these thresholds. Test rows are included in a phase only when their observed
target is at or below the frozen threshold; every engine has at least 307
queries in the 5% phase. Across the three seeds, equal-engine normalized MAE
is 0.01223 (SD 0.00048), 0.01401 (SD 0.00027), and 0.01833 (SD 0.00021) for
the 5%, 10%, and 20% horizons. Their source-stratified device bootstrap
intervals are 0.00974–0.01477, 0.01066–0.01743, and 0.01366–0.02323,
respectively. The horizon derivation, raw-data digests, per-device errors and
seed summaries are in `artifacts/formal/results/nc_p_critical_horizons.json`
and the adjacent `nc_p_tabdpt130_critical_*` reports. These are retrospective
RUL-error strata for baseline characterization, not online warning results.

The selected XJTU-SY PHMD-split LSTM configuration (window 50, learning rate
1e-4, batch 512) was evaluated for all three fixed seeds. Across four test
bearings, normalized-HI device-macro MAE/RMSE are 0.18348 (seed SD 0.00145) /
0.22039 (SD 0.00075), inverse-RUL MAE/RMSE are 122.56 (SD 0.47) / 144.92
(SD 0.41) acquisition minutes, and device-macro PHM score is 0.28087
(SD 0.00542). The source of device variation is visible in the per-bearing
reports; the normalized MAE's four-bearing bootstrap 95% interval is
0.15051–0.22448, separately from seed variation. Raw RUL-minute results are
offline inverse-transform metrics using the PHMD benchmark's known bearing
lifetimes; they are not online warning estimates. The split's public test
history has prior access, so these scores are reproducibility results, not
blind confirmation. Per-bearing and three-seed outputs are in
`artifacts/formal/results/xjtu_lstm_*`.

The TabDPT 1.3 configuration was selected on seed 72 using the registered
validation loss and then held fixed for all three final seeds. The selected
candidate used context size 2,048, eight ensembles and inference batch 512.
The measured per-seed elapsed time was about 193 seconds, with sampled process
peak memory about 1,012 MiB. Seed-88/101 attempts made before the evaluator fix
were aborted before metrics and are not included. The corrected runs completed
on physical GPUs 0 and 2 while an older TabDPT run occupied physical GPU 1;
their manifests record physical and logical device IDs.

The NC-P XGBoost seed-72 validation grid completed all five window/stride
candidates with `test=false`. Validation losses were 0.027496 (1/1), 0.027943
(5/1), 0.032501 (10/5), 0.033827 (20/5) and 0.043784 (50/50); the selected
configuration is 1/1. Its three selected-configuration test runs completed on
CPU in about 20.3 seconds each. Equal-engine normalized MAE/RMSE are 0.12336 /
0.15471, PHM score is 0.29329, and equal-engine raw-unit NASA score is 6.33798
(source-stratified device bootstrap 95% interval 3.885–10.407).
The NASA score and all per-device errors repeat exactly across the three seeds.
The 5%, 10%, and 20% critical-phase normalized MAE values are 0.08525, 0.08959,
and 0.09745. The three `predictions.nc`
files have the same SHA-256 and all 16 per-device metrics match, so this XGBoost
setup is deterministic under the registered configuration; its zero seed SD
must not be interpreted as independent seed evidence. Per-seed device metrics,
critical errors, prediction audits and the aggregated report are retained under
`artifacts/formal/results/nc_p_xgboost_*`.

The TabDPT 1.1.13 seed-72 window-1 validation run completed on 2026-09-28 at
13:22 Asia/Shanghai. Its `val/loss` was 0.00977; all eight ensembles took
2 h 6 min and the tracked run, including preprocessing, took 8,522 seconds.
The sampled peak memory on its assigned physical GPU 1 was 3,866 MiB. Test
evaluation was disabled and confirmed in the log.

All five seed-72 validation candidates were rerun after the per-engine window
boundary fix, with unique Hydra `experiment_group` values and isolated output
groups. Their validation losses were 0.009772 (1/1), 0.012012 (5/1), 0.010536
(10/5), 0.013515 (20/5), and 0.022941 (50/50). Window-1 is selected and its
frozen configuration SHA-256 is
`fde4020eeab69f2bc176386b8baa8994de31a5ef000634f0d2996db8da08677f`. The
selected seed-72 test run completed on physical GPU 0 in 30,463 seconds
(8.46 h), with a tracked peak of 3,868 MiB. Its test predictions reconcile
against all 16 held-out engines and 196,153 query rows at `atol=1e-4`,
`rtol=1e-6`. Device-macro normalized MAE/RMSE are 0.07015/0.09491, PHM score
0.40488, NASA score 1.51611, and critical-stage normalized MAE at the frozen
5%/10%/20% horizons is 0.01995/0.02035/0.02596. The per-device metrics,
prediction audit, critical-stage metrics, NASA metrics, and full tracked command
are retained in `artifacts/formal/results/nc_p_tabdpt11_*_seed72.json` and the
run manifest. Seeds 88 and 101 are now running concurrently on physical GPUs 2 and 3 under the same frozen window-1 configuration.

Selection uses the plan's configured validation loss (`val/loss`, normalized
query-weighted MSE). The headline test table is separately device-macro. These
are different estimands and are both retained; the test scores did not affect
selection.

## Audited three-seed snapshot (2026-09-28)

This is the completed subset of the planned baselines, not a claim that the
entire model/dataset grid is finished. NC-P scores average 16 held-out engines
(four per data source); XJTU PHMD scores average four held-out bearings. Values
below are normalized MAE/RMSE, followed by PHM score and, for NC-P, raw-unit
NASA score. Parentheses are sample standard deviations across the three fixed
seeds. Device-bootstrap intervals are reported separately in the per-model
sections above. The two datasets are different tasks and their metric values
must not be read as a direct ranking across domains.

| Dataset / model | Normalized MAE | Normalized RMSE | PHM score | NASA score | Critical-stage normalized MAE (5% / 10% / 20%) |
|---|---:|---:|---:|---:|---:|
| NC-P / TabDPT 1.3.0 | 0.07206 (0.00070) | 0.09712 (0.00093) | 0.41839 (0.00101) | 1.39440 (0.01628) | 0.01223 / 0.01401 / 0.01833 |
| NC-P / XGBoost 3.1.3 | 0.12336 (0.00000) | 0.15471 (0.00000) | 0.29329 (0.00000) | 6.33798 (0.00000) | 0.08525 / 0.08959 / 0.09745 |
| NC-P / LSTM | 0.07529 (0.00362) | 0.09789 (0.00432) | 0.39981 (0.01060) | 1.56450 (0.17237) | 0.02325 / 0.02237 / 0.02557 |
| XJTU-SY PHMD / LSTM | 0.18348 (0.00145) | 0.22039 (0.00075) | 0.28087 (0.00542) | n/a | not computed |
| XJTU-SY PHMD / TabDPT 1.1.13 | 0.20130 (0.00375) | 0.23907 (0.00494) | 0.25581 (0.00201) | n/a | not computed |
| XJTU-SY PHMD / XGBoost 3.1.3 | 0.21063 (0.00000) | 0.26145 (0.00000) | 0.23328 (0.00000) | n/a | not computed |
| XJTU-SY PHMD / TabDPT 1.3.0 | 0.22060 (0.00523) | 0.25744 (0.00799) | 0.24110 (0.00286) | n/a | not computed |

Both XGBoost rows have zero seed deviation and identical prediction-file
SHA-256 values across all three executions. They establish deterministic replay
under each configuration, not independent stochastic-seed evidence. XJTU
inverse-RUL errors are 122.56 (0.47) MAE and 144.92 (0.41) RMSE acquisition
minutes for LSTM; 133.13 (3.79) / 163.97 (4.52) for TabDPT 1.1.13; 133.53
(0.00) / 167.32 (0.00) for XGBoost; and 147.89 (5.86) / 173.07 (7.35) for
TabDPT 1.3.0. They are offline inverse-transform metrics using benchmark
lifetimes. Existing access to the public XJTU test history is disclosed above.

The frozen configurations and audited tables are retained in
`artifacts/formal/results/nc_p_tabdpt130_three_seed_summary.json`,
`artifacts/formal/results/nc_p_xgboost_three_seed_summary.json`, and
`artifacts/formal/results/nc_p_lstm_three_seed_summary.json`,
`artifacts/formal/results/xjtu_lstm_three_seed_summary.json`, and
`artifacts/formal/results/xjtu_xgboost_three_seed_summary.json`,
`artifacts/formal/results/xjtu_tabdpt11_three_seed_summary.json`, and
`artifacts/formal/results/xjtu_tabdpt130_three_seed_summary.json`.

For NC-P LSTM, the normalized-MAE engine bootstrap interval is 0.06666–0.08381
and RMSE interval is 0.08806–0.10712; they resample test engines within source
and are separate from the seed deviations in the table. Critical-stage MAE
device-bootstrap intervals at the 5%, 10%, and 20% horizons are
0.01693–0.03110, 0.01609–0.03048, and 0.01887–0.03396. The raw-unit NASA
score and per-engine critical-stage summaries are in
`artifacts/formal/results/nc_p_lstm_nasa_three_seed_summary.json` and
`artifacts/formal/results/nc_p_lstm_critical_three_seed_summary.json`.

NC-P LSTM seed 72/88/101 took 2,307/1,015/1,199 seconds on physical GPUs
0/3/2, respectively. PID-attributed peak VRAM sampled by the tracker was 664
MiB for each run. It passed prediction-versus-per-engine metric reconciliation
for all 196,153 rows and 16 engines at `atol=1e-4, rtol=1e-6`; the largest
absolute metric discrepancy was 2.13e-5. The source-level test MAE is DS01
0.06797, DS04 0.08113, DS05 0.05892, and DS07 0.09315. These four-engine
source means are descriptive, not independently replicated source estimates.

XJTU XGBoost selected window 50 / train stride 50 from the seed-72 validation
losses 0.09630, 0.09254, 0.08754, 0.09615 and 0.07709 for the five registered
window/stride candidates. The validation-only grid took 317 seconds total on
CPU; each of the three final CPU runs took about 61.5 seconds. All four test
bearings and 2,261 prediction rows passed the per-device metrics audit at
`atol=1e-4, rtol=1e-6`.

XJTU TabDPT 1.3.0 selected window 50 / stride 50 on seed-72 validation; the
five candidate validation losses were 0.09717, 0.10706, 0.10025, 0.11349, and
0.05753. The three final runs took 30.6–40.9 seconds and sampled 1,282 MiB of
PID-attributed VRAM each. All three prediction files passed the four-bearing
metric audit at `atol=1e-4, rtol=1e-6`. Normalized MAE/RMSE have four-bearing
bootstrap intervals 0.17171–0.27177 / 0.20920–0.31733; the small bearing count
limits interpretation. Inverse-RUL errors remain offline transforms using
benchmark lifetimes, and prior public-test access is disclosed above.

XJTU TabDPT 1.1.13 also selected window 50 / stride 50. Its validation losses
for the same ordered candidates were 0.09662, 0.10136, 0.10035, 0.09896, and
0.06447. Its three final runs took about 30.7 seconds each and peaked at
1,430 MiB PID-attributed VRAM. The normalized-MAE/RMSE bearing-bootstrap
intervals were 0.14372–0.24540 / 0.17306–0.30321. Both TabDPT versions used
window-50 contexts, but their native context and reduction mechanisms differ;
the test results cannot isolate a version effect.

The first XJTU TabDPT 1.3.0 validation attempt failed before loader creation
because the isolated environment lacked the PHMD reader. The selected Git
revision and its declared compatibility pins were installed, the full
environment lock was regenerated, `uv pip check` passed, and all five
validation-only candidates then completed. The failed run remains in its
manifest and is excluded from candidate selection.

The PHMD split's eight XJTU training bearings have median maximum RUL 354
acquisition intervals. Their frozen 5%, 10%, and 20% warning horizons are
17.7, 35.4, and 70.8 acquisition intervals. The derivation counts ordered raw
training acquisition files and records their listing hashes in
`artifacts/formal/results/xjtu_critical_horizons.json`; no validation or test
bearing lifetime contributes. The first TabDPT 1.3 full-distribution warning
study is now complete for seeds 72, 88, and 101; detailed device-balanced Brier,
AUPRC, detection, false-alarm episodes, lead time, execution costs, and the
point/full-output protocol discrepancy are recorded in
`docs/research/CANDIDATE_EVIDENCE_MATRIX.md` and each seed manifest under
`artifacts/research/xjtu_warning_study_seed*_gpu*_20260929/`. The native CDF,
residual ECDF, and direct event classifier have distinct calibration/utility
trade-offs on four public test bearings. The prior public-test access remains
disclosed above, so this is descriptive replication evidence rather than blind
external validation. Event labels come from the raw RUL timeline; no test
lifetime is used to convert an HI prediction into online remaining time.

### Baseline-version completeness audit

The agreed core comparison evaluates TabDPT v1.1.13 and v1.3.0 separately.
An additional literature check found the official v1.2.0 / TabDPT-Turbo
release (June 2026), which changes the default to long context without
retrieval, introduces new weights, and reports roughly 120× average speedup on
TabArena. Since v1.3 is described by its authors as similar to v1.2 with
additional predictive improvements, omitting v1.2 would leave the proposed
compute-budget question without its closest recent efficiency comparator.
The v1.1.13/v1.3.0 results remain valid as the preregistered core comparison,
but no broad efficiency-superiority claim should be made until v1.2 is run as a
supplementary version baseline. The isolated Python 3.12 / torch 2.9.1
environment and TabDPT 1.2.0 package are installed and import successfully;
the official weight revision is pinned, but the weight file could not be
retrieved through the current proxy because TLS connections terminate with
EOF. The adapter passes the native `context_size=None` default, uses the v1.2
prediction-time `batch_size` API, and records the actual context row count; it
does not cap context at 2,048 rows. No v1.2 model run was launched and no weight
digest is claimed. Its throughput comparison remains pending until the
official checkpoint is available and verified.
Because all three versions change weights and training/interface details,
between-version score differences are descriptive and cannot identify the
effect of any one architectural change.

The NC-P LSTM seeds completed on physical GPUs 0, 3, and 2. Manifests
distinguish physical IDs from `cuda:0`; the historical runs used GPUs 0/2/3,
and the current task-specific authorization covers physical GPUs 1–6. The
NC-P TabDPT 1.1.13 seed-72 test is complete and audited, and seeds 88 and 101
use the same frozen configuration. The original full-context TabPFN validation
remains unsupported: its 265,359-row input is about 26.5 times TabPFN 2.2.1's
10,000-row intended range, and the observed run ended with a CUDA kernel
configuration error, not a confirmed OOM. A separately named 10,000-row,
unit-balanced temporal-context seed-72 validation grid is incomplete. Its 1/1
candidate is audited at `val/loss=0.0097866`; 5/1 and 10/5 remain active,
20/5's first attempt failed from measured same-GPU contention, and its isolated
retry plus 50/50 remain pending. All selection jobs use `test=false`; no NC-P
TabPFN test score is claimed. A fixed-input query-batch comparison helper is
implemented and still needs a cache-backed input export and a GPU slot with
enough memory for the fitted context.
On XJTU, the default cached fit mode OOMed during validation fitting at 6,557
rows and 460 features. A separate low-memory path reproduced deterministically,
but on a matched 2,049-row by 460-feature training slice and 398 validation
queries cached and low-memory predictions differed by up to 0.10086 (mean
absolute difference 0.01015), above `atol=1e-4, rtol=1e-4`. These execution
paths therefore remain separate protocols. All five XJTU low-memory plus
32-query-yield validation candidates completed with `test=false`; their losses
for windows 1/1, 5/1, 10/5, 20/5, and 50/50 were 0.099257, 0.122849, 0.116539,
0.105462, and 0.040901. Window 50/50 is frozen under SHA-256
`4e70b9558c1fbc88a9ea68133141c88f5ada25fefbc0318478c8d5dda1f849c5`. Its
validation run took 3,130 seconds with a tracked peak of 12,534 MiB on physical
GPU 4. The seed-72 final evaluation uses this same low-memory execution
protocol on physical GPU 4. Seed 72 completed in 21,957 s (6.10 h), with
12,698 MiB tracked peak VRAM. Its saved 2,261-row prediction artifact was
recomputed against all six normalized and denormalized per-bearing regression
metrics; the maximum absolute difference was 0.00489 for MSE and remained
within the metric-specific `atol=1e-4, rtol=1e-6` tolerance. The bearing query
counts are 52, 161, 533, and 1,515. The device-macro normalized MAE is 0.17742
while the query-weighted MAE is 0.22105; report both because the bearing
trajectory lengths are highly unequal. The prediction digest is
`e8469cc4923aa4c443cecde376c3164bf978c3cfb5cb37c123fbfa7b2e1e43fb`; the
audit is `artifacts/formal/results/xjtu_tabpfn_lowmem_yield32_prediction_audit_seed72.json`.
This remains a separate low-memory/yield32 result: the matched cached versus
low-memory slice failed equivalence, and 50/50 expands 460 features to 23,000,
far beyond the 500-feature intended range. It is not pooled into the
paper-matched cached baseline table. The audited seed-72 result is a separate
execution protocol; seeds 88 and 101 are now evaluating the same frozen 50/50
low-memory configuration on physical GPUs 4 and 5. Candidate
selection and run manifests are in
`artifacts/formal/results/xjtu_tabpfn_lowmem_yield32_selection_seed72.json`
and the adjacent `artifacts/formal/xjtu_tabpfn_lowmem_yield32_w50_s50_seed*_final/` directories.

| Window / stride | Training rows × features | Validation loss | End-to-end seconds | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|
| 1 / 1 | 6,557 × 460 | 0.099257 | 676 | 23,736 |
| 5 / 1 | 6,557 × 2,300 | 0.122849 | 3,624 | 17,162 |
| 10 / 5 | 1,316 × 4,600 | 0.116539 | 1,125 | 19,898 |
| 20 / 5 | 1,316 × 9,200 | 0.105462 | 2,956 | 11,176 |
| 50 / 50 | 134 × 23,000 | 0.040901 | 3,130 | 12,534 |

These are measured run costs with cached preprocessing, not a matched inference
throughput benchmark: window 1 ran on GPU 2 and the others on GPU 4. Window and
training stride also change together, which changes both the training rows and
validation query set. The loss sweep therefore freezes the best candidate for
this protocol but does not isolate a causal effect of longer windows or
sparser supervision. A controlled fixed-query, fixed-context analysis is still
required for the temporal-information research question. The comparison report
remains `artifacts/formal/results/xjtu_tabpfn_execution_mode_comparison.json`.
