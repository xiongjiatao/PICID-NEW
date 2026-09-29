# NC-P device validation and blind-test eligibility protocol

Protocol v2 frozen 2026-09-29, before any successful output from the grouped-CV
runner. It supersedes the earlier draft that subsampled query rows and named
classic C-MAPSS as a blind-test candidate. The earlier launch failed its
environment/version check before model execution and produced no predictions.
The target study is [TFM-PHM, arXiv:2606.05481v1](https://arxiv.org/html/2606.05481v1).
Each run stores the resolved command, code and data hashes, package and weight
versions, GPU mapping, concurrent tasks, seed, fold, exact sampling metadata,
predictions, metrics, and resource measurements.

## What this protocol does and does not reproduce

The primary reproduction follows the project's audited PICID NC-P
configuration: for DS01, DS04, DS05, and DS07, source-local units 1–5 train,
unit 6 validates, and units 7–10 test. TFM-PHM reports NC-P in its main results,
but Appendix A.3 tabulates only the distinct DS02 split; it does not fully
specify this NC-P unit assignment. We identify this as the frozen PICID
reproduction protocol, not as a split explicitly listed in the paper. The
NC-P target is RUL on the benchmark's transformed timeline; its fixed
feature/target transforms and native-unit mapping are inherited from the
audited PICID configuration. The paper's broader protocol searches five
fit-predict window/stride settings on validation, evaluates every test instance,
and reports five-seed results. This document does not replace that split or
relabel previously accessed test results as blind evidence.

The XJTU-SY external reproduction remains the paper's PHMD 8/3/4 bearing split
and health-index target `HI = 1 - runtime / total_lifetime`; it is not direct
RUL. For the separate warning analysis, candidate horizons are frozen from
training-bearing lifetimes and warning labels are derived on each raw timeline.
Do not convert an XJTU test bearing's HI prediction into an online RUL using its
realized total lifetime. These four test bearings were already accessed, so
this remains a public benchmark replay.

The added analysis is a **development-pool, source-stratified, device-held-out
cross-validation** over NC-P training units 1–5. The same source-local unit is
held out in all four sources per fold; over five folds, each of the 20
development engines is evaluated once. Every fold still has context from all
four sources, so it does not test transfer to an unseen source. These assets
have been used in earlier development work, so this estimates device-level
variability inside the development pool. It is not independent confirmation,
an external-domain test, or a blind test. The canonical NC-P test engines and
all four XJTU-SY PHMD test bearings were previously accessed; reruns on them
are protocol replays.

The benchmark's fixed scaler constants follow the existing paper-matched
configuration, but their derivation population is not documented. A prior
streaming comparison found them close to the canonical NC-P training-unit
statistics without resolving their provenance. Therefore the fixed-scaler
analysis is the reproduction path, while a train-fold-fitted scaling analysis
is a required sensitivity before making a strong device-generalization claim.
The paper's Appendix A.3 prose says min-max scaling, while its detailed Appendix
C.2.3 specifies fixed N-CMAPSS scalers, 60-step non-overlapping aggregation,
and the RUL factor 0.01; the audited PICID transform follows the detailed schema
and code. Record this textual discrepancy and the scaler provenance; do not
silently replace the fixed benchmark transform.

## Frozen NC-P device-level comparison

For each of the five folds, hold out units 1, 2, 3, 4, or 5 across DS01/04/05/07
as one group. The other 16 source-plus-unit identities provide context. Source
and unit remain a composite key in all preprocessing, sampling, metrics, and
prediction files. No row-level random split is allowed.

Evaluate the five paper fit-predict candidates `(window, stride)` = `(1,1)`,
`(5,1)`, `(10,5)`, `(20,5)`, and `(50,50)`. The stride restricts eligible
**training-context endpoints only**. Every transformed-time row from each held-
out engine is a query, in chronological order; no test/query subsampling is
allowed. Every candidate and seed within a fold receives byte-identical query
IDs, times, and targets. Histories are built separately inside each engine and
use the repository's audited causal left-edge padding rule.

The supplementary grouped-CV run uses TabDPT v1.3.0, eight ensembles, and the
project's frozen 2,048-row context configuration (`configs/model/
tabdpt130_fit_predict.yaml`). The paper specifies the candidate grid, validation
selection, and test-row policy; the 2,048 value is the repository's resolved
model budget, not a number disclosed by the paper itself. Context endpoints
are sampled from training devices only, at indices allowed by the candidate
stride. Allocation is max-min water-filled across devices: a short engine is
capped at its available endpoints and its unused quota is redistributed as
evenly as possible. No endpoint may repeat. The same seed/fold/stride uses the
same endpoints for candidates with that stride, so `(1,1)` versus `(5,1)` and
`(10,5)` versus `(20,5)` isolate window width at fixed context timestamps.

The project comparison uses five paper seeds: 72, 88, 101, 666, and 226688.
Report each seed's equal-device macro and mean ± sample standard deviation
across seeds. For each device, average its five seed scores before computing
paired candidate-minus-`(1,1)` differences. The 95% percentile interval uses
10,000 bootstrap draws, sampling five engines with replacement within each
source and averaging the four source means equally. Seeds are not independent
assets, and timestamps are not bootstrap units. Since the five CV training
sets overlap, these device bootstrap intervals are descriptive resampling
summaries, not formal coverage guarantees for a new fleet.

Report all five candidates, including worse results. The held-out folds cannot
select a winner or a new mechanism. Per device, report normalized-target MAE,
MSE, and RMSE; native-RUL MAE, MSE, and RMSE; and the native-unit asymmetric
NASA score used for direct-RUL tasks. Also report supplemental critical-stage
MAE at 5%, 10%, and 20% of the median maximum lifetime computed from the 16
training engines in that fold only. Keep equal-device macro metrics separate
from query-weighted values. Critical-stage scores are descriptive and do not
replace the paper metrics.

The full registered comparison is 5 seeds × 5 folds × 5 candidates = 125
candidate fit/predict evaluations. Query count, data-preparation time, model
initialization, fit time, prediction time, output serialization, steady-state
throughput, peak allocated/reserved memory, candidate wall time, and full-run
wall time are recorded by candidate. Before scheduling the full grid, run one
fold end to end and report measured throughput, peak memory, concurrent GPU
jobs, and an updated ETA. Transient zero GPU utilization is not a stop
condition; use process heartbeat and completed candidate counts. Do not
interrupt unrelated GPU processes.

## Frozen blind-test eligibility gate

No untouched, custodian-controlled asset set has yet been verified. Therefore
the blind evaluation is **pending**, not an executable benchmark-specific
protocol and not a result. Freeze this gate now; do not substitute a convenient
dataset to make the status appear complete.

Before a future blind run, an independent custodian must identify an original
paper dataset/task and asset split, audit prior access to raw trajectories,
labels, predictions, and aggregate metrics, and confirm that neither assets nor
labels informed feature/model/configuration selection. The custodian must retain
test targets, publish hashes for the source archive and permitted inputs, and
release only the inputs allowed by the task definition. Before scores are
released, deposit hashes of code, configuration, dependencies, weights, and
prediction artifacts. Any access uncertainty disqualifies the set from a blind
claim. If the audit fails, report the benchmark as an ordinary public-data
replay or acquire prospective industrial assets with a documented custodian.

Classic NASA C-MAPSS FD001–FD004 is a distinct benchmark and is not a substitute
for TFM-PHM's N-CMAPSS NC-P or DS02 tasks. TFM-PHM's DS02 split is train units
2/5/10/16/20, validation 18, and test 11/14/15; it is also a separate task, and
its published test results prevent a claim of globally unseen labels. The
current local dataset inventory contains NC-P sources DS01/04/05/07 but no
verified DS02 raw data. Neither dataset is presently eligible as a blind set.

If an eligible cohort is secured, task-specific details must be preregistered
before the custodian opens labels: source and asset IDs; target and units;
training/validation/test roles; preprocessing fit population; allowed query
history; model versions and weight digests; context and compute budgets;
validation-only selection rule; primary and secondary metrics; seed set; failure
handling; and device-level uncertainty procedure. A dataset name alone is not
a complete blind-test protocol.

## Validity and stop rules

Reject a grouped-CV run if composite engine IDs merge, a held-out engine enters
context or fitted preprocessing, any engine query timeline is incomplete,
repeated, reordered, or differs across candidates/seeds, or predictions depend
on future features. Reject the fixed-scaler generalization claim if the scaler
provenance limitation is omitted or the train-fold scaling sensitivity reverses
the conclusion. Retain failed, interrupted, and OOM attempts in their manifests
but exclude them from successful-result means. A real negative result under the
matched protocol remains in the comparison table.
