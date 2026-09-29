# NC-P device validation and blind-test preregistration

Protocol frozen: 2026-09-29, before any output from the grouped cross-validation
runner in this file. The code revision, lockfiles, data manifest, and weights are
recorded in each run manifest. This registration supplements the TFM-PHM
reproduction; it does not replace the paper's fixed split or claim an exact
numeric reproduction.

## Evidence status and limits

The target paper uses device-disjoint train/validation/test sets and selects
fit-predict window/stride settings using validation loss. It reports five random
seeds: 72, 88, 101, 666, and 226688. This study keeps those seeds and the paper's
five candidate pairs. The four NC-P unit-6 validation engines remain the
validation-selection set in the primary reproduction.

The additional NC-P analysis below is a **development-pool, source-stratified
device-held-out cross-validation**, not a blind test. Its 20 engines come from
canonical training units 1–5 in DS01/04/05/07. Those data and the scientific
question have appeared in earlier development work. The fold predictions are
new, but this history means the result cannot be described as untouched
confirmatory evidence. The benchmark's fixed feature-scaler constants follow
the paper/code protocol; their derivation population was not disclosed. A prior
streaming comparison found them close to NC-P training statistics, but does not
resolve provenance. Report this as a limitation and run a train-fold-fitted
scaling sensitivity before making a strong asset-generalization claim.

Neither the NC-P canonical test engines (units 7–10 in each source) nor the
four XJTU-SY PHMD test bearings are eligible for blind evaluation: test metrics
and prediction histories were previously accessed. Any rerun on them is a
locked-protocol benchmark replay and must disclose that access history.

## Expanded NC-P device-level validation

The source-local device identity is the pair `(n_DS, unit)` throughout. Fold
`k` holds out unit `k+1` from each of DS01, DS04, DS05, and DS07; the other 16
engines are training-context devices. Across five folds, every one of the 20
engines is queried exactly once. No row-level random split is permitted.

Each fold evaluates TabDPT v1.3.0 with the original fit-predict candidates
`(window, stride) = (1,1), (5,1), (10,5), (20,5), (50,50)`. Use 8 inference
ensembles, at most 1,024 context rows balanced equally over the 16 training
engines (64 per engine), and 512 deterministic, timeline-spanning query points
per held-out engine. The same query indices are reused across all five
candidates and all seeds within a fold. A candidate's training endpoints are
restricted to multiples of its registered stride; the causal history window is
constructed separately inside each engine. The default left-edge behavior is
edge padding, as in the existing temporal-control implementation. These fixed
context/query budgets are a controlled supplementary analysis; they are not
the paper's full-row fit-predict table. Candidates with a shared stride reuse
the same sampled training endpoints, so the `(1,1)`/`(5,1)` and
`(10,5)`/`(20,5)` contrasts change history width without changing context
timestamps.

Only training-device RUL labels may set critical-stage boundaries. In each
fold, compute the median of the maximum native RUL among its 16 training
engines, then define 5%, 10%, and 20% horizons from that median. Convert the
paper's normalized target back to native RUL units with the frozen constant
target factor. The held-out engines do not contribute to their own horizon.

The primary endpoint is the equal-device macro MAE in native RUL units over the
512 common queries. Prespecified secondary endpoints are equal-device macro
RMSE and MAE within the fold-derived 5%, 10%, and 20% critical horizons. Report
all per-engine scores. For each candidate, report the five seed-level
device-macro means and their mean ± sample standard deviation. For paired
candidate-minus-`(1,1)` differences, average each engine's score across seeds,
then obtain a 95% percentile interval from 10,000 cluster bootstrap replicates:
sample five engines with replacement within each source and weight the four
source means equally. Seeds are not independent devices; timestamps and query
rows are not bootstrap units. Do not claim significance or a population-level
false-alarm guarantee from these 20 development engines.

All five candidate scores are reported. The held-out-fold scores are not used
to select a winner or invent a new temporal mechanism. Any later method
selection requires a new nested design and a separate untouched evaluation.
Failures, OOMs, or interrupted folds are retained as failed manifests and are
never averaged as successful results. Every run records the full command,
physical and logical GPU, concurrent task schedule, seed, fold, input/weight
hashes, device assignments, exact context/query endpoints, timing, throughput,
and peak memory.

## Prospective project-blinded external evaluation

The candidate external benchmark is the **classic NASA C-MAPSS FD001–FD004
release**, not N-CMAPSS DS02. The repository artifact search found no prior
FD001–FD004 prediction or metrics artifacts, but that search does not establish
absence of human or external exposure. The source archive contains a nested
`data_set.zip`; its test labels have not been audited or scored in this phase.
Classic C-MAPSS is a public benchmark, so any blinding can only be
project-custodian blinding, not globally unknown labels.

Before opening any C-MAPSS test RUL file, an independent custodian must verify
source provenance and project access history, retain the official test RUL
files, and provide only the official training split plus an engine-disjoint
development validation split. If provenance or prior access is uncertain, the
benchmark is not eligible as a blind test and a genuinely prospective
industrial asset cohort must be enrolled instead. The custodian records the
archive and each released input-file SHA-256 without disclosing test targets.

For each FD subset, sort official training engine IDs, assign 20% to validation
using the lowest SHA-256 ranks of
`picid-cmapss-blind-v1|<FD-subset>|<engine-id>`, and use the remaining 80% for
candidate fitting. The registered methods are TabPFN, TabDPT 1.1.13, TabDPT
1.3.0, XGBoost 3.1.3, and LSTM. For the four fit-predict methods, test the
paper's five `(window, stride)` pairs. For LSTM, test sequence lengths 1, 10,
and 50 crossed with learning rates `1e-3`, `5e-4`, and `1e-4`, with batch size
512 and at most 200 epochs; use the existing early stopping and best-validation
checkpoint rule. Use the frozen paper-intent preprocessing and target units;
fit any data-dependent feature transform on the 80% training engines only.
Select one configuration per method and FD subset by minimum configured
validation loss pooled over validation rows and the five registered seeds;
break exact ties by smaller window/sequence length, then smaller stride, then
lower learning rate. No candidate may be selected from official test inputs or
labels.

After selection, refit once on all official training engines with the selected
configuration and the registered epoch rule. Training RUL is derived
cycle-by-cycle only from each run-to-end training engine. At test time, emit
exactly one prediction per engine at its last observed cycle. The evaluator
joins that prediction to the custodian-held official terminal RUL label. No
test-derived preprocessing, selection, calibration, threshold, or retry is
allowed.

Report FD001, FD002, FD003, and FD004 separately and with an equal-subset
macro; per-engine MAE, RMSE, and the registered NASA asymmetric score are
primary outputs. Use the five registered seeds, report mean ± sample standard
deviation across seeds, and separately bootstrap test engines within each FD
subset. Publish every registered baseline and every failure. The custodian
releases metrics only after the frozen code/config/weights/prediction artifact
hashes are deposited. A code defect that invalidates the protocol voids that
evaluation; any replacement split must be registered before its labels are
opened.

This external protocol is **frozen but not executed**. It becomes a blind
evaluation only after the independent custodian's eligibility check and sealed
label handling are in place. Until then, no result may be called blind.

## Stop and interpretation rules

The expanded NC-P cross-validation is stopped as a generalization argument if
fold membership merges source-local unit IDs, any fitted preprocessing sees a
held-out engine, query sets differ by candidate, or the unknown scaler
provenance is ignored in the claim. A new contribution claim is stopped if the
simple registered window/stride controls explain the observed gain, if the
gain is confined to one source/device, or if an improvement disappears under
the training-only scaling sensitivity. A benchmark result is stopped as blind
if an author or model-selection process has already accessed its test labels.
