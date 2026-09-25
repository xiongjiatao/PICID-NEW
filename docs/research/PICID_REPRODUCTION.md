# PICID reproduction and research preparation

## Scope and provenance

The three mandatory references and their distinct roles are recorded in
[REQUIRED_REFERENCES.md](REQUIRED_REFERENCES.md). That evidence contract governs
the execution and scientific reporting described here.

The unmodified source snapshot is commit `d9d93ee`. The sibling `picid-release`
directory is read-only. This repository is an academic research derivative of
PICID; the original LICENSE.txt and authorship are retained. Initial import is
a local source snapshot, not a reconstruction of upstream Git history.

First-stage models: TabPFN, TabDPT, XGBoost, LSTM. Main data: N-CMAPSS sources
DS01/04/05/07; bearing validation: XJTU-SY, first with the paper PHMD split,
then the existing held-out-condition protocol. These are independently fitted
dataset experiments, not zero-shot transfer between turbofans and bearings.

Paper seeds: 72, 88, 101, 666, 226688. First smoke seed: 72. Fit-predict
window/stride candidates: (1,1), (5,1), (10,5), (20,5), (50,50).
Five random seeds quantify algorithm randomness, not independent asset evidence.

## Execution gates

1. Preserve the import snapshot and record file hashes. Use an isolated Git
   worktree for every subsequent source change.
2. Install the original `uv.lock`, including the paper-specific TabPFN Git fork.
   A generic PyPI TabPFN substitute does not establish exact reproduction.
3. Compose all eight model/dataset configurations; audit split IDs, target
   semantics, scaling, aggregation, and data availability before fitting.
4. Test input causality, train-only preprocessing, context membership, aligned
   labels, cache equality, and model checkpoints. Composition alone is not a pass.
5. Run seed 72 end-to-end; measure elapsed time and peak GPU memory before a
   multi-seed queue. Use only physical GPUs 0/1/2, explicit CUDA_VISIBLE_DEVICES,
   and record logical cuda:0 separately. Never stop unrelated workloads.

All runtime artifacts belong in ignored `artifacts/`. Record command, commit,
resolved configuration, dependency versions, data/checkpoint digests, GPU mapping,
concurrent processes, exit code, and final status. No automatic experiment launch
is permitted after a blocked protocol audit. No performance results exist yet.

## Source-level findings requiring verification

- `WindowedAggregationTransform` accepts `agg`, while several original configs
  specify `aggregation`. Because **kwargs absorbs the latter, requested `last`
  can silently run as the default `mean`. This affects timestamps/targets and
  potentially the XJTU descriptor sequence. Preserve this as released-code
  evidence; a corrected protocol must have a separate identity.
  `audit_reproduction.py --protocol aggregation_corrected` emits separate
  configs with explicit `agg=last` overrides; it does not edit original transforms
  or declare the remaining provenance/data/runtime gates passed.
- Old paper shell scripts reference experiment paths absent from the current
  tree. Compose the explicit current paths rather than launching old scripts.
- XJTU PHMD membership is 8 training, 3 validation, 4 test bearings; current
  in-domain defaults are different and require the explicit split override.
- The XJTU target is HI. The current HealthIndexTransform documents runtime /
  total lifetime, with inverse scaling by lifetime. Reported inverse-scaled
  metrics must be reconciled with the manuscript's HI metric wording before
  comparing numbers. Total lifetime may define offline labels, never model input
  or an operational RUL conversion using a held-out device's future lifetime.
- Existing XJTU domain_shift folds all hold out condition 3; they are not three
  independent held-out-condition evaluations.
- N-CMAPSS uses precomputed standard scalers; their training provenance must be
  checked, rather than assuming fit-on-train from the generic pipeline.

## Warning study preregistration (design only)

N-CMAPSS warning labels use remaining flight cycles. XJTU warning labels use
elapsed operating time and the recorded terminal time, separately from HI.
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
