# PICID reproduction and research preparation

## Scope and provenance

The three mandatory papers have been cross-reviewed in full in
[FULL_TEXT_CROSS_REVIEW.md](FULL_TEXT_CROSS_REVIEW.md). This protocol follows
the target TFM-PHM paper and treats the PICID infrastructure paper as the
framework specification.

The unmodified source snapshot is commit `d9d93ee`. The sibling `picid-release`
directory is read-only. This repository is an academic research derivative of
PICID; the original LICENSE.txt and authorship are retained. Initial import is
a local source snapshot, not a reconstruction of upstream Git history.

Targeted reproduction models: TabPFN, TabDPT, XGBoost, and LSTM. The main task is
N-CMAPSS NC-P over DS01/04/05/07. XJTU-SY is reproduced separately with the
TFM-PHM paper's PHMD split (8/3/4); its existing leave-condition evaluation is a
separate generalization result. These are independently fitted tasks, not
zero-shot transfer between turbofans and bearings.

Paper seeds: 72, 88, 101, 666, 226688. First smoke seed: 72. Fit-predict
window/stride candidates: (1,1), (5,1), (10,5), (20,5), (50,50).
Five random seeds quantify algorithm randomness, not independent asset evidence.

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
   time/peak memory; then repeat the paper's five seeds. Use only physical GPUs
   0/1/2 with explicit CUDA_VISIBLE_DEVICES and record logical cuda:0 separately.

All runtime artifacts belong in ignored `artifacts/`. Record command, commit,
resolved configuration, dependency versions, data/checkpoint digests, GPU mapping,
concurrent processes, exit code, and final status. Archive validation and
configuration preflight passed. The first full XJTU seed-72 LSTM run is active on
physical GPU0 (`CUDA_VISIBLE_DEVICES=0`, logical `cuda:0`); at the current status
snapshot it is generating `time_domain_features` for Bearing3_1, so model fitting
and performance metrics remain pending.

## Resolved findings and remaining evidence

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
