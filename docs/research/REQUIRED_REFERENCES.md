# Mandatory three-paper reference contract

The user designated these local PDFs as mandatory on 2026-09-26. Their full-text
cross-review and code reconciliation are recorded in
[FULL_TEXT_CROSS_REVIEW.md](FULL_TEXT_CROSS_REVIEW.md). They are not three
independent replications of a proposed predictor. All supplied PDFs identify
arXiv v1; do not assert peer-reviewed acceptance without separate verification.

| Reference | Role | Evidence read in this bootstrap | Required artifact |
|---|---|---|---|
| Theiler et al., *From paper to benchmark: agentic, framework-based reproduction of under-specified methods in machine health intelligence*, [2605.28371](https://arxiv.org/abs/2605.28371) | Reproduction process | Full 37-page text; main method/results, appendices B/C/E | Staged paper-to-framework binding; explicit assumptions; separated runnable/scientific verdicts |
| Telyatnikov et al., *Picid: A Modular Evaluation Infrastructure for Reproducible PHM Across Tasks and Domains*, [2605.28345](https://arxiv.org/abs/2605.28345) | Evaluation contracts | Full 67-page text; formalization, library, transform/data/evaluator appendices | Train-only fit contract, target alignment, split regimes, N-CMAPSS fixed scalers, XJTU target/split, metrics |
| Theiler et al., *Towards Unified and Data-Efficient Prognostics and Health Management with Tabular Foundation Models*, [2606.05481](https://arxiv.org/abs/2606.05481) | Model and experiment replication; closest prior work | Full 43-page text; methods, model details, results, dataset and reproducibility appendices | TFM tabularization/ICL, splits, target/scaling, sampling, five seeds, metrics, code correspondence |

The extracted full texts were cross-checked against current PICID code and config.
This is a review of the three supplied papers, not a review of every citation in
their bibliographies.

## Six-slot binding and assumptions

| Slot | Existing binding / evidence | Current discrepancy or missing evidence |
|---|---|---|
| task | N-CMAPSS RUL; XJTU normalized RUL/health trajectory, TFM Appendix A and locked PHMD reader | XJTU `rul_key` now states input semantics; HI inverse returns minutes |
| datasource | MultiSourceLoader DS01/04/05/07; XJTU_SYLoader PHMD split | Raw HDF5 files missing locally; filename variants and split IDs need census |
| transform | Shared N-CMAPSS aggregation and XJTU descriptors | Alias bug fixed; intent and pre-fix mean behavior get distinct audit protocol names |
| sequencer | Original five window/stride candidates | Actual query timestamps and prefix-only access still require runtime tests |
| model | Locked TabPFN fork, TabDPT, paper-reported XGBoost, LSTM wrappers | Imported upstream mislabeled the sklearn GradientBoosting control as XGBoost; the improved copy now calls locked XGBoost 3.1.3. It retains 1,000 rounds/seed 42 but uses library defaults for undisclosed tree settings. Keep this assumption visible; runtime checks alone are not PHM evidence |
| evaluator | RUL/per-unit evaluators | Report normalized HI and inverse acquisition-minute per-unit metrics separately; full model/runtime replay remains outstanding |

Explicit assumptions: seed 72 is the first smoke run (execution ordering, not a
paper claim); 5/10/20% training-median-life warning horizons are a proposed new
protocol (not specified in the papers). Each resolved config records either
`paper_intent_fixed` or `released_behavior_legacy` for aggregation behavior.

Verification follows *From paper to benchmark*: static checks, model sanity, and
result matching are separate gates. Gradient-flow/microbatch memorization checks
apply to trainable LSTM; frozen ICL models instead require valid support/query
contracts and deterministic replay. A passed test suite or composed configuration
does not imply successful result reproduction.
