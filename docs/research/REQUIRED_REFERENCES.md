# Mandatory three-paper reference contract

The user designated these local PDFs as mandatory on 2026-09-26. They serve
different roles and must be checked together. They are not three independent
replications of our proposed method. All three supplied PDFs identify arXiv v1;
do not assert peer-reviewed acceptance without separate verification.

| Reference | Role | Evidence read in this bootstrap | Required artifact |
|---|---|---|---|
| Theiler et al., *From paper to benchmark: agentic, framework-based reproduction of under-specified methods in machine health intelligence*, [2605.28371](https://arxiv.org/abs/2605.28371) | Reproduction process | pp. 1–7, especially §3.1–3.5 | Paper-indexed six-slot binding; explicit assumptions; static/sanity/result-matching status and failure attribution |
| Telyatnikov et al., *Picid: A Modular Evaluation Infrastructure for Reproducible PHM Across Tasks and Domains*, [2605.28345](https://arxiv.org/abs/2605.28345) | Evaluation contracts | pp. 1–6, especially §3 and §4.1 | Train-only transform state, time-support/target alignment, device split and evaluation-unit evidence |
| Theiler et al., *Towards Unified and Data-Efficient Prognostics and Health Management with Tabular Foundation Models*, [2606.05481](https://arxiv.org/abs/2606.05481) | Model and experiment replication; closest prior work | §3–5, §6.0.5, Appendix A and C relevant passages | Exact target, window/stride, context sampling, seed/configuration and metric correspondence |

This is a section-level reading record, not a claim that every page and cited
paper has already been reviewed. Appendices on task contracts, data variants,
and per-unit evaluation remain required before formal performance claims.

## Six-slot binding and assumptions

| Slot | Existing binding / evidence | Current discrepancy or missing evidence |
|---|---|---|
| task | N-CMAPSS RUL; XJTU HI, TFM paper Appendix A | HI inverse-scaled reporting differs from a literal HI-error interpretation; inspect ground-truth transport |
| datasource | MultiSourceLoader DS01/04/05/07; XJTU_SYLoader PHMD split | Raw HDF5 files missing locally; filename variants and split IDs need census |
| transform | Shared N-CMAPSS aggregation and XJTU descriptors | Released configs use `aggregation`, class uses `agg`; default mean conflicts with requested last |
| sequencer | Original five window/stride candidates | Actual query timestamps and prefix-only access still require runtime tests |
| model | Locked TabPFN fork, TabDPT, XGBoost, LSTM wrappers | Environment installed; real checkpoint/runtime checks are not yet evidence of PHM performance |
| evaluator | RUL/per-unit evaluators | Prove inverse scaling semantics and compare pooled versus device-macro metrics |

Explicit assumptions: seed 72 is the first smoke run (execution ordering, not a
paper claim); 5/10/20% training-median-life warning horizons are a proposed new
protocol (not specified in the papers); `aggregation_corrected` is an opt-in
repair of configuration intent (not an exact reproduction of released behavior).
Each resolved configuration and failure report must retain that protocol name.

Verification follows *From paper to benchmark*: static checks, model sanity, and
result matching are separate gates. Gradient-flow/microbatch memorization checks
apply to trainable LSTM; frozen ICL models instead require valid support/query
contracts and deterministic replay. A passed test suite or composed configuration
does not imply successful result reproduction.
