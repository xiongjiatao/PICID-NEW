# Initial novelty evidence matrix — not a completed novelty claim

Checked 2026-09-26. The supplied 43-page PHM PDF is the primary local replication
reference. Other entries below were checked on primary publication/abstract
pages; their full methods must be read before algorithm-level novelty claims.

| Prior work | Verified scope | Overlap with proposed work | Required comparison / missing evidence |
|---|---|---|---|
| [From paper to benchmark, arXiv 2605.28371 (2026)](https://arxiv.org/abs/2605.28371) | Six framework slots, explicit assumptions and separate verification gates (§3, local PDF) | Agentic experiment automation and framework binding are already studied | Follow this methodology; do not claim automation itself as either new predictive method |
| [Theiler et al., arXiv 2606.05481 (2026)](https://arxiv.org/abs/2606.05481) | PHM tabularization, TabPFN/TabDPT, temporal windows, data efficiency and representative context | Generic TFM+PHM and context subsampling are already covered | Same splits, representation and context budget; current code versus supplied PDF needs reconciliation |
| [PICID infrastructure, arXiv 2605.28345 (2026)](https://arxiv.org/abs/2605.28345) | Reusable PHM evaluation infrastructure | A new wrapper or reproducible runner is engineering infrastructure | Preserve data/evaluation contracts; do not claim a novel predictor for infrastructure work |
| [DeepHit, AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11842) | Direct discrete event-time distribution modeling, competing risks | Multi-horizon event probabilities alone are established | Compare a small event-time model with the same covariates; detailed dynamic-update and ranking mechanisms still need full-text review |
| [Learn then Test, arXiv 2110.01052](https://arxiv.org/abs/2110.01052) | Model-agnostic calibration for finite-sample risk control | Independent risk calibration is not itself a new method | Specify independent statistical units and actual loss; 2021 initial version, v5 2022 |
| [Conformal Risk Control, arXiv 2208.02814](https://arxiv.org/abs/2208.02814) | Conformal control of risk under stated assumptions | Generic conformal calibration is existing methodology | Distinguish expectation guarantees from high-probability guarantees; verify applicability to device-level dependent trajectories |

Candidate questions, not established contributions:

1. Does task-specific temporal evidence improve failure-proximity ranking beyond
   the frozen TFM's point prediction and native predictive distribution?
   Required controls: point-threshold, native predictive CDF, empirical residual
   CDF, small direct classifier/event-time model, dynamic-evidence ablation.
2. At equal inference cost, can cross-condition warning performance improve
   without exploiting held-out-device terminal information?
   Required controls: original random/block context selection, random device-
   balanced context, simple nearest-neighbor retrieval, calibrated versus raw
   probabilities, and condition-level results.

The two questions may overlap with other recent work. A single authoritative
paper cannot prove two new methods novel, and this matrix does not establish
that novelty.

## 2026-09-28 evidence refresh

Full-text HTML was additionally reviewed for [End-to-End Compression for Tabular Foundation Models (TACO, arXiv:2602.05649)](https://arxiv.org/html/2602.05649v1), [Tabular Foundation Models Can Do Survival Analysis (arXiv:2601.22259)](https://arxiv.org/html/2601.22259v1), [Tabular Foundation Models for Clinical Survival Analysis via Survival-Aware Adaptation (arXiv:2606.12006)](https://arxiv.org/html/2606.12006v1), and [SurvPFN (arXiv:2606.04564)](https://arxiv.org/html/2606.04564v1). TACO rules out a generic learned-table-compression claim. The 2026 survival papers rule out generic dynamic TFM failure-risk formulation as novelty; any remaining warning direction needs a more specific device-level industrial calibration question, while the available evaluation units remain small. Candidate contribution directions and stop criteria are detailed in [CANDIDATE_EVIDENCE_MATRIX.md](CANDIDATE_EVIDENCE_MATRIX.md).

## Evidence refresh: 2026-09-28

The [official TabDPT v1.2.0 release](https://github.com/layer6ai-labs/TabDPT-inference/releases/tag/v1.2.0) and [TabDPT-Turbo paper](https://arxiv.org/abs/2608.01400) reveal a major missing efficiency comparator in the current v1.1.13/v1.3.0 plan: v1.2 changes to long-context, no-retrieval defaults and reports approximately 120× average speedup on TabArena. The v1.2 package is installed in the isolated Python 3.12 / torch 2.9.1 environment, but the pinned official checkpoint download currently terminates with a proxy TLS EOF; no weight digest or model result is available. Therefore the current version comparison is valid for its stated pair, but it cannot support a general claim that its approach is compute-efficient relative to current released TabDPT.

The broad warning direction is also substantially occupied. Alomari's [2026 IEEE Access framework](https://doi.org/10.1109/ACCESS.2026.3685622) jointly predicts RUL and failure proximity, adds conformal intervals, and evaluates N-CMAPSS ID/OOD generalization. Wang et al.'s [2026 XJTU-SY conformal RUL study](https://doi.org/10.36001/phme.2026.v9i1.4902) directly covers probabilistic bearing RUL and calibration on the same dataset. This evidence rejects generic “RUL-to-cross-condition warning with uncertainty” as a novelty claim; only a narrower repeated-device alert-burden/lead-time evaluation question remains, and the four XJTU test bearings are too few to establish strong deployment guarantees. See the refreshed evidence and finite-sample constraints in [CANDIDATE_EVIDENCE_MATRIX.md](CANDIDATE_EVIDENCE_MATRIX.md).
