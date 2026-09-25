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

The two questions may overlap with other recent work. Dedicated full-text
retrieval/context, survival-TFM, and industrial early-warning literature searches
remain outstanding. A single authoritative paper cannot prove two new methods
novel, and this matrix does not establish that novelty.
