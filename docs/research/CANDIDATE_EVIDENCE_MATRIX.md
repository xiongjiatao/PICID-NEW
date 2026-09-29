# PICID candidate contribution evidence matrix

Evidence reviewed 2026-09-29. This is a research decision record, not a novelty claim. The local 43-page TFM-PHM source and the official TabDPT v1.3 release were also checked. Recheck this landscape before method submission because relevant preprints are appearing rapidly.

Device-level validation and the prospective project-blinded C-MAPSS protocol
are frozen in [DEVICE_VALIDATION_AND_BLIND_TEST_PREREGISTRATION.md](DEVICE_VALIDATION_AND_BLIND_TEST_PREREGISTRATION.md).
The planned five-fold NC-P analysis uses the 20 canonical training engines, so
it is development cross-validation rather than untouched confirmatory evidence;
the previously accessed NC-P and XJTU test assets cannot be relabeled as blind.

## Candidate A — preserving temporal degradation evidence under a fixed inference budget

**Question.** Given identical observed history rows, context count, query points, model weights, and measured latency/memory budget, which temporal statistics must survive context reduction for cross-device RUL estimation?

**What is already covered.** TFM-PHM already turns PHM history into table rows, tests five window/stride settings, and reports that time tabularization changes performance. Its N-CMAPSS result is specifically DS02: neither TabPFN nor TabDPT improves with longer adjacent windows there; the authors argue useful degradation context may require many flight cycles, beyond their tested windows. This is a warning against equating a larger window with preserved degradation evidence, while the current NC-P subset combination DS01/04/05/07 remains a different, still-unmeasured setting. TACO (ICML 2026 spotlight) learns an end-to-end latent compressed training table for a tabular foundation model and reports large inference/memory reductions. Its main evaluation is tabular classification; its long-table demonstration uses time-split MetroPT-3 fault classification and chunked latent compression, not RUL or device-level degradation evidence. The official release provides TabPFN-TACO and TabPFN-POT checkpoints trained from scratch rather than using TabPFN/TabICL pretrained weights, and documents a `TACOClassifier` inference API; a drop-in regression reproduction is not yet verified. Thus generic learned context compression is occupied, but whether a RUL model must preserve temporal phase or transitions is not answered by that result. Plain PCA, representative-context retrieval, batching, or an engineering cache cannot be a contribution by themselves. TabDPT has native retrieval, feature reduction and v1.3 context reduction choices that need direct controls.

**Direct RUL-foundation-model prior art found 2026-09-29.** Two 2026 Chronos-2 studies materially narrow this candidate. El-Ghoussani et al. use a frozen Chronos-2 backbone and an MLP head for sensor-stream RUL on two industrial device types; their v3 preprint reports longer contexts improving performance up to about 80 steps. Its data are repair-log labeled, hourly-resampled and chronologically split 85:15, with overlap windows discarded at the split. This directly covers frozen temporal embeddings plus a light RUL head and the generic claim that longer context helps. Its disclosed protocol does not specify a holdout of independent assets, so it does not establish cross-device generalization; the two industrial datasets also differ from the NC-P/XJTU protocol. The preprint lists calibrated uncertainty from Chronos-2 probabilistic outputs, heterogeneous channel handling, and robustness to regime shifts as future work; these are open evaluation needs, not novelty by themselves. Abdouni et al. (PHM Society European Conference 2026) report a frozen Chronos-2 Wide & Deep RUL model on all four C-MAPSS subsets. They compress each sensor's 768-dimensional embedding to 8 dimensions, flatten sensor positions, and fuse these features with normalized raw sensors and operating-regime clusters; they report a mean RMSE of 10.32 over the four subsets and ten runs. This directly covers learned compression plus raw-sensor/condition fusion for RUL, including multi-regime C-MAPSS. Its published protocol holds out engines for train/validation but does not provide the same NC-P DS01/04/05/07 split, and its results are author-reported, not independently reproduced here. The paper explicitly lists N-CMAPSS validation as future work, which leaves a useful dataset-shift test but does not restore novelty to its compression/fusion mechanism. Accordingly, “frozen foundation embeddings + lightweight RUL head,” “longer context improves RUL,” and “compress embeddings then fuse raw sensors/conditions” are not viable novelty claims. A TabPFN/TabDPT-specific context-row question remains distinct, but the evidence currently supports only a controlled evaluation question, not a method gap.

**Material baseline omission found 2026-09-28.** The official [TabDPT v1.2 release](https://github.com/layer6ai-labs/TabDPT-inference/releases) introduces TabDPT-Turbo, long-context inference, no retrieval by default, new weights, and an author-reported average 120× speedup on TabArena. The [Turbo paper](https://arxiv.org/abs/2608.01400) describes row-based attention plus long-context pretraining to avoid retrieval. The official repository says v1.3 is similar to v1.2 with additional model/recipe changes. Therefore the frozen v1.1.13/v1.3.0 pair does not yet support a general claim that a new mechanism improves compute-bounded temporal evidence: v1.2 is the nearest released efficiency comparator and must be added as a supplementary baseline before making that claim. The isolated Python 3.12 / torch 2.9.1 environment and v1.2.0 package are installed and import successfully, but the pinned official checkpoint download currently ends with a proxy TLS EOF. No v1.2 model run or weight digest is claimed. The adapter now leaves `context_size=None` to preserve the official unbounded context default and uses the v1.2 prediction-time batch API. The v1.2 result should be reported as a separate version with its own package/weight digest and context semantics; aggregate version differences cannot be attributed to retrieval or any single change.

**Release check on 2026-09-29.** Official TabDPT v1.3.1 was released on September 21; its release page lists moving checkpoint loading out of estimator initialization as the change. The agreed v1.3.0 remains the frozen scientific baseline, and v1.3.1 is recorded as an inference-code update to evaluate separately only if initialization cost is part of a later efficiency claim. This does not change the current version comparison or justify pooling results across versions.

**Unresolved, narrower hypothesis.** Under an equal query-time and context-row budget, TabPFN/TabDPT's table-context mechanisms may discard ordering, local derivatives, or degradation transitions that a chronological window model retains. This is a falsifiable comparison against the new Chronos-2 RUL work, TACO, native TabDPT retrieval/reduction, and simple statistics—not a claim that temporal compression itself is novel. It becomes a method candidate only if a reproducible NC-P/XJTU failure mode survives device-disjoint validation and those matched controls.

**Required protocol.** Keep device split, query set, original history support, context rows, batch size and seed fixed. Compare original chronological windows, flattened-window baseline, training-fitted PCA, fixed multiscale statistics, uniform/device-balanced row subsampling, TabDPT native retrieval/subsampling, and TACO if its public implementation/checkpoint is runnable under a compatible protocol. Match both context-row count and measured peak memory/query latency. Report per-device MAE/RMSE, critical-stage errors, and confidence intervals bootstrapped over devices; report seed variation separately. Audit every produced window against its source device and raw time indices before fitting.

The current `unit_balanced_temporal` TabPFN fit-context sampler is one explicit
device-balanced subsampling control under the v2 10,000-row intended range. It
uses training `unit_id` only, keeps the validation/test query rows untouched,
and is an engineering baseline rather than a proposed contribution. NC-P
device-disjoint validation controls are now complete for seeds 72, 88, and
101. They compare raw chronological windows of 1, 10, and 50 rows, an
additional context-only PCA(128) transform on W50, and fixed multiscale
statistics on W50. All arms use the same 2,048 context rows, 1,024 query rows
on each of the same four validation engines, eight TabDPT v1.3 ensembles, and
the model's native PCA/context reduction. The validation queries are held
constant across representations and seeds.

| Representation | Input dimensions | All-stage normalized MAE | Critical 5% MAE | Critical 20% MAE |
|---|---:|---:|---:|---:|
| Raw W1 | 18 | 0.07493 ± 0.00038 | 0.01944 ± 0.00134 | 0.03428 ± 0.00142 |
| Raw W10 | 180 | 0.08585 ± 0.00157 | 0.01946 ± 0.00078 | 0.03514 ± 0.00228 |
| Raw W50 | 900 | 0.10578 ± 0.00085 | 0.01652 ± 0.00061 | 0.04074 ± 0.00072 |
| Context-only PCA(128) + W50 | 128 | 0.10468 ± 0.00244 | 0.01657 ± 0.00127 | 0.03996 ± 0.00255 |
| Fixed multiscale statistics + W50 | 270 | 0.09065 ± 0.00201 | 0.02274 ± 0.00124 | 0.03914 ± 0.00196 |

Values are mean ± sample SD over three stochastic seeds, not 12 independent
devices: the same four engines are reused in each seed. W1 has the best
all-stage error. W50 lowers the macro critical-5% error by about 15% relative
to W1, while raising all-stage MAE by about 41%; its critical-20% error is
worse. This small critical-5% difference is source-heterogeneous: it is
strongest on DS04 and does not reproduce uniformly on DS05/DS07. There is one
validation engine per source, so this cannot establish a source-level effect.
Context-only PCA barely changes W50's critical-5% score, and fixed multiscale
statistics do not preserve its small gain. These results motivate a larger,
device-resampled test of the trade-off; they do not yet show a general benefit
from longer histories or identify a novel mechanism. No test-set result was
used. Peak allocated memory was 0.484 GiB; wall time varied under concurrent
GPU load and is not suitable for ranking these representations. Per-seed
manifests and per-representation metrics are under
`artifacts/research/ncp_temporal_control_seed*_gpu*_20260929/`.

**Evidence needed before implementation.** The first validation-only accuracy comparison is now available, but its small, source-heterogeneous critical-5% gain does not isolate a degradation transition or survive a meaningful device-resampling analysis. The unit-boundary invariant has a synthetic regression test, and each generated history is constrained to one engine. Next, vary history support and query locations separately on additional device-disjoint validation folds, include TabDPT v1.2/Turbo when its checkpoint is obtainable, and report cold-start/steady-state latency under a recorded concurrency profile. Implement a mechanism only if a reproducible phase-specific information loss remains after those matched controls.

Before making a broad “industrial time-series foundation model” claim, add a compatible Chronos-2 RUL baseline or explicitly scope conclusions to the tabular foundation models evaluated. The two new studies show that omitting TSFMs would leave a direct recent RUL baseline family untested. Their tasks and data protocols differ from NC-P and XJTU, so their published metrics must not be ranked directly against PICID results.

**Falsification / stop rule.** Stop or reformulate if equal-budget PCA/statistics/native TabDPT preserve critical-stage metrics, if no robust phase-specific error appears across device resamples, if the gain disappears under matched chronological query points, or if a released general-purpose compressor already provides the same temporal guarantee. Current status: **not established**. The pilot batch-size tests in `artifacts/formal/ncp_tabdpt*_batch_preflight` do not test this hypothesis.

## Candidate B — reliable failure-proximity alerts from dynamic predictions

**Question.** Can a model produce useful, calibrated probabilities that a device enters a predeclared failure-proximity horizon, using only information available at the current timestamp and calibration devices independent of evaluation devices?

**What is already covered.** TFM-PHM studies predictive maintenance/prognostics and interprets TabPFN output probabilistically. TabDPT v1.3 adds a native full regression distribution with quantile/statistic helpers. “Predict an event probability,” a residual empirical CDF, a generic discrete-time classifier, or ordinary conformal intervals are not new alone. The 2026 *Tabular Foundation Models Can Do Survival Analysis* paper studies static and dynamic censored outcomes by time-discretized binary tasks and gives a consistency result under conditional independent censoring. *Tabular Foundation Models for Clinical Survival Analysis via Survival-Aware Adaptation* studies TabPFN/TabDPT/TabICL, interval reformulation and an MTLR head. SurvPFN studies a pretrained event-time density with an explicit censoring-aware likelihood. More directly, Alomari's 2026 IEEE Access paper combines sequence modeling, multitask RUL and failure-proximity outputs, conformal prediction intervals, and ID/OOD evaluation across multiple N-CMAPSS datasets; its framing directly overlaps the broad proposed “cross-condition RUL-to-warning” direction. A 2026 PHM Society paper also evaluates probabilistic RUL with conformal calibration on the same XJTU-SY full-lifecycle bearing dataset and reports a large coverage correction. These works make generic TFM survival, dynamic event risk, uncertainty calibration, and cross-condition warning occupied ground.

**New adverse evidence found 2026-09-29.** A July 2026 Zenodo artifact accompanies a manuscript submitted to *Reliability Engineering & System Safety* and describes a corpus-scale N-CMAPSS conformal study across nine subsets, 99 engines, and 20 unit-level replications. Its code uses a stratified 40/30/30 fit/calibration/test split by engines, evaluates split conformal, CQR, CV+, life-stage Mondrian intervals, covariate-weighted leave-one-subset-out calibration, and a one-sided safe-RUL floor. The released aggregate reports 0.898 coverage at nominal 0.90 and 0.098 late-prediction risk at target 0.10; leave-one-subset-out coverage is poor on DS01 and DS04, so the work itself reports a meaningful cross-condition failure boundary. This is direct prior art for calibrated RUL and conservative failure-proximity signals on the same corpus, including subsets used by NC-P. It does not report alert episodes, alarms per asset, or warning lead-time utility, so those remain possible operational evaluation questions rather than established method novelty.

The released artifact includes README, protocol, code, and JSON/CSV results, but not the submitted manuscript PDF. Its conformal quantiles are fitted over timestamp-level residuals after engine-disjoint splitting; the code also uses engine-cluster bootstrap for reported coverage intervals. The formal guarantee for temporally clustered calibration scores therefore needs checking against the manuscript before repeating the authors' finite-sample guarantee language. The reported results are authors' artifact outputs, not independently rerun here.

**Industrial gap to test, not presume.** The remaining plausible gap is deployment-aligned, device-level warning under limited independent failures and condition changes, with historical dependence, repeated alerts and a fixed maintenance horizon all accounted for. The stated NC-P split has 16 held-out engines (four in each of DS01, DS04, DS05 and DS07); the XJTU PHMD split has four test bearings. Those counts support device-level analysis but remain small for high-confidence false-alarm guarantees, especially for source-conditional claims with only four NC-P engines per source. N-CMAPSS benchmark terminal labels and XJTU dataset end points are operational proxies, not verified field maintenance events. This gap may be too weakly sampled to support a paper claim.

**Required comparisons.** RUL threshold; TabDPT 1.3 full CDF; TabPFN full output/residual ECDF; separately held-out device calibration; discrete-time event classifier and a small DeepHit/MTLR-style model; calibration off/on; condition and device-wise leave-out. Derive 5%, 10%, 20% horizons from training-device median life in native units, then freeze them. For XJTU derive labels from the raw timeline and never use an evaluation bearing's realized total life to create an online horizon.

**Compute-budget control.** The three probability-study runs each used a single explicitly recorded physical GPU and completed in 242–246 seconds, with 0.424 GiB peak allocated memory. These end-to-end times include loading, prediction, calibration, and alert metrics; they are descriptive under concurrent GPU load, not a clean latency ranking. The run manifests record the full command, physical/logical device, concurrent processes, seed, output path, and stage timings. The full-distribution route is a separately named inference protocol because its predicted means differ from the saved point-output route by 0.0135–0.0243, above the registered 1e-4 tolerance. Do not pool its probabilities with the point-prediction baseline or attribute their difference to calibration.

**Frozen XJTU horizon definition.** The eight PHMD training bearings have
median maximum RUL 354 acquisition intervals (RUL is `N-1` for `N` ordered
acquisitions). This fixes candidate horizons at 17.7, 35.4, and 70.8 intervals.
They are calculated from training-bearing filename indices only and recorded in
`artifacts/formal/results/xjtu_critical_horizons.json`. The warning study has
now been evaluated for seeds 72, 88, and 101 using eight-fold leave-one-training-
bearing-out predictions to fit Platt mappings and empirical alert thresholds.
Events are constructed from each bearing's raw native-RUL timeline. No test
bearing's lifetime is used to convert predicted HI to online RUL.

| Horizon | Probability source | Brier uncalibrated → Platt | AUPRC | Test warning-window detection | False-alarm episodes / bearing | Mean lead among detected (intervals) |
|---|---|---:|---:|---:|---:|---:|
| 5% | TabDPT native CDF | 0.0982 ± 0.0029 → 0.0943 ± 0.0017 | 0.5165 ± 0.1183 | 0.083 ± 0.144 | 0.00 ± 0.00 | 6.00 |
| 10% | TabDPT native CDF | 0.1929 ± 0.0038 → 0.1730 ± 0.0036 | 0.5082 ± 0.0422 | 0.083 ± 0.144 | 1.25 ± 1.75 | 7.00 |
| 20% | TabDPT native CDF | 0.2666 ± 0.0088 → 0.2140 ± 0.0073 | 0.6935 ± 0.0180 | 0.667 ± 0.144 | 2.00 ± 0.25 | 39.28 ± 0.95 |
| 5% | Residual ECDF | 0.0931 ± 0.0020 → 0.0945 ± 0.0019 | 0.4852 ± 0.0680 | 0.500 ± 0.000 | 4.75 ± 1.09 | 10.33 ± 2.84 |
| 10% | Residual ECDF | 0.1917 ± 0.0032 → 0.1749 ± 0.0041 | 0.4779 ± 0.0206 | 0.333 ± 0.144 | 1.58 ± 2.13 | 11.50 ± 8.67 |
| 20% | Residual ECDF | 0.2587 ± 0.0096 → 0.2187 ± 0.0079 | 0.6711 ± 0.0115 | 0.417 ± 0.144 | 1.00 ± 1.15 | 23.33 ± 15.78 |
| 5% | Direct event logistic | 0.0780 ± 0.0000 → 0.0924 ± 0.0000 | 0.6527 ± 0.0000 | 1.000 ± 0.000 | 0.50 ± 0.00 | 10.50 ± 0.00 |
| 10% | Direct event logistic | 0.1540 ± 0.0000 → 0.1763 ± 0.0000 | 0.6299 ± 0.0000 | 1.000 ± 0.000 | 1.25 ± 0.00 | 12.00 ± 0.00 |
| 20% | Direct event logistic | 0.1846 ± 0.0000 → 0.2268 ± 0.0000 | 0.7716 ± 0.0000 | 1.000 ± 0.000 | 1.00 ± 0.00 | 33.00 ± 0.00 |

Values are seed means ± sample SD; direct logistic scores are identical across
seeds. Lead time is conditional on test bearings detected within the warning
window, and seed runs with no detected bearing contribute no lead value. Alert
thresholds were selected from out-of-fold training-bearing scores at an
empirical device-macro FPR near 5%; this is not a population FPR guarantee.
Platt calibration improves the native CDF Brier score at each horizon but does
not yield strong 5%/10% detection on the four public test bearings. Residual
ECDF has higher short-horizon detection and lead but more 5% false-alarm
episodes. The direct event classifier ranks better and detects all four
bearings in this split, while Platt worsens its Brier scores at all horizons.
These are competing trade-offs, not evidence that one approach is calibrated
and operationally superior.
Per-seed reports are in
`artifacts/research/xjtu_warning_study_seed*_gpu*_20260929/manifest.json`.

**Metrics and uncertainty.** Device-balanced Brier/log loss and calibration plots, event-risk ranking/AUPRC, detection at calibration-fixed empirical false-alarm levels, alarms per device, missed events and lead-time distribution. Bootstrap whole devices. State explicitly when calibration sample size cannot support a claimed bound; do not treat correlated timestamps as independent Bernoulli trials. Compare prediction-distribution coverage separately from decision quality.

**Finite-sample calibration constraint.** Treat a complete device/bearing as the exchangeable statistical unit. Under standard split conformal, the finite-sample order-statistic index is `ceil((n+1)(1-alpha))`; a 90% interval needs at least nine independent calibration units for a finite quantile. XJTU has only eight PHMD training bearings in total, while its three validation bearings are also insufficient; NC-P has four validation engines. Thus a 90% device-level split-conformal guarantee cannot be claimed from the existing XJTU set or NC-P validation set. Adding calibration engines inside NC-P's 20 training units is possible only by reducing the fitting pool, and must be explicitly budgeted before fitting. Resampling timestamp residuals cannot manufacture independent devices. The alert study may still report descriptive calibration/utility curves, but its guarantee and uncertainty language must match the actual number of held-out assets.

**Decision after literature refresh and first evaluation.** The broad direction—“turn RUL predictions into cross-condition failure warnings with uncertainty”—is already covered and is rejected as a novelty claim. The observed XJTU results do not establish a superior warning method: native CDF calibration improves Brier but misses most 5%/10% events; residual ECDF buys short-horizon detection at a larger false-alarm burden; the direct event classifier ranks and detects better here but becomes less calibrated after Platt scaling. Only four public test bearings were evaluated, and their test histories had been accessed in earlier baseline work. These results are descriptive replication evidence, not blind external validation or a reliable false-alarm guarantee. A possible next study is a prospectively frozen, device-disjoint comparison of alert burden and lead time, with enough independent assets and a deployment-defined intervention label. Until then, this remains an evaluation question, not an established method gap.

**Falsification / stop rule.** Stop if the native predictive CDF or a simple calibrated RUL threshold matches warning utility at the same false-alarm burden, if independent calibration devices are too few, if results depend on evaluation terminal life, or if horizon labels cannot represent an observable maintenance decision. Current status: **not established; likely an evaluation/design contribution unless the mechanism addresses a measured failure mode**.

## Source register

- Theiler et al., [Towards Unified and Data-Efficient Prognostics and Health Management with Tabular Foundation Models](https://arxiv.org/html/2606.05481v1), especially its time-tabularization experiments and probabilistic-output discussion.
- El-Ghoussani et al., [Time-Series Foundation Model Embeddings for Remaining Useful Life Estimation](https://arxiv.org/html/2606.11990v3), arXiv v3 dated August 11, 2026. The full text reports frozen Chronos-2 embeddings with an MLP RUL head, two industrial sensor datasets, chronological 85:15 splits, and context-length analysis through 80 steps; the described split does not specify independent-asset holdout.
- Abdouni et al., [Leveraging Time Series Foundation Models Embeddings for Remaining Useful Life Prediction](https://doi.org/10.36001/phme.2026.v9i1.4906), PHM Society European Conference 2026, 9(1), 1–7. The full paper reports frozen Chronos-2 compression-and-flattening plus raw sensor/regime fusion on C-MAPSS FD001–FD004, engine-disjoint train/validation selection, ten runs, and mean RMSE 10.32; these are author-reported results and have not been independently reproduced here. Its stated future work includes N-CMAPSS validation and edge-latency reduction.
- Zabërgja et al., [End-to-End Compression for Tabular Foundation Models](https://arxiv.org/html/2602.05649v1), ICML 2026 spotlight; see also the authors' [official TACO implementation](https://github.com/machinelearningnuremberg/TACO). It learns a latent compressed table jointly with its predictor; this is direct overlap with generic context compression.
- [Tabular Foundation Models Can Do Survival Analysis](https://arxiv.org/html/2601.22259v1), especially its dynamic conditional survival formulation, interval tasks and assumptions.
- [Tabular Foundation Models for Clinical Survival Analysis via Survival-Aware Adaptation](https://arxiv.org/html/2606.12006v1), especially its TabPFN/TabDPT/TabICL evaluation and MTLR head.
- [SurvPFN: Towards Foundation Models for Survival Predictions](https://arxiv.org/html/2606.04564v1), especially its censored density/ranking objective and event-time distribution.
- [Official TabDPT v1.3.0 release](https://github.com/layer6ai-labs/TabDPT-inference/releases/tag/v1.3.0), which adds full probabilistic regression output and `BarDistribution` helpers.
- [Official TabDPT v1.3.1 release](https://github.com/layer6ai-labs/TabDPT-inference/releases/tag/v1.3.1), released September 21, 2026; the release page lists checkpoint-loading relocation as its change. v1.3.0 remains the frozen baseline for this protocol.
- [Official TabDPT v1.2 release](https://github.com/layer6ai-labs/TabDPT-inference/releases/tag/v1.2.0) and [TabDPT-Turbo paper](https://arxiv.org/abs/2608.01400), for the recent long-context/no-retrieval speed baseline. The speed figure is the authors' TabArena claim, not an observed PICID result.
- Alomari, [A Unified Uncertainty-Aware Multi-Task Framework for Robust Remaining Useful Life Prediction Under Distribution Shift](https://doi.org/10.1109/ACCESS.2026.3685622), IEEE Access 14 (2026), 60940–60962. Full author manuscript checked at [ResearchGate](https://www.researchgate.net/publication/404015361_A_Unified_Uncertainty-Aware_Multi-Task_Framework_for_Robust_Remaining_Useful_Life_Prediction_under_Distribution_Shift); the paper directly combines RUL, failure proximity, conformal intervals and N-CMAPSS ID/OOD evaluation.
- Wang, Vidal, and Pozo, [Uncertainty-Aware Bearing Remaining Useful Life Prediction Based on Conformal Prediction](https://doi.org/10.36001/phme.2026.v9i1.4902), PHM Society European Conference 2026, DOI 10.36001/phme.2026.v9i1.4902. It uses the XJTU-SY full-lifecycle dataset and calibrates probabilistic RUL intervals; the venue's [full paper/PDF](https://www.papers.phmsociety.org/index.php/phme/article/download/4902/2943) is linked here.
- Yan, [Audited conformal calibration for RUL prognostics on N-CMAPSS: code and results](https://zenodo.org/records/21281080), Zenodo record published July 9, 2026, DOI 10.5281/zenodo.21281080. The record describes an accompanying manuscript submitted to *Reliability Engineering & System Safety*. Its abstract is direct adverse prior art for N-CMAPSS RUL calibration; the full ZIP manuscript has not yet been inspected, so no finer novelty conclusion is made.
- Żukowska et al., [Towards Long-Context Time Series Foundation Models](https://openreview.net/pdf?id=FNjddk8ckA), NeurIPS 2024 Workshop, for compressive memory as prior art in long multivariate temporal context. Its task is forecasting, so transfer to RUL remains an inference, not direct evidence of equivalent performance.
