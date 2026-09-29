# TabPFN completion protocol — 2026-09-30

The completion controller is `scripts/research/complete_tabpfn.py`. Local plans,
commands and controller logs are in `artifacts/formal/tabpfn_completion_20260930`.
These are ongoing runs, not completed results.

| Protocol | Selection | Final evaluation | Assigned physical GPU |
|---|---|---|---|
| XJTU low_memory / yield32 | Existing frozen 50/50, seed72 validation | Existing audited72; restart88 and101 | 0 and2 |
| NC-P balanced10k / cached / yield32 | Reuse audited1/1 and5/1; evaluate10/5,20/5,50/50 with test=false | Freeze only after all five succeed; then72,88,101 | 1 |

Separate result namespaces are
`artifacts/formal/results/xjtu_tabpfn_lowmem_yield32` and
`artifacts/formal/results/nc_p_tabpfn_balanced10k_cached_yield32`.
Reports use the frozen configuration digest; aggregation rejects mixed digests.
Existing process attempts, failures and interrupted logs are preserved. Reuse
requires exit0, exact method overrides, validation-only log evidence and finite
validation metrics. Final stages require saved prediction/per-device metric
reconciliation before inclusion. A failed candidate is retained and subsequent
candidates may run, but no incomplete-grid selection is allowed.

The XJTU low-memory execution failed numerical equivalence with the cached
execution and remains a separate alternative, not a paper-matched reproduction.
NC-P caps fit context at10,000 device-balanced temporal rows; the original
full-context failure remains part of the record. No speed or accuracy gain is
attributed to this budget change without a matched comparison. Window50 exceeds
the intended TabPFN feature range and is reported as such, not silently omitted.

XJTU admission budgets use16,000 MiB (observed seed72 peak12,698 MiB). NC-P10/5
uses5,000 MiB;20/5 and50/50 reserve23,000 MiB plus1,024 MiB headroom. For50/50,
this is a feasibility ceiling, **not a measured peak or a guarantee of fit**.
The controller waits for this memory budget without interrupting other jobs;
it also requires64,000 MiB host memory available before launch. It records
resource waiting separately from computation. Failure does not trigger a hidden
reduction in context, ensemble size, window length or test query count.

A prior XJTU seed72 final took21,957 s; this is an observed reference, not an ETA
for co-scheduled seeds. NC-P candidate runtime and feasibility remain unresolved.
The user later authorized parallel work on physical GPUs0–6, superseding the
0–2 restriction above. Existing Chronos2 and PICID TabDPT jobs are retained. The reported process-tree RSS of about208GB
sums shared forked memory and is not evidence of208GB unique physical memory;
host MemAvailable is checked directly for scheduling.


## Parallel scheduling update — 2026-09-30

With user authorization expanded to GPU0–6, the still-pending NC-P 20/5 and
50/50 seed72 validation stages are launched on free GPUs3 and4. Their output
paths and commands match the queued controller plan, so it can verify and reuse
them after its 10/5 stage finishes instead of running either candidate twice.
Both admission checks require23,000 MiB estimated peak plus1,024 MiB reserve;
this is only a ceiling and the 50/50 candidate may fail. Existing GPU0–2 tasks
continue. GPUs5–6 remain available for additional stages after observed memory
and process checks.
