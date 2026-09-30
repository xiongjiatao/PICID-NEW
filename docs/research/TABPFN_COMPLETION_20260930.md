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

## Results and current status — 2026-09-30

XJTU TabPFN's three-seed `low_memory` / `yield32` alternate is now complete.
All three prediction exports reconcile with per-bearing metrics (2,261 rows on
four bearings per seed). Across seeds, normalized device-macro MAE is
0.178463 (sample SD 0.001536); offline-denormalized device-macro MAE is
127.343 (SD 2.423), and PHM score is 0.297971 (SD 0.003923). The bearing-level
95% bootstrap interval for normalized macro MAE is [0.10790, 0.24903], based
on four test bearings; it is not a population-level guarantee. Public test
access and the low-memory/cached execution difference remain disclosed.

The native TabPFN `memory_saving_mode=True` retry3 OOMed under GPU contention:
its tracked peak was 15.85 GiB, another process used 2.18 GiB, and only 5.65 GiB
remained for a 6.47-GiB allocation. That external process was not stopped.
Retry4 used the same memory-saving mode on an otherwise empty GPU0 and still
OOMed: this process held 22.32 GiB, only 1.36 GiB was free, and the next
allocation required 6.47 GiB. PyTorch reported 7.29 GiB reserved but
unallocated, so retry5 keeps the training rows and model settings fixed while
testing `PYTORCH_ALLOC_CONF=expandable_segments:True`. The environment setting
is stored in the run manifest and, if selected, carried into every final seed.
The first retry failed before data loading because a nested symlink was created
in an existing empty `datasets` directory; that attempt is retained. The later
retry resolved the canonical dataset path and reached model fitting.

Retry5 reproduced the same 22.32-GiB OOM, so allocator configuration alone did
not solve it. A local overlay now raises TabPFN 2.2.1's activation chunk factor
from8 to16. It is built from the locked package into the result tree; neither
the virtual environment nor installed dependency is edited. The patch changes
only inference activation partitioning. A seed72 synthetic probe with 256 rows,
360 features and32 queries produced bit-identical predictions for factors8 and
16 (maximum difference0; tolerance1e-4); allocated peak was591 MiB for both, so
this probe supports numerical equivalence but does not establish high-context
memory reduction. Each run manifest records and verifies the patched package
hash and overlay-manifest hash. NC-P w20 retry6 is a separate, explicitly
identified validation attempt using the factor16 overlay.

NC-P 10/5 remains in validation on GPU1 and 50/50 remains in validation on
GPU4. The original 20/5 default execution, retry4 and retry5 all failed with
OOM. The factor16 reconciled controller reuses audited 1/1 and 5/1, waits for
the in-flight 10/5 and 50/50, then validates 20/5 retry6. Its final-seed memory
estimate uses the selected candidate's measured peak, with the tracked runner
preserving 1 GiB of headroom. It freezes a configuration only after every
registered candidate has a finite validation result. If 20/5 still fails,
there is no five-candidate selection or final three-seed claim. If all five
complete, the selected configuration is evaluated on seeds72/88/101 in
parallel on GPUs2/3/6, with per-seed prediction audits before aggregation.
Retry6 failed on GPU0 with the verified factor16 overlay and the same OOM
allocation request; its controller is waiting on the matching existing 10/5
validation and will retain that failure rather than launch a duplicate.

The five-candidate search remains incomplete because 20/5 could not fit after
three isolated OOM attempts. A separate resource-feasible selection plan uses
only the four completed-or-in-flight candidates at 1/1, 5/1, 10/5 and 50/50;
its frozen configuration hashes the explicit 20/5 exclusion and all failure
manifests. Any resulting three-seed evaluation is a resource-feasible subset
baseline, not a complete five-candidate search or a paper-exact result.
