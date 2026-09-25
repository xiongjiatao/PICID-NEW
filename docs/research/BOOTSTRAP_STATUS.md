# Execution status — 2026-09-26

## Completed

- Stopped the four old project GPU workers (PIDs 3912378, 3948670, 413980,
  413670) and two D_select controllers (3912309, 3948605), by user request.
  All six exited after SIGTERM. GPU process query was empty afterwards.
- Imported 1,812 source/config/test/document files as `d9d93ee` and pushed `main`
  to `xiongjiatao/PICID-NEW`. HTTPS push failed due to a stale VS Code credential
  socket; existing SSH authentication succeeded. HTTPS fetch URL is retained;
  the same repository uses SSH for push. No credentials were copied into Git.
- Installed the supplied frozen uv.lock into the copy's independent `.venv`:
  Python 3.12.14, torch 2.9.1, TabPFN 2.2.1 (paper fork at ad335050),
  TabDPT 1.1.13 (f43d7fdc), PHMD 2025.0.4 (512426b4), pytest 9.0.2.
- All eight selected dataset/model configurations compose, both released and
  explicit aggregation-corrected variants. Released variants have a confirmed
  mismatch: `aggregation=last` yields a mean. Numeric counterexample [1,4,10]
  produces 5; explicit `agg=last` produces 10. Core model/transform defaults remain
  untouched. Corrected variants remove this mismatch without implying a full pass.
- Targeted verification: 55 tests passed (audit, data extraction, XJTU split,
  aggregation and HI tests); Ruff and `git diff --check` passed. This is not a
  full-repository test run and does not establish runtime causality or accuracy.
- Three required PDFs are indexed by filename, SHA256 and role in generated
  audit artifacts; section-level evidence is recorded in REQUIRED_REFERENCES.md.

## Pending before model experiments

- N-CMAPSS raw DS01/04/05/07 HDF5 files were not found in the searched project/cache
  paths. NASA's official repository lists a reachable 15,760,443,389-byte archive:
  [NASA source](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
  The data-preparation script downloads this archive and extracts just those four
  sources. It records the original DS01-005 filename before making a local DS01
  alias by byte-preserving extraction. Live status belongs in ignored artifacts.
- Existing local XJTU raw data is present, but its equivalence to the locked PHMD
  loader layout has not been established; do not substitute silently.
- Resolve precomputed N-CMAPSS scaler provenance, XJTU target/inverse-scale
  interpretation, and model checkpoint provenance. Runtime causality, prefix
  mutation, cache equivalence and full seed-72 evaluation are still outstanding.
- Wider full-text novelty comparison and two-method decisions remain open.

No PHM performance claim, five-seed result, or real maintenance-event guarantee
has been established. The current gate is `BLOCKED_PROTOCOL_OR_DATA`, not a
successful scientific reproduction. Data preparation may proceed on CPU while
the GPU model-experiment gate remains closed.
