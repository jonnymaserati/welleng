# Vendored ISCWSA error-model JSON schema

Source: <https://github.com/iscwsa/error-models>
Pinned commit: **`c7af7848b8407256550f8c19fd75c5626dbd6787`** (vendored 2026-04-16)

This directory mirrors the `schemas/` tree from that repository at
the pinned commit. Files are unmodified — see the upstream repo for
the canonical authoring history.

## When ISCWSA's schema changes

1. Update the pin: `cd /tmp && git clone https://github.com/iscwsa/error-models && cd error-models && git rev-parse HEAD`
2. Replace this directory's contents from `schemas/`.
3. Update the SHA above.
4. Run the conformance harness (`pytest tests/test_iscwsa_json_conformance.py`).
   The suite exercises every converted OWSG tool model under both the
   legacy hand-coded path and the new JSON+interpreter path; any
   schema change that breaks the agreement matrix shows up as a
   failing test, not a silent regression.

The agreement matrix at the previous pinned SHA is recorded in
`tests/iscwsa_json_baseline.txt` so we can diff "before vs after" for
any proposed schema revision.

## Why pin?

- ISCWSA's schema is in active early-stage development (their own
  `mwd_rev5.jsonc` carries TODOs about UUID format, hashing, and
  field semantics). A floating dependency would silently surface
  upstream churn as welleng test failures.
- A pinned vendored copy keeps welleng builds reproducible and lets
  us advance deliberately when there's an actual schema change worth
  tracking.
