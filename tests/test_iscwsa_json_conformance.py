"""Locks in the parallel-paths conformance matrix.

The harness in ``welleng.errors.conformance`` runs every term that exists in
*both* the legacy hand-coded dispatcher and the new JSON+interpreter path
on the ISCWSA Standard Test Well 1, and reports whether the two agree.

These tests pin the agreement count per tool model. Regressions in either
path (a hand-coded weight function changing, or the interpreter losing
precision on a formula) light up here. The schema-gap categories
(``MDPrev`` / ``AzPrev`` / ``IncPrev`` / per-tool calibration constants)
are also pinned so that any future schema fix that closes a gap shows up
as a deliberate test update rather than slipping silently into the totals.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from welleng.errors.conformance import (
    compare_model,
    standard_test_survey,
)

JSON_ROOT = Path(__file__).parent.parent / "welleng" / "errors" / "iscwsa_json" / "owsg_a"


# (json filename, expected MATCH count, expected DIFFER count, expected
# INTERP_FAILED count, expected NO_LEGACY count, total terms).
# Numbers come from the audited 2026-04-16 baseline run.
EXPECTED = [
    ("MWD+SRGM.json",   28, 2, 4, 1, 35),
    ("GYRO-NS.json",     5, 1, 5, 7, 18),
    ("GYRO-NS-CT.json",  5, 1, 7, 6, 19),
    ("GYRO-MWD.json",    5, 1, 5, 7, 18),
]


def _classify(r):
    if not r.interp_available:
        return "INTERP_FAILED"
    if not r.legacy_available:
        return "NO_LEGACY"
    if r.max_abs_diff < 1e-10:
        return "MATCH"
    if r.max_abs_diff < 1e-6:
        return "NEAR_MATCH"
    return "DIFFER"


@pytest.fixture(scope="module")
def survey():
    return standard_test_survey()


@pytest.mark.parametrize(
    "json_name,exp_match,exp_differ,exp_interp_fail,exp_no_legacy,exp_total",
    EXPECTED,
    ids=[e[0].replace(".json", "") for e in EXPECTED],
)
def test_conformance_matrix(survey, json_name, exp_match, exp_differ,
                            exp_interp_fail, exp_no_legacy, exp_total):
    path = JSON_ROOT / json_name
    if not path.exists():
        pytest.skip(f"JSON tool model not shipped: {json_name}")

    results = compare_model(str(path), survey)
    counts: dict[str, int] = {}
    for r in results:
        c = _classify(r)
        counts[c] = counts.get(c, 0) + 1

    assert len(results) == exp_total, f"expected {exp_total} terms, got {len(results)}"
    assert counts.get("MATCH", 0) == exp_match, (
        f"{json_name}: expected {exp_match} MATCH, got {counts}"
    )
    assert counts.get("DIFFER", 0) == exp_differ, (
        f"{json_name}: expected {exp_differ} DIFFER, got {counts}"
    )
    assert counts.get("INTERP_FAILED", 0) == exp_interp_fail, (
        f"{json_name}: expected {exp_interp_fail} INTERP_FAILED, got {counts}"
    )
    assert counts.get("NO_LEGACY", 0) == exp_no_legacy, (
        f"{json_name}: expected {exp_no_legacy} NO_LEGACY, got {counts}"
    )


def test_known_schema_gaps_present(survey):
    """The catalogued schema gaps should still be detected.

    If any of these terms suddenly evaluates cleanly, either the schema
    has gained a new variable (good — update this test) or the interpreter
    is silently substituting something wrong (bad — investigate).
    """
    path = JSON_ROOT / "GYRO-NS-CT.json"
    if not path.exists():
        pytest.skip("GYRO-NS-CT.json not shipped")

    results = compare_model(str(path), survey)
    by_name = {r.name: r for r in results}

    cross_station = {"XYM3E", "XYM4E", "XCLA", "XCLH"}
    per_tool_calibration = {"GXY-RN", "GXY-GD", "GXY-GRW"}

    for term in cross_station | per_tool_calibration:
        if term not in by_name:
            continue
        r = by_name[term]
        assert not r.interp_available, (
            f"{term} unexpectedly evaluated — schema gap closed? "
            f"Update the conformance matrix expectation."
        )


def test_mwd_canonical_terms_match_to_machine_precision(survey):
    """The simple MWD terms (no cross-station deps, no calibration consts)
    must agree to machine precision between legacy and interpreter.

    Acts as the canary: if a MATCH term ever drifts above 1e-12, either
    the YAML magnitude diverged from the JSON value, the unit-conversion
    table got out of sync, or the interpreter lost precision on a formula.
    """
    path = JSON_ROOT / "MWD+SRGM.json"
    if not path.exists():
        pytest.skip("MWD+SRGM.json not shipped")

    results = compare_model(str(path), survey)
    matched = [r for r in results if _classify(r) == "MATCH"]
    assert len(matched) >= 25, (
        f"expected >=25 MATCH terms on MWD+SRGM, got {len(matched)}"
    )
    for r in matched:
        assert r.max_abs_diff <= 1e-12, (
            f"{r.name} regressed: max |Δ| = {r.max_abs_diff:.2e}"
        )
