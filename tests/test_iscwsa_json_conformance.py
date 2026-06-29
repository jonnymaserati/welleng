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
# Re-baselined 2026-06-25 after fixing standard_test_survey() (it built a
# 90-radian well by double-converting inc; see conformance.py). On the
# corrected well one singular term per model moves DIFFER -> MATCH: the
# vertical section now exercises the interpreter's SING substitution, which
# agrees with the legacy hand-coded singular branch (was masked by the broken
# well, where no station had inc < 0.0001 deg). Totals unchanged.
# Re-baselined 2026-06-29: binding the previous-station arrays (MDPrev/IncPrev/
# AzPrev) in conformance._bindings_from_survey lets the cross-station terms (XCLA/
# XCLH, XYM3E/XYM4E) evaluate in the interpreter -- they move INTERP_FAILED ->
# NO_LEGACY (they evaluate but welleng's legacy hand-coded dispatcher never
# implemented them, so there is nothing to diff against). The only remaining
# INTERP_FAILED are the recurrence/noise gyro terms (GXY-RN/GD/GRW).
EXPECTED = [
    ("MWD+SRGM.json",   29, 1, 0,  5, 35),
    ("GYRO-NS.json",     6, 0, 1, 11, 18),
    ("GYRO-NS-CT.json",  6, 0, 3, 10, 19),
    ("GYRO-MWD.json",    6, 0, 1, 11, 18),
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
    """Lock which terms evaluate vs remain a gap.

    Cross-station terms (XCLA/XCLH, XYM3E/XYM4E) now evaluate -- the prev-station
    arrays (MDPrev/IncPrev/AzPrev) are bound (2026-06-29), matching production. The
    remaining interpreter gaps are the recurrence/noise gyro terms (GXY-RN noise,
    GXY-GD drift, GXY-GRW random walk), which the single-vectorised conformance eval
    cannot express (they need the station-by-station recurrence path). If one of
    those suddenly evaluates, the harness gained the recurrence path -- update this.
    """
    path = JSON_ROOT / "GYRO-NS-CT.json"
    if not path.exists():
        pytest.skip("GYRO-NS-CT.json not shipped")

    results = compare_model(str(path), survey)
    by_name = {r.name: r for r in results}

    # Now evaluate (were INTERP_FAILED before the prev-station bindings):
    for term in ("XCLA", "XCLH", "XYM3E", "XYM4E"):
        if term in by_name:
            assert by_name[term].interp_available, (
                f"{term} no longer evaluates — prev-station bindings regressed?"
            )

    # Still a gap (recurrence/noise terms the vectorised harness can't express):
    for term in ("GXY-RN", "GXY-GD", "GXY-GRW"):
        if term in by_name:
            assert not by_name[term].interp_available, (
                f"{term} unexpectedly evaluated — harness gained the recurrence "
                f"path? Update the conformance matrix expectation."
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
