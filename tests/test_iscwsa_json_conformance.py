"""The hand-coded weight functions agree with the OWSG tool-model formulas.

The harness in ``welleng.errors.conformance`` evaluates every term of an OWSG JSON model
two ways, through the hand-coded weight function the YAML tool models call and through
the JSON formula the interpreter evaluates, on the ISCWSA Standard Test Well 1, and
reports the largest difference in the error vector away from the vertical singularity.

Every term of every shipped model is checked. A term may only be left unpaired or
differing for a reason stated in ``NOT_PAIRED`` or ``FORMULA_IS_NOT_THE_PRODUCTION_PATH``.
Each weight is checked against the ISCWSA Definition itself in
``tests/test_iscwsa_weights_definition.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from welleng.errors.conformance import compare_model, standard_test_survey

JSON_ROOT = Path(__file__).parent.parent / "welleng" / "errors" / "iscwsa_json"
MODELS = sorted(JSON_ROOT.glob("owsg_*/*.json"))

# Terms evaluated only from the JSON formula: welleng has no hand-coded twin.
NOT_PAIRED = {
    "AXYZ-MIS", "AXYZ-SF", "AXYZ-ZB", "EXT-REF",
    "GXY-B1R", "GXY-B1S", "GXY-B2R", "GXY-B2S", "GXY-G1", "GXY-G4",
}
# Station-to-station recurrences: the formula cannot be evaluated one station at a time.
RECURRENCE = {"GXY-GD", "GXY-GRW", "GXY-RN", "GZ-GD"}
# XCLA/XCLH: production computes these with the hand-coded functions for every model (the
# cross-station max(.) is not a pointwise weight), so the JSON formula is not what runs.
FORMULA_IS_NOT_THE_PRODUCTION_PATH = {"XCLA", "XCLH"}

TOL = 1e-10


@pytest.fixture(scope="module")
def survey():
    return standard_test_survey()


def test_every_model_is_checked():
    assert len(MODELS) == 100


@pytest.mark.parametrize("path", MODELS, ids=[p.name for p in MODELS])
def test_hand_coded_and_formula_agree(survey, path):
    problems = []
    for r in compare_model(str(path), survey):
        if r.name in RECURRENCE:
            if r.interp_available:
                problems.append(f"{r.name} now evaluates pointwise: move it out of RECURRENCE")
            continue
        if not r.interp_available:
            problems.append(f"{r.name}: formula does not evaluate ({r.note})")
            continue
        if not r.legacy_available:
            if r.name not in NOT_PAIRED:
                problems.append(f"{r.name}: no hand-coded function reached ({r.note})")
            continue
        if r.name in NOT_PAIRED:
            problems.append(f"{r.name} now has a hand-coded twin: move it out of NOT_PAIRED")
        if r.name in FORMULA_IS_NOT_THE_PRODUCTION_PATH:
            continue
        if not r.max_abs_diff <= TOL:
            problems.append(
                f"{r.name} ({r.legacy_function}): max |diff| {r.max_abs_diff:.3e}, "
                f"relative {r.max_rel_diff:.3e}"
            )
    assert not problems, f"{path.name}:\n  " + "\n  ".join(problems)
