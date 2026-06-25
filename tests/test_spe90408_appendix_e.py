"""Absolute validation of the gyro error model against SPE 90408 Appendix E.

SPE 90408-MS (Torkildsen et al. 2004) Appendix E publishes position
covariance matrices for six synthetic "Example Models" on the ISCWSA standard
well bores, with a stated acceptance criterion: a correct implementation
agrees to within +/-1% of every tabulated value (or +/-2 units where the
value is < 200), "verified within these limits by independent
implementations".

This pins **Example Model #1** (XY accelerometer + XY stationary gyro, 0-150
deg) on **ISCWSA Well #1** to Table E1. Unlike the conformance harness (which
compares the JSON interpreter against the legacy welleng weight functions and
so cancels any common scale error), this checks welleng's gyro output against
absolute, externally-published numbers -- exercising the vertical-singularity
substitution (the well is vertical 0-1200 m) and the depth-term propagation.

The model config + magnitudes are SPE 90408-MS Appendix D (Table D1/D3 + D7);
the weight functions are Tables 2 (XY accel), 4 (XY stationary gyro), 9
(misalignment Alt.3), 10 (depth). Fixture:
``tests/test_data/spe90408_example_models/example_1.json``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import welleng as we
from welleng.survey import Survey, SurveyHeader
from welleng.errors.tool_errors import _json_to_em_adapter

DATA = Path(__file__).parent / "test_data"

# SPE 90408-MS Appendix E, Table E1, Example Model #1, well bore ISCWSA #1.
# depth(m) -> [NN, NE, NV, EE, EV, VV] in m^2 (covariance unit = depth unit^2).
REF_E1_MODEL1 = {
    1200: [19, 0, 0, 18, 0, 2],
    2100: [1439, -373, -2, 144, -6, 9],
    5100: [134488, -35992, -18, 9807, -49, 137],
}

# Paper's acceptance: within +/-1%, or +/-2 units where |value| < 200.
REL_TOL = 0.01
ABS_TOL_SMALL = 2.0
SMALL = 200


def _build_well1() -> Survey:
    """ISCWSA Standard Test Well #1 from the raw ISCWSA test data.

    inc/azi are stored in *degrees* in the file (max 90 / 75); the header
    angles are radians. (The conformance helper ``standard_test_survey()``
    double-converts inc via ``np.degrees`` -> a 90-radian well; do not use it
    for an absolute check.)
    """
    with open(DATA / "error_mwdrev5_1_iscwsa_data.json") as f:
        d = json.load(f)
    sv, h = d["survey"], d["header"]
    sh = SurveyHeader(
        name="iscwsa-1", latitude=h["latitude"], b_total=h["b_total"],
        dip=np.degrees(h["dip"]), declination=np.degrees(h["declination"]),
        convergence=np.degrees(h.get("convergence", 0.0)),
        G=h["G"], azi_reference=h["azi_reference"],
        earth_rate=h.get("earth_rate"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Survey(
            md=np.array(sv["md"]), inc=np.array(sv["inc"]),
            azi=np.array(sv["azi"]), header=sh,
        )


def _example_model_cov(survey: Survey, fixture: str) -> np.ndarray:
    """Run a synthetic Example-Model JSON tool through the interpreter on the
    given survey and return the summed NEV covariance, shape (n, 3, 3).

    The example models are test fixtures, not registered tools, so we borrow
    a wired ToolError (its survey + propagation machinery) from a real model
    and swap in the example's adapted ``em``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        em = we.error.ErrorModel(survey, error_model="GYRO-NS-CT")
    te = em.errors
    with open(DATA / "spe90408_example_models" / fixture) as f:
        adapted = _json_to_em_adapter(json.load(f))
    te.em = adapted
    te.errors = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for code, entry in adapted["codes"].items():
            te.errors[code] = te._call_interpreter(
                code, entry["_iscwsa_term"], entry["magnitude"],
                entry["propagation"],
            )
    n = len(survey.md)
    cov = np.zeros((n, 3, 3))
    for v in te.errors.values():
        assert v is not None and getattr(v, "cov_NEV", None) is not None, (
            "an Example-Model term produced no covariance (propagation mode "
            "unhandled by _generate_error?)"
        )
        cov += np.asarray(v.cov_NEV)
    assert np.all(np.isfinite(cov)), "non-finite covariance (singularity/inf)"
    return cov


@pytest.mark.parametrize("depth", list(REF_E1_MODEL1))
def test_example_model_1_appendix_e(depth):
    survey = _build_well1()
    cov = _example_model_cov(survey, "example_1.json")

    md = np.asarray(survey.md)
    i = int(np.argmin(np.abs(md - depth)))
    assert abs(md[i] - depth) < 1e-6, (
        f"no station at checkpoint {depth} m (nearest {md[i]} m)"
    )

    c = cov[i]
    got = {
        "NN": c[0, 0], "NE": c[0, 1], "NV": c[0, 2],
        "EE": c[1, 1], "EV": c[1, 2], "VV": c[2, 2],
    }
    ref = dict(zip(("NN", "NE", "NV", "EE", "EV", "VV"), REF_E1_MODEL1[depth]))

    failures = []
    for name, g in got.items():
        r = ref[name]
        if abs(r) < SMALL:
            ok = abs(g - r) <= ABS_TOL_SMALL
        else:
            ok = abs((g - r) / r) <= REL_TOL
        if not ok:
            failures.append(f"{name}: got {g:.2f}, ref {r} "
                            f"(Δ {g - r:+.2f})")
    assert not failures, (
        f"SPE 90408 Appendix E Model #1 @ {depth} m outside ±1%/±2u:\n  "
        + "\n  ".join(failures)
    )
