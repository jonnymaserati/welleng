"""Continuous-gyro recurrence terms (GXY-GD drift, GXY-GRW random walk).

The ISCWSA Rev5 continuous-gyro weight functions are *recurrence* terms: the
azimuth contribution at each station accumulates from the previous station
(``XY_Gyro_Drift`` = running drift integral, ``XY_Gyro_Random_Walk`` = running
random walk), gated by the ``XY Static Gyro End Inc`` inclination -- below the
gate the continuous survey is not running and the term contributes zero.

These cannot be expressed by a single vectorised formula evaluation, so the
JSON+interpreter path (``ToolError._call_interpreter``) detects the recurrence
state variable and evaluates the term station-by-station. This test pins that
the contribution is (a) exactly zero below the gate, (b) non-zero and monotone
above it, and (c) equal to the closed-form running integral the deleted
SPE 90408 Table 7 hand-coded functions produced.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from welleng.survey import Survey, SurveyHeader


def _build_survey():
    # Simple build well crossing the static-gyro gate (17 deg): vertical to
    # ~60 deg over 2000 m, constant azimuth.
    md = np.arange(0.0, 2001.0, 100.0)
    inc = np.clip((md / 2000.0) * 60.0, 0.0, 60.0)
    azi = np.full_like(md, 30.0)
    sh = SurveyHeader(latitude=60.0, azi_reference="true")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Survey(md=md, inc=inc, azi=azi, header=sh, error_model="GYRO-NS-CT")


def _closed_form(survey, speed, gate, power, mode, mag, scale):
    md = np.asarray(survey.md)
    inc = np.asarray(survey.inc_rad)
    n = len(md)
    acc = 0.0
    e = np.zeros(n)
    # Accumulate the SPE 90408-PA Table 7 coefficient h_i, then apply the
    # magnitude LINEARLY (Eqs 5c / 6c: dA_i = v * h_i). Drift: h_i = acc
    # (power=1). Random walk: h_i = sqrt(acc) (power=2) -- the sqrt is the
    # coefficient's, NOT applied to the magnitude.
    for k in range(1, n):
        if inc[k] > gate:
            s = np.sin((inc[k] + inc[k - 1]) / 2.0) ** power
            acc += (md[k] - md[k - 1]) / (speed * s)
            h_i = acc if mode == "drift" else np.sqrt(acc)
            e[k] = h_i * (mag * scale)
        else:
            acc = 0.0
    return e


@pytest.mark.parametrize(
    "code,power,mode",
    [("GXY-GD", 1, "drift"), ("GXY-GRW", 2, "random_walk")],
)
def test_continuous_gyro_running_integral(code, power, mode):
    survey = _build_survey()
    err = survey.err
    hdr = err.errors.em["header"]
    speed = float(hdr["GXYRunningSpeed"])
    gate = float(hdr["XY Static Gyro End Inc"])

    term = err.errors.errors[code]
    e_dia = np.asarray(term.e_DIA)[:, 2]
    inc = np.asarray(survey.inc_rad)

    # (a) exactly zero below the gate
    below = inc <= gate
    assert np.all(e_dia[below] == 0.0), f"{code}: non-zero contribution below gate"

    # (b) above the gate it switches on and is monotone non-decreasing
    above = np.where(inc > gate)[0]
    assert above.size > 0, "test well must cross the static-gyro gate"
    assert e_dia[above[-1]] > 0.0, f"{code}: zero contribution above gate"
    assert np.all(np.diff(e_dia[above]) >= -1e-12), f"{code}: not monotone above gate"

    # (c) equals the SPE 90408-PA Table 7 coefficient times the magnitude
    #     (Eqs 5c / 6c: dA_i = v * h_i), to machine precision. `scale` is
    #     hard-coded to deg->rad here, independent of _MAG_UNIT_TO_BASE, so
    #     this also guards the "deg/sqr(hr)" unit-conversion lookup.
    mag = float(err.errors.em["codes"][code]["magnitude"])
    scale = np.pi / 180.0  # both GXY-GD (deg/hr) and GXY-GRW (deg/sqr(hr)) -> rad
    ref = _closed_form(survey, speed, gate, power, mode, mag, scale)
    assert np.allclose(e_dia, ref, atol=1e-12, rtol=0.0), (
        f"{code}: max|e_DIA - closed_form| = {np.max(np.abs(e_dia - ref)):.3e}"
    )

    # and it produces a non-trivial NEV covariance (was identically zero
    # before the recurrence path existed)
    assert np.max(np.abs(np.asarray(term.cov_NEV))) > 0.0

    # (d) regression guard: binding MDPrev activates cross-station terms
    #     (XYM3E/XYM4E: Max(1, sqrt(10/(MD-MDPrev)))) whose station-0
    #     MD-MDPrev==0 would otherwise yield inf/nan. The whole-survey
    #     covariance must stay finite.
    assert np.all(np.isfinite(np.asarray(err.errors.cov_NEVs))), (
        "non-finite survey covariance (station-0 cross-station blow-up)"
    )
