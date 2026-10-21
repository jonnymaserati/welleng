"""XYM3E / XYM4E at a vertical first surveyed station (ISCWSA Rev5.13 Sec 11.5 eq 33).

The station-spacing amplifier M = Max(1, sqrt(10 / (MD - MDPrev))) at station 1
belongs to the interval from the tie-on to station 1. It only exceeds 1 when that
interval is under 10 m, which the ISCWSA test wells never are, so a short first
interval is needed to see it. North (XYM3E) and east (XYM4E) take the same weight.
"""
import numpy as np
import pytest

import welleng as we

MDS = np.array([0.0, 1.0, 30.0, 60.0, 90.0, 120.0])


def _err(model):
    sh = we.survey.SurveyHeader(
        name="short-first-interval", latitude=60.0, b_total=50000.0, dip=70.0,
        declination=0.0, convergence=0.0, azi_reference="grid",
    )
    zero = np.zeros_like(MDS)
    return we.survey.Survey(
        md=MDS, inc=zero, azi=zero, header=sh, error_model=model,
    ).err.errors


def test_first_station_amplifier_uses_the_first_interval():
    errors = _err("ISCWSA MWD Rev5.11")
    north = errors.errors["XYM3E"].e_NEV[1, 0]
    east = errors.errors["XYM4E"].e_NEV[1, 1]
    mag = np.radians(0.3)
    amp = max(1.0, np.sqrt(10.0 / (MDS[1] - MDS[0])))
    expected = amp * (MDS[2] + MDS[1] - 2 * MDS[0]) / 2 * mag
    assert north == pytest.approx(expected, rel=1e-12)
    assert east == pytest.approx(expected, rel=1e-12)
