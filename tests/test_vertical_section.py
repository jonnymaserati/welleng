"""Vertical section: the stations' horizontal displacement projected onto
the section azimuth -- exact at the stations, correct through north, and
without modifying the survey."""
import math
import warnings

import numpy as np
import pytest

import welleng as we

warnings.filterwarnings("ignore")


def _survey(azi, inc=60.0, ref="grid", **hdr):
    md = np.arange(0.0, 30.0 * len(azi), 30.0)
    h = we.survey.SurveyHeader(azi_reference=ref, b_total=50000.0, dip=60.0,
                               declination=hdr.pop("declination", 0.0),
                               convergence=hdr.pop("convergence", 0.0), **hdr)
    return we.survey.Survey(md=md, inc=np.full(len(md), inc), azi=azi,
                            header=h)


def test_through_north_is_the_northing():
    """Azimuth 350 -> 10 deg at 60 deg inclination: the section along north
    is the northing. Averaging 350 and 10 gives 180 and runs it backwards."""
    s = _survey([350.0, 355.0, 0.0, 5.0, 10.0])
    vs = s.get_vertical_section(0.0)
    assert np.allclose(vs, s.n - s.n[0], atol=1e-9)
    assert np.all(np.diff(vs) > 0)


def test_does_not_modify_the_survey():
    s = _survey([0.0, 10.0, 20.0], inc=0.0)       # vertical stations
    before = {k: np.array(getattr(s, k), copy=True)
              for k in ("azi_grid_rad", "azi_true_rad", "azi_mag_rad")}
    s.get_vertical_section(90.0)
    for k, v in before.items():
        assert np.array_equal(getattr(s, k), v), k


@pytest.mark.parametrize("ref", ["grid", "true", "magnetic"])
def test_along_the_well_is_the_horizontal_displacement(ref):
    """A constant-azimuth well viewed along its own azimuth (in the header's
    reference) shows its full horizontal displacement."""
    s = _survey([30.0] * 12, ref=ref, convergence=1.7, declination=2.3)
    vs = s.get_vertical_section(30.0)
    disp = np.hypot(s.n - s.n[0], s.e - s.e[0])
    assert np.allclose(vs, disp, atol=1e-9)


def test_origin():
    s = _survey([45.0] * 6)
    vs = s.get_vertical_section(0.0, origin=(s.n[0] - 100.0, s.e[0]))
    assert np.allclose(vs, s.n - s.n[0] + 100.0, atol=1e-9)


def test_set_vertical_section_keeps_the_header_in_radians():
    s = _survey([350.0, 0.0, 10.0])
    s.set_vertical_section(45.0, deg=True)
    assert math.isclose(s.header.vertical_section_azimuth, math.radians(45.0))
    assert np.allclose(s.vertical_section, s.get_vertical_section(45.0))
