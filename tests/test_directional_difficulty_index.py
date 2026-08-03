"""DDI (IADC/SPE 59196, Oag & Williams). No numeric worked example exists in the
paper, so these validate self-consistency, the definition, and unit-invariance."""
import numpy as np
import pytest

import welleng as we


def _build_survey(depth_unit="meters", scale=1.0):
    # a build-and-hold: vertical, build to 60 deg, hold — gives departure + dogleg
    md = np.array([0, 500, 1500, 3000], dtype=float) * scale
    inc = np.array([0, 0, 60, 60], dtype=float)
    azi = np.array([45, 45, 45, 45], dtype=float)
    return we.survey.Survey(md=md, inc=inc, azi=azi, depth_unit=depth_unit)


def test_ddi_matches_manual_formula():
    s = _build_survey()
    ddi = s.directional_difficulty_index()          # TD float
    # hand-recompute: log10(MD_ft * AHD_ft * cum|dogleg_deg| / TVD_ft) at TD
    to_ft = 1.0 / 0.3048
    ahd = np.hypot(s.n, s.e)
    tort = np.cumsum(np.abs(np.degrees(s.dogleg)))
    expect = np.log10(s.md[-1] * ahd[-1] * tort[-1] / s.tvd[-1] * to_ft)
    assert ddi == pytest.approx(expect, rel=1e-9)


def test_ddi_data_flag():
    s = _build_survey()
    arr = s.directional_difficulty_index(data=True)
    assert isinstance(arr, np.ndarray) and arr.shape == s.md.shape
    assert s.directional_difficulty_index() == pytest.approx(arr[-1])


def test_ddi_unit_invariant_m_vs_ft():
    # DDI is defined in feet, so the SAME well in metres vs feet must agree
    ddi_m = _build_survey("meters", 1.0).directional_difficulty_index(depth_units="m")
    ddi_ft = _build_survey(
        "feet", 1.0 / 0.3048
    ).directional_difficulty_index(depth_units="ft")
    assert ddi_m == pytest.approx(ddi_ft, rel=1e-9)


def _jwell_ft(hold_inc, hold_len_ft):
    # build-and-hold in feet, a la SPE-59196 Fig 4 designer J-wells
    md = np.array([0.0, 2000.0, 5000.0, 5000.0 + hold_len_ft])
    inc = np.array([0.0, 0.0, hold_inc, hold_inc])
    azi = np.full(4, 30.0)
    return we.survey.Survey(md=md, inc=inc, azi=azi, depth_unit="feet")


def test_ddi_reproduces_spe59196_range_and_trend():
    # Figs 7-9: DDI increases monotonically with departure/inclination for the
    # J-well family, spanning the published ~5-7.6 range.
    fam = [(30, 4000), (45, 9000), (60, 14000), (75, 22000), (85, 34000)]
    ddis = [
        _jwell_ft(i, hl).directional_difficulty_index(depth_units="ft")
        for i, hl in fam
    ]
    assert all(np.isfinite(d) for d in ddis)
    assert np.all(np.diff(ddis) > 0)          # monotonic increase (Figs 7-9)
    assert 4.5 < ddis[0] and ddis[-1] < 8.0    # inside the paper's range
    # a moderate ERD well lands in the Fig 13/14 field band (~6.0-6.8)
    mid = _jwell_ft(60, 14000).directional_difficulty_index(depth_units="ft")
    assert 6.0 <= mid <= 6.8
