"""Tests for welleng.steering — the design <-> adjusted ("error domain")
trajectory map and formation-top prognosis."""
import numpy as np
import pytest

import welleng as we
from welleng.steering import project_to, formation_prognosis


def _design():
    return we.survey.Survey(
        md=[0, 300, 900, 1800, 2400, 3000.], inc=[0, 0, 40, 40, 80, 80.],
        azi=np.zeros(6), header=we.survey.SurveyHeader(name="d"),
    )


# --------------------------------------------------------------------------- #
# project_to — the domain switch
# --------------------------------------------------------------------------- #
def test_offset_zero_at_surface():
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    p = project_to(a, d, 0.0)
    assert np.allclose(p.offset_nev, 0.0, atol=1e-9)


def test_adjusted_is_shallow_of_design_downhole():
    """The survey-interval bias shallows TVD: adjusted sits high-side of the plan
    and above it in TVD, growing with depth."""
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    p_mid = project_to(a, d, 1800.)
    p_td = project_to(a, d, 3000.)
    # adjusted shallower => its tvd < design tvd => offset in tvd is negative
    assert p_td.offset_nev[2] < 0
    assert p_mid.offset_nev[2] < 0
    # magnitude grows with depth
    assert abs(p_td.offset_nev[2]) > abs(p_mid.offset_nev[2])
    # high-side offset is positive (adjusted on the up/high side of the hole)
    assert p_td.offset_highside > 0


def test_project_to_target_station_matches_interpolate():
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    p = project_to(a, d, 2100.)
    nd = d.interpolate_md(2100.)
    assert np.allclose(p.pos_nev, nd.pos_nev)
    assert p.tvd == pytest.approx(nd.pos_nev[2])


def test_project_to_antisymmetric_offset():
    """Swapping source/target negates the NEV offset (same displacement)."""
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    ad = project_to(a, d, 2400.)
    da = project_to(d, a, 2400.)
    assert np.allclose(ad.offset_nev, -da.offset_nev)


def test_project_to_out_of_range_raises():
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    with pytest.raises(ValueError):
        project_to(a, d, 9e9)


def test_vertical_highside_is_nan():
    """High side is undefined in a vertical hole."""
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    p = project_to(a, d, 150.)   # still vertical section
    assert np.isnan(p.offset_highside)


# --------------------------------------------------------------------------- #
# formation_prognosis
# --------------------------------------------------------------------------- #
def test_top_shallower_than_prognosis():
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    res = formation_prognosis(d, a, [(1800., "Top A"), (3000., "Top B")])
    assert [r.name for r in res] == ["Top A", "Top B"]
    # actual TVD is shallower than the same-MD prognosis => negative residual
    assert res[0].tvd_residual < 0
    assert res[1].tvd_residual < res[0].tvd_residual   # grows with depth


def test_top_md_residual_positive_extra_md():
    """To reach the formation's TVD the shallow well must drill MORE md."""
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    res = formation_prognosis(d, a, [(3000., "Top B")])
    r = res[0]
    # design reaches r.tvd_actual at a shallower md than the drilled 3000 m
    assert not np.isnan(r.md_prognosed)
    assert r.md_prognosed < r.md_actual
    assert r.md_residual == pytest.approx(r.md_actual - r.md_prognosed)


def test_top_tvd_residual_matches_project_to():
    """formation_prognosis TVD residual == the project_to tvd offset at same MD."""
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    md = 2400.
    r = formation_prognosis(d, a, [(md, "T")])[0]
    p = project_to(a, d, md)
    assert r.tvd_residual == pytest.approx(p.offset_nev[2])


def test_top_beyond_design_tvd_nan_md_residual():
    d = _design()
    a = d.maximum_curvature(dls_noise=1.0)
    # a fabricated survey deeper than the design's max TVD is hard to force here;
    # instead check the _md_at_tvd None-path via an unreachable TVD directly
    from welleng.steering import _md_at_tvd
    assert _md_at_tvd(d, 9e9) is None
