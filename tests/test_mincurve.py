"""Heavy coverage for welleng.utils.MinCurve.

MinCurve is the foundational, local-coordinate, units- and azimuth-reference-
agnostic minimum-curvature geometry kernel (Survey builds on it, and the api /
wellpath designer use it directly). These tests pin its geometry, its agnostic
contract, the curvature-radius <-> DLS relationship, edge cases and invariants,
plus parity with Survey.
"""
import numpy as np
import pytest

from welleng.utils import MinCurve, radius_from_dls, dls_from_radius
from welleng.survey import Survey


# --- fixtures ----------------------------------------------------------------
def _vertical():
    return MinCurve(np.array([0.0, 100.0, 200.0]),
                    np.radians([0.0, 0.0, 0.0]),
                    np.radians([0.0, 0.0, 0.0]))


def _constant_build(step_md=100.0, d_inc_deg=10.0, n=10):
    """Planar (azi=0) constant-build well -> constant curvature radius."""
    md = np.arange(n + 1) * step_md
    inc = np.radians(np.arange(n + 1) * d_inc_deg)
    azi = np.zeros(n + 1)
    return MinCurve(md, inc, azi), step_md, np.radians(d_inc_deg)


def _mixed():
    return MinCurve(np.array([0.0, 100.0, 250.0, 500.0, 800.0]),
                    np.radians([0.0, 15.0, 45.0, 90.0, 90.0]),
                    np.radians([0.0, 30.0, 60.0, 90.0, 120.0]))


# --- geometry correctness ----------------------------------------------------
def test_vertical_well_geometry():
    mc = _vertical()
    assert np.allclose(mc.dogleg, 0.0)                       # no turning
    assert np.allclose(mc.poss[:, 2], [0.0, 100.0, 200.0])   # straight down (z)
    assert np.allclose(mc.poss[:, :2], 0.0)                  # no N/E drift
    assert np.all(np.isinf(mc.curve_radius))                 # straight -> inf radius


def test_constant_build_has_constant_radius():
    mc, step_md, d_inc = _constant_build()
    expected_R = step_md / d_inc                             # arc length / angle
    assert np.allclose(mc.curve_radius[1:], expected_R)
    # planar azi=0 => dogleg per step == delta inclination
    assert np.allclose(mc.dogleg[1:], d_inc)


def test_positions_are_cumulative_and_local():
    mc = _mixed()
    # local: first station at the origin (no start offset baked in)
    assert np.allclose(mc.poss[0], 0.0)
    # cumulative: poss == cumsum of per-station deltas
    deltas = np.array([mc.delta_x, mc.delta_y, mc.delta_z]).T
    assert np.allclose(mc.poss, np.cumsum(deltas, axis=0))


# --- agnostic contract -------------------------------------------------------
def test_no_unit_or_datum_state():
    """MinCurve holds no unit / start / datum state (that is Survey's)."""
    mc = _mixed()
    for attr in ("unit", "start_xyz", "start_nev", "header"):
        assert not hasattr(mc, attr), f"MinCurve should not hold {attr!r}"


def test_units_agnostic_md_values_pass_through():
    """Same numeric md/inc/azi -> same geometry regardless of what unit the
    caller *considers* md to be in (the kernel never sees a unit)."""
    md = np.array([0.0, 100.0, 250.0])
    a = MinCurve(md, np.radians([0.0, 30.0, 60.0]), np.radians([0.0, 10.0, 20.0]))
    b = MinCurve(md, np.radians([0.0, 30.0, 60.0]), np.radians([0.0, 10.0, 20.0]))
    assert np.allclose(a.poss, b.poss) and np.allclose(a.curve_radius, b.curve_radius)


# --- curvature radius <-> DLS ------------------------------------------------
def test_curve_radius_is_arclength_over_dogleg():
    mc = _mixed()
    assert np.allclose(mc.curve_radius[1:], mc.delta_md[1:] / mc.dogleg[1:])


def test_dls_is_the_callers_convention_from_radius():
    """MinCurve exposes only the convention-free curve_radius; DLS is the
    caller's units-aware derivation (deg per coeff md-units), scaling linearly
    with the coefficient."""
    mc = _mixed()
    dls30 = np.degrees(30.0 / mc.curve_radius)
    dls100 = np.degrees(100.0 / mc.curve_radius)
    assert np.allclose(dls100, dls30 * (100.0 / 30.0))


def test_radius_reproduces_legacy_dls():
    """deg(coeff / curve_radius) equals the legacy degrees(dogleg)/dmd*coeff."""
    mc = _mixed()
    legacy = np.zeros_like(mc.dogleg)
    legacy[1:] = np.degrees(mc.dogleg[1:]) / mc.delta_md[1:] * 30.0
    assert np.allclose(np.degrees(30.0 / mc.curve_radius), legacy)


def test_radius_dls_roundtrip():
    mc, _step_md, _d_inc = _constant_build()
    R = mc.curve_radius[1]
    # utils converters round-trip on the curvature radius
    assert radius_from_dls(dls_from_radius(R)) == pytest.approx(R, rel=1e-6)


# --- edge cases / invariants -------------------------------------------------
def test_straight_section_inf_radius():
    mc = MinCurve(np.array([0.0, 500.0]), np.radians([30.0, 30.0]), np.radians([45.0, 45.0]))
    assert np.isinf(mc.curve_radius[1])                      # straight -> inf radius
    assert np.degrees(30.0 / mc.curve_radius[1]) == 0.0      # -> 0 DLS for the caller


def test_requires_at_least_two_stations():
    with pytest.raises(AssertionError):
        MinCurve(np.array([0.0]), np.array([0.0]), np.array([0.0]))


def test_rf_at_least_one():
    mc = _mixed()
    assert np.all(mc.rf >= 1.0 - 1e-12)


# --- parity with Survey (the oracle relationship) ----------------------------
def test_survey_dls_matches_mincurve():
    """Survey derives its DLS from MinCurve; they must agree (metric)."""
    md = [0.0, 100.0, 250.0, 500.0]
    inc = [0.0, 15.0, 45.0, 90.0]
    azi = [0.0, 30.0, 60.0, 90.0]
    s = Survey(md=md, inc=inc, azi=azi)
    mc = MinCurve(np.array(md), np.radians(inc), s.azi_grid_rad)
    assert np.allclose(s.dls, np.degrees(30.0 / mc.curve_radius))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --- interpolate (exact min-curve MD->TVD, no Survey) ------------------------
def _curved():
    # a build-and-turn survey with real doglegs
    md = np.array([0.0, 300.0, 600.0, 1000.0, 1600.0, 2200.0])
    inc = np.radians([0.0, 0.0, 30.0, 60.0, 90.0, 90.0])
    azi = np.radians([0.0, 0.0, 20.0, 45.0, 90.0, 135.0])
    return md, inc, azi


def test_interpolate_matches_survey_interpolate_md():
    md, inc, azi = _curved()
    mc = MinCurve(md, inc, azi)
    sv = Survey(md=md, inc=inc, azi=azi, deg=False)  # Survey start at origin
    # off-station query mds spanning several intervals
    qs = np.array([150.0, 455.0, 780.5, 1333.3, 1900.0])
    got = mc.interpolate(qs)  # LOCAL [East, North, TVD]; pos_nev is [N, E, V]
    for q, g in zip(qs, got):
        node = sv.interpolate_md(float(q))
        np.testing.assert_allclose(g[0], node.pos_nev[1], atol=1e-8)  # East
        np.testing.assert_allclose(g[1], node.pos_nev[0], atol=1e-8)  # North
        np.testing.assert_allclose(g[2], node.pos_nev[2], atol=1e-8)  # TVD


def test_interpolate_is_the_arc_not_a_chord():
    md, inc, azi = _curved()
    mc = MinCurve(md, inc, azi)
    # station tvds for a linear reference
    tvd_st = mc.poss[:, 2]
    q = 780.5  # mid a dogleg interval
    arc = mc.interpolate(q)[2]
    chord = np.interp(q, md, tvd_st)
    assert abs(arc - chord) > 1e-3  # arc must differ from the linear chord


def test_interpolate_scalar_array_and_out_of_range():
    md, inc, azi = _curved()
    mc = MinCurve(md, inc, azi)
    assert mc.interpolate(500.0).shape == (3,)          # scalar -> (3,)
    assert mc.interpolate([100.0, 500.0]).shape == (2, 3)
    assert np.isnan(mc.interpolate(md[-1] + 500.0)).all()  # beyond end -> nan
    # exact at a station == that station's cumulative position
    np.testing.assert_allclose(mc.interpolate(md[2]), mc.poss[2], atol=1e-9)


# --- tvd_turning_points + interpolate_tvd (moved from Survey) + angles --------
def test_tvd_turning_points_matches_survey():
    # builds through horizontal (85 -> 95) -> a turning point INSIDE interval 1
    md = np.array([0., 1000., 1100., 1300.])
    inc = np.radians([0, 85, 95, 95.])
    azi = np.zeros(4)
    mc = MinCurve(md, inc, azi)
    tp = mc.tvd_turning_points()
    assert len(tp) == 1 and 1000.0 < tp[0] < 1100.0
    # inc is 90 deg (horizontal tangent) at the turning point
    _, inc_star, _ = mc.interpolate(float(tp[0]), angles=True)
    np.testing.assert_allclose(np.degrees(inc_star), 90.0, atol=1e-6)
    # parity: Survey delegates to MinCurve for the same MDs
    sv = Survey(md=md, inc=inc, azi=azi, deg=False)
    np.testing.assert_allclose(sv.tvd_turning_points(), tp)


def test_tvd_turning_points_empty_when_monotonic():
    mc = MinCurve(np.array([0., 500., 1000.]),
                  np.radians([0, 30, 60.]), np.zeros(3))
    assert len(mc.tvd_turning_points()) == 0


def test_interpolate_tvd_inverse_multi_crossing():
    # rise -> dip -> rise: a target in the dip band is reached multiple times
    md = np.array([0., 800., 1000., 1200., 1600.])
    mc = MinCurve(md, np.radians([0, 70, 110, 70, 40.]), np.zeros(5))
    fine = np.linspace(0.0, 1600.0, 800001)
    tvdf = mc.interpolate(fine)[:, 2]
    target = None
    for frac in np.linspace(0.05, 0.98, 60):
        t = tvdf.min() + frac * (tvdf.max() - tvdf.min())
        f = tvdf - t
        if (np.sign(f[:-1]) != np.sign(f[1:])).sum() >= 2:
            target = t
            break
    assert target is not None
    got = mc.interpolate_tvd(target)
    assert len(got) >= 2
    for g in got:                                  # each solved md hits the tvd
        np.testing.assert_allclose(mc.interpolate(g)[2], target, atol=1e-4)


def test_interpolate_tvd_monotonic_inverts_interpolate():
    mc = MinCurve(np.array([0., 500., 1000., 1500.]),
                  np.radians([0, 20, 40, 60.]), np.zeros(4))
    tvd_q = mc.interpolate(1234.0)[2]
    back = mc.interpolate_tvd(tvd_q)
    assert back.size == 1
    np.testing.assert_allclose(back[0], 1234.0, atol=1e-5)


def test_interpolate_angles_return():
    md, inc, azi = _curved()
    mc = MinCurve(md, inc, azi)
    pos, i, a = mc.interpolate(455.0, angles=True)
    assert pos.shape == (3,) and np.isfinite(i) and np.isfinite(a)
    # array form
    p, ii, aa = mc.interpolate([150.0, 780.5], angles=True)
    assert p.shape == (2, 3) and ii.shape == (2,) and aa.shape == (2,)


# --- get_dogleg antiparallel/U-turn domain clamp (welleng #307) ---------------
def test_get_dogleg_horizontal_uturn_is_pi_not_nan():
    from welleng.utils import get_dogleg
    # inc1==inc2==90, azi differing by 180 -> antiparallel tangents; the
    # haversine arg is exactly 1.0 and FP rounding can push it past 1 -> NaN
    # without the clamp. Must return pi.
    dl = get_dogleg(np.radians(90.0), np.radians(0.0),
                    np.radians(90.0), np.radians(180.0))
    assert np.isfinite(dl)
    np.testing.assert_allclose(dl, np.pi, atol=1e-9)


def test_get_dogleg_unchanged_on_in_domain_values():
    from welleng.utils import get_dogleg
    assert get_dogleg(0.0, 0.0, 0.0, 0.0) == 0.0
    np.testing.assert_allclose(
        np.degrees(get_dogleg(np.radians(30.0), 0.0, np.radians(60.0), 0.0)),
        30.0, atol=1e-9)
