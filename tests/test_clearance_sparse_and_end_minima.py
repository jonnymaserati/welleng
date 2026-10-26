"""Separation-factor minima the station sampling can hide.

A. Sparse offset: the closest approach lies far along a long offset leg whose
   two stations are both distant, so the leg next to the nearest offset
   station is the wrong one to search.
B. End of the reference: an approach still closing at the last reference
   station has its minimum inside the last leg, which ``argrelmin`` cannot
   flag.

Oracles are independent of the station sampling: the exact geometry of the
approach, and the same wells densified with ``interpolate_mds`` so the
minimum sits at or next to a station.
"""
import numpy as np
import pytest

import welleng as we
from welleng.clearance import IscwsaClearance, MahalanobisClearance


def _header():
    return we.survey.SurveyHeader(deg=True, b_total=50000.0, dip=70.0,
                                  declination=1.0)


@pytest.fixture(scope="module")
def sparse_offset_pair():
    # reference: vertical at N=0, E=50, stations every 30 m
    md = np.arange(0.0, 2001.0, 30.0)  # grid-lint: ok straight well
    ref = we.survey.Survey(md=md, inc=np.zeros_like(md), azi=np.zeros_like(md),
                           header=_header(), start_nev=[0, 50, 0],
                           error_model="ISCWSA MWD Rev4")
    # offset: a 2000 m horizontal tangent due north at E=0, TVD 1000 m, then
    # two 90 deg turns (R=150 m) and 700 m back south. Its nearest station to
    # the reference is the last one, 390 m away; the tangent passes at 50 m.
    q = 150.0 * np.pi / 2
    omd = np.array([0.0, 2000.0, 2000.0 + q, 2000.0 + 2 * q, 2700.0 + 2 * q])
    off = we.survey.Survey(md=omd, inc=np.full(5, 90.0),
                           azi=np.array([0.0, 0.0, 90.0, 180.0, 180.0]),
                           header=_header(), start_nev=[-1000, 0, 1000],
                           error_model="ISCWSA MWD Rev4")
    return ref, off


def test_sparse_offset_finds_the_tangent_approach(sparse_offset_pair):
    """The tangent passes 50 m from the reference at TVD 1000 m (exact
    geometry); the nearest offset station is 390.6 m away."""
    ref, off = sparse_offset_pair
    c = IscwsaClearance(ref, off)
    i = int(np.argmin(np.ravel(c.distance_cc)))
    assert float(np.ravel(c.distance_cc)[i]) == pytest.approx(50.0, abs=1e-3)
    assert c.ref.md[i] == pytest.approx(1000.0, abs=0.5)
    assert c.off.md[i] == pytest.approx(1000.0, abs=0.5)


def test_sparse_offset_mahalanobis_matches_densified_offset(sparse_offset_pair):
    ref, off = sparse_offset_pair
    mds = np.arange(10.0, off.md[-1], 10.0)  # grid-lint: ok densified oracle
    dense = off.interpolate_mds(mds)
    want = MahalanobisClearance(ref, dense).min_sf
    got = MahalanobisClearance(ref, off).min_sf
    assert got == pytest.approx(want, rel=1e-5)


def test_sparse_offset_iscwsa_matches_densified_offset(sparse_offset_pair):
    """The offset covariance between stations comes from the error model, the
    rule interpolate_mds uses, so the separation factor does not depend on
    where the offset's stations happen to be."""
    ref, off = sparse_offset_pair
    mds = np.arange(10.0, off.md[-1], 10.0)  # grid-lint: ok densified oracle
    want = np.ravel(IscwsaClearance(ref, off.interpolate_mds(mds)).sf).min()
    got = np.ravel(IscwsaClearance(ref, off).sf).min()
    assert got == pytest.approx(want, rel=1e-6)


@pytest.fixture(scope="module")
def end_approach_pair():
    # reference: vertical, stations every 100 m to 1000 m
    md = np.arange(0.0, 1001.0, 100.0)  # grid-lint: ok straight well
    ref = we.survey.Survey(md=md, inc=np.zeros_like(md), azi=np.zeros_like(md),
                           header=_header(), error_model="ISCWSA MWD Rev4")
    # offset: horizontal due east at TVD 950 m, 8 m north of the reference
    omd = np.arange(0.0, 2001.0, 100.0)  # grid-lint: ok straight well
    off = we.survey.Survey(md=omd, inc=np.full_like(omd, 90.0),
                           azi=np.full_like(omd, 90.0), header=_header(),
                           start_nev=[8, -1000, 950],
                           error_model="ISCWSA MWD Rev4")
    return ref, off


def test_minimum_inside_the_last_reference_leg_is_found(end_approach_pair):
    """The last station (1000 m) is the lowest station value; the minimum is
    at TVD 950 m, inside the last leg (exact geometry: 8 m centre to centre)."""
    ref, off = end_approach_pair
    c = IscwsaClearance(ref, off)
    sf = np.ravel(c.sf)
    i = int(np.argmin(sf))
    assert c.ref.md[i] == pytest.approx(950.0, abs=0.5)
    assert float(np.ravel(c.distance_cc)[i]) == pytest.approx(8.0, abs=1e-3)
    # densified reference: the minimum is then an interior station
    mds = np.arange(10.0, 1000.0, 10.0)  # grid-lint: ok densified oracle
    dense = IscwsaClearance(ref.interpolate_mds(mds), off)
    assert sf.min() == pytest.approx(np.ravel(dense.sf).min(), rel=1e-6)
    # and the Mahalanobis path, which searches both end legs on its own
    assert sf.min() == pytest.approx(MahalanobisClearance(ref, off).min_sf, rel=1e-5)


def test_no_point_is_inserted_where_the_first_leg_only_falls(end_approach_pair):
    """The first station is above its neighbour, so the first leg holds no
    minimum and nothing is inserted there."""
    ref, off = end_approach_pair
    c = IscwsaClearance(ref, off)
    assert np.all(np.isin(c.ref.md[c.ref.md < 900.0], ref.md))


def test_offset_radius_at_the_closest_point_not_the_nearest_station(
        sparse_offset_pair):
    """Offset hole radius 0.3 m on the tangent and 0.1 m beyond it. The
    approach is on the tangent, whose stations are not the nearest one, so
    the radius must come from the leg (the preceding station's, the rule
    interpolate_mds carries), as the densified offset does."""
    ref, off = sparse_offset_pair
    off = we.survey.Survey(md=off.md, inc=off.inc_deg, azi=off.azi_grid_deg,
                           header=_header(), start_nev=[-1000, 0, 1000],
                           radius=np.array([0.3, 0.1, 0.1, 0.1, 0.1]),
                           error_model="ISCWSA MWD Rev4")
    mds = np.arange(10.0, off.md[-1], 10.0)  # grid-lint: ok densified oracle
    dense = off.interpolate_mds(mds)
    want = np.ravel(IscwsaClearance(ref, dense).sf).min()
    assert np.ravel(IscwsaClearance(ref, off).sf).min() == pytest.approx(want, rel=1e-6)
    want = MahalanobisClearance(ref, dense).min_sf
    assert MahalanobisClearance(ref, off).min_sf == pytest.approx(want, rel=1e-5)


def test_reference_radius_inside_a_refined_leg(end_approach_pair):
    """Reference radius 0.5 m to 900 m and 0.1 m at 1000 m. The minimum is
    inside the last leg, where the preceding station's radius applies (the
    rule interpolate_mds carries): the value minimised is the value
    reported, and both equal the densified reference's."""
    ref, off = end_approach_pair
    rad = np.where(ref.md < 950.0, 0.5, 0.1)
    ref = we.survey.Survey(md=ref.md, inc=ref.inc_deg, azi=ref.azi_grid_deg,
                           header=_header(), radius=rad,
                           error_model="ISCWSA MWD Rev4")
    mds = np.arange(10.0, 1000.0, 10.0)  # grid-lint: ok densified oracle
    want = np.ravel(IscwsaClearance(ref.interpolate_mds(mds), off).sf).min()
    assert np.ravel(IscwsaClearance(ref, off).sf).min() == pytest.approx(want, rel=1e-6)
    # the search's own value at its minimum
    c0 = IscwsaClearance(ref, off, minimize_sf=False)
    n = len(ref.md)
    _, searched = c0._interval_minimum(n - 1, np.array([ref.delta_md[n - 1], 0.0]))
    assert searched == pytest.approx(want, rel=1e-6)
