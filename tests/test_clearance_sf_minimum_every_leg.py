"""The separation-factor minimum inside a leg whose stations are not minima.

Reference: vertical, stations every 500 m. The offset, 52.7 m east at
surface, swings toward it across the reference leg 500-1000 m and then
converges slowly below 1500 m, so the stations read 0.612, 0.598, 0.576 down
to 1500 m: the station minimum is at 1500 m, and neither station of the leg
holding the closest approach is a local minimum. Searching only next to the
station minimum reported 0.549; the wells overlap (SF below zero) at about
MD 689 m. The oracle is a brute-force scan of the same separation factor at
1 m along every reference leg.
"""
import numpy as np

import welleng as we
from welleng.clearance import IscwsaClearance

HEADER = dict(b_total=50_000., dip=72., declination=1.)


def _well(md, inc, azi, east):
    return we.survey.Survey(
        md=np.asarray(md, float), inc=np.asarray(inc, float),
        azi=np.asarray(azi, float), header=we.survey.SurveyHeader(**HEADER),
        error_model="ISCWSA MWD Rev5.11", start_nev=[0., east, 0.],
        start_xyz=[east, 0., 0.])


def _pair():
    ref = _well([0, 500, 1000, 1500, 2000, 2500, 3000], [0] * 7, [0] * 7, 0.0)
    md = np.arange(0, 3001, 100.)  # grid-lint: ok survey stations
    peak, centre, deep = 37.61936557162207, 722.5562605605994, 9.619026726518346
    inc = np.zeros(len(md))
    rise = (md >= centre - 200) & (md < centre)
    fall = (md >= centre) & (md < centre + 200)
    inc[rise] = peak * (md[rise] - (centre - 200)) / 200
    inc[fall] = peak * (1 - (md[fall] - centre) / 200)
    inc[md >= 1500] = deep
    off = _well(md, inc, np.full(len(md), 270.), 52.748770484047)
    return ref, off


def _brute_force_minimum(c):
    """Minimum of the separation factor at 1 m along every reference leg."""
    best = float(np.nanmin(c.sf))
    md = np.asarray(c.ref.md)
    for j in range(len(md) - 1):
        length = md[j + 1] - md[j]
        for x in np.arange(1.0, length, 1.0):  # grid-lint: ok brute-force oracle
            best = min(best, c._get_sf_min(np.array([x]), j,
                                           np.array([0.0, length])))
    return best


def test_a_minimum_inside_a_leg_without_a_station_minimum_is_found():
    ref, off = _pair()
    stations = IscwsaClearance(ref, off, minimize_sf=False)
    sf = np.ravel(stations.sf)
    # the station minimum is at 1500 m; stations 500 and 1000 are not minima
    assert np.argmin(sf) == 3
    assert sf[1] > sf[2] > sf[3]
    oracle = _brute_force_minimum(stations)
    assert oracle < 0.0                       # the wells overlap

    found = float(np.nanmin(IscwsaClearance(ref, off).sf))
    assert found <= oracle + 1e-6
    assert found < 0.0
