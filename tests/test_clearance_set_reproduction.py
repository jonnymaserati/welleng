"""The ISCWSA standard clearance well set, reproduced two ways.

1. Against ISCWSA's own Rev 4 calculator (the MWD Error Model Example Spreadsheet,
   Revision 4), recalculated with each well of the set: core's default reproduces it
   at every station. Where the two differ, the difference is pinned, so a change in
   either direction fails.
2. Against the set's own listing (r4, 17 May 2017), whose covariances were
   generated with the first station's attitude error spanning the first leg and the
   surface position exact: ``first_station_allowance=True`` with ``tie_on=True``.
   Its azimuths are grid north (ED50 / UTM zone 31N), each well with the grid
   convergence of its own slot, which the listing states to 0.01 deg.

The listing is tabulated to 4 decimal places and is not consistent with itself at the
last digit (test below), so it is reproduced to its rounding plus a stated tolerance.
"""
import json

import numpy as np
from pyproj import Proj

from welleng.clearance import IscwsaClearance
from welleng.survey import Survey, make_survey_header

with open("tests/test_data/clearance_iscwsa_well_data.json") as f:
    SET = json.load(f)["wells"]
with open("tests/test_data/clearance_set_iscwsa_rev4_workbook.json") as f:
    WORKBOOK = json.load(f)["sigma_hla"]

# slot easting and northing (m, ED50 / UTM zone 31N) from each well's sheet
SLOT = {
    "Reference well": (500000.00, 6651566.71), "01 - well": (500099.96, 6651566.71),
    "02 - well": (500000.00, 6651666.67), "03 - well": (500010.00, 6651566.71),
    "04 - well": (500019.99, 6651566.71), "05 - well": (499500.20, 6651516.73),
    "06 - well": (500099.96, 6649267.63), "07 - well": (500079.97, 6649267.63),
    "08 - well": (499100.36, 6651066.91), "09 - well": (500000.00, 6650567.11),
    "10 - well": (500000.00, 6651566.71), "11 - well": (499000.40, 6650167.27),
}


def _slot_convergence(name):
    """Grid convergence at the well's slot, deg (positive east)."""
    p = Proj("EPSG:23031")
    lon, lat = p(*SLOT[name], inverse=True)
    return p.get_factors(lon, lat).meridian_convergence


def _survey(w, allowance=None, tie_on=None, convergence=0.0):
    h = dict(w["header"])
    h["convergence"] = np.deg2rad(convergence)
    sh = make_survey_header(h)
    sh.first_station_allowance = allowance
    sh.tie_on = tie_on
    return Survey(md=w["MD"], inc=w["IncDeg"], azi=w["AziDeg"], n=w["N"], e=w["E"],
                  tvd=w["TVD"], header=sh, error_model="ISCWSA MWD Rev4",
                  start_nev=[w["N"][0], w["E"][0], w["TVD"][0]], deg=True,
                  unit="meters")


def _sigma_hla(s, n):
    c = np.asarray(s.cov_hla)
    return np.sqrt(np.stack([c[:, 0, 0], c[:, 1, 1], c[:, 2, 2]], axis=1))[:n]


def test_core_reproduces_the_iscwsa_rev4_calculator_on_the_set():
    # max |core - calculator| per well, in metres, over every station below MD 0
    # (the calculator reports 0 at the datum, core the datum's DRFR).
    pinned = {"06 - well": 4.7155e-05, "07 - well": 1.8606e-04, "08 - well": 2.0780e-02,
              "10 - well": 7.0271e-03, "11 - well": 3.2631e-02}
    for name, w in SET.items():
        got = _sigma_hla(_survey(w), len(w["MD"]))
        d = np.abs(got - np.asarray(WORKBOOK[name]))[1:].max()
        if name in pinned:
            assert abs(d - pinned[name]) < 1e-6, (name, d)
        else:
            assert d < 1e-9, (name, d)


def test_the_option_reproduces_the_set_listing():
    # 4 dp listing: rounding half-width 5e-5 m; the remaining tolerance is the set's
    # own implementation difference, pinned per well where it is larger. The
    # sidetrack (10) carries its kick-off station's error uncorrelated (ISCWSA
    # common practice 2017, section 2.8); its residual is an exchange of variance
    # between H and A in the build, under 0.2 mm on the hold.
    pinned = {"10 - well": 3.9303e-03}
    for name, w in SET.items():
        allowance = "uncorrelated" if name == "10 - well" else True
        s = _survey(w, allowance, True, _slot_convergence(name))
        got = _sigma_hla(s, len(w["MD"]))
        want = np.array([w["sigmaH"], w["sigmaL"], w["sigmaA"]]).T
        d = np.abs(got - want).max()
        if name in pinned:
            assert abs(d - pinned[name]) < 1e-6, (name, d)
        else:
            assert d < 1.5e-4, (name, d)


def test_the_option_reproduces_the_set_separation_factors():
    S = {n: _survey(w, True, True, _slot_convergence(n)) for n, w in SET.items()}
    for n, s in S.items():
        s.radius = np.full(len(s.md), 0.4572 if n == "Reference well" else 0.3048)
    for name, s in S.items():
        if name in ("Reference well", "10 - well"):
            continue
        r = IscwsaClearance(S["Reference well"], s, minimize_sf=False)
        sf = np.asarray(r.sf)[~np.asarray(r.ref.interpolated, dtype=bool)]
        want = np.asarray(SET[name]["SF"], dtype=float)
        m = np.isfinite(want) & (want < 10.0)    # the set's stated range: SF < 10
        assert np.abs(sf[:len(want)] - want)[m].max() < 0.002, name


def test_the_listing_differs_from_itself_at_its_last_digit():
    # wells 05 and 09 share every input through their vertical section (MD, inc 0,
    # azi 0), and a station's covariance depends only on the stations up to the next
    # one, yet the listing gives them different values: no calculation from the
    # published inputs returns both.
    a, b = SET["05 - well"], SET["09 - well"]
    for md in (90.0, 120.0):
        i, j = a["MD"].index(md), b["MD"].index(md)
        assert (a["IncDeg"][i], a["AziDeg"][i]) == (0, 0)
        assert (b["IncDeg"][j], b["AziDeg"][j]) == (0, 0)
        assert a["MD"][:i + 2] == b["MD"][:j + 2]
        assert a["IncDeg"][:i + 2] == b["IncDeg"][:j + 2]
        assert a["sigmaH"][i] != b["sigmaH"][j]


def test_the_tie_on_station_carried_uncorrelated():
    """first_station_allowance="uncorrelated": the first station's own attitude
    error over half its first leg, from every non-depth source through its own
    weights, added once and independently beyond the first station."""
    w = SET["10 - well"]
    plain = _survey(w, None, True)
    tied = _survey(w, "uncorrelated", True)
    tie = tied.err.errors.cov_NEV_tie
    # vertical kick-off: the misalignments XYM3/XYM4 give each lateral axis
    # (0.5 x 30 m x 0.1 deg)^2; the accelerometer terms add a little; no vertical
    xym = (0.5 * 30.0 * np.deg2rad(0.1)) ** 2
    assert xym < tie[0, 0] < 1.1 * xym and xym < tie[1, 1] < 1.1 * xym
    assert abs(tie[2, 2]) < 1e-12
    # zero at the first station, the same constant at every later one
    d = np.asarray(tied.cov_nev) - np.asarray(plain.cov_nev)
    assert np.allclose(d[0], 0.0) and np.allclose(d[1:], tie)
    # the interior covariance takes it over the first half-leg, whole beyond
    assert np.allclose(tied.err.cov_nev_at(w["MD"][1]), np.asarray(tied.cov_nev)[1])
    q = w["MD"][0] + 15.0
    half = tied.err.cov_nev_at(q) - plain.err.cov_nev_at(q)
    assert np.allclose(half, tie)
    # it is not the coherent allowance as well
    assert not tied.err._first_station_allowance


def test_first_station_allowance_takes_four_values():
    import pytest
    from welleng.survey import SurveyHeader
    for v in (None, True, False, "uncorrelated"):
        SurveyHeader(first_station_allowance=v)
    with pytest.raises(ValueError, match="first_station_allowance"):
        SurveyHeader(first_station_allowance="coherent")


def test_slot_convergence_is_the_listings():
    # each sheet's "Convergence at slot", to its 0.01 deg; and without it well 05,
    # the one hold at an oblique magnetic azimuth, is short by 0.72 mm in sigma L
    stated = {"05 - well": -0.01, "08 - well": -0.01, "11 - well": -0.02}
    for name in SLOT:
        assert round(_slot_convergence(name), 2) + 0.0 == stated.get(name, 0.0), name
    w = SET["05 - well"]
    want = np.array(w["sigmaL"])
    for conv, d in ((0.0, 7.1990e-04), (_slot_convergence("05 - well"), 1.5e-4)):
        got = _sigma_hla(_survey(w, True, True, conv), len(w["MD"]))[:, 1]
        e = np.abs(got - want).max()
        assert (abs(e - d) < 1e-6) if conv == 0.0 else e < d, (conv, e)
