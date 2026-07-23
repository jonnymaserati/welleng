"""Rev5+ surface tie-on (Definition of ISCWSA Error Model §4.7.1.1, eq. 32).

The slot attitude carries its own measurement error (magnitude = a downhole
survey), modelled by doubling the FIRST survey station's inc AND azi weighting
columns -- the full middle and right-hand columns of eq. (10), i.e.
inc-{N,E,V} + azi-{N,E}.

The ISCWSA reference well #1 is vertical-North at station 1 (inc=azi=0), so only
inc-N is non-zero there and doubling inc-N alone reproduces the reference; the
E/V/azi terms only bite for a DEVIATED first survey (immediate kickoff, platform
slot, composed sidetrack). These tests exercise that deviated case, which the
conformance workbook does not.
"""
import numpy as np

from welleng.survey import Survey, SurveyHeader

MAG = dict(b_total=50000, dip=70, declination=0, convergence=0,
           azi_reference="grid")


def _drdp_row1(inc1_deg, azi1_deg, model):
    """drdp row for station 1, given a first-downhole-survey attitude."""
    s = Survey(
        md=[0., 30., 100., 200.],
        inc=[0., inc1_deg, 30., 45.],
        azi=[0., azi1_deg, 40., 50.],
        header=SurveyHeader(**MAG),
        error_model=model,
    )
    return np.asarray(s.err.drdp, float)[1]


def test_rev5_surface_tieon_doubles_full_inc_and_azi_columns():
    """For a deviated slot, rev5 doubles the WHOLE inc+azi columns (cols 3:9),
    not just inc-N. rev4 (pre-tie-on) uses the same base weighting undoubled, so
    the rev5 row is exactly 2x the rev4 row on those columns."""
    row5 = _drdp_row1(20., 45., "ISCWSA MWD Rev5.11")[3:9]
    row4 = _drdp_row1(20., 45., "ISCWSA MWD Rev4")[3:9]
    np.testing.assert_allclose(row5, 2.0 * row4, rtol=1e-9, atol=1e-12)
    # inc-E (col4), inc-V (col5), azi-N (col6), azi-E (col7) are all genuinely
    # non-zero for a deviated slot -- exactly the terms the old inc-N-only code
    # dropped.
    assert np.all(np.abs(row4[1:5]) > 1e-6), row4


def test_rev4_has_no_surface_tieon():
    """rev4 predates the surface tie-on: station 1 is not doubled at all."""
    row4 = _drdp_row1(20., 45., "ISCWSA MWD Rev4")
    # base balanced-tangential weighting, undoubled
    half = 0.5 * (30. - 0.)
    si, ci = np.sin(np.radians(20.)), np.cos(np.radians(20.))
    sa, ca = np.sin(np.radians(45.)), np.cos(np.radians(45.))
    expect = np.array([half * ci * ca, half * ci * sa, -half * si,
                       -half * si * sa, half * si * ca, 0.0])
    np.testing.assert_allclose(row4[3:9], expect, rtol=1e-9, atol=1e-12)


def test_vertical_north_slot_reproduces_reference_behaviour():
    """ISCWSA #1 geometry at station 1 (inc=azi=0): only inc-N is non-zero, so
    the completed doubling reproduces the historical inc-N-only result -> the
    conformance workbook value is unchanged."""
    row5 = _drdp_row1(0., 0., "ISCWSA MWD Rev5.11")
    half = 0.5 * (30. - 0.)
    np.testing.assert_allclose(row5[3], 2.0 * half, rtol=1e-9)   # inc-N doubled
    np.testing.assert_allclose(row5[4:9], 0.0, atol=1e-12)       # rest vanish
