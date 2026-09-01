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

from welleng.error import ErrorModel
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


def test_surface_tieon_only_at_true_surface_root():
    """The tie-on applies only at a true surface root (md[0] == 0). A survey
    starting below the datum (md[0] != 0, e.g. a composed/hierarchy sub-survey)
    carries station 0 externally and gets NO slot allowance -- mirroring the
    station-0 depth (DRFR) gate. Same geometry rooted at md=0 IS doubled, so the
    below-datum row is exactly half the surface-rooted row on cols 3:9."""
    model = "ISCWSA MWD Rev5.11"
    surf = Survey(md=[0., 30., 100., 200.], inc=[0., 20., 30., 45.],
                  azi=[0., 45., 40., 50.], header=SurveyHeader(**MAG),
                  error_model=model)
    deep = Survey(md=[1000., 1030., 1100., 1200.], inc=[0., 20., 30., 45.],
                  azi=[0., 45., 40., 50.], header=SurveyHeader(**MAG),
                  error_model=model)
    r_surf = np.asarray(surf.err.drdp, float)[1, 3:9]
    r_deep = np.asarray(deep.err.drdp, float)[1, 3:9]
    np.testing.assert_allclose(r_deep, 0.5 * r_surf, rtol=1e-9, atol=1e-12)
    assert np.any(np.abs(r_deep) > 1e-6)   # base weighting present, undoubled


def test_vertical_north_slot_reproduces_reference_behaviour():
    """ISCWSA #1 geometry at station 1 (inc=azi=0): only inc-N is non-zero, so
    the completed doubling reproduces the historical inc-N-only result -> the
    conformance workbook value is unchanged."""
    row5 = _drdp_row1(0., 0., "ISCWSA MWD Rev5.11")
    half = 0.5 * (30. - 0.)
    np.testing.assert_allclose(row5[3], 2.0 * half, rtol=1e-9)   # inc-N doubled
    np.testing.assert_allclose(row5[4:9], 0.0, atol=1e-12)       # rest vanish


def test_surface_tieon_gate_keys_on_framework_and_revision():
    """The Rev5 surface tie-on is keyed on model metadata (framework +
    revision_number), NOT a name-string suffix. Verdict on known cases:
    fires for ISCWSA-standard Rev5+, silent for Rev4/earlier AND for a COMPASS
    IPM import (which carries its OWN tie-on and whose vendor-descriptor name
    would slip a string-suffix gate)."""
    w = ErrorModel._wants_surface_tieon
    # FIRES -- ISCWSA-standard Rev5+ (registry string + OWSG/ISCWSA-JSON dict)
    assert w("ISCWSA MWD Rev5.11") is True
    assert w({"metadata": {"framework": "ISCWSA Rev5", "revision_number": 5,
                           "short_name": "MWD+SRGM"}}) is True
    # SILENT -- Rev4 and earlier predate the slot allowance
    assert w("ISCWSA MWD Rev4") is False
    assert w("ISCWSA MWD Rev3") is False           # JJ: pre-rev4 must not fire
    assert w({"metadata": {"framework": "ISCWSA Rev4",
                           "revision_number": 4}}) is False
    # SILENT -- COMPASS IPM import (own tie-on); a vendor-descriptor name
    # ("...mag-corr") slips a string-suffix gate, the framework keys it out.
    assert w({"metadata": {"framework": "COMPASS IPM",
                           "short_name": "Magnetic, std, mag-corr"}}) is False


def _first_station_sigma_v(md0, tie_on):
    """sigma_V at the first station for a survey rooted at md0, given tie_on."""
    md = [md0, md0 + 30., md0 + 100., md0 + 200.]
    s = Survey(md=md, inc=[0., 20., 30., 45.], azi=[0., 45., 40., 50.],
               header=SurveyHeader(tie_on=tie_on, **MAG), error_model="ISCWSA MWD Rev5.11")
    return float(np.sqrt(np.asarray(s.err.errors.cov_NEVs)[0, 2, 2])), s.err._first_station_is_root


def test_tie_on_flag_role_governs_first_station_root():
    """The first-station role (root vs tie-on) governs the seed, not the md
    value. Default (None) infers from md==0 -- a below-datum start is a tie-on
    with ~zero first-station uncertainty (reference behaviour). tie_on=False
    forces a SUBSEA root at md!=0, re-engaging the station-0 seed (placeholder
    subsea path). tie_on=True forces a tie-on even at md==0."""
    # default at surface root (md==0): seeded
    sv_surf, root_surf = _first_station_sigma_v(0.0, None)
    assert root_surf is True and sv_surf > 0.0
    # default below datum (md!=0): tie-on, no first-station seed (matches reference)
    sv_tie, root_tie = _first_station_sigma_v(91.0, None)
    assert root_tie is False and sv_tie == 0.0
    # SUBSEA opt-in: force root at md!=0 -> seed re-engages (magnitude ~ the
    # surface-root seed; not identical -- the depth error's scale component is
    # evaluated at the station's own md, so a 91 m root seeds slightly more)
    sv_subsea, root_subsea = _first_station_sigma_v(91.0, False)
    assert root_subsea is True and sv_subsea > 0.0
    np.testing.assert_allclose(sv_subsea, sv_surf, rtol=0.05)
    # force tie-on at md==0 -> no seed
    sv_forced_tie, root_forced_tie = _first_station_sigma_v(0.0, True)
    assert root_forced_tie is False and sv_forced_tie == 0.0
