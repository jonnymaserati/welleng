"""Validation tests for the Tortuosity Index (TI) and Modified Tortuosity
Index (MTI).

References
----------
- Ashok et al., IADD presentation "Measuring Wellbore Tortuosity".
- D'Angelo et al., SPE/IADC-194099-MS "Unplanned Tortuosity Index".
- Corcutt, https://jonnymaserati.github.io tortuosity-index post series.

The published method gives no raw survey data (proprietary wells), so these
tests validate against (a) analytic closed forms, (b) dimensional / qualitative
invariants, and (c) regression anchors re-baselined against the current
implementation. See docs/dev/VALIDATION.md.
"""
import numpy as np
import welleng as we


def iscwsa_1():
    sh = we.survey.SurveyHeader(
        name="ISCWSA No. 1", latitude=60, longitude=2, G=9.80665,
        b_total=50_000, dip=72, declination=-4,
        vertical_section_azimuth=75, azi_reference='magnetic')
    md, inc, azi = np.array([
        [0.0, 0.0, 0.0], [1200.0, 0.0, 0.0], [2100.0, 60.0, 75.0],
        [5100.0, 60.0, 75.0], [5400.0, 90.0, 75.0], [8000.0, 90.0, 75.0]]).T
    return we.survey.Survey(md, inc, azi, header=sh)


def iscwsa_2():
    header = we.survey.SurveyHeader(
        name="ISCWSA No. 2", latitude=28, longitude=-90, G=9.80665,
        b_total=48_000, dip=58, declination=2,
        vertical_section_azimuth=21, azi_reference='true')
    md, inc, azi = np.array([
        [0.0, 0.0, 0.0], [2000.0, 0.0, 0.0], [3600.0, 32.0, 2.0],
        [5000.0, 32.0, 2.0], [5525.54, 32.0, 32.0], [6051.08, 32.0, 62.0],
        [6576.62, 32.0, 92.0], [7102.16, 32.0, 122.0], [9398.5, 60.0, 220.0],
        [12500.0, 60.0, 220.0]]).T
    return we.survey.Survey(md * 0.3048, inc, azi, header=header)


def test_straight_hole_is_zero():
    """A perfectly straight hole has zero tortuosity (TI and MTI)."""
    sh = we.survey.SurveyHeader(name="straight")
    s = we.survey.Survey(
        md=np.array([0., 500., 1000., 2000.]),
        inc=np.zeros(4), azi=np.zeros(4), header=sh
    ).interpolate_survey(step=30)
    assert np.nanmax(np.abs(we.survey.tortuosity_index(s))) == 0.0
    assert np.nanmax(np.abs(s.modified_tortuosity_index(dls_noise=None))) == 0.0


def test_single_arc_matches_analytic():
    """A single continuous arc must equal the closed form
    TI = 1/2 * (kappa / L_c_ft) * (L_arc / L_chord - 1).
    """
    sh = we.survey.SurveyHeader(name="arc")
    s = we.survey.Survey(
        md=np.array([0., 300., 600.]),
        inc=np.array([0., 30., 60.]),
        azi=np.zeros(3), header=sh
    ).interpolate_survey(step=5)

    data = we.survey.tortuosity_index(s, data=True)
    # the whole arc is a single curve-turn section
    assert list(data['n_sections']) == [1]

    l_c = s.md[-1] - s.md[0]
    chord = np.linalg.norm(s.pos_nev[-1] - s.pos_nev[0])
    analytic = 0.5 * (1e7 / (l_c / 0.3048)) * ((l_c / chord) - 1)
    assert np.isclose(data['ti'][-1], analytic, rtol=1e-9)


def test_mti_is_dimensionless():
    """The MTI of a survey in metres equals that of the same survey in feet."""
    s_m = iscwsa_1()
    s_ft = we.survey.Survey(
        md=s_m.md / 0.3048, inc=s_m.inc_deg, azi=s_m.azi_grid_deg,
        header=we.survey.SurveyHeader(name="ft"))
    mti_m = s_m.modified_tortuosity_index(dls_noise=None)[-1]
    mti_ft = s_ft.modified_tortuosity_index(dls_noise=None)[-1]
    assert np.isclose(mti_ft, mti_m, rtol=1e-9)


def test_ti_scales_with_length_unit():
    """TI is NOT dimensionless (by design): it scales with the unit of length.
    The same geometry in feet vs metres differs by the 0.3048 factor.
    """
    s_m = iscwsa_1()
    s_ft = we.survey.Survey(
        md=s_m.md / 0.3048, inc=s_m.inc_deg, azi=s_m.azi_grid_deg,
        header=we.survey.SurveyHeader(name="ft"))
    ti_m = we.survey.tortuosity_index(s_m)[-1]
    ti_ft = we.survey.tortuosity_index(s_ft)[-1]
    assert np.isclose(ti_ft / ti_m, 0.3048, rtol=1e-9)


def test_maximum_curvature_increases_mti():
    """The maximum-curvature pre-processing makes a survey more tortuous, and
    interpolating before it increases it further (the survey-frequency result).
    """
    s = iscwsa_2()
    mti_minc = s.modified_tortuosity_index(dls_noise=None)[-1]
    mti_maxc = s.modified_tortuosity_index(step=None, dls_noise=1.0)[-1]
    mti_maxc_30 = s.modified_tortuosity_index(step=30, dls_noise=1.0)[-1]
    assert mti_maxc > mti_minc
    assert mti_maxc_30 > mti_maxc


def test_kappa_override_and_legacy_kapa():
    """The 'kappa' scale factor must be honourable; the legacy 'kapa' typo is
    still accepted for back-compatibility (regression for the silent-ignore
    bug where only the misspelt key worked).
    """
    s = iscwsa_1().interpolate_survey(step=30)
    base = we.survey.tortuosity_index(s)[-1]                 # kappa=1e7
    via_kappa = we.survey.tortuosity_index(s, kappa=1.0)[-1]
    via_kapa = we.survey.tortuosity_index(s, kapa=1.0)[-1]
    assert np.isclose(via_kappa, base * 1e-7, rtol=1e-9)     # override is live
    assert np.isclose(via_kapa, via_kappa, rtol=1e-9)        # legacy still works


def test_coeff_override():
    """coeff sets the length unit used for L_c normalization in TI."""
    s = iscwsa_1().interpolate_survey(step=30)
    ti_ft = we.survey.tortuosity_index(s, coeff=0.3048)[-1]
    ti_m = we.survey.tortuosity_index(s, coeff=1.0)[-1]
    assert np.isclose(ti_ft / ti_m, 0.3048, rtol=1e-9)


def test_dls_tol_is_forwarded_and_splits_sections():
    """Survey.tortuosity_index must forward dls_tol to the engine (regression
    for the bug where the method hardcoded dls_tol=None). With dls_tol set, two
    arcs in the same plane but with different DLS are split into two sections.
    """
    sh = we.survey.SurveyHeader(name="two-dls")
    # 0->30 deg over 300 m (fast), 30->45 deg over 450 m (slow); same plane.
    s = we.survey.Survey(
        md=np.array([0., 300., 750.]),
        inc=np.array([0., 30., 45.]),
        azi=np.zeros(3), header=sh
    ).interpolate_survey(step=10)

    n_normal = len(we.survey.tortuosity_index(s, dls_tol=None, data=True)['n_sections'])
    n_dls = len(we.survey.tortuosity_index(s, dls_tol=0.01, data=True)['n_sections'])
    assert n_normal == 1
    assert n_dls == 2

    # the method must produce the same result as the module function -> forwarded
    via_method = s.tortuosity_index(dls_tol=0.01)[-1]
    via_module = we.survey.tortuosity_index(s, dls_tol=0.01)[-1]
    assert np.isclose(via_method, via_module, rtol=1e-9)
    # and it must differ from the dls_tol=None case (proves it is not ignored)
    assert not np.isclose(via_method, s.tortuosity_index(dls_tol=None)[-1], rtol=1e-6)


def test_tortuosity_views():
    """total / remaining / local readings of the profile behave correctly."""
    s = iscwsa_1().interpolate_survey(step=30)
    prof = s.modified_tortuosity_index(dls_noise=None)
    v = s.tortuosity_views(modified=True, dls_noise=None)
    assert {'md', 'total', 'remaining', 'local'} <= set(v)
    assert np.isclose(v['total'], prof[-1])
    assert np.isclose(v['remaining'][0], v['total'])     # all of it still ahead at the start
    assert np.isclose(v['remaining'][-1], 0.0)           # nothing left at TD
    assert np.all(v['local'] >= -1e-12)                  # MTI non-decreasing -> local >= 0

    # local tortuosity is higher in a build than in a tangent hold
    i_build = np.argmin(np.abs(v['md'] - 1800.0))
    i_hold = np.argmin(np.abs(v['md'] - 3500.0))
    assert v['local'][i_build] > v['local'][i_hold]

    # target_md, module-helper consistency, and the TI (non-modified) variant
    vt = s.tortuosity_views(modified=True, dls_noise=None, target_md=5100.0)
    assert vt['remaining'][0] > 0.0
    assert np.isclose(we.survey.tortuosity_views(prof, s.md)['total'], prof[-1])
    assert s.tortuosity_views(modified=False)['total'] > 0.0


def test_regression_anchors():
    """Lock current TI/MTI values for the ISCWSA reference wells. These are
    re-baselined against the current implementation (the 2022 blog absolutes
    drifted ~20% with maximum_curvature / interpolate evolution; the qualitative
    invariants above are the scientific check)."""
    s1 = iscwsa_1().interpolate_survey(step=30)
    s2 = iscwsa_2()
    # Deterministic anchors: no maximum-curvature pre-processing -> tight.
    assert np.isclose(we.survey.tortuosity_index(s1)[-1], 18.641285781891, rtol=1e-9)
    assert np.isclose(s1.modified_tortuosity_index(dls_noise=None)[-1], 0.605503933178, rtol=1e-9)
    assert np.isclose(s2.modified_tortuosity_index(dls_noise=None)[-1], 0.524669579439, rtol=1e-9)
    # dls_noise=1.0 routes through maximum_curvature -> interpolate + 3D
    # sectionization, whose section boundaries sit on a floating-point threshold.
    # A borderline station can flip a boundary under sub-epsilon, run-to-run
    # float variation (e.g. threaded-BLAS reductions), giving a discrete ~0.3%
    # jump in the final MTI. A 1e-9 anchor is therefore flaky across runs/Python
    # versions (observed on CI 3.11); anchor these to 1% so they still catch gross
    # regressions without pinning the knife-edge. Qualitative invariants above are
    # the scientific check.
    assert np.isclose(s2.modified_tortuosity_index(step=None, dls_noise=1.0)[-1], 0.7308, rtol=1e-2)
    assert np.isclose(s2.modified_tortuosity_index(step=30, dls_noise=1.0)[-1], 0.8199, rtol=1e-2)
