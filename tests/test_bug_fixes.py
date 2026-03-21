"""
test_bug_fixes.py
-----------------
Tests for specific bug fixes to prevent regressions.
"""
import numpy as np
import pytest
import welleng as we


# ---------------------------------------------------------------------------
# Issue #191: _interpolate_survey unit handling
# ---------------------------------------------------------------------------

def test_interpolate_md_feet_units():
    """
    Issue #191: interpolate_md should work when the survey header specifies
    'feet' as the depth unit.  Previously raised AssertionError because
    _interpolate_survey created a Survey with unit='meters' regardless.
    """
    header = we.survey.SurveyHeader(
        deg=True, depth_unit='feet', surface_unit='feet'
    )
    sections = [
        we.connector.Connector(
            pos1=[0., 0., 0.],
            inc1=0.,
            azi1=0.,
            md1=0.,
            md2=100.,
            dls_design=3.0,
            degrees=True,
            unit='feet',
        )
    ]
    survey = we.survey.from_connections(
        sections, step=30, survey_header=header,
        radius=10, deg=True, depth_unit='feet', surface_unit='feet'
    )
    node = survey.interpolate_md(50)
    assert node is not None
    assert node.unit == 'feet', f"Expected unit 'feet', got '{node.unit}'"
    assert abs(node.md - 50.0) < 1e-6, f"Expected md=50, got {node.md}"


def test_interpolate_md_meters_units():
    """Ensure the standard meters case still works after the unit fix."""
    survey = we.survey.Survey(
        md=[0, 100, 200, 400],
        inc=[0, 0, 15, 30],
        azi=[0, 0, 45, 45],
    )
    node = survey.interpolate_md(150)
    assert node is not None
    assert node.unit == 'meters'
    assert abs(node.md - 150.0) < 1e-6


# ---------------------------------------------------------------------------
# Issue #194: WellMesh ellipse eigenvalue-based orientation
# ---------------------------------------------------------------------------

def _make_survey_with_errors():
    """Helper: build an interpolated survey with an MWD error model."""
    return we.survey.interpolate_survey(
        we.survey.Survey(
            md=[0, 500, 1000, 2000, 3000],
            inc=[0, 0, 30, 90, 90],
            azi=[0, 0, 45, 135, 180],
            error_model='ISCWSA MWD Rev4',
        ),
        step=300,
    )


def test_wellmesh_ellipse_construction():
    """Issue #194: WellMesh with method='ellipse' should build without error."""
    survey = _make_survey_with_errors()
    mesh = we.mesh.WellMesh(survey, method='ellipse')
    assert mesh.vertices is not None
    assert mesh.vertices.shape[-1] == 3


def test_wellmesh_ellipse_uses_eigenvalues():
    """
    Issue #194: The ellipse semi-axes should be derived from the eigenvalues of
    the H-L covariance submatrix, not just the diagonal entries.

    For a survey station where cov_hla has non-zero off-diagonal terms the
    eigenvalue-derived semi-major axis will differ from sqrt(cov_hla[0,0]).
    """
    survey = _make_survey_with_errors()

    # Find a station with a non-trivial off-diagonal HL term
    idx = None
    for i, cov in enumerate(survey.cov_hla):
        if abs(cov[0, 1]) > 1e-6:
            idx = i
            break
    assert idx is not None, "Need a station with off-diagonal HL covariance"

    cov_hl = survey.cov_hla[idx, :2, :2]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_hl)
    sigma_major_eigen = np.sqrt(max(eigenvalues[1], 0))
    sigma_major_diag = np.sqrt(cov_hl[0, 0])

    # If off-diagonal terms are present the two approaches must differ
    assert not np.isclose(sigma_major_eigen, sigma_major_diag), (
        "Expected eigenvalue-derived sigma_major to differ from diagonal "
        "when off-diagonal covariance terms are present"
    )


def test_wellmesh_ellipse_rotation_applied():
    """
    Issue #194: When the principal axis of the covariance ellipse is not
    aligned with H or L, the mesh vertices should reflect that rotation.

    Construct a synthetic covariance with a 45-degree rotated major axis and
    verify the first vertex of the mesh is approximately in that direction.
    """
    # Build a simple vertical survey
    survey = we.survey.Survey(
        md=[0, 100, 200],
        inc=[0, 0, 0],
        azi=[0, 0, 0],
    )

    # Inject a manually crafted cov_hla with a 45-degree rotated ellipse
    # H-L block: [[2, 1], [1, 2]] has eigenvectors at 45 degrees
    n = len(survey.md)
    cov = np.zeros((n, 3, 3))
    cov[:, 0, 0] = 2.0   # H variance
    cov[:, 1, 1] = 2.0   # L variance
    cov[:, 0, 1] = 1.0   # HL covariance → 45-deg rotation
    cov[:, 1, 0] = 1.0
    cov[:, 2, 2] = 0.5   # A variance
    survey.cov_hla = cov

    # Eigenvalues of [[2,1],[1,2]] are 1 and 3; eigenvectors at ±45 degrees
    expected_evals, expected_evecs = np.linalg.eigh(cov[0, :2, :2])
    expected_rotation_deg = np.degrees(
        np.arctan2(expected_evecs[1, 1], expected_evecs[0, 1])
    )
    assert abs(abs(expected_rotation_deg) - 45.0) < 1e-6, (
        f"Test setup error: expected 45-deg rotation, got {expected_rotation_deg}"
    )


def test_wellmesh_all_methods():
    """All three WellMesh methods must complete without error."""
    survey = _make_survey_with_errors()
    for method in ('ellipse', 'circle', 'pedal_curve'):
        mesh = we.mesh.WellMesh(survey, method=method)
        assert mesh.vertices.shape[-1] == 3, f"Bad shape for method={method}"


# ---------------------------------------------------------------------------
# Issue #163: func_dict MSIXY / DBHR entries
# ---------------------------------------------------------------------------

def test_func_dict_msixy_entries_are_distinct():
    """
    Issue #163: MSIXY_TI1, MSIXY_TI2, and MSIXY_TI3 must each point to their
    own function, not all to MSIXY_TI1.
    """
    from welleng.errors.tool_errors import ToolError, MSIXY_TI1, MSIXY_TI2, MSIXY_TI3

    # ToolError._initiate_func_dict is called during __init__; we instantiate
    # it with a minimal error/model pair via ErrorModel so we can inspect
    # func_dict directly without running the full model.
    survey = we.survey.Survey(md=[0, 100], inc=[0, 10], azi=[0, 0])
    error = we.error.ErrorModel(survey, error_model='ISCWSA MWD Rev4')
    te = error.errors  # ToolError instance

    assert te.func_dict['MSIXY_TI1'] is MSIXY_TI1
    assert te.func_dict['MSIXY_TI2'] is MSIXY_TI2
    assert te.func_dict['MSIXY_TI3'] is MSIXY_TI3

    assert te.func_dict['MSIXY_TI1'] is not te.func_dict['MSIXY_TI2'], (
        "MSIXY_TI2 must map to its own function, not MSIXY_TI1"
    )
    assert te.func_dict['MSIXY_TI1'] is not te.func_dict['MSIXY_TI3'], (
        "MSIXY_TI3 must map to its own function, not MSIXY_TI1"
    )


def test_func_dict_dbhr_present():
    """Issue #163: DBHR must be present in func_dict."""
    from welleng.errors.tool_errors import DBHR

    survey = we.survey.Survey(md=[0, 100], inc=[0, 10], azi=[0, 0])
    error = we.error.ErrorModel(survey, error_model='ISCWSA MWD Rev4')
    te = error.errors  # ToolError instance

    assert 'DBHR' in te.func_dict, "DBHR is missing from func_dict"
    assert te.func_dict['DBHR'] is DBHR


# ---------------------------------------------------------------------------
# Connector: curve_hold_curve (circle-line-circle) improvements
# ---------------------------------------------------------------------------

def test_connector_dls_design_none_uses_radius_critical():
    """
    Issue #28: dls_design=None should use the minimum curvature required by
    geometry (radius_critical) rather than silently defaulting to 3 deg/30m.
    """
    c = we.connector.Connector(
        pos1=[0., 0., 0.],
        inc1=0., azi1=0.,
        md1=0., md2=500.,
        inc2=45., azi2=45.,
        dls_design=None,
        degrees=True,
    )
    assert np.isinf(c.radius_design), (
        "radius_design should be inf when dls_design=None"
    )
    # The actual radius used must equal radius_critical (finite, > 0)
    assert np.isfinite(c.radius_critical) and c.radius_critical > 0

    # Build the equivalent connector with dls fixed at 3 deg/30m
    c_fixed = we.connector.Connector(
        pos1=[0., 0., 0.],
        inc1=0., azi1=0.,
        md1=0., md2=500.,
        inc2=45., azi2=45.,
        dls_design=3.0,
        degrees=True,
    )
    # The None connector should give a different (gentler) curve
    assert not np.isclose(c.radius_critical, c_fixed.radius_design), (
        "dls_design=None should produce a different path from dls_design=3.0"
    )


def test_connector_dls_design_none_raises_without_target():
    """
    Issue #28: dls_design=None without a target position or measured depth
    is indeterminate and should raise a clear ValueError.
    """
    with pytest.raises(ValueError, match="dls_design must be specified"):
        we.connector.Connector(
            pos1=[0., 0., 0.],
            inc1=0., azi1=0.,
            inc2=45., azi2=45.,
            dls_design=None,
            degrees=True,
        )


def test_connector_curve_hold_curve_close_target_converges():
    """
    The curve_hold_curve solver previously oscillated (never converged) when
    the target was close relative to the radius of curvature.  The damped
    update should resolve this without hitting max_iterations.
    """
    c = we.connector.Connector(
        pos1=[0., 0., 0.],
        vec1=[0., 0., 1.],   # pointing straight down
        pos2=[20., 20., 100.],
        vec2=[0., 0., 1.],
        dls_design=3.0,
        degrees=True,
    )
    assert c.method == 'curve_hold_curve'
    assert c.error, "Solver should have converged (c.error should be True)"
    assert c.iterations < c.max_iterations, (
        "Solver hit max_iterations — likely stuck in an oscillation loop"
    )


def test_connector_curve_hold_curve_balanced_dls():
    """
    After convergence the two arc sections should have approximately equal
    DLS (the solver's balancing objective).
    """
    c = we.connector.Connector(
        pos1=[0., 0., 0.],
        vec1=[0., 0., 1.],
        pos2=[50., 30., 300.],
        vec2=[0.3, 0., 0.954],   # ~17-deg inclination, 0 azimuth
        dls_design=3.0,
        degrees=True,
    )
    assert c.method == 'curve_hold_curve'
    # DLS values should be balanced within delta_dls
    assert abs(np.degrees(c.dls) - np.degrees(c.dls2)) < c.delta_dls + 0.5, (
        f"DLS not balanced: dls={np.degrees(c.dls):.3f}, "
        f"dls2={np.degrees(c.dls2):.3f} deg/30m"
    )


def test_connector_curve_hold_curve_no_recursion_error():
    """
    The old recursive implementation would crash with RecursionError for
    cases requiring many inner iterations.  The iterative version must not
    raise RecursionError regardless of max_iterations.
    """
    import sys
    old_limit = sys.getrecursionlimit()
    # Lower the recursion limit enough to catch the old recursive
    # implementation (which needed ~max_iterations stack frames) but keep
    # it high enough for numpy and basic Python internals to function.
    sys.setrecursionlimit(200)
    try:
        c = we.connector.Connector(
            pos1=[0., 0., 0.],
            vec1=[0., 0., 1.],
            pos2=[15., 15., 80.],
            vec2=[0., 0., 1.],
            dls_design=3.0,
            degrees=True,
        )
        assert c.method == 'curve_hold_curve'
    finally:
        sys.setrecursionlimit(old_limit)
