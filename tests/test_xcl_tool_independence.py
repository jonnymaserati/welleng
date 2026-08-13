"""XCLA/XCLH (extended course length, Codling SPE-187249) are GEOMETRIC,
tool-independent terms: the survey-interval position error depends on the
wellbore path, not the survey instrument. So on the SAME well they must give the
SAME covariance across MWD and gyro tool models.

Regression for the 0.25.0 item-1 fix: gyro models previously routed XCLA/XCLH
through the pointwise JSON interpreter (which can't express the cross-station
Max(Δangle, tortuosity·course-length) recurrence -> a different, unvalidated
form), while MWD used the hand-coded weight functions validated against the
ISCWSA diagnostics. Both now route through the hand-coded path.
"""
import numpy as np

import welleng as we


def _cov(model, term):
    md = np.linspace(0, 3000, 40)
    inc = np.linspace(2, 72, 40)
    azi = np.linspace(10, 88, 40)
    s = we.survey.Survey(
        md=md, inc=inc, azi=azi, deg=True,
        header=we.survey.SurveyHeader(
            b_total=50000, dip=70, declination=0, azi_reference="grid"),
        error_model=model,
    )
    return np.asarray(s.err.errors.errors[term].cov_NEV)


MODELS = ["MWD+SRGM", "GYRO-NS", "GYRO-MWD"]


def test_xcla_is_tool_independent():
    ref = _cov("MWD+SRGM", "XCLA")
    for m in MODELS[1:]:
        assert np.allclose(_cov(m, "XCLA"), ref, atol=1e-12), (
            f"XCLA cov differs between MWD+SRGM and {m} -- XCLA is geometric and "
            f"must be tool-independent"
        )


def test_xclh_is_tool_independent():
    ref = _cov("MWD+SRGM", "XCLH")
    for m in MODELS[1:]:
        assert np.allclose(_cov(m, "XCLH"), ref, atol=1e-12)


def test_xcla_is_the_validated_position_form():
    # The hand-coded form puts weight in the depth/inclination position columns
    # (not the interpreter's azimuth-only form). Guard that gyro didn't revert to
    # the interpreter form (which had e_DIA ~ [0, 0, azi]).
    s = we.survey.Survey(
        md=np.linspace(0, 3000, 40), inc=np.linspace(2, 72, 40),
        azi=np.linspace(10, 88, 40), deg=True,
        header=we.survey.SurveyHeader(
            b_total=50000, dip=70, declination=0, azi_reference="grid"),
        error_model="GYRO-NS",
    )
    e = np.asarray(s.err.errors.errors["XCLA"].e_DIA)[-1]
    assert abs(e[0]) > 1e-6, "gyro XCLA reverted to the interpreter (no depth weight)"


def test_xcl_dia_carries_no_first_station_doubling():
    """XCL is an INTERVAL error (station k depends only on the interval k-1..k), so the
    own-only DIA form carries NO station-0 contribution and NO first-station surface-
    tie-on doubling (Def. of ISCWSA Error Model §4.7.1.1, eq. 32) -- matching the
    released position-direct XCL, which carries none. (Refinement noted by T. Allen,
    TALLENTECH, 2026.) Locks the property against a future regression that routed XCL
    through the shared tie-on-doubled Jacobian.
    """
    h = we.survey.SurveyHeader(
        b_total=50000, dip=70, declination=0, azi_reference="grid")
    h.xcl_representation = "dia"
    s = we.survey.Survey(
        md=np.linspace(0, 3000, 40), inc=np.linspace(2, 72, 40),
        azi=np.linspace(10, 88, 40), deg=True, header=h, error_model="MWD+SRGM")
    for term in ("XCLH", "XCLA"):
        e_DIA = np.asarray(s.err.errors.errors[term].e_DIA)
        assert np.all(e_DIA[0] == 0.0), (
            f"{term} own-only DIA has a non-zero station-0 perturbation -- an interval "
            f"error must carry no first-station contribution or tie-on doubling")
