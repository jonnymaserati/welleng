"""NEV covariance is in the axes of the survey's positions (grid north).

Oracle: frame invariance. Well A has grid convergence g and grid azimuths a;
well B has convergence 0 and azimuths a + g. Both have the same true
azimuths, so they are one physical well with one error model: B's NEV axes
are true north, A's are grid north. A's positions and every NEV covariance it
outputs must be B's turned by -g about vertical, and its HLA covariance,
which has no north, must equal B's.
"""
import numpy as np
import pytest

import welleng as we

G_DEG = 2.0
MD = np.array([0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0])
INC = np.array([1.0, 15.0, 35.0, 60.0, 75.0, 75.0])
AZI = np.array([20.0, 25.0, 40.0, 70.0, 95.0, 95.0])


def _survey(convergence, azi, model, inc=INC):
    h = we.survey.SurveyHeader(deg=True, b_total=50000.0, dip=70.0,
                               declination=1.0, convergence=convergence,
                               azi_reference="grid")
    return we.survey.Survey(md=MD, inc=inc, azi=azi, header=h,
                            error_model=model)


def _rot():
    g = np.radians(G_DEG)
    return np.array([[np.cos(g), np.sin(g), 0.0],
                     [-np.sin(g), np.cos(g), 0.0],
                     [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("inc0", [1.0, 0.0])   # 0: a vertical station, singular weights
@pytest.mark.parametrize("model", ["ISCWSA MWD Rev4", "ISCWSA MWD Rev5.11",
                                   "MWD+SRGM"])
def test_covariance_turns_with_the_positions(model, inc0):
    inc = np.r_[inc0, INC[1:]]
    a = _survey(G_DEG, AZI, model, inc)
    b = _survey(0.0, AZI + G_DEG, model, inc)
    R = _rot()
    pa = np.column_stack([a.n, a.e, a.tvd])
    pb = np.column_stack([b.n, b.e, b.tvd])
    np.testing.assert_allclose(pa, pb @ R.T, atol=1e-9)
    scale = np.abs(b.cov_nev).max()
    for name in ("cov_nev", "cov_nev_random", "cov_nev_systematic",
                 "cov_nev_global"):
        np.testing.assert_allclose(
            getattr(a, name), R @ getattr(b, name) @ R.T,
            atol=1e-12 * scale, err_msg=name)
    np.testing.assert_allclose(a.cov_hla, b.cov_hla, atol=1e-12 * scale)
    # a vertical station stores grid azimuth 0, so its true azimuth is 0 + g
    # in A and 0 in B: the two are one well only beyond the leg leaving it
    for md in (150.0, 750.0, 1350.0) if inc0 else (750.0, 1350.0):
        np.testing.assert_allclose(
            a.err.cov_nev_at(md), R @ b.err.cov_nev_at(md) @ R.T,
            atol=1e-12 * scale)
    for (sa, sb) in zip(a.err.errors.errors.values(), b.err.errors.errors.values()):
        np.testing.assert_allclose(sa.e_NEV, sb.e_NEV @ R.T,
                                   atol=1e-12 * np.sqrt(scale))
        np.testing.assert_allclose(sa.cov_NEV, R @ sb.cov_NEV @ R.T,
                                   atol=1e-12 * scale)


def test_separation_factor_does_not_depend_on_the_frame():
    """Two wells described with grid convergence give the same separation
    factor as the same wells described in true north."""
    off_azi = np.array([200.0, 205.0, 210.0, 230.0, 260.0, 260.0])
    ref_a, ref_b = (_survey(G_DEG, AZI, "ISCWSA MWD Rev4"),
                    _survey(0.0, AZI + G_DEG, "ISCWSA MWD Rev4"))

    def off(conv, azi):
        h = we.survey.SurveyHeader(deg=True, b_total=50000.0, dip=70.0,
                                   declination=1.0, convergence=conv,
                                   azi_reference="grid")
        return we.survey.Survey(md=MD, inc=INC, azi=azi, header=h,
                                start_nev=[150.0, 300.0, 0.0] if conv == 0.0
                                else list(_rot() @ [150.0, 300.0, 0.0]),
                                error_model="ISCWSA MWD Rev4")

    sa = we.clearance.MahalanobisClearance(ref_a, off(G_DEG, off_azi)).min_sf
    sb = we.clearance.MahalanobisClearance(ref_b, off(0.0, off_azi + G_DEG)).min_sf
    assert sa == pytest.approx(sb, rel=1e-6)
    iscwsa = we.clearance.IscwsaClearance
    ia = np.ravel(iscwsa(ref_a, off(G_DEG, off_azi)).sf).min()
    ib = np.ravel(iscwsa(ref_b, off(0.0, off_azi + G_DEG)).sf).min()
    assert ia == pytest.approx(ib, rel=1e-6)
