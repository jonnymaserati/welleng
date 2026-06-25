"""Regression guard for the vertical-singular branch of XYM3E / XYM4E on
high-frequency surveys (issue #203).

The ISCWSA example workbooks all sample at ΔMD ≥ Lmin = 10 m, so the
misalignment damping term ``M = max(1, sqrt(Lmin/ΔMD))`` is 1 everywhere and
the SING-branch ``e*`` formula is never exercised with M > 1. These tests
build a synthetic vertical well at ΔMD = 5 m (⇒ M = √2) to cover that path.

Per *Definition of ISCWSA Error Model Rev5.13*:
  - §11.5 rows 11V/12V: VertWftFn = [M, 0, 0] (XYM3E) / [0, M, 0] (XYM4E)
  - eq (20):  e_k   = σ · (D_{k+1} − D_{k−1})/2 · VertWftFn
  - eq (21):  e*_k  = σ · (D_k     − D_{k−1})/2 · VertWftFn
  - eq (33):  e_1   = σ · (D_2 + D_1 − 2·D_0)/2 · VertWftFn  (first station)
  - eq (34):  e*_1  = σ · (D_1 − D_0)          · VertWftFn  (full first interval)

i.e. M multiplies BOTH e and e*. The bug fixed in #203 was that the SING
branch applied M to ``e`` but dropped it from ``e*`` (both the general line and
the first-station override), so on high-frequency wells e* was short by a
factor of M.
"""

import numpy as np

from welleng.survey import Survey, SurveyHeader

MODEL = "ISCWSA MWD Rev5.11"
DMD = 5.0          # course length, m — below Lmin=10 so M > 1
LMIN = 10.0
N = 6


def _vertical_highfreq_err(dmd=DMD, n=N, model=MODEL):
    md = np.arange(n) * dmd
    sh = SurveyHeader()
    sh.error_model = model
    survey = Survey(
        md=md, inc=np.zeros(n), azi=np.zeros(n),
        header=sh, error_model=model,
    )
    return survey.err, md


def _expected_e_estar(md, M, mag):
    """Spec eqs (20)/(21)/(33)/(34) along the active singular axis."""
    e = np.zeros(len(md))
    estar = np.zeros(len(md))
    # e: centred pair, with first/last station overrides
    e[1:-1] = M * (md[2:] - md[:-2]) / 2 * mag
    e[1] = M * (md[2] + md[1] - 2 * md[0]) / 2 * mag       # eq (33)
    e[-1] = M * (md[-1] - md[-2]) / 2 * mag
    # e*: half back-interval, full interval at first station
    estar[1:] = M * (md[1:] - md[:-1]) / 2 * mag           # eq (21)
    estar[1] = M * (md[1] - md[0]) * mag                   # eq (34)
    return e, estar


def test_xym3e_xym4e_damping_applies_to_estar():
    err, md = _vertical_highfreq_err()
    M = max(1.0, np.sqrt(LMIN / DMD))
    assert M > 1.0  # sanity: this well actually exercises the damping path

    # Recover the per-source 1-sigma magnitude from XYM3E's diagonal e_DIA:
    # at inc = 0 the inc-axis weight reduces to coeff, so e_DIA[1] = coeff[0]·mag
    # and coeff[0] = M for the first interval.
    xym3 = err.errors.errors["XYM3E"]
    mag = xym3.e_DIA[1, 1] / M

    e_exp, estar_exp = _expected_e_estar(md, M, mag)

    # XYM3E is singular on the North axis (col 0); XYM4E on East (col 1).
    np.testing.assert_allclose(
        np.asarray(xym3.e_NEV)[:, 0], e_exp, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(xym3.e_NEV_star)[:, 0], estar_exp, rtol=1e-12, atol=1e-14
    )

    xym4 = err.errors.errors["XYM4E"]
    np.testing.assert_allclose(
        np.asarray(xym4.e_NEV)[:, 1], e_exp, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(xym4.e_NEV_star)[:, 1], estar_exp, rtol=1e-12, atol=1e-14
    )


def test_estar_is_not_the_undamped_value():
    """Explicitly pin the #203 failure mode: a mid SING station's e* must carry
    M, i.e. it must NOT equal the undamped σ·(Dk−Dk-1)/2."""
    err, md = _vertical_highfreq_err()
    M = max(1.0, np.sqrt(LMIN / DMD))
    xym3 = err.errors.errors["XYM3E"]
    mag = xym3.e_DIA[1, 1] / M

    estar2 = np.asarray(xym3.e_NEV_star)[2, 0]
    damped = M * (md[2] - md[1]) / 2 * mag      # correct (eq 21)
    undamped = (md[2] - md[1]) / 2 * mag        # the old, buggy value

    assert np.isclose(estar2, damped, rtol=1e-12)
    assert not np.isclose(estar2, undamped, rtol=1e-6)


def test_highfreq_sing_covariance_matches_propagation():
    """cov_NN[k] = e*_k² + Σ_{i<k} e_i² (welleng's random-propagation split)."""
    err, md = _vertical_highfreq_err()
    M = max(1.0, np.sqrt(LMIN / DMD))
    xym3 = err.errors.errors["XYM3E"]
    mag = xym3.e_DIA[1, 1] / M

    e_exp, estar_exp = _expected_e_estar(md, M, mag)
    cov_exp = estar_exp ** 2 + np.concatenate(([0.0], np.cumsum(e_exp ** 2)[:-1]))

    np.testing.assert_allclose(
        np.asarray(xym3.cov_NEV)[:, 0, 0], cov_exp, rtol=1e-12, atol=1e-14
    )
