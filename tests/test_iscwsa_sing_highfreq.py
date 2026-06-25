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

import warnings

import numpy as np
import pytest

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


# --------------------------------------------------------------------------
# Broader SING-branch sweep across both MWD models
# --------------------------------------------------------------------------

def _err(md, inc, azi, model):
    """Build an Error for an arbitrary geometry with magnetic params set, so
    field-dependent terms (ABXY/MBXY/...) evaluate on non-vertical wells."""
    sh = SurveyHeader()
    sh.error_model = model
    sh.latitude = 60.0
    sh.b_total = 50000.0
    sh.dip = 72.0
    sh.declination = -4.0
    md = np.asarray(md, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Survey(
            md=md, inc=np.asarray(inc, float), azi=np.asarray(azi, float),
            header=sh, error_model=model,
        ).err


@pytest.mark.parametrize("model, code, axis, damped", [
    ("ISCWSA MWD Rev4",    "XYM3",  0, False),
    ("ISCWSA MWD Rev4",    "XYM4",  1, False),
    ("ISCWSA MWD Rev5.11", "XYM3E", 0, True),
    ("ISCWSA MWD Rev5.11", "XYM4E", 1, True),
])
def test_misalignment_damping_only_for_rev5(model, code, axis, damped):
    """Per Rev5.13 §4.6 the damping term M is applied to XYM3E/XYM4E (Rev5,
    random) but NOT to XYM3/XYM4 (Rev4, systematic). On a vertical well,
    halving the course length (ΔMD 10→5 m ⇒ M: 1→√2) must scale a mid SING
    station's e* by pure geometry (×0.5) for the undamped Rev4 terms and by
    geometry×M (×0.5√2) for the damped Rev5 terms. Guards both the #203 bug
    (M missing from Rev5 e*) and its inverse (M wrongly added to Rev4)."""
    z4 = np.zeros(4)
    reg = _err(np.arange(0, 31, 10.0), z4, z4, model)   # ΔMD=10 ⇒ M=1
    hf = _err(np.arange(0, 16, 5.0), z4, z4, model)     # ΔMD=5  ⇒ M=√2
    es_reg = np.asarray(reg.errors.errors[code].e_NEV_star)[2, axis]
    es_hf = np.asarray(hf.errors.errors[code].e_NEV_star)[2, axis]
    ratio = es_hf / es_reg
    expected = 0.5 * np.sqrt(2) if damped else 0.5
    assert np.isclose(ratio, expected, rtol=1e-6), (
        f"{model}/{code}: e* hf/reg ratio {ratio:.5f}, expected {expected:.5f}"
    )


def test_xym3e_xym4e_vertical_symmetry():
    """On a vertical, due-north well the singular weight functions are [M,0,0]
    (XYM3E) and [0,M,0] (XYM4E) per Rev5.13 §11.5, so XYM3E's NN covariance
    must equal XYM4E's EE covariance station-for-station."""
    md = np.arange(0, 121, 30.0)
    err = _err(md, np.zeros_like(md), np.zeros_like(md), "ISCWSA MWD Rev5.11")
    nn = np.asarray(err.errors.errors["XYM3E"].cov_NEV)[:, 0, 0]
    ee = np.asarray(err.errors.errors["XYM4E"].cov_NEV)[:, 1, 1]
    np.testing.assert_allclose(nn, ee, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("model", ["ISCWSA MWD Rev4", "ISCWSA MWD Rev5.11"])
@pytest.mark.parametrize("geometry", ["vertical", "highfreq_vertical", "horizontal"])
def test_all_sources_finite_on_singular_geometries(model, geometry):
    """No error source may leak NaN/inf covariance on geometries that exercise
    the singular (vertical) or 1/sin(inc) (near-horizontal) branches. A blanket
    regression net for the whole weight-function table, not just the misalignment
    family."""
    if geometry == "vertical":
        md = np.arange(0, 301, 30.0)
        inc = np.zeros_like(md)
    elif geometry == "highfreq_vertical":
        md = np.arange(0, 51, 5.0)
        inc = np.zeros_like(md)
    else:  # near-horizontal: exercises 1/sin(inc) without an exact-90° cot blowup
        md = np.arange(0, 301, 30.0)
        inc = np.full_like(md, 89.0)
        inc[0] = 0.0
    err = _err(md, inc, np.zeros_like(md), model)
    bad = sorted(
        s for s, e in err.errors.errors.items()
        if not np.all(np.isfinite(np.asarray(e.cov_NEV)))
    )
    assert not bad, f"{model}/{geometry}: non-finite covariance in {bad}"
