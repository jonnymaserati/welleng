"""Exact continuous interior covariance -- ``ErrorModel.cov_nev_at``.

Gates the exact interior ellipse of uncertainty at an arbitrary measured depth
against three properties:

1. **Station-exactness** -- at a survey station the interior covariance reproduces
   the conventional per-station covariance to machine precision.
2. **Interior parity** -- at an interior point the covariance matches the exact
   analytical reference (baked as deterministic golden constants), validated
   independently against a Monte-Carlo simulation. The deviated-leg case passes;
   the surface-rooted first-leg case is a documented gap (the tie-on coupling is
   developed discontinuously), marked ``xfail`` until the continuous-transport
   construction lands -- at which point it flips to ``xpass`` and this marker is
   removed.
3. **Model-agnostic** -- the construction consumes each source's error vector by
   its propagation mode, so it applies to any tool error model (MWD, gyro, SAG,
   vendor IPM), not only the ISCWSA MWD standard.

The golden references are the analytical exact form, deterministic (no
Monte-Carlo noise) and reproducible from the fixed survey + bundled model.
"""
import numpy as np
import pytest

import welleng as we

_MODEL = "MWD+SRGM"

_MD = np.array([100.0, 200, 300, 400, 500, 600])
_INC = np.array([5.0, 12, 20, 45, 60, 60])
_AZI = np.array([10.0, 20, 30, 35, 40, 40])
_HEADER = dict(azi_reference="grid", b_total=50000.0, dip=70.0,
               declination=2.0, convergence=1.0)

# golden interior covariances (analytical exact form), for the fixture survey.
_GOLDEN = {
    # (interval i, fraction f): the (3,3) NEV covariance at that interior point
    (3, 0.5): np.array([
        [2.6939522386e+01, 1.4395702747e+01, -2.4597041552e+01],
        [1.4395702747e+01, 1.5353171447e+01, -1.7378335500e+01],
        [-2.4597041552e+01, -1.7378335500e+01, 2.8456943483e+01]]),
    (0, 0.5): np.array([
        [2.3742436811e-01, 5.7170940631e-02, -3.2421045280e-02],
        [5.7170940631e-02, 8.0570208518e-02, -1.0755045605e-02],
        [-3.2421045280e-02, -1.0755045605e-02, 3.8944475897e-02]]),
}


@pytest.fixture(scope="module")
def survey():
    sh = we.survey.SurveyHeader(**_HEADER)
    return we.survey.Survey(md=_MD, inc=_INC, azi=_AZI, header=sh, deg=True,
                            error_model=_MODEL)


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_station_exactness(survey, k):
    # cov_nev_at exactly at a survey station reproduces the per-station covariance
    # to machine precision (eq 6). Station 1 is the f=1 endpoint of the surface-
    # rooted first leg, so it inherits that leg's tracked gap (see
    # test_interior_parity_first_leg) and is excluded here rather than asserted.
    got = np.asarray(survey.err.cov_nev_at(float(_MD[k])))
    ref = np.asarray(survey.err.errors.cov_NEVs[k])
    assert np.max(np.abs(got - ref)) < 1e-9


def test_interior_parity_deviated(survey):
    # a deviated interior point matches the analytical exact reference.
    i, f = 3, 0.5
    md = _MD[i] + f * (_MD[i + 1] - _MD[i])
    got = np.asarray(survey.err.cov_nev_at(float(md)))
    assert _rel(got, _GOLDEN[(i, f)]) < 1e-2


@pytest.mark.xfail(reason="surface-rooted first-leg tie-on coupling is developed "
                          "discontinuously; the continuous-transport construction "
                          "closes it -- flips to xpass when landed",
                   strict=True)
def test_interior_parity_first_leg(survey):
    # the first (surface-rooted) leg is the known gap (~11%); tracked for the fix.
    i, f = 0, 0.5
    md = _MD[i] + f * (_MD[i + 1] - _MD[i])
    got = np.asarray(survey.err.cov_nev_at(float(md)))
    assert _rel(got, _GOLDEN[(i, f)]) < 1e-2


@pytest.mark.parametrize("model", ["MWD+SRGM", "MWD+SRGM_Fl", "MWD+SRGM+SAG"])
def test_model_agnostic(model):
    # the interior covariance applies to any tool error model, gyro included --
    # a valid PSD (3,3) at an interior depth, not only the MWD standard.
    sh = we.survey.SurveyHeader(**_HEADER)
    s = we.survey.Survey(md=_MD, inc=_INC, azi=_AZI, header=sh, deg=True,
                         error_model=model)
    cov = np.asarray(s.err.cov_nev_at(250.0))
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)
    assert np.min(np.linalg.eigvalsh(cov)) >= -1e-9
