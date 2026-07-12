"""Geometric torsion + curvature-rate on Survey (SPE-105068-PA Eq. 17/18)."""
import numpy as np

import welleng as we


def _survey(md, inc, azi):
    return we.survey.Survey(
        md=np.array(md, float), inc=np.array(inc, float), azi=np.array(azi, float),
        header=we.survey.SurveyHeader(),
    )


def test_torsion_zero_on_planar():
    # a 2D (constant-azimuth) build has zero geometric torsion
    md = np.arange(0, 2001, 100.0)
    s = _survey(md, np.linspace(0, 80, len(md)), np.full_like(md, 45.0))
    tau = s.torsion()
    assert tau.shape == md.shape
    assert np.nanmax(np.abs(tau)) < 1e-6


def test_torsion_constant_on_small_circle():
    # constant inclination + constant azimuth rate = a curve of constant curvature AND
    # torsion -> torsion must be constant across the interior (the strong signature)
    md = np.arange(0, 2001, 100.0)
    s = _survey(md, np.full_like(md, 45.0), np.linspace(0, 180, len(md)))
    tau = s.torsion()[1:-1]
    assert tau.mean() > 0
    assert tau.std() / tau.mean() < 1e-9          # constant to ~machine precision


def test_torsion_validates_helix_closed_form():
    # VALIDATION (not just consistency): a cylinder helix (constant inc + linear azimuth) has a
    # closed-form torsion. With tau = h/(a^2+h^2), kappa = a/(a^2+h^2), and h/a = cot(inc), the
    # exact relation is tau = kappa * cot(inc). Checks the VALUE against the closed form.
    md = np.arange(0, 4001, 50.0)
    for inc0 in (30.0, 45.0, 60.0):
        s = _survey(md, np.full_like(md, inc0), np.linspace(0, 360, len(md)))
        tau = s.torsion()
        dmd = np.diff(md)
        kappa = np.zeros_like(md)
        kappa[1:] = np.where(dmd > 0, s.dogleg[1:] / dmd, 0.0)   # curvature = dogleg / interval
        interior = slice(3, -3)
        ratio = float((tau[interior] / kappa[interior]).mean())
        expected = 1.0 / np.tan(np.radians(inc0))               # cot(inc)
        assert abs(ratio - expected) / expected < 2e-3          # min-curve discretization ~4e-4


def test_curvature_rate_zero_on_constant_dls():
    # a uniform build (constant DLS) has dκ/ds ~ 0 in its interior (excluding the KOP
    # onset at station 0, a real curvature discontinuity on a min-curve survey)
    md = np.arange(0, 2001, 100.0)
    s = _survey(md, np.linspace(0, 60, len(md)), np.full_like(md, 30.0))
    cr = s.curvature_rate()
    assert cr.shape == md.shape
    assert np.nanmax(np.abs(cr[2:-1])) < 1e-9      # flat interior


def test_curvature_rate_spikes_at_dls_transition():
    # build then hold -> curvature drops to zero at the transition -> a dκ/ds spike there
    md = np.arange(0, 2001, 100.0)
    inc = np.concatenate([np.linspace(0, 60, 11), np.full(10, 60.0)])
    s = _survey(md, inc, np.full_like(md, 30.0))
    cr = np.abs(s.curvature_rate())
    # the build->hold transition is around station 10; it must carry a nonzero rate
    assert cr[9:12].max() > 1e-6
