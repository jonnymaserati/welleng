"""Symbolic derivation of the closed-form TVD crossing and turning point.

Sawaryn & Thorogood (2005, SPE-84246-PA), *Interpolation at a Plane*
(Eqs. 25-27 + Eq. 1) and *Turning Point* (Eq. 31), specialised to a
horizontal plane, as implemented in ``welleng.utils._arc_tvd_crossings`` and
``welleng.utils._horizontal_tangent_delta``.

A minimum-curvature arc of dogleg ``alpha`` and length ``L`` has radius
``L / alpha``; its unit tangent at subtended angle ``d`` is the SLERP of the end
tangents, so the vertical component is::

    u(d) = (sin(alpha - d) u1 + sin(d) u2) / sin(alpha)

The implementation evaluates the tangent in the Rodrigues ``u``-form
(``welleng.utils._arc_tangent``); the first proof shows the two are the same
vector, so everything derived from the blend holds for the code.

Each test proves one step with SymPy (residual simplifies to zero), then pins
the implemented code against the proven form numerically.
"""

import numpy as np
import pytest

from welleng.utils import (
    MinCurve,
    _arc_tvd_crossings,
    _horizontal_tangent_delta,
)


@pytest.fixture
def sp():
    """SymPy for the proofs only; the numeric pins below run without it."""
    return pytest.importorskip("sympy")


def _setup(sp):
    d, s, alpha, L, u1, u2 = sp.symbols("d s alpha L u1 u2", real=True)
    U = (sp.sin(alpha - d) * u1 + sp.sin(d) * u2) / sp.sin(alpha)
    A = u1 * sp.sin(alpha)
    B = u1 * sp.cos(alpha) - u2
    return d, s, alpha, L, u1, u2, U, A, B


def test_rodrigues_u_form_equals_the_slerp_blend(sp):
    """cos(d) v1 + sin(d) (v2 - cos(alpha) v1) / sin(alpha) == SLERP blend."""
    d, alpha = sp.symbols("d alpha", real=True)
    v1 = sp.Matrix(sp.symbols("v1x v1y v1z", real=True))
    v2 = sp.Matrix(sp.symbols("v2x v2y v2z", real=True))
    u = (v2 - sp.cos(alpha) * v1) / sp.sin(alpha)
    rodrigues = sp.cos(d) * v1 + sp.sin(d) * u
    slerp = (sp.sin(alpha - d) * v1 + sp.sin(d) * v2) / sp.sin(alpha)
    diff = (rodrigues - slerp).applyfunc(lambda e: sp.simplify(sp.expand_trig(e)))
    assert diff == sp.zeros(3, 1)


def test_vertical_travel_reduces_to_a_sin_plus_b_cos(sp):
    """z(d) * alpha sin(alpha) / L == a sin d + b cos d - b."""
    d, s, alpha, L, u1, u2, U, A, B = _setup(sp)
    dz = (L / alpha) * sp.integrate(U.subs(d, s), (s, 0, d))
    lhs = dz * alpha * sp.sin(alpha) / L
    rhs = A * sp.sin(d) + B * sp.cos(d) - B
    assert sp.simplify(sp.expand_trig(lhs - rhs)) == 0


def test_half_angle_roots_solve_the_crossing_equation(sp):
    """d = 2 atan((a +/- sqrt(a^2 + b^2 - c^2)) / (b + c)) solves a sin + b cos = c.

    With t = tan(d/2) the equation is the quadratic (b+c) t^2 - 2 a t + (c-b) = 0;
    both roots are proven.
    """
    a, b, c, t = sp.symbols("a b c t", real=True)
    quad = (b + c) * t**2 - 2 * a * t + (c - b)
    for sign in (1, -1):
        root = (a + sign * sp.sqrt(a**2 + b**2 - c**2)) / (b + c)
        assert sp.simplify(sp.expand(quad.subs(t, root))) == 0
    # and the quadratic IS the crossing equation under the half-angle map
    sin_d = 2 * t / (1 + t**2)
    cos_d = (1 - t**2) / (1 + t**2)
    assert sp.simplify((a * sin_d + b * cos_d - c) * (1 + t**2) + quad) == 0


def test_turning_point_zeroes_the_vertical_tangent(sp):
    """tan(d) = -sin(alpha) u1 / (u2 - cos(alpha) u1) makes u(d) == 0."""
    d, _, alpha, _, u1, u2, *_ = _setup(sp)
    d_tp = sp.atan2(-sp.sin(alpha) * u1, u2 - sp.cos(alpha) * u1)
    num = sp.sin(alpha - d) * u1 + sp.sin(d) * u2
    expanded = sp.expand_trig(num)
    # cos/sin of atan2(y, x) are x/r and y/r; substitute and clear r
    r = sp.sqrt((sp.sin(alpha) * u1) ** 2 + (u2 - sp.cos(alpha) * u1) ** 2)
    sub = expanded.subs({sp.cos(d): (u2 - sp.cos(alpha) * u1) / r,
                         sp.sin(d): (-sp.sin(alpha) * u1) / r})
    assert sp.simplify(sub * r) == 0
    assert d_tp is not None


# -- the implemented code, pinned against the proven forms --------------------

def _arcs(n=300, seed=11):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        inc = rng.uniform(0, np.pi, 2)
        azi = rng.uniform(0, 2 * np.pi, 2)
        mc = MinCurve(np.array([0.0, rng.uniform(5, 300)]), inc, azi)
        if mc.dogleg[1] > 1e-6:
            yield mc


def test_implemented_crossings_land_on_the_target_tvd():
    """Every root the code returns reproduces the target TVD on the arc."""
    checked = 0
    for mc in _arcs():
        alpha_, dmd = mc.dogleg[1], mc.delta_md[1]
        c1, c2 = np.cos(mc.inc)
        z_end = mc.poss[1, 2]
        for frac in (0.25, 0.5, 0.75):
            target = frac * z_end
            for dd in _arc_tvd_crossings(c1, c2, alpha_, dmd, target):
                z = mc.interpolate(dd / alpha_ * dmd)[2]
                assert z == pytest.approx(target, abs=1e-7)
                checked += 1
    assert checked > 300


def test_implemented_turning_point_is_horizontal():
    """Where the code reports a turning point, the inclination there is 90 deg."""
    checked = 0
    for mc in _arcs():
        alpha_, dmd = mc.dogleg[1], mc.delta_md[1]
        c1, c2 = np.cos(mc.inc)
        dd = _horizontal_tangent_delta(c1, c2, alpha_)
        if dd is None:
            continue
        inc_at, _ = mc.inc_azi_at(dd / alpha_ * dmd)
        assert inc_at == pytest.approx(np.pi / 2, abs=1e-9)
        checked += 1
    assert checked > 20
