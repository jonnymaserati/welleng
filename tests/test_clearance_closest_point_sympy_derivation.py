"""Symbolic proof behind the closed-form closest point on a minimum-curvature
arc (``welleng.clearance._closest_x_on_arc``) and the shared-station rule
that relies on it (``Clearance._closest_offset_point``).

The arc is ``p(th) = C + R (sin th * a - cos th * b)`` with ``a``, ``b`` the
in-plane unit vectors and ``n`` the plane normal. Writing
``Q - C = qa a + qb b + qn n``:

1. ``|p(th) - Q|^2 = R^2 + qa^2 + qb^2 + qn^2 - 2 R f(th)`` with
   ``f(th) = qa sin th - qb cos th`` -- distance falls as ``f`` rises, and the
   out-of-plane ``qn`` does not move the minimiser.
2. ``f(th) = A cos(th - th*)`` with ``A = sqrt(qa^2 + qb^2)`` and
   ``th* = atan2(qa, -qb)``: ``f <= A`` for every ``th``, equal only at
   ``th*`` (mod 2 pi). So an interior ``th*`` is the GLOBAL closest point on
   the whole circle, hence on the arc -- no endpoint of the arc is closer.
   That is what lets a leg's interior closest point outrank the station it
   shares with a neighbouring leg.
3. ``f`` has no other local maximum, so when ``th*`` lies off the arc the
   closest point is an endpoint. A minimum-curvature dogleg is in
   ``[0, pi]`` and ``atan2`` returns ``th*`` in ``(-pi, pi]``, so
   ``th* + 2 pi k`` can fall on the arc only for ``k = 0`` -- the single
   check ``0 <= th* <= dogleg`` is complete.
"""
import numpy as np
import sympy as sp

from welleng.clearance import _closest_x_on_arc

th, R = sp.symbols("theta R", positive=True)
qa, qb, qn = sp.symbols("q_a q_b q_n", real=True)
A = sp.sqrt(qa ** 2 + qb ** 2)
f = qa * sp.sin(th) - qb * sp.cos(th)


def test_distance_is_affine_in_f():
    p = sp.Matrix([R * sp.sin(th), -R * sp.cos(th), 0])     # p - C in (a, b, n)
    q = sp.Matrix([qa, qb, qn])                              # Q - C
    d2 = (p - q).dot(p - q)
    assert sp.simplify(d2 - (R ** 2 + qa ** 2 + qb ** 2 + qn ** 2 - 2 * R * f)) == 0


def test_f_is_a_single_cosine_peaking_at_theta_star():
    # sin(th*) = qa / A, cos(th*) = -qb / A  <=>  th* = atan2(qa, -qb)
    s_star, c_star = qa / A, -qb / A
    cos_shift = sp.cos(th) * c_star + sp.sin(th) * s_star    # cos(th - th*)
    assert sp.simplify(f - A * cos_shift) == 0
    # A^2 - f^2 is a perfect square -> f <= A for every theta
    assert sp.trigsimp(sp.expand(
        A ** 2 - f ** 2 - (qa * sp.cos(th) + qb * sp.sin(th)) ** 2)) == 0
    # and the bound is attained at th*
    assert sp.simplify(qa * s_star - qb * c_star - A) == 0


def test_interior_closest_point_beats_both_arc_ends():
    """Numerical consequence on real arcs: an interior closed-form point is no
    further than either end of its arc (the property the shared-station rule
    uses). Random arcs with dogleg up to pi and points all around them."""
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(4000):
        dogleg = rng.uniform(1e-3, np.pi - 1e-6)
        dmd = rng.uniform(1.0, 100.0)
        t0 = rng.normal(size=3)
        t0 /= np.linalg.norm(t0)
        v = rng.normal(size=3)
        v -= v @ t0 * t0
        v /= np.linalg.norm(v)
        t1 = np.cos(dogleg) * t0 + np.sin(dogleg) * v
        P0 = rng.normal(size=3) * 10
        Q = P0 + rng.normal(size=3) * dmd
        Rad = dmd / dogleg
        C = P0 + Rad * v

        def pos(x):
            a = x / Rad
            return C + Rad * (np.sin(a) * t0 - np.cos(a) * v)

        x = _closest_x_on_arc(P0, t0, t1, dmd, dogleg, Q)
        if 0.0 < x < dmd:
            checked += 1
            d = np.linalg.norm(pos(x) - Q)
            assert d <= np.linalg.norm(pos(0.0) - Q) + 1e-9
            assert d <= np.linalg.norm(pos(dmd) - Q) + 1e-9
    assert checked > 500
