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


# Reconditioned form. With d = Q - P0, da = d.a, db = d.b and kappa = dogleg / L:
# qa = da and -qb = R - db, so th* = atan2(da, R - db) = atan2(da kappa, 1 - db kappa)
# (atan2 is unchanged by a positive common factor). For a near-straight segment R is
# huge and Q - C loses every digit of the offset to cancellation against R b; the
# scaled form keeps them. The endpoint test multiplies f(dogleg) - f(0) by kappa too.
d_a, d_b, kap, dl = sp.symbols("d_a d_b kappa delta", real=True)


def test_reconditioned_theta_star_and_endpoint_rule():
    Rk = 1 / kap
    q_a, q_b = d_a, d_b - Rk                                 # Q - C in (a, b)
    # same atan2 direction: (sin, cos) of th* proportional to (q_a, -q_b) and to the
    # scaled pair, with the positive factor kappa
    assert sp.simplify(d_a * kap - kap * q_a) == 0
    assert sp.simplify((1 - d_b * kap) - kap * (-q_b)) == 0
    # f(dogleg) - f(0) = kappa^-1 [d_a kappa sin dl - (1 - d_b kappa) 2 sin^2(dl/2)]
    fd = q_a * sp.sin(dl) - q_b * sp.cos(dl)
    f0 = -q_b
    rhs = (d_a * kap * sp.sin(dl) - (1 - d_b * kap) * 2 * sp.sin(dl / 2) ** 2) / kap
    assert sp.simplify(sp.expand_trig(fd - f0 - rhs)) == 0


def _mp_closest_distance(P0, t0, t1, L, dogleg, Q, x):
    """Distance from Q to the arc point at offset x, and the true minimum over the
    arc, both in 50-digit arithmetic from the same float inputs."""
    import mpmath as mp
    mp.mp.dps = 50
    v = lambda u: mp.matrix([mp.mpf(float(c)) for c in u])            # noqa: E731
    dot = lambda p, q: sum(p[i] * q[i] for i in range(3))             # noqa: E731
    a, t1m, P, Qm = v(t0), v(t1), v(P0), v(Q)
    b = t1m - dot(t1m, a) * a
    b = b / mp.sqrt(dot(b, b))
    dg, Lm = mp.mpf(float(dogleg)), mp.mpf(float(L))
    R = Lm / dg

    def dist(xx):
        th = mp.mpf(xx) / R
        p = P + R * b + R * (mp.sin(th) * a - mp.cos(th) * b)
        return mp.sqrt(dot(p - Qm, p - Qm))

    d = Qm - P
    th = mp.atan2(dot(d, a), R - dot(d, b))
    cands = [mp.mpf(0), Lm] + ([th * R] if 0 <= th <= dg else [])
    return float(dist(x)), float(min(dist(c) for c in cands))


def test_near_straight_arcs_find_the_true_closest_point():
    """Doglegs from 1e-9 rad to pi: the closed-form point is the closest point on
    the arc to within 1e-9 m, against the same formula in 50-digit arithmetic."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for lo in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-3, 1e-1):
        for _ in range(150):
            dogleg = min(lo * rng.uniform(1, 10), np.pi - 1e-6)
            t0 = rng.normal(size=3)
            t0 /= np.linalg.norm(t0)
            v = np.cross(t0, rng.normal(size=3))
            v /= np.linalg.norm(v)
            t1 = np.cos(dogleg) * t0 + np.sin(dogleg) * v
            L = rng.uniform(1.0, 500.0)
            P0 = rng.normal(size=3) * 100
            Q = P0 + rng.normal(size=3) * 300
            x = _closest_x_on_arc(P0, t0, t1, L, dogleg, Q)
            assert 0.0 <= x <= L
            got, best = _mp_closest_distance(P0, t0, t1, L, dogleg, Q, x)
            worst = max(worst, got - best)
    assert worst < 1e-9, worst
