"""Symbolic proofs behind ``MahalanobisClearance``.

1. ``_quad_form_inv``: the linear-solve path ``d' S^-1 d`` and the eigen path
   ``sum_i (v_i' d)^2 / lambda_i`` are the same quantity, because
   ``S = V diag(lambda) V'`` with ``V`` orthogonal gives
   ``S^-1 = V diag(1/lambda) V'``. Proved for a general rotation (Euler-Rodrigues
   from an unnormalised quaternion) and general positive eigenvalues.

2. Mahalanobis separation factor >= pedal (ISCWSA) separation factor, with the
   same floored covariance ``S = C_ref + C_off + sigma_pa^2 I`` in both. With
   ``dp = (D - R) u`` for unit ``u`` the two are
   ``(D - R) sqrt(u' S^-1 u) / k`` and ``(D - R) / (k sqrt(u' S u))``, so the
   claim is ``(u' S u)(u' S^-1 u) >= 1``. In S's eigenbasis, with ``w`` the
   components of ``u``:
   ``(sum l_i w_i^2)(sum w_i^2 / l_i) - (sum w_i^2)^2
   = sum_{i<j} w_i^2 w_j^2 (l_i - l_j)^2 / (l_i l_j) >= 0`` -- equality only
   when ``u`` lies in one eigenspace (a spherical ellipsoid, where the two
   rules coincide).
"""
import itertools

import sympy as sp


def test_solve_and_eigen_quadratic_forms_are_identical():
    a, b, c, e = sp.symbols("a b c e", real=True)
    n2 = a ** 2 + b ** 2 + c ** 2 + e ** 2
    # rotation from an unnormalised quaternion (a, b, c, e): orthogonal for any
    # non-zero quaternion
    V = sp.Matrix([
        [a*a + b*b - c*c - e*e, 2*(b*c - a*e), 2*(b*e + a*c)],
        [2*(b*c + a*e), a*a - b*b + c*c - e*e, 2*(c*e - a*b)],
        [2*(b*e - a*c), 2*(c*e + a*b), a*a - b*b - c*c + e*e],
    ]) / n2
    assert sp.simplify(V.T * V - sp.eye(3)) == sp.zeros(3)
    lam = sp.symbols("l1:4", positive=True)
    d = sp.Matrix(sp.symbols("d1:4", real=True))
    S = V * sp.diag(*lam) * V.T
    S_inv_eig = V * sp.diag(*[1 / x for x in lam]) * V.T
    assert sp.simplify(S * S_inv_eig - sp.eye(3)) == sp.zeros(3)
    proj = V.T * d
    eig_form = sum(proj[i] ** 2 / lam[i] for i in range(3))
    assert sp.simplify((d.T * S_inv_eig * d)[0] - eig_form) == 0


def test_mahalanobis_is_never_below_pedal():
    lam = sp.symbols("l1:4", positive=True)
    w = sp.symbols("w1:4", real=True)
    lhs = (sum(li * x ** 2 for li, x in zip(lam, w))
           * sum(x ** 2 / li for li, x in zip(lam, w))
           - sum(x ** 2 for x in w) ** 2)
    rhs = sum(w[i] ** 2 * w[j] ** 2 * (lam[i] - lam[j]) ** 2 / (lam[i] * lam[j])
              for i, j in itertools.combinations(range(3), 2))
    assert sp.simplify(lhs - rhs) == 0
    # every term of rhs is a square over a positive product: rhs >= 0, and with
    # |u| = 1 (sum w^2 = 1) this is (u'Su)(u'S^-1u) >= 1.


def test_pedal_equals_mahalanobis_for_a_spherical_ellipsoid():
    lam, D, R, k = sp.symbols("lam D R k", positive=True)
    w = sp.symbols("w1:4", real=True)
    unit = {w[2]: sp.sqrt(1 - w[0] ** 2 - w[1] ** 2)}
    maha = (D - R) * sp.sqrt(sum(x ** 2 / lam for x in w)) / k
    pedal = (D - R) / (k * sp.sqrt(sum(lam * x ** 2 for x in w)))
    assert sp.simplify((maha - pedal).subs(unit)) == 0
