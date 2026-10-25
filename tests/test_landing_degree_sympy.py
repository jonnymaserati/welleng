"""Durable derivation test: the Sawaryn LINE-landing closure is an irreducible
degree-10 polynomial in the along-line distance ``k``.

This pins WHY ``solve_clc_landing`` roots ``c0(k)`` numerically (brentq scan)
rather than by a formula: the closure is not solvable in radicals. If anyone
later tries to replace the numerical root with a "closed form", this test states
the obstruction (an irreducible degree-10 polynomial → Abel-Ruffini). It also
guards against a regression that silently lowers the closure's degree.

Ref: Sawaryn (2021) SPE-204111-PA Eq. 15 (degree-10) and Appendix C (landing).
"""
import sympy as sp

from welleng.sawaryn_analytical import _eq15_coeff


def test_line_landing_closure_is_irreducible_degree_10():
    k = sp.symbols('k', real=True)
    # concrete rational constants -> the generic degree of the closure in k
    mu = sp.Rational(1, 3)
    eps1, eps4, psi0_2, R1, R2 = (
        sp.Integer(2), sp.Integer(3), sp.Integer(100), sp.Integer(50), sp.Integer(60)
    )
    Q = psi0_2 + 2 * eps4 * k + k**2          # = L^2, quadratic in k
    L = sp.sqrt(Q)
    g1, g4 = (eps1 + mu * k) / L, (eps4 + k) / L
    c0 = sp.expand(sp.simplify(
        _eq15_coeff(0, sp.Integer(1), g1, g4, mu, R1 / L, R2 / L)
    ))
    # c0 = A(k) + B(k)*sqrt(Q); rationalise: c0 = 0  <=>  A^2 = B^2 Q
    A = c0.subs(L, 0)
    B = sp.simplify((c0 - A) / L)
    assert not B.has(L), "radical did not separate cleanly"
    num = sp.expand(sp.numer(sp.together(A**2 - B**2 * Q)))
    factors = sorted(sp.Poly(f, k).degree() for f, _ in sp.factor_list(num)[1])
    # one irreducible degree-10 factor (squared by the rationalisation)
    assert factors == [10], f"expected [10], got {factors}"
    # degree 10 >= 5 and irreducible => not solvable in radicals (Abel-Ruffini)
    assert factors[0] >= 5
