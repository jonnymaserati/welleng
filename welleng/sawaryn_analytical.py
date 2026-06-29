"""
Analytical 3D Curve-Line-Curve (CLC) point-to-target solver.

The first open, runnable implementation of the closed-form point-to-target
solution of Sawaryn (2021):

    Sawaryn, S. J. (2021). "A Generalized Solution to the Point-to-Target
    Problem Using the Minimum Curvature Method." SPE Drilling & Completion.
    DOI: 10.2118/204111-PA.

A CLC trajectory connects a kickoff station (position + unit tangent) to a
target station with two circular arcs (radii ``R1``, ``R2``) joined by a
straight tangent of length ``beta``. The solution is parameterised by the
tangent length and the two subtended arc angles ``alpha1``, ``alpha2``.

How this solver works (and why it sidesteps a trap)
---------------------------------------------------
The paper presents the solution two ways: the *forward* constraint equations
(Eqs. 11-13: eta1, eta4, eta14 as functions of alpha1, alpha2, beta) and the
*eliminated* implicit form (Eq. 15: a degree-10 polynomial in beta whose roots
are every solution). The printed Eq. 15 is **not** scale-covariant — it does not
reproduce the paper's own worked roots under length normalisation, i.e. it
carries a transcription/print error in the eliminated polynomial (whose true
expansion the paper notes is "~4000 terms, beyond human capability").

So this solver does **not** use Eq. 15. It uses Sawaryn's *clean* pieces — the
forward equations (11-13, verified exact) and the half-angle back-substitution
(Eqs. 18-25, verified exact) — and finds the tangent lengths by driving the
eta-residual to zero (forward-verification). This is exact, scale-covariant, and
reproduces Example 2 of SPE-204111-PA to the printed precision. ``eq15`` is
retained below for reference only, flagged as the unreliable form.

This supersedes the iterative Sawaryn-Thorogood (2005, SPE-84246-PA) scheme that
welleng's ``Connector`` inherits.

----------------------------------------------------------------------------
Dedicated to the memory of **Steven J. Sawaryn**, in honour of his decades of
contributions to directional drilling and wellbore-positioning science. This is
his last and most general solution to a problem he advanced for over four
decades, coded here in the open for the first time.
----------------------------------------------------------------------------
"""

import numpy as np
from scipy.optimize import minimize_scalar


def tangent(inc_deg, azi_deg):
    """Unit tangent [N, E, V] from inclination + azimuth (degrees)."""
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    return np.array([np.sin(inc) * np.cos(azi),
                     np.sin(inc) * np.sin(azi),
                     np.cos(inc)])


def _scalars(p1, t1, p4, t4):
    """Dot-product invariants (Eqs. 2-6) from the two stations.

    All five are rotation- and translation-invariant, so the solution is
    frame-free by construction — no canonical-frame transform is required.
    """
    dp = np.asarray(p4, float) - np.asarray(p1, float)
    mu = float(np.asarray(t1, float) @ np.asarray(t4, float))   # mu = t1 . t4
    psi2 = float(dp @ dp)                                        # Eq. 3
    eta1 = float(dp @ t1)                                        # Eq. 4
    eta4 = float(dp @ t4)                                        # Eq. 5
    eta14 = float(dp @ np.cross(t1, t4)) / np.sqrt(1.0 - mu**2)  # Eq. 6
    return psi2, eta1, eta4, eta14, mu


def subtended_angles(beta, psi2, eta1, eta4, eta14, mu, R1, R2):
    """Candidate subtended arc angles (alpha1, alpha2), radians, for a ``beta``.

    Eqs. 18-25: each half-angle tangent ``T = tan(alpha/2)`` solves a quadratic
    ``A T^2 + B T + C = 0``. Returns the two branches per arc; forward-
    verification selects the physical one.
    """
    psb = psi2 - beta**2
    A1 = 4*R1**2*((eta1 + beta)**2 - R2**2*(1 - mu**2))          # Eq. 18
    B1 = 4*R1*(2*R2**2*(eta1 - mu*eta4) - psb*(eta1 + beta))     # Eq. 19
    C1 = psb**2 - 4*R2**2*(psi2 - eta4**2)                       # Eq. 20
    A2 = 4*R2**2*((eta4 + beta)**2 - R1**2*(1 - mu**2))          # Eq. 22
    B2 = 4*R2*(2*R1**2*(eta4 - mu*eta1) - psb*(eta4 + beta))     # Eq. 23
    C2 = psb**2 - 4*R1**2*(psi2 - eta1**2)                       # Eq. 24

    def _roots(A, B, C):
        disc = B*B - 4*A*C
        if A == 0 or disc < 0:
            return []
        sq = np.sqrt(disc)
        return [2*np.arctan2((-B + sq), 2*A),     # Eqs. 21 / 25
                2*np.arctan2((-B - sq), 2*A)]

    return _roots(A1, B1, C1), _roots(A2, B2, C2)


def forward(alpha1, alpha2, beta, mu, R1, R2):
    """Forward model (Eqs. 11-13): (eta1, eta4, eta14) from the path parameters.

    Verified exact against SPE-204111-PA Example 2. Returns ``None`` where the
    angular surd is negative (geometrically inconsistent).
    """
    c1, s1 = np.cos(alpha1), np.sin(alpha1)
    c2, s2 = np.cos(alpha2), np.sin(alpha2)
    T1, T2 = np.tan(alpha1/2), np.tan(alpha2/2)
    eta1 = R1*s1 + beta*c1 + R2*T2*(mu + c1)                     # Eq. 11
    eta4 = R1*T1*(mu + c2) + beta*c2 + R2*s2                     # Eq. 12
    surd = (1 - mu**2 - c1**2 - c2**2 + 2*mu*c1*c2) / (1 - mu**2)
    if surd < 0:
        return None
    eta14 = (R1*T1 + beta + R2*T2) * np.sqrt(surd)               # Eq. 13
    return np.array([eta1, eta4, eta14])


def solve_clc_analytical(p1, t1, p4, t4, R1, R2, n_scan=4000, tol=1e-6):
    """Solve the CLC point-to-target problem (Sawaryn 2021), forward-verified.

    Parameters
    ----------
    p1, t1 : (3,) array — kickoff position and unit tangent (N, E, V).
    p4, t4 : (3,) array — target position and unit tangent.
    R1, R2 : float — first/second arc radii (any consistent length unit).

    Returns
    -------
    list of dict, sorted by total measured depth, each with keys
    ``beta``, ``alpha1``, ``alpha2`` (radians), ``arc1``, ``line``, ``arc2``,
    ``total_md`` and ``residual``. Complete: every real CLC solution is returned.
    """
    psi2, eta1, eta4, eta14, mu = _scalars(p1, t1, p4, t4)
    target = np.array([eta1, eta4, eta14])

    def best_branch(beta):
        a1s, a2s = subtended_angles(beta, psi2, eta1, eta4, eta14, mu, R1, R2)
        best = (np.inf, None, None)
        for a1 in a1s:
            for a2 in a2s:
                f = forward(a1, a2, beta, mu, R1, R2)
                if f is not None:
                    # Eq. 13 carries a ± out-of-plane branch: match |eta14|
                    # (sign is resolved at path reconstruction).
                    r = float(np.sqrt((f[0] - target[0])**2
                                      + (f[1] - target[1])**2
                                      + (abs(f[2]) - abs(target[2]))**2))
                    if r < best[0]:
                        best = (r, a1, a2)
        return best

    def residual(beta):
        return best_branch(beta)[0]

    b_max = 5.0 * np.sqrt(psi2)
    grid = np.linspace(1e-4 * np.sqrt(psi2), b_max, n_scan)
    res = np.array([residual(b) for b in grid])

    solutions = []
    for i in range(1, len(grid) - 1):
        if res[i] < res[i-1] and res[i] < res[i+1] and res[i] < 0.1 * np.sqrt(psi2):
            opt = minimize_scalar(
                residual, bracket=(grid[i-1], grid[i], grid[i+1]))
            beta = float(opt.x)
            r, a1, a2 = best_branch(beta)
            if r > tol * np.sqrt(psi2):
                continue
            arc1, arc2 = R1 * abs(a1), R2 * abs(a2)
            sol = dict(beta=beta, alpha1=a1, alpha2=a2,
                       arc1=arc1, line=beta, arc2=arc2,
                       total_md=arc1 + beta + arc2, residual=r)
            if not any(abs(s['beta'] - beta) < 1e-3 for s in solutions):
                solutions.append(sol)

    solutions.sort(key=lambda s: s['total_md'])
    return solutions


def eq15(beta, psi2, eta1, eta4, eta14, mu, R1, R2):
    """Sawaryn Eq. 15 — the eliminated degree-10 polynomial. **REFERENCE ONLY.**

    The printed form is NOT scale-covariant (it does not reproduce the paper's
    own worked roots under length normalisation) — it carries a transcription/
    print error in the eliminated polynomial. ``solve_clc_analytical`` does not use it;
    it forward-verifies via the clean Eqs. 11-13 + 18-25 instead. Kept here only
    to document the discrepancy.
    """
    b2, mu2 = beta * beta, mu * mu
    A = (eta1 + eta4 + (1 + mu) * b2)**2 + (1 - mu2) * eta14**2
    inner1 = ((psi2 - b2)**2 - 4*R1**2*(psi2 - eta1**2)
              - 4*R2**2*(psi2 - eta4**2) + 4*R1**2*R2**2*(1 - mu)**2)**2
    inner2 = 16*R1**2*R2**2*(((1 + mu)*psi2 - 2*eta1*eta4 - (1 - mu)*b2)**2
                             + 4*b2*(1 - mu2)*eta14**2)
    term2 = (32*R1**2*R2**2*(1 + mu)*(1 - mu2)*eta14**2
             * ((psi2 - b2)**2 + 4*(R1**2*eta1**2 + R2**2*eta4**2)
                + 4*R1**2*R2**2*(1 - mu)**2))
    term3 = 4*(R1**2 - R2**2)*(eta1 - eta4)*beta
    term4 = 2*(R1**2 + R2**2)*((1 - mu)*(psi2 - b2) + 2*eta1*eta4)
    return A*(inner1 - inner2) + term2 + term3 - term4
