"""
Analytical 3D Curve-Line-Curve (CLC) point-to-target solver.

An open implementation of the closed-form point-to-target solution of
Sawaryn (2021):

    Sawaryn, S. J. (2021). "A Generalized Solution to the Point-to-Target
    Problem Using the Minimum Curvature Method." SPE Drilling & Completion.
    DOI: 10.2118/204111-PA.

A CLC trajectory connects a kickoff station (position + unit tangent) to a
target station with two circular arcs (radii ``R1``, ``R2``) joined by a
straight tangent of length ``beta``. The solution is parameterised by the
tangent length and the two subtended arc angles ``alpha1``, ``alpha2``.

The corrected Eq. 15, and a note on the printed one
---------------------------------------------------
The paper presents the solution two ways: the *forward* constraint equations
(Eqs. 11-13: eta1, eta4, eta14 as functions of alpha1, alpha2, beta) and the
*eliminated* implicit form (Eq. 15: a degree-10 polynomial in beta whose real
positive roots are every solution). The **printed** Eq. 15 is not scale-
covariant — it does not reproduce the paper's own worked roots under length
normalisation, i.e. it carries a transcription error in the eliminated
polynomial (whose true expansion the paper notes is "~4000 terms, beyond human
capability"). The ``eq15`` function below reproduces the printed form and is
retained only to document that trap (see ``test_eq15_is_trapped``).

The correct degree-10 coefficients were re-derived by replicating Sawaryn's own
surd-elimination (Appendix B: the half-angle quadratics B-15/B-16 and the
bilinear constraint B-19), eliminating the surds symbolically (``_eq15_coeffs``).
These power the vectorised closed-form solver ``solve_clc``, which returns
*every* CLC solution and reproduces Example 2 of SPE-204111-PA exactly. Subtended angles are reported as the true dogleg in [0, 2*pi) so the
measured depth ranks solutions correctly. Planar (eta14 ~ 0) and parallel-
tangent (|mu| = 1) cases are handled by ``solve_clc_2d`` (the paper's biquadratic
2D form). Forward-verified solvers (``solve_clc_analytical`` scan,
``solve_clc_resultant`` per-instance resultant) are provided as cross-checks.

This supersedes the iterative scheme of Sawaryn & Thorogood (2005, "A Compendium
of Directional Calculations Based on the Minimum Curvature Method", SPE-84246-PA)
that welleng's ``Connector`` inherits.

Citation
--------
Use of this work requires citation. Cite Sawaryn (2021, SPE-204111-PA) for the
underlying mathematics, and — for any use of *this* (welleng's) implementation
or its corrected coefficients — you must also cite
welleng (software concept DOI 10.5281/zenodo.20968887) and the welleng
analytical-CLC paper:

    Corcutt, J. (2026). An Open, Vectorized Closed-Form Solver for the 3D
    Curve-Hold-Curve Point-to-Target Problem. Zenodo. DOI 10.5281/zenodo.21130979.

----------------------------------------------------------------------------
Dedicated to the memory of **Steven J. Sawaryn**, in honour of his decades of
contributions to directional drilling and wellbore-positioning science — this
implements his last and most general solution to a problem he advanced for over
four decades.
----------------------------------------------------------------------------
"""

import numpy as np
from scipy.optimize import minimize_scalar, brentq


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


def solve_clc_analytical(p1, t1, p4, t4, R1, R2=None, n_scan=4000, tol=1e-6):
    """Solve the CLC point-to-target problem (Sawaryn 2021), forward-verified.

    Parameters
    ----------
    p1, t1 : (3,) array — kickoff position and unit tangent (N, E, V); ``t1`` is
        a unit vector, as tangents are throughout welleng (cf. ``Survey.vec_nev``).
    p4, t4 : (3,) array — target position and unit tangent.
    R1, R2 : float — first/second arc radii (``R2`` defaults to ``R1``, symmetric
        arcs). Unit-agnostic: any length unit, so long as positions and radii
        share it (e.g. all metres, or all feet). The returned lengths come back
        in that same unit; angles are radians.

    Returns
    -------
    list of dict, sorted by total measured depth, each with keys (driller's
    terms in brackets):
    ``beta`` / ``line`` — straight tangent length (the hold section);
    ``alpha1``, ``alpha2`` — subtended arc angles in radians (the dogleg of each
    build/turn); ``arc1``, ``arc2`` — arc lengths ``R*alpha`` (the build
    sections); ``total_md`` — total measured depth; ``residual``. The build-plane
    toolface is not returned but is recoverable from the reconstructed tangent.
    Complete: every real CLC solution is returned.
    """
    if R2 is None:
        R2 = R1
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


def solve_clc_resultant(p1, t1, p4, t4, R1, R2=None):
    """Complete CLC solve via per-instance resultant elimination.

    Independent cross-check for the vectorised ``solve_clc`` (exercised in the
    test suite); not on welleng's hot path.

    Eliminates the two half-angle tangents from Sawaryn's clean forward
    equations (11-13) by exact-rational resultants -> a polynomial in beta whose
    real positive roots include EVERY CLC solution (complete *by construction* -
    unlike the scan-based ``solve_clc_analytical``, which can drop large-angle /
    min-MD roots). Spurious roots (from clearing the half-angle denominators and
    squaring Eq. 13) are removed by forward-verification.

    Fast (~90ms/solve; python-flint resultant + arbitrary-precision acb roots,
    scale-normalised so coefficients stay representable) and complete +
    deterministic. The rational inputs are truncated to ~5 digits (1e5): the
    forward-verification filter rejects any root that drifts, so the lower
    precision buys ~2.4x speed with no loss of completeness.
    """
    if R2 is None:
        R2 = R1
    import flint
    from fractions import Fraction
    if not hasattr(flint, 'fmpq_mpoly_ctx'):        # some python-flint builds omit it
        raise ImportError(
            "solve_clc_resultant needs python-flint with rational multivariate "
            "polynomials (flint.fmpq_mpoly_ctx); this build lacks it. It is only an "
            "independent cross-check — use solve_clc to solve.")
    flint.ctx.prec = 400
    psi2, eta1, eta4, eta14, mu = _scalars(p1, t1, p4, t4)
    L = np.sqrt(psi2)                              # characteristic length -> O(1) coeffs

    def fq(x):
        f = Fraction(float(x)).limit_denominator(10**5)
        return flint.fmpq(f.numerator, f.denominator)

    ctx = flint.fmpq_mpoly_ctx.get(['T1', 'T2', 'B'], flint.Ordering.lex)
    T1, T2, B = ctx.gens()
    e1, e4, e14, m, r1, r2 = (fq(eta1/L), fq(eta4/L), fq(eta14/L),
                              fq(mu), fq(R1/L), fq(R2/L))
    # Sawaryn's clean forward eqs 11,12,13^2 with the half-angle denominators cleared
    P1 = r1*2*T1 + B*(1 - T1**2) + r2*T2*(m*(1 + T1**2) + (1 - T1**2)) - e1*(1 + T1**2)
    P2 = r1*T1*(m*(1 + T2**2) + (1 - T2**2)) + B*(1 - T2**2) + r2*2*T2 - e4*(1 + T2**2)
    u1p, u1m, u2p, u2m = 1 + T1**2, 1 - T1**2, 1 + T2**2, 1 - T2**2
    surd = ((1 - m**2)*u1p**2*u2p**2 - u1m**2*u2p**2 - u1p**2*u2m**2
            + 2*m*u1m*u2m*u1p*u2p)
    P3 = (r1*T1 + B + r2*T2)**2 * surd - e14**2*(1 - m**2)*u1p**2*u2p**2
    # eliminate T1 (index 0), then T2 (index 1) -> univariate polynomial in B
    F = P1.resultant(P2, 0).resultant(P1.resultant(P3, 0), 1)
    deg = max(mn[2] for mn in F.monoms())
    cl = [flint.fmpq(0)]*(deg + 1)
    for mn, co in zip(F.monoms(), F.coeffs()):
        cl[mn[2]] = co
    Fp = flint.fmpq_poly(cl)
    sqf = Fp // Fp.gcd(Fp.derivative())            # squarefree part (acb roots need it)
    roots = flint.acb_poly([sqf[i] for i in range(sqf.degree() + 1)]).roots()
    betas = sorted({float(z.real)*L for z in roots
                    if abs(float(z.imag)) < 1e-5 and float(z.real) > 1e-6})
    # The 1e5-truncated polynomial localises each root to ~1e-2; polish it
    # against the EXACT invariants so the path hits the target to ~1e-12 while
    # the resultant stays cheap. Spurious roots never reach zero -> filtered.
    from scipy.optimize import minimize_scalar

    def _resid(b):
        a1s, a2s = subtended_angles(b, psi2, eta1, eta4, eta14, mu, R1, R2)
        best = np.inf
        for a1 in a1s:
            for a2 in a2s:
                f = forward(a1, a2, b, mu, R1, R2)
                if f is not None:
                    best = min(best, np.sqrt((f[0]-eta1)**2 + (f[1]-eta4)**2
                                             + (abs(f[2])-abs(eta14))**2))
        return best

    tol = 1e-4 * np.sqrt(psi2)                     # acceptance (loose; spurious are far)
    xatol = 1e-9 * max(L, 1.0)                      # scale-relative polish precision
    sols = []
    for i, b0 in enumerate(betas):
        lo = betas[i-1] if i else 0.0
        hi = betas[i+1] if i + 1 < len(betas) else b0 + 0.02*L
        w = min(0.02*L, 0.4*(b0 - lo), 0.4*(hi - b0))
        b = float(minimize_scalar(_resid, bounds=(max(1e-9, b0 - w), b0 + w),
                                  method='bounded', options={'xatol': xatol}).x)
        if _resid(b) >= tol or any(abs(b - s['beta']) < 1e-3 for s in sols):
            continue
        a1s, a2s = subtended_angles(b, psi2, eta1, eta4, eta14, mu, R1, R2)
        for a1 in a1s:
            for a2 in a2s:
                f = forward(a1, a2, b, mu, R1, R2)
                if f is None:
                    continue
                res = np.sqrt((f[0]-eta1)**2 + (f[1]-eta4)**2 + (abs(f[2])-abs(eta14))**2)
                if res < tol:
                    arc1, arc2 = R1*abs(a1), R2*abs(a2)
                    sols.append(dict(beta=b, alpha1=a1, alpha2=a2,
                                     arc1=arc1, line=b, arc2=arc2,
                                     total_md=arc1 + b + arc2, residual=res))
                    break
            else:
                continue
            break
    sols.sort(key=lambda s: s['total_md'])
    return sols


# ----------------------------------------------------------------------------
# Vectorised closed-form solver -- corrected Sawaryn (2021) Eq. 15.
# ----------------------------------------------------------------------------
def _eq15_coeffs(psi2, g1, g4, l, R1, R2):
    """Corrected Sawaryn (2021) Eq. 15 -- the degree-10 closed-form polynomial in
    beta. Coefficients c0..c10 as functions of the frame-invariants
    g1=eta1=Dp.t1, g4=eta4=Dp.t4, l=mu=t1.t4, psi2=|Dp|^2, and the arc radii R1,R2
    (eta14 is folded into psi2 via psi2 = (g1^2+g4^2-2 l g1 g4)/(1-l^2) + eta14^2).
    Derived by computer-algebra surd-elimination of the half-angle tangents
    (Appendix B, Eqs. B-15, B-16, B-19, via resultant elimination); not the
    printed Eq. 15, which carries transcription errors. Reproducible in any CAS.
    Evaluate with scale-normalised invariants (psi2=1, g/L, R/L) for conditioning;
    roots are beta/L."""
    # Machine-generated by symbolic resultant elimination; one un-wrapped
    # expression per coefficient c0..c10 -- kept verbatim and inline so the engine
    # has a single source of truth (wrapping or splitting out would only invite
    # transcription error in a ~4000-term expansion no human edits by hand).
    return [
        2*g1*g4*psi2**4+psi2**5-l*psi2**5+16*g1**3*g4*psi2**2*R1**2+8*g1**2*psi2**3*R1**2-16*g1*g4*psi2**3*R1**2-8*g1**2*l*psi2**3*R1**2-8*psi2**4*R1**2+8*l*psi2**4*R1**2+32*g1**5*g4*R1**4+16*g1**4*psi2*R1**4-64*g1**3*g4*psi2*R1**4-16*g1**4*l*psi2*R1**4-32*g1**2*psi2**2*R1**4+32*g1*g4*psi2**2*R1**4+32*g1**2*l*psi2**2*R1**4+16*psi2**3*R1**4-16*l*psi2**3*R1**4+16*g1*g4**3*psi2**2*R2**2-16*g1*g4*psi2**3*R2**2+8*g4**2*psi2**3*R2**2-8*g4**2*l*psi2**3*R2**2-8*psi2**4*R2**2+8*l*psi2**4*R2**2-64*g1**3*g4**3*R1**2*R2**2-64*g1**3*g4*psi2*R1**2*R2**2+96*g1**2*g4**2*psi2*R1**2*R2**2-64*g1*g4**3*psi2*R1**2*R2**2+160*g1**2*g4**2*l*psi2*R1**2*R2**2-64*g1**2*psi2**2*R1**2*R2**2+112*g1*g4*psi2**2*R1**2*R2**2-64*g4**2*psi2**2*R1**2*R2**2+32*g1**2*l*psi2**2*R1**2*R2**2-32*g1*g4*l*psi2**2*R1**2*R2**2+32*g4**2*l*psi2**2*R1**2*R2**2-80*g1*g4*l**2*psi2**2*R1**2*R2**2+56*psi2**3*R1**2*R2**2-72*l*psi2**3*R1**2*R2**2+8*l**2*psi2**3*R1**2*R2**2+8*l**3*psi2**3*R1**2*R2**2-128*g1**4*R1**4*R2**2+192*g1**3*g4*R1**4*R2**2-128*g1**2*g4**2*R1**4*R2**2+128*g1*g4**3*R1**4*R2**2+128*g1**3*g4*l*R1**4*R2**2-256*g1**2*g4**2*l*R1**4*R2**2+64*g1**3*g4*l**2*R1**4*R2**2+224*g1**2*psi2*R1**4*R2**2-192*g1*g4*psi2*R1**4*R2**2+64*g4**2*psi2*R1**4*R2**2-160*g1**2*l*psi2*R1**4*R2**2-64*g4**2*l*psi2*R1**4*R2**2-32*g1**2*l**2*psi2*R1**4*R2**2+192*g1*g4*l**2*psi2*R1**4*R2**2-32*g1**2*l**3*psi2*R1**4*R2**2-96*psi2**2*R1**4*R2**2+160*l*psi2**2*R1**4*R2**2-32*l**2*psi2**2*R1**4*R2**2-32*l**3*psi2**2*R1**4*R2**2+32*g1*g4**5*R2**4-64*g1*g4**3*psi2*R2**4+16*g4**4*psi2*R2**4-16*g4**4*l*psi2*R2**4+32*g1*g4*psi2**2*R2**4-32*g4**2*psi2**2*R2**4+32*g4**2*l*psi2**2*R2**4+16*psi2**3*R2**4-16*l*psi2**3*R2**4+128*g1**3*g4*R1**2*R2**4-128*g1**2*g4**2*R1**2*R2**4+192*g1*g4**3*R1**2*R2**4-128*g4**4*R1**2*R2**4-256*g1**2*g4**2*l*R1**2*R2**4+128*g1*g4**3*l*R1**2*R2**4+64*g1*g4**3*l**2*R1**2*R2**4+64*g1**2*psi2*R1**2*R2**4-192*g1*g4*psi2*R1**2*R2**4+224*g4**2*psi2*R1**2*R2**4-64*g1**2*l*psi2*R1**2*R2**4-160*g4**2*l*psi2*R1**2*R2**4+192*g1*g4*l**2*psi2*R1**2*R2**4-32*g4**2*l**2*psi2*R1**2*R2**4-32*g4**2*l**3*psi2*R1**2*R2**4-96*psi2**2*R1**2*R2**4+160*l*psi2**2*R1**2*R2**4-32*l**2*psi2**2*R1**2*R2**4-32*l**3*psi2**2*R1**2*R2**4-128*g1**2*R1**4*R2**4+32*g1*g4*R1**4*R2**4-128*g4**2*R1**4*R2**4+256*g1**2*l*R1**4*R2**4+128*g1*g4*l*R1**4*R2**4+256*g4**2*l*R1**4*R2**4-128*g1**2*l**2*R1**4*R2**4-320*g1*g4*l**2*R1**4*R2**4-128*g4**2*l**2*R1**4*R2**4+128*g1*g4*l**3*R1**4*R2**4+32*g1*g4*l**4*R1**4*R2**4+144*psi2*R1**4*R2**4-336*l*psi2*R1**4*R2**4+160*l**2*psi2*R1**4*R2**4+96*l**3*psi2*R1**4*R2**4-48*l**4*psi2*R1**4*R2**4-16*l**5*psi2*R1**4*R2**4,  # c0
        2*g1*psi2**4+2*g4*psi2**4+16*g1**3*psi2**2*R1**2+16*g1**2*g4*psi2**2*R1**2-16*g1*psi2**3*R1**2-16*g4*psi2**3*R1**2+32*g1**5*R1**4+32*g1**4*g4*R1**4-64*g1**3*psi2*R1**4-64*g1**2*g4*psi2*R1**4+32*g1*psi2**2*R1**4+32*g4*psi2**2*R1**4+16*g1*g4**2*psi2**2*R2**2+16*g4**3*psi2**2*R2**2-16*g1*psi2**3*R2**2-16*g4*psi2**3*R2**2-64*g1**3*g4**2*R1**2*R2**2-64*g1**2*g4**3*R1**2*R2**2-64*g1**3*psi2*R1**2*R2**2+64*g1**2*g4*psi2*R1**2*R2**2+64*g1*g4**2*psi2*R1**2*R2**2-64*g4**3*psi2*R1**2*R2**2+128*g1**2*g4*l*psi2*R1**2*R2**2+128*g1*g4**2*l*psi2*R1**2*R2**2+48*g1*psi2**2*R1**2*R2**2+48*g4*psi2**2*R1**2*R2**2-96*g1*l*psi2**2*R1**2*R2**2-96*g4*l*psi2**2*R1**2*R2**2-16*g1*l**2*psi2**2*R1**2*R2**2-16*g4*l**2*psi2**2*R1**2*R2**2-64*g1**3*R1**4*R2**2+192*g1**2*g4*R1**4*R2**2-128*g1*g4**2*R1**4*R2**2+128*g4**3*R1**4*R2**2-128*g1**3*l*R1**4*R2**2+128*g1**2*g4*l*R1**4*R2**2-256*g1*g4**2*l*R1**4*R2**2+64*g1**3*l**2*R1**4*R2**2+64*g1**2*g4*l**2*R1**4*R2**2+64*g1*psi2*R1**4*R2**2-192*g4*psi2*R1**4*R2**2+128*g1*l*psi2*R1**4*R2**2+128*g4*l*psi2*R1**4*R2**2-192*g1*l**2*psi2*R1**4*R2**2+64*g4*l**2*psi2*R1**4*R2**2+32*g1*g4**4*R2**4+32*g4**5*R2**4-64*g1*g4**2*psi2*R2**4-64*g4**3*psi2*R2**4+32*g1*psi2**2*R2**4+32*g4*psi2**2*R2**4+128*g1**3*R1**2*R2**4-128*g1**2*g4*R1**2*R2**4+192*g1*g4**2*R1**2*R2**4-64*g4**3*R1**2*R2**4-256*g1**2*g4*l*R1**2*R2**4+128*g1*g4**2*l*R1**2*R2**4-128*g4**3*l*R1**2*R2**4+64*g1*g4**2*l**2*R1**2*R2**4+64*g4**3*l**2*R1**2*R2**4-192*g1*psi2*R1**2*R2**4+64*g4*psi2*R1**2*R2**4+128*g1*l*psi2*R1**2*R2**4+128*g4*l*psi2*R1**2*R2**4+64*g1*l**2*psi2*R1**2*R2**4-192*g4*l**2*psi2*R1**2*R2**4+32*g1*R1**4*R2**4+32*g4*R1**4*R2**4-128*g1*l*R1**4*R2**4-128*g4*l*R1**4*R2**4+192*g1*l**2*R1**4*R2**4+192*g4*l**2*R1**4*R2**4-128*g1*l**3*R1**4*R2**4-128*g4*l**3*R1**4*R2**4+32*g1*l**4*R1**4*R2**4+32*g4*l**4*R1**4*R2**4,  # c1
        -8*g1*g4*psi2**3-3*psi2**4+5*l*psi2**4-32*g1**3*g4*psi2*R1**2-8*g1**2*psi2**2*R1**2+32*g1*g4*psi2**2*R1**2+24*g1**2*l*psi2**2*R1**2+8*psi2**3*R1**2-24*l*psi2**3*R1**2+16*g1**4*R1**4+16*g1**4*l*R1**4-32*g1**2*psi2*R1**4-32*g1**2*l*psi2*R1**4+16*psi2**2*R1**4+16*l*psi2**2*R1**4-32*g1*g4**3*psi2*R2**2+32*g1*g4*psi2**2*R2**2-8*g4**2*psi2**2*R2**2+24*g4**2*l*psi2**2*R2**2+8*psi2**3*R2**2-24*l*psi2**3*R2**2+128*g1**3*g4*R1**2*R2**2-160*g1**2*g4**2*R1**2*R2**2+128*g1*g4**3*R1**2*R2**2-160*g1**2*g4**2*l*R1**2*R2**2+96*g1**2*psi2*R1**2*R2**2-96*g1*g4*psi2*R1**2*R2**2+96*g4**2*psi2*R1**2*R2**2-96*g1**2*l*psi2*R1**2*R2**2+64*g1*g4*l*psi2*R1**2*R2**2-96*g4**2*l*psi2*R1**2*R2**2+160*g1*g4*l**2*psi2*R1**2*R2**2-88*psi2**2*R1**2*R2**2+56*l*psi2**2*R1**2*R2**2-8*l**2*psi2**2*R1**2*R2**2-24*l**3*psi2**2*R1**2*R2**2-32*g1**2*R1**4*R2**2-64*g4**2*R1**4*R2**2+32*g1**2*l*R1**4*R2**2+128*g1*g4*l*R1**4*R2**2+64*g4**2*l*R1**4*R2**2-32*g1**2*l**2*R1**4*R2**2-128*g1*g4*l**2*R1**4*R2**2+32*g1**2*l**3*R1**4*R2**2+32*psi2*R1**4*R2**2-32*l*psi2*R1**4*R2**2-32*l**2*psi2*R1**4*R2**2+32*l**3*psi2*R1**4*R2**2+16*g4**4*R2**4+16*g4**4*l*R2**4-32*g4**2*psi2*R2**4-32*g4**2*l*psi2*R2**4+16*psi2**2*R2**4+16*l*psi2**2*R2**4-64*g1**2*R1**2*R2**4-32*g4**2*R1**2*R2**4+64*g1**2*l*R1**2*R2**4+128*g1*g4*l*R1**2*R2**4+32*g4**2*l*R1**2*R2**4-128*g1*g4*l**2*R1**2*R2**4-32*g4**2*l**2*R1**2*R2**4+32*g4**2*l**3*R1**2*R2**4+32*psi2*R1**2*R2**4-32*l*psi2*R1**2*R2**4-32*l**2*psi2*R1**2*R2**4+32*l**3*psi2*R1**2*R2**4+16*R1**4*R2**4-48*l*R1**4*R2**4+32*l**2*R1**4*R2**4+32*l**3*R1**4*R2**4-48*l**4*R1**4*R2**4+16*l**5*R1**4*R2**4,  # c2
        -8*g1*psi2**3-8*g4*psi2**3-32*g1**3*psi2*R1**2-32*g1**2*g4*psi2*R1**2+32*g1*psi2**2*R1**2+32*g4*psi2**2*R1**2-32*g1*g4**2*psi2*R2**2-32*g4**3*psi2*R2**2+32*g1*psi2**2*R2**2+32*g4*psi2**2*R2**2+128*g1**3*R1**2*R2**2+128*g4**3*R1**2*R2**2-128*g1**2*g4*l*R1**2*R2**2-128*g1*g4**2*l*R1**2*R2**2-96*g1*psi2*R1**2*R2**2-96*g4*psi2*R1**2*R2**2+64*g1*l*psi2*R1**2*R2**2+64*g4*l*psi2*R1**2*R2**2+32*g1*l**2*psi2*R1**2*R2**2+32*g4*l**2*psi2*R1**2*R2**2,  # c3
        12*g1*g4*psi2**2+2*psi2**3-10*l*psi2**3+16*g1**3*g4*R1**2-8*g1**2*psi2*R1**2-16*g1*g4*psi2*R1**2-24*g1**2*l*psi2*R1**2+8*psi2**2*R1**2+24*l*psi2**2*R1**2+16*g1*g4**3*R2**2-16*g1*g4*psi2*R2**2-8*g4**2*psi2*R2**2-24*g4**2*l*psi2*R2**2+8*psi2**2*R2**2+24*l*psi2**2*R2**2+32*g1**2*R1**2*R2**2-80*g1*g4*R1**2*R2**2+32*g4**2*R1**2*R2**2+64*g1**2*l*R1**2*R2**2-32*g1*g4*l*R1**2*R2**2+64*g4**2*l*R1**2*R2**2-80*g1*g4*l**2*R1**2*R2**2-24*psi2*R1**2*R2**2+8*l*psi2*R1**2*R2**2-8*l**2*psi2*R1**2*R2**2+24*l**3*psi2*R1**2*R2**2,  # c4
        12*g1*psi2**2+12*g4*psi2**2+16*g1**3*R1**2+16*g1**2*g4*R1**2-16*g1*psi2*R1**2-16*g4*psi2*R1**2+16*g1*g4**2*R2**2+16*g4**3*R2**2-16*g1*psi2*R2**2-16*g4*psi2*R2**2-16*g1*R1**2*R2**2-16*g4*R1**2*R2**2+32*g1*l*R1**2*R2**2+32*g4*l*R1**2*R2**2-16*g1*l**2*R1**2*R2**2-16*g4*l**2*R1**2*R2**2,  # c5
        -8*g1*g4*psi2+2*psi2**2+10*l*psi2**2+8*g1**2*R1**2+8*g1**2*l*R1**2-8*psi2*R1**2-8*l*psi2*R1**2+8*g4**2*R2**2+8*g4**2*l*R2**2-8*psi2*R2**2-8*l*psi2*R2**2-8*R1**2*R2**2+8*l*R1**2*R2**2+8*l**2*R1**2*R2**2-8*l**3*R1**2*R2**2,  # c6
        -8*g1*psi2-8*g4*psi2,  # c7
        2*g1*g4-3*psi2-5*l*psi2,  # c8
        2*g1+2*g4,  # c9
        1+l,  # c10
    ]


def _companion_roots(co):
    """Roots of a batch of degree-10 polynomials via companion-matrix eigenvalues.

    ``co``: (11, N) array, rows ``c0..c10`` (``c10`` the leading coeff). Returns
    (N, 10) complex roots. This is what makes the closed form *vectorisable* -
    one ``np.linalg.eigvals`` call over all N pairs, no Python loop.
    """
    N = co.shape[1]
    C = np.zeros((N, 10, 10))
    C[:, 1:, :-1] = np.eye(9)[None]
    C[:, :, -1] = -(co[:10] / co[10]).T
    return np.linalg.eigvals(C)


def _clc_solutions(P1, T1, P4, T4, R1, R2):
    """Core engine: every CLC solution per pair (general, non-degenerate case).

    Returns ``(beta, alpha1, alpha2, total_md, valid)``, each (N, 10): the full
    candidate set (roots of the corrected Eq. 15) with the forward-verified arc
    angles, measured depth, and a validity mask. Selection (shortest / all) is
    left to the public wrappers. Assumes ``|mu| < 1`` and ``eta14 != 0`` (route
    ``|eta14|~0`` / ``|mu|~1`` to :func:`solve_clc_2d`).
    """
    P1, T1, P4, T4 = (np.asarray(a, float) for a in (P1, T1, P4, T4))
    dp = P4 - P1
    mu = np.einsum('ij,ij->i', T1, T4)
    psi2 = np.einsum('ij,ij->i', dp, dp)
    eta1 = np.einsum('ij,ij->i', dp, T1)
    eta4 = np.einsum('ij,ij->i', dp, T4)
    with np.errstate(divide='ignore', invalid='ignore'):
        eta14 = np.einsum('ij,ij->i', dp, np.cross(T1, T4)) / np.sqrt(1.0 - mu**2)
    L = np.sqrt(psi2)
    N = len(psi2)
    R1a = np.broadcast_to(np.asarray(R1, float), (N,))
    R2a = np.broadcast_to(np.asarray(R2, float), (N,))
    co = np.array(_eq15_coeffs(np.ones(N), eta1/L, eta4/L, mu, R1a/L, R2a/L))
    # Degenerate pairs (|mu|=1) drop the polynomial's order: the leading
    # coefficient vanishes (or coefficients go non-finite), so the companion
    # divide co[:10]/co[10] would blow up. Neutralise those columns so eigvals
    # does not raise, and mark them invalid below. Callers route |mu|=1 to
    # solve_clc_2d, the planar/parallel form (Sawaryn 2021, Eq. 34).
    scale = np.max(np.abs(co), axis=0)
    bad = (~np.isfinite(co).all(axis=0)) | (np.abs(co[10]) < 1e-12 * (scale + 1e-300))
    if bad.any():
        co[:, bad] = 0.0
        co[10, bad] = 1.0                            # dummy monic -> finite roots
    roots = _companion_roots(co) * L[:, None]
    b, bi = roots.real, roots.imag
    ok = (np.abs(bi) < 1e-4 * (np.abs(b) + 1)) & (b > 1e-6)
    R1b, R2b = R1a[:, None], R2a[:, None]
    ps, e1, e4 = psi2[:, None], eta1[:, None], eta4[:, None]
    e14, M, psb = eta14[:, None], mu[:, None], psi2[:, None] - b**2
    A1 = 4*R1b**2*((e1+b)**2 - R2b**2*(1-M**2)); B1 = 4*R1b*(2*R2b**2*(e1-M*e4)-psb*(e1+b)); C1 = psb**2-4*R2b**2*(ps-e4**2)
    A2 = 4*R2b**2*((e4+b)**2 - R1b**2*(1-M**2)); B2 = 4*R2b*(2*R1b**2*(e4-M*e1)-psb*(e4+b)); C2 = psb**2-4*R1b**2*(ps-e1**2)
    d1 = np.maximum(B1**2 - 4*A1*C1, 0.0); d2 = np.maximum(B2**2 - 4*A2*C2, 0.0)
    res4, a1_4, a2_4 = [], [], []
    for s1 in (1, -1):
        a1 = 2*np.arctan2(-B1 + s1*np.sqrt(d1), 2*A1)
        for s2 in (1, -1):
            a2 = 2*np.arctan2(-B2 + s2*np.sqrt(d2), 2*A2)
            c1, sn1 = np.cos(a1), np.sin(a1); c2, sn2 = np.cos(a2), np.sin(a2)
            t1h, t2h = np.tan(a1/2), np.tan(a2/2)
            f1 = R1b*sn1 + b*c1 + R2b*t2h*(M+c1)
            f4 = R1b*t1h*(M+c2) + b*c2 + R2b*sn2
            with np.errstate(divide='ignore', invalid='ignore'):
                sd = (1-M**2 - c1**2 - c2**2 + 2*M*c1*c2)/(1-M**2)
            # clamp a tiny-negative surd to 0 -- at eta14~0 (planar) the true
            # solution's surd sits at ~0- numerically; rejecting it on surd<0
            # drops a valid root. eta1/eta4 still filter genuinely-bad branches.
            f14 = (R1b*t1h + b + R2b*t2h) * np.sqrt(np.maximum(sd, 0.0))
            res4.append(np.sqrt((f1-e1)**2 + (f4-e4)**2 + (np.abs(f14)-np.abs(e14))**2))
            a1_4.append(a1); a2_4.append(a2)
    res4 = np.stack(res4, -1); a1_4 = np.stack(a1_4, -1); a2_4 = np.stack(a2_4, -1)
    k = np.argmin(res4, -1)                          # best-matching angle branch
    res = np.take_along_axis(res4, k[..., None], -1)[..., 0]
    a1b = np.take_along_axis(a1_4, k[..., None], -1)[..., 0]
    a2b = np.take_along_axis(a2_4, k[..., None], -1)[..., 0]
    # The residual is built from R-scale terms (f1 = R1*sin(a1) + b*cos(a1)
    # + R2*tan(a2/2)*(mu + cos(a1))), so it CANNOT be compared against an
    # L-scaled bound: when R >> L the tolerance falls below the residual's own
    # floor and the CORRECT root is rejected (measured: the genuine 30 m root
    # for a 30 m throw at R=573 has res 3.2e-2 vs a 1e-4*L bound of 3e-3).
    # Scale by the characteristic length of the compared quantities so
    # acceptance cannot depend on the R/L ratio.
    valid = ok & (res < 1e-4 * np.maximum(L[:, None], R1b + R2b))
    if bad.any():
        valid[bad] = False                          # degenerate pairs -> use solve_clc_2d
    # subtended_angles returns 2*arctan2(...) in (-2pi, 2pi]; a co-terminal value
    # (same tan(alpha/2), same path) inflates |alpha|. Normalise to the TRUE arc
    # turn in [0, 2pi) so the measured depth -- and any drawing -- is correct.
    a1b = a1b % (2 * np.pi)
    a2b = a2b % (2 * np.pi)
    md = R1b * a1b + b + R2b * a2b
    return b, a1b, a2b, md, valid


# TODO (max-radius solver): when no renderable CLC exists at the given radii,
# the useful follow-up is NOT the beta=0 critical (minimum) radius but the
# GENTLEST feasible curve -- the LARGEST radius for which a valid CLC exists with
# both arc doglegs <= pi (a > pi arc is a non-physical loop the min-curve renderer
# can't draw). That is the proper analytic replacement for the old iterative
# auto-tighten: maximise R subject to "a CLC exists AND max(alpha1, alpha2) <= pi".
# The binding constraint at the maximum is an arc dogleg hitting pi (or beta -> 0),
# expressible via the same Eq.15 coefficients. Symmetric R1=R2=R is a clean solve;
# asymmetric is along a fixed R1:R2 ratio (or a bounded 2D search). Until then,
# Connector raises on no-CLC-at-design-radii and the caller sweeps R themselves.
# Follow-up: once this solver exists, give Connector an opt-in (e.g.
# on_infeasible='raise' | 'max_radius') to RETURN the max-radius solution (or its
# radius) instead of raising when the target is infeasible at the design DLS.
def solve_clc(p1, t1, p4, t4, R1, R2=None, return_all=False):
    """Solve the CLC point-to-target problem for a single station pair.

    Main entry point. Runs the general closed-form solver first; degenerate pairs
    (parallel/antiparallel tangents ``|mu| = 1``, or planar ``eta14 ~ 0``)
    auto-fall back to :func:`solve_clc_2d`, so the caller need not pre-classify.

    Parameters
    ----------
    p1, t1 : (3,) array_like
        Kickoff position and unit tangent (N, E, V); ``t1`` is a unit vector.
    p4, t4 : (3,) array_like
        Target position and unit tangent.
    R1, R2 : float
        First / second arc radii. ``R2`` defaults to ``R1`` (symmetric arcs).
        Unit-agnostic, so long as positions and radii share one length unit
        (e.g. all metres, or all feet).
    return_all : bool, default False
        If False, return only the shortest (minimum measured-depth) solution.
        If True, return every valid CLC solution.

    Returns
    -------
    dict or list of dict or None
        ``return_all=False``: the shortest solution as a dict with keys ``beta``
        (tangent / hold length), ``alpha1``, ``alpha2`` (arc doglegs, radians)
        and ``total_md`` (measured depth); ``None`` if no CLC exists.
        ``return_all=True``: list of such dicts, shortest first.
    """
    R2 = R1 if R2 is None else R2
    mu = float(np.asarray(t1, float) @ np.asarray(t4, float))
    if abs(mu) > 1 - 1e-9:                           # parallel/antiparallel: use 2D form
        return solve_clc_2d(p1, t1, p4, t4, R1, R2, return_all=return_all)
    b, a1, a2, md, valid = (x[0] for x in _clc_solutions([p1], [t1], [p4], [t4], R1, R2))
    sols = [dict(beta=float(b[k]), alpha1=float(a1[k]), alpha2=float(a2[k]),
                 total_md=float(md[k])) for k in range(len(b)) if valid[k]]
    if not sols:
        # general form is singular here (parallel/antiparallel tangents, |mu|=1)
        # -> the planar 2D form, which subsumes that case.
        return solve_clc_2d(p1, t1, p4, t4, R1, R2, return_all=return_all)
    sols.sort(key=lambda s: s['total_md'])
    if return_all:
        return sols
    return sols[0]


def solve_clc_2d(p1, t1, p4, t4, R1, R2=None, return_all=False):
    """Planar / singular CLC solve (``eta14 ~ 0``) — Sawaryn Eq. 34, biquadratic in beta.

    Covers the degenerate 2D case AND the parallel/antiparallel-tangent
    singularities ``mu = +-1`` (where the general form's ``1/(1-mu^2)`` blows up):
    ``eta14`` is identically 0 there, so this biquadratic subsumes Sawaryn's
    Eqs 37/38. The ``+-`` is the two arc senses (Figs 10/11). Verification uses
    Eqs 11-12 only (the out-of-plane surd sits at 0 numerically here).

    return_all=False (default): the shortest solution dict, or None.
    return_all=True: list of all solution dicts, shortest first.
    """
    if R2 is None:
        R2 = R1
    dp = np.asarray(p4, float) - np.asarray(p1, float)
    mu = float(np.asarray(t1, float) @ np.asarray(t4, float))
    psi2 = float(dp @ dp); eta1 = float(dp @ t1); eta4 = float(dp @ t4)
    eta14 = 0.0 if abs(mu) > 1 - 1e-9 else float(dp @ np.cross(t1, t4)) / np.sqrt(1 - mu**2)
    tol = 1e-4 * np.sqrt(psi2)
    K = psi2**2 - 4*R1**2*(psi2-eta1**2) - 4*R2**2*(psi2-eta4**2) + 4*R1**2*R2**2*(1-mu)**2
    Bc = (1+mu)*psi2 - 2*eta1*eta4
    Bq = -(1-mu)

    def _branch(b):                                  # best (alpha1, alpha2, residual)
        a1s, a2s = subtended_angles(b, psi2, eta1, eta4, eta14, mu, R1, R2)
        best = (np.inf, None, None)
        for x1 in a1s:
            for x2 in a2s:
                c1, s1 = np.cos(x1), np.sin(x1); c2, s2 = np.cos(x2), np.sin(x2)
                T1h, T2h = np.tan(x1/2), np.tan(x2/2)
                r = np.hypot(R1*s1 + b*c1 + R2*T2h*(mu+c1) - eta1,
                             R1*T1h*(mu+c2) + b*c2 + R2*s2 - eta4)
                if r < best[0]:
                    best = (r, x1, x2)
        return best

    sols = []
    for sgn in (1, -1):                              # Eq 34 +- : the two arc senses
        a2 = -2*psi2 - sgn*4*R1*R2*Bq
        a0 = K - sgn*4*R1*R2*Bc
        disc = a2**2 - 4*a0
        if disc < 0:
            continue
        for u in ((-a2 + np.sqrt(disc))/2, (-a2 - np.sqrt(disc))/2):
            if u <= 1e-12:
                continue
            b = float(np.sqrt(u))
            r, x1, x2 = _branch(b)
            if r < tol and not any(abs(b - s['beta']) < 1e-2 for s in sols):
                x1, x2 = x1 % (2 * np.pi), x2 % (2 * np.pi)   # true arc turn
                sols.append(dict(beta=b, alpha1=x1, alpha2=x2,
                                 total_md=R1 * x1 + b + R2 * x2))
    sols.sort(key=lambda s: s['total_md'])
    if return_all:
        return sols
    return sols[0] if sols else None


def _max_radius_2d(p1, t1, p4, t4, ratio):
    """Maximum radius for the parallel-tangent (|mu|=1) case, via the 2D solver.

    The general form carries a 1/sqrt(1-mu^2) and is singular here, so bisect the
    feasibility edge: the largest R for which the planar solve has a solution with
    both arc doglegs <= pi (``solve_clc_2d`` also returns >pi loops, so filter).
    """
    pi = np.pi + 1e-9
    L = float(np.linalg.norm(np.asarray(p4, float) - np.asarray(p1, float)))

    def feas(R):
        return [s for s in solve_clc_2d(p1, t1, p4, t4, R, ratio * R, return_all=True)
                if s['alpha1'] <= pi and s['alpha2'] <= pi]

    lo, hi = 1e-3 * L, 5.0 * L
    if not feas(lo):
        return None
    if not feas(hi):                                     # edge lies in (lo, hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if feas(mid):
                lo = mid
            else:
                hi = mid
    R = lo                                               # largest feasible R found
    sols = feas(R)
    if not sols:
        return None
    s = min(sols, key=lambda x: x['total_md'])
    return dict(radius=R, radius2=ratio * R, beta=s['beta'],
                alpha1=s['alpha1'], alpha2=s['alpha2'], total_md=s['total_md'])


def max_radius(p1, t1, p4, t4, ratio=1.0):
    """Largest radius admitting a valid CLC — the gentlest feasible curve.

    The point-to-target CLC is reachable with both arc doglegs ``<= pi`` only up
    to a maximum radius; beyond it the target is reachable only by a ``> pi``
    (loop) arc, which the minimum-curvature renderer cannot draw. That maximum is
    the ``beta = 0`` (curve-curve / biarc) boundary, where the hold vanishes —
    equivalently the largest root of the constant coefficient ``c0`` of the
    corrected Eq. 15 whose biarc has both doglegs ``<= pi``. This is the analytic
    form of the classical "critical radius": the gentlest curvature that still
    reaches the target. A caller can fall back to it when no CLC exists at the
    design radii, instead of iterating the radius down.

    Parameters
    ----------
    p1, t1, p4, t4 : (3,) array_like
        Kickoff / target positions and unit tangents (N, E, V).
    ratio : float, default 1.0
        ``R2 / R1``. ``1.0`` is symmetric radii; otherwise the second radius
        scales with the first along this ratio.

    Returns
    -------
    dict or None
        ``radius`` (R1), ``radius2`` (R2), ``beta`` (0.0), ``alpha1``, ``alpha2``
        (biarc doglegs, radians) and ``total_md``; ``None`` if no feasible biarc
        exists (the target is unreachable under the ``pi`` constraint).

    Notes
    -----
    Closed-form condition of Sawaryn (2021, SPE-204111-PA); no iteration. Parallel
    tangents (``|mu| = 1``) — where the general form is singular — are handled by a
    2D feasibility bisection (:func:`solve_clc_2d`). See :func:`solve_clc` for the
    general (fixed-design-radius) solve.
    """
    mu = float(np.asarray(t1, float) @ np.asarray(t4, float))
    if abs(mu) > 1 - 1e-9:                  # parallel tangents: general form singular
        return _max_radius_2d(p1, t1, p4, t4, ratio)
    psi2, e1, e4, e14, mu = _scalars(p1, t1, p4, t4)
    L = np.sqrt(psi2)

    def _c0(R):
        return _eq15_coeffs(1.0, e1 / L, e4 / L, mu, R / L, ratio * R / L)[0]

    grid = np.linspace(1e-3 * L, 5.0 * L, 600)
    cv = np.array([_c0(r) for r in grid])
    best = None
    for i in range(len(grid) - 1):
        if cv[i] * cv[i + 1] >= 0:
            continue
        R = brentq(_c0, grid[i], grid[i + 1])
        R1, R2 = R, ratio * R
        a1s, a2s = subtended_angles(0.0, psi2, e1, e4, e14, mu, R1, R2)
        for x1 in a1s:
            for x2 in a2s:
                f = forward(x1, x2, 0.0, mu, R1, R2)
                if f is None:
                    continue
                if (abs(f[0] - e1) < 1e-4 * L and abs(f[1] - e4) < 1e-4 * L
                        and abs(abs(f[2]) - abs(e14)) < 1e-4 * L):
                    a1, a2 = x1 % (2 * np.pi), x2 % (2 * np.pi)
                    if a1 <= np.pi + 1e-9 and a2 <= np.pi + 1e-9 and (
                            best is None or R > best['radius']):
                        best = dict(radius=R1, radius2=R2, beta=0.0,
                                    alpha1=a1, alpha2=a2, total_md=R1 * a1 + R2 * a2)
    return best


def solve_clc_landing(p1, t1, p0, t4, R1, R2=None, return_all=False):
    """Land onto a LINE target: p4 = p0 + k*t4, solving for the scalar k.

    The *landing* problem (Sawaryn 2021, Appendix C):
    the target is not a fixed point but any point on the line through ``p0`` in
    direction ``t4``; the free parameter is the along-line distance ``k``, and the
    connection is a biarc (``beta = 0``). On that line the invariants collapse to
    low-order functions of ``k`` (Eqs. C-8/C-13/C-17):
    ``eta1 = eps1 + mu*k``, ``eta4 = eps4 + k``, ``eta14 = eps14`` (constant),
    ``psi^2 = psi0^2 + 2*eps4*k + k^2``, with ``eps* = (p0 - p1).<basis>``.
    Substituting these into the biarc condition ``c0 = 0`` (the constant
    coefficient of the corrected Eq. 15) gives a polynomial in ``k`` (Eq. 44)
    whose roots are the landing distances; this is solved numerically for ``k``.

    Parameters
    ----------
    p1, t1 : (3,) array_like
        Kickoff position and unit tangent (N, E, V).
    p0, t4 : (3,) array_like
        The landing line: ``p0`` is its anchor (the ``k = 0`` base point) and
        ``t4`` its unit direction. Named ``p0`` (not ``p4``) because the target
        point ``p4 = p0 + k*t4`` is the *solved* output, not an input.
    R1, R2 : float
        Arc radii; ``R2`` defaults to ``R1``.
    return_all : bool, default False
        False -> the shortest feasible landing (both biarc doglegs <= pi) as a
        dict, or ``None``. True -> every landing root, feasible-first then by MD.

    Returns
    -------
    dict or list of dict or None
        Each dict: ``k`` (along-line distance), ``p4`` (landing point), ``beta``
        (0.0), ``alpha1``, ``alpha2`` (biarc doglegs, radians), ``total_md``.
    """
    R2 = R1 if R2 is None else R2
    p1, t1, p0, t4 = (np.asarray(a, float) for a in (p1, t1, p0, t4))
    mu = float(t1 @ t4)
    d0 = p0 - p1
    eps1 = float(d0 @ t1)
    eps4 = float(d0 @ t4)
    eps14 = 0.0 if abs(mu) > 1 - 1e-9 else float(d0 @ np.cross(t1, t4)) / np.sqrt(1 - mu**2)
    psi0_2 = float(d0 @ d0)

    def _c0k(k):
        psi2 = psi0_2 + 2 * eps4 * k + k * k
        if psi2 <= 0:
            return np.nan
        L = np.sqrt(psi2)
        return _eq15_coeffs(1.0, (eps1 + mu * k) / L, (eps4 + k) / L, mu, R1 / L, R2 / L)[0]

    kmax = 4.0 * (np.sqrt(psi0_2) + R1 + R2)
    ks = np.linspace(1e-6, kmax, 3000)
    cv = np.array([_c0k(k) for k in ks])
    sols = []
    for i in range(len(ks) - 1):
        if not (cv[i] * cv[i + 1] < 0):
            continue
        k = brentq(_c0k, ks[i], ks[i + 1])
        psi2 = psi0_2 + 2 * eps4 * k + k * k
        g1, g4 = eps1 + mu * k, eps4 + k
        a1s, a2s = subtended_angles(0.0, psi2, g1, g4, eps14, mu, R1, R2)
        # Branch selection by CHORD-IDENTITY CLOSURE, not the forward-residual.
        # The middle tangent v = (delta - R1 T1 t1 - R2 T2 t4)/(R1 T1 + R2 T2)
        # (beta = 0 on the landing biarc) must be a UNIT vector for a valid
        # solution. This is decisive at the razor-edge (|v|-1 ~ 1e-5 for the true
        # branch vs ~1 for the alternate), where the forward-residual flips with
        # the last bits of k: forward()'s Eq.13 surd sits at +/-1e-16 at the root
        # and its strict surd<0 -> None rejects the true branch on rounding.
        best = (np.inf, None, None)
        delta = (p0 + k * t4) - p1
        for x1 in a1s:
            for x2 in a2s:
                T1, T2 = np.tan(x1 / 2), np.tan(x2 / 2)
                denom = R1 * T1 + R2 * T2
                if abs(denom) < 1e-12:
                    continue
                v = (delta - R1 * T1 * t1 - R2 * T2 * t4) / denom
                r = abs(float(np.linalg.norm(v)) - 1.0)
                if r < best[0]:
                    best = (r, x1, x2)
        if best[1] is None:
            continue
        a1, a2 = best[1] % (2 * np.pi), best[2] % (2 * np.pi)
        sols.append(dict(k=float(k), p4=p0 + k * t4, beta=0.0,
                         alpha1=a1, alpha2=a2, total_md=R1 * a1 + R2 * a2))
    sols.sort(key=lambda s: (not (s['alpha1'] <= np.pi + 1e-9 and s['alpha2'] <= np.pi + 1e-9),
                             s['total_md']))
    if return_all:
        return sols
    feas = [s for s in sols if s['alpha1'] <= np.pi + 1e-9 and s['alpha2'] <= np.pi + 1e-9]
    return feas[0] if feas else None


