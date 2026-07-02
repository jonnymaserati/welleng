"""Validate the Sawaryn (2021, SPE-204111-PA) analytical CLC solver against the
paper's own worked Example 2.

Same kickoff/target as Example 1, R1=1250 ft, R2=1750 ft → tangent-length roots
β = (1072.6, 1630.2, 1789.95, 2356.9) ft, principal β=1072.6 with subtended arc
angles α1=13.953°, α2=13.109°.

The solver is forward-verified (Eqs. 11-13 + 18-25); it does NOT use the printed
Eq. 15, which carries a transcription/print error (see ``test_eq15_is_trapped``).
"""

import numpy as np
import pytest

from welleng.sawaryn_analytical import (
    tangent, _scalars, forward, subtended_angles, solve_clc_analytical, eq15,
    solve_clc_resultant,
    _eq15_coeffs, solve_clc, solve_clc_batch, solve_clc_2d, max_radius,
    solve_clc_landing, solve_clc_r_sweep, solve_clc_r_grid, _build_r_scales,
)

P1 = np.array([8000.0, 8000.0, 6000.0])
P4 = np.array([9500.0, 8800.0, 6500.0])
T1 = tangent(75.0, 15.0)
T4 = tangent(85.0, 30.0)
R1, R2 = 1250.0, 1750.0


def test_scalars_match_example1():
    # Paper Example 1: ψ²=3.14e6, η1=1728.93, η4=1736.15, η14=252.95, μ=0.95202
    psi2, eta1, eta4, eta14, mu = _scalars(P1, T1, P4, T4)
    assert psi2 == pytest.approx(3.14e6, rel=1e-4)
    assert eta1 == pytest.approx(1728.93, abs=0.02)
    assert eta4 == pytest.approx(1736.15, abs=0.02)
    assert eta14 == pytest.approx(252.95, abs=0.02)
    assert mu == pytest.approx(0.95202, abs=1e-4)


def test_forward_model_reproduces_example2():
    # Eqs. 11-13 from the known principal solution must give the η invariants.
    _, eta1, eta4, eta14, mu = _scalars(P1, T1, P4, T4)
    f = forward(np.radians(13.953), np.radians(13.109), 1072.6, mu, R1, R2)
    assert f[0] == pytest.approx(eta1, abs=0.05)
    assert f[1] == pytest.approx(eta4, abs=0.05)
    assert f[2] == pytest.approx(eta14, abs=0.05)


def test_back_substitution_is_exact_at_principal():
    # Eqs. 18-25: at the true principal β=1072.6, the angles must be exact.
    psi2, eta1, eta4, eta14, mu = _scalars(P1, T1, P4, T4)
    a1s, a2s = subtended_angles(1072.6, psi2, eta1, eta4, eta14, mu, R1, R2)
    assert any(np.degrees(a) == pytest.approx(13.953, abs=0.01) for a in a1s)
    assert any(np.degrees(a) == pytest.approx(13.109, abs=0.01) for a in a2s)


def test_solve_clc_reproduces_example2_exactly():
    # The full forward-verified solver: all four roots + the principal angles.
    sols = solve_clc_analytical(P1, T1, P4, T4, R1, R2)
    betas = sorted(s['beta'] for s in sols)
    expected = [1072.6, 1630.2, 1789.95, 2356.9]
    assert len(betas) == 4
    for got, exp in zip(betas, expected):
        assert got == pytest.approx(exp, abs=0.2)   # exact to printed precision
    principal = min(sols, key=lambda s: s['beta'])
    assert np.degrees(principal['alpha1']) == pytest.approx(13.953, abs=0.01)
    assert np.degrees(principal['alpha2']) == pytest.approx(13.109, abs=0.01)
    assert principal['residual'] < 1e-3


def test_solve_clc_resultant_is_complete_on_example2():
    # The resultant solver is complete *by construction* (every valid solution's
    # beta is a root of the eliminated polynomial); spurious roots filtered by
    # forward-verification. Recovers all four Example-2 roots exactly.
    pytest.importorskip("flint")  # optional 'analytical' extra (python-flint)
    sols = solve_clc_resultant(P1, T1, P4, T4, R1, R2)
    betas = sorted(s['beta'] for s in sols)
    expected = [1072.6, 1630.2, 1789.95, 2356.9]
    assert len(betas) == 4
    for got, exp in zip(betas, expected):
        assert got == pytest.approx(exp, abs=0.2)
    principal = min(sols, key=lambda s: s['beta'])
    assert np.degrees(principal['alpha1']) == pytest.approx(13.953, abs=0.01)
    assert np.degrees(principal['alpha2']) == pytest.approx(13.109, abs=0.01)
    # roots found cheap (1e5-truncated polynomial) then Newton-polished against the
    # exact invariants -> path hits the target well within a ft on a ~1772-ft case.
    assert all(s['residual'] < 1e-4 for s in sols)


def test_eq15_is_trapped():
    # Document Sawaryn's trap: the printed Eq. 15 does NOT vanish at the true
    # principal root (the forward-verified solver is exact there instead).
    psi2, eta1, eta4, eta14, mu = _scalars(P1, T1, P4, T4)
    scale = abs(eq15(500.0, psi2, eta1, eta4, eta14, mu, R1, R2))
    assert abs(eq15(1072.6, psi2, eta1, eta4, eta14, mu, R1, R2)) > 1e-3 * scale


# --- Vectorised closed-form engine (corrected Eq. 15) ------------------------

def _build_clc(dl1, tf1, dl2, tf2, dist):
    """Forward-construct a CLC path from the origin with tangent [0,0,1]; returns
    (target position, target unit tangent). For construct-and-recover tests."""
    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    c1, s1 = np.cos(dl1), np.sin(dl1)
    ct1, st1 = np.cos(tf1), np.sin(tf1)
    a = np.array([(1 - c1) * ct1, (1 - c1) * st1, s1])
    v = np.array([s1 * ct1, s1 * st1, c1])
    b = a + dist * v
    R = Rz(tf1) @ Ry(dl1) @ Rz(tf2)
    c2, s2 = np.cos(dl2), np.sin(dl2)
    return b + R @ np.array([1 - c2, 0, s2]), R @ np.array([s2, 0, c2])


def test_corrected_eq15_vanishes_at_example2_roots():
    # Unlike the printed (trapped) Eq. 15, the DERIVED degree-10 polynomial
    # vanishes at all four paper roots (scale-normalised evaluation).
    psi2, eta1, eta4, eta14, mu = _scalars(P1, T1, P4, T4)
    L = np.sqrt(psi2)
    co = _eq15_coeffs(1.0, eta1 / L, eta4 / L, mu, R1 / L, R2 / L)
    scale = max(abs(c) for c in co)
    for beta in (1072.6, 1630.2, 1789.95, 2356.9):
        # vanishes to float-coefficient precision (~1e-6 relative); the printed
        # Eq. 15 by contrast is O(scale) here (see test_eq15_is_trapped).
        assert abs(np.polyval(co[::-1], beta / L)) < 1e-4 * scale


def test_solve_clc_batch_reproduces_example2():
    out = solve_clc_batch(P1[None], T1[None], P4[None], T4[None], R1, R2,
                          return_all=True)
    betas = sorted(out['beta'][0][out['valid'][0]])
    assert len(betas) == 4
    for got, exp in zip(betas, [1072.6, 1630.2, 1789.95, 2356.9]):
        assert got == pytest.approx(exp, abs=0.2)


def test_solve_clc_default_returns_shortest():
    shortest = solve_clc(P1, T1, P4, T4, R1, R2)                 # default
    allsol = solve_clc(P1, T1, P4, T4, R1, R2, return_all=True)
    assert len(allsol) == 4
    assert shortest['total_md'] == pytest.approx(min(s['total_md'] for s in allsol))
    assert shortest['beta'] == pytest.approx(1072.6, abs=0.2)


def test_solve_clc_batch_constructed_recovery():
    # build CLC paths, the solver must return the build tangent length for each.
    rng = np.random.default_rng(2024)
    N = 60
    P4s, T4s, truth = [], [], []
    for _ in range(N):
        dl1, tf1, dl2, tf2, d = (rng.uniform(0.3, 2.0), rng.uniform(-np.pi, np.pi),
                                 rng.uniform(0.3, 2.0), rng.uniform(-np.pi, np.pi),
                                 rng.uniform(0.3, 1.5))
        p3, v3 = _build_clc(dl1, tf1, dl2, tf2, d)
        P4s.append(p3); T4s.append(v3); truth.append(d)
    O = np.zeros((N, 3)); V = np.tile([0, 0, 1.], (N, 1))
    out = solve_clc_batch(O, V, np.array(P4s), np.array(T4s), 1.0, 1.0, return_all=True)
    rec = sum(bool(np.any(out['valid'][i] & (np.abs(out['beta'][i] - truth[i]) < 1e-2)))
              for i in range(N))
    assert rec == N


def test_solve_clc_2d_handles_planar_and_parallel():
    O, V = np.zeros(3), np.array([0, 0, 1.])
    # planar (eta14 = 0): tf1 = tf2 = 0
    p3, v3 = _build_clc(0.6, 0.0, 0.5, 0.0, 0.8)
    sols = solve_clc_2d(O, V, p3, v3, 1.0, 1.0, return_all=True)
    assert any(abs(s['beta'] - 0.8) < 1e-2 for s in sols)
    # mu = 1 parallel tangents: fwd(a, 0, a, pi, b) returns v3 = [0,0,1]
    p3, v3 = _build_clc(0.8, 0.0, 0.8, np.pi, 0.6)
    assert abs(V @ v3 - 1.0) < 1e-9                              # confirm mu = 1
    s = solve_clc_2d(O, V, p3, v3, 1.0, 1.0)
    assert s is not None and abs(s['beta'] - 0.6) < 1e-2


def test_closed_form_matches_resultant_on_example2():
    pytest.importorskip("flint")
    cf = sorted(solve_clc(P1, T1, P4, T4, R1, R2, return_all=True),
                key=lambda s: s['beta'])
    rs = sorted(solve_clc_resultant(P1, T1, P4, T4, R1, R2), key=lambda s: s['beta'])
    assert len(cf) == len(rs) == 4
    for a, b in zip(cf, rs):
        assert a['beta'] == pytest.approx(b['beta'], abs=0.05)


def test_solve_clc_recovers_planar_eta14_zero():
    # Planar (eta14 ~ 0) target: the surd in the forward model sits at ~0- numerically,
    # so a naive surd>=0 filter drops the valid root. Regression guard for the
    # asymmetric-batch completeness gap -- the generator's own path must be recovered.
    O, V = np.zeros(3), np.array([0, 0, 1.])
    p3, v3 = _build_clc(0.9, 0.0, 0.7, 0.0, 1.1)            # tf1 = tf2 = 0 -> planar
    _, _, _, e14, _ = _scalars(O, V, p3, v3)
    assert abs(e14) < 1e-6 * float(np.linalg.norm(p3))      # confirm planar
    sols = solve_clc(O, V, p3, v3, 1.0, 1.0, return_all=True)
    assert any(abs(s['beta'] - 1.1) < 1e-2 for s in sols)


def test_shortest_aligns_with_generator():
    # Regression for the |alpha|-vs-dogleg mis-ranking: subtended_angles returns
    # 2*arctan2(...), which can be a negative co-terminal of the true dogleg, so
    # scoring MD with |alpha| once labelled a 298-deg arc loop the 'shortest'.
    # For random asymmetric constructions the solver must (a) recover the
    # generator's own path -- arc1/line endpoints aligned -- and (b) rank by the
    # true dogleg, so no shortest MD exceeds the (known valid) generator's.
    O, V = np.zeros(3), np.array([0, 0, 1.])

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    rng = np.random.default_rng(2024)
    for _ in range(60):
        dl1, tf1 = rng.uniform(0.3, 2.2), rng.uniform(-np.pi, np.pi)
        dl2, tf2 = rng.uniform(0.3, 2.2), rng.uniform(-np.pi, np.pi)
        dist = rng.uniform(0.4, 1.8)
        r1, r2 = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)
        c1, s1 = np.cos(dl1), np.sin(dl1)
        ct1, st1 = np.cos(tf1), np.sin(tf1)
        gtA = r1 * np.array([(1-c1)*ct1, (1-c1)*st1, s1])
        v = np.array([s1*ct1, s1*st1, c1])
        gtB = gtA + dist * v
        R = Rz(tf1) @ Ry(dl1) @ Rz(tf2)
        c2, s2 = np.cos(dl2), np.sin(dl2)
        p4 = gtB + R @ (r2 * np.array([1-c2, 0, s2]))
        t4 = R @ np.array([s2, 0, c2])
        gt_md = r1*dl1 + dist + r2*dl2

        sols = solve_clc(O, V, p4, t4, r1, r2, return_all=True)
        assert sols
        # (a) generator path recovered -- some solution's arc1/line points align
        aligned = False
        for s in sols:
            T1, T2 = np.tan(s['alpha1']/2), np.tan(s['alpha2']/2)
            t2 = (p4 - r1*T1*V - r2*T2*t4) / (r1*T1 + s['beta'] + r2*T2)
            pos1 = r1*T1*(V + t2)
            pos2 = pos1 + s['beta']*t2
            # roots come from companion-matrix eigenvalues, not exact arithmetic,
            # so allow a little slack rather than demanding exact alignment.
            if np.linalg.norm(pos1-gtA) < 1e-3 and np.linalg.norm(pos2-gtB) < 1e-3:
                aligned = True
                break
        assert aligned, "generator path not among the solutions"
        # (b) ranking sane -- the true min-MD never exceeds a known valid path
        assert min(s['total_md'] for s in sols) <= gt_md + 1e-3


def test_r2_defaults_to_r1():
    # R2=None must mirror R2=R1 (symmetric arcs) across the public entry points.
    O, V = np.zeros(3), np.array([0, 0, 1.])
    p4, t4, R1 = np.array([1500., 800, 500]), tangent(85, 30), 1200.
    a = solve_clc(O, V, p4, t4, R1, return_all=True)
    b = solve_clc(O, V, p4, t4, R1, R1, return_all=True)
    assert [round(s['beta'], 6) for s in a] == [round(s['beta'], 6) for s in b]
    P, T = np.array([[0., 0, 0]]), np.array([[0., 0, 1.]])
    d = solve_clc_batch(P, T, p4[None], t4[None], R1)
    e = solve_clc_batch(P, T, p4[None], t4[None], R1, R1)
    assert np.allclose(d['total_md'], e['total_md'], equal_nan=True)


def test_max_radius_gentlest_feasible():
    # max_radius = the largest R giving a valid CLC with both arc doglegs <= pi,
    # i.e. the beta=0 biarc boundary (the analytic critical radius). The biarc
    # must reach the target with both arcs <= pi, and just above that radius the
    # target is reachable only by a > pi (loop) arc.
    O, V = np.zeros(3), np.array([0, 0, 1.])

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    rng = np.random.default_rng(11)
    for j in range(25):
        dl1, tf1 = rng.uniform(0.4, 1.8), rng.uniform(-np.pi, np.pi)
        dl2, tf2 = rng.uniform(0.4, 1.8), rng.uniform(-np.pi, np.pi)
        dist, r = rng.uniform(0.4, 1.5), rng.uniform(0.6, 1.4)
        c1, s1 = np.cos(dl1), np.sin(dl1)
        ct1, st1 = np.cos(tf1), np.sin(tf1)
        a = r * np.array([(1-c1)*ct1, (1-c1)*st1, s1])
        v = np.array([s1*ct1, s1*st1, c1])
        b = a + dist * v
        R = Rz(tf1) @ Ry(dl1) @ Rz(tf2)
        c2, s2 = np.cos(dl2), np.sin(dl2)
        p4 = b + R @ (r * np.array([1-c2, 0, s2]))
        t4 = R @ np.array([s2, 0, c2])

        mr = max_radius(O, V, p4, t4)
        assert mr is not None, j
        Rm, a1, a2 = mr['radius'], mr['alpha1'], mr['alpha2']
        assert a1 <= np.pi + 1e-6 and a2 <= np.pi + 1e-6, (j, a1, a2)
        assert mr['beta'] == 0.0
        # the beta=0 biarc reaches the target
        T1h, T2h = np.tan(a1/2), np.tan(a2/2)
        t2 = (p4 - Rm*T1h*V - Rm*T2h*t4) / (Rm*T1h + Rm*T2h)
        end = Rm*T1h*(V + t2) + Rm*T2h*(t2 + t4)
        assert np.linalg.norm(end - p4) < 1e-6 * (np.linalg.norm(p4) + 1), j
        # it is the maximum: just above, no CLC with both arcs <= pi exists
        above = solve_clc(O, V, p4, t4, Rm*1.05, Rm*1.05, return_all=True)
        assert not any(s['alpha1'] <= np.pi + 1e-9 and s['alpha2'] <= np.pi + 1e-9
                       for s in above), (j, Rm)


def test_parallel_and_antiparallel_tangents_handled():
    # |mu|=1 must not crash. Antiparallel (mu=-1) makes the general polynomial drop
    # order (leading coeff -> 0), which would blow up the companion divide; parallel
    # (mu=+1) too. Both route to the 2D form (Sawaryn Eq. 34) and must reconstruct.
    O = np.zeros(3)

    def recon(t1, t4, p4, s):
        T1h, T2h = np.tan(s['alpha1'] / 2), np.tan(s['alpha2'] / 2)
        t2 = (p4 - O - T1h * t1 - T2h * t4) / (T1h + s['beta'] + T2h)
        end = O + T1h * (t1 + t2) + s['beta'] * t2 + T2h * (t2 + t4)
        return np.linalg.norm(end - p4)

    for t1, t4, p4 in (
        (np.array([0., 0, 1.]), np.array([0., 0, 1.]), np.array([1.5, 0, 4.])),   # mu=+1
        (np.array([0., 0, 1.]), np.array([0., 0, -1.]), np.array([1.5, 0, -1.])),  # mu=-1
    ):
        s = solve_clc(O, t1, p4, t4, 1.0, 1.0)                # must not raise
        assert s is not None
        assert recon(t1, t4, p4, s) < 1e-6
        fb = solve_clc_batch([O], [t1], [p4], [t4], 1.0, 1.0)  # batch must not raise
        assert fb['found'][0]
        assert abs(fb['total_md'][0] - s['total_md']) < 1e-6
    # a mixed batch (antiparallel + general) must not crash on the degenerate row
    mb = solve_clc_batch(
        np.array([[0., 0, 0], [8000., 8000, 6000]]),
        np.array([[0., 0, 1.], tangent(75, 15)]),
        np.array([[1.5, 0, -1.], [9500., 8800, 6500]]),
        np.array([[0., 0, -1.], tangent(85, 30)]),
        np.array([1.0, 1250.]), np.array([1.0, 1750.]))
    assert mb['found'][0] and mb['found'][1]


def test_max_radius_parallel_tangents():
    # |mu|=1 (parallel tangents): the general form is singular, so max_radius
    # routes to a 2D feasibility bisection. Vertical S: t1 || t4, p4=[2,0,3].
    O = np.array([0., 0., 0.]); t1 = np.array([1., 0., 0.]); t4 = np.array([1., 0., 0.])
    p4 = np.array([2., 0., 3.])
    mr = max_radius(O, t1, p4, t4)
    assert mr is not None
    Rm = mr['radius']
    assert 1.05 < Rm < 1.12                              # the biarc boundary ~1.083
    assert mr['alpha1'] <= np.pi + 1e-9 and mr['alpha2'] <= np.pi + 1e-9
    pi = np.pi + 1e-9

    def feasible(R):
        return any(s['alpha1'] <= pi and s['alpha2'] <= pi
                   for s in solve_clc_2d(O, t1, p4, t4, R, R, return_all=True))
    assert feasible(Rm * 0.99)                           # drillable just below
    assert not feasible(Rm * 1.01)                       # gone just above


def test_landing_reproduces_example4():
    # SPE-204111-PA Example 4 (Wang 2019 landing): line target p4 = p0 + k*t4,
    # solve for k. Published: real roots k = 37.27, 567.52; the (largest / Wang)
    # solution k=567.52 has alpha1=30.774, alpha2=16.385, p4=(533.30,-194.11,3280.8).
    p1 = np.array([167.78, -81.72, 3293.92]); t1 = tangent(82, 328)
    p0 = np.array([0., 0., 3280.80]);         t4 = tangent(90, 340)
    R = 469.94
    allsol = solve_clc_landing(p1, t1, p0, t4, R, R, return_all=True)
    ks = [s['k'] for s in allsol]
    assert any(abs(k - 37.27) < 0.1 for k in ks), ks
    assert any(abs(k - 567.52) < 0.1 for k in ks), ks
    # default = the feasible (both arcs <= pi) landing = Wang's k=567.52
    s = solve_clc_landing(p1, t1, p0, t4, R, R)
    assert abs(s['k'] - 567.52) < 0.1
    assert abs(np.degrees(s['alpha1']) - 30.774) < 0.05
    assert abs(np.degrees(s['alpha2']) - 16.385) < 0.05
    assert np.allclose(s['p4'], [533.30, -194.11, 3280.8], atol=0.15)


def test_build_r_scales_snaps_design():
    # linspace of scale factors, with the design (1.0) snapped in when bracketed
    s = _build_r_scales(0.9, 1.1, 11)
    assert len(s) == 11
    assert s[0] == 0.9 and s[-1] == 1.1
    assert np.any(s == 1.0)                              # design radius present
    # range not bracketing 1.0 -> no snap
    assert not np.any(_build_r_scales(1.1, 1.3, 5) == 1.0)


def test_r_sweep_feature():
    # a feature (not new maths): the fixed-radius solver run over a radius range
    # (given as scale factors on the design radii) in one batched call. The
    # design radius is always in the sweep; feasible up to max_radius.
    R1d, R2d = 1250., 1750.
    sw = solve_clc_r_sweep(P1, T1, P4, T4, R1d, R2d,
                           scale_min=0.9, scale_max=1.1, n_steps=11)
    di = sw['design_index']
    # design row: scale 1.0, actual design radii, matches a direct solve
    assert sw['scale'][di] == 1.0
    assert sw['radius1'][di] == R1d and sw['radius2'][di] == R2d
    s0 = solve_clc(P1, T1, P4, T4, R1d, R2d)
    assert abs(s0['total_md'] - sw['total_md'][di]) < 1e-6
    # R2/R1 ratio preserved across the sweep
    assert np.allclose(sw['radius2'] / sw['radius1'], R2d / R1d)
    # gentler curve costs MD (monotone in scale); doglegs renderable (<= pi)
    assert np.all(sw['feasible'])
    assert np.all(np.diff(sw['total_md']) > 0)
    assert np.all(sw['alpha1'] <= np.pi + 1e-9)
    assert np.all(sw['alpha2'] <= np.pi + 1e-9)
    # every feasible row matches a direct solve at that radius
    for i in range(len(sw['scale'])):
        s = solve_clc(P1, T1, P4, T4, sw['radius1'][i], sw['radius2'][i])
        assert abs(s['total_md'] - sw['total_md'][i]) < 1e-6

    # explicit `values` path: full range incl R=0 and past the critical radius
    Rmax = max_radius(P1, T1, P4, T4)['radius']
    chord = np.linalg.norm(P4 - P1)
    swv = solve_clc_r_sweep(P1, T1, P4, T4, 1250.,
                            values=[0.0, 500., Rmax * 0.98, Rmax * 1.03])
    # R = 0 -> pure tangent: beta = MD = chord (arcs collapse to instant turns)
    assert swv['feasible'][0]
    assert abs(swv['beta'][0] - chord) < 1e-9
    assert abs(swv['total_md'][0] - chord) < 1e-9
    # feasible below the critical radius, infeasible above it
    assert swv['feasible'][:-1].all()
    assert not swv['feasible'][-1]
    assert np.isnan(swv['total_md'][-1])

    # scales and values are mutually exclusive
    with pytest.raises(ValueError):
        solve_clc_r_sweep(P1, T1, P4, T4, 1250., scales=[1.0], values=[1250.])


def test_r_sweep_planar_vertical_s_trap():
    # Parallel-tangent (|mu|=1) vertical S: the general form is singular, so the
    # batch must route these rows to solve_clc_2d (else the sweep wrongly reports
    # infeasible). And the operational trap: a clean 90/90 S at the design radius
    # becomes INFEASIBLE if the radius is raised ~10% (the drillable S vanishes;
    # only >pi loops remain), while reducing R keeps it feasible and simpler.
    O = np.array([0., 0., 0.]); t1 = np.array([1., 0., 0.]); t4 = np.array([1., 0., 0.])
    p4 = np.array([2., 0., 3.])                          # a 90/90 S at R = 1.0
    assert abs(t1 @ t4) == 1.0                           # exactly parallel (mu=1)

    # the singular row is handled, not skipped: batch agrees with the 2D solver
    fb = solve_clc_batch([O], [t1], [p4], [t4], 1.0, 1.0)
    s2 = solve_clc_2d(O, t1, p4, t4, 1.0, 1.0)
    assert fb['found'][0] and s2 is not None
    assert abs(fb['total_md'][0] - s2['total_md']) < 1e-6
    assert abs(np.degrees(fb['alpha1'][0]) - 90.0) < 0.5

    sw = solve_clc_r_sweep(O, t1, p4, t4, 1.0, values=[0.8, 1.0, 1.1])
    # reduce R -> feasible and gentler; design -> feasible ~90 deg
    assert sw['feasible'][0] and sw['feasible'][1]
    assert np.degrees(sw['alpha1'][0]) < np.degrees(sw['alpha1'][1])  # smaller R, smaller dogleg
    assert abs(np.degrees(sw['alpha1'][1]) - 90.0) < 0.5
    # raise R ~10% -> no drillable S (only >pi loops); the trap
    assert not sw['feasible'][2]
    assert np.isnan(sw['total_md'][2])


def test_r_grid_independent_axes():
    # independent R1 x R2 sweep: a 2D grid, one batched solve, design cell exact.
    R1d, R2d = 1250., 1750.
    g = solve_clc_r_grid(P1, T1, P4, T4, R1d, R2d,
                         scale_min=0.8, scale_max=1.2, n_steps=9)
    i, j = g['design_index']
    K1, K2 = len(g['r1_scales']), len(g['r2_scales'])
    assert g['feasible'].shape == (K1, K2) == g['total_md'].shape
    # design cell = (1.0, 1.0) -> the design radii, matching a direct solve
    assert g['r1_scales'][i] == 1.0 and g['r2_scales'][j] == 1.0
    assert g['radius1'][i] == R1d and g['radius2'][j] == R2d
    s = solve_clc(P1, T1, P4, T4, R1d, R2d)
    assert g['feasible'][i, j]
    assert abs(g['total_md'][i, j] - s['total_md']) < 1e-6
    # axis 0 is R1, axis 1 is R2: an off-diagonal cell matches its direct solve
    assert abs(g['total_md'][0, -1] -
               solve_clc(P1, T1, P4, T4, g['radius1'][0], g['radius2'][-1])['total_md']) < 1e-6
    # NaN exactly where infeasible
    assert np.all(np.isnan(g['total_md'][~g['feasible']]))
