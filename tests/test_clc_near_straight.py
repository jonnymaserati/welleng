"""CLC solves in the NEAR-STRAIGHT regime (small tangent change, R >> L).

This is the regime a well planner actually spends its time in -- local
connections, bridges, shortcuts -- and the regime where the validity bound was
wrong. `_clc_solutions` accepted a root when its forward-reconstruction
residual fell below ``1e-4 * L`` (L = |p4 - p1|), but that residual is built
from R-scale terms::

    f1 = R1*sin(a1) + b*cos(a1) + R2*tan(a2/2)*(mu + cos(a1))

so for R >> L the bound sits below the residual's own floor and the CORRECT
root is discarded -- leaving a long-way (>pi) root as the "shortest valid"
solution. Symptom in the field: a 30 m near-collinear move returned a 7230 m
double-loop (240x). Fixed by scaling the bound with the radius (9e103bb).

The oracle here is deliberately SELF-CONSISTENT rather than a predicted
geometry: a naive "arc needed (R*dtheta) fits inside the throw" predictor is
wrong in 3D because it ignores lateral closure. So instead we assert
(a) whatever is returned must reconstruct the target, and (b) if a direct root
that reconstructs exists, a direct root must be the one selected.
"""

import numpy as np
import pytest

from welleng.sawaryn_analytical import solve_clc

def _uv(inc, azi):
    i, a = np.radians(inc), np.radians(azi)
    return np.array([np.sin(i) * np.cos(a), np.sin(i) * np.sin(a), np.cos(i)])


def _rot(v, axis, ang):
    k = axis / np.linalg.norm(axis)
    return (v * np.cos(ang) + np.cross(k, v) * np.sin(ang)
            + k * (k @ v) * (1 - np.cos(ang)))


def _reconstruct(p1, v1, p4, v4, R1, R2, beta, a1, a2):
    """Rebuild the CLC end pose from a solution and return the target miss.

    Arc 1 turns v1 -> t (the hold direction) through a1 in the plane containing
    v1 and t; the hold runs beta along t; arc 2 turns t -> v4 through a2. The
    hold direction is recovered from the solution by turning v1 through a1 about
    the axis that also carries t into v4 -- i.e. the plane pair is fixed by
    requiring the second arc to land on v4. We solve for t directly: it is the
    unit vector with v1.t = cos(a1) and t.v4 = cos(a2).
    """
    n = np.cross(v1, v4)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.inf                    # planar/parallel -> solve_clc_2d
    n /= nn
    # t in the (v1, v4) plane basis, satisfying both dot constraints
    c1, c2 = np.cos(a1), np.cos(a2)
    mu = float(v1 @ v4)
    det = 1.0 - mu ** 2
    x = (c1 - mu * c2) / det
    y = (c2 - mu * c1) / det
    t = x * v1 + y * v4
    perp2 = 1.0 - (x * c1 + y * c2)
    if perp2 > 1e-12:                    # out-of-plane component
        t = t + np.sqrt(perp2) * n
    tn = np.linalg.norm(t)
    if tn < 1e-12:
        return np.inf
    t /= tn
    # arc 1: v1 -> t about their common normal
    ax1 = np.cross(v1, t)
    if np.linalg.norm(ax1) < 1e-14:
        p2 = p1 + 0.0 * v1
    else:
        ax1 /= np.linalg.norm(ax1)
        # chord of a circular arc: R * (rotate(p-centre)) ; centre = p1 + R*(ax1 x v1)
        ctr = p1 + R1 * np.cross(ax1, v1)
        p2 = ctr + _rot(p1 - ctr, ax1, a1)
    p3 = p2 + beta * t
    ax2 = np.cross(t, v4)
    if np.linalg.norm(ax2) < 1e-14:
        p_end = p3
    else:
        ax2 /= np.linalg.norm(ax2)
        ctr2 = p3 + R2 * np.cross(ax2, t)
        p_end = ctr2 + _rot(p3 - ctr2, ax2, a2)
    return float(np.linalg.norm(p_end - p4))


def _is_direct(sol):
    return (abs(float(sol["alpha1"])) <= np.pi + 1e-9
            and abs(float(sol["alpha2"])) <= np.pi + 1e-9)


# near-straight grid: small tangent changes, throws from 10 m to 300 m, and
# design DLS spanning R/L from ~1 to ~100
GRID = [
    (throw, dturn, dls)
    for throw in (10.0, 30.0, 100.0, 300.0)
    for dturn in (0.25, 1.0, 3.0)
    for dls in (1.0, 3.0, 6.0, 15.0)
]


def _case(throw, dturn):
    """A 3D near-straight pose pair: target ahead along v1, tangents differing
    by dturn in BOTH inclination and azimuth (so eta14 != 0, the general form).
    """
    v1 = _uv(20.0, 30.0)
    v2 = _uv(20.0 + dturn, 30.0 + dturn)
    p1 = np.zeros(3)
    return p1, v1, p1 + throw * v1, v2


@pytest.mark.parametrize("throw,dturn,dls", GRID)
def test_returned_solution_reconstructs_the_target(throw, dturn, dls):
    """Whatever is selected must actually land on the target.

    This is the fundamental gate: a loosened acceptance bound must not start
    admitting roots that do not solve the problem.

    Scoped to DIRECT solutions. :func:`_reconstruct` recovers the hold direction
    from ``cos(alpha)``, which cannot distinguish 359.96 deg from -0.04 deg, so
    the helper is only a valid oracle for arcs <= pi. Long-way solutions are
    covered by ``test_connector_analytical.test_random_pose_battery_*``, which
    checks endpoint accuracy through welleng's own renderer.
    """
    p1, v1, p4, v4 = _case(throw, dturn)
    R = np.degrees(30.0) / dls
    sol = solve_clc(p1, v1, p4, v4, R, R)
    if sol is None:
        pytest.skip("no CLC at this radius")
    if not _is_direct(sol):
        pytest.skip("long-way solution -- reconstruction helper is direct-only")
    miss = _reconstruct(p1, v1, p4, v4, R, R, float(sol["beta"]),
                        float(sol["alpha1"]), float(sol["alpha2"]))
    assert miss < 1e-3 * max(throw, 1.0), (
        f"throw {throw} dturn {dturn} DLS {dls}: reconstruction miss "
        f"{miss:.4g} m (md {float(sol['total_md']):.1f})"
    )


@pytest.mark.parametrize("throw,dturn,dls", GRID)
def test_direct_root_preferred_when_one_reconstructs(throw, dturn, dls):
    """If a DIRECT root that reconstructs the target exists, take a direct root.

    Self-consistent oracle: enumerate the accepted roots and check the
    selection. This is the regression that pins the near-straight defect -- the
    30 m / 7230 m case had a reconstructing direct root available and returned
    the loop instead.
    """
    p1, v1, p4, v4 = _case(throw, dturn)
    R = np.degrees(30.0) / dls
    every = solve_clc(p1, v1, p4, v4, R, R, return_all=True)
    if not every:
        pytest.skip("no CLC at this radius")
    tol = 1e-3 * max(throw, 1.0)
    good_direct = [
        s for s in every
        if _is_direct(s)
        and _reconstruct(p1, v1, p4, v4, R, R, float(s["beta"]),
                         float(s["alpha1"]), float(s["alpha2"])) < tol
    ]
    if not good_direct:
        pytest.skip("no reconstructing direct root at this radius (physics)")
    chosen = solve_clc(p1, v1, p4, v4, R, R)
    assert _is_direct(chosen), (
        f"throw {throw} dturn {dturn} DLS {dls}: a direct root exists "
        f"(md {min(float(s['total_md']) for s in good_direct):.1f}) but the "
        f"solver returned a loop (md {float(chosen['total_md']):.1f})"
    )


@pytest.mark.parametrize("throw,dturn", [
    (30.0, 0.5),
    (30.0, 1.18),
    pytest.param(100.0, 0.5, marks=pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN REMAINING GAP in the same class as 9e103bb. At L=100, "
            "R=1719 (DLS 1) the direct root (md 99.7, ~= the 100 m throw) has "
            "reconstruction residual 0.4302 against the radius-scaled bound "
            "1e-4*(L+2R) = 0.3438 -- rejected by 25%, so a loop (md 10899.8) "
            "is returned even though DLS 0.75 and DLS 3 both return direct. "
            "The residual floor grows faster than the bound does, because the "
            "degree-10 root-find is ill-conditioned at large R/L. Loosening "
            "the constant further would be fitting to this case; the "
            "principled fix is a joint Newton polish of (beta, alpha1, alpha2) "
            "against the reconstruction equations, which collapses the "
            "residual and makes any reasonable bound sufficient. (A Newton "
            "polish of beta ALONE does not help -- measured 1.0x -- because "
            "the residual is dominated by the angle-recovery branch.) "
            "strict=True so this flips loudly when that lands."
        ),
    )),
    (100.0, 1.0),
    (300.0, 0.5),
])
def test_direct_solution_persists_as_the_radius_tightens(throw, dturn):
    """Once a direct solution appears, tightening the radius must not lose it.

    The physically-sound half of the monotonicity criterion. (The converse is
    NOT required: at a gentle enough limit the direct solution genuinely ceases
    to exist -- turning 1.18 deg at R=1719 m needs 35.4 m of arc against a 30 m
    throw -- so the root set may legitimately shrink as R grows.)
    """
    p1, v1, p4, v4 = _case(throw, dturn)
    seen_direct_at = None
    for dls in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 40.0):
        R = np.degrees(30.0) / dls
        sol = solve_clc(p1, v1, p4, v4, R, R)
        if sol is None:
            continue
        if _is_direct(sol):
            seen_direct_at = dls
        elif seen_direct_at is not None:
            pytest.fail(
                f"throw {throw} dturn {dturn}: direct at DLS {seen_direct_at} "
                f"but a loop at the TIGHTER DLS {dls} "
                f"(md {float(sol['total_md']):.1f})"
            )


def test_the_field_case_is_a_short_connection():
    """The reported case: 30 m throw, 1.18 deg tangent change, DLS 3 -> ~30 m.

    Returned 7230 m (240x) before 9e103bb.
    """
    p1 = np.array([-105.4, -495.4, 582.0])
    v1 = np.array([0.0, 0.0, 1.0])
    p4 = np.array([-105.5, -495.7, 612.0])
    v4 = np.array([-0.01, -0.018, 1.0])
    v4 = v4 / np.linalg.norm(v4)
    R = np.degrees(30.0) / 3.0
    sol = solve_clc(p1, v1, p4, v4, R, R)
    assert sol is not None
    assert float(sol["total_md"]) == pytest.approx(30.0, abs=0.5)
    assert _is_direct(sol)


# ---------------------------------------------------------------------------
# Scene-scaled coverage.
#
# The grid above is ANCHORED: one origin, one base attitude, target exactly
# along v1, zero lateral offset. That is not what a caller submits -- an online
# CLC endpoint or a planner hands over arbitrary poses anywhere in space. A CLC
# solve is pure geometry and must be ambivalent to scene placement, so these
# cases vary position, attitude and lateral offset.
#
# Endpoint fidelity is deliberately NOT re-tested here: it is already covered
# through welleng's own renderer by
# test_connector_analytical.test_random_pose_battery_no_spurious_rejection
# (which is what caught an incorrect arc normalisation at 86 cases). These
# tests cover the properties that battery does NOT: scene invariance, direct
# preference, and the near-straight long-way rate.
# ---------------------------------------------------------------------------

def _scene_case(rng):
    """A near-straight pose pair anywhere in a scene, any attitude, WITH lateral
    offset -- i.e. what an online CLC caller or a planner actually submits."""
    inc0 = rng.uniform(0.5, 92.0)
    azi0 = rng.uniform(0.0, 360.0)
    v1 = _uv(inc0, azi0)
    v4 = _uv(np.clip(inc0 + rng.uniform(-2.5, 2.5), 0.05, 179.0),
             azi0 + rng.uniform(-2.5, 2.5))
    throw = rng.uniform(20.0, 150.0)
    p1 = rng.normal(scale=2000.0, size=3) + np.array([0.0, 0.0, 2500.0])
    lat = np.cross(v1, [0.0, 0.0, 1.0])
    lat = (np.array([1.0, 0.0, 0.0]) if np.linalg.norm(lat) < 1e-9
           else lat / np.linalg.norm(lat))
    return p1, v1, p1 + throw * v1 + rng.uniform(-5.0, 5.0) * lat, v4


def _rand_rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


@pytest.mark.parametrize("seed", range(12))
def test_solution_is_invariant_to_scene_placement(seed):
    """Rigidly moving the whole problem must not change the solution.

    Translating and rotating (p1, v1, p4, v4) together must leave beta, both arc
    angles and the measured depth unchanged -- pure geometry has no preferred
    origin or axis. Guards against origin-dependence creeping into the closed
    form: the bare Sawaryn form drops the p1 offset and is valid only at the
    origin, which is why connector._solve_chc_analytical carries an explicit
    "dp = pos_target - pos1 keeps this frame-free" note.
    """
    rng = np.random.default_rng(seed)
    p1, v1, p4, v4 = _scene_case(rng)
    R = np.degrees(30.0) / 3.0
    base = solve_clc(p1, v1, p4, v4, R, R)
    if base is None:
        pytest.skip("no CLC for this pair")
    scale = max(float(base["total_md"]), 1.0)
    for _ in range(3):
        t, q = rng.normal(scale=5000.0, size=3), _rand_rot(rng)
        moved = solve_clc(q @ p1 + t, q @ v1, q @ p4 + t, q @ v4, R, R)
        assert moved is not None, "solvable in place but not when moved"
        assert abs(float(moved["beta"]) - float(base["beta"])) < 1e-6 * scale
        assert abs(float(moved["total_md"])
                   - float(base["total_md"])) < 1e-6 * scale
        for k in ("alpha1", "alpha2"):
            # rotation round-off, absolute, on angles up to 2*pi
            assert abs(float(moved[k]) - float(base[k])) < 1e-6


@pytest.mark.parametrize("seed", range(24))
def test_scene_direct_root_preferred_over_a_loop(seed):
    """Anywhere in the scene: if an accepted direct root of sane length exists,
    the solver must not return a long-way one instead."""
    rng = np.random.default_rng(1000 + seed)
    p1, v1, p4, v4 = _scene_case(rng)
    R = np.degrees(30.0) / 3.0
    every = solve_clc(p1, v1, p4, v4, R, R, return_all=True)
    if not every:
        pytest.skip("no CLC for this pair")
    L = float(np.linalg.norm(p4 - p1))
    sane_direct = [s for s in every
                   if _is_direct(s) and float(s["total_md"]) < 1.5 * L]
    if not sane_direct:
        pytest.skip("no sane direct root accepted at this radius")
    chosen = solve_clc(p1, v1, p4, v4, R, R)
    assert _is_direct(chosen), (
        f"seed {seed}: direct root available (md "
        f"{min(float(s['total_md']) for s in sane_direct):.1f}, L {L:.1f}) but "
        f"got a loop (md {float(chosen['total_md']):.1f})"
    )


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN GAP, same class as the L=100/R=1719 case above and MUCH broader than "
    "the anchored grid suggested. Over 300 scene-realistic near-straight pairs "
    "(any attitude, any scene position, 20-150 m throw, <=2.5 deg tangent "
    "change, <=5 m lateral) the long-way rate is 47.0% at DLS 3 (28.7% at DLS 6, "
    "17.3% at DLS 10), worst md/L 167x. These are a planner's ordinary local "
    "connections, so the residual-vs-bound gap is not a corner case. Contrast "
    "welleng-pathfinder's 400-pair battery at R/L ~ 0.5-2.9, where an 87.8% loop "
    "rate is GENUINE geometry and unchanged by 9e103bb: loops are real in tight "
    "geometry and largely artefact in near-straight geometry. Fix is the joint "
    "Newton polish of (beta, alpha1, alpha2) noted above."
))
def test_near_straight_scene_loop_rate_is_low():
    """Long-way solutions should be rare when the poses are nearly collinear."""
    rng = np.random.default_rng(7)
    R = np.degrees(30.0) / 3.0
    solved = loops = 0
    for _ in range(300):
        p1, v1, p4, v4 = _scene_case(rng)
        try:
            sol = solve_clc(p1, v1, p4, v4, R, R)
        except Exception:
            continue
        if sol is None:
            continue
        solved += 1
        if not _is_direct(sol):
            loops += 1
    assert solved > 200
    assert loops / solved < 0.05, (
        f"{loops}/{solved} ({100 * loops / solved:.1f}%) near-straight scene "
        "pairs returned a long-way solution"
    )
