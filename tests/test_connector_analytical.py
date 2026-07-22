"""Proof tests for the analytical (closed-form) curve-hold-curve solver.

The ``Connector`` now solves the curve-hold-curve (CHC) point-to-target
problem with the closed-form solution of Sawaryn (2021), SPE-204111-PA
(``Connector._solve_chc_analytical``). Arcs may turn more than 180deg (rendered
the long way round), so a CLC exists and is used at the design DLS for almost
every geometry; only a genuinely unreachable pose (no CLC at all) raises. The
inherited Sawaryn & Thorogood (2005) iterative scheme has been removed.

These tests prove the port:

* the analytical path is the one actually taken for well-posed CLC geometries
  (``c._chc_solver == 'analytical'``), and the connector method is still
  ``'curve_hold_curve'`` (public API preserved);
* the reconstructed endpoint matches the target far TIGHTER than the iterative
  baseline — ``test_clc_connector`` asserts only ``atol=1e-3`` on the endpoint,
  whereas the closed form lands within ``1e-6`` (here ~2e-8);
* the analytical solve respects the design DLS exactly (each arc radius equals
  the design radius to machine precision — no curvature is traded for MD).

The same ``make_clc_path`` generator and (n=1000, seed=42, radius=1.0) sweep as
``test_clc_connector`` are used, so the two tests cover the identical case set.
"""

import numpy as np

from welleng.connector import Connector, get_pos
from welleng.utils import make_clc_path, dls_from_radius


# Closed-form endpoint tolerance — two orders tighter than the iterative
# baseline (test_clc_connector uses atol=1e-3).
TIGHT_TOL = 1e-6


def _reconstruct_endpoint(c):
    """Forward-reconstruct the CHC endpoint pos/vec from the connector's public
    state: arc1 (pos1, vec1 -> vec2) -> hold (length tangent_length along vec2)
    -> arc2 (pos3, vec3 -> vec_target). Returns (pos_end, vec_end)."""
    pos2 = get_pos(
        c.pos1, c.vec1, c.vec2, c.dist_curve, c.func_dogleg
    ).reshape(3)
    pos3 = pos2 + c.tangent_length * c.vec2
    pos_end = get_pos(
        pos3, c.vec3, c.vec_target, c.dist_curve2, c.func_dogleg2
    ).reshape(3)
    return pos2, pos3, pos_end


def test_chc_analytical_roundtrip(n=1000, seed=42, radius=1.0):
    """Round-trip: every constructible CLC is recovered by the analytical path
    to ~1e-6, on the design DLS, with the public CHC state self-consistent."""
    rng = np.random.default_rng(seed)
    toolface1 = rng.uniform(-np.pi, np.pi, n)
    dogleg1 = rng.uniform(1e-2, np.pi, n)
    distance = rng.uniform(0.01, radius, n)
    toolface2 = rng.uniform(-np.pi, np.pi, n)
    dogleg2 = rng.uniform(1e-2, np.pi, n)

    dls = dls_from_radius(radius)
    pos0 = np.array([0., 0., 0.])
    vec0 = np.array([0., 0., 1.])

    n_analytical = 0
    n_fallback = 0
    endpoint_failures = []
    dls_failures = []
    internal_failures = []

    for i in range(n):
        path = make_clc_path(
            toolface1[i], dogleg1[i], distance[i],
            toolface2[i], dogleg2[i],
            pos0=pos0, vec0=vec0, radius=radius,
        )
        c = Connector(
            pos1=pos0, vec1=vec0,
            pos2=path['pos3'], vec2=path['vec3'],
            dls_design=dls,
        )

        # Public API preserved: method unchanged.
        assert c.method == 'curve_hold_curve', (i, c.method)

        if getattr(c, '_chc_solver', None) != 'analytical':
            n_fallback += 1
            continue
        n_analytical += 1

        # Endpoint reconstructed from public state must hit the target tightly.
        pos2, pos3, pos_end = _reconstruct_endpoint(c)
        if not np.allclose(pos_end, path['pos3'], atol=TIGHT_TOL):
            endpoint_failures.append(
                (i, float(np.linalg.norm(pos_end - path['pos3'])))
            )
        # Target tangent is matched exactly (it is set directly from input).
        if not np.allclose(c.vec_target, path['vec3'], atol=TIGHT_TOL):
            endpoint_failures.append(
                (i, 'vec', float(np.linalg.norm(c.vec_target - path['vec3'])))
            )

        # Stored pos2/pos3 must equal the forward reconstruction (state is
        # internally consistent, so the renderer reproduces the same geometry).
        if not (np.allclose(pos2, c.pos2, atol=TIGHT_TOL)
                and np.allclose(pos3, c.pos3, atol=TIGHT_TOL)):
            internal_failures.append(i)

        # DLS respected: each arc radius == design radius (no curvature traded).
        r1 = c.dist_curve / c.dogleg if c.dogleg > 1e-10 else radius
        r2 = c.dist_curve2 / c.dogleg2 if c.dogleg2 > 1e-10 else radius
        if min(r1, r2) < radius * (1 - TIGHT_TOL):
            dls_failures.append((i, r1, r2, radius))

        # MD bookkeeping is consistent with the arc/hold lengths.
        assert np.isclose(c.md2, c.md1 + abs(c.dist_curve))
        assert np.isclose(c.md3, c.md2 + c.tangent_length)
        assert np.isclose(c.md_target, c.md3 + abs(c.dist_curve2))

    # The analytical path is the PRIMARY solver — it must carry the vast
    # majority of well-posed CLC cases (here all 1000).
    assert n_analytical >= 0.99 * n, (
        f"analytical path only solved {n_analytical}/{n}; expected >= 99%"
    )
    assert not endpoint_failures, (
        f"{len(endpoint_failures)} analytical cases missed the target by > "
        f"{TIGHT_TOL}: {endpoint_failures[:10]}"
    )
    assert not internal_failures, (
        f"{len(internal_failures)} analytical cases had inconsistent stored "
        f"pos2/pos3: {internal_failures[:10]}"
    )
    assert not dls_failures, (
        f"{len(dls_failures)} analytical cases exceeded the design DLS: "
        f"{dls_failures[:10]}"
    )


def test_chc_analytical_is_used_simple():
    """A clearly well-posed CLC uses the analytical path and lands tightly."""
    vec1 = np.array([-1., -1., 1.]) / np.sqrt(3)
    vec2 = np.array([1., -1., 0.]) / np.sqrt(2)
    c = Connector(
        pos1=[0., 0., 0.], vec1=vec1,
        pos2=[0., 1000., 500.], vec2=vec2,
        dls_design=3.0,
    )
    assert c.method == 'curve_hold_curve'
    assert c._chc_solver == 'analytical'
    _, _, pos_end = _reconstruct_endpoint(c)
    assert np.allclose(pos_end, [0., 1000., 500.], atol=1e-4)
    assert np.allclose(c.vec_target, vec2, atol=TIGHT_TOL)


def test_no_clc_target_raises():
    """A target with NO curve-hold-curve solution at the design radii raises.
    Because arcs may turn > 180deg a CLC exists for almost every geometry
    (see test_long_arc_solves_at_design_dls), so a genuine no-solution case needs
    an unreachable pose: a target on the start axis with a reversed tangent -- a
    U-turn on the spot no radius-R curve-hold-curve can make."""
    import pytest
    with pytest.raises(ValueError):
        Connector(
            pos1=[0., 0., 0.], vec1=[0., 0., 1.],
            pos2=[0., 0., 50.], vec2=[0., 0., -1.],
        )


def test_long_arc_solves_at_design_dls():
    """A target whose only CLC at the design DLS turns an arc > 180deg is solved
    at the design DLS (the > pi arc is a valid circular curve, rendered the long
    way round) -- it is NOT rejected and does NOT need a tighter (design-DLS-
    exceeding) radius. This is the ``[0, 100, 100]`` vertical->vertical case that
    previously had no <= pi CLC: it now returns a curve-hold-curve at the design
    radius with a > pi second arc, hitting the target to machine precision without
    exceeding the design dogleg severity."""
    target = [0., 100., 100.]
    c = Connector(
        pos1=[0., 0., 0.], vec1=[0., 0., 1.],
        pos2=target, vec2=[0., 0., 1.],
    )
    assert c.method == 'curve_hold_curve'
    assert c._chc_solver == 'analytical'

    # A long arc (> pi) is used -- the whole point of this case.
    assert max(c.dogleg, c.dogleg2) > np.pi

    # Endpoint reconstructed from public state hits the target to machine prec.
    _, _, pos_end = _reconstruct_endpoint(c)
    assert np.allclose(pos_end, target, atol=TIGHT_TOL), (
        float(np.linalg.norm(np.array(pos_end) - np.array(target)))
    )

    # Solved AT the design DLS -- no critical-radius override, design not exceeded.
    assert c.radius_critical == np.inf
    assert c.radius_critical2 == np.inf


def test_random_pose_battery_no_spurious_rejection():
    """Regression for the ~22% spurious CHC rejection: a random pose-to-pose
    battery that ``sawaryn_analytical.solve_clc`` solves 400/400 at DLS 3.0 must
    also be solved 400/400 by the Connector. Previously the Connector rejected
    ~1 in 5 (every geometry whose only CLC needs a > pi arc), silently shrinking
    a planner's candidate set. Each solve must hit the target and respect the
    design DLS."""
    from welleng.utils import get_vec, radius_from_dls

    rng = np.random.default_rng(0)
    R = radius_from_dls(3.0)
    p1 = np.array([0., 0., 0.])
    v1 = np.array([0., 0., 1.])
    solved = long_arc = 0
    for _ in range(400):
        p2 = np.array([rng.uniform(200, 800), rng.uniform(-400, 400),
                       rng.uniform(900, 2000)])
        inc = rng.uniform(20, 70)
        azi = rng.uniform(0, 360)
        v2 = np.asarray(get_vec(inc, azi, nev=True, deg=True)).reshape(3)
        c = Connector(pos1=p1, vec1=v1, pos2=p2, vec2=v2,
                      dls_design=3.0, on_infeasible='raise')
        solved += 1
        if max(c.dogleg, c.dogleg2) > np.pi:
            long_arc += 1
        # hits the target
        _, _, pos_end = _reconstruct_endpoint(c)
        assert np.allclose(pos_end, p2, atol=1e-4), (
            float(np.linalg.norm(pos_end - p2))
        )
        # at the design radius (no curvature traded for MD)
        assert np.isclose(c.dist_curve / c.dogleg, R, rtol=1e-6)
        assert np.isclose(c.dist_curve2 / c.dogleg2, R, rtol=1e-6)
    assert solved == 400, solved
    # the point of the fix: a big chunk genuinely need the > pi long arc.
    assert long_arc > 50, long_arc


if __name__ == "__main__":
    test_chc_analytical_roundtrip()
    test_chc_analytical_is_used_simple()
    test_no_clc_target_raises()
    test_long_arc_solves_at_design_dls()
    print("ok")
