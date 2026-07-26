"""The degenerate-coplanar CLC boundary: a single arc plus a hold (alpha2 = 0).

A planner hits this by construction: it spawns a child pose-to-POINT (arrival
tangent unbound), stores the returned tangent as that node's pose, then
re-solves the chain pose-to-POSE from the stored poses. Every such leg sits
exactly on the boundary -- the two tangents and the displacement are coplanar
and the second arc is exactly zero.

Two independent failures met there, both reported as "solvable as pose-to-point,
refused as pose-to-pose with its own arrival tangent":

1. The arc quadratic is TANGENT at alpha2 = 0, so its discriminant's true value
   is 0 and it is formed as a difference of two ~4e24 terms -- measured -5.4e08,
   i.e. -1.2e-16 relative. An absolute ``disc < 0`` test read that as infeasible
   and dropped BOTH roots.
2. Once admitted, the zero turn comes back as a tiny NEGATIVE angle, which
   ``% (2*pi)`` mapped to a full loop -- reporting an 822 m path as 7094 m.
"""

import numpy as np
import pytest

from welleng.sawaryn_analytical import solve_clc

AZIS = (-9.46, 0.0, 100.0, 250.0)
TURNS = (2.0, 5.0, 10.0, 24.1837, 45.0, 60.0, 80.0, 120.0, 170.0)


def _single_arc_then_hold(R, alpha1, hold, azi):
    """Pose pair whose exact connection is one arc of ``alpha1`` then a hold."""
    p1 = np.zeros(3)
    t1 = np.array([0.0, 0.0, 1.0])
    n = np.array([np.cos(azi), np.sin(azi), 0.0])
    t4 = np.cos(alpha1) * t1 + np.sin(alpha1) * n
    p4 = R * np.sin(alpha1) * t1 + R * (1 - np.cos(alpha1)) * n + hold * t4
    return p1, t1, p4, t4


@pytest.mark.parametrize("turn", TURNS)
@pytest.mark.parametrize("R", (300.0, 1000.0, 2500.0))
@pytest.mark.parametrize("hold", (10.0, 50.0, 400.0, 2000.0))
def test_boundary_leg_is_solved_exactly(turn, R, hold):
    a1 = np.radians(turn)
    s = solve_clc(*_single_arc_then_hold(R, a1, hold, np.radians(-9.46)), R)
    assert s is not None
    # A tangency is intrinsically half-precision in the root, so the recovered
    # parameters carry ~2e-08 rad / ~1e-07 relative -- sub-mm on the path.
    assert s["total_md"] == pytest.approx(R * a1 + hold, rel=1e-6)
    assert s["alpha1"] == pytest.approx(a1, abs=1e-6)
    assert s["alpha2"] == pytest.approx(0.0, abs=1e-6)   # NOT 2*pi
    assert s["beta"] == pytest.approx(hold, rel=1e-6)


@pytest.mark.parametrize("azi", AZIS)
def test_boundary_is_plane_independent(azi):
    R, a1, hold = 1000.0, np.radians(24.1837), 400.0
    s = solve_clc(*_single_arc_then_hold(R, a1, hold, np.radians(azi)), R)
    assert s is not None
    assert s["total_md"] == pytest.approx(R * a1 + hold, rel=1e-9)


def test_zero_turn_never_reported_as_a_full_loop():
    """The specific regression: alpha2 = 0 co-terminal with 2*pi.

    tan(alpha/2) cannot tell them apart and the closure equations are satisfied
    identically by both, so the physical (shorter) representative must be
    returned -- otherwise a full loop of arc is silently added to the md.
    """
    R = 2500.0
    s = solve_clc(
        *_single_arc_then_hold(R, np.radians(45.0), 100.0, 0.0), R
    )
    assert s is not None
    assert s["alpha2"] < np.pi
    assert s["total_md"] < R * np.radians(45.0) + 100.0 + 1.0


def test_pose_to_point_result_restated_as_a_pose_still_solves():
    """End-to-end statement of the consumer's failure, via Connector."""
    from welleng.connector import Connector

    p1, v1 = [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    target = [300.0, -50.0, 800.0]
    c1 = Connector(pos1=p1, vec1=v1, pos2=target, dls_design=3.0)
    c2 = Connector(
        pos1=p1, vec1=v1, pos2=target, vec2=c1.vec_target, dls_design=3.0
    )
    assert c2.md_target == pytest.approx(c1.md_target, rel=1e-6)


def test_forward_accepts_every_exactly_coplanar_triple():
    """``forward`` must gate the Gram determinant the same way the solvers do.

    A coplanar triple (``mu = cos(alpha1 + alpha2)``) has determinant exactly 0,
    so it comes back either side of zero -- 8567 of the 57600 triples below
    evaluate NEGATIVE, down to -2.0e-15. An absolute ``surd < 0`` test rejected
    every one of them, which is how ``max_radius`` lost a genuine radius on a
    planar pose (a batch consumer, ``|eta14|/L = 8.8e-17``, all four branches
    refused at -5.5e-14). Aligning the test on ``_GRAM_TOL`` also took the
    collinear batteries' median closure from 8.0e-09 to 4.2e-14.
    """
    from welleng.sawaryn_analytical import forward

    grid = np.radians(np.linspace(1.0, 120.0, 240))
    rejected = [
        (a1, a2)
        for a1 in grid
        for a2 in grid
        if forward(a1, a2, 500.0, np.cos(a1 + a2), 1000.0, 1000.0) is None
    ]
    assert not rejected, (
        f"{len(rejected)} coplanar triples refused, e.g. {rejected[:5]}")


@pytest.mark.xfail(
    reason="OPEN: near-straight regime (turn < ~1 deg with R/L >> 1). Eq 34's "
           "biquadratic loses the true beta root to cancellation in K, so the "
           "solver returns a long-way arc instead. TWO consumer-visible faces, "
           "which need different detection: ungated it is a CORKSCREW (true md "
           "404.4, returned 3556); under direct_only=True the loop is refused "
           "and it surfaces as a false 'unreachable' None (a downstream consumer). "
           "Pre-existing; not the coplanar-boundary defect this module covers.",
    strict=True,
)
def test_near_straight_boundary_leg():
    R, a1, hold = 500.0, np.radians(0.5), 400.0
    s = solve_clc(*_single_arc_then_hold(R, a1, hold, 0.0), R)
    assert s is not None
    assert s["total_md"] == pytest.approx(R * a1 + hold, rel=1e-6)
