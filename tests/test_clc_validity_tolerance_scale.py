"""CLC validity tolerance must scale with the RADIUS, not just the separation.

`_clc_solutions` accepts a root when its forward-reconstruction residual is
small. That residual is built from R-scale terms::

    f1 = R1*sin(a1) + b*cos(a1) + R2*tan(a2/2)*(mu + cos(a1))

so comparing it against an L-scaled bound (``1e-4 * L``, L = |p4 - p1|) fails
whenever R >> L: the bound falls below the residual's own floor and the CORRECT
root is discarded, leaving a long-way (>pi) root as the "shortest valid"
solution. Reported by a downstream consumer from a real relief-well sweep, where a
30 m near-collinear move at DLS 3 returned a 7230 m double-loop (240x).

NB the arcs of the surviving long-way root are genuine >pi arcs, not a wrap
artefact -- forcing all arcs to <=pi breaks endpoint reconstruction on 86 cases
of the random pose battery. The defect is purely the acceptance bound.
"""

import numpy as np

from welleng.sawaryn_analytical import solve_clc

# poses from the real sweep: 30 m apart, tangents 1.18 deg apart
P1 = np.array([-105.4, -495.4, 582.0])
V1 = np.array([0.0, 0.0, 1.0])
P2 = np.array([-105.5, -495.7, 612.0])
V2 = np.array([-0.01, -0.018, 1.0]) / np.linalg.norm([-0.01, -0.018, 1.0])
L = float(np.linalg.norm(P2 - P1))


def _solve(dls):
    R = np.degrees(30.0) / dls
    return solve_clc(P1, V1, P2, V2, R, R)


def test_direct_root_found_for_near_collinear_move():
    """A 30 m throw with a 1.18 deg turn is a ~30 m CLC at DLS 3, not 7230 m."""
    s = _solve(3.0)
    assert s is not None
    assert float(s["total_md"]) < 1.5 * L
    assert np.degrees(float(s["alpha1"])) < 5.0
    assert np.degrees(float(s["alpha2"])) < 5.0


def test_no_ratio_artefact_across_the_feasible_range():
    """Acceptance must not flip on the R/L ratio.

    At R where the arc needed to turn the tangents still fits inside the throw,
    the direct solution exists and must be returned. (At much gentler DLS it
    genuinely does NOT exist -- turning 1.18 deg at R=1719 m takes 35.4 m of
    arc, more than the 30 m throw -- so the root set legitimately shrinks
    there; that is physics, not a defect.)
    """
    for dls in (3.0, 4.0, 4.219024, 4.219025, 6.0, 10.0):
        s = _solve(dls)
        assert s is not None, f"no solution at DLS {dls}"
        md = float(s["total_md"])
        assert md < 1.5 * L, f"DLS {dls}: got {md:.1f} m for a {L:.1f} m throw"


def test_tolerance_is_radius_scaled_not_separation_scaled():
    """Regression on the bound itself: the old 1e-4*L rejected a res~3.2e-2
    root at R=573 (bound 3e-3). Assert the accepted solution reconstructs."""
    R = np.degrees(30.0) / 3.0
    s = solve_clc(P1, V1, P2, V2, R, R)
    beta = float(s["beta"])
    a1, a2 = float(s["alpha1"]), float(s["alpha2"])
    # arc + hold + arc must account for the throw
    assert beta > 0
    assert abs(R * a1 + beta + R * a2 - float(s["total_md"])) < 1e-6
