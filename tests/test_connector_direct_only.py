"""Connector ``direct_only``: solver success is not a DLS feasibility test.

Since PR #305 the renderer handles arcs of any angle, so a long-way (>pi) arc
is a valid CLC at essentially any radius. Consequently a search of the form
"does a CLC exist at this DLS" succeeds at almost ANY DLS -- returning a
multi-kilometre corkscrew for a short pose-to-pose move. ``direct_only=True``
rejects those, restoring a meaningful reachability predicate.

Regression for the downstream collapse this caused in a consumer's
``required_dls`` bisection (it returned its floor for ~every pose pair, so a
feasibility filter certified corkscrews as drillable).
"""

import numpy as np
import pytest

import welleng as we

POS1 = np.array([300., -200., 1400.])
POS2 = np.array([700., 100., 2000.])


def _uv(inc, azi):
    i, a = np.radians(inc), np.radians(azi)
    return np.array([np.sin(i) * np.cos(a), np.sin(i) * np.sin(a), np.cos(i)])


VEC1, VEC2 = _uv(20., 320.), _uv(88., 300.)
# the direct requirement for this pose pair (both arcs first <= pi) ~6.87 deg/30m
DLS_DIRECT = 6.88


def _connect(dls, direct_only):
    return we.connector.Connector(
        pos1=POS1, vec1=VEC1, pos2=POS2, vec2=VEC2,
        dls_design=dls, direct_only=direct_only,
    )


def test_loop_solution_exists_at_absurdly_low_dls():
    """Default behaviour (unchanged): a >pi loop solves at a tiny DLS."""
    c = _connect(0.5, False)
    straight = np.linalg.norm(POS2 - POS1)
    assert c.md_target > 10 * straight            # a corkscrew, not a path
    assert max(abs(c.dogleg), abs(c.dogleg2)) > np.pi


@pytest.mark.parametrize("dls", [0.5, 3.0, 6.0])
def test_direct_only_rejects_the_loop(dls):
    with pytest.raises(ValueError, match="DIRECT"):
        _connect(dls, True)


@pytest.mark.parametrize("dls", [DLS_DIRECT, 8.0, 10.0])
def test_direct_only_accepts_a_genuine_direct_solution(dls):
    c = _connect(dls, True)
    assert abs(c.dogleg) <= np.pi + 1e-9
    assert abs(c.dogleg2) <= np.pi + 1e-9
    # a direct solve is a sane length, not a multi-km loop
    assert c.md_target < 2.0 * np.linalg.norm(POS2 - POS1)


def test_direct_only_does_not_change_an_accepted_solution():
    """The guard only rejects; it must not perturb the geometry it accepts."""
    a = _connect(8.0, False)
    b = _connect(8.0, True)
    assert a.md_target == pytest.approx(b.md_target, rel=0, abs=0)
    assert np.allclose(a.pos_target, b.pos_target, rtol=0, atol=0)


def test_bisection_is_meaningful_again():
    """The consumer-facing point: an ungated DLS bisection collapses to its
    floor; the gated one recovers the true direct requirement."""
    def required(direct_only, lo=0.5, hi=10.0, tol=0.01):
        def ok(d):
            try:
                _connect(d, direct_only)
                return True
            except ValueError:
                return False
        assert ok(hi)
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if ok(mid):
                hi = mid
            else:
                lo = mid
        return hi

    assert required(False) < 0.6                       # collapsed to the floor
    assert required(True) == pytest.approx(DLS_DIRECT, abs=0.05)
