"""Analytic closest-point for IscwsaClearance (feat/iscwsa-analytic-closest-point).

The per-station scipy Powell search for the closest point on the offset well
(the ~85% hot path of IscwsaClearance) is replaced by a closed-form point-to-arc
solve: a min-curvature segment is a planar circular arc, so the closest point is
analytic. These tests pin the kernel against the scipy search it replaces,
including the off-arc (endpoint) case, and the frame it must operate in.

The end-to-end ISCWSA reference-value validation (incl. the datum-shifted
well 10) lives in tests/test_clearance_iscwsa.py and is the integration reference.
"""
import numpy as np
import pytest
from scipy import optimize

from welleng.survey import Survey, SurveyHeader, _interpolate_pos_nev
from welleng.clearance import _closest_x_on_arc


def _offset(n=40, shift=8.0):
    md = np.linspace(0, 30 * (n - 1), n)
    inc = np.clip(np.linspace(0, 90, n), 0, 60)
    azi = (np.linspace(0, 120, n) + shift) % 360
    return Survey(md=md, inc=inc, azi=azi, header=SurveyHeader())


def _local(survey):
    """Positions + unit tangents in the local (n, e, tvd) frame the clearance
    and _interpolate_pos_nev use."""
    pos = np.column_stack([survey.n, survey.e, survey.tvd])
    inc = np.asarray(survey.inc_rad, float)
    azi = np.asarray(survey.azi_grid_rad, float)
    tan = np.column_stack([
        np.sin(inc) * np.cos(azi), np.sin(inc) * np.sin(azi), np.cos(inc)])
    return pos, tan


def _scipy_x(survey, seg, Q):
    b = survey.md[seg + 1] - survey.md[seg]
    res = optimize.minimize(
        lambda x: np.linalg.norm(_interpolate_pos_nev(survey, x[0], seg) - Q),
        [b / 2], method='Powell', bounds=[(0, b)])
    return float(res.x[0]), float(res.fun)


def test_kernel_matches_scipy_and_is_never_worse():
    off = _offset()
    pos, tan = _local(off)
    rng = np.random.default_rng(0)
    max_dx = 0.0
    for seg in range(len(off.md) - 1):
        segmd = off.md[seg + 1] - off.md[seg]
        for _ in range(8):
            Q = pos[seg] + rng.normal(scale=60, size=3)
            xa = _closest_x_on_arc(pos[seg], tan[seg], tan[seg + 1],
                                   segmd, off.dogleg[seg + 1], Q)
            da = np.linalg.norm(_interpolate_pos_nev(off, xa, seg) - Q)
            xs, ds = _scipy_x(off, seg, Q)
            max_dx = max(max_dx, abs(xa - xs))
            # analytic is the true minimum -> never a worse distance than scipy
            assert da <= ds + 1e-6
    assert max_dx < 1e-3   # agree to well below scipy's own tolerance


def test_endpoint_case_picks_true_nearest_end():
    # A point beyond the end of a curved segment: the closest arc point is the
    # END (x = seg_md), and a naive atan2-clamp can land on the wrong end.
    off = _offset(n=6)
    pos, tan = _local(off)
    seg = 2
    segmd = off.md[seg + 1] - off.md[seg]
    # place Q far past the segment end along the end tangent
    Q = pos[seg + 1] + 500.0 * tan[seg + 1]
    xa = _closest_x_on_arc(pos[seg], tan[seg], tan[seg + 1], segmd,
                           off.dogleg[seg + 1], Q)
    xs, _ = _scipy_x(off, seg, Q)
    assert abs(xa - xs) < 1e-2
    assert xa > segmd * 0.9      # at/near the far end, not clamped to 0


def test_straight_segment_projects_onto_tangent():
    P0 = np.zeros(3)
    t = np.array([0.0, 0.0, 1.0])
    Q = np.array([5.0, 0.0, 12.0])
    x = _closest_x_on_arc(P0, t, t, 30.0, 0.0, Q)   # dogleg = 0 -> straight
    assert np.isclose(x, 12.0)                      # projection along tangent
    # clamped to the segment
    assert _closest_x_on_arc(P0, t, t, 30.0, 0.0, np.array([0., 0., 99.])) == 30.0
    assert _closest_x_on_arc(P0, t, t, 30.0, 0.0, np.array([0., 0., -5.])) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
