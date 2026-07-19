"""arc_inc_azi_extrema — exact inclination/azimuth extrema over min-curvature arcs.

Verified against dense sampling. The two results pinned: inclination extrema at
the closed-form critical points, and
azimuth strictly monotonic along a circular arc (extrema = endpoints, signed
swing exact including reflex arcs and multi-wrap).
"""
import numpy as np
import pytest

from welleng.utils import arc_inc_azi_extrema


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _make_arcs(n, seed, amax=2 * np.pi * 0.999):
    rng = np.random.default_rng(seed)
    va = _unit(rng.normal(size=(n, 3)))
    nhat = _unit(np.cross(va, rng.normal(size=(n, 3))))
    u = _unit(np.cross(nhat, va))
    alpha = rng.uniform(0.02, amax, n)
    vb = va * np.cos(alpha)[:, None] + u * np.sin(alpha)[:, None]
    return va, vb, u, alpha


def test_matches_dense_sampling():
    va, vb, u, alpha = _make_arcs(3000, seed=1)
    r = arc_inc_azi_extrema(va, vb, alpha)
    e_inc = e_swing = 0.0
    for i in range(len(alpha)):
        if abs(np.sin(alpha[i])) < 1e-6:      # antiparallel: u unrecoverable
            continue
        th = np.linspace(0, alpha[i], 4000)
        t = va[i][None, :] * np.cos(th)[:, None] + u[i][None, :] * np.sin(th)[:, None]
        inc = np.arccos(np.clip(t[:, 2], -1, 1))
        e_inc = max(e_inc, abs(inc.min() - r['inc_min'][i]),
                    abs(inc.max() - r['inc_max'][i]))
        if np.hypot(t[:, 0], t[:, 1]).min() < 1e-3:   # passes vertical: azi singular
            continue
        azi = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
        e_swing = max(e_swing, abs((azi[-1] - azi[0]) - r['azi_swing'][i]))
    assert e_inc < 1e-3          # inclination extrema exact
    assert e_swing < 1e-6        # azimuth swing exact (incl. reflex + multi-wrap)


def test_azimuth_monotonic_direction_is_sign_K():
    va, vb, u, alpha = _make_arcs(2000, seed=2)
    r = arc_inc_azi_extrema(va, vb, alpha)
    K = va[:, 0] * u[:, 1] - va[:, 1] * u[:, 0]
    ok = np.abs(K) > 1e-6
    # swing sign follows K; a near-vertical arc can flip, so allow the ones that
    # pass vertical to differ
    good = ok & ~r['passes_vertical']
    assert np.all(np.sign(r['azi_swing'][good]) == np.sign(K[good]))


def test_endpoints_and_reflex():
    # a reflex arc (dogleg > pi) must still reproduce vec_b azimuth at the end
    va = np.array([[1.0, 0.0, 0.5]])
    va = _unit(va)
    u = _unit(np.array([[0.0, 1.0, 0.0]]))          # in-plane perp
    alpha = np.array([1.2 * np.pi])                  # reflex
    vb = va * np.cos(alpha)[:, None] + u * np.sin(alpha)[:, None]
    r = arc_inc_azi_extrema(va, vb, alpha)
    assert np.isclose(r['azi_start'][0], np.arctan2(va[0, 1], va[0, 0]))
    assert np.isclose(r['azi_end'][0], np.arctan2(vb[0, 1], vb[0, 0]))


def test_straight_segment_constant():
    va = _unit(np.array([[0.4, 0.3, 0.87]]))
    r = arc_inc_azi_extrema(va, va.copy(), np.array([0.0]))
    inc = np.arccos(np.clip(va[0, 2], -1, 1))
    assert np.isclose(r['inc_min'][0], inc) and np.isclose(r['inc_max'][0], inc)
    assert r['azi_swing'][0] == 0.0
    assert not r['passes_vertical'][0]


def test_vertical_passage_flagged():
    # start vertical, curl slightly off: min inclination hits 0 -> flagged
    va = np.array([[0.0, 0.0, 1.0]])
    u = _unit(np.array([[0.3, 0.1, 0.0]]))
    alpha = np.array([0.4])
    vb = va * np.cos(alpha)[:, None] + u * np.sin(alpha)[:, None]
    r = arc_inc_azi_extrema(va, vb, alpha)
    assert bool(r['passes_vertical'][0])
    assert r['inc_min'][0] < 1e-4


def test_horizontal_plane_swing_equals_alpha():
    # tangent confined to the horizontal plane (V=0): |t_EN|=1 everywhere, so the
    # azimuth swings at unit rate -> total swing = +/- alpha exactly, and inc is
    # constant at 90 deg. A near-2pi arc therefore covers (almost) all azimuths.
    va = np.array([[1.0, 0.0, 0.0]])
    u = np.array([[0.0, 1.0, 0.0]])
    for a in (0.5, 0.9 * np.pi, 1.5 * np.pi, 1.99 * np.pi):
        alpha = np.array([a])
        vb = va * np.cos(alpha)[:, None] + u * np.sin(alpha)[:, None]
        r = arc_inc_azi_extrema(va, vb, alpha)
        assert np.isclose(abs(r['azi_swing'][0]), a, atol=1e-9)
        assert np.isclose(r['inc_min'][0], np.pi / 2) and \
            np.isclose(r['inc_max'][0], np.pi / 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
