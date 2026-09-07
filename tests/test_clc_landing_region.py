"""Tests for solve_clc_landing_region — min-MD CLC landing into a Target region.

Validated against an independent brute-force min-MD oracle (dense grid of
point-to-target solve_clc calls over the region), plus a frame-convention check
and the point/infeasible edge cases.
"""
import numpy as np
import pytest

from welleng.sawaryn_analytical import (
    solve_clc, solve_clc_landing_region, _target_frame,
)
from welleng.target import Target

P1 = np.array([0.0, 0.0, 0.0])
T1 = np.array([0.0, 0.0, 1.0])          # vertical kickoff
T4 = np.array([1.0, 0.0, 0.0])          # land pointing North
R1 = 200.0


def _oracle(target, n=41):
    """Independent min-MD over a dense grid of point-to-target solves."""
    center, u, v = _target_frame(target)
    g = target.geometry
    if target.shape == "rectangle":
        (a0, b0), (a1, b1) = g["pos1"], g["pos2"]
        lo = (min(a0, a1), min(b0, b1))
        hi = (max(a0, a1), max(b0, b1))
        r = None
    else:  # circle
        r = g["radius"]
        lo, hi = (-r, -r), (r, r)

    def inside(a, b):
        return r is None or a * a + b * b <= r * r + 1e-9
    best = np.inf
    for a in np.linspace(lo[0], hi[0], n):
        for b in np.linspace(lo[1], hi[1], n):
            if not inside(a, b):
                continue
            s = solve_clc(P1, T1, center + a * u + b * v, T4, R1)
            if s and s["alpha1"] <= np.pi + 1e-9 and s["alpha2"] <= np.pi + 1e-9:
                best = min(best, s["total_md"])
    return best


def test_rectangle_interior_matches_oracle():
    # box straddling the easiest-to-reach point -> interior optimum
    t = Target("box", 200, 0, 300, "rectangle", pos1=[-40, -40], pos2=[40, 40])
    sol = solve_clc_landing_region(P1, T1, T4, t, R1)
    orc = _oracle(t)
    assert sol is not None
    # solver refines the grid, so it is never worse than the oracle...
    assert sol["total_md"] <= orc + 1e-3
    # ...and agrees with it to grid resolution
    assert orc - sol["total_md"] < 0.5
    # the landing point is inside the box and on the target plane
    a, b = sol["ab"]
    assert -40 <= a <= 40 and -40 <= b <= 40
    assert np.allclose(sol["p4"], [200 + a, b, 300])


def test_rectangle_boundary_optimum():
    # box offset so the nearest reachable point is on an edge/corner
    t = Target("box", 150, 40, 300, "rectangle", pos1=[-50, -30], pos2=[50, 30])
    sol = solve_clc_landing_region(P1, T1, T4, t, R1)
    orc = _oracle(t)
    assert sol is not None
    assert sol["total_md"] <= orc + 1e-3
    assert orc - sol["total_md"] < 0.5


def test_circle_matches_oracle():
    t = Target("disk", 150, 40, 300, "circle", radius=40)
    sol = solve_clc_landing_region(P1, T1, T4, t, R1)
    orc = _oracle(t)
    assert sol is not None
    assert sol["total_md"] <= orc + 1e-3          # boundary theta-solve refines
    assert abs(sol["total_md"] - orc) < 0.5
    a, b = sol["ab"]
    assert a * a + b * b <= 40 ** 2 + 1e-6


def test_point_equals_solve_clc():
    t = Target("pt", 200, 0, 300, "point")
    sol = solve_clc_landing_region(P1, T1, T4, t, R1)
    direct = solve_clc(P1, T1, t.position, T4, R1)
    assert sol is not None and direct is not None
    assert sol["total_md"] == pytest.approx(direct["total_md"])
    assert np.allclose(sol["p4"], [200, 0, 300])


def test_frame_convention_dip_90():
    # dip=90 about North tilts the East axis to point straight down (V)
    t = Target("d", 100, 0, 500, "rectangle", pos1=[-1, -1], pos2=[1, 1], dip=90)
    center, u, v = _target_frame(t)
    assert np.allclose(center, [100, 0, 500])
    assert np.allclose(u, [1, 0, 0])          # North unchanged
    assert np.allclose(v, [0, 0, 1], atol=1e-9)   # East -> Down


@pytest.mark.parametrize("dip,ori", [(35, 0), (35, 60), (90, 0)])
def test_dipped_oriented_plane_matches_oracle(dip, ori):
    # the target plane is at an arbitrary orientation, not just horizontal NE
    t = Target("t", 180, 20, 300, "rectangle", pos1=[-40, -40], pos2=[40, 40],
               dip=dip, orientation=ori)
    sol = solve_clc_landing_region(P1, T1, T4, t, R1)
    orc = _oracle(t)
    assert sol is not None
    assert sol["total_md"] <= orc + 1e-3
    assert orc - sol["total_md"] < 0.5
    # the solved landing point lies in the tilted target plane
    c, u, v = _target_frame(t)
    assert abs((sol["p4"] - c) @ np.cross(u, v)) < 1e-9


def _md_pt(p4):
    s = solve_clc(P1, T1, p4, T4, R1)
    if s and s["alpha1"] <= np.pi + 1e-9 and s["alpha2"] <= np.pi + 1e-9:
        return s["total_md"]
    return np.inf


def test_polygon_matches_oracle():
    verts = [[-40, -30], [40, -30], [0, 50]]
    t = Target("poly", 180, 0, 300, "polygon", vertices=verts)
    sol = solve_clc_landing_region(P1, T1, T4, t, R1)
    assert sol is not None
    # in-plane grid oracle over the triangle's bounding box, inside-poly only
    from welleng.sawaryn_analytical import _point_in_polygon
    c, u, v = _target_frame(t)
    vs = np.array(verts, float)
    best = np.inf
    for a in np.linspace(vs[:, 0].min(), vs[:, 0].max(), 61):
        for b in np.linspace(vs[:, 1].min(), vs[:, 1].max(), 61):
            if _point_in_polygon(a, b, vs):
                best = min(best, _md_pt(c + a * u + b * v))
    assert sol["total_md"] <= best + 1e-3
    assert best - sol["total_md"] < 0.5


def test_ellipse_matches_oracle():
    t = Target("ell", 160, 0, 300, "ellipse", radius_1=50, radius_2=25)
    sol = solve_clc_landing_region(P1, T1, T4, t, R1)
    assert sol is not None
    c, u, v = _target_frame(t)
    best = np.inf
    for a in np.linspace(-50, 50, 61):
        for b in np.linspace(-25, 25, 61):
            if (a / 50) ** 2 + (b / 25) ** 2 <= 1.0:
                best = min(best, _md_pt(c + a * u + b * v))
    assert sol["total_md"] <= best + 1e-3
    assert best - sol["total_md"] < 0.5


def test_cube_matches_oracle():
    t = Target("cube", 180, 0, 300, "cube", half_extents=[40, 40, 40])
    sol = solve_clc_landing_region(P1, T1, T4, t, R1)
    assert sol is not None
    center = np.array([180.0, 0, 300])
    best = np.inf
    for x in np.linspace(-40, 40, 21):
        for y in np.linspace(-40, 40, 21):
            for z in np.linspace(-40, 40, 21):
                best = min(best, _md_pt(center + [x, y, z]))
    assert sol["total_md"] <= best + 1e-3
    assert best - sol["total_md"] < 0.5


def test_sphere_matches_surface_oracle():
    t = Target("sph", 180, 0, 300, "sphere", radius=40)
    sol = solve_clc_landing_region(P1, T1, T4, t, R1)
    assert sol is not None
    center = np.array([180.0, 0, 300])
    # the min sits on the boundary (external kickoff) -> dense surface oracle
    best = np.inf
    for th in np.linspace(0, np.pi, 61):
        for ph in np.linspace(0, 2 * np.pi, 61):
            d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            best = min(best, _md_pt(center + 40 * d))
    assert abs(sol["total_md"] - best) < 0.5
    assert np.sum((sol["p4"] - center) ** 2) <= 40 ** 2 + 1e-3


def test_gaussian_cov_ellipsoid_matches_surface_oracle():
    cov = np.diag([40.0, 25, 30]) ** 2
    t = Target("cov", 180, 0, 300, "gaussian", cov=cov)
    sol = solve_clc_landing_region(P1, T1, T4, t, R1, k=1.0)
    assert sol is not None
    center = np.array([180.0, 0, 300])
    L = np.linalg.cholesky(cov)
    best = np.inf
    for th in np.linspace(0, np.pi, 61):
        for ph in np.linspace(0, 2 * np.pi, 61):
            d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            best = min(best, _md_pt(center + L @ d))
    assert abs(sol["total_md"] - best) < 0.5
    # solved point is within the 1-sigma ellipsoid
    dd = np.linalg.solve(L, sol["p4"] - center)
    assert dd @ dd <= 1.0 + 1e-3


def test_unreachable_region_returns_none():
    # a tiny target directly above the kickoff, tangent North: no feasible CLC
    t = Target("bad", 0, 0, -50, "circle", radius=1)
    assert solve_clc_landing_region(P1, T1, T4, t, R1) is None
