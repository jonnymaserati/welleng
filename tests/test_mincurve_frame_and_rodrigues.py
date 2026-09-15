"""MinCurve's output basis, and what the arc kernel actually computes.

Both prompted by a consumer: it was swapping every result from core's
[E, N, V] into its own [N, E, V], and it described the tangent step as SLERP.
"""
import numpy as np
import pytest

from welleng.utils import MinCurve, arc_step, get_vec

MD = np.array([0.0, 100.0, 300.0, 600.0, 900.0])
INC = np.radians([0.0, 30.0, 60.0, 90.0, 88.0])
AZI = np.radians([0.0, 20.0, 45.0, 80.0, 95.0])


# --- the frame is a basis, not a second algorithm -------------------------- #
def test_nev_is_exactly_the_default_with_two_columns_swapped():
    """Bitwise, not approximately. If these ever differ by an ulp it means the
    two frames have acquired separate arithmetic, which is the thing the
    single-kernel design exists to prevent."""
    env = MinCurve(MD, INC, AZI)
    nev = MinCurve(MD, INC, AZI, frame="nev")
    assert np.array_equal(env.poss[:, [1, 0, 2]], nev.poss)


def test_interpolate_follows_the_frame():
    q = np.array([50.0, 137.0, 450.0, 880.0])
    env = MinCurve(MD, INC, AZI)
    nev = MinCurve(MD, INC, AZI, frame="nev")
    assert np.array_equal(env.interpolate(q)[:, [1, 0, 2]], nev.interpolate(q))


def test_the_default_is_unchanged():
    """Survey consumes poss as [E, N, V] and applies its own axis swap, so the
    default must not move."""
    mc = MinCurve(MD, INC, AZI)
    assert mc.frame == "env"
    north = MinCurve(np.array([0.0, 100.0]), np.radians([90.0, 90.0]),
                     np.radians([0.0, 0.0]))
    assert north.poss[-1][1] == pytest.approx(100.0)      # column 1 is NORTH
    assert north.poss[-1][0] == pytest.approx(0.0)


def test_due_north_and_due_east_probes_in_nev():
    """The way to establish an axis order -- reading delta_x/delta_y invites
    the wrong guess, which is how the consumer's transpose happened."""
    kw = dict(md=np.array([0.0, 100.0]), inc=np.radians([90.0, 90.0]),
              frame="nev")
    north = MinCurve(azi=np.radians([0.0, 0.0]), **kw)
    east = MinCurve(azi=np.radians([90.0, 90.0]), **kw)
    assert north.poss[-1][0] == pytest.approx(100.0)      # column 0 is NORTH
    assert east.poss[-1][1] == pytest.approx(100.0)       # column 1 is EAST


def test_an_unknown_frame_is_refused_and_says_what_the_options_mean():
    with pytest.raises(ValueError, match="easting"):
        MinCurve(MD, INC, AZI, frame="xyz")


# --- what the kernel computes --------------------------------------------- #
def test_the_tangent_is_rodrigues_and_slerp_is_the_same_rotation():
    """The consumer called this SLERP. The kernel writes it as Rodrigues, and
    for unit tangents the two are the SAME rotation in the arc plane -- SLERP
    of two unit vectors IS a rotation of the first about their plane normal.
    Neither name is wrong about the geometry; they differ in conditioning.
    """
    v1 = get_vec(30.0, 20.0, deg=True)[0]
    v2 = get_vec(75.0, 65.0, deg=True)[0]
    th = float(np.arccos(np.clip(v1 @ v2, -1, 1)))
    dmd, x = 300.0, np.array([137.0])
    phi = th * x / dmd

    _, t_core = arc_step(v1[None, :], v2[None, :], np.array([th]),
                         np.array([dmd]), x)

    # Shoemake's SLERP blend
    t_slerp = ((np.sin(th - phi)[:, None] * v1
                + np.sin(phi)[:, None] * v2) / np.sin(th))

    # Rodrigues, rotating v1 about the plane normal
    k = np.cross(v1, v2)
    k = k / np.linalg.norm(k)
    t_rod = (np.cos(phi)[:, None] * v1
             + np.sin(phi)[:, None] * np.cross(k, v1)
             + (1 - np.cos(phi))[:, None] * k * (k @ v1))

    assert np.allclose(t_core, t_slerp, rtol=0, atol=5e-16)
    assert np.allclose(t_core, t_rod, rtol=0, atol=5e-16)


def test_the_tangent_is_a_unit_vector_through_the_leg():
    """Whatever it is called, it must stay on the sphere."""
    mc = MinCurve(MD, INC, AZI)
    for q in np.linspace(MD[0] + 1e-9, MD[-1], 97):
        _, inc_i, azi_i = mc.interpolate(np.array([q]), angles=True)
        v = get_vec(inc_i[0], azi_i[0], deg=False)[0]
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)


def test_rodrigues_has_no_per_query_one_over_sin():
    """Why the kernel uses the Rodrigues form rather than the SLERP blend: the
    1/sin(theta) is a ONE-TIME set-up on the in-plane basis vector, so a
    near-zero dogleg needs no special-cased branch per query. A tiny dogleg
    must simply give the straight-line answer."""
    v1 = get_vec(45.0, 30.0, deg=True)[0]
    v2 = get_vec(45.0 + 1e-9, 30.0, deg=True)[0]
    th = float(np.arccos(np.clip(v1 @ v2, -1, 1)))
    disp, tang = arc_step(v1[None, :], v2[None, :], np.array([th]),
                          np.array([100.0]), np.array([50.0]))
    assert np.all(np.isfinite(disp)) and np.all(np.isfinite(tang))
    assert np.allclose(disp[0], 50.0 * v1, rtol=0, atol=1e-6)
