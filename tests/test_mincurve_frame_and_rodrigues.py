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


# --- attitude without paying for a position ------------------------------- #
def test_inc_azi_at_matches_interpolate_bitwise():
    """Two ways to get an attitude must never be two ANSWERS. Asserted
    bitwise: the moment these can differ there are two implementations of the
    same quantity, which is the defect this whole arc kernel exists to avoid."""
    mc = MinCurve(MD, INC, AZI)
    q = np.linspace(MD[0], MD[-1], 257)
    _, inc_i, azi_i = mc.interpolate(q, angles=True)
    inc_o, azi_o = mc.inc_azi_at(q)
    assert np.array_equal(inc_i, inc_o)
    assert np.array_equal(np.mod(azi_i, 2 * np.pi), azi_o)


def test_inc_azi_at_is_nan_outside_the_survey():
    """Not the nearest station's attitude -- that is a fabricated pose, and an
    attitude is what orients everything downstream of it."""
    mc = MinCurve(MD, INC, AZI)
    inc, azi = mc.inc_azi_at(np.array([-10.0, 5000.0]))
    assert np.all(np.isnan(inc)) and np.all(np.isnan(azi))


def test_inc_azi_at_scalar_returns_floats():
    mc = MinCurve(MD, INC, AZI)
    inc, azi = mc.inc_azi_at(450.0)
    assert isinstance(inc, float) and isinstance(azi, float)
    assert 0.0 <= azi < 2 * np.pi


def test_inc_azi_at_honours_the_frame():
    """Attitude is frame-independent -- inc/azi are angles, not columns."""
    a = MinCurve(MD, INC, AZI).inc_azi_at(np.array([137.0, 450.0]))
    b = MinCurve(MD, INC, AZI, frame="nev").inc_azi_at(np.array([137.0, 450.0]))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_arc_step_still_returns_both_halves():
    """The split is internal. arc_step is public and consumers call it."""
    v1 = get_vec(30.0, 20.0, deg=True)
    v2 = get_vec(75.0, 65.0, deg=True)
    th = np.array([float(np.arccos(np.clip(v1[0] @ v2[0], -1, 1)))])
    disp, tang = arc_step(v1, v2, th, np.array([300.0]), np.array([137.0]))
    assert disp.shape == (1, 3) and tang.shape == (1, 3)
    assert np.linalg.norm(tang[0]) == pytest.approx(1.0)


def test_interpolate_angles_also_honours_the_frame():
    """The frame option created a SECOND place where column order matters, and
    the first fix missed this one. An attitude is frame-independent, so a
    wrong basis here returns a confident wrong pose that nothing downstream
    can catch."""
    q = np.array([137.0, 450.0, 880.0])
    _, i_env, a_env = MinCurve(MD, INC, AZI).interpolate(q, angles=True)
    _, i_nev, a_nev = MinCurve(MD, INC, AZI,
                               frame="nev").interpolate(q, angles=True)
    assert np.allclose(i_env, i_nev, rtol=0, atol=1e-15)
    assert np.allclose(np.mod(a_env, 2 * np.pi), np.mod(a_nev, 2 * np.pi),
                       rtol=0, atol=1e-15)


def test_the_nev_attitude_is_the_true_one_not_a_transposed_one():
    """Anchored against the input: a leg built due north at 90 deg inclination
    must read back azi 0, in EITHER frame. A transpose would read 90."""
    md = np.array([0.0, 100.0])
    inc = np.radians([90.0, 90.0])
    azi = np.radians([0.0, 0.0])
    for frame in ("env", "nev"):
        i, a = MinCurve(md, inc, azi, frame=frame).inc_azi_at(50.0)
        assert np.degrees(i) == pytest.approx(90.0, abs=1e-9)
        assert np.degrees(a) % 360 == pytest.approx(0.0, abs=1e-9)


def test_leg_inc_azi_matches_inc_azi_at():
    """The single-leg scalar attitude (cov_nev_at's) and inc_azi_at share the
    arc and the in-plane vector; only the scalar arithmetic differs, so they
    agree to a few ulp -- on curved, straight, vertical and near-pi legs."""
    md = np.arange(0.0, 331.0, 30.0)
    inc = np.radians([0, 0, 5, 30, 30, 90, 179.0, 1.0, 1.0, 60, 60, 120])
    azi = np.radians([0, 0, 10, 350, 350, 20, 200.0, 25, 25, 359.9, 0.1, 90])
    for mc in (MinCurve(MD, INC, AZI), MinCurve(MD, INC, AZI, frame="nev"),
               MinCurve(md, inc, azi)):
        for i in range(len(mc.md) - 1):
            for x in np.linspace(0.0, mc.delta_md[i + 1], 7):
                a = mc._leg_inc_azi(i, float(x))
                b = mc.inc_azi_at(float(mc.md[i] + x))
                assert abs(a[0] - b[0]) < 1e-14
                d = abs(a[1] - b[1])
                # azimuth is undefined at vertical; elsewhere compare mod 2 pi
                assert a[0] < 1e-12 or min(d, 2 * np.pi - d) < 1e-13


def test_leg_frames_rebuild_the_kernel_tangent():
    """cos(phi) v1 + sin(phi) u, from the public leg frame, is the tangent
    interpolate(angles=True) returns -- including through north and near pi."""
    md = np.arange(0.0, 211.0, 30.0)
    inc = np.radians([0.3, 20, 60, 90, 88, 150, 2.0, 179.0])
    azi = np.radians([10, 350, 5, 90, 95, 300, 20, 200])
    mc = MinCurve(md, inc, azi)
    v1, u, curved = mc.leg_frames()
    for i in range(len(md) - 1):
        for f in (0.1, 0.5, 0.9):
            x = f * mc.delta_md[i + 1]
            _, qi, qa = mc.interpolate(md[i] + x, angles=True)
            want = np.array([np.sin(qi) * np.sin(qa), np.sin(qi) * np.cos(qa),
                             np.cos(qi)])
            if curved[i]:
                phi = x * mc.dogleg[i + 1] / mc.delta_md[i + 1]
                got = np.cos(phi) * v1[i] + np.sin(phi) * u[i]
            else:
                got = v1[i]
            assert np.allclose(got, want, atol=1e-12)
