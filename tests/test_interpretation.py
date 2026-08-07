"""Tests for welleng.interpretation (sensor->survey, QC, MSA).

The oracle for the forward model is an analytic round-trip: build ideal tool-frame
sensor readings from a known (inc, azi, toolface) + reference field, then confirm
the navigation equations recover the inputs. MSA is validated by injecting a known
tool error into ideal readings and recovering it in closed form.
"""
import numpy as np
import pytest

from welleng.interpretation import (
    sensor_to_survey,
    gyro_to_survey,
    earth_rate_components,
    EARTH_RATE,
    GeomagReference,
    georef_checks,
    dual_depth_difference,
    estimate_sensor_errors,
)

G_REF, B_REF, DIP = 9.81, 50000.0, np.radians(70.0)


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _ideal(inc, azi, tf, g=G_REF, bt=B_REF, dip=DIP):
    """Ideal (Gz-axial) accelerometer + magnetometer for a pose."""
    Re2t = (_Rz(azi) @ _Ry(inc) @ _Rz(tf)).T
    b_earth = np.array([bt * np.cos(dip), 0, bt * np.sin(dip)])
    return Re2t @ np.array([0, 0, g]), Re2t @ b_earth


# --- forward model -----------------------------------------------------------
@pytest.mark.parametrize("inc0,azi0,tf0", [(10, 30, 45), (60, 135, 200), (89, 350, 12),
                                           (5, 80, 300), (120, 220, 90)])
def test_forward_roundtrip(inc0, azi0, tf0):
    g, b = _ideal(*np.radians([inc0, azi0, tf0]))
    inc, azi, tf = sensor_to_survey(g, b)
    assert inc == pytest.approx(inc0, abs=1e-9)
    assert (azi - azi0 + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-9)


def test_forward_axis_first_convention():
    g, b = _ideal(*np.radians([40, 100, 210]))
    # re-order to axial-FIRST (x = axial) and read back with axis='x'
    g_xfirst = g[[2, 0, 1]]
    b_xfirst = b[[2, 0, 1]]
    inc, azi, _ = sensor_to_survey(g_xfirst, b_xfirst, axis="x")
    assert inc == pytest.approx(40, abs=1e-9)
    assert (azi - 100 + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-9)


def test_forward_declination_applied():
    g, b = _ideal(*np.radians([30, 120, 60]))
    _, azi_mag, _ = sensor_to_survey(g, b)
    _, azi_grid, _ = sensor_to_survey(g, b, declination=2.5, grid_convergence=1.0)
    assert (azi_grid - azi_mag) % 360 == pytest.approx(1.5, abs=1e-9)


def test_forward_array_and_scalar_shapes():
    poses = np.radians([[10, 30, 45], [60, 135, 200]])
    gs = np.array([_ideal(*p)[0] for p in poses])
    bs = np.array([_ideal(*p)[1] for p in poses])
    inc, azi, tf = sensor_to_survey(gs, bs)
    assert inc.shape == (2,) and azi.shape == (2,)
    assert isinstance(sensor_to_survey(gs[0], bs[0])[0], float)


# --- gyro (gyrocompassing) ---------------------------------------------------
def _ideal_gyro(inc, azi, tf, lat, rate=EARTH_RATE):
    """Ideal (Gz-axial) accelerometer + rate-gyro for a pose at latitude ``lat``.

    Earth-rate vector in NED: horizontal ``rate*cos(lat)`` (north), vertical
    ``rate*sin(lat)`` up (Down-component negative).
    """
    Re2t = (_Rz(azi) @ _Ry(inc) @ _Rz(tf)).T
    phi = np.radians(lat)
    w_ned = np.array([rate * np.cos(phi), 0.0, -rate * np.sin(phi)])
    return Re2t @ np.array([0, 0, G_REF]), Re2t @ w_ned


@pytest.mark.parametrize("inc0,azi0,tf0,lat", [(10, 30, 45, 60), (60, 135, 200, 40),
                                               (89, 350, 12, 71), (30, 220, 90, 5)])
def test_gyro_roundtrip_true_north(inc0, azi0, tf0, lat):
    g, w = _ideal_gyro(*np.radians([inc0, azi0, tf0]), lat)
    inc, azi, tf = gyro_to_survey(g, w)
    assert inc == pytest.approx(inc0, abs=1e-9)
    assert (azi - azi0 + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-7)


def test_gyro_no_declination_unlike_mag():
    # same pose: gyro gives TRUE north; mag would need +declination to match
    g, w = _ideal_gyro(*np.radians([40, 100, 60]), 55)
    _, azi_true, _ = gyro_to_survey(g, w)
    assert (azi_true - 100 + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-7)


def test_earth_rate_components_latitude():
    h, v = earth_rate_components(0.0)      # equator: all horizontal
    assert h == pytest.approx(EARTH_RATE) and v == pytest.approx(0.0)
    h, v = earth_rate_components(90.0)     # pole: all vertical
    assert h == pytest.approx(0.0, abs=1e-12) and v == pytest.approx(EARTH_RATE)


# --- QC ----------------------------------------------------------------------
def test_qc_clean_station_passes():
    g, b = _ideal(*np.radians([20, 60, 100]))
    ref = GeomagReference(b_total=B_REF, dip=np.degrees(DIP), g_total=G_REF)
    r = georef_checks(g, b, ref)
    assert r.passed.all()
    assert abs(r.d_b[0]) < 1e-6 and abs(r.d_dip[0]) < 1e-9


def test_qc_flags_magnetic_interference():
    g, b = _ideal(*np.radians([20, 60, 100]))
    b = b + np.array([0, 0, 1500.0])  # axial interference spike (nT)
    ref = GeomagReference(b_total=B_REF, dip=np.degrees(DIP), g_total=G_REF)
    r = georef_checks(g, b, ref)
    assert r.flag_b[0]                 # total-field flags
    assert not r.flag_g[0]             # gravity unaffected
    assert not r.passed[0]


# --- dual-depth-difference (SPE-133417 Table 1a/1b/1c) -----------------------
@pytest.mark.parametrize(
    "ddD, sig_pipe, sig_wll, tol_pub, verdict",
    [
        (0.6, 1.4, 3.4, 3.7, True),    # Table 1a @ 1000 m
        (2.2, 11.0, 16.5, 19.8, True),  # Table 1a @ 4000 m
        (11.0, 1.4, 3.4, 3.7, False),   # Table 1b @ 1000 m -> fail
        (44.0, 11.0, 16.5, 19.8, False),  # Table 1b @ 4000 m -> fail
        (3.4, 1.4, 3.4, 3.7, True),     # Table 1c @ 1000 m (marginal pass)
        (13.8, 11.0, 16.5, 19.8, True),  # Table 1c @ 4000 m
    ],
)
def test_dual_depth_difference_reproduces_spe133417_table1(
    ddD, sig_pipe, sig_wll, tol_pub, verdict
):
    r = dual_depth_difference(ddD, sig_pipe, sig_wll)
    assert float(r.tolerance) == pytest.approx(tol_pub, abs=0.05)
    assert bool(r.passed) is verdict


def test_dual_depth_difference_vectorised():
    r = dual_depth_difference([0.6, 44.0], [1.4, 11.0], [3.4, 16.5])
    assert list(r.passed) == [True, False]


# --- MSA ---------------------------------------------------------------------
def _geometry_rich(n=60, seed=7):
    rng = np.random.default_rng(seed)
    inc = np.radians(rng.uniform(5, 85, n))
    azi = np.radians(rng.uniform(0, 360, n))
    tf = np.radians(rng.uniform(0, 360, n))
    return np.array([_ideal(i, a, t)[1] for i, a, t in zip(inc, azi, tf)])


def test_msa_closed_form_recovers_injected_error():
    Btrue = _geometry_rich()
    b_inj = np.array([180.0, -120.0, 250.0])
    s_inj = np.array([0.002, -0.0015, 0.003])
    Bmeas = (1 + s_inj) * Btrue + b_inj
    ref = GeomagReference(b_total=B_REF, dip=np.degrees(DIP), g_total=G_REF)
    res = estimate_sensor_errors(Bmeas, ref, sensor="mag")
    # 2nd-order linearisation error is tiny relative to the injected magnitudes
    assert np.abs(res.bias - b_inj).max() < 2.0
    assert np.abs(res.scale - s_inj).max() < 1e-4
    assert res.n_stations == 60


def test_msa_reports_axial_weakly_observable_on_narrow_azimuth():
    # near-constant azimuth (tangent section) -> axial poorly observed
    rng = np.random.default_rng(3)
    n = 40
    inc = np.radians(rng.uniform(30, 46, n))
    azi = np.radians(rng.uniform(276, 300, n))       # narrow
    tf = np.radians(rng.uniform(0, 360, n))
    Btrue = np.array([_ideal(i, a, t)[1] for i, a, t in zip(inc, azi, tf)])
    ref = GeomagReference(b_total=B_REF, dip=np.degrees(DIP), g_total=G_REF)
    res = estimate_sensor_errors(Btrue, ref, sensor="mag", noise=70.0)
    # axial std must be much larger than cross-axial (the observability law)
    assert res.std[2] > 5 * max(res.std[0], res.std[1])
    assert not res.axial_estimable          # gated off


def test_msa_min_stations_raises():
    Btrue = _geometry_rich(n=5)
    ref = GeomagReference(b_total=B_REF, dip=np.degrees(DIP), g_total=G_REF)
    with pytest.raises(ValueError, match="at least"):
        estimate_sensor_errors(Btrue, ref, sensor="mag")


def test_msa_correlation_matrix_shape_and_diag():
    res = estimate_sensor_errors(_geometry_rich(), GeomagReference(B_REF, 70.0, G_REF),
                                 sensor="mag", noise=50.0)
    assert res.correlation.shape == (6, 6)
    assert np.allclose(np.diag(res.correlation), 1.0, atol=1e-9)
