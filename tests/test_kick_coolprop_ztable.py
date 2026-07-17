"""ZTable (precomputed CoolProp Z(P,T) surface) validation.

Skipped when the optional CoolProp backend is not installed. When it is, the table
must reproduce direct CoolProp flashes to interpolation tolerance and round-trip
through the disk cache.
"""
import numpy as np
import pytest

pytest.importorskip("CoolProp")

from welleng.kick_tolerance.gas_z_coolprop import ZTable, fluid_z_density

COMP = {"methane": 0.9, "co2": 0.1}
PBOX = (15.0, 7500.0)
TBOX = (560.0, 720.0)


@pytest.fixture(scope="module")
def table():
    return ZTable(COMP, PBOX, TBOX, cache=False)


def test_ztable_matches_coolprop(table):
    """Interpolated Z / density track direct CoolProp flashes to <1e-3."""
    for p in np.linspace(200, 7300, 9):
        for t in np.linspace(570, 710, 5):
            z_ref, rho_ref = fluid_z_density(COMP, float(p), float(t))
            assert float(table.z(p, t)) == pytest.approx(z_ref, abs=1e-3)
            assert float(table.rho_ppg(p, t)) == pytest.approx(rho_ref, rel=2e-3)


def test_ztable_vectorised(table):
    """Vectorised lookup equals element-wise scalar lookup."""
    ps = np.array([1000.0, 3000.0, 6000.0])
    zs = table.z(ps, 640.0)
    assert zs.shape == ps.shape
    for p, z in zip(ps, zs):
        assert float(table.z(float(p), 640.0)) == pytest.approx(float(z))


def test_ztable_clamps_outside_box(table):
    """Lookups outside the box clamp to the edge (no extrapolation / NaN)."""
    z_lo = float(table.z(PBOX[0] - 500.0, 640.0))
    z_edge = float(table.z(PBOX[0], 640.0))
    assert np.isfinite(z_lo) and z_lo == pytest.approx(z_edge)


def test_ztable_disk_cache_roundtrip():
    """Second construction loads from the disk cache (identical grids)."""
    a = ZTable(COMP, PBOX, TBOX, cache=True)
    b = ZTable(COMP, PBOX, TBOX, cache=True)
    assert np.array_equal(a._Z, b._Z)
    assert float(a.z(3200.0, 640.0)) == float(b.z(3200.0, 640.0))


def test_analytical_coolprop_route_runs_and_differs_from_methane():
    """The analytical solver's CoolProp route (gas_composition=) runs end-to-end
    and gives a physically sensible, different answer from the methane H-Y default:
    a dense, poorly-expanding CO2-rich influx is less buoyant -> a HIGHER kick
    tolerance than light methane."""
    from welleng.kick_tolerance import WellSection, analytical_kick_tolerance
    secs = [WellSection(0.0, 6500.0, 0.066, False),
            WellSection(6500.0, 10500.0, 0.046, True)]
    pp = (np.array([0.0, 10500.0]), np.array([10.5, 11.0]))
    fp = (np.array([0.0, 10500.0]), np.array([14.0, 14.0]))
    common = dict(bhp_psi=6402.0, rho_mud_ppg=12.0, gas_bh_state=(None, 660.0, None, None))
    methane = analytical_kick_tolerance(secs, pp, fp, **common).max_influx_bbl
    co2rich = analytical_kick_tolerance(
        secs, pp, fp, gas_composition={"methane": 0.2, "co2": 0.8}, **common).max_influx_bbl
    assert 30.0 < methane < 90.0
    assert 30.0 < co2rich < 150.0
    assert co2rich > methane                      # dense CO2 kick is milder
