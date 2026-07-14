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
