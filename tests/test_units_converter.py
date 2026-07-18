"""Tests for the fast, generic Units boundary converter (welleng.units.Units)."""
import math

import numpy as np
import pytest

from welleng.units import Units, CANONICAL, ureg


def test_to_and_from_canonical_roundtrip():
    u = Units(length="ft", angle="degree", pressure="psi")
    for quantity, val in [("length", 1234.5), ("angle", 42.0), ("pressure", 5000.0)]:
        canon = u.to_canonical(val, quantity)
        assert u.from_canonical(canon, quantity) == pytest.approx(val)


def test_canonical_values_are_si():
    u = Units(length="ft", angle="degree")
    assert u.to_canonical(1000.0, "length") == pytest.approx(304.8)        # ft -> m
    assert u.to_canonical(180.0, "angle") == pytest.approx(math.pi)        # deg -> rad
    assert u.from_canonical(math.pi, "angle") == pytest.approx(180.0)      # rad -> deg


def test_generic_convert_pair():
    u = Units()
    assert u.convert(1.0, "bar", "pascal") == pytest.approx(1e5)
    assert u.convert(1.0, "inch", "mm") == pytest.approx(25.4)
    assert u.convert(1.0, "ppg", "kg/m**3") == pytest.approx(119.826, abs=1e-2)


def test_affine_temperature_offset():
    """Temperature is affine (non-zero offset), not a pure scale."""
    u = Units()
    assert u.convert(32.0, "degF", "K") == pytest.approx(273.15)
    assert u.convert(0.0, "degC", "K") == pytest.approx(273.15)
    # a pure-scale converter would give 0 K for 32 degF -> guard against that
    assert u.convert(32.0, "degF", "K") != pytest.approx(0.0)


def test_array_conversion_vectorises():
    u = Units(length="ft")
    arr = np.array([100.0, 1000.0, 2500.0])
    out = u.to_canonical(arr, "length")
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, arr * 0.3048)


def test_same_unit_is_noop_identity():
    u = Units()
    x = np.array([1.0, 2.0, 3.0])
    assert u.convert(x, "meter", "meter") is x        # no copy, no work


def test_default_system_is_si_lengths_metric_angles_degrees():
    u = Units()
    assert u.system["length"] == "meter"
    assert u.system["angle"] == "degree"
    # length already SI -> to_canonical is a passthrough
    assert u.to_canonical(500.0, "length") == 500.0


def test_matches_pint_ground_truth():
    """The cached-factor result equals a direct pint conversion."""
    u = Units()
    for src, dst, val in [("ft", "m", 3280.0), ("psi", "bar", 145.0),
                          ("degF", "K", 100.0), ("ppg", "sg", 8.345)]:
        assert u.convert(val, src, dst) == pytest.approx(
            ureg.Quantity(val, src).to(dst).magnitude
        )


def test_factor_offset_is_cached():
    u = Units()
    u.convert(1.0, "ft", "m")
    assert ("ft", "m") in u._cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
