"""Tests for the opt-in pint-backed unit helpers (welleng.units)."""
import math

import pytest

from welleng import units
from welleng.units import ureg


def test_single_registry_shared():
    """Quantities from the helpers interoperate with the shared ureg."""
    q = units.length(1.0, 'm') + ureg('100 cm')
    assert units.to(q, 'm') == pytest.approx(2.0)


def test_angles_round_trip():
    assert units.to_rad(units.deg(180)) == pytest.approx(math.pi)
    assert units.to_deg(units.rad(math.pi)) == pytest.approx(180.0)


def test_length_conversion():
    assert units.to(units.length(1, 'ft'), 'm') == pytest.approx(0.3048)
    assert units.to(units.length(1, 'inch'), 'mm') == pytest.approx(25.4)


def test_pressure_conversion():
    assert units.to(units.pressure(1, 'bar'), 'Pa') == pytest.approx(1e5)
    assert units.to(units.pressure(14.5037738, 'psi'), 'bar') == pytest.approx(1.0, rel=1e-4)


def test_mud_weight_ppg_sg():
    # 1.0 sg (fresh water) ~= 8.345 ppg
    assert units.to(units.mud_weight(1.0, 'sg'), 'ppg') == pytest.approx(8.345, abs=1e-2)
    assert units.to(units.mud_weight(8.345, 'ppg'), 'sg') == pytest.approx(1.0, abs=1e-2)


def test_hydrostatic_gradient_field_factor():
    """dP/dz = rho*g reproduces the field 0.052 psi/ft per ppg constant."""
    grad = units.hydrostatic_gradient(units.mud_weight(1.0, 'ppg'))
    assert units.to(grad, 'psi/ft') == pytest.approx(0.052, abs=5e-4)


def test_gradient_round_trip():
    mw = units.mud_weight(12.5, 'ppg')
    grad = units.hydrostatic_gradient(mw)
    back = units.mud_weight_from_gradient(grad)
    assert units.to(back, 'ppg') == pytest.approx(12.5)


def test_converters():
    assert units.to_si(units.length(1, 'km')) == pytest.approx(1000.0)
    assert units.magnitude(units.pressure(5000, 'psi')) == pytest.approx(5000.0)


def test_dimensionality_guard_rejects_wrong_unit():
    with pytest.raises(ValueError):
        units.length(1, 'psi')       # not a length
    with pytest.raises(ValueError):
        units.pressure(1, 'meter')   # not a pressure
    with pytest.raises(ValueError):
        units.mud_weight(1, 'psi')   # not a density


def test_ft_lbf_custom_unit_preserved():
    """The pre-existing custom ft_lbf unit still resolves."""
    assert units.to(ureg('1 ft_lbf'), 'N*m') == pytest.approx(1.35582, rel=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_two_arg_to_si_and_to_overloads():
    from welleng import units as u
    # 2-arg / 3-arg boilerplate-free forms match the Quantity forms
    assert u.to_si(1.0, "bar") == u.to_si(u.pressure(1, "bar")) == 100000.0
    assert u.to(1.0, "bar", "Pa") == u.to(u.pressure(1, "bar"), "Pa") == 100000.0
    # a numeric-factor drilling unit (YP) round-trips
    assert abs(u.to_si(1.0, "lbf/(100*ft**2)") - 0.4788026) < 1e-4


def test_is_valid_unit_predicate():
    from welleng import units as u
    for good in ("in**2", "kg/m**3", "m/hr", "Pa*s", "lbf/(100*ft**2)", "ppg", "sg"):
        assert u.is_valid_unit(good), good
    for bad in ("not_a_unit", "xyzzy", ""):
        assert not u.is_valid_unit(bad), bad
