"""Tests for the thermal-domain unit helpers in welleng.units."""
import pytest

from welleng import units as u


def test_temperature_is_offset():
    # 60 degF = 288.71 K (offset applied, not a scale)
    assert u.to(u.temperature(60, "degF"), "kelvin") == pytest.approx(288.706, abs=1e-2)
    assert u.to_si(u.temperature(100, "degF")) == pytest.approx(310.928, abs=1e-2)


def test_temperature_delta_is_a_scale_not_offset():
    # a 1 degF difference is 0.5556 K difference (NO 255 K offset)
    dt = u.to(u.temperature_delta(1, "delta_degF"), "kelvin")
    assert dt == pytest.approx(5 / 9, abs=1e-4)


def test_offset_and_delta_differ_by_the_absolute_zero_offset():
    # the whole point of two helpers: same number, wildly different meaning
    offset = u.to(u.temperature(1, "degF"), "kelvin")
    delta = u.to(u.temperature_delta(1, "delta_degF"), "kelvin")
    assert offset > 255 and delta < 1  # would silently collide with one helper


@pytest.mark.parametrize("helper,val,unit,target,expected", [
    (u.mass_rate, 1, "pound/hour", "kg/s", 1.2600e-4),
    (u.specific_heat, 1, "Btu/(pound*delta_degF)", "J/(kg*kelvin)", 4186.8),
    (u.thermal_conductivity, 1, "Btu/(hour*foot*delta_degF)", "W/(m*kelvin)", 1.7307),
    (u.thermal_diffusivity, 1, "foot**2/hour", "m**2/s", 2.5806e-5),
    (u.heat_transfer_coefficient, 1, "Btu/(hour*foot**2*delta_degF)",
     "W/(m**2*kelvin)", 5.6783),
    (u.temperature_gradient, 1, "delta_degF/foot", "kelvin/meter", 1.8227),
])
def test_field_to_si_conversions(helper, val, unit, target, expected):
    assert u.to(helper(val, unit), target) == pytest.approx(expected, rel=1e-3)


def test_defaults_are_si():
    cp = u.to(u.specific_heat(4186.8), "Btu/(pound*delta_degF)")
    assert cp == pytest.approx(1.0, rel=1e-3)
    assert u.to(u.mass_rate(1.0), "kg/s") == 1.0


@pytest.mark.parametrize("helper", [
    u.specific_heat, u.thermal_conductivity, u.thermal_diffusivity,
    u.heat_transfer_coefficient, u.temperature_gradient, u.mass_rate,
])
def test_wrong_dimensionality_rejected(helper):
    with pytest.raises(ValueError):
        helper(1, "meter")
