"""Optional, opt-in unit helpers for welleng (pint-backed).

welleng's core API works in plain floats — SI inside the engine, field units at
the API boundary. This module is a convenience layer for users who *want*
unit-tagged quantities and safe conversions: build a quantity, convert it, or
strip it back to a float to feed the core. Importing welleng does not require
using any of this, and the core signatures stay plain ``float``.

Everything shares the single welleng ``ureg`` registry. Mixing pint registries
breaks quantity interoperation, so always build quantities through these helpers
(or ``ureg`` itself) rather than instantiating a fresh ``UnitRegistry``.

Examples
--------
>>> from welleng import units
>>> round(units.to(units.length(1000, 'ft'), 'm'), 6)
304.8
>>> round(units.to(units.hydrostatic_gradient(units.mud_weight(12.5, 'ppg')), 'psi/ft'), 4)
0.6494
>>> round(units.to_rad(units.deg(180)), 6)
3.141593
"""
from typing import Optional, Tuple, Union

import numpy as np
import pint

ureg: pint.UnitRegistry = pint.UnitRegistry()
Q_ = ureg.Quantity

# scalar or array value moving through the boundary converter
Numeric = Union[float, np.ndarray]

# --- custom units -------------------------------------------------------------
# TODO import custom units from file instead of defining here
ureg.define('ft_lbf = ft * lbf')
# Drilling mud-density units absent from pint's default registry. ``ppg`` is
# pounds per US gallon; ``sg`` (specific gravity) is treated as a density
# relative to fresh water (1 sg = 1000 kg/m^3) so it converts against ppg/kg·m⁻³.
ureg.define('ppg = pound / gallon')
ureg.define('sg = 1000 * kilogram / meter ** 3')

Number = Union[int, float]
UnitLike = Union[str, pint.Unit]


# --- angles -------------------------------------------------------------------
def deg(value: Number) -> pint.Quantity:
    """A quantity of ``value`` degrees.

    >>> deg(90).to('radian').magnitude
    1.5707963267948966
    """
    return value * ureg.degree


def rad(value: Number) -> pint.Quantity:
    """A quantity of ``value`` radians."""
    return value * ureg.radian


def to_deg(angle: pint.Quantity) -> float:
    """Magnitude of ``angle`` in degrees (float).

    >>> round(to_deg(rad(3.141592653589793)), 6)
    180.0
    """
    return float(angle.to(ureg.degree).magnitude)


def to_rad(angle: pint.Quantity) -> float:
    """Magnitude of ``angle`` in radians (float)."""
    return float(angle.to(ureg.radian).magnitude)


# --- typed quantity constructors ----------------------------------------------
def _quantity(
    value: Number, unit: UnitLike, dimensionality: str, name: str
) -> pint.Quantity:
    q = ureg.Quantity(value, unit)
    if not q.check(dimensionality):
        raise ValueError(
            f"{name} unit {str(unit)!r} is not a {name} ({dimensionality})"
        )
    return q


def length(value: Number, unit: UnitLike = 'meter') -> pint.Quantity:
    """A length quantity (default metres).

    >>> round(to(length(1, 'ft'), 'm'), 6)
    0.3048
    """
    return _quantity(value, unit, '[length]', 'length')


def pressure(value: Number, unit: UnitLike = 'psi') -> pint.Quantity:
    """A pressure quantity (default psi)."""
    return _quantity(value, unit, '[pressure]', 'pressure')


def mud_weight(value: Number, unit: UnitLike = 'ppg') -> pint.Quantity:
    """A mud-weight (density) quantity (default ppg).

    >>> round(to(mud_weight(1.2, 'sg'), 'ppg'), 3)
    10.014
    """
    return _quantity(value, unit, '[mass] / [length] ** 3', 'mud weight')


def force(value: Number, unit: UnitLike = 'newton') -> pint.Quantity:
    """A force quantity (default newtons)."""
    return _quantity(value, unit, '[force]', 'force')


def torque(value: Number, unit: UnitLike = 'newton * meter') -> pint.Quantity:
    """A torque quantity (default N·m).

    Note: torque and energy share a dimensionality in pint, so this does not
    reject energy units.
    """
    return _quantity(value, unit, '[force] * [length]', 'torque')


# --- thermal-domain constructors ----------------------------------------------
# CRITICAL: an absolute temperature is an OFFSET unit (degF <-> kelvin add a
# constant), whereas every gradient / heat-capacity / conductivity term uses a
# temperature DIFFERENCE (delta_degF <-> kelvin, a pure scale). Mixing the
# two silently injects ~255 K. Use :func:`temperature` for a state (a reading)
# and :func:`temperature_delta` for any per-degree term.
def temperature(value: Number, unit: UnitLike = 'degF') -> pint.Quantity:
    """An absolute temperature (default degF) — an OFFSET unit.

    Convert with :func:`to`/:func:`to_si` (kelvin): the constant offset is
    applied. Do NOT use this for gradients or specific heats — use
    :func:`temperature_delta`.

    >>> round(to(temperature(60, 'degF'), 'kelvin'), 2)
    288.71
    """
    return _quantity(value, unit, '[temperature]', 'temperature')


def temperature_delta(
    value: Number, unit: UnitLike = 'delta_degF'
) -> pint.Quantity:
    """A temperature DIFFERENCE (default delta_degF) — a pure scale.

    For the per-degree part of gradients, specific heats and conductivities.

    >>> round(to(temperature_delta(1, 'delta_degF'), 'kelvin'), 4)
    0.5556
    """
    return _quantity(value, unit, '[temperature]', 'temperature difference')


def mass_rate(value: Number, unit: UnitLike = 'kilogram / second') -> pint.Quantity:
    """A mass flow rate (default kg/s)."""
    return _quantity(value, unit, '[mass] / [time]', 'mass rate')


def specific_heat(
    value: Number, unit: UnitLike = 'joule / (kilogram * kelvin)'
) -> pint.Quantity:
    """A specific heat capacity (default J/(kg·K)). Uses delta temperature."""
    return _quantity(
        value, unit, '[energy] / [mass] / [temperature]', 'specific heat')


def thermal_conductivity(
    value: Number, unit: UnitLike = 'watt / (meter * kelvin)'
) -> pint.Quantity:
    """A thermal conductivity (default W/(m·K)). Uses delta temperature."""
    return _quantity(
        value, unit, '[power] / [length] / [temperature]',
        'thermal conductivity')


def thermal_diffusivity(
    value: Number, unit: UnitLike = 'meter ** 2 / second'
) -> pint.Quantity:
    """A thermal diffusivity (default m²/s)."""
    return _quantity(
        value, unit, '[length] ** 2 / [time]', 'thermal diffusivity')


def heat_transfer_coefficient(
    value: Number, unit: UnitLike = 'watt / (meter ** 2 * kelvin)'
) -> pint.Quantity:
    """A heat-transfer coefficient (default W/(m²·K)). Uses delta temperature."""
    return _quantity(
        value, unit, '[power] / [length] ** 2 / [temperature]',
        'heat transfer coefficient')


def temperature_gradient(
    value: Number, unit: UnitLike = 'kelvin / meter'
) -> pint.Quantity:
    """A temperature gradient (default K/m) — delta-based (K/m, degF/ft)."""
    return _quantity(
        value, unit, '[temperature] / [length]', 'temperature gradient')


# --- converters / guards ------------------------------------------------------
def to(quantity, unit: UnitLike, to_unit: Optional[UnitLike] = None) -> float:
    """Magnitude expressed in a target unit (float). Two call forms:

    ``to(quantity, unit)`` -- express an existing Quantity in ``unit``; or
    ``to(value, from_unit, to_unit)`` -- a value + its unit-string, converted
    to ``to_unit`` (drops the ``Q_(value, from_unit)`` boilerplate).

    >>> to(pressure(1, 'bar'), 'Pa')
    100000.0
    >>> to(1.0, 'bar', 'Pa')
    100000.0
    """
    if to_unit is not None:
        quantity = quantity * ureg(unit)
        unit = to_unit
    return float(quantity.to(unit).magnitude)


def to_si(quantity, unit: Optional[UnitLike] = None) -> float:
    """Magnitude in SI base units (float). Two call forms:

    ``to_si(Q_(value, unit))`` or the boilerplate-free ``to_si(value, unit)``.

    >>> to_si(pressure(1, 'bar'))
    100000.0
    >>> to_si(1.0, 'bar')
    100000.0
    """
    if unit is not None:
        quantity = quantity * ureg(unit)
    return float(quantity.to_base_units().magnitude)


def magnitude(quantity: pint.Quantity) -> float:
    """The bare magnitude of ``quantity`` (float), unit unchanged."""
    return float(quantity.magnitude)


def is_valid_unit(unit_str: UnitLike) -> bool:
    """True if ``unit_str`` is a unit the registry understands.

    A public predicate for validating a user-supplied unit string at an API/GUI
    boundary BEFORE conversion, without a ``try``/``except`` around ``Q_``.

    >>> is_valid_unit('lbf/(100*ft**2)')
    True
    >>> is_valid_unit('not_a_unit')
    False
    """
    # A blank string is a user error at an API/GUI edge, not "dimensionless".
    if not isinstance(unit_str, str) or not unit_str.strip():
        return False
    # Parse as an expression (not a strict Unit) so composite units carrying a
    # numeric factor -- e.g. YP in lbf/(100*ft**2) -- pass; matches the ``ureg``
    # path the 2-arg to_si / to use.
    try:
        ureg(unit_str)
        return True
    except Exception:
        return False


# --- drilling helpers ---------------------------------------------------------
def hydrostatic_gradient(
    mud_weight_q: pint.Quantity, unit: UnitLike = 'psi/ft'
) -> pint.Quantity:
    """Vertical hydrostatic pressure gradient of a static mud column, dP/dz = ρg.

    >>> round(to(hydrostatic_gradient(mud_weight(10, 'ppg')), 'psi/ft'), 4)
    0.5195
    """
    return (mud_weight_q * ureg('standard_gravity')).to(unit)


def mud_weight_from_gradient(
    gradient_q: pint.Quantity, unit: UnitLike = 'ppg'
) -> pint.Quantity:
    """Equivalent mud weight (density) of a hydrostatic gradient, ρ = (dP/dz)/g."""
    return (gradient_q / ureg('standard_gravity')).to(unit)


#: Hydrostatic gradient of a 1 ppg fluid at STANDARD gravity [psi/ft].
#: 1 lb / 231 in^3 x 12 in/ft = 0.05194805..., computed here rather than typed.
#:
#: **Do not quote this to more figures than gravity justifies.** g varies with
#: latitude by about +-0.27% about standard (9.7803 m/s^2 at the equator to
#: 9.8322 at the poles), so the physically achievable range is roughly 0.051809
#: to 0.052083 -- a 0.53% spread. Any digit beyond the fourth is a statement
#: about where the well is, not about arithmetic.
#:
#: **But it mostly CANCELS, which is why the literature does not fuss about it.**
#: Where pressures are expressed as equivalent mud weights -- the industry
#: convention -- the constant divides out of a hydrostatic balance. For the
#: kick-tolerance gas height, ``BHP = PP.g.TD`` and ``FRAC = LOT.g.shoe`` are
#: both gradient-derived, so
#:
#:     h = [rho_mud.(TD - shoe) - (PP.TD - LOT.shoe)] / (rho_mud - rho_gas)
#:
#: contains no g at all; and the Boyle ratio ``FRAC/BHP = (LOT.shoe)/(PP.TD)``
#: cancels it as well. With ideal gas the kick tolerance is COMPLETELY
#: independent of this constant, and welleng's 0.0521 versus the exact value
#: makes no difference to it.
#:
#: g survives only where an ABSOLUTE pressure enters that is not gradient-derived
#: with the same constant -- an annular-pressure-loss term in psi, or a real-gas
#: Z(P, T) evaluation, which needs a true absolute pressure. Those are
#: second-order.
#:
#: (Recorded because it was got wrong once: comparing two models by matching
#: their PSI values, rather than their equivalent mud weights, makes the constant
#: appear to matter by several percent. It is an artefact of the comparison.)
PSI_PER_PPG_PER_FT: float = float(
    (ureg('1 lb/gal') * ureg('standard_gravity')).to('psi/ft').magnitude
)


def gravity_at_latitude(
    latitude_deg: Number, altitude_m: Number = 0.0
) -> float:
    """Normal gravity [m/s^2] at a latitude, WGS84 (Somigliana), with a
    free-air correction for altitude.

    >>> round(gravity_at_latitude(0.0), 5)
    9.78033
    >>> round(gravity_at_latitude(90.0), 5)
    9.83218
    """
    phi = np.radians(np.asarray(latitude_deg, dtype=float))
    sin2 = np.sin(phi) ** 2
    g = 9.7803253359 * (1.0 + 0.00193185265241 * sin2) / np.sqrt(
        1.0 - 0.00669437999013 * sin2
    )
    return float(g - 3.086e-6 * float(altitude_m))


def hydrostatic_gradient_at_latitude(
    mud_weight_q: pint.Quantity,
    latitude_deg: Number,
    unit: UnitLike = 'psi/ft',
    altitude_m: Number = 0.0,
) -> pint.Quantity:
    """Hydrostatic gradient using LOCAL gravity rather than standard gravity.

    A North Sea well (58 deg N) and a Gulf of Mexico well (28 deg N) differ by
    0.26% in gravity. Note this rarely changes an answer -- see the cancellation
    note on :data:`PSI_PER_PPG_PER_FT`. Use it where an absolute pressure is
    genuinely needed, not to "improve" a hydrostatic balance expressed in
    equivalent mud weights, where the constant divides out.

    >>> round(to(hydrostatic_gradient_at_latitude(mud_weight(1, 'ppg'), 58.0),
    ...          'psi/ft'), 7)
    0.0520059
    """
    g = gravity_at_latitude(latitude_deg, altitude_m) * ureg('m/s**2')
    return (mud_weight_q * g).to(unit)


# --- fast boundary converter --------------------------------------------------
# Canonical SI unit per named quantity. welleng engines compute in these; the
# `Units` boundary converts user I/O to/from them. Performance-critical callers
# work in canonical units directly and skip the converter entirely (zero cost);
# a single-well / interactive user pays one trivial conversion at each edge.
CANONICAL: dict = {
    "length": "meter",
    "angle": "radian",
    "pressure": "pascal",
    "density": "kilogram / meter ** 3",
    "force": "newton",
    "torque": "newton * meter",
    "temperature": "kelvin",
}

# Default user system (metric-ish field defaults); override per-quantity.
_DEFAULT_SYSTEM: dict = {
    "length": "meter",
    "angle": "degree",
    "pressure": "pascal",
    "density": "kilogram / meter ** 3",
    "force": "newton",
    "torque": "newton * meter",
    "temperature": "kelvin",
}


class Units:
    """Fast, generic unit-conversion boundary — pint at setup, numpy at runtime.

    welleng engines compute in canonical SI (see :data:`CANONICAL`). This converts
    user inputs to canonical on ingest and canonical to user units on output, and
    ONLY there — performance-critical callers use the canonical core directly and
    bypass this entirely.

    Speed: the affine ``(factor, offset)`` for each unit pair is computed ONCE via
    pint and cached; conversion is then pure numpy arithmetic (``value * factor
    (+ offset)``), scalar or array, with no pint on the hot path. Affine units
    (e.g. temperature) carry a non-zero offset; multiplicative units do not.

    Generic + reusable across welleng modules (survey, drilling, kick, api).

    Examples
    --------
    >>> u = Units(length="ft", angle="degree")
    >>> round(u.to_canonical(1000.0, "length"), 4)        # ft -> m
    304.8
    >>> round(u.from_canonical(3.14159265, "angle"), 3)   # rad -> deg
    180.0
    >>> round(u.convert(100.0, "psi", "bar"), 6)          # generic pair
    6.894757
    """

    def __init__(self, **system: str) -> None:
        self.system: dict = {**_DEFAULT_SYSTEM, **system}
        self._cache: dict = {}

    def _factor_offset(self, src: UnitLike, dst: UnitLike) -> Tuple[float, float]:
        key = (str(src), str(dst))
        fo = self._cache.get(key)
        if fo is None:
            zero = float(ureg.Quantity(0.0, src).to(dst).magnitude)
            one = float(ureg.Quantity(1.0, src).to(dst).magnitude)
            fo = (one - zero, zero)   # affine: y = x * factor + offset
            self._cache[key] = fo
        return fo

    def convert(self, value: Numeric, src: UnitLike, dst: UnitLike) -> Numeric:
        """Convert ``value`` (scalar or ndarray) from ``src`` to ``dst`` units."""
        if str(src) == str(dst):
            return value
        factor, offset = self._factor_offset(src, dst)
        return value * factor + offset if offset else value * factor

    def to_canonical(self, value: Numeric, quantity: str) -> Numeric:
        """User-units ``value`` -> canonical SI for the named ``quantity``."""
        return self.convert(value, self.system[quantity], CANONICAL[quantity])

    def from_canonical(self, value: Numeric, quantity: str) -> Numeric:
        """Canonical-SI ``value`` -> the user's units for the named ``quantity``."""
        return self.convert(value, CANONICAL[quantity], self.system[quantity])
