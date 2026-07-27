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
from typing import Tuple, Union

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


# --- converters / guards ------------------------------------------------------
def to(quantity: pint.Quantity, unit: UnitLike) -> float:
    """Magnitude of ``quantity`` expressed in ``unit`` (float).

    >>> to(pressure(1, 'bar'), 'Pa')
    100000.0
    """
    return float(quantity.to(unit).magnitude)


def to_si(quantity: pint.Quantity) -> float:
    """Magnitude of ``quantity`` in SI base units (float)."""
    return float(quantity.to_base_units().magnitude)


def magnitude(quantity: pint.Quantity) -> float:
    """The bare magnitude of ``quantity`` (float), unit unchanged."""
    return float(quantity.magnitude)


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
#: 9.8322 at the poles), so the physically achievable range of this constant is
#: roughly 0.051809 to 0.052083 -- a 0.53% spread. Any digit beyond the fourth
#: is a statement about where the well is, not about arithmetic.
#:
#: That matters more than it looks for kick tolerance, where the answer is a
#: small difference of two large hydrostatic terms. The amplification is
#: ~20x on a typical case, so LATITUDE ALONE is worth ~10% of the kick
#: tolerance: the same well design tolerates a different influx in the North Sea
#: than in the Gulf of Mexico. The amplification grows without bound as a well
#: approaches balance, i.e. it is largest exactly when the tolerance is smallest.
#:
#: Note ``welleng.kick_tolerance`` uses 0.0521, which is +0.292% and therefore
#: OUTSIDE the planetary range (it exceeds polar gravity). That value is
#: inherited from SPE-208788-PA and reproducing the paper's worked example
#: digit-for-digit depends on it -- validate with the source's own constant, and
#: quote the deviation.
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

    Worth using where the answer is sensitive to it -- see the note on
    :data:`PSI_PER_PPG_PER_FT`. A North Sea well (58 deg N) and a Gulf of Mexico
    well (28 deg N) differ by 0.26% in gravity alone.

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
