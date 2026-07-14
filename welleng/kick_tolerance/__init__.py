"""Tier-0 single-bubble kick-tolerance subpackage (clean-room).

Public single-bubble kick-tolerance closed form (Eqs. A-1...A-9 of the public
SPE paper SPE-208788-PA) with a clean-room Hall & Yarborough (1973) real-gas
Z-factor backend for the Tier-0 pure-methane influx.

Layout
------
* ``core``        -- the margin logic (KickInputs/KickResult, drill/swab cases).
* ``gas_z``       -- clean-room Hall & Yarborough (1973) Z-factor backend.
* ``envelope``    -- deterministic worst-credible-case barrier envelope.
* ``monotonicity``-- per-case SymPy monotonicity analysis feeding the envelope.

``core`` and ``gas_z`` depend only on the standard library plus numpy/scipy, so
they import with no optional dependencies. ``envelope`` and ``monotonicity``
require SymPy (the optional ``kick`` extra); when SymPy is absent the envelope
API is exposed as a stub that raises a helpful ImportError on use, so that
``import welleng`` (and the core kick-tolerance API) still works without it.
"""

from .core import (
    KickInputs,
    KickResult,
    drill_kick,
    swab_kick,
    resolve_gas_properties,
)
from .gas_z import (
    hall_yarborough_z,
    gas_density_ppg,
    methane_properties,
)
# Optional CoolProp real-EOS mixture backend (CO2 / CCUS). The import itself is
# safe without CoolProp -- the dependency is only imported when fluid_z_density
# is actually called (raises a helpful ImportError then, pointing at welleng[kick]).
from .gas_z_coolprop import fluid_z_density
# Gas-migration engine: bubble migration up the section-wise annulus checked
# against the pore/fracture (PPFP) envelope at every exposed depth (conservative,
# safe-side). numpy-only -- no optional dependency. The step trajectory is the
# animation data.
from .migration import (
    WellSection,
    MigrationStep,
    MigrationResult,
    KickToleranceResult,
    migrate,
    max_influx_circulated,
    pressure_at_depth,
    linear_temp_profile,
)
# Catalogue-backed geometry: true annular capacity (bore - string), casing IDs
# from the API-5CT catalogue. catalog is imported lazily inside the builders.
from .geometry import annular_capacity, cased_section, open_hole_section
# NOGEPA-50 static single-shoe formula (the mandated baseline the migration
# engine's static reduction reproduces).
from .nogepa import nogepa_drilling_kick_tolerance, NogepaResult

try:  # envelope/monotonicity need SymPy (optional 'kick' extra)
    from .envelope import evaluate_envelope, EnvelopeResult
except ImportError as _envelope_import_error:  # pragma: no cover - optional dep
    _ENVELOPE_IMPORT_ERROR = _envelope_import_error

    def evaluate_envelope(*args, **kwargs):
        """Stub raised when the optional ``kick`` extra (SymPy) is missing."""
        raise ImportError(
            "evaluate_envelope requires the optional 'kick' extra "
            "(SymPy). Install it with: pip install welleng[kick]"
        ) from _ENVELOPE_IMPORT_ERROR

    EnvelopeResult = None

__all__ = [
    "KickInputs",
    "KickResult",
    "drill_kick",
    "swab_kick",
    "resolve_gas_properties",
    "evaluate_envelope",
    "EnvelopeResult",
    "hall_yarborough_z",
    "gas_density_ppg",
    "methane_properties",
    "fluid_z_density",
    "WellSection",
    "MigrationStep",
    "MigrationResult",
    "KickToleranceResult",
    "migrate",
    "max_influx_circulated",
    "pressure_at_depth",
    "linear_temp_profile",
    "annular_capacity",
    "cased_section",
    "open_hole_section",
    "nogepa_drilling_kick_tolerance",
    "NogepaResult",
]
