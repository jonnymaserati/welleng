"""Tier-0 single-bubble kick-tolerance subpackage (clean-room).

Public single-bubble kick-tolerance closed form (Eqs. A-1...A-9 of the public
SPE paper SPE-208788-PA) with a clean-room Hall & Yarborough (1973) real-gas
Z-factor backend for the Tier-0 pure-methane influx.

Model class & scope (what this is, and is NOT)
----------------------------------------------
This is a **single-bubble, quasi-static, deterministic** kick-tolerance calculator.
One coherent (insoluble) gas influx is sized by the shoe / weak-zone pressure
balance (the closed form) and marched up the annulus with Boyle + real-gas Z +
optional temperature expansion (the migration engine). It is **CONSERVATIVE
(safe-side)**: a single coherent slug imposes the GREATEST shoe pressure, so the
tolerable influx it returns is a LOWER BOUND on what a real dispersed / multiphase
influx would allow. The literature confirms the direction -- dynamic multiphase
simulators return a HIGHER kick tolerance (SPE-208788-PA; Kiani Nassab et al.,
SPE-202426-PA, Fig. 12: static single-bubble 13-16 bbl vs dynamic ~17 bbl).

It is **NOT a transient two-phase flow simulator**: no gas/liquid slip or holdup,
no flow-regime transitions, no gas dissolution into the mud, no transient annulus
temperature (that is OLGA / Drillbench / "Simulator D" territory). Use this as a
fast, reproducible, conservative well-control **barrier check**. A full multiphase
model would only RELAX (raise) the tolerance -- a casing-design margin-recovery
tool, not a safety improvement -- and its transient two-phase hydraulics belong
with a hydraulics kernel, not here.

WORST-CASE PORE PRESSURE -- the stated convention (READ THIS)
--------------------------------------------------------------
**welleng uses PP + kick intensity, floored at the mud hydrostatic.**

    scenario_P_td = max( (PP + kick_intensity) * g * D_td ,  rho_mud * g * D_td )

This is a CHOICE, and it is the single biggest source of disagreement between
kick-tolerance numbers in the industry. Santos & Sonnemann, *Transitional Kick
Tolerance*, SPE-159175-MS (2012), names the two families in common use:

* **PP+KI** -- "Predicted pore pressure + kick intensity". What we implement.
* **MW+KI** -- "Current Mud Density + kick intensity", i.e. the worst case is
  referenced to the fluid in the hole rather than to the predicted pore
  pressure.

They give different answers, and the paper is blunt about the consequence: *"one
big issue in the industry today is the lack of consistency about KT calculation,
leading to significant confusion and increased risk"*, from *"methods which
provide widely diverging results"*. It also warns that "several, slightly
different ways" exist WITHIN each family -- our mud-hydrostatic floor is one such
variation, so quoting "PP+KI" alone does not pin the number either.

Quote the convention with the number. Two engineers can both be right and
disagree by a wide margin, and neither will know why unless it is stated.

Not implemented: **Transitional Kick Tolerance** (the same paper) -- KT as a
curve over the whole hole section rather than a single value at section TD. The
paper's criticism that the industry focuses "almost exclusively on the KT at the
end of a hole section" applies to this package as it stands.

STRING POSITION -- the stated convention (READ THIS)
----------------------------------------------------
**A kick tolerance is a number about a stated configuration, and this is ours.**
The engine can place the string anywhere; the convention below is what the
REPORTED quantity means, not what the code is capable of. Quote it with the
number.

* **Drilling case** (``drill_kick``, and the sectioned solvers used for it) --
  the **BHA is ON BOTTOM**. There is no string-position variable: the geometry is
  fixed and the kick margin is carried in the bottom-hole pressure. Nothing is
  searched over, because nothing moves.

* **Swab case** (``swab_kick``, tripping) -- the string position IS a variable,
  because the BHA can intercept the bubble at any depth as it is run in. The
  worst interception is **COMPUTED, not assumed.**

  Industry commonly assumes the worst case is *bubble top at the shoe with the
  bit at the bubble's bottom*. We do not assume it, because we can afford to
  solve for it -- and it is **NOT conservative**. The worst case has the bubble
  top at the BINDING depth, which is the shoe only when the shoe is what binds.
  Measured against the true worst: the shoe assumption is +0.68%, +0.79% and
  **+25.42%** high on a well whose weak zone sits below the shoe. It has the
  right shape and the wrong depth.

  The true worst is determinate, not a search: the gas length ``L(d)`` is
  density-driven and capacity-independent, so for each candidate binding depth
  ``d`` the worst string position is ``d + L(d)``.

UNITS -- the field-units contract (READ THIS)
---------------------------------------------
**Every input and output of this subpackage is in US oilfield field units**,
deliberately, to match the validation papers (SPE-208788, NOGEPA, API RP 59) and
industry KT practice. This diverges from welleng's SI convention: it is the ONE
field-units subpackage. Unit conversion is a BOUNDARY concern -- a caller (API, GUI,
notebook) converts to/from these units with Pint (``welleng.units`` = a pint
registry), the engine itself never converts. The canonical units are:

    quantity              unit        engine parameters (examples)
    --------------------  ----------  -----------------------------------------
    density / mud weight  ppg         rho_mud, rho_mud_ppg, rho_gas_s
    pressure (equiv. MW)  ppg         PP, kick_intensity, P_lot, pp/fp profiles
    pressure (absolute)   psi         P_apl, bhp_psi, and all returned pressures
    depth / length        ft          D_td, D_lot, section top/bottom, profile depths
                                       -- ALL depths are TRUE VERTICAL DEPTH (TVD)
    temperature           degF        T_s, T_td  (KickInputs)
    temperature           degR        T_bh in gas_bh_state; temp-profile callables
    volume                bbl         influx, kick tolerance, kt_threshold
    annular capacity      bbl/ft      V_dpa, WellSection.annular_capacity_bbl_per_ft
    inclination           deg         inc_shoe
    Z factor / fractions  -           Z_s, Z_td, gas mole fractions (dimensionless)

``gas_bh_state`` tuple = ``(P_bh [psi], T_bh [degR], Z [-], rho_gas [ppg])`` (any
element may be ``None`` -> computed). PP / fracture / kick-intensity are
mud-weight-EQUIVALENT densities (ppg), NOT absolute pressures. Temperatures are
degF on ``KickInputs`` but degR inside ``gas_bh_state`` and the temp-profile
callables -- note the difference. Per-parameter units are also restated in each
function/dataclass docstring.

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
    DEFAULT_MODEL_REVISION,
    MODEL_REVISIONS,
    KickInputs,
    KickResult,
    drill_kick,
    swab_kick,
    influx_column,
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
from .gas_z_coolprop import (
    fluid_z_density,
    fluid_presets,
    fluid_aliases,
    FLUID_PRESETS,
)
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
    _max_influx_circulated,
    pressure_at_depth,
    linear_temp_profile,
)
# Analytical (breakpoint) kick-tolerance solver: the migration-form KT evaluated
# only at the breakpoints of P(gas position) -- the exact worst position, no march.
from .analytical import (
    analytical_kick_tolerance, AnalyticalKickTolerance,
    max_influx_contained_at_surface, swab_worst_bit,
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
    "MODEL_REVISIONS",
    "DEFAULT_MODEL_REVISION",
    "KickInputs",
    "KickResult",
    "drill_kick",
    "swab_kick",
    "influx_column",
    "resolve_gas_properties",
    "evaluate_envelope",
    "EnvelopeResult",
    "hall_yarborough_z",
    "gas_density_ppg",
    "methane_properties",
    "fluid_z_density",
    "fluid_presets",
    "fluid_aliases",
    "FLUID_PRESETS",
    "WellSection",
    "MigrationStep",
    "MigrationResult",
    "KickToleranceResult",
    "migrate",
    "pressure_at_depth",
    "linear_temp_profile",
    "analytical_kick_tolerance",
    "AnalyticalKickTolerance",
    "max_influx_contained_at_surface",
    "swab_worst_bit",
    "annular_capacity",
    "cased_section",
    "open_hole_section",
    "nogepa_drilling_kick_tolerance",
    "NogepaResult",
]
