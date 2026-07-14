"""Single-bubble kick-tolerance margin module (clean-room).

Implements the public single-bubble kick-tolerance closed form, Eqs. A-1...A-9,
as derived in Appendix A of the public SPE paper SPE-208788-PA (Thorogood et al.,
"Kick Tolerance ...", SPE Drilling & Completion, September 2022, pp. 242-243).
The equations are transcribed faithfully from that public source only.

Clean-room scope
----------------
Gas properties (the real-gas compressibility factors ``Z_s``, ``Z_td`` and the
influx gas density at the shoe ``rho_gas_s``) are COMPUTED by the clean-room
Hall & Yarborough (1973) backend in ``gas_z.py`` from the influx composition
(Tier 0 = pure methane) and the model's own pressure/temperature conditions.
They may still be INJECTED as overrides (any of ``Z_s``, ``Z_td``,
``rho_gas_s`` left as ``None`` is computed; a numeric value is used verbatim),
which keeps mixtures / CO2-bearing systems (e.g. a CoolProp HEOS backend)
pluggable. No external, third-party, or private gas-property data source is
referenced.

Where the gas properties are evaluated
--------------------------------------
The influx gas is a single migrating bubble. Its properties are evaluated once
per scenario at two physical stations and shared by both the drill and swab
cases (as in the reference derivation):

  * TD station -- (P_td, T_td), with P_td the A-1 bottom-hole pressure at the
    maximum-credible pore pressure. Gives ``Z_td`` and (unused) ``rho_gas_td``.
  * Shoe station -- (P_shoe, T_s). ``P_shoe`` is the influx-gas pressure at the
    casing shoe: the bottom-hole pressure reduced by the static influx-gas
    column over the open-hole section (a standard, source-free gas-gradient
    step). Gives ``Z_s`` and ``rho_gas_s``.

Note on the reference case: the reference paper's printed shoe Z (1.123) sits
above the value at the shoe *fracture* pressure P_lot - P_apl (~1.04); it
corresponds to a near-bottom-hole influx-gas pressure, which is what the static
gas column recovers. Reproducing the paper's Table-1 gas properties therefore
lands the kick volumes within ~1-1.3% -- the paper's own Table-1 inconsistency,
not a backend error.

Two first-class cases (kept SEPARATE by design)
-----------------------------------------------
  * ``drill_kick``  -- the drillability gate. A-7 tolerable-influx capacity
    evaluated at the maximum-credible pore pressure (PP + kick intensity).
    This is the section PASS/FAIL.
  * ``swab_kick``   -- the unmitigated free-trip limit (A-8), i.e. A-7 with the
    bottom-hole pressure set to the mud hydrostatic. This is a MITIGABLE
    operational limit: a swab failure flags "pump out of the hole" -- and pumping
    out is itself the drill case, which is already assessed. Swab NEVER overrides
    drill; the two are reported separately and are never blind-min'd / AND-gated.

Units
-----
Depths ft; densities ppg; pressures psi; temperatures input in degF and
converted to absolute degR internally; annular capacity ``V_dpa`` in bbl/ft so
that capacities come out in bbl. Gravitational constant g = 0.0521
psi.ppg^-1.ft^-1 (oilfield units); atmospheric pressure P_atm = 14.7 psi.

Note (from the derivation): the gas density at TD, rho_gas_td, appears in NO
equation of the closed form -- only rho_gas_s does (via the A-5 constant A).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Optional

from .gas_z import gas_density_ppg, hall_yarborough_z

# --- Constants (public, oilfield units) -------------------------------------
G_PSI_PER_PPG_FT = 0.0521      # gravitational constant g  [psi.ppg^-1.ft^-1]
P_ATM_PSI = 14.7               # atmospheric pressure P_atm [psi]
RANKINE_OFFSET = 460.0         # degF -> degR (paper convention; A back-solves
                               #               to ~241.2 with this offset)


def ppg_to_psi(rho_ppg: float, depth_ft: float) -> float:
    """Hydrostatic / gradient pressure: g * rho * depth  [psi]."""
    return G_PSI_PER_PPG_FT * rho_ppg * depth_ft


def fahrenheit_to_rankine(t_degf: float) -> float:
    """Absolute temperature in degR (paper uses a +460 offset)."""
    return t_degf + RANKINE_OFFSET


@dataclass
class KickInputs:
    """Operational + gas-property inputs for a single-bubble kick-tolerance case.

    Operational (design-box) inputs:
        rho_mud        : drilling fluid density                       [ppg]
        PP             : formation pore pressure (mud-weight equiv.)  [ppg]
        kick_intensity : kick intensity added to PP for max-credible  [ppg]
        P_lot          : formation strength / LOT at the shoe         [ppg]
        P_apl          : annular pressure loss                        [psi]
        D_td           : measured/true depth at TD                    [ft]
        D_lot          : shoe (casing / LOT) depth                    [ft]
        T_s            : temperature at the shoe                      [degF]
        T_td           : temperature at TD                            [degF]
        V_dpa          : annular capacity, drillpipe-in-hole          [bbl/ft]

    Gas-property inputs (COMPUTED by the clean-room Hall-Yarborough backend when
    left as ``None``; a numeric value overrides -- e.g. a mixture backend):
        Z_s        : real-gas Z factor at the shoe station            [-]
        Z_td       : real-gas Z factor at the TD station              [-]
        rho_gas_s  : influx gas density at the shoe station           [ppg]

    Design threshold:
        kt_threshold : required tolerable-kick volume (margin datum)  [bbl]

    Trajectory (deviated wells):
        inc_shoe : hole inclination at the shoe [deg]; 0.0 = vertical (the
            SPE-208788 Table-1 case). Converts the gas column's VERTICAL height
            H_gas to an along-hole length L_gas = H_gas / cos(inc) before it
            multiplies the per-MD annular capacity V_dpa (Nassab et al.,
            SPE-202426-PA). Net effect: A -> A / cos(inc_shoe). D_td / D_lot are
            TRUE VERTICAL depths; all hydrostatic terms are TVD-referenced.
    """

    rho_mud: float
    PP: float
    kick_intensity: float
    P_lot: float
    P_apl: float
    D_td: float
    D_lot: float
    T_s: float
    T_td: float
    V_dpa: float
    Z_s: Optional[float] = None
    Z_td: Optional[float] = None
    rho_gas_s: Optional[float] = None
    kt_threshold: float = 25.0
    inc_shoe: float = 0.0


@dataclass
class KickResult:
    """Result of a single kick-tolerance case."""

    case: str            # "drill" or "swab"
    P_td: float          # bottom-hole pressure used            [psi]
    A: float             # A-5 constant                          [bbl]
    B: float             # A-6 constant                          [psi]
    capacity: float      # tolerable influx volume (A-7 / A-8)   [bbl]
    threshold: float     # required tolerable-kick volume        [bbl]
    margin: float        # capacity - threshold                  [bbl]
    passed: bool         # margin >= 0
    pp_at_threshold: float  # A-9 pore pressure at kt_threshold  [ppg]


# --- Gas-property backend (clean-room Hall & Yarborough 1973) ---------------

def scenario_P_td(inp: KickInputs) -> float:
    """A-1 bottom-hole pressure at the maximum-credible pore pressure [psi].

    The single scenario pressure at which the influx gas properties are
    evaluated (shared by the drill and swab cases).
    """
    P_pp_psi = ppg_to_psi(inp.PP + inp.kick_intensity, inp.D_td)
    return max(P_pp_psi, ppg_to_psi(inp.rho_mud, inp.D_td))


def _shoe_gas_pressure(inp: KickInputs, P_td: float) -> float:
    """Influx-gas pressure at the casing shoe [psi].

    ``P_td`` reduced by the static influx-gas column over the open-hole section
    (D_td - D_lot). The column is integrated with the Hall-Yarborough density
    at the mean pressure/temperature and converged by fixed-point iteration.
    """
    T_s_r = fahrenheit_to_rankine(inp.T_s)
    T_td_r = fahrenheit_to_rankine(inp.T_td)
    interval = inp.D_td - inp.D_lot
    P_shoe = P_td
    for _ in range(64):
        p_mean = 0.5 * (P_td + P_shoe)
        t_mean = 0.5 * (T_td_r + T_s_r)
        z_mean = hall_yarborough_z(p_mean, t_mean)
        rho_mean = gas_density_ppg(p_mean, t_mean, z_mean)
        P_new = P_td - G_PSI_PER_PPG_FT * rho_mean * interval
        if abs(P_new - P_shoe) < 1e-6:
            P_shoe = P_new
            break
        P_shoe = P_new
    return P_shoe


def resolve_gas_properties(inp: KickInputs) -> tuple[float, float, float]:
    """Return (Z_s, Z_td, rho_gas_s), computing any left as ``None``.

    TD station:   (P_td, T_td)  -> Z_td.
    Shoe station: (P_shoe, T_s) -> Z_s, rho_gas_s.

    Injected numeric values are used verbatim (mixture-backend override).
    """
    P_td = scenario_P_td(inp)
    T_s_r = fahrenheit_to_rankine(inp.T_s)
    T_td_r = fahrenheit_to_rankine(inp.T_td)

    Z_td = inp.Z_td if inp.Z_td is not None else hall_yarborough_z(P_td, T_td_r)

    if inp.Z_s is None or inp.rho_gas_s is None:
        P_shoe = _shoe_gas_pressure(inp, P_td)
        Z_s_c = hall_yarborough_z(P_shoe, T_s_r)
        rho_gas_s_c = gas_density_ppg(P_shoe, T_s_r, Z_s_c)
    else:
        Z_s_c = rho_gas_s_c = None  # not needed; both injected

    Z_s = inp.Z_s if inp.Z_s is not None else Z_s_c
    rho_gas_s = inp.rho_gas_s if inp.rho_gas_s is not None else rho_gas_s_c
    return Z_s, Z_td, rho_gas_s


# --- Closed-form building blocks --------------------------------------------

def constant_A(inp: KickInputs) -> float:
    """A-5:  A = (P_lot - P_apl) * T_td * Z_td * V_dpa
                 / [ g * T_s * Z_s * (rho_mud - rho_gas_s) ]   [bbl]

    P_lot is converted to a pressure at the shoe depth (g * P_lot * D_lot).
    Temperatures are absolute (degR). Gas properties (Z_s, Z_td, rho_gas_s) are
    computed by the clean-room Hall-Yarborough backend unless injected.

    Deviated wells (Nassab et al., SPE-202426-PA): the gas column's vertical
    height H_gas occupies an along-hole length L_gas = H_gas / cos(inc_shoe),
    which is what multiplies the per-MD annular capacity V_dpa. This scales A by
    1 / cos(inc_shoe). inc_shoe = 0 (vertical) recovers the SPE-208788 Table-1
    form exactly. (Exact survey-integral form -- absorbing BHA change-of-section
    -- is a documented follow-up; this constant-inclination form is the standard
    published convention.)
    """
    Z_s, Z_td, rho_gas_s = resolve_gas_properties(inp)
    P_lot_psi = ppg_to_psi(inp.P_lot, inp.D_lot)
    T_s_r = fahrenheit_to_rankine(inp.T_s)
    T_td_r = fahrenheit_to_rankine(inp.T_td)
    num = (P_lot_psi - inp.P_apl) * T_td_r * Z_td * inp.V_dpa
    den = G_PSI_PER_PPG_FT * T_s_r * Z_s * (inp.rho_mud - rho_gas_s)
    return (num / den) / cos(radians(inp.inc_shoe))


def constant_B(inp: KickInputs) -> float:
    """A-6:  B = P_lot - P_apl + g * rho_mud * (D_td - D_lot)   [psi]

    (the maximum pore pressure at which the tolerable influx is zero).
    """
    P_lot_psi = ppg_to_psi(inp.P_lot, inp.D_lot)
    return P_lot_psi - inp.P_apl + G_PSI_PER_PPG_FT * inp.rho_mud * (inp.D_td - inp.D_lot)


def P_td_from_A1(inp: KickInputs, P_pp_psi: float) -> float:
    """A-1:  P_td = max{P_pp, D_td * g * rho_mud}   [psi]."""
    return max(P_pp_psi, ppg_to_psi(inp.rho_mud, inp.D_td))


def influx_volume_A7(A: float, B: float, P_td: float) -> float:
    """A-7:  V_gas_td = A * (B - P_td) / (P_td + P_atm)   [bbl].

    Tolerable single-bubble influx volume (zero when P_pp == B).
    """
    return A * (B - P_td) / (P_td + P_ATM_PSI)


def pp_at_threshold_A9(A: float, B: float, V_threshold: float) -> float:
    """A-9:  P_td = (A * B - V_gas_td * P_atm) / (V_gas_td + A)   [psi],

    the bottom-hole pressure corresponding to a chosen kick-tolerance volume,
    returned as an equivalent pore-pressure mud weight [ppg] at D_td.
    """
    return (A * B - V_threshold * P_ATM_PSI) / (V_threshold + A)


def _psi_to_ppg(p_psi: float, depth_ft: float) -> float:
    """Inverse gradient: pressure -> equivalent mud weight [ppg] at depth."""
    return p_psi / (G_PSI_PER_PPG_FT * depth_ft)


# --- The two first-class cases ----------------------------------------------

def drill_kick(inp: KickInputs) -> KickResult:
    """Drillability gate (section PASS/FAIL).

    A-7 tolerable-influx capacity at the MAXIMUM-CREDIBLE pore pressure,
    P_pp = (PP + kick_intensity) expressed as a pressure at D_td, per A-1.
    """
    A = constant_A(inp)
    B = constant_B(inp)
    P_pp_psi = ppg_to_psi(inp.PP + inp.kick_intensity, inp.D_td)
    P_td = P_td_from_A1(inp, P_pp_psi)
    capacity = influx_volume_A7(A, B, P_td)
    margin = capacity - inp.kt_threshold
    P_td_thresh = pp_at_threshold_A9(A, B, inp.kt_threshold)
    return KickResult(
        case="drill",
        P_td=P_td,
        A=A,
        B=B,
        capacity=capacity,
        threshold=inp.kt_threshold,
        margin=margin,
        passed=margin >= 0.0,
        pp_at_threshold=_psi_to_ppg(P_td_thresh, inp.D_td),
    )


def swab_kick(inp: KickInputs) -> KickResult:
    """Unmitigated free-trip limit (A-8) -- a MITIGABLE operational limit.

    A-8 is A-7 with the bottom-hole pressure set to the mud hydrostatic,
    P_td = g * rho_mud * D_td (pore pressure <= mud hydrostatic). A swab
    failure means "pump out of the hole" -- which is the drill case, assessed
    separately. Swab NEVER overrides drill and is never blind-min'd with it.
    """
    A = constant_A(inp)
    B = constant_B(inp)
    P_td = ppg_to_psi(inp.rho_mud, inp.D_td)  # A-8 substitution
    capacity = influx_volume_A7(A, B, P_td)   # == A-8 closed form
    margin = capacity - inp.kt_threshold
    P_td_thresh = pp_at_threshold_A9(A, B, inp.kt_threshold)
    return KickResult(
        case="swab",
        P_td=P_td,
        A=A,
        B=B,
        capacity=capacity,
        threshold=inp.kt_threshold,
        margin=margin,
        passed=margin >= 0.0,
        pp_at_threshold=_psi_to_ppg(P_td_thresh, inp.D_td),
    )


def annular_capacity_dpa(hole_id_in: float, pipe_od_in: float) -> float:
    """Convenience: annular capacity V_dpa = (hole^2 - pipe^2) / 1029.4 [bbl/ft].

    Per the A-3 simplification the influx occupies the drillpipe-in-hole annulus
    at the shoe (pipe_od = drillpipe OD). This is a geometry helper only; V_dpa
    may equally be supplied directly as an input.
    """
    return (hole_id_in ** 2 - pipe_od_in ** 2) / 1029.4
