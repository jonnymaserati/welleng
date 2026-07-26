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
  * Influx station -- (P_shoe, T_influx). ``P_shoe`` is the influx-gas pressure
    at the casing shoe: the bottom-hole pressure reduced by the static
    influx-gas column over the open-hole section (a standard, source-free
    gas-gradient step). Gives ``Z_s`` and ``rho_gas_s``.

    ``T_influx`` is REVISION-DEPENDENT (see ``MODEL_REVISIONS``). As published
    it is the shoe temperature ``T_s``; from welleng 0.26.0 the default is the
    INFLUX COLUMN's mean temperature, because the column hangs BELOW the shoe
    and the shoe is its cool end. Evaluating at the shoe made the gas denser
    than the column, inflating A and OVERSTATING the tolerable influx by ~3% on
    the Table-1 case (anti-conservative). See ``_influx_temperature``.

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
from functools import lru_cache
from math import cos, radians
from typing import Optional

from .gas_z import gas_density_ppg, hall_yarborough_z, methane_properties

# --- Constants (public, oilfield units) -------------------------------------
G_PSI_PER_PPG_FT = 0.0521      # gravitational constant g  [psi.ppg^-1.ft^-1]
P_ATM_PSI = 14.7               # atmospheric pressure P_atm [psi]
RANKINE_OFFSET = 460.0         # degF -> degR (paper convention; A back-solves
                               #               to ~241.2 with this offset)


# --- Model revisions --------------------------------------------------------
#
# A kick-tolerance number ends up in a signed-off well programme, so a result
# computed under an earlier model must stay reproducible EXACTLY after the model
# changes, by naming the revision it was computed under (a well-control
# reproducibility requirement). Hence a named, FROZEN revision rather than a boolean: a boolean
# cannot carry a second change, and a revision identifier is what a report can
# cite. The revision travels on ``KickResult``, so a stored calculation records
# what produced it.
#
# Published revisions are FROZEN. Never re-tune one; add a new identifier.
#
# This is deliberately NOT folded into ``ideal_gas`` or ``gas_density_mode``.
# Those are user-selectable PHYSICS OPTIONS chosen on the merits of a case; a
# revision is a REPRODUCIBILITY ANCHOR that freezes every semantic in the path.
# Overloading them would let pinning a revision silently override a physics
# choice the user made deliberately.
MODEL_REVISIONS = {
    # As published: the influx state is lumped at the SHOE temperature T_s.
    # Reproduces SPE-208788-PA Table 1 and is the validation anchor.
    "spe-208788": "influx gas state evaluated at the shoe temperature T_s",
    # The influx column hangs BELOW the shoe, so the shoe is its COOL end. The
    # state is lumped at the column-mean temperature instead. See
    # ``_influx_temperature``.
    "column-mean-2026": "influx gas state evaluated at the gas-column mean "
                        "temperature (default from welleng 0.26.0)",
}
DEFAULT_MODEL_REVISION = "column-mean-2026"


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
                         DRILL CASE ONLY. The swab case IGNORES kick_intensity
                         by design: its bottom-hole pressure is the mud
                         hydrostatic, independent of PP and KI (Nassab
                         SPE-202426-PA, Eqs 8-9). See ``swab_kick``.
        P_lot          : formation strength / LOT at the shoe         [ppg]
        P_apl          : surface-side pressure margin at the shoe      [psi]
                         (applied choke/back pressure + safety margin). Annular
                         friction is conventionally ZERO here (well killing is
                         circulated at a reduced/kill rate, so annular dP is small);
                         a user who wants to include it simply ADDS it to this term.
        D_td           : measured/true depth at TD                    [ft]
        D_lot          : shoe (casing / LOT) depth                    [ft]
        T_s            : temperature at the shoe                      [degF]
        T_td           : temperature at TD                            [degF]
        V_dpa          : annular capacity, drillpipe-in-hole          [bbl/ft]

    Gas-property inputs (COMPUTED when left as ``None``; a numeric value
    overrides). Default backend = clean-room Hall-Yarborough for pure methane;
    set ``fluid`` to use the CoolProp real-EOS mixture backend instead:
        Z_s        : real-gas Z factor at the shoe station            [-]
        Z_td       : real-gas Z factor at the TD station              [-]
        rho_gas_s  : influx gas density at the shoe station           [ppg]
        fluid      : optional gas COMPOSITION (mole fractions), e.g.
                     {"Methane": 0.9, "CO2": 0.1}. When set, Z/density are
                     computed via CoolProp (real EOS, CO2/CCUS mixtures) --
                     requires welleng[kick]. None => pure-methane Hall-Yarborough.

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
    fluid: Optional[dict] = None
    kt_threshold: float = 25.0
    inc_shoe: float = 0.0
    model_revision: str = DEFAULT_MODEL_REVISION
    # Frozen model revision (see MODEL_REVISIONS). Pin ``"spe-208788"`` to
    # reproduce a number computed before welleng 0.26.0, or the paper's Table 1.
    ideal_gas: bool = False
    # Ideal-gas REFERENCE mode: Z == 1 everywhere and isothermal (the influx-gas
    # column and expansion use a single temperature). Reproduces a textbook
    # single-bubble Boyle's-law kick-tolerance hand-calc / spreadsheet exactly,
    # for cross-checking welleng against ideal-gas references. NOT for design:
    # it discards real-gas compressibility (Hall-Yarborough / CoolProp), which
    # is the physically correct default. An explicit Z_s/Z_td/rho_gas_s override
    # still wins over this flag.

    @classmethod
    def from_survey(cls, survey, shoe_md: float, td_md: float, **params):
        """Build inputs from a welleng ``Survey``, reading the trajectory-derived
        quantities off the survey so callers don't hand-transcribe them.

        ``D_lot`` / ``D_td`` (true vertical depths at the shoe and TD) and
        ``inc_shoe`` (hole inclination at the shoe) are interpolated from the
        survey at ``shoe_md`` / ``td_md`` via min-curvature (``interpolate_md``);
        every other input (``rho_mud``, ``PP``, ``P_lot``, ``V_dpa``, ...) is
        passed through in ``params``. The survey is duck-typed — any object with
        ``interpolate_md(md) -> node`` exposing ``pos_nev`` and ``inc_deg`` works.

        ``Node`` has no direct ``.tvd``; its position is ``pos_nev = [North, East,
        Vertical]`` (welleng NEV convention), so ``pos_nev[2]`` IS the TVD. This is
        the GLOBAL (datum-referenced, ``start_nev``-inclusive) TVD -- which is the
        correct depth for the hydrostatic terms (the mud column is referenced to the
        surface datum, not the survey start). It equals ``survey.tvd`` only at a zero
        datum (``survey.tvd`` is the LOCAL depth, per the ``Survey`` docstring); on a
        datum-shifted / sidetrack survey they differ by ``start_nev``, and the global
        ``pos_nev[2]`` is the one KT wants. Reduces to MD for a vertical well.
        """
        shoe = survey.interpolate_md(shoe_md)
        td = survey.interpolate_md(td_md)
        return cls(
            D_lot=float(shoe.pos_nev[2]),   # pos_nev = [N, E, TVD] -> [2] is TVD
            D_td=float(td.pos_nev[2]),
            inc_shoe=float(shoe.inc_deg),
            **params,
        )


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
    model_revision: str = DEFAULT_MODEL_REVISION
    # The frozen revision that PRODUCED this number. Travels with the result so
    # a stored calculation can be re-run verbatim; cite it in any report.
    T_influx: Optional[float] = None
    # Influx temperature the gas state was evaluated at [degF] -- T_s under
    # "spe-208788", the column mean under "column-mean-2026". Reported because
    # it is the quantity the revision changes.
    rho_influx: Optional[float] = None
    # Influx gas density the A-5 constant ACTUALLY used [ppg]. NOT necessarily
    # the density at the shoe: under "column-mean-2026" it is the column-mean
    # value, and an injected ``rho_gas_s`` overrides either.
    H_gas: Optional[float] = None
    # Influx-column height at this capacity [ft TVD], from A-2 with
    # ``rho_influx``. Reported under BOTH revisions.
    #
    # ``(H_gas, rho_influx)`` are the pair a caller needs to draw the limiting
    # pressure profile, and they are the ONLY pair that closes on the fracture
    # pressure exactly -- see ``influx_column`` for the identity and for why
    # back-deriving H from ``capacity`` instead does not close.


# --- Gas-property backend (clean-room Hall & Yarborough 1973) ---------------

def scenario_P_td(inp: KickInputs) -> float:
    """A-1 DRILL bottom-hole pressure at the maximum-credible pore pressure [psi].

    The pressure at which the DRILL influx gas properties are evaluated. This is
    the drill scenario (PP + kick intensity, floored at mud hydrostatic); the
    SWAB case stations its gas at its own bottom-hole pressure (mud hydrostatic)
    instead -- see ``resolve_gas_properties``/``constant_A`` ``P_td`` and
    ``swab_kick``. (Sharing this pressure with swab leaked kick_intensity into
    the swab number, anti-conservatively -- Nassab SPE-202426-PA Eqs 8-9.)
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
    # Ideal-gas reference mode: Z == 1, isothermal (single temperature T_s).
    t_mean = T_s_r if inp.ideal_gas else 0.5 * (T_td_r + T_s_r)
    interval = inp.D_td - inp.D_lot
    P_shoe = P_td
    for _ in range(64):
        p_mean = 0.5 * (P_td + P_shoe)
        z_mean = 1.0 if inp.ideal_gas else hall_yarborough_z(p_mean, t_mean)
        rho_mean = gas_density_ppg(p_mean, t_mean, z_mean)
        P_new = P_td - G_PSI_PER_PPG_FT * rho_mean * interval
        if abs(P_new - P_shoe) < 1e-6:
            P_shoe = P_new
            break
        P_shoe = P_new
    return P_shoe


def _influx_temperature(inp: KickInputs, P_td: float) -> tuple[float, float]:
    """Temperature the influx gas state is evaluated at [degR], and H_gas [ft].

    The A-5 constant's ``T_s``, ``Z_s`` and ``rho_gas_s`` describe the INFLUX,
    at the condition A-2 defines: gas top at the casing shoe. The column
    therefore hangs BELOW the shoe, over the interval that carries the
    geothermal gradient -- so the shoe is its COOL END, not a representative
    value for it. Evaluating there makes the gas denser than the column, which
    SHRINKS the ``(rho_mud - rho_gas_s)`` deficit in A-5's denominator, inflates
    A, and passes straight through A-7 as an OVERSTATED tolerable influx. The
    error is anti-conservative (raised by an external reviewer, 2026-07-26).

    Note the asymmetry this removes: the pressure side already integrates the
    column (``_shoe_gas_pressure``); only the temperature side did not.

    Support in the primary sources: SPE-202426-PA (Nassab et al.) evaluates its
    worked influx at the AVERAGE across the influx -- 89 degC at the top and
    92 degC at the bottom giving "an average influx temperature of approximately
    91 degC" -- i.e. the symbol denotes the influx's state, not the temperature
    at the shoe depth. NOGEPA-50 Section 5.4.3's "constant temperature is
    conservative" is a DIFFERENT comparison (holding the influx at bottomhole
    temperature as it migrates up the hole) and does not bear on which
    temperature represents the static column.

    The column height is an OUTPUT -- ``H = (B - P_td) / [g (rho_mud -
    rho_gas)]`` from A-2, and ``rho_gas`` depends on the temperature at the
    column's mid-depth -- so this is a fixed point. It converges monotonically
    in a handful of iterations from H = 0 (the shoe value, i.e. the previous
    model), and the correction is signed: a warmer column always gives a
    SMALLER A.

    The ``"spe-208788"`` revision (and ``ideal_gas``, which is isothermal by
    definition) returns ``T_s`` unchanged.
    """
    T_s_r = fahrenheit_to_rankine(inp.T_s)
    if inp.model_revision == "spe-208788" or inp.ideal_gas:
        return T_s_r, None
    interval = inp.D_td - inp.D_lot
    if interval <= 0.0 or inp.T_td == inp.T_s:
        return T_s_r, None                             # nothing to average over
    # A single evaluation costs ~7 gas-property solves, and the closed form asks
    # for the same station three times per case (constant_A, its
    # resolve_gas_properties, and the result's own reporting). Memoised on the
    # scalars it actually depends on, which also keeps an envelope sweep cheap.
    return _influx_temperature_impl(
        inp.T_s, inp.T_td, inp.D_lot, inp.D_td, inp.rho_mud, inp.P_lot,
        inp.P_apl, P_td,
        None if inp.fluid is None else tuple(sorted(inp.fluid.items())),
    )


@lru_cache(maxsize=8192)
def _influx_temperature_impl(T_s, T_td, D_lot, D_td, rho_mud, P_lot, P_apl,
                             P_td, fluid_key):
    """Fixed point behind :func:`_influx_temperature` (scalars only, cacheable)."""
    fluid = None if fluid_key is None else dict(fluid_key)
    inp = KickInputs(
        rho_mud=rho_mud, PP=0.0, kick_intensity=0.0, P_lot=P_lot, P_apl=P_apl,
        D_td=D_td, D_lot=D_lot, T_s=T_s, T_td=T_td, V_dpa=1.0, fluid=fluid,
    )
    interval = D_td - D_lot
    grad = (T_td - T_s) / interval                     # degF/ft
    B = constant_B(inp)
    P_shoe = _shoe_gas_pressure(inp, P_td)
    H = 0.0
    T_r = fahrenheit_to_rankine(T_s)
    for _ in range(64):
        T_r = fahrenheit_to_rankine(T_s + grad * 0.5 * H)
        rho = _influx_density(inp, P_shoe, T_r)
        deficit = rho_mud - rho
        if deficit <= 0.0:
            break                                      # no buoyant column
        H_new = (B - P_td) / (G_PSI_PER_PPG_FT * deficit)
        H_new = min(max(H_new, 0.0), interval)         # the column is IN the
        #                                                open hole (A-2's
        #                                                simplification (1))
        if abs(H_new - H) < 1e-9:
            H = H_new
            break
        H = H_new
    return T_r, H


def _influx_density(inp: KickInputs, p_psia: float, t_r: float) -> float:
    """Influx gas density [ppg] at ``(p_psia, t_r)`` on the active backend."""
    if inp.ideal_gas:
        return gas_density_ppg(p_psia, t_r, 1.0)
    if inp.fluid is not None:
        from .gas_z_coolprop import fluid_z_density

        return fluid_z_density(inp.fluid, p_psia, t_r)[1]
    return methane_properties(p_psia, t_r)[1]


def resolve_gas_properties(
    inp: KickInputs, P_td: Optional[float] = None
) -> tuple[float, float, float]:
    """Return (Z_s, Z_td, rho_gas_s), computing any left as ``None``.

    TD station:   (P_td, T_td)  -> Z_td.
    Shoe station: (P_shoe, T_s) -> Z_s, rho_gas_s.

    ``P_td`` is the bottom-hole pressure the influx gas is stationed at. Default
    ``None`` uses the DRILL scenario pressure ``scenario_P_td(inp)``; the swab
    case passes its own P_td (mud hydrostatic) so kick_intensity does not leak
    into the swab gas properties (Nassab SPE-202426-PA Eqs 8-9).

    Backend: pure-methane Hall-Yarborough by default; if ``inp.fluid`` is a
    composition dict, the CoolProp real-EOS mixture backend (CO2 / CCUS) is used.
    Injected numeric Z_s/Z_td/rho_gas_s values are used verbatim (override).
    """
    if inp.model_revision not in MODEL_REVISIONS:
        raise ValueError(
            f"unknown kick-tolerance model_revision {inp.model_revision!r}; "
            f"known revisions: {sorted(MODEL_REVISIONS)}"
        )
    if P_td is None:
        P_td = scenario_P_td(inp)
    # The influx station's temperature is revision-dependent -- T_s as
    # published, the gas-column mean thereafter (see _influx_temperature).
    T_s_r, _ = _influx_temperature(inp, P_td)
    T_td_r = fahrenheit_to_rankine(inp.T_td)

    if inp.ideal_gas:
        # Ideal-gas reference: Z == 1 everywhere, isothermal (single T_s). An
        # explicit numeric override still wins.
        def gas(p_psia, t_r):                          # (Z=1, ideal rho_ppg)
            return 1.0, gas_density_ppg(p_psia, T_s_r, 1.0)
        T_td_r = T_s_r                                 # isothermal
    elif inp.fluid is not None:
        from .gas_z_coolprop import fluid_z_density   # optional CoolProp backend

        def gas(p_psia, t_r):                          # (Z, rho_ppg) for the mixture
            return fluid_z_density(inp.fluid, p_psia, t_r)
    else:
        gas = methane_properties                       # (Z, rho_ppg) clean-room H-Y

    Z_td = inp.Z_td if inp.Z_td is not None else gas(P_td, T_td_r)[0]

    if inp.Z_s is None or inp.rho_gas_s is None:
        P_shoe = _shoe_gas_pressure(inp, P_td)
        Z_s_c, rho_gas_s_c = gas(P_shoe, T_s_r)
    else:
        Z_s_c = rho_gas_s_c = None  # not needed; both injected

    Z_s = inp.Z_s if inp.Z_s is not None else Z_s_c
    rho_gas_s = inp.rho_gas_s if inp.rho_gas_s is not None else rho_gas_s_c
    return Z_s, Z_td, rho_gas_s


# --- Closed-form building blocks --------------------------------------------

def constant_A(inp: KickInputs, P_td: Optional[float] = None) -> float:
    """A-5:  A = (P_lot - P_apl) * T_td * Z_td * V_dpa
                 / [ g * T_s * Z_s * (rho_mud - rho_gas_s) ]   [bbl]

    P_lot is converted to a pressure at the shoe depth (g * P_lot * D_lot).
    Temperatures are absolute (degR). Gas properties (Z_s, Z_td, rho_gas_s) are
    computed by the clean-room Hall-Yarborough backend unless injected, stationed
    at ``P_td`` (default: the drill scenario pressure; swab passes mud
    hydrostatic -- see ``resolve_gas_properties``).

    Deviated wells (Nassab et al., SPE-202426-PA): the gas column's vertical
    height H_gas occupies an along-hole length L_gas = H_gas / cos(inc_shoe),
    which is what multiplies the per-MD annular capacity V_dpa. This scales A by
    1 / cos(inc_shoe). inc_shoe = 0 (vertical) recovers the SPE-208788 Table-1
    form exactly. (Exact survey-integral form -- absorbing BHA change-of-section
    -- is a documented follow-up; this constant-inclination form is the standard
    published convention.)
    """
    if P_td is None:
        P_td = scenario_P_td(inp)
    Z_s, Z_td, rho_gas_s = resolve_gas_properties(inp, P_td)
    P_lot_psi = ppg_to_psi(inp.P_lot, inp.D_lot)
    # SAME influx temperature the Z_s / rho_gas_s above were evaluated at --
    # this and resolve_gas_properties must never disagree on the station.
    T_s_r, _ = _influx_temperature(inp, P_td)
    # Ideal-gas reference mode is isothermal: the T_td/T_s expansion ratio -> 1.
    T_td_r = T_s_r if inp.ideal_gas else fahrenheit_to_rankine(inp.T_td)
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


def influx_column(
    inp: KickInputs, P_td: Optional[float] = None
) -> tuple[float, float, float]:
    """The limiting influx column: ``(T_influx [degF], H_gas [ft], rho_influx)``.

    The three quantities A-5 was actually evaluated with. Use these to draw or
    check the limiting pressure profile -- they are the only set that closes.

    THE IDENTITY. Reconstructing the shoe pressure from this column,

        P_shoe = P_td - g [ rho_influx * H + rho_mud * (D_td - D_lot - H) ]

    returns the binding fracture pressure ``g * P_lot * D_lot - P_apl``
    EXACTLY, for any ``rho_influx``, because A-2 defines H as
    ``(B - P_td) / [g (rho_mud - rho_influx)]`` and B carries the mud column
    over the same interval. That is the definition of the limiting condition:
    at the tolerable influx the shoe sits ON the fracture pressure. Verified to
    0.00 psi under both revisions by ``test_kick_model_revision.py``.

    DO NOT back-derive H from ``KickResult.capacity`` instead. Expanding the
    capacity to the shoe goes through A-4, whose pressure bookkeeping is not
    algebraically identical to A-5/A-7 as printed (A-4 pairs
    ``P_lot - P_apl + P_atm`` with ``P_td``, while A-7 pairs ``P_lot - P_apl``
    with ``P_td + P_atm``). On the Table-1 case that route misses the fracture
    pressure by -3.93 psi -- IDENTICALLY under both revisions, so it is the
    paper's own bookkeeping and not the influx-temperature correction. A
    profile built that way was already off before 0.26.0; it just did not move.

    ``rho_influx`` respects an injected ``rho_gas_s``, so the identity holds for
    a caller supplying its own gas properties too.
    """
    if P_td is None:
        P_td = scenario_P_td(inp)
    T_r, _ = _influx_temperature(inp, P_td)
    _, _, rho_influx = resolve_gas_properties(inp, P_td)
    deficit = inp.rho_mud - rho_influx
    H = ((constant_B(inp) - P_td) / (G_PSI_PER_PPG_FT * deficit)
         if deficit > 0.0 else 0.0)
    return T_r - RANKINE_OFFSET, H, rho_influx


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
    T_influx, H_gas, rho_influx = influx_column(inp, P_td)
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
        model_revision=inp.model_revision,
        T_influx=T_influx,
        rho_influx=rho_influx,
        H_gas=H_gas,
    )


def swab_kick(inp: KickInputs) -> KickResult:
    """Unmitigated free-trip limit (A-8) -- a MITIGABLE operational limit.

    A-8 is A-7 with the bottom-hole pressure set to the mud hydrostatic,
    P_td = g * rho_mud * D_td (pore pressure <= mud hydrostatic). A swab
    failure means "pump out of the hole" -- which is the drill case, assessed
    separately. Swab NEVER overrides drill and is never blind-min'd with it.

    ``kick_intensity`` is DELIBERATELY UNUSED here. For a swabbed kick the
    bottom-hole pressure is the mud hydrostatic, independent of formation PP
    (and therefore of any intensity margin added to PP): Nassab SPE-202426-PA,
    Eqs 8-9 -- "Ptd ... is independent of formation PP and is equal to mud
    hydrostatic pressure. This concept is misunderstood ... in many KT models
    that assume Ptd is equal to PP for both underbalanced and swabbed kicks[,]
    ... [leading] to an overestimated KT value in swabbing conditions." Carrying
    PP or PP+KI into the swab bottom-hole pressure would over-report swab KT
    (unsafe); dropping KI for swab is the correct, conservative default. This
    includes the influx gas STATIONING: A is evaluated with the swab P_td (mud
    hydrostatic), not the drill scenario pressure, so kick_intensity does not
    leak into Z/rho either.

    CONSCIOUS DIVERGENCE FROM SPE-208788-PA. That paper's worked example shares
    ONE A constant across A-7 (drill) and A-8 (swab) -- so its published swab
    figure (43.79 bbl, Table-1) stations the swab gas at the drill max-credible
    pressure (PP + KI), carrying kick intensity into a swab number. We reproduce
    that figure under its own convention in the validation suite, but our MODEL
    does not adopt it: a swab kick is not a drill kick, so it carries no kick
    intensity (Nassab SPE-202426-PA, Eqs 8-9). The difference is ~1.5% and only
    when PP+KI > mud (outside the swab model's PP <= mud domain).
    """
    P_td = ppg_to_psi(inp.rho_mud, inp.D_td)  # A-8 substitution
    A = constant_A(inp, P_td)                 # station gas at the swab P_td (no KI leak)
    B = constant_B(inp)
    T_influx, H_gas, rho_influx = influx_column(inp, P_td)
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
        model_revision=inp.model_revision,
        T_influx=T_influx,
        rho_influx=rho_influx,
        H_gas=H_gas,
    )


def annular_capacity_dpa(hole_id_in: float, pipe_od_in: float) -> float:
    """Convenience: annular capacity V_dpa = (hole^2 - pipe^2) / 1029.4 [bbl/ft].

    Per the A-3 simplification the influx occupies the drillpipe-in-hole annulus
    at the shoe (pipe_od = drillpipe OD). This is a geometry helper only; V_dpa
    may equally be supplied directly as an input.
    """
    return (hole_id_in ** 2 - pipe_od_in ** 2) / 1029.4
