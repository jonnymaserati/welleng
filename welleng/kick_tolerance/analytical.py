"""Analytical (breakpoint) kick-tolerance solver.

The migration :func:`~welleng.kick_tolerance.migration._max_influx_circulated`
marches a single gas bubble up the annulus and bisects the influx. That is exact
in the limit but (a) costs a fine march and (b) can UNDER-sample a narrow
breakpoint of the imposed-pressure-vs-position curve. This module computes the
kick tolerance by evaluating the imposed pressure only at the BREAKPOINTS of that
curve -- the positions where the binding constraint can turn -- and taking the
worst. It evaluates the true worst gas position (no sampling gap) in
O(breakpoints).

Density convention (the ``gas_density_mode`` switch, shared with the engine):

* ``"conservative"`` -- the whole influx is expanded to the gas-TOP pressure and
  placed at that lightest, constant density (longest column -> highest shallow
  loading -> lowest KT). This is the INHERENT CONSERVATISM of the numeric
  single-bubble model; the analytical solver reproduces that safe-side bound
  exactly. Default.
* ``"exact"`` -- integrate the TRUE (pressure-dependent) gas density up the
  column. This is MORE ACCURATE -- it recovers the real tolerance the conservative
  bound forfeits. It is not "less conservative"; it is the accurate value, without
  the numeric model's built-in cushion. CAVEAT: on a BORDERLINE result (computed
  max KT ~ the prescribed/regulatory max), the operator must apply their own
  margin, precisely because the accurate value carries no hidden slack.

The breakpoints of ``P(gas position)`` (for a fixed influx) are the configs where
either gas face crosses a discrete change in the problem:

    1. gas TOP at a section boundary or a PP/FP breakpoint      (classic shoe case)
    2. gas BOTTOM at a section boundary  -- the fill alignments  (was the missed set)
    3. the DEEPEST bubble: gas bottom at TD                      (initial position)
    4. the binding exposed depth switches (a PP/FP breakpoint under the gas/mud)

Evaluating only (1) -- the naive "min over interfaces" -- is NOT conservative:
on a tight/BHA bottom section the worst position is (2)/(3), where the gas bottom
crosses the BHA-top capacity discontinuity and fills the tight section to TD, so
the gas top sits INTERIOR to the open hole (not on any boundary).

Geometry assumption (single-bubble model class): the bubble caps are
PERPENDICULAR to the well centerline (an annular slug), NOT horizontal
gravity-segregated interfaces. Gas geometry is 1-D along the wellbore axis; in a
deviated well the along-hole (MD) length maps to a TVD extent by ``cos(inc)`` (a
later refinement -- this module is TVD-native like the rest of the engine).

Each candidate's breach influx is solved DIRECTLY (no marching, no outer influx
bisection):

* families 1/4 (gas TOP at a boundary) -- CLOSED FORM. At breach the gas-top
  pressure equals FP(d), so the gas length follows from the column pressure
  balance: conservative = a constant-density (linear) column -> a linear solve;
  exact = an exponential column ``FP.exp(k.L) = A + b.L`` -> a Lambert-W-form root
  (a 3-iter Newton, dependency-free). The length is density-driven hence
  CAP-independent; only the bbl total walks the per-section capacities.
* families 2/3 (gas BOTTOM at a boundary + TD -- the interior/tight-BHA binding a
  gas-top pin can't express) -- ALSO CLOSED FORM. Pinning the binding depth makes the
  bit depth cancel out of the balance, leaving ``rho_mud.g.L - dP_gas(L) = C(d)`` with
  ``C`` a function of the binding depth alone: the same Lambert-W form as above, once
  per (gas-bottom pin, envelope depth) pair. The influx never needed bisecting.
  :func:`_breach_v_gas_bottom` retains the bracketed secant as the REFERENCE
  implementation the closed form is tested against; it is not on the solve path.

The kick tolerance is the MIN breach influx over all candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

from .migration import (
    G_PSI_PER_PPG_FT,
    WellSection,
    _as_ppg_callable,
    _fill_up,
    _profile_breakpoints,
    _resolve_bh_state,
    _as_temp_callable,
    _z_at,
    pressure_at_depth,
    ppg_to_psi,
    ProfileLike,
    TempProfileLike,
)

_CHUNK_FT = 100.0   # sub-segment length for the exact-mode closed-form integration


@lru_cache(maxsize=200_000)
def _z_cached(p_psi_int: int, t_tenths: int) -> float:
    return _z_at(float(p_psi_int), t_tenths / 10.0)


def _z(p_psi: float, t_rankine: float) -> float:
    """H-Y Z(P,T), memoised on (1 psi, 0.1 degR) -- the SAME validated backend as
    the migration engine, but the analytical solver re-queries the same (P,T) many
    times across the influx bisection and the candidate set. Rounding to 1 psi /
    0.1 degR changes Z by < 1e-4 (well inside the H-Y band); the migration path is
    unchanged (it still calls the un-cached backend)."""
    return _z_cached(int(round(p_psi)), int(round(t_rankine * 10.0)))


def _temperature_for_case(case, temp_profile, geothermal):
    """Resolve the gas-column temperature profile from the WELL-CONTROL CASE.

    Which temperature is correct is a property of the case, not of what the
    caller happened to type into a temperature box:

    * **drill** -- the kill is CIRCULATED, so the annulus tends to a steady state
      that tends to TD temperature. Isothermal at the bottom-hole temperature is
      that limit, and it is within 1.5% of a realistic circulating profile on the
      safe side. A formation gradient is the WRONG profile here and reads 7-12%
      HIGH: cooler shallow gas is denser, the column is heavier, the shoe sees
      less, and more influx looks tolerable. Supplying ``geothermal`` for a drill
      case is therefore refused rather than silently used.
    * **swab** -- a TRIP, no circulation, so the column relaxes toward formation.
      A geothermal profile is required; there is nothing sane to invent if it is
      absent.

    An explicit ``temp_profile`` always wins -- that is the advanced path, and it
    is how a measured or modelled circulating profile is supplied (non-monotonic
    profiles are supported). ``case=None`` reproduces the pre-0.27 precedence
    exactly: ``temp_profile > geothermal > isothermal``.
    """
    if case is None:
        return temp_profile if temp_profile is not None else geothermal
    if case not in ("drill", "swab"):
        raise ValueError(f"case must be 'drill', 'swab' or None, not {case!r}")
    if temp_profile is not None:
        return temp_profile
    if case == "swab":
        if geothermal is None:
            raise ValueError(
                "case='swab' is a trip with no circulation, so the gas column "
                "relaxes toward FORMATION temperature -- supply geothermal=(tvd, "
                "degR) or an explicit temp_profile."
            )
        return geothermal
    if geothermal is not None:
        raise ValueError(
            "case='drill' is a CIRCULATED kill, so the annulus tends to TD "
            "temperature, not to the formation gradient. A geothermal profile "
            "reads 7-12% HIGH here (denser shallow gas, heavier column, less "
            "pressure at the shoe). Drop geothermal to use the isothermal "
            "bottom-hole limit, or pass an explicit circulating temp_profile."
        )
    return None                                # isothermal at T_bh


def max_influx_contained_at_surface(
    sections: Sequence[WellSection],
    *,
    burst_pressure_psi: float,
    bhp_psi: float,
    rho_mud_ppg: float,
    gas_bh_state,
    bottom_tvd: Optional[float] = None,
    temp_profile: TempProfileLike = None,
    z_fn=None,
) -> float:
    """Largest bottom-hole influx whose migration to surface still keeps the
    surface pressure within ``burst_pressure_psi`` [bbl].

    The worst case for the casing is the whole influx arrived at surface: the gas
    has expanded, it has displaced the mud that was carrying the load, and the
    surface pressure is at its highest. With the bubble occupying ``[0, L]`` and
    mud beneath it to TD, and the bottom pinned at the bottom-hole pressure::

        P_s * exp(k * L)  +  rho_mud * g * (TD - L)  =  BHP

    One scalar equation in the gas length ``L``, solved directly -- set ``P_s`` to
    the allowable and the equation gives the longest tolerable column, which the
    per-section capacities then turn into an influx. Surface pressure rises
    monotonically with ``L`` (gas replacing mud), so the root is unique.

    Compare the result with the kick tolerance from
    :func:`analytical_kick_tolerance`: if it is SMALLER, the casing is the binding
    barrier rather than the formation, and the open-hole answer alone overstates
    what the well can take.

    INDICATIVE ONLY -- this is not a casing design calculation. It credits no
    external backup, assumes the influx arrives coherently at surface with mud
    beneath it, and ignores axial load, bending, temperature derating, wear and
    connection ratings.

    Returns
    -------
    float
        The influx [bbl at bottom-hole conditions]. ``0.0`` if the well cannot be
        shut in within the allowable even with no influx at all (i.e. the mud
        column alone already puts surface above the rating).
    """
    ss = sorted(sections, key=lambda s: s.top_tvd)
    td = float(bottom_tvd) if bottom_tvd is not None else max(s.bottom_tvd for s in ss)
    g = G_PSI_PER_PPG_FT
    zf = z_fn or _z
    P_bh, T_bh_r, Z_bh, rho_bh = _resolve_bh_state(gas_bh_state, bhp_psi)
    temp_fn = _as_temp_callable(temp_profile, T_bh_r)

    P_s = float(burst_pressure_psi)
    # No influx at all: a full mud column already puts surface at BHP - rho.g.TD.
    if P_s <= bhp_psi - g * rho_mud_ppg * td:
        return 0.0

    T_c = float(temp_fn(0.5 * td))
    L, k = 0.0, None
    for _ in range(60):                       # outer: Z at the column mean pressure
        P_top = P_s
        P_bot_guess = P_s if k is None else P_s * np.exp(k * L)
        Z_c = zf(max(0.5 * (P_top + P_bot_guess), 1.0), T_c)
        k = g * rho_bh * Z_bh * T_bh_r / (P_bh * Z_c * T_c)
        prev = L
        for _ in range(40):                   # inner: Newton on the balance
            e = np.exp(k * L)
            f = P_s * e + g * rho_mud_ppg * (td - L) - bhp_psi
            df = P_s * k * e - g * rho_mud_ppg
            if abs(df) < 1e-12:
                break
            step = f / df
            L = min(max(L - step, 0.0), td)
            if abs(step) < 1e-9:
                break
        if abs(L - prev) < 1e-8:
            break

    if L <= 0.0:
        return 0.0

    volume, P = 0.0, P_s
    for s in ss:
        top, bot = max(0.0, s.top_tvd), min(L, s.bottom_tvd)
        if bot <= top:
            continue
        P_bot = P * np.exp(k * (bot - top))
        volume += s.capacity_per_tvd_ft * (P_bot - P) / (g * rho_bh)
        P = P_bot
    return float(volume)


@dataclass
class AnalyticalKickTolerance:
    """Result of :func:`analytical_kick_tolerance`."""

    max_influx_bbl: float          # bottom-hole influx at first fracture [bbl]
    binding_gas_top_tvd: float     # gas-top TVD of the binding config [ft]
    binding_gas_bottom_tvd: float  # gas-bottom TVD of the binding config [ft]
    binding_depth_tvd: float       # exposed depth where the envelope is tight [ft]
    open_hole_unconstrained: bool  # True if the OPEN HOLE does not constrain the kick
    #                                tolerance at the provided fracture pressure -- the
    #                                shoe holds through full open-hole displacement. NOT
    #                                "unlimited": a real limit exists (casing burst above
    #                                the shoe), it is simply NOT assessed here (we stop at
    #                                the shoe), and the fracture pressure is uncertain.
    breakpoints: dict              # {label: influx-at-breach} for inspection
    surface_containment_bbl: Optional[float] = None
    #                                Largest influx the CASING could hold with the
    #                                bubble arrived at surface, when the cased
    #                                sections carry a burst allowable. None when no
    #                                section does (the default) -- the casing is then
    #                                simply not assessed, as before.
    casing_binds: bool = False
    #                                True when that is SMALLER than the open-hole
    #                                answer, i.e. the casing is the binding barrier
    #                                and the formation-only number overstates what the
    #                                well can take. INDICATIVE -- see
    #                                :func:`max_influx_contained_at_surface`.
    already_fractured: bool = False
    #                                True when the MUD COLUMN ALONE already meets or
    #                                exceeds the fracture pressure at some exposed
    #                                depth, with NO influx in the hole. The tolerance
    #                                is then 0.0 and the section is undrillable on
    #                                these inputs -- it is NOT the same condition as
    #                                ``open_hole_unconstrained`` even though both
    #                                leave the breach search with no candidate. See
    #                                the guard in :func:`analytical_kick_tolerance`.
    maasp_psi: float = float("nan")
    #                                Maximum allowable annular surface pressure over
    #                                the WHOLE exposed hole [psi] -- see
    #                                :func:`~welleng.kick_tolerance.migration.maasp`.
    maasp_governing_tvd: float = float("nan")
    maasp_governed_by_shoe: bool = True
    #                                False when a weak zone BELOW the shoe governs, so
    #                                the conventional shoe-only MAASP overstates the
    #                                closed-in limit.


def _top_for_bottom(gas_bottom, influx_bbl_bh, sections_sorted, bottom_tvd, *,
                    bhp_psi, rho_mud_ppg, gas_bh, temp_fn,
                    gas_density_mode="conservative", z_fn=None):
    """Gas TOP when the bubble BOTTOM is pinned at ``gas_bottom`` and holds
    ``influx_bbl_bh`` of bottom-hole gas, in the SAME density convention as
    :func:`~welleng.kick_tolerance.migration.pressure_at_depth` so geometry and
    pressure are consistent.

    The pressure at the pinned bottom is known from the mud column below the BHP.

    * ``"conservative"`` -- the whole influx is expanded to the gas-TOP pressure
      (lowest -> lightest, LONGEST column -> highest shallow loading) at that
      constant density; ``P_gt = P_gb - rho(P_gt).g.L`` (``L`` from the per-section
      capacities via :func:`_fill_up`) is a short fixed point. Safe-side; matches
      the legacy numeric single-bubble model.
    * ``"exact"`` -- integrate the TRUE (pressure-dependent) density up the column.
      Over a sub-segment with ~constant ``c = rho_bh.Z_bh.T_bh/(P_bh.Z.T)`` the
      pressure is EXPONENTIAL and the gas mass is closed-form
      ``mass = cap.(P - P.exp(-g.c.dz))/g``; walk sub-segments (``_CHUNK_FT``) and
      invert in the last. MORE ACCURATE -- recovers the real tolerance the
      conservative bound forfeits (a shorter column here). On a BORDERLINE result
      (max KT ~ the prescribed limit) the accurate value has no built-in cushion,
      so treat it with the operator's own margin.
    """
    P_bh, T_bh_r, Z_bh, rho_bh = gas_bh
    g = G_PSI_PER_PPG_FT
    zf = z_fn or _z                                       # H-Y default, else provider
    P_gb = max(bhp_psi - g * rho_mud_ppg * (bottom_tvd - gas_bottom), 1.0)

    if gas_density_mode == "exact":
        P = P_gb
        mass_target = influx_bbl_bh * rho_bh          # ppg.bbl (rho_bh in ppg)
        mass = 0.0
        z = float(gas_bottom)
        while z > 1e-9 and mass < mass_target:
            seg_top_limit = max(0.0, z - _CHUNK_FT)
            cap = None
            for sec in sections_sorted:               # section just above z; snap to its top
                if sec.top_tvd < z <= sec.bottom_tvd:
                    cap = sec.capacity_per_tvd_ft
                if sec.top_tvd < z and sec.top_tvd > seg_top_limit:
                    seg_top_limit = sec.top_tvd
            if cap is None:
                cap = sections_sorted[0].capacity_per_tvd_ft
            seg = z - seg_top_limit
            T = float(temp_fn(0.5 * (z + seg_top_limit)))
            Z = zf(max(P, 1.0), T)
            c = rho_bh * Z_bh * T_bh_r / (P_bh * Z * T)    # rho = c.P  [ppg/psi]
            k = g * c
            P_seg_top = P * np.exp(-k * seg)
            dm = cap * (P - P_seg_top) / g
            if mass + dm >= mass_target and dm > 0.0:
                arg = 1.0 - (mass_target - mass) * g / (cap * P)
                return max(z + np.log(max(arg, 1e-12)) / k, 0.0)
            mass += dm
            P = P_seg_top
            z = seg_top_limit
        return max(z, 0.0)

    # "conservative": constant gas-top (lightest) density, fixed point in P_gt.
    gas_top = float(gas_bottom)
    P_gt = P_gb
    for _ in range(30):
        T_gt = float(temp_fn(gas_top))
        Z_gt = zf(max(P_gt, 1.0), T_gt)
        Ve = influx_bbl_bh * (P_bh * Z_gt * T_gt) / (max(P_gt, 1.0) * Z_bh * T_bh_r)
        gas_top = _fill_up(gas_bottom, Ve, sections_sorted)
        rho_top = rho_bh * max(P_gt, 1.0) * Z_bh * T_bh_r / (P_bh * Z_gt * T_gt)  # ppg
        P_new = max(P_gb - rho_top * g * (gas_bottom - gas_top), 1.0)
        if abs(P_new - P_gt) < 1e-4:
            P_gt = P_new
            break
        P_gt = 0.5 * (P_gt + P_new)
    return max(gas_top, 0.0)


# Convergence of the gas-bottom-pinned influx solve (``_breach_v_gas_bottom``).
# The loop must END ON A TOLERANCE, never on the iteration cap -- the cap is a
# runaway backstop. ``test_gas_bottom_solve_converges_not_truncates`` asserts that.
_SECANT_TOL_PSI = 0.25       # margin tolerance
_SECANT_TOL_BBL = 1e-3       # bracket width on the influx
_SECANT_MAX_ITER = 100


def _min_margin(gas_top, gas_bottom, exposed_depths, pp_psi, fp_psi, *,
                bottom_tvd, bhp_psi, rho_mud_ppg, gas_bh, gas_density_mode,
                temp_profile, z_fn=None):
    """(worst margin, binding depth) over the exposed depths for a config.

    The margin is the min of the FP margin (``FP-P >= 0``: no breakdown) and the
    PP margin (``P-PP >= 0``: no further influx).
    """
    P = pressure_at_depth(
        exposed_depths, gas_top_tvd=gas_top, gas_bottom_tvd=gas_bottom,
        bottom_tvd=bottom_tvd, bhp_psi=bhp_psi, rho_mud_ppg=rho_mud_ppg,
        gas_bh=gas_bh, gas_density_mode=gas_density_mode, temp_profile=temp_profile,
        z_fn=z_fn,
    )
    both = np.minimum(fp_psi - P, P - pp_psi)
    j = int(np.argmin(both))
    return float(both[j]), float(exposed_depths[j])


def analytical_kick_tolerance(
    sections: Sequence[WellSection],
    pp: ProfileLike,
    fp: ProfileLike,
    *,
    bhp_psi: float,
    rho_mud_ppg: float,
    gas_bh_state,
    gas_density_mode: str = "conservative",
    temp_profile: TempProfileLike = None,
    geothermal: TempProfileLike = None,
    gas_composition=None,
    fluid_table=None,
    gas_model: str = "real",
    check_depths: Optional[Sequence[float]] = None,
    case: Optional[str] = None,
) -> AnalyticalKickTolerance:
    """Max bottom-hole influx tolerable over the whole migration, by breakpoints.

    Breakpoint alternative to
    :func:`~welleng.kick_tolerance.migration._max_influx_circulated`: evaluates the
    imposed pressure only at the breakpoints of ``P(gas position)`` (§module
    docstring) and bisects the influx to the first fracture. ``gas_density_mode``
    selects the safe-side numeric bound (``"conservative"``, default) or the more
    accurate true-density value (``"exact"``) -- see the module docstring for the
    density convention, the borderline caveat, the geometry assumption and the
    breakpoint families.

    Real-gas backend (optional): pass ``gas_composition`` (mole fractions, e.g.
    ``{"methane": 0.9, "co2": 0.1}``) or a prebuilt ``fluid_table``
    (:class:`~welleng.kick_tolerance.gas_z_coolprop.ZTable`) to use CoolProp real-EOS
    Z(P,T) for a mixture / CO2 / CCUS influx (built once over the case P,T box, then
    interpolated). Default (neither) = the clean-room Hall-Yarborough methane backend.

    Units (US oilfield -- see the subpackage docstring's UNITS contract):
        sections     : :class:`WellSection` with TVD ``top``/``bottom`` [ft] and
                       ``annular_capacity_bbl_per_ft`` [bbl/ft].
        pp, fp       : pore / fracture profile ``(tvd_ft, ppg)`` tables or callables
                       -- mud-weight-EQUIVALENT density [ppg] vs TVD [ft].
        bhp_psi      : bottom-hole (kill) pressure [psi].
        rho_mud_ppg  : drilling-fluid density [ppg].
        gas_bh_state : ``(P_bh [psi], T_bh [degR], Z [-], rho_gas [ppg])`` (any None
                       -> computed). Returns influx / kick tolerance in [bbl], binding
                       depths in [ft].
        case : 'drill' | 'swab' | None. Selects the TEMPERATURE convention when no
            explicit ``temp_profile`` is given -- see :func:`_temperature_for_case`.
            'drill' is a circulated kill (isothermal at T_bh, the TD-temperature
            limit) and REFUSES a geothermal profile, which reads 7-12% high;
            'swab' is a trip and REQUIRES one. ``None`` keeps the pre-0.27
            precedence temp_profile > geothermal > isothermal.
        check_depths : optional explicit TVDs [ft] at which to enforce the
                       pore-fracture (PP-FP) envelope, OVERRIDING the auto-enumerated
                       exposed depths (open-hole boundaries + PP/FP breakpoints + gas
                       faces). Use it to pin the checked constraint set: e.g.
                       ``[shoe_tvd]`` gives SINGLE-SHOE semantics (only the casing shoe
                       strength-checked; deeper FP jumps are not binding), the free-tier
                       convention. ``None`` (default) = the full sections-aware
                       multi-depth check. Gas-position enumeration is unaffected -- the
                       gas may still sit anywhere; the envelope is simply enforced only
                       at these depths. NOTE: the constraint checked at each depth is
                       the full envelope ``min(FP - P, P - PP)`` -- the fracture bound
                       AND the pore bound. For a sane single-shoe well the imposed
                       pressure P exceeds PP at the shoe, so the fracture bound is the
                       active (binding) one there; the pore bound rides along and is
                       slack. If you need a strictly fracture-only check, filter on the
                       returned binding mechanism.
    """
    if gas_model not in ("real", "ideal"):
        raise ValueError(f"gas_model must be 'real' or 'ideal', got {gas_model!r}")
    ss = sorted(sections, key=lambda s: s.top_tvd)
    bottom_tvd = max(s.bottom_tvd for s in ss)
    # Ideal-gas REFERENCE mode: Z == 1 everywhere and isothermal. Reproduces a
    # textbook single-bubble Boyle's-law result for cross-checking welleng
    # against ideal-gas references; NOT for design (discards real-gas
    # compressibility). Overrides any gas_composition/fluid_table.
    if gas_model == "ideal":
        if gas_composition is not None or fluid_table is not None:
            raise ValueError("gas_model='ideal' cannot be combined with a "
                             "real-gas composition / fluid_table")
        P_bh0 = gas_bh_state[0] if gas_bh_state[0] is not None else bhp_psi
        T_bh0 = gas_bh_state[1]
        gas_bh_state = (P_bh0, T_bh0, 1.0, gas_bh_state[3])   # Z_bh := 1
        temp_profile = ([0.0, bottom_tvd], [T_bh0, T_bh0])     # isothermal
        geothermal = None
    # Real-gas Z provider (CoolProp) when a composition / table is given, else H-Y.
    z_fn = (lambda p, t: 1.0) if gas_model == "ideal" else None
    if gas_model == "real" and (fluid_table is not None or gas_composition is not None):
        table = fluid_table
        if table is None:
            from .gas_z_coolprop import ZTable
            t_bh = gas_bh_state[1]
            t_surf = float(temp_profile(0.0)) if callable(temp_profile) else (
                float(geothermal(0.0)) if callable(geothermal) else t_bh)
            table = ZTable(gas_composition,
                           (14.7, bhp_psi * 1.1),
                           (min(t_surf, t_bh) - 5.0, max(t_surf, t_bh) + 5.0))
        z_fn = lambda p, t: float(table.z(p, t))          # noqa: E731
        # bottom-hole anchor from the same fluid if the user didn't supply Z/rho
        P_bh0 = gas_bh_state[0] if gas_bh_state[0] is not None else bhp_psi
        T_bh0 = gas_bh_state[1]
        Z_bh0 = gas_bh_state[2] if gas_bh_state[2] is not None else float(table.z(P_bh0, T_bh0))
        rho_bh0 = gas_bh_state[3] if gas_bh_state[3] is not None else float(table.rho_ppg(P_bh0, T_bh0))
        gas_bh_state = (P_bh0, T_bh0, Z_bh0, rho_bh0)
    gas_bh = _resolve_bh_state(gas_bh_state, bhp_psi)
    zf = z_fn or _z                                       # Z(P,T): H-Y default, else provider
    temp_profile = _temperature_for_case(case, temp_profile, geothermal)
    temp_fn = _as_temp_callable(temp_profile, gas_bh[1])
    pp_fn, fp_fn = _as_ppg_callable(pp), _as_ppg_callable(fp)

    # Section boundaries (gas-face candidates) and the exposed open-hole depths at
    # which the envelope is enforced.
    boundaries = sorted({s.top_tvd for s in ss} | {s.bottom_tvd for s in ss})
    pf_breaks = [b for b in (_profile_breakpoints(pp) + _profile_breakpoints(fp))
                 if 0.0 < b < bottom_tvd]

    # Explicit FP-enforcement depths (single-shoe / pinned-constraint override).
    check_arr = (None if check_depths is None
                 else np.array(sorted({float(x) for x in check_depths}), dtype=float))

    # This solver is exact BECAUSE it knows every depth at which the binding
    # constraint can turn. A callable profile has no discrete breakpoints
    # (_profile_breakpoints returns []), so a weak zone between section boundaries
    # is never in the candidate set and is never evaluated -- silently, and in the
    # NON-conservative direction. Measured: the same weak zone at 8000 ft gives
    # 44.4 bbl as a (tvd, ppg) table and 58.2 bbl as a callable, a 31% over-report.
    # A caller who pins ``check_depths`` has told us where to look and may use a
    # callable; otherwise refuse rather than quietly answer the wrong question.
    if check_arr is None and (callable(pp) or callable(fp)):
        raise ValueError(
            "analytical_kick_tolerance needs the depths at which the pore/fracture "
            "gradient can turn, and a callable profile does not expose them -- a "
            "weak zone between section boundaries would be silently missed. Pass "
            "the profile as a (tvd, ppg) table, or pin the depths to enforce with "
            "check_depths=[...]. To keep a callable without either, use "
            "_max_influx_circulated, which marches instead of enumerating."
        )

    def exposed_for(gas_top, gas_bottom):
        # FP/PP margin over a mud/gas column is piecewise-monotone; its extremum
        # is at a breakpoint depth: open-hole boundaries, PP/FP breaks, and the
        # gas faces themselves. Enforce the envelope only where formation is open.
        if check_arr is not None:
            return check_arr                                  # caller pins the checked set
        cand = set(pf_breaks)
        for s in ss:
            if s.is_open_hole:
                cand.update((s.top_tvd, s.bottom_tvd))
        cand.update((gas_top, gas_bottom))
        d = np.array(sorted(x for x in cand
                            if any(s.is_open_hole and s.top_tvd <= x <= s.bottom_tvd
                                   for s in ss)), dtype=float)
        return d

    # Physical ceiling: total exposed open-hole volume (single bubble below shoe).
    v_hole = sum(s.capacity_per_tvd_ft * (s.bottom_tvd - s.top_tvd)
                 for s in ss if s.is_open_hole)

    # The casing above the shoe is not exposed to fracture, but it has a finite
    # internal pressure rating, and the worst case for it is the influx arrived at
    # SURFACE. Assessed only when a cased section carries an allowable (i.e. it was
    # built by ``cased_section`` with a grade); the weakest rating governs.
    _ratings = [s.burst_pressure_psi for s in ss
                if not s.is_open_hole and s.burst_pressure_psi is not None]
    surface_bbl = (
        max_influx_contained_at_surface(
            ss, burst_pressure_psi=min(_ratings), bhp_psi=bhp_psi,
            rho_mud_ppg=rho_mud_ppg, gas_bh_state=gas_bh_state,
            bottom_tvd=bottom_tvd, temp_profile=temp_profile, z_fn=z_fn,
        ) if _ratings else None
    )

    # A BOUNDARY is any discrete change in the problem -- a section-capacity
    # change OR a PP/FP breakpoint. Both gas faces are candidate-pinned at every
    # boundary (JJ): gas BOTTOM at a boundary + TD (families 2/3), gas TOP at a
    # boundary (families 1/4).
    bset = sorted(set(boundaries) | set(pf_breaks))
    bottom_pins = [b for b in bset if b > 0.0] + [bottom_tvd]
    top_pins = [b for b in bset if 0.0 < b < bottom_tvd]
    if check_arr is not None:
        # Family 1/4 (gas-top-at-d) enforces FP AT d, so it must also be pinned to
        # the checked set; gas BOTTOM enumeration (bottom_pins) is left untouched.
        top_pins = [float(x) for x in check_arr if 0.0 < x < bottom_tvd]

    P_bh, T_bh_r, Z_bh, rho_bh = gas_bh
    g = G_PSI_PER_PPG_FT

    def _breach_v_gas_top(d):
        """CLOSED-FORM influx at which the imposed pressure at gas-top depth ``d``
        reaches FP(d) (families 1/4). The gas length L is density-driven, so it is
        CAP-INDEPENDENT -- a single closed form even across sections (conservative =
        constant-density linear column; exact = exponential column, a Lambert-W-form
        root by a 3-iter Newton, dependency-free). Only the bbl total walks the
        per-section capacities. Returns ``(V, gas_bottom)`` or None if a gas-top-at-d
        config cannot reach FP or would not fit above TD.

        FRACTURE-SIDE ONLY (by design): this closed form solves the FP breach for a
        gas-top-pinned config; it does not check the pore-side bound (P >= PP). That is
        complete for the kick-tolerance MINIMUM for two reasons. (1) Physics: a pore-
        side breach (imposed P falling to PP above the gas) is a LARGE-influx / high-
        gas-position regime -- the column is heavily gas-lightened -- so it is not the
        minimum breach influx that sets KT; the KT minimum is the shoe/weak-zone FP
        limit, caught here, or an interior config caught by the gas-BOTTOM-pinned
        family (``_breach_v_gas_bottom`` -> ``_min_margin``), which DOES enforce the
        full ``min(FP - P, P - PP)`` envelope. (2) Empirical guard: the analytical KT is
        cross-checked against the thorough march (``_max_influx_circulated``, which
        enforces the full PP-FP envelope at every station) in
        ``test_conservative_matches_migration_standard`` -- analytical <= march + tol.
        If this FP-only solve ever over-reported by missing a pore-side binding, the
        analytical result would exceed the march and that test would fail."""
        FP = ppg_to_psi(fp_fn(np.array([d]))[0], d)
        A = bhp_psi - g * rho_mud_ppg * (bottom_tvd - d)      # mud pressure at d
        b = g * rho_mud_ppg
        if FP <= A:
            return None                                        # already >= FP with no gas
        T_d = float(temp_fn(d))
        if gas_density_mode == "conservative":
            Z_t = zf(FP, T_d)
            rho_top = rho_bh * FP * Z_bh * T_bh_r / (P_bh * Z_t * T_d)
            if b <= rho_top * g:
                return None
            L = (FP - A) / (b - rho_top * g)                   # constant-density (linear)
        else:                                                  # exact exponential column
            L = (FP - A) / b                                   # linear seed
            for _ in range(6):                                 # Newton on FP*exp(kL)=A+bL
                Z_c = zf(0.5 * (FP + A + b * max(L, 0.0)), T_d)
                k = g * rho_bh * Z_bh * T_bh_r / (P_bh * Z_c * T_d)
                e = np.exp(k * L)
                fL, dfL = FP * e - (A + b * L), FP * k * e - b
                if abs(dfL) < 1e-12:
                    break
                Ln = L - fL / dfL
                if abs(Ln - L) < 1e-4:
                    L = Ln
                    break
                L = max(Ln, 0.0)
        gas_bottom = d + L
        if L <= 0.0 or gas_bottom > bottom_tvd + 1e-6:
            return None                                        # cannot fit as gas-top-at-d
        gas_bottom = min(gas_bottom, bottom_tvd)
        V = 0.0
        P = FP
        for s in ss:                                           # bbl total across sections
            top = max(d, s.top_tvd)
            bot = min(gas_bottom, s.bottom_tvd)
            if bot <= top:
                continue
            if gas_density_mode == "conservative":
                Z_t = zf(FP, T_d)
                rho_top = rho_bh * FP * Z_bh * T_bh_r / (P_bh * Z_t * T_d)
                V += rho_top * s.capacity_per_tvd_ft * (bot - top) / rho_bh
            else:
                Z_c = zf(0.5 * (P + A + b * (bot - d)), T_d)
                k = g * rho_bh * Z_bh * T_bh_r / (P_bh * Z_c * T_d)
                P_bot = P * np.exp(k * (bot - top))
                V += s.capacity_per_tvd_ft * (P_bot - P) / (g * rho_bh)
                P = P_bot
        return V, gas_bottom

    def _margin_bottom(b, V):
        """(min margin, gas_top, binding depth) for the config with gas BOTTOM
        pinned at ``b`` and influx ``V``."""
        gt = _top_for_bottom(b, V, ss, bottom_tvd, bhp_psi=bhp_psi,
                             rho_mud_ppg=rho_mud_ppg, gas_bh=gas_bh, temp_fn=temp_fn,
                             gas_density_mode=gas_density_mode, z_fn=z_fn)
        d = exposed_for(gt, b)
        if not d.size:
            return np.inf, gt, np.nan
        mv, db = _min_margin(gt, b, d, ppg_to_psi(pp_fn(d), d), ppg_to_psi(fp_fn(d), d),
                             bottom_tvd=bottom_tvd, bhp_psi=bhp_psi,
                             rho_mud_ppg=rho_mud_ppg, gas_bh=gas_bh,
                             gas_density_mode=gas_density_mode, temp_profile=temp_profile,
                             z_fn=z_fn)
        return mv, gt, db

    def _breach_v_gas_top_at(z, d):
        """CLOSED FORM. Influx for the gas TOP pinned at boundary ``z`` with the
        envelope exactly tight at an exposed depth ``d`` ABOVE it.

        The existing gas-top family pins the top at ``d`` and enforces FP at the
        SAME depth -- the diagonal of (face position x binding depth). The worst
        config need not be on that diagonal: with a long tight bottom section the
        gas top sits on the BHA-top boundary while the shoe is what binds, and
        that pairing was never enumerated. Measured on a 2500 ft tight section:
        the diagonal-only set returns 18.69 bbl, whose worst gas position has a
        -52 psi margin at the shoe; the true answer is 17.3.

        Unknown is the gas BOTTOM. With the gas spanning ``[z, gb]``::

            P(gb) = BHP - rho_mud*g*(TD - gb)
            P(z)  = P(gb) * exp(-k*(gb - z))
            P(d)  = P(z) - rho_mud*g*(z - d)  =  FP(d)

        One scalar equation in ``gb``; same Newton as everywhere else here.

        Returns ``(V, gas_bottom)`` or None when no such config fits.
        """
        if not (d < z < bottom_tvd):
            return None
        FPd = ppg_to_psi(fp_fn(np.array([d]))[0], d)
        bm = g * rho_mud_ppg
        T_d = float(temp_fn(z))
        P_z_target = FPd + bm * (z - d)        # gas-top pressure the envelope demands
        if P_z_target <= 0.0:
            return None

        gb, k = min(z + 1.0, bottom_tvd), None
        for _ in range(30):                    # outer: Z at the column mean pressure
            P_gb = bhp_psi - bm * (bottom_tvd - gb)
            Z_c = zf(max(0.5 * (P_gb + P_z_target), 1.0), T_d)
            k = g * rho_bh * Z_bh * T_bh_r / (P_bh * Z_c * T_d)
            prev = gb
            for _ in range(30):                # inner: Newton on P(z) - target
                P_gb = bhp_psi - bm * (bottom_tvd - gb)
                e = np.exp(-k * (gb - z))
                f = P_gb * e - P_z_target
                df = bm * e - P_gb * k * e
                if abs(df) < 1e-12:
                    break
                step = f / df
                gb = min(max(gb - step, z + 1e-9), bottom_tvd)
                if abs(step) < 1e-9:
                    break
            if abs(gb - prev) < 1e-7:
                break

        if not (z < gb <= bottom_tvd + 1e-6):
            return None
        gb = min(gb, bottom_tvd)
        # The Newton CLAMPS at the bounds, so a demand the well cannot meet comes
        # back as a bound rather than as a failure. Verify the equation is actually
        # satisfied before believing the answer -- an unreachable gas-top pressure
        # (P_z_target above anything the column can deliver) otherwise yields a
        # spurious tiny influx and collapses the kick tolerance.
        resid = (bhp_psi - bm * (bottom_tvd - gb)) * np.exp(-k * (gb - z)) - P_z_target
        if abs(resid) > 1e-6 * max(1.0, abs(P_z_target)):
            return None

        V, P = 0.0, P_z_target
        for s in ss:
            top, bot = max(z, s.top_tvd), min(gb, s.bottom_tvd)
            if bot <= top:
                continue
            P_bot = P * np.exp(k * (bot - top))
            for _ in range(3):
                Z_s = zf(max(0.5 * (P + P_bot), 1.0), T_d)
                k_s = g * rho_bh * Z_bh * T_bh_r / (P_bh * Z_s * T_d)
                P_bot = P * np.exp(k_s * (bot - top))
            V += s.capacity_per_tvd_ft * (P_bot - P) / (g * rho_bh)
            P = P_bot
        return (V, gb) if V > 0.0 else None

    def _breach_v_gas_bottom_at(b, d):
        """CLOSED FORM. Influx for gas BOTTOM pinned at ``b`` with the fracture
        envelope exactly tight at an exposed depth ``d`` ABOVE the gas top.

        The bit depth cancels out of the shoe balance, leaving one scalar equation
        in the gas length ``L``. With the gas bottom pinned, its pressure is known
        from the mud beneath it, and the mud above the gas carries the load up to
        ``d``::

            FP(d) = P_b*exp(-k*L) - rho_mud*g*(b - L - d)

        i.e. ``rho_mud*g*L - dP_gas(L) = FP(d) - BHP + rho_mud*g*(TD - d)``, whose
        right-hand side depends only on ``d``. Newton on ``L`` -- the same
        Lambert-W form :func:`_breach_v_gas_top` already solves for the gas-TOP
        pin, which is why the influx never needed bisecting.

        Depths INSIDE the gas need no solve: with the bottom pinned, the pressure
        there is fixed and independent of the influx, so the binding config is the
        one with the gas top exactly at ``d`` -- already enumerated as family 1/4.

        Returns ``(V, gas_top)`` or None when no such config fits the hole.
        """
        P_b = bhp_psi - g * rho_mud_ppg * (bottom_tvd - b)
        if P_b <= 0.0:
            return None
        FPd = ppg_to_psi(fp_fn(np.array([d]))[0], d)
        C = FPd - bhp_psi + g * rho_mud_ppg * (bottom_tvd - d)
        bm = g * rho_mud_ppg
        T_d = float(temp_fn(d))

        L, k = max(C / bm, 1.0), None
        for _ in range(30):                    # outer: Z at the column mean pressure
            P_top = P_b if k is None else P_b * np.exp(-k * L)
            Z_c = zf(max(0.5 * (P_b + P_top), 1.0), T_d)
            k = g * rho_bh * Z_bh * T_bh_r / (P_bh * Z_c * T_d)
            prev = L
            for _ in range(30):                # inner: Newton on Phi(L) - C
                e = np.exp(-k * L)
                f = bm * L - P_b * (1.0 - e) - C
                df = bm - P_b * k * e
                if abs(df) < 1e-12:
                    break
                step = f / df
                L = max(L - step, 1e-9)
                if abs(step) < 1e-9:
                    break
            if abs(L - prev) < 1e-7:
                break

        gas_top = b - L
        if not (0.0 < L) or gas_top <= d:
            return None                        # d must sit above the gas top
        # Same guard: the Newton floors L, so an unsatisfiable balance returns a
        # bound instead of failing. Check the residual before believing it.
        if abs(bm * L - P_b * (1.0 - np.exp(-k * L)) - C) > 1e-6 * max(1.0, abs(C)):
            return None

        # bbl total: walk the per-section capacities, Z re-evaluated per section
        # exactly as the gas-top-pinned family does.
        V, P = 0.0, P_b * np.exp(-k * L)
        for s in ss:
            top, bot = max(gas_top, s.top_tvd), min(b, s.bottom_tvd)
            if bot <= top:
                continue
            # Z at the SECTION's own mean pressure -- two passes, because the
            # section's bottom pressure is what we are computing.
            P_bot = P * np.exp(k * (bot - top))
            for _ in range(3):
                Z_s = zf(max(0.5 * (P + P_bot), 1.0), T_d)
                k_s = g * rho_bh * Z_bh * T_bh_r / (P_bh * Z_s * T_d)
                P_bot = P * np.exp(k_s * (bot - top))
            V += s.capacity_per_tvd_ft * (P_bot - P) / (g * rho_bh)
            P = P_bot
        return (V, gas_top) if V > 0.0 else None

    def _breach_v_gas_bottom(b):
        """Influx where the config with gas BOTTOM pinned at ``b`` first breaches
        (families 2/3 -- the interior/tight-BHA binding the closed form can't pin).
        Margin is monotone decreasing in V; a bracketed SECANT on the influx
        (falls back to bisection if it leaves the bracket). Returns
        ``(V, gas_top, binding_depth)`` or None if it never breaches within the hole."""
        m_hi, _, _ = _margin_bottom(b, v_hole)
        if m_hi > 0.0:
            return None                                   # never breaches in the hole
        lo, hi = 0.0, v_hole
        m_lo = _margin_bottom(b, lo)[0]                   # large positive (no gas)
        v = hi
        # ILLINOIS-modified false position. Plain false position STAGNATES here:
        # the margin is strongly convex in V, so one endpoint is retained every
        # iteration and the bracket never halves. Under the old 24-iteration cap
        # that truncated mid-flight and returned the truncated value as if it had
        # converged -- 3.7% low on a shoe-bound case, and erratic enough to fake a
        # cusp in KT-vs-bit-position. Halving the retained endpoint's margin forces
        # the bracket down and restores superlinear convergence; the cap is now a
        # backstop rather than the thing that ends the loop.
        retained = 0
        for _ in range(_SECANT_MAX_ITER):
            denom = m_hi - m_lo
            v = hi - m_hi * (hi - lo) / denom if abs(denom) > 1e-9 else 0.5 * (lo + hi)
            if not (lo < v < hi):
                v = 0.5 * (lo + hi)
            m, _, _ = _margin_bottom(b, v)
            if abs(m) < _SECANT_TOL_PSI or (hi - lo) < _SECANT_TOL_BBL:
                break
            if m > 0.0:
                if retained > 0:
                    m_hi *= 0.5                           # Illinois
                lo, m_lo, retained = v, m, max(retained, 0) + 1
            else:
                if retained < 0:
                    m_lo *= 0.5                           # Illinois
                hi, m_hi, retained = v, m, min(retained, 0) - 1
        m, gt, db = _margin_bottom(b, v)
        return v, gt, db

    # Per-candidate breach influx; the kick tolerance is the MIN over candidates.
    v_star = np.inf
    best = (np.nan, np.nan, np.nan)
    for d in top_pins:                                    # families 1/4: closed form
        r = _breach_v_gas_top(d)
        if r is not None and r[0] < v_star:
            v_star, best = r[0], (d, r[1], d)
    # Fixed envelope depths -- the depths at which the constraint can turn, with the
    # gas faces excluded (those are families 1/4).
    _env_d = (check_arr if check_arr is not None else np.array(sorted(
        x for x in (set(pf_breaks)
                    | {s.top_tvd for s in ss if s.is_open_hole}
                    | {s.bottom_tvd for s in ss if s.is_open_hole})
        if any(s.is_open_hole and s.top_tvd <= x <= s.bottom_tvd for s in ss)),
        dtype=float))

    for b in bottom_pins:                                 # families 2/3: CLOSED FORM
        for d in _env_d:
            r = _breach_v_gas_bottom_at(b, float(d))
            if r is not None and r[0] < v_star:
                v_star, best = r[0], (r[1], b, float(d))

    # OFF-DIAGONAL: gas TOP pinned at a boundary while a DIFFERENT depth binds.
    # With a long tight bottom section the gas top sits on the BHA-top boundary and
    # the shoe is what breaches; that pairing is on neither diagonal above.
    for zt in top_pins:
        for d in _env_d:
            if float(d) >= float(zt):
                continue
            r = _breach_v_gas_top_at(float(zt), float(d))
            if r is not None and r[0] < v_star:
                v_star, best = r[0], (float(zt), r[1], float(d))

    from .migration import maasp as _maasp
    try:
        _mr = _maasp(ss, fp, rho_mud_ppg=rho_mud_ppg,
                     check_depths=list(_env_d) if len(_env_d) else None)
        _mk = dict(maasp_psi=_mr.maasp_psi,
                   maasp_governing_tvd=_mr.governing_tvd,
                   maasp_governed_by_shoe=_mr.governed_by_shoe)
    except ValueError:                       # no open hole -- MAASP undefined
        _mk = {}

    # ALREADY FRACTURED, NO INFLUX. An empty breach-candidate set has TWO physically
    # OPPOSITE causes, and the per-depth solves return None for both: the shoe is far
    # too strong to breach (genuinely unconstrained), or the MUD COLUMN ALONE already
    # meets FP so there is no intact state to grow a bubble from (`FP <= A` above).
    # Collapsing the second into "the shoe holds through full open-hole displacement"
    # reports the full open-hole capacity for a well that is losing returns before any
    # gas enters -- the unsafe direction, and the same defect class as the clamped
    # bubble height in `core.drill_kick`. Separate them explicitly.
    # Raised by welleng-api 2026-07-27 (Finding D): their design-curve sweep shifts FP
    # down by 2 ppg, hit this at the weakest shoe, and got `open_hole_unconstrained`
    # with the SAME volume as the strongest shoe in the sweep.
    if not np.isfinite(v_star):
        _mud_breach = [
            float(d) for d in _env_d
            if ppg_to_psi(float(fp_fn(np.array([float(d)]))[0]), float(d))
            <= bhp_psi - g * rho_mud_ppg * (bottom_tvd - float(d))
        ]
        if _mud_breach:
            return AnalyticalKickTolerance(
                0.0, np.nan, np.nan, float(min(_mud_breach)), False,
                {"note": (
                    "The mud column ALONE already meets or exceeds the fracture "
                    f"pressure at {min(_mud_breach):.1f} ft with NO influx in the "
                    "hole, so there is no tolerable kick: the well is losing returns "
                    "before any gas enters. Tolerance reported as 0.0. This is NOT "
                    "'the shoe holds' -- it is the opposite. Reduce mud weight or "
                    "set casing before this section."
                )},
                surface_containment_bbl=surface_bbl,
                casing_binds=False,
                already_fractured=True,
                **_mk,
            )

    # If no fracture breach is reachable within the exposed hole, the shoe holds to
    # full displacement -- route through the casing-burst / full-displacement handling
    # below (v_hole caps the displacement bisection).
    if not np.isfinite(v_star) or v_star >= v_hole - 1e-9:
        v_star, best = v_hole, (np.nan, np.nan, np.nan)

    gt, gb, dbind = best

    # Santos SPE-140113: a single coherent bubble cannot be LONGER than the open hole
    # it occupies. If the fracture-breach influx above needs a bubble longer than the
    # open hole, that gas-top-at-shoe worst case is UNREACHABLE -- by the time the
    # bubble tail clears TD its top is already above the shoe. So the SHOE HOLDS
    # through full open-hole displacement and the OPEN HOLE does not constrain the kick
    # tolerance at the provided fracture pressure. This is NOT "unlimited", and we do
    # NOT claim what the limit IS: this assessment stops at the shoe. We assert only
    # what we checked -- the open hole is not the constraint here. Report the full
    # open-hole gas capacity with that flag, NOT a misleading fracture KT. [JJ]
    from .migration import _fill_down, pressure_at_depth

    oh_len = sum(s.bottom_tvd - s.top_tvd for s in ss if s.is_open_hole)

    _OPEN_HOLE_UNCONSTRAINED_NOTE = {
        "note": ("The open hole does not constrain the kick tolerance at the provided "
                 "(uncertain) fracture pressure: the shoe holds through full open-hole "
                 "displacement. The reported volume is the full open-hole gas capacity, "
                 "NOT the kick tolerance -- this is not 'unlimited'. Limits are NOT "
                 "assessed here: above the shoe (e.g. casing burst as the gas reaches "
                 "surface) is outside this open-hole check, and sub-shoe leak-off into "
                 "permeable formations is not modelled. The governing limit lies beyond "
                 "what is assessed."),
    }

    # Open-hole capacity WITHOUT a full march (was ~5 s of thorough marches per call;
    # a downstream perf regression). The bubble is longest at its most-expanded
    # position -- gas top at surface -- so a single-position evaluation there gives the
    # max gas length. This mirrors the migration's per-position calc EXACTLY (same
    # P_rep seed, Boyle expansion, _fill_down and damped fixed point on the gas-top
    # pressure via pressure_at_depth), just at the one governing position instead of
    # marching all of them -- ~100x cheaper, same numbers (cross-checked analytical vs
    # march in test_kick_analytical).
    def _gas_len_at_surface(v_bh: float) -> float:
        gas_top = 0.0
        P_rep = max(bhp_psi - g * rho_mud_ppg * (bottom_tvd - gas_top), 1.0)
        T_local = float(temp_fn(gas_top))
        gas_len = 0.0
        for _ in range(100):
            Z = zf(P_rep, T_local)
            V = v_bh * (P_bh * Z * T_local) / (P_rep * Z_bh * T_bh_r)
            gas_bottom, gas_len = _fill_down(gas_top, V, ss, bottom_tvd)
            P_new = max(float(pressure_at_depth(
                gas_top, gas_top_tvd=gas_top, gas_bottom_tvd=gas_bottom,
                bottom_tvd=bottom_tvd, bhp_psi=bhp_psi, rho_mud_ppg=rho_mud_ppg,
                gas_bh=gas_bh, gas_density_mode=gas_density_mode,
                temp_profile=temp_profile, n_sub=20, z_fn=z_fn)), 1.0)
            if abs(P_new - P_rep) < 1e-4:
                break
            P_rep = 0.5 * (P_rep + P_new)
        return gas_len

    if _gas_len_at_surface(v_star) > oh_len:               # fracture breach unreachable
        lo, hi = 0.0, v_star                              # -> full open-hole displacement
        for _ in range(40):                               # bisect gas_len == oh_len
            if hi - lo <= 1e-2:
                break
            mid = 0.5 * (lo + hi)
            if _gas_len_at_surface(mid) <= oh_len:
                lo = mid
            else:
                hi = mid
        shoe = min((s.top_tvd for s in ss if s.is_open_hole), default=np.nan)
        return AnalyticalKickTolerance(
            float(lo), 0.0, float(oh_len),
            float(shoe), True, dict(_OPEN_HOLE_UNCONSTRAINED_NOTE),
            surface_bbl, surface_bbl is not None and surface_bbl < float(lo),
            **_mk)

    return AnalyticalKickTolerance(
        float(v_star), float(gt), float(gb),
        float(dbind) if dbind == dbind else np.nan, False, {},
        surface_bbl, surface_bbl is not None and surface_bbl < float(v_star),
        **_mk)


def swab_worst_bit(
    sections_for_bit,
    pp: ProfileLike,
    fp: ProfileLike,
    *,
    bhp_psi: float,
    rho_mud_ppg: float,
    gas_bh_state,
    bottom_tvd: float,
    shoe_tvd: float,
    gas_density_mode: str = "conservative",
    temp_profile: TempProfileLike = None,
    geothermal: TempProfileLike = None,
    **kwargs,
):
    """Worst string position for the SWAB case, and the kick tolerance there.

    Tripping in, the BHA can intercept the bubble at any depth, so unlike the
    drilling case (BHA on bottom, geometry fixed) the string position is a
    variable. This finds the worst one WITHOUT searching over it.

    Industry commonly assumes the worst case is *bubble top at the shoe with the
    bit at the bubble's bottom.* That is **not conservative** -- it hard-codes the
    shoe as the binding depth. Measured against the true worst it is +0.68%,
    +0.79% and **+25.42%** high, the last on a well whose weak zone sits below the
    shoe. Right shape, wrong depth.

    The worst position is DETERMINATE. The gas length ``L(d)`` is density-driven
    and capacity-independent, so with the bubble top pinned at a candidate binding
    depth ``d``::

        bit* = d + L(d),      L(d) from   FP(d)*exp(k*L) = A + b*L

    one closed-form solve per candidate depth, no search. ``bit*`` is itself a
    breakpoint -- it is exactly where the gas top lands on the pressure knot, so
    the binding candidate switches there -- hence both sides of each are
    evaluated and the worse taken.

    Do NOT be tempted by the fixed point ``bit := gas_bottom(bit)``: once the bit
    is deep enough the gas bottom equals the bit identically, so every position
    satisfies it and any iteration "converges" to wherever it started.

    Parameters
    ----------
    sections_for_bit : callable
        ``bit_tvd -> list[WellSection]``. Builds the annulus for a string
        position: the BHA/collar annulus above the bit, open hole below it.
    pp, fp : ProfileLike
        Pore and fracture profiles as ``(tvd, ppg)`` tables.
    bhp_psi, rho_mud_ppg, gas_bh_state
        As :func:`analytical_kick_tolerance`.
    bottom_tvd, shoe_tvd
        Well TD and casing shoe [ft TVD].
    **kwargs
        Passed through to :func:`analytical_kick_tolerance`.

    Returns
    -------
    (AnalyticalKickTolerance, float)
        The result at the worst string position, and that position [ft TVD].
    """
    g = G_PSI_PER_PPG_FT
    fp_fn = _as_ppg_callable(fp)
    P_bh, T_bh_r, Z_bh, rho_bh = _resolve_bh_state(gas_bh_state, bhp_psi)
    temp_fn = _as_temp_callable(temp_profile if temp_profile is not None else geothermal,
                                T_bh_r)

    def _length_at(d):
        """Cap-independent gas length with the bubble top pinned at ``d``."""
        FP = ppg_to_psi(fp_fn(np.array([d]))[0], d)
        A = bhp_psi - g * rho_mud_ppg * (bottom_tvd - d)
        bm = g * rho_mud_ppg
        if FP <= A:
            return None
        T_d = float(temp_fn(d))
        L = (FP - A) / bm
        for _ in range(40):
            Z_c = _z(0.5 * (FP + A + bm * max(L, 0.0)), T_d)
            k = g * rho_bh * Z_bh * T_bh_r / (P_bh * Z_c * T_d)
            e = np.exp(k * L)
            f, df = FP * e - (A + bm * L), FP * k * e - bm
            if abs(df) < 1e-12:
                break
            Ln = L - f / df
            if abs(Ln - L) < 1e-7:
                L = Ln
                break
            L = max(Ln, 0.0)
        return L if L > 0.0 else None

    # Candidate binding depths: the shoe plus every pressure knot in the open hole.
    candidates = {float(shoe_tvd)}
    for prof in (pp, fp):
        candidates.update(x for x in _profile_breakpoints(prof)
                          if shoe_tvd < x < bottom_tvd)

    bits = set()
    for d in candidates:
        L = _length_at(d)
        if L is None:
            continue
        star = d + L
        # bit* is a breakpoint: the binding candidate switches across it, so take
        # both sides rather than landing exactly on the switch.
        for bit in (star - 1e-3, star, star + 1e-3):
            if shoe_tvd < bit <= bottom_tvd:
                bits.add(min(bit, bottom_tvd))

    if not bits:
        raise ValueError(
            "no admissible string position: the fracture profile is never "
            "reachable from the bottom-hole pressure over the open hole."
        )

    best, best_bit = None, None
    for bit in sorted(bits):
        r = analytical_kick_tolerance(
            sections_for_bit(bit), pp, fp, bhp_psi=bhp_psi,
            rho_mud_ppg=rho_mud_ppg, gas_bh_state=gas_bh_state,
            gas_density_mode=gas_density_mode, temp_profile=temp_profile,
            geothermal=geothermal, **kwargs,
        )
        if best is None or r.max_influx_bbl < best.max_influx_bbl:
            best, best_bit = r, bit
    return best, best_bit
