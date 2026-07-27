"""Analytical (breakpoint) kick-tolerance solver.

The migration :func:`~welleng.kick_tolerance.migration.max_influx_circulated`
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
  gas-top pin can't express) -- a secant on the influx, using :func:`_top_for_bottom`
  to place the gas for each trial influx.

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
) -> AnalyticalKickTolerance:
    """Max bottom-hole influx tolerable over the whole migration, by breakpoints.

    Breakpoint alternative to
    :func:`~welleng.kick_tolerance.migration.max_influx_circulated`: evaluates the
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
    if temp_profile is None:
        temp_profile = geothermal
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
        cross-checked against the thorough march (``max_influx_circulated``, which
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
    for b in bottom_pins:                                 # families 2/3: secant on V
        r = _breach_v_gas_bottom(b)
        if r is not None and r[0] < v_star:
            v_star, best = r[0], (r[1], b, r[2])

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
            surface_bbl, surface_bbl is not None and surface_bbl < float(lo))

    return AnalyticalKickTolerance(
        float(v_star), float(gt), float(gb),
        float(dbind) if dbind == dbind else np.nan, False, {},
        surface_bbl, surface_bbl is not None and surface_bbl < float(v_star))
