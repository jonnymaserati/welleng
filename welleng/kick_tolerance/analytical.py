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

Every candidate is located by ONE primitive, :func:`_top_for_bottom`: pin the gas
BOTTOM at a depth (its pressure is then known from the mud column below the BHP)
and fill the influx UP the annulus to read off the gas top -- exact mode by a
closed-form exponential mass integral, conservative mode by a short constant-
density fixed point. Family (1) inverts the same primitive (bisect the bottom that
puts the top on a boundary).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

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


@dataclass
class AnalyticalKickTolerance:
    """Result of :func:`analytical_kick_tolerance`."""

    max_influx_bbl: float          # bottom-hole influx at first fracture [bbl]
    binding_gas_top_tvd: float     # gas-top TVD of the binding config [ft]
    binding_gas_bottom_tvd: float  # gas-bottom TVD of the binding config [ft]
    binding_depth_tvd: float       # exposed depth where the envelope is tight [ft]
    is_unlimited: bool             # True if the whole exposed hole tolerates gas
    breakpoints: dict              # {label: influx-at-breach} for inspection


def _top_for_bottom(gas_bottom, influx_bbl_bh, sections_sorted, bottom_tvd, *,
                    bhp_psi, rho_mud_ppg, gas_bh, temp_fn,
                    gas_density_mode="conservative"):
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
                    cap = sec.annular_capacity_bbl_per_ft
                if sec.top_tvd < z and sec.top_tvd > seg_top_limit:
                    seg_top_limit = sec.top_tvd
            if cap is None:
                cap = sections_sorted[0].annular_capacity_bbl_per_ft
            seg = z - seg_top_limit
            T = float(temp_fn(0.5 * (z + seg_top_limit)))
            Z = _z(max(P, 1.0), T)
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
        Z_gt = _z(max(P_gt, 1.0), T_gt)
        Ve = influx_bbl_bh * (P_bh * Z_gt * T_gt) / (max(P_gt, 1.0) * Z_bh * T_bh_r)
        gas_top = _fill_up(gas_bottom, Ve, sections_sorted)
        rho_top = rho_bh * max(P_gt, 1.0) * Z_bh * T_bh_r / (P_bh * Z_gt * T_gt)  # ppg
        P_new = max(P_gb - rho_top * g * (gas_bottom - gas_top), 1.0)
        if abs(P_new - P_gt) < 1e-4:
            P_gt = P_new
            break
        P_gt = 0.5 * (P_gt + P_new)
    return max(gas_top, 0.0)


def _min_margin(gas_top, gas_bottom, exposed_depths, pp_psi, fp_psi, *,
                bottom_tvd, bhp_psi, rho_mud_ppg, gas_bh, gas_density_mode,
                temp_profile):
    """(worst margin, binding depth) over the exposed depths for a config.

    The margin is the min of the FP margin (``FP-P >= 0``: no breakdown) and the
    PP margin (``P-PP >= 0``: no further influx).
    """
    P = pressure_at_depth(
        exposed_depths, gas_top_tvd=gas_top, gas_bottom_tvd=gas_bottom,
        bottom_tvd=bottom_tvd, bhp_psi=bhp_psi, rho_mud_ppg=rho_mud_ppg,
        gas_bh=gas_bh, gas_density_mode=gas_density_mode, temp_profile=temp_profile,
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
    """
    ss = sorted(sections, key=lambda s: s.top_tvd)
    bottom_tvd = max(s.bottom_tvd for s in ss)
    gas_bh = _resolve_bh_state(gas_bh_state, bhp_psi)
    if temp_profile is None:
        temp_profile = geothermal
    temp_fn = _as_temp_callable(temp_profile, gas_bh[1])
    pp_fn, fp_fn = _as_ppg_callable(pp), _as_ppg_callable(fp)

    # Section boundaries (gas-face candidates) and the exposed open-hole depths at
    # which the envelope is enforced.
    boundaries = sorted({s.top_tvd for s in ss} | {s.bottom_tvd for s in ss})
    pf_breaks = [b for b in (_profile_breakpoints(pp) + _profile_breakpoints(fp))
                 if 0.0 < b < bottom_tvd]

    def exposed_for(gas_top, gas_bottom):
        # FP/PP margin over a mud/gas column is piecewise-monotone; its extremum
        # is at a breakpoint depth: open-hole boundaries, PP/FP breaks, and the
        # gas faces themselves. Enforce the envelope only where formation is open.
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
    v_hole = sum(s.annular_capacity_bbl_per_ft * (s.bottom_tvd - s.top_tvd)
                 for s in ss if s.is_open_hole)

    # A BOUNDARY is any discrete change in the problem -- a section-capacity
    # change OR a PP/FP breakpoint. Both gas faces are candidate-pinned at every
    # boundary (JJ): gas BOTTOM at a boundary + TD (families 2/3), gas TOP at a
    # boundary (families 1/4).
    bset = sorted(set(boundaries) | set(pf_breaks))
    bottom_pins = [b for b in bset if b > 0.0] + [bottom_tvd]
    top_pins = [b for b in bset if 0.0 < b < bottom_tvd]

    def _top_at(gas_bottom, V):
        return _top_for_bottom(gas_bottom, V, ss, bottom_tvd, bhp_psi=bhp_psi,
                               rho_mud_ppg=rho_mud_ppg, gas_bh=gas_bh,
                               temp_fn=temp_fn, gas_density_mode=gas_density_mode)

    def worst_margin(V):
        best_m, best = np.inf, (np.nan, np.nan, np.nan)

        def consider(gt, gb):
            nonlocal best_m, best
            d = exposed_for(gt, gb)
            if not d.size:
                return
            mv, db = _min_margin(gt, gb, d, ppg_to_psi(pp_fn(d), d),
                                 ppg_to_psi(fp_fn(d), d), bottom_tvd=bottom_tvd,
                                 bhp_psi=bhp_psi, rho_mud_ppg=rho_mud_ppg,
                                 gas_bh=gas_bh, gas_density_mode=gas_density_mode,
                                 temp_profile=temp_profile)
            if mv < best_m:
                best_m, best = mv, (gt, gb, db)

        # families 2/3: gas BOTTOM pinned at each boundary + TD
        for b in bottom_pins:
            consider(_top_at(b, V), b)

        # families 1/4: gas TOP pinned at each boundary -- find the gas bottom that
        # lands the top there. _top_at is monotone increasing in gas_bottom, so a
        # secant converges in a few steps (vs a 40-iter bisection); bracket-guarded.
        for bnd in top_pins:
            f_hi = _top_at(bottom_tvd, V) - bnd
            if f_hi < -1e-6:
                continue  # even the deepest bubble already sits above bnd
            lo, hi = bnd, bottom_tvd            # f(lo) = _top_at(bnd,V)-bnd <= 0 <= f_hi
            f_lo = _top_at(lo, V) - bnd
            gb = hi
            for _ in range(20):
                denom = f_hi - f_lo
                cand = hi - f_hi * (hi - lo) / denom if abs(denom) > 1e-12 else 0.5 * (lo + hi)
                if not (lo <= cand <= hi):       # secant left the bracket -> bisect
                    cand = 0.5 * (lo + hi)
                fc = _top_at(cand, V) - bnd
                gb = cand
                if abs(fc) < 0.25:               # top placed within 0.25 ft
                    break
                if fc > 0.0:
                    hi, f_hi = cand, fc
                else:
                    lo, f_lo = cand, fc
            if abs(_top_at(gb, V) - bnd) > 1.0:   # could not place the top -> skip
                continue
            consider(bnd, gb)

        return best_m, best

    # If the whole hole tolerates gas, tolerance is unlimited (full displacement).
    m_ceil, _ = worst_margin(v_hole)
    if m_ceil >= 0.0:
        return AnalyticalKickTolerance(v_hole, np.nan, np.nan, np.nan, True, {})

    # Bisect the influx to first breach (worst_margin is monotone decreasing in V).
    lo, hi = 0.0, v_hole
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        m, _ = worst_margin(mid)
        if m > 0.0:
            lo = mid
        else:
            hi = mid
    vstar = lo
    _, (gt, gb, dbind) = worst_margin(vstar)
    return AnalyticalKickTolerance(
        float(vstar), float(gt), float(gb),
        float(dbind) if dbind == dbind else np.nan, False, {})
