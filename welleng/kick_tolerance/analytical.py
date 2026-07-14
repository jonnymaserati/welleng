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

    P_bh, T_bh_r, Z_bh, rho_bh = gas_bh
    g = G_PSI_PER_PPG_FT

    def _breach_v_gas_top(d):
        """CLOSED-FORM influx at which the imposed pressure at gas-top depth ``d``
        reaches FP(d) (families 1/4). The gas length L is density-driven, so it is
        CAP-INDEPENDENT -- a single closed form even across sections (conservative =
        constant-density linear column; exact = exponential column, a Lambert-W-form
        root by a 3-iter Newton, dependency-free). Only the bbl total walks the
        per-section capacities. Returns ``(V, gas_bottom)`` or None if a gas-top-at-d
        config cannot reach FP or would not fit above TD."""
        FP = ppg_to_psi(fp_fn(np.array([d]))[0], d)
        A = bhp_psi - g * rho_mud_ppg * (bottom_tvd - d)      # mud pressure at d
        b = g * rho_mud_ppg
        if FP <= A:
            return None                                        # already >= FP with no gas
        T_d = float(temp_fn(d))
        if gas_density_mode == "conservative":
            Z_t = _z(FP, T_d)
            rho_top = rho_bh * FP * Z_bh * T_bh_r / (P_bh * Z_t * T_d)
            if b <= rho_top * g:
                return None
            L = (FP - A) / (b - rho_top * g)                   # constant-density (linear)
        else:                                                  # exact exponential column
            L = (FP - A) / b                                   # linear seed
            for _ in range(6):                                 # Newton on FP*exp(kL)=A+bL
                Z_c = _z(0.5 * (FP + A + b * max(L, 0.0)), T_d)
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
                Z_t = _z(FP, T_d)
                rho_top = rho_bh * FP * Z_bh * T_bh_r / (P_bh * Z_t * T_d)
                V += rho_top * s.annular_capacity_bbl_per_ft * (bot - top) / rho_bh
            else:
                Z_c = _z(0.5 * (P + A + b * (bot - d)), T_d)
                k = g * rho_bh * Z_bh * T_bh_r / (P_bh * Z_c * T_d)
                P_bot = P * np.exp(k * (bot - top))
                V += s.annular_capacity_bbl_per_ft * (P_bot - P) / (g * rho_bh)
                P = P_bot
        return V, gas_bottom

    def _margin_bottom(b, V):
        """(min margin, gas_top, binding depth) for the config with gas BOTTOM
        pinned at ``b`` and influx ``V``."""
        gt = _top_for_bottom(b, V, ss, bottom_tvd, bhp_psi=bhp_psi,
                             rho_mud_ppg=rho_mud_ppg, gas_bh=gas_bh, temp_fn=temp_fn,
                             gas_density_mode=gas_density_mode)
        d = exposed_for(gt, b)
        if not d.size:
            return np.inf, gt, np.nan
        mv, db = _min_margin(gt, b, d, ppg_to_psi(pp_fn(d), d), ppg_to_psi(fp_fn(d), d),
                             bottom_tvd=bottom_tvd, bhp_psi=bhp_psi,
                             rho_mud_ppg=rho_mud_ppg, gas_bh=gas_bh,
                             gas_density_mode=gas_density_mode, temp_profile=temp_profile)
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
        for _ in range(24):
            denom = m_hi - m_lo
            v = hi - m_hi * (hi - lo) / denom if abs(denom) > 1e-9 else 0.5 * (lo + hi)
            if not (lo < v < hi):
                v = 0.5 * (lo + hi)
            m, _, _ = _margin_bottom(b, v)
            if abs(m) < 0.25 or (hi - lo) < 1e-3:         # 0.25 psi / 0.001 bbl
                break
            if m > 0.0:
                lo, m_lo = v, m
            else:
                hi, m_hi = v, m
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

    # If nothing breaches within the exposed hole, the tolerance is unlimited.
    if not np.isfinite(v_star) or v_star >= v_hole - 1e-9:
        return AnalyticalKickTolerance(v_hole, np.nan, np.nan, np.nan, True, {})

    gt, gb, dbind = best
    return AnalyticalKickTolerance(
        float(v_star), float(gt), float(gb),
        float(dbind) if dbind == dbind else np.nan, False, {})
