"""Shut-in gas migration: the closure is CONSTANT VOLUME, not pinned pressure.

The circulated cases hold bottom-hole pressure constant with the choke, so the
gas expands as it rises and the annulus pressure is a consequence. A SHUT-IN
well is closed: nothing leaves, so the total volume is fixed and the gas can
only expand by whatever the mud gives back under compression.

That single substitution -- **BHP pinned -> volume pinned** -- changes the
character of the answer completely:

* The gas carries its pressure upward. With ``V`` and ``n`` fixed,
  ``P = rho.Z.R.T/M`` varies only through ``Z.T``, so the bubble arrives at the
  shoe still close to formation pressure. Everything below it is then
  over-pressured by the mud it displaced, so **BHP RISES as the gas migrates**,
  and so does surface pressure. That is the hazard, and it is the opposite of
  the circulated case.
* **The rigid limit is DEGENERATE, not conservative.** With no relief the gas
  density is fixed, so the pressure it imposes when it reaches the shoe does not
  depend on how much gas there is: measured 5913 psi for a 5 bbl kick and for a
  120 bbl kick, identical. There is no tolerance to compute -- either formation
  pressure exceeds the shoe strength or it does not, whatever the kick size.
* **The tolerance is created ENTIRELY by the relief.** On a 10,500 ft well with
  12.6 ppg pore against a 14 ppg shoe:

  =================  ==============
  c [1/psi]          tolerance [bbl]
  =================  ==============
  0 (rigid)          0.00
  1e-6               1.97
  3e-6 (WBM)         5.39
  5e-6 (OBM)         8.26
  1.2e-5             15.55
  =================  ==============

  So compressibility is not a refinement here. It is constitutive.

Scope
-----
This reports **whether and where a barrier is breached**, not what follows from
the breach. Consequences of failure are out of scope by design.

The relief is an EFFECTIVE compressibility: mud, casing ballooning and formation
compliance all contribute and all have the same sign. Lumping them into one
number is honest; calling it "mud compressibility" would not be. Excluding
ballooning is conservative in the same direction as excluding the mud's own
compressibility.

``welleng.fluid.Fluid`` is NOT used as the source: its density responds to
applied pressure two orders of magnitude too weakly (see the warning on that
class). ``c`` is an explicit input until the welleng-drilling fluid model is
ported, at which point it becomes ``c(P, T)``.

Not modelled, and each breaks the closure in the UNSAFE direction: a second
influx, and losses at the shoe taking mud away. Neither belongs in a
single-bubble model; both are stated boundaries.

.. warning::
   **This module has NO external validation anchor.** Its tests assert internal
   consistency -- monotone in the compressibility, monotone in the influx, the
   rigid limit degenerate, the closure solved as a root rather than iterated --
   and nothing more. The circulated path is anchored to NOGEPA-50 Section 3.2;
   this one is anchored to nothing.

   The natural anchor is the BP Exploration kick-tolerance workbook, which
   computes BOTH a shut-in and a circulating tolerance and takes the lower. That
   workbook is proprietary: it may be used to check numbers locally, but nothing
   from it -- values, formulae or structure -- may enter this repository.

   Until that check is done, treat the output as indicative and do not put it in
   front of a well programme.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .migration import (
    G_PSI_PER_PPG_FT,
    WellSection,
    _as_ppg_callable,
    _as_temp_callable,
    _profile_breakpoints,
    _resolve_bh_state,
    ppg_to_psi,
    ProfileLike,
    TempProfileLike,
)
from .analytical import _z

#: Effective compressibility of a water-based mud system [1/psi]. Indicative.
WBM_COMPRESSIBILITY = 3.0e-6
#: ... and of an oil-based system, which is the more compressible.
OBM_COMPRESSIBILITY = 5.0e-6


@dataclass
class ShutInBreach:
    """Where a shut-in migration first breaches, or that it does not."""

    breached: bool
    breach_depth_tvd: Optional[float]       # the exposed depth that goes first
    breach_gas_top_tvd: Optional[float]     # bubble top position when it goes
    max_sicp_psi: float                     # highest surface pressure reached
    max_bhp_psi: float                      # highest bottom-hole pressure reached
    gas_expansion_frac: float               # relief the mud gave back, as a fraction


def _bubble_pressure(volume_bbl, influx_bbl, gas_top, *, cap_bbl_per_ft,
                     bhp_initial_psi, gas_bh, temp_fn):
    """Gas pressure for a bubble of ``volume_bbl`` with its top at ``gas_top``.

    Mass is conserved and the well is closed, so ``P.V/(Z.T)`` is invariant:
    the pressure follows from the volume, not from a hydrostatic balance.
    """
    P_bh, T_bh_r, Z_bh, _ = gas_bh
    length = volume_bbl / cap_bbl_per_ft
    t_local = float(temp_fn(gas_top + 0.5 * length))
    pressure = bhp_initial_psi * influx_bbl / volume_bbl
    for _ in range(40):
        z = _z(max(pressure, 1.0), t_local)
        updated = (bhp_initial_psi * influx_bbl * (z * t_local)
                   / (Z_bh * T_bh_r) / volume_bbl)
        if abs(updated - pressure) < 1e-10:
            return updated, length
        pressure = updated
    return pressure, length


def _state(influx_bbl, gas_top, *, compressibility, well_mud_bbl, cap_bbl_per_ft,
           bhp_initial_psi, bottom_tvd, rho_mud_ppg, gas_bh, temp_fn):
    """(P_gas, volume, SICP, BHP) with the bubble top at ``gas_top``.

    The closure -- gas volume equals what the compressed mud has given back --
    is a ROOT, not a fixed point to iterate. Iterating it diverges: the gas
    pressure goes as 1/V, so at small volumes the implied relief overshoots and
    lands in a different basin, which shows up as a tolerance that is not
    monotone in the compressibility. Bracketed and solved instead.
    """
    g = G_PSI_PER_PPG_FT
    sicp_initial = bhp_initial_psi - g * rho_mud_ppg * bottom_tvd

    def field(volume):
        pressure, length = _bubble_pressure(
            volume, influx_bbl, gas_top, cap_bbl_per_ft=cap_bbl_per_ft,
            bhp_initial_psi=bhp_initial_psi, gas_bh=gas_bh, temp_fn=temp_fn)
        sicp = pressure - g * rho_mud_ppg * gas_top
        bhp = pressure + g * rho_mud_ppg * (bottom_tvd - gas_top - length)
        mean_rise = 0.5 * ((sicp - sicp_initial) + (bhp - bhp_initial_psi))
        return pressure, sicp, bhp, mean_rise

    if compressibility <= 0.0:
        pressure, sicp, bhp, _ = field(influx_bbl)
        return pressure, influx_bbl, sicp, bhp

    def closure(volume):
        return volume - influx_bbl - well_mud_bbl * compressibility * field(volume)[3]

    lo, hi = influx_bbl, influx_bbl
    for _ in range(200):
        hi *= 1.2
        if closure(hi) > 0.0:
            break
    else:                                       # pragma: no cover - runaway guard
        raise ValueError(
            "the volume closure does not bracket; check c and well_mud_bbl"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hi - lo < 1e-10:
            break
        if closure(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    volume = 0.5 * (lo + hi)
    pressure, sicp, bhp, _ = field(volume)
    return pressure, volume, sicp, bhp


def shut_in_migration(
    sections: Sequence[WellSection],
    pp: ProfileLike,
    fp: ProfileLike,
    *,
    influx_bbl: float,
    bhp_initial_psi: float,
    rho_mud_ppg: float,
    gas_bh_state,
    well_mud_bbl: float,
    bottom_tvd: Optional[float] = None,
    compressibility_per_psi: float = 0.0,
    burst_pressure_psi: Optional[float] = None,
    temp_profile: TempProfileLike = None,
    geothermal: TempProfileLike = None,
) -> ShutInBreach:
    """Migrate a shut-in influx and report the first barrier breach.

    Pressure rises monotonically as the bubble rises, so the FIRST breach is the
    answer and the bubble positions worth evaluating are the breakpoints -- the
    section boundaries and the pressure-profile knots, where either the capacity
    or the limit changes. No march.

    ``compressibility_per_psi`` is an EFFECTIVE value covering mud, casing and
    formation compliance together. Leaving it at zero gives the rigid limit,
    which is degenerate rather than conservative -- see the module docstring.
    """
    ss = sorted(sections, key=lambda s: s.top_tvd)
    td = float(bottom_tvd) if bottom_tvd is not None else max(s.bottom_tvd for s in ss)
    gas_bh = _resolve_bh_state(gas_bh_state, bhp_initial_psi)
    temp_fn = _as_temp_callable(
        temp_profile if temp_profile is not None else geothermal, gas_bh[1])
    fp_fn = _as_ppg_callable(fp)

    open_hole = [s for s in ss if s.is_open_hole]
    if not open_hole:
        raise ValueError("no open-hole section: nothing to breach")
    cap = open_hole[-1].capacity_per_tvd_ft

    checks = sorted({s.top_tvd for s in open_hole} | {s.bottom_tvd for s in open_hole}
                    | {b for b in _profile_breakpoints(fp) if 0.0 < b < td})
    shoe = min(s.top_tvd for s in open_hole)
    positions = sorted({p for p in (checks + [shoe]) if 0.0 <= p < td}, reverse=True)

    max_sicp, max_bhp, expansion = -np.inf, -np.inf, 0.0
    for gas_top in positions:
        pressure, volume, sicp, bhp = _state(
            influx_bbl, gas_top, compressibility=compressibility_per_psi,
            well_mud_bbl=well_mud_bbl, cap_bbl_per_ft=cap,
            bhp_initial_psi=bhp_initial_psi, bottom_tvd=td,
            rho_mud_ppg=rho_mud_ppg, gas_bh=gas_bh, temp_fn=temp_fn)
        max_sicp, max_bhp = max(max_sicp, sicp), max(max_bhp, bhp)
        expansion = max(expansion, volume / influx_bbl - 1.0)

        for depth in checks:
            if depth < gas_top:                 # in the mud above the bubble
                imposed = pressure - G_PSI_PER_PPG_FT * rho_mud_ppg * (gas_top - depth)
            elif depth <= gas_top + volume / cap:
                imposed = pressure             # inside the bubble
            else:
                continue                        # below it: not yet reached
            if imposed > ppg_to_psi(fp_fn(np.array([depth]))[0], depth):
                return ShutInBreach(True, float(depth), float(gas_top),
                                    float(max_sicp), float(max_bhp), float(expansion))
        if burst_pressure_psi is not None and sicp > burst_pressure_psi:
            return ShutInBreach(True, 0.0, float(gas_top),
                                float(max_sicp), float(max_bhp), float(expansion))

    return ShutInBreach(False, None, None, float(max_sicp), float(max_bhp),
                        float(expansion))
