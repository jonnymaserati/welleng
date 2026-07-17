"""NOGEPA Industry Standard No. 50 static single-shoe kick-tolerance formula.

Faithful implementation of the mandated Netherlands well-control standard's
"current industry practice" closed form (NOGEPA Industry Standard No. 50 --
*Kick Tolerances for Well Design and Drilling Operations*, 2020-12-09, §3.2). This
is the STATIC single-shoe check: the gas bubble top at the casing shoe, the shoe
"normally assumed" the weakest exposed zone, temperature and compressibility (Z)
neglected. It is provided so welleng can REPRODUCE the mandated calculation exactly
and so the gas-migration engine's static reduction can be validated against it.

NOTE (from the standard itself): NOGEPA-50 §3.2 states KT "should be calculated in
terms of kick volume which can be circulated out ... [and] should not be
oversimplified," lists the factors this formula neglects (gas velocity/distribution,
P & T, gas composition/dispersion/solubility, friction -- "static conditions"), and
notes the shoe is only *normally* assumed weakest. The migration engine
(:mod:`welleng.kick_tolerance.migration`) is the not-oversimplified circulate-out
calculation the standard calls for; this module is the static baseline.

Drilling-kick equations (NOGEPA-50 p.12-13, verbatim), gas top at the shoe::

    G_mud = mud_ppg * 0.052                        [psi/ft]
    P_frac = LOT_ppg * Z_CSG * 0.052               [psi]  (fracture pressure at shoe)
    H    = (P_f - (Z_OH - Z_CSG)*G_mud - P_frac) / (G_gas - G_mud)   [ft]  gas height
    V1   = H * CAP_ann                             [bbl]  influx volume at the shoe
    V2   = P1 * V1 / P2   (Boyle to bottom hole)   [bbl]  = kick tolerance
           with P1 = P_frac (at shoe), P2 = P_f (formation pressure at TD)

``CAP_ann`` is the annulus capacity between pipe and hole (KT is "hole geometry
minus the drill string"); ``G_gas`` is the gas gradient, 0.05-0.15 psi/ft.
"""
from __future__ import annotations

from dataclasses import dataclass

NOGEPA_G = 0.052   # NOGEPA ppg -> psi/ft gradient constant (mud/fracture)


@dataclass
class NogepaResult:
    """NOGEPA-50 static drilling-kick tolerance and its intermediates."""

    kick_tolerance_bbl: float   # V2 -- KT at bottom-hole conditions   [bbl]
    gas_height_ft: float        # H  -- gas column height at the shoe   [ft]
    influx_at_shoe_bbl: float   # V1 -- influx volume at the shoe       [bbl]
    p_frac_shoe_psi: float      # fracture pressure at the shoe         [psi]


def nogepa_drilling_kick_tolerance(
    *,
    formation_pressure_psi: float,   # P_f  at section TD
    z_oh_ft: float,                  # Z_OH vertical depth of open hole at TD
    z_csg_ft: float,                 # Z_CSG vertical depth of casing shoe
    lot_ppg: float,                  # leak-off test at the shoe
    mud_ppg: float,                  # (maximum) mud weight
    cap_ann_bbl_per_ft: float,       # CAP_ann pipe-to-hole annulus capacity
    g_gas_psi_per_ft: float = 0.1,   # gas gradient (NOGEPA range 0.05-0.15)
) -> NogepaResult:
    """NOGEPA-50 §3.2 static drilling-kick tolerance [bbl] (the ``V2`` value).

    Reproduces the mandated formula exactly (T and Z ignored, per the standard).
    The migration engine reduces to this when the casing shoe genuinely governs.
    """
    g_mud = mud_ppg * NOGEPA_G
    p_frac = lot_ppg * z_csg_ft * NOGEPA_G
    denom = g_gas_psi_per_ft - g_mud            # < 0 (gas lighter than mud)
    if denom == 0:
        raise ValueError("G_gas equals G_mud; gas height is undefined")
    H = (formation_pressure_psi - (z_oh_ft - z_csg_ft) * g_mud - p_frac) / denom
    v1 = H * cap_ann_bbl_per_ft
    v2 = p_frac * v1 / formation_pressure_psi    # P1*V1/P2, Boyle to bottom hole
    return NogepaResult(
        kick_tolerance_bbl=v2,
        gas_height_ft=H,
        influx_at_shoe_bbl=v1,
        p_frac_shoe_psi=p_frac,
    )
