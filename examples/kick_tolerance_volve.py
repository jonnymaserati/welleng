"""Maximum gas kick tolerance on public Volve field data (well F-13).

Self-contained field demonstration for the welleng kick-tolerance engine. It
solves the INVERSE problem -- the maximum single gas influx that can be
circulated out within the pore/fracture envelope over the whole migration -- and
reports the binding depth and mechanism, both isothermal (at the EDM bottom-hole
temperature) and with the real Volve geothermal gradient.

Data: Equinor Volve EDM, well F-13 (wellbore 0OL2SUGd7c), CC BY 4.0 -- attribute
Equinor and the Volve licence partners. The pore/fracture/survey/geometry
profiles are shipped alongside this script as
``examples/data/volve_f13_kick_tolerance_profiles.npz`` (derived from the public
Volve EDM export), so the example runs without the ~211 MB source XML.

Run:  python examples/kick_tolerance_volve.py

Reproduces the field-demonstration figures in the note "The welleng
kick-tolerance engine: derivation, provenance and public validation against
published methods" (Corcutt, 2026): 79.4 bbl isothermal, 86.0 bbl with the real
Volve geothermal gradient, both binding at 7904 ft TVD by fracture.
"""
from __future__ import annotations

import os

import numpy as np

from welleng.kick_tolerance import max_influx_circulated, WellSection

HERE = os.path.dirname(__file__)
NPZ = os.path.join(HERE, "data", "volve_f13_kick_tolerance_profiles.npz")

# Real Volve F-13 geothermal anchors (EDM CD_TEMP_GRADIENT group nWfrh) [ft, degF].
VOLVE_GEO_TVD = np.array([380.0, 8996.0])          # mudline, explicit EDM point
VOLVE_GEO_DEGF = np.array([40.0, 204.8])
DP_OD_IN = 5.0                                      # drill-pipe OD for the annulus


def volve_geothermal_rankine(td_tvd: float):
    """(tvd, T_rankine) table: mudline + the EDM anchor, extrapolated to TD."""
    grad = (VOLVE_GEO_DEGF[1] - VOLVE_GEO_DEGF[0]) / (VOLVE_GEO_TVD[1] - VOLVE_GEO_TVD[0])
    t_td = VOLVE_GEO_DEGF[0] + grad * (td_tvd - VOLVE_GEO_TVD[0])
    tvd = np.array([VOLVE_GEO_TVD[0], VOLVE_GEO_TVD[1], td_tvd])
    degf = np.array([VOLVE_GEO_DEGF[0], VOLVE_GEO_DEGF[1], t_td])
    return tvd, degf + 460.0, grad, t_td


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    pp = (d["pp_tvd"], d["pp_ppg"])
    fp = (d["fp_tvd"], d["fp_ppg"])
    shoe = float(d["tvd_shoe"]); td = float(d["td_tvd"])
    mud = float(d["mud_ppg"]); bhp = float(d["bhp_psi"]); bht = float(d["bht_degf"])

    cap_cased = (12.415 ** 2 - DP_OD_IN ** 2) / 1029.4   # 13-3/8" cased, 5" DP
    cap_open = (8.5 ** 2 - DP_OD_IN ** 2) / 1029.4        # 8.5" open hole, 5" DP
    sections = [
        WellSection(top_tvd=0.0, bottom_tvd=shoe,
                    annular_capacity_bbl_per_ft=cap_cased, is_open_hole=False),
        WellSection(top_tvd=shoe, bottom_tvd=td,
                    annular_capacity_bbl_per_ft=cap_open, is_open_hole=True),
    ]
    common = dict(sections=sections, pp=pp, fp=fp, bhp_psi=bhp, rho_mud_ppg=mud,
                  gas_bh_state=(None, bht + 460.0, None, None), n_steps=150,
                  v_cap_bbl=300.0)

    geo_tvd, geo_r, grad, t_td = volve_geothermal_rankine(td)

    print(f"Volve F-13 (0OL2SUGd7c): 13-3/8\" shoe TVD {shoe:.0f} / TD {td:.0f} ft")
    print(f"  mud {mud} ppg, BHP {bhp:.0f} psi, BHT {bht:.0f} degF")
    print(f"  annular cap (5\" DP): cased {cap_cased:.4f} / open {cap_open:.4f} bbl/ft")
    print(f"  Volve geothermal: {grad:.4f} degF/ft -> {t_td:.0f} degF @ TD")
    iso = max_influx_circulated(**common)
    geo = max_influx_circulated(**common, geothermal=(geo_tvd, geo_r))
    for name, r in [("isothermal @ BHT", iso), ("real Volve geothermal", geo)]:
        print(f"  {name:24s}  MAX gas KT = {r.max_influx_bbl:6.1f} bbl "
              f"  binds @ TVD {r.binding_tvd:.0f} ft  by {r.limited_by}")


if __name__ == "__main__":
    main()
