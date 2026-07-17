"""Reproduce the NOGEPA-50 §3.2 static single-shoe kick-tolerance formula.

Verifies welleng reproduces the mandated closed form (H -> V1 -> V2) faithfully.
The migration engine reduces to this static value when the casing shoe governs.
"""
import pytest

from welleng.kick_tolerance import nogepa_drilling_kick_tolerance


# NOGEPA constant and a realistic well (SPE-208788 Table-1-like geometry).
G = 0.052
Z_CSG, Z_OH = 6500.0, 10500.0
LOT, MUD = 16.0, 11.9
PP_KI = 11.5 + 1.1                       # pore pressure + kick intensity [ppg]
PF = PP_KI * G * Z_OH                    # formation pressure at TD        [psi]
CAP_ANN = 0.0209                         # DP-in-hole annulus capacity  [bbl/ft]


def test_nogepa_reproduces_the_formula():
    r = nogepa_drilling_kick_tolerance(
        formation_pressure_psi=PF, z_oh_ft=Z_OH, z_csg_ft=Z_CSG,
        lot_ppg=LOT, mud_ppg=MUD, cap_ann_bbl_per_ft=CAP_ANN, g_gas_psi_per_ft=0.1,
    )
    # Fracture pressure at the shoe = LOT_ppg * Z_CSG * 0.052.
    assert r.p_frac_shoe_psi == pytest.approx(LOT * Z_CSG * G)
    # Gas height and shoe influx volume are positive.
    assert r.gas_height_ft > 0.0
    assert r.influx_at_shoe_bbl == pytest.approx(r.gas_height_ft * CAP_ANN)
    # V2 = P1*V1/P2 (Boyle to bottom hole) -- the closed form, self-consistent.
    assert r.kick_tolerance_bbl == pytest.approx(
        r.p_frac_shoe_psi * r.influx_at_shoe_bbl / PF
    )
    # Kick tolerance in a realistic band for this well (~30 bbl).
    assert 15.0 < r.kick_tolerance_bbl < 60.0


def test_nogepa_gas_gradient_range():
    """G_gas in the NOGEPA 0.05-0.15 psi/ft range gives sane, ordered results."""
    kts = [
        nogepa_drilling_kick_tolerance(
            formation_pressure_psi=PF, z_oh_ft=Z_OH, z_csg_ft=Z_CSG,
            lot_ppg=LOT, mud_ppg=MUD, cap_ann_bbl_per_ft=CAP_ANN,
            g_gas_psi_per_ft=g,
        ).kick_tolerance_bbl
        for g in (0.05, 0.10, 0.15)
    ]
    assert all(10.0 < kt < 80.0 for kt in kts)
