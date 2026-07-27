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


# --- the REDUCTION that anchors the closed form ----------------------------- #

def test_bubble_state_closed_form_reduces_to_nogepa_exactly():
    """welleng's default closed form IS NOGEPA-50 Section 3.2 under NOGEPA's
    assumptions. This is the closed form's external anchor.

    NOGEPA ignores T and Z (Z = 1, isothermal) and uses a CONSTANT gas gradient.
    Match those and the two are algebraically identical -- because both build the
    influx column the same way:

        P_f = p_frac + g_mud*(z_oh - z_csg - H) + g_gas*H

    i.e. BHP pinned, MUD beneath the bubble, the bubble's own weight across it,
    its top at the shoe fracture limit.

    WHY THIS TEST EXISTS. Before it, every paper reproduction in the suite
    exercised the closed form against SPE-208788's printed TABLE -- whose gas
    properties imply a whole-open-hole gas-gradient pressure that the paper's own
    A-2 simplification (1) excludes. Matching that table therefore CONCEALED a
    1290 psi error in where the gas was evaluated. NOGEPA is an independent,
    mandated formula and it cannot be satisfied by the wrong construction.
    See docs/dev/KICK_CLOSED_FORM_AUDIT.md.
    """
    from welleng.kick_tolerance.nogepa import NOGEPA_G

    cap = (6.125 ** 2 - 4.0 ** 2) / 1029.4
    z_csg, z_oh, lot, mud = 6500.0, 10500.0, 16.0, 11.9
    g_gas = 0.10                                   # NOGEPA range 0.05-0.15
    p_f = 12.6 * NOGEPA_G * z_oh                   # max-credible PP at TD

    nog = nogepa_drilling_kick_tolerance(
        formation_pressure_psi=p_f, z_oh_ft=z_oh, z_csg_ft=z_csg, lot_ppg=lot,
        mud_ppg=mud, cap_ann_bbl_per_ft=cap, g_gas_psi_per_ft=g_gas,
    )

    # the same construction, written out -- NOT calling welleng, so this is an
    # independent statement of the algebra rather than a tautology
    g_mud = mud * NOGEPA_G
    p_frac = lot * z_csg * NOGEPA_G
    h = (p_f - (z_oh - z_csg) * g_mud - p_frac) / (g_gas - g_mud)
    v2 = (h * cap) * (p_frac / p_f)                # Boyle, Z=1, isothermal

    assert h == pytest.approx(nog.gas_height_ft, rel=1e-12)
    assert v2 == pytest.approx(nog.kick_tolerance_bbl, rel=1e-12)


def test_the_old_gas_pressure_basis_could_NOT_satisfy_nogepa():
    """The regression guard: the superseded basis misses NOGEPA by ~6%.

    `column-mean-2026` (and `spe-208788`) evaluate the influx after a GAS gradient
    over the whole open hole. Pinned so that nobody restores that basis and
    believes the paper-table tests still cover them -- they do, and that is the
    problem.
    """
    from welleng.kick_tolerance.core import (
        KickInputs, annular_capacity_dpa, drill_kick,
    )

    cap = annular_capacity_dpa(6.125, 4.0)
    kw = dict(rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0,
              P_apl=210.0, D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0,
              V_dpa=cap)
    correct = drill_kick(KickInputs(model_revision="bubble-state", **kw)).capacity
    for superseded in ("spe-208788", "column-mean-2026"):
        old = drill_kick(KickInputs(model_revision=superseded, **kw)).capacity
        assert old < correct, f"{superseded} should UNDER-report vs bubble-state"
        assert 0.02 < (correct / old - 1) < 0.12, (
            f"{superseded}: gap to the corrected basis is "
            f"{100 * (correct / old - 1):.2f}%, outside the audited 2-12% band"
        )
