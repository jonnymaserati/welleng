"""SPE/IADC-140113-MS (Santos, Catak, Valluri 2011) Well-A validation.

welleng's single-bubble tolerable-influx `capacity` is algebraically Santos's V1
(the influx-volume-on-bottom: a single gas bubble Boyle-expanded from bottom to the
drill-pipe annulus at the shoe-fracture limit). This locks the reproduction of the
paper's fully-worked Well-A @4500 ft.

Santos's base case is "constant temperature, constant density, no compressibility
(Z=1)", so Z_s=Z_td=1.0 and T_s=T_td are the correct comparison (not welleng's
Hall-Yarborough default). Influx density rho_k = 1.9 ppg and kick intensity 0.5 ppg
are the paper's shared assumptions (Table 1 text).

Safety margin: the paper's printed values are self-consistent at 235 psi. The
conventional margin is choke-operator-error 100 + choke-line-friction 100 = 200 psi;
the extra ~35 psi is the ANNULAR-FRICTION component Santos's text notes. Kick
tolerance is quasi-static, so welleng does not COMPUTE this annular loss -- it is
supplied as a static input via P_apl, exactly as any hydraulics-derived annular loss
would be. Annular friction is conventionally taken as zero (the margin is a
drill-pipe-side well-killing margin), so welleng's default excludes it; the term is
supplied here only to reproduce Santos's method completely.

welleng vs Santos differ by exactly one refinement: welleng carries atmospheric
pressure in Boyle's law (capacity = A*(B-Pp)/(Pp+P_atm)) where Santos's simplified
Eq 3 omits it (V1 = A*(B-Pp)/Pp) -- a <0.5% effect at these depths.
"""
import pytest

from welleng.kick_tolerance import KickInputs, drill_kick
from welleng.kick_tolerance.core import constant_A, constant_B, P_ATM_PSI

# SPE/IADC-140113 Table 1, Well A + shared assumptions.
V_DPA_A = (14.5 ** 2 - 5.5 ** 2) / 1029.4          # 14.5" bit, 5.5" DP -> 0.1749 bbl/ft
SM_CONVENTIONAL = 200.0                             # choke error 100 + choke line 100
SM_FULL = 235.0                                     # + ~35 psi annular friction (paper)


def _well_a(p_apl):
    return KickInputs(
        rho_mud=11.5, PP=12.0, kick_intensity=0.0,     # PP = MW + 0.5 kick intensity
        P_lot=15.0, P_apl=p_apl, D_td=4500.0, D_lot=3000.0,
        T_s=150.0, T_td=150.0, V_dpa=V_DPA_A,
        Z_s=1.0, Z_td=1.0, rho_gas_s=1.9,              # Santos: Z=1, rho_k=1.9 ppg
    )


def test_santos_well_a_reproduced_with_full_margin():
    """With Santos's full 235-psi margin (incl. the annular-friction term), welleng's
    capacity reproduces Santos's printed Well-A V1 (50.9 bbl) to <0.5%."""
    v1 = drill_kick(_well_a(SM_FULL)).capacity
    assert v1 == pytest.approx(50.9, rel=0.005)


def test_santos_annular_friction_is_the_reproduction_step():
    """The conventional (annular-friction = 0) margin gives ~61 bbl -- ~18% above the
    paper's printed 50.9. This is the documented gap: annular friction is
    conventionally zero (welleng's default posture), and Santos's printed values carry
    a non-zero annular component. Not an implementation error."""
    v_conventional = drill_kick(_well_a(SM_CONVENTIONAL)).capacity
    v_full = drill_kick(_well_a(SM_FULL)).capacity
    assert v_conventional == pytest.approx(60.9, abs=0.5)   # ~18% above 50.9
    assert v_full < v_conventional                          # annular friction lowers KT


def test_welleng_capacity_is_santos_V1_with_P_atm_refinement():
    """welleng.capacity == Santos V1 up to the atmospheric-pressure term:
    welleng = A*(B-Pp)/(Pp+P_atm); Santos V1 = A*(B-Pp)/Pp. Same model, welleng more
    rigorous by the P_atm carry (~0.5% here)."""
    inp = _well_a(SM_FULL)
    A, B = constant_A(inp), constant_B(inp)
    P_pp = drill_kick(inp).P_td                            # A-1 bottom-hole pressure
    welleng_v1 = drill_kick(inp).capacity
    santos_v1 = A * (B - P_pp) / P_pp                      # Santos Eq 3 (no P_atm)
    welleng_formula = A * (B - P_pp) / (P_pp + P_ATM_PSI)
    assert welleng_v1 == pytest.approx(welleng_formula, rel=1e-9)
    assert welleng_v1 == pytest.approx(santos_v1, rel=0.006)   # differ only by P_atm
    assert welleng_v1 < santos_v1                             # P_atm in denominator
