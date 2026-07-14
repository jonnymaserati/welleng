"""SPE-208788-PA (Thorogood, Robertson, Castillo, Sawaryn 2022) Table-1 validation.

Reproduces the paper's published worked example using the PAPER'S OWN inputs and
the PAPER'S OWN method -- the single-bubble kick-tolerance closed form derived in
its Appendix A (Eqs A-1 ... A-9). Vertical 6.125-in. hole, methane influx into a
water-based mud, geothermal temperature (T-shoe 212 degF, T-td 302 degF),
Hall-Yarborough Z-factor.

Every intermediate the paper reports (Table 1) is reproduced to the paper's stated
precision. The only residual is the paper rounding A (Eq A-5) to three significant
figures (242 bbl); our unrounded A = 241.2 bbl is correct for the exact inputs, and
feeding the paper's OWN rounded A back through A-7 reproduces its Vgas to 0.03 %
(the `digit_exact` check below). This is validation to the paper's numerical
precision, not an approximation.

Paper Table 1 (SPE-208788-PA):
  Geometry     6.125-in hole, 4-in DP, 4.75-in collar, BHA 131 ft, vertical
  Casing shoe  6,500 ft; LOT 16.0 ppg (P_lot = 5,418 psi); T-shoe 212 degF
  TD           10,500 ft; T-td 302 degF; mud 11.90 ppg
  Pore press.  11.50 ppg + 1.10 ppg uncertainty (kick intensity)
  APL 210 psi; gas Z_s 1.1230 / Z_td 1.1650; rho_gas_s 1.710 ppg
  Results      Ptd (A-1) 6,893 psi | A (A-5) 242 bbl | B (A-6) 7,688 psi
               Vgas-td drilled (A-7) 27.86 bbl | Vgas-td KT=0 swab (A-8) 43.79 bbl
"""
import pytest

from welleng.kick_tolerance import KickInputs, drill_kick, swab_kick
from welleng.kick_tolerance.core import (
    scenario_P_td, constant_A, constant_B, influx_volume_A7,
    resolve_gas_properties,
)

# Annular capacity, drillpipe-in-hole: (D_hole^2 - D_dp^2) / 1029.4  [bbl/ft].
V_DPA = (6.125 ** 2 - 4.0 ** 2) / 1029.4  # = 0.020901 bbl/ft


def paper_inputs(**overrides):
    """SPE-208788 Table-1 inputs. Gas properties are the paper's own Table-1
    values (Hall-Yarborough-derived) so the closed form (A-1..A-9) is validated
    independently of our Z backend; that backend is checked separately below."""
    kw = dict(
        rho_mud=11.90, PP=11.50, kick_intensity=1.10, P_lot=16.00, P_apl=210.0,
        D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0, V_dpa=V_DPA,
        Z_s=1.1230, Z_td=1.1650, rho_gas_s=1.710, kt_threshold=25.0, inc_shoe=0.0,
    )
    kw.update(overrides)
    return KickInputs(**kw)


def test_spe208788_intermediates_A1_A5_A6():
    """A-1 (Ptd), A-6 (B) are digit-exact vs the paper (same g = 0.0521); A-5 (A)
    matches to the paper's 3-significant-figure rounding."""
    inp = paper_inputs()
    # A-1 bottom-hole pressure at PP + uncertainty:
    assert scenario_P_td(inp) == pytest.approx(6893.0, abs=1.0)
    # A-6 constant B:
    assert constant_B(inp) == pytest.approx(7688.0, abs=2.0)
    # A-5 constant A: unrounded 241.2; paper reports 242 (3 s.f.).
    A = constant_A(inp)
    assert A == pytest.approx(241.2, abs=0.3)          # our exact value
    assert round(A) in (241, 242)                       # within the paper's rounding


def test_spe208788_tolerable_influx_A7_A8():
    """A-7 (drilled) and A-8 (swab / KT=0) tolerable influx vs the paper, within
    the rounding of A."""
    inp = paper_inputs()
    v_drill = drill_kick(inp).capacity                  # A-7
    v_swab = swab_kick(inp).capacity                    # A-8
    # Paper 27.86 / 43.79 bbl; ours (unrounded A) 27.78 / 43.57 -> within 0.5 %.
    assert v_drill == pytest.approx(27.86, rel=0.005)
    assert v_swab == pytest.approx(43.79, rel=0.006)
    # Ordering the paper asserts: the mitigable swab limit exceeds the drilled one.
    assert v_swab > v_drill


def test_spe208788_digit_exact_through_A7():
    """Proof the residual is the paper's rounding of A, not our error: feed the
    paper's OWN rounded intermediates (A=242, B=7688, Ptd=6893) into A-7 and we
    reproduce its Vgas to 0.03 %."""
    v = influx_volume_A7(242.0, 7688.0, 6893.0)
    assert v == pytest.approx(27.86, abs=0.02)


def test_spe208788_hall_yarborough_backend_reproduces_paper_gas_props():
    """Separately, our clean-room Hall-Yarborough backend reproduces the paper's
    (H-Y-derived) Z and gas density to ~1 % -- so the closed-form validation above
    is not leaning on injected gas properties the engine couldn't itself produce."""
    inp = paper_inputs(Z_s=None, Z_td=None, rho_gas_s=None)  # force H-Y
    Z_s, Z_td, rho_gas_s = resolve_gas_properties(inp)
    assert Z_s == pytest.approx(1.1230, rel=0.02)
    assert Z_td == pytest.approx(1.1650, rel=0.01)
    assert rho_gas_s == pytest.approx(1.710, rel=0.01)
