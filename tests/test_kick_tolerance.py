"""Golden test for the clean-room kick-tolerance module.

Fixtures are the PUBLISHED Table-1 numbers from the public SPE paper
SPE-208788-PA. The gas properties (Z_s, Z_td, rho_gas_s) are now COMPUTED by the
clean-room Hall & Yarborough (1973) backend (``gas_z.py``) from pure methane and
the model's own P/T conditions -- NOT injected. ``test_gas_z.py`` validates that
backend against CoolProp and the paper's printed Z values.

Exact assertions (independent of gas properties):
    P_td = 6893 psi   (max-credible PP = PP + kick = 12.6 ppg governs)
    B    = 7688 psi

~1% assertions (kick outputs):
    V_drilled  ~= 27.86 bbl   (computed backend: 27.58, 1.0%)
    V_swab     ~= 43.79 bbl   (computed backend: 43.25, 1.2%)
    PP @ KT=25 ~= 12.74 ppg   (computed backend: 12.72, 0.1%)

Why ~1% and NOT bit-exact
-------------------------
The paper's Table-1 is internally inconsistent. Its printed shoe Z (1.123) sits
above the value at the shoe *fracture* pressure P_lot - P_apl (~1.04); it
corresponds to a near-bottom-hole influx-gas pressure. The clean-room backend
evaluates the influx gas at the physically-defensible shoe pressure (bottom-hole
pressure reduced by the static influx-gas column), giving Z_s ~= 1.134,
rho_gas_s ~= 1.72, and A ~= 239.4. The kick volumes then land within ~1-1.3% of
the published cells. That residual is the PAPER'S OWN Table-1 inconsistency; we
do NOT tune any constant/pressure to force a closer match.
"""

import math
from dataclasses import replace

import pytest

from welleng.kick_tolerance.core import (
    KickInputs,
    annular_capacity_dpa,
    constant_A,
    constant_B,
    drill_kick,
    influx_volume_A7,
    ppg_to_psi,
    resolve_gas_properties,
    swab_kick,
)

# Documented residual from the paper's Table-1 gas-property inconsistency.
KICK_VOLUME_REL_TOL = 0.013


def make_inputs() -> KickInputs:
    # Table-1 published fixtures (public SPE-208788-PA). Gas properties are
    # COMPUTED by the Hall-Yarborough backend (Z_s / Z_td / rho_gas_s = None).
    return KickInputs(
        rho_mud=11.9,
        PP=11.5,
        kick_intensity=1.1,
        P_lot=16.0,
        P_apl=210.0,
        D_td=10500.0,
        D_lot=6500.0,
        T_s=212.0,
        T_td=302.0,
        # A-3 simplification: drillpipe (4") in the 6.125" hole at the shoe.
        V_dpa=annular_capacity_dpa(6.125, 4.0),
        # Gas properties left as None -> computed from pure methane.
        kt_threshold=25.0,
    )


def test_gas_properties_computed():
    """Backend reproduces the paper's printed Table-1 gas properties (~1%)."""
    Z_s, Z_td, rho_gas_s = resolve_gas_properties(make_inputs())
    assert math.isclose(Z_td, 1.1650, rel_tol=0.005)     # paper 1.1650
    assert math.isclose(Z_s, 1.1230, rel_tol=0.011)      # paper 1.1230
    assert math.isclose(rho_gas_s, 1.710, rel_tol=0.011)  # paper 1.710


def test_P_td_exact():
    """A-1: P_td governed by max-credible PP (12.6 ppg) -> 6893 psi."""
    res = drill_kick(make_inputs())
    assert round(res.P_td) == 6893


def test_B_exact():
    """A-6: B = 7688 psi."""
    res = drill_kick(make_inputs())
    assert round(res.B) == 7688


def test_A_constant_paper_band():
    """A-5 with the computed backend lands at ~239.4 (documented Table-1 band).

    The published A cell is 242; back-solving individual Table-1 cells gives
    ~241.2/242.1/242.5. The computed-gas forward evaluation is ~239.4 -- within
    the paper's own ~1% spread.
    """
    A = constant_A(make_inputs())
    assert 238.0 < A < 243.0
    assert math.isclose(A, 239.4, rel_tol=3e-3)


def test_V_drilled_within_1pct():
    """A-7 tolerable influx at max-credible PP ~= 27.86 bbl (~1%)."""
    res = drill_kick(make_inputs())
    assert math.isclose(res.capacity, 27.86, rel_tol=KICK_VOLUME_REL_TOL)


def test_swab_reproduces_spe208788_shared_A():
    """Reproduce SPE-208788-PA Table-1 swab figure (43.79 bbl) under ITS OWN
    convention: one A constant shared across A-7/A-8, i.e. swab gas stationed at
    the drill max-credible pressure (PP+KI). Validation = reproduce the source
    result with the source method.

    Our shipped model (``swab_kick``) does NOT do this -- it stations the swab
    gas at mud hydrostatic (KI-independent), a conscious divergence documented
    in ``swab_kick`` per Nassab SPE-202426-PA. See
    ``test_swab_kick_is_kick_intensity_invariant``.
    """
    inp = make_inputs()
    A_shared = constant_A(inp)                       # scenario-stationed (208788 shared A)
    B = constant_B(inp)
    P_td_swab = ppg_to_psi(inp.rho_mud, inp.D_td)    # A-8 substitution
    v_swab_208788 = influx_volume_A7(A_shared, B, P_td_swab)
    assert math.isclose(v_swab_208788, 43.79, rel_tol=KICK_VOLUME_REL_TOL)


def test_swab_model_diverges_from_208788_consciously():
    """Our swab_kick (mud-stationed gas) sits BELOW the 208788 shared-A figure
    for this underbalanced-KI fixture -- conservative direction, ~1.5%."""
    inp = make_inputs()
    assert inp.PP + inp.kick_intensity > inp.rho_mud   # underbalanced-KI: the two diverge
    ours = swab_kick(inp).capacity
    assert ours < 43.79
    assert math.isclose(ours, 43.15, rel_tol=0.01)


def test_pp_at_threshold_within_1pct():
    """A-9 pore pressure at KT = 25 bbl ~= 12.74 ppg (~1%)."""
    res = drill_kick(make_inputs())
    assert math.isclose(res.pp_at_threshold, 12.74, rel_tol=0.01)


def test_swab_never_overrides_drill_separate_results():
    """Drill and swab are independent first-class results (never blind-min'd)."""
    inp = make_inputs()
    d = drill_kick(inp)
    s = swab_kick(inp)
    assert d.case == "drill" and s.case == "swab"
    # Different bottom-hole pressures -> genuinely separate cases.
    assert d.P_td != s.P_td
    # B is a gas-free constant (shared); A differs because the influx gas is
    # stationed at each case's own P_td (drill: PP+KI scenario; swab: mud
    # hydrostatic) -- make_inputs is underbalanced (PP+KI 12.6 > mud 11.9).
    assert math.isclose(d.B, s.B)
    assert not math.isclose(d.A, s.A)
    assert d.capacity != s.capacity


def test_swab_kick_is_kick_intensity_invariant():
    """swab_kick must not depend on kick_intensity, INCLUDING via gas stationing.

    Nassab SPE-202426-PA Eqs 8-9: the swab bottom-hole pressure is the mud
    hydrostatic, independent of PP/KI. Regression for the leak where swab gas
    Z/rho were stationed at scenario_P_td (PP+KI) -> KI crept in anti-
    conservatively. make_inputs is underbalanced so the leak, if present, shows.
    """
    base = make_inputs()
    caps = [swab_kick(replace(base, kick_intensity=ki)).capacity
            for ki in (0.0, 1.1, 2.5, 5.0)]
    assert max(caps) - min(caps) < 1e-9
    # sanity: drill DOES respond to KI (the invariance is swab-specific)
    drills = [drill_kick(replace(base, kick_intensity=ki)).capacity
              for ki in (0.0, 5.0)]
    assert abs(drills[0] - drills[1]) > 1.0


def test_deviated_scales_A_by_inverse_cos_inc_shoe():
    """Deviated form (Nassab et al., SPE-202426-PA): the gas column's vertical
    height H_gas becomes an along-hole length L_gas = H_gas / cos(inc_shoe), so
    A scales by 1 / cos(inc_shoe). inc_shoe = 0 recovers the vertical Table-1
    form; a deviated well tolerates a LARGER influx volume for the same
    fracture-limiting vertical gas column."""
    A_vert = constant_A(make_inputs())               # inc_shoe = 0.0 (vertical)
    for inc in (15.0, 30.0, 45.0, 60.0):
        A_dev = constant_A(replace(make_inputs(), inc_shoe=inc))
        assert math.isclose(
            A_dev, A_vert / math.cos(math.radians(inc)), rel_tol=1e-12
        )
    # inc_shoe = 0 is exactly the vertical constant (regression guard)
    assert math.isclose(constant_A(replace(make_inputs(), inc_shoe=0.0)), A_vert)
    # deviated -> larger tolerable influx capacity than vertical
    dev = drill_kick(replace(make_inputs(), inc_shoe=45.0)).capacity
    vert = drill_kick(make_inputs()).capacity
    assert dev > vert


def test_from_survey_reads_tvd_and_inclination():
    """KickInputs.from_survey pulls D_lot/D_td (TVD) and inc_shoe off a welleng
    Survey via min-curvature interpolation, so callers don't hand-transcribe them."""
    import numpy as np
    import welleng as we

    survey = we.survey.Survey(
        md=np.array([0.0, 1000.0, 2000.0, 3000.0]),
        inc=np.array([0.0, 0.0, 30.0, 60.0]),
        azi=np.array([0.0, 0.0, 45.0, 45.0]),
    )
    params = dict(
        rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
        T_s=212.0, T_td=302.0, V_dpa=annular_capacity_dpa(6.125, 4.0),
    )
    inp = KickInputs.from_survey(survey, shoe_md=2500.0, td_md=3000.0, **params)
    # deviated section (inc 30@2000 -> 60@3000): inclination at shoe ~45 deg
    assert math.isclose(inp.inc_shoe, 45.0, abs_tol=1.0)
    # TVD is shallower than MD for the deviated stations
    assert inp.D_lot < 2500.0 and inp.D_td < 3000.0
    # produces a usable inputs object
    assert drill_kick(inp).case == "drill"


if __name__ == "__main__":
    inp = make_inputs()
    d = drill_kick(inp)
    s = swab_kick(inp)
    print(f"A          = {d.A:.4f} bbl")
    print(f"P_td(drill)= {d.P_td:.2f} psi   B = {d.B:.2f} psi")
    print(f"V_drilled  = {d.capacity:.4f} bbl")
    print(f"P_td(swab) = {s.P_td:.2f} psi")
    print(f"V_swab     = {s.capacity:.4f} bbl")
    print(f"PP @ KT=25 = {d.pp_at_threshold:.4f} ppg")
