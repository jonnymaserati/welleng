"""The influx bubble's own state.

`_bubble_state` replaced `_shoe_gas_pressure`, which ran a GAS gradient over the
whole open hole -- a construction A-2's simplification (1) explicitly excludes,
since the gas column does not fill the entire open hole. Mud sits beneath the
bubble and sets its bottom pressure.

Its docstring has claimed these assertions live here since it was written. They
did not. They do now.
"""
import pytest

import welleng.kick_tolerance.core as core
from welleng.kick_tolerance.core import (
    G_PSI_PER_PPG_FT,
    KickInputs,
    annular_capacity_dpa,
    fahrenheit_to_rankine,
    ppg_to_psi,
    scenario_P_td,
)

def p_top(inp):
    return ppg_to_psi(inp.P_lot, inp.D_lot) - inp.P_apl + 14.7


CASE = dict(rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
            D_td=10500.0, D_lot=6500.0, V_dpa=annular_capacity_dpa(6.125, 4.0))


def test_the_two_routes_to_the_bubble_bottom_pressure_agree():
    """The self-check the construction rests on, and the reason it is right.

    The pressure at the bottom of the bubble can be reached two independent
    ways: down from its top through the bubble's OWN weight, or up from bottom
    hole through the MUD beneath it. A wrong construction does not close.
    """
    inp = KickInputs(T_s=212.0, T_td=302.0, **CASE)
    p_td = scenario_P_td(inp)
    _, h, _, rho_bar = core._bubble_state(inp, p_td)

    p_top = ppg_to_psi(inp.P_lot, inp.D_lot) - inp.P_apl + 14.7
    from_the_top = p_top + G_PSI_PER_PPG_FT * rho_bar * h
    from_below = p_td + 14.7 - G_PSI_PER_PPG_FT * inp.rho_mud * (
        inp.D_td - inp.D_lot - h)

    assert from_the_top == pytest.approx(from_below, abs=0.05)




def test_the_modelled_gas_column_is_convectively_unstable():
    """A finding, pinned so it is not rediscovered as a bug.

    Across the bubble the pressure rises only ~2% while the imposed geothermal
    gradient lifts T ~5%, and rho ~ P/(Z.T), so the MODEL makes the gas heavier
    at the top of its own column than at the bottom. That is not an equilibrium:
    the imposed gradient is roughly an order of magnitude superadiabatic (g/cp
    for methane ~0.002-0.003 degF/ft against 0.0225 here), so a real bubble would
    convect and mix toward near-isothermal.

    The consequence for the code is narrow and worth stating: do NOT mass-weight
    the influx temperature. Weighting by this density profile weights by an
    artefact of solving a static column under an imposed profile. The reported
    temperature is the mid-height value of the imposed linear profile, which is
    the honest summary of what was actually assumed.
    """
    inp = KickInputs(T_s=212.0, T_td=302.0, **CASE)
    _, h, _, _ = core._bubble_state(inp, scenario_P_td(inp))
    grad = (inp.T_td - inp.T_s) / (inp.D_td - inp.D_lot)

    top = core._influx_density(inp, p_top(inp), fahrenheit_to_rankine(inp.T_s))
    bottom = core._influx_density(
        inp, p_top(inp) + G_PSI_PER_PPG_FT * 1.45 * h,
        fahrenheit_to_rankine(inp.T_s + grad * h),
    )

    assert top > bottom, "expected the modelled column to be density-inverted"
    assert grad > 0.01, "geothermal gradient is far above any adiabat"


def test_the_influx_temperature_is_the_mid_height_value():
    """Pinned: the reported temperature is the mid-height value of the imposed
    linear profile, inside the two mud-interface temperatures."""
    inp = KickInputs(T_s=212.0, T_td=302.0, **CASE)
    t_bar, h, _, _ = core._bubble_state(inp, scenario_P_td(inp))
    grad = (inp.T_td - inp.T_s) / (inp.D_td - inp.D_lot)

    assert t_bar == pytest.approx(
        fahrenheit_to_rankine(inp.T_s + grad * 0.5 * h), rel=1e-9)
    assert fahrenheit_to_rankine(inp.T_s) < t_bar < fahrenheit_to_rankine(
        inp.T_s + grad * h)


def test_the_bulk_density_is_pinned_by_the_endpoint_pressures():
    """Why the convective instability does not reach the kick tolerance.

    Both faces of the bubble are fixed by the pressure balance -- its top by the
    shoe fracture limit, its bottom by the mud beneath it -- so

        INT rho dh = dP / g

    is determined, and the bulk mean density rho_bar = dP/(g*H) with it. However
    the column redistributes internally, the mean density is the same, and so is
    the buoyancy term that drives the A constant.

    The internal ARRANGEMENT of the mass never enters the answer. That is why the
    density profile being convectively unstable
    (:func:`test_the_modelled_gas_column_is_convectively_unstable`) is a caution
    about the reported internal state and not about the kick tolerance.
    """
    for t_s, t_td in ((212.0, 302.0), (100.0, 300.0), (212.0, 212.0), (60.0, 350.0)):
        inp = KickInputs(T_s=t_s, T_td=t_td, **CASE)
        _, h, _, rho_bar = core._bubble_state(inp, scenario_P_td(inp))

        # the bubble-bottom pressure by the INDEPENDENT route: bottom hole less
        # the mud column beneath the bubble
        p_bottom = scenario_P_td(inp) + 14.7 - G_PSI_PER_PPG_FT * inp.rho_mud * (
            inp.D_td - inp.D_lot - h)
        implied = (p_bottom - p_top(inp)) / (G_PSI_PER_PPG_FT * h)

        assert implied == pytest.approx(rho_bar, rel=1e-9)
