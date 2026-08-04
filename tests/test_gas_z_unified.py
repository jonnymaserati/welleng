"""Tests for the unified classic gas_z() wrapper + Sutton pseudo-criticals."""
import pytest

from welleng.kick_tolerance import (
    gas_z,
    hall_yarborough_z,
    standing_pseudo_criticals,
    sutton_pseudo_criticals,
)
from welleng.kick_tolerance.gas_z import METHANE_PPC_PSIA, METHANE_TPC_RANKINE

P, T = 5400.0, 660.0  # psia, degR (~200 degF)


def test_default_is_methane_hall_yarborough():
    # no gravity/composition -> Tier-0 methane H-Y, exactly
    assert gas_z(P, T) == hall_yarborough_z(P, T)


def test_gas_gravity_uses_standing_by_default():
    g = 0.686
    tpc, ppc = standing_pseudo_criticals(g)
    assert gas_z(P, T, gas_gravity=g) == hall_yarborough_z(P, T, tpc, ppc)


def test_sutton_correlation_selectable_and_differs():
    g = 0.9
    tpc_su, ppc_su = sutton_pseudo_criticals(g)
    z_sutton = gas_z(P, T, gas_gravity=g, pc_correlation="sutton")
    assert z_sutton == hall_yarborough_z(P, T, tpc_su, ppc_su)
    # Sutton and Standing give different pseudo-criticals -> different Z
    assert z_sutton != gas_z(P, T, gas_gravity=g, pc_correlation="standing")


def test_explicit_pseudo_criticals_override_gravity():
    z = gas_z(P, T, gas_gravity=0.9, t_pc_rankine=METHANE_TPC_RANKINE,
              p_pc_psia=METHANE_PPC_PSIA)
    assert z == hall_yarborough_z(P, T)  # explicit methane criticals win


def test_sutton_matches_published_form():
    # Sutton (1985) at g=0.7
    tpc, ppc = sutton_pseudo_criticals(0.7)
    assert tpc == pytest.approx(169.2 + 349.5 * 0.7 - 74.0 * 0.7**2)
    assert ppc == pytest.approx(756.8 - 131.0 * 0.7 - 3.6 * 0.7**2)


def test_bad_method_and_correlation_raise():
    with pytest.raises(ValueError):
        gas_z(P, T, method="nope")
    with pytest.raises(ValueError):
        gas_z(P, T, pc_correlation="nope")


def test_coolprop_path_or_graceful_fallback():
    comp = {"Methane": 0.9, "CO2": 0.1}
    try:
        import CoolProp  # noqa: F401
    except ImportError:
        # CoolProp absent: composition alone must raise, but with a gravity it
        # falls back to Hall-Yarborough rather than failing.
        with pytest.raises(ImportError):
            gas_z(P, T, composition=comp, method="coolprop")
        z = gas_z(P, T, composition=comp, gas_gravity=0.65, method="coolprop")
        assert z == hall_yarborough_z(P, T, *standing_pseudo_criticals(0.65))
    else:
        # CoolProp present: auto-selects the EOS path for a composition
        z = gas_z(P, T, composition=comp)
        assert 0.5 < z < 1.5


def test_coolprop_method_needs_composition():
    with pytest.raises(ValueError):
        gas_z(P, T, method="coolprop")
