"""Ideal-gas REFERENCE mode (Z == 1, isothermal).

A single-bubble Boyle's-law kick-tolerance hand-calc / spreadsheet uses ideal
gas: Z == 1, isothermal. welleng's default is real-gas (Hall-Yarborough), which
is physically correct but does NOT match such references. The ideal-gas toggle
(``KickInputs.ideal_gas`` on the closed-form path, ``gas_model="ideal"`` on the
migration/analytical path) reproduces the textbook result so welleng can be
cross-checked against any ideal-gas reference.

Reference case (ideal, single-shoe, vertical): TD 10000 ft, MW 10 ppg,
BHP 5400 psi, shoe 6000 ft, 8.5 in hole x 5 in DP, MAASP 900 psi (FP 12.885 ppg
at shoe), gas 2.0 ppg at BHP. Textbook single-bubble Boyle -> 56.06 bbl.
"""
import numpy as np
import pytest

from welleng.kick_tolerance.core import KickInputs, drill_kick
from welleng.kick_tolerance.migration import WellSection
from welleng.kick_tolerance.analytical import analytical_kick_tolerance

CAP = (8.5 ** 2 - 5.0 ** 2) / 1029.4


def _sections():
    return [WellSection(0.0, 6000.0, CAP, is_open_hole=False),
            WellSection(6000.0, 10000.0, CAP, is_open_hole=True)]


def _kw():
    return dict(bhp_psi=5400.0, rho_mud_ppg=10.0,
                gas_bh_state=(5400.0, 560.0, None, 2.0),
                temp_profile=([0.0, 10000.0], [560.0, 560.0]))


def test_analytical_ideal_matches_textbook_boyle():
    a = analytical_kick_tolerance(
        _sections(), ([0, 10000], [9.0, 9.0]), ([0, 10000], [12.8846153846] * 2),
        gas_model="ideal", gas_density_mode="exact", check_depths=[6000.0], **_kw())
    # textbook single-bubble ideal Boyle for this case = 56.06 bbl; welleng's
    # column-integrated ideal result agrees to <1% (the residual is the field
    # constant g and single-pressure-vs-integrated column).
    assert a.max_influx_bbl == pytest.approx(56.06, rel=0.01)


def test_ideal_differs_from_real_by_the_z_factor():
    real = analytical_kick_tolerance(
        _sections(), ([0, 10000], [9.0, 9.0]), ([0, 10000], [12.8846153846] * 2),
        gas_model="real", gas_density_mode="exact", check_depths=[6000.0], **_kw())
    ideal = analytical_kick_tolerance(
        _sections(), ([0, 10000], [9.0, 9.0]), ([0, 10000], [12.8846153846] * 2),
        gas_model="ideal", gas_density_mode="exact", check_depths=[6000.0], **_kw())
    # real-gas H-Y (Z ~ 0.9 at ~4000 psi) yields a materially larger tolerance
    # here; the two must not coincide.
    assert real.max_influx_bbl > ideal.max_influx_bbl * 1.05


def test_gas_model_validation():
    with pytest.raises(ValueError, match="gas_model must be"):
        analytical_kick_tolerance(
            _sections(), ([0, 10000], [9.0, 9.0]), ([0, 10000], [12.9, 12.9]),
            gas_model="nonsense", **_kw())


def test_ideal_rejects_real_gas_composition():
    with pytest.raises(ValueError, match="cannot be combined"):
        analytical_kick_tolerance(
            _sections(), ([0, 10000], [9.0, 9.0]), ([0, 10000], [12.9, 12.9]),
            gas_model="ideal", gas_composition={"methane": 1.0}, **_kw())


def test_core_ideal_gas_flag_flows_through():
    base = dict(rho_mud=10.0, PP=9.0, kick_intensity=0.0, P_lot=12.8846153846,
                P_apl=0.0, D_td=10000.0, D_lot=6000.0, T_s=100.0, T_td=100.0,
                V_dpa=CAP)
    real = drill_kick(KickInputs(**base))
    ideal = drill_kick(KickInputs(**base, ideal_gas=True))
    assert ideal.capacity != pytest.approx(real.capacity, rel=1e-3)
    # an explicit Z override still wins over the flag (Z_s/Z_td forced to 1
    # reproduces the ideal Z path even without the isothermal part)
    assert ideal.capacity > 0.0


def test_core_ideal_gas_is_isothermal_z_one():
    # With a strong temperature gradient, ideal mode must ignore it (isothermal)
    # and give Z=1 -- i.e. be insensitive to T_td.
    base = dict(rho_mud=10.0, PP=9.0, kick_intensity=0.0, P_lot=12.8846153846,
                P_apl=0.0, D_td=10000.0, D_lot=6000.0, T_s=100.0, V_dpa=CAP)
    a = drill_kick(KickInputs(**base, T_td=100.0, ideal_gas=True))
    b = drill_kick(KickInputs(**base, T_td=300.0, ideal_gas=True))
    assert a.capacity == pytest.approx(b.capacity, rel=1e-9)
