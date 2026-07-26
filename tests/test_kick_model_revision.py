"""Kick-tolerance model revisions, and the A-5 influx-temperature correction.

A kick-tolerance number ends up in a signed-off well programme, so a result
computed under an earlier model has to stay reproducible EXACTLY once the model
changes. Hence a named FROZEN revision that travels with the result.

The correction itself (a batch consumer, external reviewer, 2026-07-26): A-5's
``T_s``/``Z_s``/``rho_gas_s`` describe the INFLUX at the condition A-2 defines
-- gas top at the casing shoe -- so the column hangs BELOW the shoe and the
shoe is its COOL END. Evaluating there makes the gas denser than the column,
shrinking the ``(rho_mud - rho_gas_s)`` deficit in A-5's denominator, inflating
A, and passing straight through A-7 as an OVERSTATED tolerable influx. The
pressure side already integrates the column; only the temperature side did not.
"""

import math
from dataclasses import replace

import pytest

from welleng.kick_tolerance.core import (
    DEFAULT_MODEL_REVISION,
    MODEL_REVISIONS,
    KickInputs,
    annular_capacity_dpa,
    constant_A,
    drill_kick,
    influx_column,
    ppg_to_psi,
    swab_kick,
)
from welleng.kick_tolerance.core import G_PSI_PER_PPG_FT

LEGACY = "spe-208788"


def case(**overrides) -> KickInputs:
    """SPE-208788 Table-1 geometry, gas properties computed."""
    kw = dict(
        rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
        D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0,
        V_dpa=annular_capacity_dpa(6.125, 4.0), kt_threshold=25.0,
    )
    kw.update(overrides)
    return KickInputs(**kw)


# --- the revision mechanism -------------------------------------------------

def test_default_is_the_corrected_revision():
    assert DEFAULT_MODEL_REVISION == "column-mean-2026"
    assert LEGACY in MODEL_REVISIONS
    assert KickInputs.__dataclass_fields__["model_revision"].default == (
        DEFAULT_MODEL_REVISION
    )


def test_revision_travels_with_the_result():
    """A stored calculation must record what produced it, or it cannot be
    re-run verbatim."""
    for rev in MODEL_REVISIONS:
        for fn in (drill_kick, swab_kick):
            res = fn(case(model_revision=rev))
            assert res.model_revision == rev
            assert res.T_influx is not None


def test_unknown_revision_raises():
    with pytest.raises(ValueError, match="unknown kick-tolerance model_revision"):
        drill_kick(case(model_revision="whatever-2099"))


def test_legacy_revision_is_frozen_at_the_published_numbers():
    """The anchor. `spe-208788` must keep reproducing SPE-208788 Table 1."""
    res = drill_kick(case(model_revision=LEGACY))
    assert round(res.P_td) == 6893                      # A-1
    assert round(res.B) == 7688                         # A-6
    assert math.isclose(res.capacity, 27.86, rel_tol=0.013)   # A-7
    assert math.isclose(res.pp_at_threshold, 12.74, rel_tol=0.01)
    assert res.T_influx == pytest.approx(212.0)         # the shoe temperature


# --- the correction ---------------------------------------------------------

def test_correction_is_anti_conservative_in_the_direction_reported():
    """The whole point: the shipped model OVERSTATED the tolerable influx."""
    legacy = drill_kick(case(model_revision=LEGACY)).capacity
    fixed = drill_kick(case()).capacity
    assert fixed < legacy
    overstatement = legacy / fixed - 1.0
    assert 0.02 < overstatement < 0.05                  # 3.1% on this case


def test_influx_is_evaluated_inside_the_column_not_at_its_cool_end():
    res = drill_kick(case())
    assert res.T_influx > 212.0                         # warmer than the shoe
    assert res.T_influx < 302.0                         # cooler than TD
    assert 0.0 < res.H_gas < (10500.0 - 6500.0)         # column is in open hole
    # the mean sits at the column mid-depth on the geothermal gradient
    grad = (302.0 - 212.0) / (10500.0 - 6500.0)
    assert res.T_influx == pytest.approx(212.0 + grad * 0.5 * res.H_gas, rel=1e-6)


def test_isothermal_is_an_exact_no_op():
    """The correction is PURELY the temperature gradient. With no gradient the
    two revisions must agree bit-for-bit -- if they do not, something other
    than the reported defect changed."""
    iso = dict(T_s=212.0, T_td=212.0)
    a_legacy = constant_A(case(model_revision=LEGACY, **iso))
    a_fixed = constant_A(case(**iso))
    assert a_fixed == a_legacy


@pytest.mark.parametrize("t_td", (232.0, 272.0, 302.0, 362.0, 412.0))
def test_correction_grows_monotonically_with_the_gradient(t_td):
    """A steeper gradient means a bigger gap between the shoe and the column,
    so a bigger correction. Monotone, and zero at zero gradient."""
    legacy = drill_kick(case(model_revision=LEGACY, T_td=t_td)).capacity
    fixed = drill_kick(case(T_td=t_td)).capacity
    delta = legacy / fixed - 1.0
    assert delta >= 0.0
    baseline = (
        drill_kick(case(model_revision=LEGACY, T_td=212.0)).capacity
        / drill_kick(case(T_td=212.0)).capacity - 1.0
    )
    assert baseline == pytest.approx(0.0, abs=1e-12)
    if t_td > 212.0:
        assert delta > 0.0


def test_ideal_gas_reference_mode_is_isothermal_and_unaffected():
    """`ideal_gas` is isothermal by definition, so no revision can move it."""
    a_legacy = constant_A(case(model_revision=LEGACY, ideal_gas=True))
    a_fixed = constant_A(case(ideal_gas=True))
    assert a_fixed == a_legacy


def test_injected_gas_properties_still_win():
    """An explicit override is verbatim under either revision -- but the
    denominator's T_s still moves, because that is the correction."""
    inj = dict(Z_s=1.10, Z_td=1.10, rho_gas_s=1.5)
    a_legacy = constant_A(case(model_revision=LEGACY, **inj))
    a_fixed = constant_A(case(**inj))
    assert a_fixed < a_legacy


def test_swab_is_corrected_too():
    """The swab case stations its gas at its own P_td; the temperature
    correction applies there as well."""
    assert swab_kick(case()).capacity < swab_kick(
        case(model_revision=LEGACY)
    ).capacity


def test_replace_preserves_the_revision():
    """`dataclasses.replace` is how consumers build sweeps; a revision must not
    be silently dropped by one."""
    inp = case(model_revision=LEGACY)
    assert replace(inp, PP=12.0).model_revision == LEGACY


# --- the limiting profile must close on the fracture pressure ---------------

def _shoe_pressure_from_column(inp, res):
    """Reconstruct the shoe pressure from the reported influx column."""
    return res.P_td - G_PSI_PER_PPG_FT * (
        res.rho_influx * res.H_gas
        + inp.rho_mud * (inp.D_td - inp.D_lot - res.H_gas)
    )


@pytest.mark.parametrize("revision", sorted(MODEL_REVISIONS))
@pytest.mark.parametrize("fn", (drill_kick, swab_kick))
def test_limiting_profile_sits_exactly_on_the_fracture_pressure(revision, fn):
    """At the tolerable influx the shoe sits ON the binding fracture pressure.

    That is what "limiting" means, and it is an identity in A-2 for ANY influx
    density -- so it must hold to 0.00 psi under both revisions. a batch consumer's
    profile drifted from -2.01 to -14.08 psi across the revisions, which is how
    the missing pieces below were found: core reported the temperature it used
    but not the DENSITY, and reported H only under the new revision, so a
    consumer had to re-derive both and drifted when they moved.
    """
    inp = case(model_revision=revision)
    res = fn(inp)
    P_frac = ppg_to_psi(inp.P_lot, inp.D_lot) - inp.P_apl
    assert _shoe_pressure_from_column(inp, res) == pytest.approx(P_frac, abs=1e-9)


@pytest.mark.parametrize("revision", sorted(MODEL_REVISIONS))
def test_column_is_reported_under_both_revisions(revision):
    """H_gas and rho_influx are what a caller draws the profile from, so
    neither may be None on either revision."""
    res = drill_kick(case(model_revision=revision))
    assert res.rho_influx is not None and 0.0 < res.rho_influx < res.P_td
    assert res.H_gas is not None and 0.0 < res.H_gas < (10500.0 - 6500.0)


def test_influx_column_matches_what_the_result_reports():
    """The public helper and the result must not disagree -- that is the
    'three sites, three answers' failure mode in miniature."""
    inp = case()
    T, H, rho = influx_column(inp)
    res = drill_kick(inp)
    assert (T, H, rho) == (res.T_influx, res.H_gas, res.rho_influx)


def test_rho_influx_honours_an_injected_density():
    """A caller supplying its own gas properties must still get a closing
    profile, so the reported density has to be the INJECTED one."""
    inp = case(rho_gas_s=1.5)
    res = drill_kick(inp)
    assert res.rho_influx == 1.5
    P_frac = ppg_to_psi(inp.P_lot, inp.D_lot) - inp.P_apl
    assert _shoe_pressure_from_column(inp, res) == pytest.approx(P_frac, abs=1e-9)


@pytest.mark.parametrize("revision", sorted(MODEL_REVISIONS))
def test_back_deriving_H_from_capacity_does_NOT_close_and_is_revision_blind(
    revision,
):
    """The trap, pinned so nobody re-discovers it as a regression.

    Expanding `capacity` back to the shoe goes through A-4, whose pressure
    bookkeeping is not algebraically identical to A-5/A-7 as printed. That route
    misses the fracture pressure -- IDENTICALLY under both revisions, which is
    what proves it is the paper's own bookkeeping and not the influx-temperature
    correction. A profile built that way was already off before 0.26.0.
    """
    from welleng.kick_tolerance.core import (
        fahrenheit_to_rankine, resolve_gas_properties, P_ATM_PSI,
    )
    inp = case(model_revision=revision)
    res = drill_kick(inp)
    Z_s, Z_td, _ = resolve_gas_properties(inp, res.P_td)
    P_frac = ppg_to_psi(inp.P_lot, inp.D_lot) - inp.P_apl
    V_s = (res.capacity * res.P_td * fahrenheit_to_rankine(res.T_influx) * Z_s
           / ((P_frac + P_ATM_PSI) * fahrenheit_to_rankine(inp.T_td) * Z_td))
    H_from_V = V_s / inp.V_dpa
    P_shoe = res.P_td - G_PSI_PER_PPG_FT * (
        res.rho_influx * H_from_V
        + inp.rho_mud * (inp.D_td - inp.D_lot - H_from_V)
    )
    assert P_shoe - P_frac == pytest.approx(-3.93, abs=0.05)   # both revisions
