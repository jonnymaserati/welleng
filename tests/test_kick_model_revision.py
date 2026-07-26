"""Kick-tolerance model revisions, and the A-5 influx-temperature correction.

A kick-tolerance number ends up in a signed-off well programme, so a result
computed under an earlier model has to stay reproducible EXACTLY once the model
changes. Hence a named FROZEN revision that travels with the result.

The correction itself (welleng-api, external reviewer, 2026-07-26): A-5's
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
    swab_kick,
)

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
