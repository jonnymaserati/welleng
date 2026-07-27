"""Kick-tolerance model revisions, and the influx gas state.

A kick-tolerance number ends up in a signed-off well programme, so a result
computed under an earlier model must stay reproducible EXACTLY once the model
changes. Hence named, FROZEN revisions that travel with the result.

THE HISTORY, because the revision names only make sense with it:

* ``spe-208788`` reproduces the paper's printed TABLE. Its gas is evaluated after
  a GAS gradient over the whole open hole -- which is what the table's printed
  Z_s = 1.1230 and rho_gas-s = 1.710 imply, and which A-2's simplification (1)
  explicitly excludes ("the gas column does not fill the entire open hole").
* ``column-mean-2026`` moved the influx TEMPERATURE to the column mean (correct,
  and supported by SPE-202426's worked average) but left that pressure basis in
  place. A second-order fix on a first-order error.
* ``bubble-state`` evaluates the gas at the BUBBLE's own state: BHP pinned, MUD
  beneath the bubble setting its bottom pressure, its top at the shoe fracture
  limit, mass-weighted mean density over the column. This reduces ALGEBRAICALLY
  EXACTLY to NOGEPA-50 Section 3.2 under NOGEPA's assumptions -- see
  ``tests/test_nogepa.py``, which is the closed form's external anchor.

Full audit, with the decomposition and the numbers:
``docs/dev/KICK_CLOSED_FORM_AUDIT.md``.
"""

import math

import pytest

from welleng.kick_tolerance.core import (
    DEFAULT_MODEL_REVISION,
    MODEL_REVISIONS,
    G_PSI_PER_PPG_FT,
    P_ATM_PSI,
    KickInputs,
    annular_capacity_dpa,
    constant_A,
    drill_kick,
    influx_column,
    ppg_to_psi,
    scenario_P_td,
    swab_kick,
)

LEGACY = "spe-208788"
SUPERSEDED = ("spe-208788", "column-mean-2026")


def case(**over):
    kw = dict(
        rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
        D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0,
        V_dpa=annular_capacity_dpa(6.125, 4.0), kt_threshold=25.0,
    )
    kw.update(over)
    return KickInputs(**kw)


# --- the mechanism ----------------------------------------------------------

def test_default_is_the_bubble_state_revision():
    assert DEFAULT_MODEL_REVISION == "bubble-state"
    # every published revision stays reachable, permanently
    assert set(SUPERSEDED) <= set(MODEL_REVISIONS)


def test_revision_and_station_travel_with_the_result():
    """A stored calculation must record what produced it, or it cannot be re-run."""
    for rev in MODEL_REVISIONS:
        for fn in (drill_kick, swab_kick):
            res = fn(case(model_revision=rev))
            assert res.model_revision == rev
            assert res.T_influx is not None
            assert res.rho_influx is not None and res.rho_influx > 0.0
            assert res.H_gas is not None and res.H_gas > 0.0


def test_unknown_revision_raises():
    with pytest.raises(ValueError, match="unknown kick-tolerance model_revision"):
        drill_kick(case(model_revision="whatever-2099"))


def test_frozen_revisions_reproduce_the_published_numbers():
    """`spe-208788` must keep reproducing SPE-208788's TABLE, forever."""
    res = drill_kick(case(model_revision=LEGACY))
    assert round(res.P_td) == 6893                            # A-1
    assert round(res.B) == 7688                               # A-6
    assert math.isclose(res.capacity, 27.86, rel_tol=0.013)   # A-7
    assert math.isclose(res.pp_at_threshold, 12.74, rel_tol=0.01)
    assert res.T_influx == pytest.approx(212.0)               # the shoe


# --- the bubble state -------------------------------------------------------

def test_the_bubble_column_CLOSES_on_the_fracture_limit():
    """The construction's self-check, and the reason to trust it.

    Two independent routes to the bubble-BOTTOM pressure must agree: the
    bubble's own weight added to its top pressure, and BHP minus the mud column
    beneath it. A wrong construction does not close. This is what the superseded
    basis fails -- it puts the gas 1290 psi high by running a gas gradient over
    the whole open hole.
    """
    from welleng.kick_tolerance.core import _bubble_state

    inp = case()
    P_td = scenario_P_td(inp)
    _, H, _, rho_bar = _bubble_state(inp, P_td)
    g = G_PSI_PER_PPG_FT
    P_top = ppg_to_psi(inp.P_lot, inp.D_lot) - inp.P_apl + P_ATM_PSI
    from_own_weight = P_top + g * rho_bar * H
    from_mud_beneath = (P_td - g * inp.rho_mud * (inp.D_td - inp.D_lot - H)
                        + P_ATM_PSI)
    assert from_own_weight == pytest.approx(from_mud_beneath, abs=1e-6), (
        f"column does not close: {from_own_weight:.4f} vs "
        f"{from_mud_beneath:.4f} psia"
    )


@pytest.mark.parametrize("superseded", SUPERSEDED)
def test_the_corrected_basis_reports_MORE_than_the_superseded_one(superseded):
    """Direction, stated because it is the safety-relevant fact.

    The superseded basis evaluates the gas too DENSE (a whole-open-hole gas
    gradient puts it 1290 psi high), and under-reports the tolerable influx by
    2-12% depending on the gradient. So the correction INCREASES the reported
    capacity: anyone who acted on a shipped number took LESS influx than they
    could safely tolerate.
    """
    old = drill_kick(case(model_revision=superseded)).capacity
    new = drill_kick(case()).capacity
    assert new > old
    assert 0.02 < (new / old - 1) < 0.12


def test_the_influx_is_evaluated_INSIDE_the_bubble():
    res = drill_kick(case())
    assert 212.0 < res.T_influx < 302.0          # between the mud interfaces
    assert 0.0 < res.H_gas < (10500.0 - 6500.0)  # column is in the open hole
    grad = (302.0 - 212.0) / (10500.0 - 6500.0)
    assert res.T_influx == pytest.approx(212.0 + grad * 0.5 * res.H_gas, rel=1e-6)


def test_influx_column_matches_what_the_result_reports():
    """The public helper and the result must not disagree -- the
    'three sites, three answers' failure mode in miniature."""
    inp = case()
    assert influx_column(inp) == (
        drill_kick(inp).T_influx, drill_kick(inp).H_gas,
        drill_kick(inp).rho_influx,
    )


def test_isothermal_still_moves_because_this_is_NOT_a_temperature_fix():
    """Guard against the old story being read back in.

    The 0.26.0 correction was temperature-only, so `isothermal == exact no-op`
    held and proved that nothing else had moved. `bubble-state` changes the
    PRESSURE basis, so it moves under a zero gradient too -- and asserting that
    keeps anyone from re-deriving the temperature-only framing.
    """
    iso = dict(T_s=212.0, T_td=212.0)
    a = constant_A(case(model_revision="column-mean-2026", **iso))
    b = constant_A(case(**iso))
    assert b != a
    assert b > a                                  # same direction as with a gradient


# --- validation inputs stay the caller's ------------------------------------

def test_injected_gas_properties_take_the_papers_algebra():
    """Injection is the PAPER-REPRODUCTION path and must stay Boyle, not mass
    conservation.

    Santos (SPE-140113) assumes "constant temperature, constant density, no
    compressibility (Z=1)". Expanding a CONSTANT-density influx by mass
    conservation gives no expansion at all, which is not that paper's model. So a
    caller who supplies gas properties gets the algebra those properties belong
    to -- which is how `tests/test_spe140113_santos.py` still reproduces 50.9 bbl.
    """
    inj = dict(Z_s=1.0, Z_td=1.0, rho_gas_s=1.9)
    assert drill_kick(case(**inj)).rho_influx == pytest.approx(1.9)
    # and it is NOT the computed bubble density
    assert drill_kick(case()).rho_influx != pytest.approx(1.9, rel=1e-3)


def test_replace_preserves_the_revision():
    from dataclasses import replace
    inp = case(model_revision=LEGACY)
    assert replace(inp, PP=12.0).model_revision == LEGACY
