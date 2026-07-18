"""Tests for the batch / sweep kick-tolerance entry points (Phase 1)."""
import numpy as np
import pytest

from welleng.kick_tolerance import (
    KickInputs, drill_kick, analytical_kick_tolerance, WellSection,
    solve_batch, batch_analytical_kick_tolerance, sweep_analytical_kick_tolerance,
    BatchCaseResult,
)
from welleng.kick_tolerance.geometry import annular_capacity

GAS = (None, 302.0 + 460.0, None, None)
COMMON = dict(bhp_psi=6402.0, rho_mud_ppg=12.0, gas_bh_state=GAS)
PP = (np.array([0.0, 10500.0]), np.array([10.5, 11.0]))
FP_UNIFORM = (np.array([0.0, 10500.0]), np.array([14.0, 14.0]))
TWO_SECTION = [WellSection(0.0, 6500.0, 0.066, False),
               WellSection(6500.0, 10500.0, 0.046, True)]


def _inputs(pp: float) -> KickInputs:
    return KickInputs(
        rho_mud=11.9, PP=pp, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
        D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0,
        V_dpa=annular_capacity(6.125, 4.0), kt_threshold=25.0,
    )


# --- closed-form batch -------------------------------------------------------
def test_solve_batch_matches_single_calls_in_order():
    cases = [_inputs(p) for p in (11.0, 11.5, 12.0)]
    out = solve_batch(cases, kind="drill")
    assert [r.index for r in out] == [0, 1, 2]
    assert all(r.ok for r in out)
    for i, c in enumerate(cases):
        assert out[i].result == drill_kick(c)      # bit-identical to the single call


def test_solve_batch_isolates_errors():
    bad = "not-a-KickInputs"                          # drill_kick(bad) raises -> isolated
    out = solve_batch([_inputs(11.5), bad, _inputs(12.0)], kind="drill")
    assert out[0].ok and out[2].ok
    assert not out[1].ok and out[1].result is None and out[1].error


def test_solve_batch_bad_kind():
    with pytest.raises(ValueError):
        solve_batch([_inputs(11.5)], kind="nope")


# --- analytical batch --------------------------------------------------------
def _acase(fp_top: float) -> dict:
    fp = (np.array([0.0, 10500.0]), np.array([fp_top, fp_top]))
    return dict(sections=TWO_SECTION, pp=PP, fp=fp,
                gas_density_mode="conservative", **COMMON)


def test_batch_analytical_matches_single_calls():
    cases = [_acase(fp) for fp in (13.5, 14.0, 14.5)]
    out = batch_analytical_kick_tolerance(cases)
    assert [r.index for r in out] == [0, 1, 2]
    assert all(r.ok for r in out)
    for i, c in enumerate(cases):
        single = analytical_kick_tolerance(**c)
        assert out[i].result.max_influx_bbl == single.max_influx_bbl   # deterministic


def test_batch_analytical_isolates_errors():
    good = _acase(14.0)
    bad = dict(good)
    bad.pop("bhp_psi")                                # missing required kwarg -> TypeError
    out = batch_analytical_kick_tolerance([good, bad])
    assert out[0].ok
    assert not out[1].ok and "TypeError" in out[1].error


# --- sweep -------------------------------------------------------------------
def test_sweep_matches_manual_batch():
    base = _acase(14.0)
    vals = [6200.0, 6402.0, 6600.0]
    swept = sweep_analytical_kick_tolerance(base, "bhp_psi", vals)
    assert [r.index for r in swept] == [0, 1, 2]
    for i, v in enumerate(vals):
        single = analytical_kick_tolerance(**{**base, "bhp_psi": v})
        assert swept[i].result.max_influx_bbl == single.max_influx_bbl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
