"""Shut-in migration: the closure is constant VOLUME, not pinned pressure."""
import warnings

import numpy as np
import pytest

from welleng.kick_tolerance import (
    OBM_COMPRESSIBILITY,
    WBM_COMPRESSIBILITY,
    WellSection,
    shut_in_migration,
)
from welleng.kick_tolerance.migration import ppg_to_psi

TD, SHOE = 10500.0, 6500.0
SECTIONS = [WellSection(0.0, SHOE, 0.066, False),
            WellSection(SHOE, TD, 0.046, True)]
PP = (np.array([0.0, TD]), np.array([12.6, 12.6]))      # underbalanced: pore > mud
FP = (np.array([0.0, TD]), np.array([14.0, 14.0]))
COMMON = dict(
    bhp_initial_psi=ppg_to_psi(12.6, TD), rho_mud_ppg=12.0,
    gas_bh_state=(None, 660.0, None, None), well_mud_bbl=800.0, bottom_tvd=TD,
    geothermal=([0.0, TD], [520.0, 660.0]),
)


def _run(influx, c):
    with warnings.catch_warnings():             # H-Y band at tiny influx
        warnings.simplefilter("ignore")
        return shut_in_migration(SECTIONS, PP, FP, influx_bbl=influx,
                                 compressibility_per_psi=c, **COMMON)


def test_the_rigid_limit_is_degenerate_not_conservative():
    """With no relief the gas density is fixed, so what it imposes when it
    reaches the shoe does not depend on how much of it there is. There is no
    tolerance to compute -- every kick breaches, and by the same margin."""
    results = [_run(v, 0.0) for v in (2.0, 6.0, 20.0, 60.0)]

    # every size breaches, and at the SAME depth -- that is the degeneracy
    assert all(r.breached for r in results)
    assert len({r.breach_depth_tvd for r in results}) == 1
    assert all(r.gas_expansion_frac == 0.0 for r in results)

    # BHP is NOT the degenerate quantity and must not be asserted as one: a
    # bigger bubble occupies more of the hole, so less mud sits beneath it and
    # BHP falls (8313 -> 7692 psi over a 30x influx range). What does not depend
    # on size is what the gas imposes when it ARRIVES at the shoe.
    bhps = [r.max_bhp_psi for r in results]
    assert all(b < a for a, b in zip(bhps, bhps[1:]))


def test_the_tolerance_is_created_entirely_by_the_relief():
    """Rigid breaches at every size; with mud compressibility a small kick
    survives. That is what makes the shut-in case a tolerance at all."""
    assert _run(2.0, 0.0).breached
    assert not _run(2.0, WBM_COMPRESSIBILITY).breached
    assert not _run(6.0, OBM_COMPRESSIBILITY).breached
    assert _run(6.0, WBM_COMPRESSIBILITY).breached      # WBM relieves less than OBM


def test_relief_is_monotone_in_compressibility_and_in_influx():
    """The bug that a fixed-point iteration hides: solved as a root, the
    tolerance rises with relief and the margin falls with influx."""
    for influx in (2.0, 6.0, 20.0):
        bhps = [_run(influx, c).max_bhp_psi
                for c in (0.0, 1e-6, WBM_COMPRESSIBILITY, OBM_COMPRESSIBILITY)]
        assert all(b <= a + 1e-9 for a, b in zip(bhps, bhps[1:])), (
            f"relief must not raise BHP: {bhps}")

    for c in (WBM_COMPRESSIBILITY, OBM_COMPRESSIBILITY):
        expansion = [_run(v, c).gas_expansion_frac for v in (2.0, 6.0, 20.0)]
        # the same absolute relief is a smaller fraction of a bigger bubble
        assert all(b <= a for a, b in zip(expansion, expansion[1:]))


def test_the_bubble_carries_bottom_hole_pressure_upward():
    """The hazard: BHP RISES as the gas migrates, the opposite of a circulated
    kill where the choke holds it flat."""
    result = _run(20.0, WBM_COMPRESSIBILITY)

    assert result.max_bhp_psi > COMMON["bhp_initial_psi"]
    assert result.breach_depth_tvd == pytest.approx(SHOE)


def test_a_burst_limit_can_bind_before_the_formation():
    """Surface pressure is a live constraint in a shut-in well."""
    generous_fp = (np.array([0.0, TD]), np.array([20.0, 20.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = shut_in_migration(SECTIONS, PP, generous_fp, influx_bbl=20.0,
                                  compressibility_per_psi=WBM_COMPRESSIBILITY,
                                  **COMMON)
        tight = shut_in_migration(SECTIONS, PP, generous_fp, influx_bbl=20.0,
                                  compressibility_per_psi=WBM_COMPRESSIBILITY,
                                  burst_pressure_psi=500.0, **COMMON)

    assert not loose.breached
    assert tight.breached
    assert tight.breach_depth_tvd == 0.0        # surface


def test_a_well_with_no_open_hole_is_refused():
    with pytest.raises(ValueError, match="no open-hole section"):
        shut_in_migration([WellSection(0.0, TD, 0.066, False)], PP, FP,
                          influx_bbl=10.0, compressibility_per_psi=0.0, **COMMON)
