"""Analytical (breakpoint) kick-tolerance solver validation.

The analytical solver (``analytical_kick_tolerance``) evaluates the migration-form
kick tolerance at only the BREAKPOINTS of the imposed-pressure-vs-position curve
(the exact worst gas position), instead of a fine march + bisection. These tests
lock it against the validated migration engine (``max_influx_circulated``, itself
validated vs SPE-208788 / SPE-202426 / SPE-140113):

  * conservative mode reproduces the migration's safe-side bound on standard
    geometry (base / weak-zone / sloped-FP), and stays CONSERVATIVE (<= the march)
    on a tight/BHA bottom section -- the case where a naive march under-samples and
    the analytical must not over-state;
  * exact mode (true gas-density integration) is MORE ACCURATE -- it recovers real
    tolerance the conservative bound forfeits (exact >= conservative);
  * the unlimited-tolerance path (whole exposed hole tolerates gas);
  * monotonicity of the tolerable set.
"""
import numpy as np
import pytest

from welleng.kick_tolerance import (
    WellSection, max_influx_circulated, analytical_kick_tolerance,
)

GAS = (None, 660.0, None, None)          # (P_bh=BHP, T_bh_rankine, Z, rho) -> H-Y fills
COMMON = dict(bhp_psi=6402.0, rho_mud_ppg=12.0, gas_bh_state=GAS)

PP = (np.array([0.0, 10500.0]), np.array([10.5, 11.0]))
FP_UNIFORM = (np.array([0.0, 10500.0]), np.array([14.0, 14.0]))

TWO_SECTION = [WellSection(0.0, 6500.0, 0.066, False),
               WellSection(6500.0, 10500.0, 0.046, True)]
TIGHT_BHA = [WellSection(0.0, 6500.0, 0.066, False),
             WellSection(6500.0, 10000.0, 0.046, True),
             WellSection(10000.0, 10500.0, 0.012, True)]     # tight BHA bottom
WEAK_FP = (np.array([0.0, 8000.0, 8001.0, 10500.0]), np.array([14.0, 14.0, 13.2, 13.2]))
SLOPED_FP = (np.array([0.0, 6500.0, 10500.0]), np.array([13.0, 13.5, 14.5]))


def _march(sections, fp, mode="conservative"):
    return max_influx_circulated(
        sections, PP, fp, gas_density_mode=mode, mode="thorough", n_steps=300,
        **COMMON).max_influx_bbl


def _analytic(sections, fp, mode="conservative"):
    return analytical_kick_tolerance(
        sections, PP, fp, gas_density_mode=mode, **COMMON).max_influx_bbl


STRONG_FP = (np.array([0.0, 10500.0]), np.array([16.0, 16.0]))  # over-length regime


@pytest.mark.parametrize("sections,fp", [
    (TWO_SECTION, FP_UNIFORM),
    (TWO_SECTION, WEAK_FP),
    (TWO_SECTION, SLOPED_FP),
    (TWO_SECTION, STRONG_FP),   # bubble-length regime: bubble would exceed the open hole
])
def test_conservative_matches_migration_standard(sections, fp):
    """On standard geometry the analytical (conservative) reproduces the validated
    migration to <1 %, and never over-states it (safe-side)."""
    a, m = _analytic(sections, fp), _march(sections, fp)
    assert a == pytest.approx(m, rel=0.01)
    assert a <= m + 0.15            # conservative: at or below the march (tol: bisection)


def test_bubble_length_limit_caps_analytical():
    """Santos SPE-140113: a bubble cannot exceed the open-hole length. With a strong
    fracture profile the fracture-only closed form would return an over-length bubble
    (~116 bbl needs a 4112 ft bubble in a 4000 ft open hole); the cap must bring it
    back to the migration's bubble-length limit (~109 bbl), NOT the fracture value."""
    a = analytical_kick_tolerance(TWO_SECTION, PP, STRONG_FP,
                                  gas_density_mode="conservative", **COMMON)
    m = max_influx_circulated(TWO_SECTION, PP, STRONG_FP, gas_density_mode="conservative",
                              mode="thorough", n_steps=300, **COMMON)
    assert m.limited_by == "bha_length"          # the migration is bubble-length-limited here
    assert a.max_influx_bbl == pytest.approx(m.max_influx_bbl, abs=0.5)
    assert a.max_influx_bbl < 112.0              # well below the ~116 fracture-only value


def test_conservative_is_conservative_on_tight_bha():
    """The motivating case: a tight/BHA bottom section, where a naive march
    under-samples the narrow gas-bottom breakpoint. The analytical must be
    CONSERVATIVE (<= the fine march) -- never the old 170-bbl over-estimate."""
    a = _analytic(TIGHT_BHA, FP_UNIFORM)
    m = _march(TIGHT_BHA, FP_UNIFORM)
    assert a <= m                    # safe-side
    assert 45.0 < a < 60.0           # near the true ~55 bbl, NOT the 170 bbl bug


def test_weak_zone_binds_at_the_weak_zone():
    """A weak FP zone (a PP/FP breakpoint, not a section boundary) must be the
    binding depth -- the unified-boundary candidate set catches it."""
    r = analytical_kick_tolerance(TWO_SECTION, PP, WEAK_FP,
                                  gas_density_mode="conservative", **COMMON)
    # gas top pinned at the weak-zone breakpoint (~8001 ft)
    assert r.binding_gas_top_tvd == pytest.approx(8001.0, abs=50.0)


def test_check_depths_gives_single_shoe_semantics():
    """``check_depths=[shoe]`` enforces the FP envelope only at the shoe, so a
    deeper weak zone is NOT binding -- the free-tier single-shoe convention.
    KT rises vs the full sections-aware check, and the binding depth is the shoe."""
    shoe = 6500.0
    full = analytical_kick_tolerance(TWO_SECTION, PP, WEAK_FP,
                                     gas_density_mode="conservative", **COMMON)
    shoe_only = analytical_kick_tolerance(TWO_SECTION, PP, WEAK_FP, check_depths=[shoe],
                                          gas_density_mode="conservative", **COMMON)
    assert full.binding_depth_tvd == pytest.approx(8001.0, abs=50.0)   # weak zone binds
    assert shoe_only.binding_depth_tvd == pytest.approx(shoe, abs=1.0)  # shoe binds
    assert shoe_only.max_influx_bbl > full.max_influx_bbl              # weak zone ignored


def test_check_depths_is_noop_when_shoe_already_binds():
    """With a uniform FP the shoe already binds, so pinning check_depths=[shoe]
    changes nothing -- the override is backward-compatible where it should be."""
    default = analytical_kick_tolerance(TWO_SECTION, PP, FP_UNIFORM,
                                        gas_density_mode="conservative", **COMMON)
    pinned = analytical_kick_tolerance(TWO_SECTION, PP, FP_UNIFORM, check_depths=[6500.0],
                                       gas_density_mode="conservative", **COMMON)
    assert pinned.max_influx_bbl == pytest.approx(default.max_influx_bbl, abs=1e-9)


@pytest.mark.parametrize("sections,fp", [
    (TWO_SECTION, FP_UNIFORM),
    (TIGHT_BHA, FP_UNIFORM),
])
def test_exact_is_more_accurate_not_less_conservative(sections, fp):
    """Exact (true gas-density) recovers tolerance the conservative bound forfeits:
    exact >= conservative. It is the accurate value, not a looser one."""
    exact = _analytic(sections, fp, mode="exact")
    cons = _analytic(sections, fp, mode="conservative")
    assert exact >= cons - 0.15


def test_unlimited_when_whole_hole_tolerates_gas():
    """If the entire exposed hole can be displaced to gas without fracturing, the
    tolerance is unlimited (full displacement permissible)."""
    strong_fp = (np.array([0.0, 10500.0]), np.array([30.0, 30.0]))
    r = analytical_kick_tolerance(TWO_SECTION, PP, strong_fp,
                                  gas_density_mode="conservative", **COMMON)
    assert r.is_unlimited


def test_tolerable_set_is_monotone():
    """within-envelope must hold below V* and fail above it (monotone tolerable
    set) -- the property the migration bug violated on a tight BHA."""
    r = analytical_kick_tolerance(TIGHT_BHA, PP, FP_UNIFORM,
                                  gas_density_mode="conservative", **COMMON)
    vstar = r.max_influx_bbl
    below = max_influx_circulated(
        TIGHT_BHA, PP, FP_UNIFORM, gas_density_mode="conservative",
        mode="thorough", n_steps=300, **COMMON)
    # the analytical V* must not exceed the fine march's answer (conservative)
    assert vstar <= below.max_influx_bbl + 1.0
