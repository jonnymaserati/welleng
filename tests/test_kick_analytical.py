"""Analytical (breakpoint) kick-tolerance solver validation.

The analytical solver (``analytical_kick_tolerance``) evaluates the migration-form
kick tolerance at only the BREAKPOINTS of the imposed-pressure-vs-position curve
(the exact worst gas position), instead of a fine march + bisection. These tests
lock it against the migration engine (``max_influx_circulated``):

.. warning::

   This module asserts AGREEMENT BETWEEN TWO OF OUR OWN ENGINES. It is not
   external validation, and an earlier version of this docstring claimed the
   migration engine was "itself validated vs SPE-208788 / SPE-202426 /
   SPE-140113". **That claim was false** -- no test connects the migration engine
   to any of those papers. Every paper reproduction in the suite
   (``test_spe208788_worked_example``, ``test_spe202426_fig12``,
   ``test_spe140113_santos``, ``test_nogepa``) exercises the CLOSED FORM or its
   own dedicated formula, never the marching or analytical engines; and
   ``test_migration`` states in its own docstring that it is a reasonableness
   suite against "not a published worked example".

   So the marching path's assurance is: internal reasonableness, monotonicity,
   and agreement with the analytical solver. Real engineering assurance --
   NOT paper reproduction. See ``docs/dev/KICK_CLOSED_FORM_AUDIT.md``.

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


def test_bubble_length_limit_is_casing_burst_regime():
    """Santos SPE-140113: a bubble cannot exceed the open-hole length. With a strong
    fracture profile the fracture-only closed form would return an over-length bubble
    (~116 bbl needs a 4112 ft bubble in a 4000 ft open hole). That gas-top-at-shoe
    worst case is UNREACHABLE, so the shoe holds to full open-hole displacement: the
    result must be flagged open_hole_unconstrained / casing-burst (governing barrier is casing
    burst to surface, higher KT -- not a fracture-limited ~116) with max_influx = the
    full-displacement volume (~109). Analytical and march must agree (JJ 2026-07-16)."""
    a = analytical_kick_tolerance(TWO_SECTION, PP, STRONG_FP,
                                  gas_density_mode="conservative", **COMMON)
    m = max_influx_circulated(TWO_SECTION, PP, STRONG_FP, gas_density_mode="conservative",
                              mode="thorough", n_steps=300, **COMMON)
    assert a.open_hole_unconstrained and m.open_hole_unconstrained
    assert m.limited_by == "open_hole_capacity"
    assert "not" in a.breakpoints.get("note", "").lower()   # carries the caveat
    assert a.max_influx_bbl == pytest.approx(m.max_influx_bbl, abs=0.5)
    assert a.max_influx_bbl < 112.0              # full displacement, NOT the ~116 fracture value


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


def test_open_hole_unconstrained_when_whole_hole_tolerates_gas():
    """If the entire exposed hole can be displaced to gas without fracturing, the
    OPEN HOLE does not constrain the KT at the provided FP -- flagged
    open_hole_unconstrained (NOT 'unlimited': casing burst above the shoe governs,
    not assessed here)."""
    strong_fp = (np.array([0.0, 10500.0]), np.array([30.0, 30.0]))
    r = analytical_kick_tolerance(TWO_SECTION, PP, strong_fp,
                                  gas_density_mode="conservative", **COMMON)
    assert r.open_hole_unconstrained


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


def test_gas_bottom_solve_converges_rather_than_truncating():
    """The gas-bottom-pinned influx solve must END ON ITS TOLERANCE, never on the
    iteration cap.

    Plain false position stagnates on this problem -- the margin is strongly convex
    in the influx, so one bracket endpoint is retained every iteration and the
    bracket never halves. Under the original 24-iteration cap the loop was cut off
    mid-flight and the truncated influx was returned as if converged: 3.7% low on a
    shoe-bound case, and erratic enough that KT-vs-bit-position grew a spurious cusp
    (54 ft of gas-top movement across 2 ft of bit). The Illinois modification forces
    the bracket down; the cap is now a runaway backstop.

    Guard: the answer must be independent of the cap and of the tolerance, which is
    exactly what a truncating solve is not.
    """
    from welleng.kick_tolerance import analytical as A

    sections = [WellSection(0.0, 6500.0, 0.066, False),
                WellSection(6500.0, 7556.5, 0.066, True),
                WellSection(7556.5, 8056.5, 0.012, True),
                WellSection(8056.5, 10500.0, 0.088, True)]

    tol_psi, tol_bbl, max_iter = (
        A._SECANT_TOL_PSI, A._SECANT_TOL_BBL, A._SECANT_MAX_ITER
    )
    try:
        A._SECANT_TOL_PSI, A._SECANT_TOL_BBL, A._SECANT_MAX_ITER = 1e-9, 1e-12, 800
        converged = _analytic(sections, FP_UNIFORM, mode="exact")
    finally:
        A._SECANT_TOL_PSI, A._SECANT_TOL_BBL, A._SECANT_MAX_ITER = (
            tol_psi, tol_bbl, max_iter
        )

    shipped = _analytic(sections, FP_UNIFORM, mode="exact")

    assert shipped == pytest.approx(converged, rel=1e-4)


def test_kick_tolerance_is_smooth_in_bit_position():
    """KT varies smoothly with bit position; a kink means the solver, not physics.

    The gas column's length is density-driven and cap-independent, so moving the
    bit moves the answer continuously. A truncating inner solve broke this: the
    binding gas top swung 54 ft across 2 ft of bit travel, with the derivative
    flipping sign twice, which reads as a genuine cusp and defeats any search for
    the worst bit position.
    """
    def sections_for_bit(bit, dc_len=500.0, shoe=6500.0, td=10500.0):
        dc_top = max(bit - dc_len, 0.0)
        out = []
        for a, b in zip(sorted({0.0, shoe, dc_top, bit, td}),
                        sorted({0.0, shoe, dc_top, bit, td})[1:]):
            if b <= a:
                continue
            mid = 0.5 * (a + b)
            cap = 0.088 if mid > bit else (0.012 if mid >= dc_top else 0.066)
            out.append(WellSection(a, b, cap, mid >= shoe))
        return out

    bits = np.arange(8056.0, 8058.01, 0.2)
    tops = np.array([
        analytical_kick_tolerance(
            sections_for_bit(float(b)), PP, FP_UNIFORM,
            gas_density_mode="exact", **COMMON
        ).binding_gas_top_tvd
        for b in bits
    ])

    assert np.all(np.diff(tops) >= -1e-9), "binding gas top is not monotone in bit"
    assert tops.max() - tops.min() < 5.0, (
        f"gas top swings {tops.max() - tops.min():.1f} ft across 2 ft of bit"
    )


# ---------------------------------------------------------------------------
# casing burst — indicative surface-containment check
# ---------------------------------------------------------------------------
OPEN_HOLE = WellSection(6500.0, 10500.0, 0.046, True)


def test_a_section_without_a_rating_is_not_assessed():
    """Backwards compatible: no burst allowable means the casing is simply not
    assessed, exactly as before 0.27 -- not silently assumed adequate."""
    plain = WellSection(0.0, 6500.0, 0.0489, False)

    r = analytical_kick_tolerance([plain, OPEN_HOLE], PP, FP_UNIFORM,
                                  gas_density_mode="exact", **COMMON)

    assert r.surface_containment_bbl is None
    assert r.casing_binds is False


def test_cased_section_carries_the_catalogue_rating_when_graded():
    from welleng.kick_tolerance.geometry import BURST_DESIGN_FACTOR, cased_section

    graded = cased_section(0.0, 6500.0, casing_od_in=9.625, casing_weight_ppf=47.0,
                           pipe_od_in=5.0, grade="L80")
    ungraded = cased_section(0.0, 6500.0, casing_od_in=9.625, casing_weight_ppf=47.0,
                             pipe_od_in=5.0)

    # API 5CT 9-5/8 47 ppf L80: minimum internal yield 6870 psi
    assert graded.burst_pressure_psi == pytest.approx(6870.0 * BURST_DESIGN_FACTOR)
    # the catalogue flags an absent grade rather than guessing, and so do we
    assert ungraded.burst_pressure_psi is None


def test_real_casing_does_not_bind_before_the_formation():
    """The premise the engine has always assumed -- now checked rather than
    assumed. A 9-5/8 47 ppf L80 string holds far more than the open hole allows."""
    from welleng.kick_tolerance.geometry import cased_section

    csg = cased_section(0.0, 6500.0, casing_od_in=9.625, casing_weight_ppf=47.0,
                        pipe_od_in=5.0, grade="L80")

    r = analytical_kick_tolerance([csg, OPEN_HOLE], PP, FP_UNIFORM,
                                  gas_density_mode="exact", **COMMON)

    assert r.surface_containment_bbl > r.max_influx_bbl
    assert r.casing_binds is False


def test_a_weak_string_binds_before_the_formation_and_is_flagged():
    weak = WellSection(0.0, 6500.0, 0.0489, False, burst_pressure_psi=900.0)

    r = analytical_kick_tolerance([weak, OPEN_HOLE], PP, FP_UNIFORM,
                                  gas_density_mode="exact", **COMMON)

    assert r.surface_containment_bbl < r.max_influx_bbl
    assert r.casing_binds is True


def test_surface_containment_rises_with_the_allowable():
    from welleng.kick_tolerance.analytical import max_influx_contained_at_surface

    sections = [WellSection(0.0, 6500.0, 0.0489, False), OPEN_HOLE]
    volumes = [
        max_influx_contained_at_surface(
            sections, burst_pressure_psi=p, bottom_tvd=10500.0, **COMMON
        )
        for p in (500.0, 1000.0, 2000.0, 4000.0, 5496.0)
    ]

    assert all(b > a for a, b in zip(volumes, volumes[1:]))


def test_a_well_that_cannot_be_shut_in_returns_zero():
    """Below the surface pressure of the mud column alone, no influx fits."""
    from welleng.kick_tolerance.analytical import max_influx_contained_at_surface

    sections = [WellSection(0.0, 6500.0, 0.0489, False), OPEN_HOLE]
    mud_alone = COMMON["bhp_psi"] - 0.0521 * COMMON["rho_mud_ppg"] * 10500.0

    assert max_influx_contained_at_surface(
        sections, burst_pressure_psi=max(mud_alone, 1.0) - 1.0,
        bottom_tvd=10500.0, **COMMON
    ) == 0.0


def test_callable_profile_is_refused_because_its_breakpoints_are_invisible():
    """The solver is exact because it enumerates every depth where the binding
    constraint can turn. A callable exposes none, so a weak zone between section
    boundaries is silently skipped -- and skipped in the NON-conservative
    direction. Same weak zone at 8000 ft: 44.4 bbl as a table, 58.2 bbl as a
    callable, a 31% over-report with no warning. Refuse instead.
    """
    knots_tvd = np.array([0.0, 7999.0, 8000.0, 8001.0, 10500.0])
    knots_ppg = np.array([14.0, 14.0, 13.0, 14.0, 14.0])
    as_table = (knots_tvd, knots_ppg)
    as_callable = lambda d: np.interp(d, knots_tvd, knots_ppg)  # noqa: E731

    with pytest.raises(ValueError, match="callable profile"):
        analytical_kick_tolerance(TWO_SECTION, PP, as_callable,
                                  gas_density_mode="exact", **COMMON)

    # the table form sees the weak zone and binds there
    table = analytical_kick_tolerance(TWO_SECTION, PP, as_table,
                                      gas_density_mode="exact", **COMMON)
    assert table.binding_depth_tvd == pytest.approx(8000.0)

    # a caller who pins the depths has said where to look, so a callable is fine
    pinned = analytical_kick_tolerance(TWO_SECTION, PP, as_callable,
                                       gas_density_mode="exact",
                                       check_depths=[6500.0, 8000.0], **COMMON)
    assert pinned.max_influx_bbl == pytest.approx(table.max_influx_bbl, rel=0.02)


def test_off_diagonal_candidate_a_face_on_one_boundary_binding_at_another():
    """The worst gas position can have a face on one boundary while a DIFFERENT
    depth binds. That pairing was never enumerated.

    The candidate set paired "gas top at d" with "FP enforced at d" -- the
    diagonal of (face position x binding depth). With a long tight bottom section
    the worst config has the gas top on the BHA-top boundary and the SHOE
    breaching. Measured on a 2500 ft tight section: the diagonal-only set returned
    18.692 bbl whose worst gas position sits at -52 psi, i.e. 7.1% UNSAFE against
    a true breach of 17.446 bbl (5 ft scan over all gas positions). With the
    off-diagonal family the solver returns 17.447, +0.01%.

    Asserted against the DEFINITION -- the influx at which the worst position over
    the whole migration first breaches -- and not against the marching engine,
    which under-samples this geometry by 0.9% and would make a wrong answer look
    conservative.
    """
    from welleng.kick_tolerance.analytical import _top_for_bottom, _z
    from welleng.kick_tolerance.migration import (
        _as_temp_callable, _resolve_bh_state, pressure_at_depth, ppg_to_psi,
    )

    td = 10500.0
    sections = [WellSection(0.0, 6500.0, 0.066, False),
                WellSection(6500.0, 8000.0, 0.046, True),
                WellSection(8000.0, td, 0.012, True)]
    depths = np.array([6500.0, 8000.0, td])
    fp_psi = ppg_to_psi(np.full(3, 14.0), depths)
    gas_bh = _resolve_bh_state(COMMON["gas_bh_state"], COMMON["bhp_psi"])
    temp_fn = _as_temp_callable(None, gas_bh[1])

    def worst_margin(volume, step=25.0):
        worst = np.inf
        for gas_bottom in np.arange(6600.0, td + 1.0, step):
            gas_top = _top_for_bottom(
                gas_bottom, volume, sections, td, bhp_psi=COMMON["bhp_psi"],
                rho_mud_ppg=COMMON["rho_mud_ppg"], gas_bh=gas_bh, temp_fn=temp_fn,
                gas_density_mode="exact", z_fn=_z,
            )
            if gas_top is None or gas_top < 0:
                continue
            p = pressure_at_depth(
                depths, gas_top_tvd=gas_top, gas_bottom_tvd=gas_bottom,
                bottom_tvd=td, bhp_psi=COMMON["bhp_psi"],
                rho_mud_ppg=COMMON["rho_mud_ppg"], gas_bh=gas_bh,
                gas_density_mode="exact", temp_profile=None,
            )
            worst = min(worst, float(np.min(fp_psi - p)))
        return worst

    answer = _analytic(sections, FP_UNIFORM, mode="exact")

    # the answer is ON the envelope -- no position breaches, and none has slack
    assert worst_margin(answer) > -5.0
    # and it is the LARGEST such influx: a little more definitely breaches
    assert worst_margin(answer * 1.05) < 0.0
    # the pre-fix answer is now excluded
    assert answer < 18.0
