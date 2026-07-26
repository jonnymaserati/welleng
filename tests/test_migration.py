"""Reasonableness tests for the gas-migration kick-tolerance engine.

This is an ASSEMBLED model (an extension beyond the static NOGEPA single-shoe
method), not a published worked example -- so the assertions are reasonableness
checks, not bit-exact targets:

  (a) a small influx stays WITHIN the PP-FP envelope (min FP margin > 0);
  (b) a large influx BREACHES the fracture pressure (within_envelope False),
      binding at/near the casing shoe;
  (c) the imposed pressure AT THE SHOE peaks when the gas top reaches the shoe
      and falls once the gas passes -- i.e. the max-shoe-pressure step matches
      the static single-shoe check;
  (d) the BHA/open-hole length flag fires when the bubble grows longer than the
      open hole it must migrate through.
"""

import numpy as np
import pytest

from welleng.kick_tolerance.migration import (
    WellSection, migrate, max_influx_circulated, pressure_at_depth,
    linear_temp_profile, G_PSI_PER_PPG_FT,
)
from welleng.kick_tolerance.gas_z import hall_yarborough_z, gas_density_ppg


# --- A simple vertical well: open hole below the shoe, casing above ----------
SHOE_TVD = 5000.0
BOTTOM_TVD = 10000.0
RHO_MUD = 12.0            # ppg
# Constant BHP a little above mud hydrostatic at TD (kill margin).
BHP = G_PSI_PER_PPG_FT * RHO_MUD * BOTTOM_TVD + 150.0   # ~6402 psi

# 8.5" open hole with 5" pipe; larger cased annulus above the shoe.
OPEN_CAP = (8.5 ** 2 - 5.0 ** 2) / 1029.4     # ~0.0459 bbl/ft
CASED_CAP = (9.625 ** 2 - 5.0 ** 2) / 1029.4  # ~0.0657 bbl/ft
OPEN_HOLE_LENGTH = BOTTOM_TVD - SHOE_TVD       # 5000 ft


def make_sections():
    return [
        WellSection(0.0, SHOE_TVD, CASED_CAP, is_open_hole=False),
        WellSection(SHOE_TVD, BOTTOM_TVD, OPEN_CAP, is_open_hole=True),
    ]


def bh_state():
    # Bottom-hole methane state; Z_bh / rho_gas computed by the HY backend.
    return (BHP, 660.0, None, None)  # 200 degF = 660 degR


# Linear PP / FP profiles in ppg vs TVD (increasing with depth).
PP_TABLE = (np.array([0.0, BOTTOM_TVD]), np.array([10.5, 11.0]))
FP_TABLE = (np.array([0.0, BOTTOM_TVD]), np.array([14.0, 14.0]))


def test_small_influx_within_envelope():
    """(a) A small influx stays inside the PP-FP window everywhere."""
    res = migrate(
        make_sections(), PP_TABLE, FP_TABLE,
        bhp_psi=BHP, influx_bbl_bh=3.0, rho_mud_ppg=RHO_MUD,
        gas_bh_state=bh_state(), n_steps=120,
    )
    assert res.within_envelope is True
    assert res.min_fp_margin_psi > 0.0
    assert res.bha_length_exceeded is False
    # The animation trajectory: one frame per step, bottom -> surface.
    assert len(res.steps) == 120
    assert res.steps[0].gas_top_tvd > res.steps[-1].gas_top_tvd


def test_large_influx_breaches_fp_at_shoe():
    """(b) A large influx breaches FP, binding at/near the shoe."""
    res = migrate(
        make_sections(), PP_TABLE, FP_TABLE,
        bhp_psi=BHP, influx_bbl_bh=45.0, rho_mud_ppg=RHO_MUD,
        gas_bh_state=bh_state(), n_steps=120,
    )
    assert res.within_envelope is False
    assert res.min_fp_margin_psi < 0.0
    # Binding is in the exposed open hole, close to the shoe (top of open hole).
    assert SHOE_TVD <= res.binding_tvd <= SHOE_TVD + 0.15 * OPEN_HOLE_LENGTH


def test_shoe_pressure_peaks_as_gas_top_reaches_shoe():
    """(c) Max imposed pressure at the shoe occurs when the gas top ~ shoe.

    This is the migration-model recovery of the static single-shoe check: the
    worst shoe loading is when the light gas column sits just below the shoe.
    """
    res = migrate(
        make_sections(), PP_TABLE, FP_TABLE,
        bhp_psi=BHP, influx_bbl_bh=20.0, rho_mud_ppg=RHO_MUD,
        gas_bh_state=bh_state(), n_steps=200,
    )
    ctx = res._ctx
    # Recompute imposed pressure at the shoe for every step (public helper).
    p_shoe = np.array([
        pressure_at_depth(
            SHOE_TVD, gas_top_tvd=s.gas_top_tvd, gas_bottom_tvd=s.gas_bottom_tvd,
            bottom_tvd=ctx["bottom_tvd"], bhp_psi=ctx["bhp_psi"],
            rho_mud_ppg=ctx["rho_mud_ppg"], gas_bh=ctx["gas_bh"],
        )
        for s in res.steps
    ])
    k = int(np.argmax(p_shoe))
    gas_top_at_peak = res.steps[k].gas_top_tvd
    # The peak shoe pressure happens with the gas top essentially at the shoe.
    assert abs(gas_top_at_peak - SHOE_TVD) < 0.03 * BOTTOM_TVD
    # And it is a genuine peak: pressure rises into it and falls out of it.
    assert p_shoe[k] > p_shoe[0]
    assert p_shoe[k] > p_shoe[-1]


def test_gas_density_mode_ordering():
    """Safe-side ordering: P(conservative) >= P(exact) >= P(bottom-hole-const).

    Bottom-hole gas is the densest; using it everywhere (the superseded
    behaviour) over-states the column weight and UNDER-states the pressure
    at/above the gas -- non-conservative for a fracture barrier. The "exact"
    mode integrates the true lighter up-hole density (the correct value); the
    "conservative" DEFAULT holds the gas-top (lightest) density constant -- the
    highest-pressure, safe-side bound. Ordering must hold at every depth at or
    above the gas.
    """
    P_bh, T = BHP, 660.0
    Z_bh = hall_yarborough_z(P_bh, T)
    rho_gas_bh = gas_density_ppg(P_bh, T, Z_bh)
    gas_bh = (P_bh, T, Z_bh, rho_gas_bh)
    gas_top, gas_bottom = 4000.0, 8000.0  # a long bubble spanning the shoe

    def bottom_hole_constant(d):
        """Superseded (non-conservative) behaviour: constant bottom-hole density."""
        total = BOTTOM_TVD - d
        lo = max(d, gas_top)
        hi = min(BOTTOM_TVD, gas_bottom)
        gas_len = max(0.0, hi - lo)
        mud_len = total - gas_len
        return BHP - G_PSI_PER_PPG_FT * (RHO_MUD * mud_len + rho_gas_bh * gas_len)

    def P(d, mode):
        return pressure_at_depth(
            d, gas_top_tvd=gas_top, gas_bottom_tvd=gas_bottom,
            bottom_tvd=BOTTOM_TVD, bhp_psi=BHP, rho_mud_ppg=RHO_MUD,
            gas_bh=gas_bh, gas_density_mode=mode,
        )

    for d in (0.0, 2000.0, gas_top, 6000.0):  # at/above the gas top
        p_cons = P(d, "conservative")
        p_exact = P(d, "exact")
        p_bh = bottom_hole_constant(d)
        assert p_cons >= p_exact - 1e-6          # conservative is the safe-side bound
        assert p_exact >= p_bh - 1e-6            # true value >= old non-conservative

    # Strictly separated above the gas (each treatment raises the shallow pressure).
    assert P(0.0, "conservative") > P(0.0, "exact") + 1e-3
    assert P(0.0, "exact") > bottom_hole_constant(0.0) + 1.0

    # Default is conservative.
    assert P(0.0, "conservative") == pytest.approx(
        pressure_at_depth(
            0.0, gas_top_tvd=gas_top, gas_bottom_tvd=gas_bottom,
            bottom_tvd=BOTTOM_TVD, bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh=gas_bh,
        )
    )

    # Below the gas all treatments reduce to the same pure-mud column.
    assert P(9000.0, "conservative") == pytest.approx(bottom_hole_constant(9000.0))
    assert P(9000.0, "exact") == pytest.approx(bottom_hole_constant(9000.0))

    # An invalid mode is rejected.
    with pytest.raises(ValueError):
        P(0.0, "bogus")


def test_bha_length_flag_fires_for_oversized_bubble():
    """(d) The bubble grows longer than the open hole -> flag fires."""
    res = migrate(
        make_sections(), PP_TABLE, FP_TABLE,
        bhp_psi=BHP, influx_bbl_bh=250.0, rho_mud_ppg=RHO_MUD,
        gas_bh_state=bh_state(), n_steps=120,
    )
    assert res.bha_length_exceeded is True
    assert max(s.gas_length_ft for s in res.steps) > OPEN_HOLE_LENGTH


def test_bha_flag_false_for_small_bubble():
    """Control for (d): a small bubble never exceeds the open hole."""
    res = migrate(
        make_sections(), PP_TABLE, FP_TABLE,
        bhp_psi=BHP, influx_bbl_bh=3.0, rho_mud_ppg=RHO_MUD,
        gas_bh_state=bh_state(), n_steps=120,
    )
    assert res.bha_length_exceeded is False


def test_backend_computes_bh_state_when_none():
    """Z_bh / rho_gas left None are computed by the Hall-Yarborough backend."""
    res = migrate(
        make_sections(), PP_TABLE, FP_TABLE,
        bhp_psi=BHP, influx_bbl_bh=5.0, rho_mud_ppg=RHO_MUD,
        gas_bh_state=(BHP, 660.0, None, None), n_steps=50,
    )
    z_expected = hall_yarborough_z(BHP, 660.0)
    assert res._ctx["Z_bh"] == pytest.approx(z_expected, rel=1e-9)
    assert res._ctx["rho_gas_ppg"] == pytest.approx(
        gas_density_ppg(BHP, 660.0, z_expected), rel=1e-9
    )


def test_max_influx_circulated_is_the_envelope_boundary():
    """The migration kick tolerance V* = max influx that can be circulated out.

    It sits between the known passing (3 bbl) and breaching (45 bbl) influxes;
    migrate() at V* is inside the envelope (safe-side boundary) and just above V*
    it fails (breaches FP or the bubble can't pass the BHA)."""
    kt = max_influx_circulated(
        make_sections(), PP_TABLE, FP_TABLE,
        bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh_state=bh_state(),
        n_steps=120, tol_bbl=0.1,
    )
    vstar = kt.max_influx_bbl
    assert 3.0 < vstar < 45.0
    # It reports WHERE/why it breaches: fracture, at/near the shoe here.
    assert kt.limited_by == "fracture"
    assert SHOE_TVD <= kt.binding_tvd <= SHOE_TVD + 0.15 * OPEN_HOLE_LENGTH
    at = migrate(
        make_sections(), PP_TABLE, FP_TABLE, bhp_psi=BHP, influx_bbl_bh=vstar,
        rho_mud_ppg=RHO_MUD, gas_bh_state=bh_state(), n_steps=120,
    )
    assert at.within_envelope and not at.bha_length_exceeded
    over = migrate(
        make_sections(), PP_TABLE, FP_TABLE, bhp_psi=BHP, influx_bbl_bh=vstar + 1.0,
        rho_mud_ppg=RHO_MUD, gas_bh_state=bh_state(), n_steps=120,
    )
    assert (not over.within_envelope) or over.bha_length_exceeded


def test_max_influx_circulated_threads_temp_profile():
    """temp_profile flows through the inverse (max-influx) solve, safe-side.

    The isothermal-at-660R baseline is the HOTTEST up-hole case (lightest gas
    column -> highest imposed shoe pressure -> least headroom). A geothermal
    profile (cooler up-hole, GEO_TEMP) makes the column heavier and the imposed
    shoe pressure lower, so MORE influx can be circulated out: V*_geo >= V*_iso.
    temp_profile=None must reproduce the isothermal V* exactly.
    """
    common = dict(
        bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh_state=bh_state(),
        n_steps=120, tol_bbl=0.1,
    )
    v_iso = max_influx_circulated(make_sections(), PP_TABLE, FP_TABLE, **common)
    v_none = max_influx_circulated(
        make_sections(), PP_TABLE, FP_TABLE, temp_profile=None, **common
    )
    v_geo = max_influx_circulated(
        make_sections(), PP_TABLE, FP_TABLE, temp_profile=GEO_TEMP, **common
    )
    # None == isothermal default, and the cooler-up-hole profile gives >= headroom.
    assert v_none.max_influx_bbl == pytest.approx(v_iso.max_influx_bbl)
    assert v_geo.max_influx_bbl >= v_iso.max_influx_bbl - 1e-9
    assert v_geo.max_influx_bbl > v_iso.max_influx_bbl  # strictly, here it moves


def test_fast_mode_matches_thorough_within_tolerance():
    """fast mode (interface-anchored coarse grid) defines the same kick-tolerance
    envelope as thorough (the fine grid) to within ~1 %, at a fraction of the cost --
    good enough for the API / GUI; thorough remains the definitive check. Same
    binding region."""
    common = dict(bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh_state=bh_state())
    thorough = max_influx_circulated(
        make_sections(), PP_TABLE, FP_TABLE, mode="thorough", **common
    )
    fast = max_influx_circulated(
        make_sections(), PP_TABLE, FP_TABLE, mode="fast", **common
    )
    assert fast.max_influx_bbl == pytest.approx(thorough.max_influx_bbl, rel=0.02)
    assert abs(fast.binding_tvd - thorough.binding_tvd) < 200.0  # same binding region


def test_max_influx_unlimited_when_hole_fills_without_fracturing():
    """If the shoe holds through full open-hole displacement, the shoe-fracture
    tolerance is unlimited (JJ): the gas-top-at-shoe worst case is unreachable and
    the governing barrier moves to CASING BURST to surface (not modeled -- higher KT).
    Flagged open_hole_unconstrained=True, limited_by='open_hole_capacity', and max_influx is the
    full-OPEN-HOLE-DISPLACEMENT influx -- LESS than the geometric hole volume, because
    the influx (measured at bottom hole) expands to fill the hole (JJ 2026-07-16)."""
    fp_high = (np.array([0.0, BOTTOM_TVD]), np.array([30.0, 30.0]))  # never fractures
    r = max_influx_circulated(
        make_sections(), PP_TABLE, fp_high,
        bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh_state=bh_state(), n_steps=80,
    )
    v_hole = OPEN_CAP * OPEN_HOLE_LENGTH
    assert r.open_hole_unconstrained is True
    assert r.limited_by == "open_hole_capacity"
    assert 0.0 < r.max_influx_bbl < v_hole          # expansion-adjusted, below geometric
    # a normal (fracturing) well is NOT flagged unlimited and reports a finite limit.
    r2 = max_influx_circulated(
        make_sections(), PP_TABLE, FP_TABLE,
        bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh_state=bh_state(), n_steps=80,
    )
    assert r2.open_hole_unconstrained is False
    assert r2.limited_by == "fracture"
    assert r2.max_influx_bbl < v_hole


def test_geothermal_is_the_default_temp_when_provided():
    """User-defined temp wins; a provided geothermal is the DEFAULT (not isothermal);
    isothermal only when neither is given.

    temp_profile=None + geothermal=G  ==  temp_profile=G  (geothermal defaulted in).
    An explicit temp_profile overrides geothermal. No geothermal, no temp_profile ->
    isothermal-at-BHT (unchanged legacy default).
    """
    common = dict(
        bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh_state=bh_state(),
        n_steps=120, tol_bbl=0.1,
    )
    # geothermal defaulted in == passing it explicitly as temp_profile.
    v_geo_default = max_influx_circulated(
        make_sections(), PP_TABLE, FP_TABLE, geothermal=GEO_TEMP, **common
    )
    v_geo_explicit = max_influx_circulated(
        make_sections(), PP_TABLE, FP_TABLE, temp_profile=GEO_TEMP, **common
    )
    assert v_geo_default.max_influx_bbl == pytest.approx(v_geo_explicit.max_influx_bbl)
    # explicit temp_profile OVERRIDES geothermal (isothermal-at-BHT wins here).
    iso = lambda d: 660.0
    v_override = max_influx_circulated(
        make_sections(), PP_TABLE, FP_TABLE,
        temp_profile=iso, geothermal=GEO_TEMP, **common
    )
    v_iso = max_influx_circulated(make_sections(), PP_TABLE, FP_TABLE, **common)
    assert v_override.max_influx_bbl == pytest.approx(v_iso.max_influx_bbl)
    # and the geothermal default actually differs from the isothermal fallback.
    assert v_geo_default.max_influx_bbl != pytest.approx(v_iso.max_influx_bbl)


def test_weak_formation_binds_deeper_than_the_shoe():
    """Answers 'is the casing shoe always the worst point?' -- NO. With a weak
    formation in the open hole below the shoe, the migration binds THERE, not at
    the shoe: the static top-of-bubble-at-shoe assumption would miss it."""
    # FP dips to a weak zone at 6500 ft TVD (well below the 5000 ft shoe).
    weak_tvd = 6500.0
    fp_weak = (
        np.array([0.0, SHOE_TVD, weak_tvd - 200, weak_tvd, weak_tvd + 200, BOTTOM_TVD]),
        np.array([14.0, 14.0, 14.0, 11.6, 14.0, 14.0]),   # a low-FP notch at 6500 ft
    )
    kt = max_influx_circulated(
        make_sections(), PP_TABLE, fp_weak,
        bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh_state=bh_state(), n_steps=150,
    )
    # The governing depth is the weak formation, NOT the shoe.
    assert kt.limited_by == "fracture"
    assert abs(kt.binding_tvd - weak_tvd) < 0.05 * BOTTOM_TVD
    assert kt.binding_tvd > SHOE_TVD + 0.1 * OPEN_HOLE_LENGTH   # clearly below the shoe


# ============================================================================
# Non-isothermal temperature profile (T(depth))
# ============================================================================
# The bottom-hole temperature in bh_state() is 660 degR (200 degF), anchored at
# BOTTOM_TVD. A GEOTHERMAL gradient rises with depth, so it is COOLER up-hole than
# that bottom-hole value -- the isothermal-at-BH baseline is the hottest case.
GEO_TEMP = linear_temp_profile(SHOE_TVD, 600.0, BOTTOM_TVD, 660.0)  # cooler up-hole


def _gas_bh_tuple():
    """Explicit bottom-hole tuple with Z/rho filled by the HY backend (T = 660 R)."""
    P_bh, T = BHP, 660.0
    Z_bh = hall_yarborough_z(P_bh, T)
    rho_gas_bh = gas_density_ppg(P_bh, T, Z_bh)
    return (P_bh, T, Z_bh, rho_gas_bh)


def test_temp_profile_none_is_isothermal_identical():
    """(a) temp_profile=None == the isothermal default, bit-for-bit.

    A constant callable returning exactly the bottom-hole T (660 R) must give the
    IDENTICAL result too: the T-ratio in the gas density is exactly 1.0 and Z is
    evaluated at the same T, so nothing changes numerically.
    """
    gas_bh = _gas_bh_tuple()
    depths = np.array([0.0, 2000.0, 4000.0, SHOE_TVD, 6000.0, 8000.0, 9000.0])
    gt, gb = 4000.0, 8000.0
    for mode in ("conservative", "exact"):
        base = pressure_at_depth(
            depths, gas_top_tvd=gt, gas_bottom_tvd=gb, bottom_tvd=BOTTOM_TVD,
            bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh=gas_bh, gas_density_mode=mode,
        )
        explicit_none = pressure_at_depth(
            depths, gas_top_tvd=gt, gas_bottom_tvd=gb, bottom_tvd=BOTTOM_TVD,
            bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh=gas_bh, gas_density_mode=mode,
            temp_profile=None,
        )
        const_tbh = pressure_at_depth(
            depths, gas_top_tvd=gt, gas_bottom_tvd=gb, bottom_tvd=BOTTOM_TVD,
            bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh=gas_bh, gas_density_mode=mode,
            temp_profile=lambda d: 660.0,  # isothermal at the bottom-hole T
        )
        assert np.array_equal(base, explicit_none)
        assert np.array_equal(base, const_tbh)

    # End-to-end: a full migrate() run is identical too (None == constant-T_bh).
    kwargs = dict(
        bhp_psi=BHP, influx_bbl_bh=20.0, rho_mud_ppg=RHO_MUD,
        gas_bh_state=bh_state(), n_steps=120,
    )
    r_iso = migrate(make_sections(), PP_TABLE, FP_TABLE, **kwargs)
    r_const = migrate(
        make_sections(), PP_TABLE, FP_TABLE, temp_profile=lambda d: 660.0, **kwargs
    )
    assert r_iso.min_fp_margin_psi == r_const.min_fp_margin_psi
    assert r_iso.binding_tvd == r_const.binding_tvd
    assert [s.p_at_binding_psi for s in r_iso.steps] == \
           [s.p_at_binding_psi for s in r_const.steps]


def test_temp_profile_direction_at_shoe():
    """(b) A geothermal profile (rises with depth) moves P(shoe) DOWN vs isothermal.

    DIRECTION (documented): the isothermal baseline holds the hot bottom-hole T
    everywhere; a geothermal gradient makes the gas COOLER up-hole -> DENSER ->
    a heavier gas column -> a LOWER imposed pressure at the shoe. Equivalently
    (the converse mechanism), HOTTER gas up-hole is LIGHTER -> a HIGHER pressure
    at the shoe. Both directions are asserted here. (So the isothermal-at-BH
    assumption is the higher-shoe-pressure, safe-side case for a fracture barrier;
    a realistic geothermal correction relaxes it.)
    """
    gas_bh = _gas_bh_tuple()
    gt, gb = 4000.0, 8000.0  # a long bubble spanning the shoe

    def P(temp):
        return pressure_at_depth(
            SHOE_TVD, gas_top_tvd=gt, gas_bottom_tvd=gb, bottom_tvd=BOTTOM_TVD,
            bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh=gas_bh, temp_profile=temp,
        )

    p_iso = P(None)
    p_geo = P(GEO_TEMP)                                      # cooler up-hole
    p_hot = P(linear_temp_profile(SHOE_TVD, 720.0, BOTTOM_TVD, 660.0))  # hotter up-hole

    assert p_geo < p_iso                     # cooler/denser up-hole -> lower P(shoe)
    assert p_hot > p_iso                     # hotter/lighter up-hole -> higher P(shoe)


def test_linear_temp_profile_and_table_forms():
    """(c) The two-point linear gradient AND a full (tvd, T) table both work."""
    # Two-point linear gradient: exact straight line, extrapolated outside [shoe, TD].
    geo = linear_temp_profile(SHOE_TVD, 600.0, BOTTOM_TVD, 660.0)
    assert float(geo(SHOE_TVD)) == pytest.approx(600.0)
    assert float(geo(BOTTOM_TVD)) == pytest.approx(660.0)
    assert float(geo(7500.0)) == pytest.approx(630.0)          # midpoint
    assert float(geo(0.0)) == pytest.approx(540.0)             # linearly extrapolated
    # Vectorised.
    vals = geo(np.array([SHOE_TVD, 7500.0, BOTTOM_TVD]))
    assert np.allclose(vals, [600.0, 630.0, 660.0])
    # Degenerate anchors are rejected.
    with pytest.raises(ValueError):
        linear_temp_profile(5000.0, 600.0, 5000.0, 660.0)

    # A full (tvd, T_rankine) field-style table is accepted and interpolated. It
    # agrees with the linear form at the shared anchor depths.
    table = (np.array([SHOE_TVD, BOTTOM_TVD]), np.array([600.0, 660.0]))
    gas_bh = _gas_bh_tuple()
    gt, gb = 4000.0, 8000.0
    p_table = pressure_at_depth(
        SHOE_TVD, gas_top_tvd=gt, gas_bottom_tvd=gb, bottom_tvd=BOTTOM_TVD,
        bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh=gas_bh, temp_profile=table,
    )
    p_iso = pressure_at_depth(
        SHOE_TVD, gas_top_tvd=gt, gas_bottom_tvd=gb, bottom_tvd=BOTTOM_TVD,
        bhp_psi=BHP, rho_mud_ppg=RHO_MUD, gas_bh=gas_bh,
    )
    assert p_table < p_iso        # table encodes cooler up-hole too -> lower P(shoe)


def test_migrate_runs_end_to_end_with_temp_profile():
    """(d) A full migration runs with a temperature profile and differs from iso."""
    kwargs = dict(
        bhp_psi=BHP, influx_bbl_bh=20.0, rho_mud_ppg=RHO_MUD,
        gas_bh_state=bh_state(), n_steps=120,
    )
    r_iso = migrate(make_sections(), PP_TABLE, FP_TABLE, **kwargs)
    r_geo = migrate(make_sections(), PP_TABLE, FP_TABLE, temp_profile=GEO_TEMP, **kwargs)
    # Ran end-to-end: full trajectory, well-formed result.
    assert len(r_geo.steps) == 120
    assert isinstance(r_geo.within_envelope, bool)
    assert r_geo.steps[0].gas_top_tvd > r_geo.steps[-1].gas_top_tvd
    # And the non-isothermal correction actually moved the answer.
    assert r_geo.min_fp_margin_psi != r_iso.min_fp_margin_psi
    # A (tvd, T) table also drives migrate() end-to-end.
    table = (np.array([SHOE_TVD, BOTTOM_TVD]), np.array([600.0, 660.0]))
    r_tab = migrate(make_sections(), PP_TABLE, FP_TABLE, temp_profile=table, **kwargs)
    assert len(r_tab.steps) == 120


if __name__ == "__main__":
    r = migrate(
        make_sections(), PP_TABLE, FP_TABLE,
        bhp_psi=BHP, influx_bbl_bh=20.0, rho_mud_ppg=RHO_MUD,
        gas_bh_state=bh_state(), n_steps=200,
    )
    print(f"within_envelope   = {r.within_envelope}")
    print(f"min FP margin     = {r.min_fp_margin_psi:.1f} psi")
    print(f"binding TVD/step  = {r.binding_tvd:.0f} ft / step {r.binding_step}")
    print(f"bha_length_exceeded = {r.bha_length_exceeded}")


# --- SIDP / SICP on MigrationStep: Rada Jancic ideal-gas hand-calc -----------
# Well-control single-bubble reference (a batch consumer spec 2026-07-22):
#   MW 10 ppg, vertical TVD 10000 ft, annular capacity 0.0459 bbl/ft,
#   influx 5 bbl -> 108.9 ft gas column, gas gradient 0.1 psi/ft, SIDP 200 psi.
# Ideal gas (Z=1, isothermal), bubble at bottom:
#   SIDP = 200.00, SICP = 245.87 (at welleng's native g=0.0521).
# Rada's hand-calc gives 200.00 / 245.75 at his rounded g=0.052 -- the 0.12 psi
# is purely that gravitational-constant rounding, not a modelling difference.
def test_rada_ideal_sidp_sicp_bubble_at_bottom():
    G = G_PSI_PER_PPG_FT                       # 0.0521
    MW, TVD, CAP, INFLUX, GGRAD = 10.0, 10000.0, 0.0459, 5.0, 0.1
    bhp = 200.0 + G * MW * TVD                 # BHP consistent with SIDP=200
    rho_gas = GGRAD / G                        # 0.1 psi/ft gas gradient -> ppg
    T_bh = 560.0                               # rankine (isothermal, ideal)
    sections = [WellSection(0.0, TVD, CAP, True)]

    res = migrate(
        sections, pp=([0, TVD], [8.0, 8.0]), fp=([0, TVD], [16.0, 16.0]),
        bhp_psi=bhp, influx_bbl_bh=INFLUX, rho_mud_ppg=MW,
        gas_bh_state=(bhp, T_bh, 1.0, rho_gas), gas_density_mode="exact",
        n_steps=50, ideal_gas=True,
    )
    bottom = res.steps[0]                       # march starts with the bubble at TD
    assert bottom.gas_length_ft == pytest.approx(108.9, abs=0.5)
    assert bottom.sidp_psi == pytest.approx(200.00, abs=0.05)
    assert bottom.sicp_psi == pytest.approx(245.87, abs=0.1)   # 245.75 @ g=0.052
    # the ideal single-bubble identity SICP = SIDP + (g_mud - g_gas)*h_gas
    assert bottom.sicp_psi == pytest.approx(
        200.0 + (G * MW - GGRAD) * bottom.gas_length_ft, abs=0.05)


def test_sidp_constant_sicp_is_a_rising_schedule():
    # SIDP is position-independent -> constant across the walk. SICP rises as the
    # bubble expands (Boyle), even in ideal mode -- it is a schedule, not flat.
    G = G_PSI_PER_PPG_FT
    MW, TVD, CAP, INFLUX = 10.0, 10000.0, 0.0459, 5.0
    bhp = 200.0 + G * MW * TVD
    sections = [WellSection(0.0, TVD, CAP, True)]
    kw = dict(
        pp=([0, TVD], [8.0, 8.0]), fp=([0, TVD], [16.0, 16.0]),
        bhp_psi=bhp, influx_bbl_bh=INFLUX, rho_mud_ppg=MW,
        gas_density_mode="exact", n_steps=40,
    )
    ideal = migrate(sections, gas_bh_state=(bhp, 560.0, 1.0, 0.1 / G),
                    ideal_gas=True, **kw)
    sidp = [s.sidp_psi for s in ideal.steps]
    sicp = [s.sicp_psi for s in ideal.steps]
    assert max(sidp) - min(sidp) < 1e-6            # SIDP constant
    assert sicp[-1] > sicp[0] + 50.0               # SICP rises up the walk

    # real gas (HY Z) gives a DIFFERENT schedule but the SAME deepest value
    real = migrate(sections, gas_bh_state=(bhp, 560.0, None, None), **kw)
    assert real.steps[0].sicp_psi == pytest.approx(ideal.steps[0].sicp_psi, abs=1.0)
    real_sicp = [s.sicp_psi for s in real.steps]
    assert abs(real_sicp[-1] - sicp[-1]) > 1.0     # schedules diverge up the walk
