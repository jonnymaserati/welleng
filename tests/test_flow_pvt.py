"""Tests for welleng.flow.pvt — black-oil PVT scalar reference oracle.

Each correlation has an acceptance test. Tests are tagged in their docstring:

* ``[VALUE-VALIDATED]`` — asserts a value against an *independent* published /
  physical anchor (known air density, water density, Bg=1 identity, an inverse
  round-trip, or a cross-oracle whose other side is independently validated).
* ``[FORM-VERIFIED]``   — the correlation FORM is transcribed from a published
  source (Bellarby 2009 prints the equation); the test hand-computes the
  published field-unit equation independently and asserts the SI function
  reproduces it (validates coefficients + the SI seam).
* ``[ASSERTED]``        — original-paper coefficients transcribed faithfully and
  cross-checked by an independent hand-computation of the published equation,
  but NOT validated against a third-party worked numeric here.
"""
import math
import warnings

import pytest

from welleng.flow import pvt


# --- SI seam constants used to build independent hand-computations -----------
PSI = 6894.757293168361            # 1 psi in Pa (independent literal)
SCF_STB = 0.028316846592 / 0.158987294928
LBFT3 = 16.018463373960138         # 1 lb/ft3 in kg/m3


def _k_to_f(t_k):
    return t_k * 1.8 - 459.67


# =============================================================================
# Definitional gas properties — VALUE-VALIDATED against physical anchors
# =============================================================================
def test_rho_gas_air_standard():
    """[VALUE-VALIDATED] Air (sg=1) at 15 °C/101.325 kPa, Z=1 → 1.225 kg/m3.

    The standard density of dry air at 15 °C is a textbook constant ≈ 1.225.
    """
    assert pvt.rho_gas(101325.0, 288.15, 1.0, 1.0) == pytest.approx(1.225, abs=2e-3)


def test_bg_identity_at_standard():
    """[VALUE-VALIDATED] Bg = 1 exactly at standard conditions, Z=1."""
    assert pvt.bg(pvt.P_SC_PA, pvt.T_SC_K, 1.0) == pytest.approx(1.0, abs=1e-12)


def test_bg_boyle_scaling():
    """[VALUE-VALIDATED] Bg halves when pressure doubles at fixed T, Z."""
    b1 = pvt.bg(10e6, 350.0, 0.9)
    b2 = pvt.bg(20e6, 350.0, 0.9)
    assert b2 == pytest.approx(b1 / 2.0, rel=1e-12)


# =============================================================================
# Gas Z-factor
# =============================================================================
def test_z_hall_yarborough_matches_kernel():
    """[VALUE-VALIDATED] SI seam reproduces the KT-validated H-Y kernel exactly.

    z_hall_yarborough is only an SI seam over the existing clean-room kernel
    welleng.kick_tolerance.gas_z.hall_yarborough_z (the KT suite is its guard).
    """
    from welleng.kick_tolerance.gas_z import hall_yarborough_z

    p_pa, t_k = 20e6, 350.0
    t_pc_k, p_pc_pa = pvt.pseudo_critical_sutton(0.7)
    z_seam = pvt.z_hall_yarborough(p_pa, t_k, t_pc_k, p_pc_pa)
    z_kernel = hall_yarborough_z(
        p_pa / PSI, t_k * 1.8, t_pc_k * 1.8, p_pc_pa / PSI
    )
    assert z_seam == z_kernel          # byte-identical: seam adds no math


def test_z_dranchuk_vs_hall_yarborough():
    """[VALUE-VALIDATED] DAK agrees with the KT-validated H-Y kernel < 1%.

    Both correlations fit the Standing-Katz chart; H-Y is independently
    validated by the kick-tolerance suite, so agreement cross-validates DAK.
    """
    t_pc_k, p_pc_pa = pvt.pseudo_critical_sutton(0.7)
    for p_pa, t_k in [(10e6, 340.0), (20e6, 360.0), (30e6, 380.0)]:
        z_hy = pvt.z_hall_yarborough(p_pa, t_k, t_pc_k, p_pc_pa)
        z_dak = pvt.z_dranchuk_abou_kassem(p_pa, t_k, t_pc_k, p_pc_pa)
        assert abs(z_dak - z_hy) / z_hy < 0.01


def test_z_dranchuk_ideal_limit():
    """[VALUE-VALIDATED] DAK → 1 as pressure → 0 (ideal-gas limit)."""
    t_pc_k, p_pc_pa = pvt.pseudo_critical_sutton(0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = pvt.z_dranchuk_abou_kassem(1.0e4, 350.0, t_pc_k, p_pc_pa)
    assert z == pytest.approx(1.0, abs=2e-3)


# =============================================================================
# Pseudo-criticals — FORM-VERIFIED (against the printed Standing/Sutton fits)
# =============================================================================
def test_pseudo_critical_standing():
    """[FORM-VERIFIED] Standing (1977) fit, hand-computed in field units."""
    g = 0.7
    t_pc_r = 168.0 + 325.0 * g - 12.5 * g * g
    p_pc_psia = 677.0 + 15.0 * g - 37.5 * g * g
    t_pc_k, p_pc_pa = pvt.pseudo_critical_standing(g)
    assert t_pc_k == pytest.approx(t_pc_r / 1.8, rel=1e-9)
    assert p_pc_pa == pytest.approx(p_pc_psia * PSI, rel=1e-9)


def test_pseudo_critical_sutton():
    """[FORM-VERIFIED] Sutton (1985), SPE-14265, hand-computed in field units."""
    g = 0.8
    t_pc_r = 169.2 + 349.5 * g - 74.0 * g * g
    p_pc_psia = 756.8 - 131.0 * g - 3.6 * g * g
    t_pc_k, p_pc_pa = pvt.pseudo_critical_sutton(g)
    assert t_pc_k == pytest.approx(t_pc_r / 1.8, rel=1e-9)
    assert p_pc_pa == pytest.approx(p_pc_psia * PSI, rel=1e-9)


# =============================================================================
# Solution GOR + bubble point (Standing) — FORM-VERIFIED (Bellarby Eq 5.11/5.12)
# =============================================================================
def test_rs_standing_form():
    """[FORM-VERIFIED] Standing (1947) Rs = Bellarby (2009) Eq 5.11.

    Independent field-unit hand-computation of the printed equation.
    """
    p_pa, t_k, api, sg = 20e6, 350.0, 35.0, 0.75
    p_psia = p_pa / PSI
    t_f = _k_to_f(t_k)
    rs_scf = sg * (
        (p_psia / 18.2 + 1.4) * 10.0 ** (0.0125 * api - 0.00091 * t_f)
    ) ** 1.2048
    expect = rs_scf * SCF_STB
    assert pvt.rs_standing(p_pa, t_k, api, sg) == pytest.approx(expect, rel=1e-12)


def test_pb_standing_form():
    """[FORM-VERIFIED] Standing (1947) Pb = Bellarby (2009) Eq 5.12."""
    rs, t_k, api, sg = 130.0, 350.0, 35.0, 0.75
    rs_scf = rs / SCF_STB
    t_f = _k_to_f(t_k)
    pb_psia = 18.2 * (
        (rs_scf / sg) ** 0.83 * 10.0 ** (0.00091 * t_f - 0.0125 * api) - 1.4
    )
    assert pvt.pb_standing(rs, t_k, api, sg) == pytest.approx(pb_psia * PSI, rel=1e-12)


def test_standing_rs_pb_round_trip():
    """[VALUE-VALIDATED] pb_standing(rs_standing(p)) recovers p to < 0.05%.

    The two are the same correlation inverted; residual is the published
    1.2048 vs 0.83 exponent rounding, not an implementation error.
    """
    p_pa, t_k, api, sg = 20e6, 350.0, 35.0, 0.75
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rs = pvt.rs_standing(p_pa, t_k, api, sg)
        pb = pvt.pb_standing(rs, t_k, api, sg)
    assert pb == pytest.approx(p_pa, rel=5e-4)


# =============================================================================
# Oil FVF — Standing FORM-VERIFIED (Bellarby Eq 5.13); V&B ASSERTED
# =============================================================================
def test_bo_standing_form():
    """[FORM-VERIFIED] Standing (1947) Bo = Bellarby (2009) Eq 5.13."""
    rs, t_k, api, sg = 130.0, 350.0, 35.0, 0.75
    rs_scf = rs / SCF_STB
    t_f = _k_to_f(t_k)
    sg_oil = 141.5 / (api + 131.5)
    expect = 0.9759 + 0.000120 * (
        rs_scf * (sg / sg_oil) ** 0.5 + 1.25 * t_f
    ) ** 1.2
    assert pvt.bo_standing(rs, t_k, api, sg) == pytest.approx(expect, rel=1e-12)


def test_bo_standing_magnitude():
    """[VALUE-VALIDATED] Saturated Bo is physically 1.0-2.0 rm3/sm3."""
    assert 1.0 < pvt.bo_standing(130.0, 350.0, 35.0, 0.75) < 2.0


def test_bo_vazquez_beggs_form():
    """[ASSERTED] V&B (1980) Bo Eq 3, API>30 class, hand-computed."""
    rs, t_k, api, sg100 = 130.0, 350.0, 35.0, 0.78
    rs_scf = rs / SCF_STB
    dt = _k_to_f(t_k) - 60.0
    ratio = api / sg100
    c1, c2, c3 = 4.670e-4, 1.100e-5, 1.337e-9
    expect = 1.0 + c1 * rs_scf + c2 * dt * ratio + c3 * rs_scf * dt * ratio
    assert pvt.bo_vazquez_beggs(rs, t_k, api, sg100) == pytest.approx(expect, rel=1e-12)


def test_bo_vazquez_beggs_class_split():
    """[ASSERTED] The ≤30 / >30 °API coefficient split is applied at 30."""
    below = pvt.bo_vazquez_beggs(130.0, 350.0, 29.9, 0.78)
    above = pvt.bo_vazquez_beggs(130.0, 350.0, 30.1, 0.78)
    assert below != above


# =============================================================================
# Undersaturated oil compressibility + FVF
# =============================================================================
def test_co_vazquez_beggs_form():
    """[ASSERTED] V&B (1980) co Eq 4, hand-computed, 1/psi → 1/Pa."""
    rs, t_k, api, sg100, p_pa = 130.0, 350.0, 35.0, 0.78, 30e6
    rs_scf = rs / SCF_STB
    t_f = _k_to_f(t_k)
    p_psia = p_pa / PSI
    co_psi = (
        -1433.0 + 5.0 * rs_scf + 17.2 * t_f - 1180.0 * sg100 + 12.61 * api
    ) / (1.0e5 * p_psia)
    assert pvt.co_vazquez_beggs(rs, t_k, api, sg100, p_pa) == pytest.approx(
        co_psi / PSI, rel=1e-12
    )


def test_co_vazquez_beggs_magnitude():
    """[VALUE-VALIDATED] co ~ 1e-9 to 1e-8 /Pa (≈ 1e-5 to 1e-4 /psi)."""
    co = pvt.co_vazquez_beggs(130.0, 350.0, 35.0, 0.78, 30e6)
    assert 1e-10 < co < 1e-8


def test_bo_undersaturated_identity_and_decay():
    """[VALUE-VALIDATED] Bo=Bob at P=Pb; Bo shrinks above Pb (co>0)."""
    bob, co, pb = 1.35, 1.2e-9, 25e6
    assert pvt.bo_undersaturated(bob, co, pb, pb) == pytest.approx(bob)
    assert pvt.bo_undersaturated(bob, co, 40e6, pb) < bob


# =============================================================================
# V&B separator-gravity normaliser + Rs
# =============================================================================
def test_gas_sg_sep100_vazquez_beggs_form():
    """[ASSERTED] V&B (1980) Eq 2 separator-gravity normaliser, hand-computed."""
    sg, api, p_sep_pa, t_sep_k = 0.75, 35.0, 2e6, 320.0
    p_sep_psia = p_sep_pa / PSI
    t_sep_f = _k_to_f(t_sep_k)
    expect = sg * (
        1.0 + 5.912e-5 * api * t_sep_f * math.log10(p_sep_psia / 114.7)
    )
    assert pvt.gas_sg_sep100_vazquez_beggs(sg, api, p_sep_pa, t_sep_k) == pytest.approx(
        expect, rel=1e-12
    )


def test_rs_vazquez_beggs_form():
    """[ASSERTED] V&B (1980) Rs Eq 1, API>30 class, hand-computed."""
    p_pa, t_k, api, sg100 = 20e6, 350.0, 35.0, 0.78
    p_psia = p_pa / PSI
    t_r = t_k * 1.8
    c1, c2, c3 = 0.0178, 1.1870, 23.9310
    rs_scf = c1 * sg100 * p_psia ** c2 * math.exp(c3 * api / t_r)
    assert pvt.rs_vazquez_beggs(p_pa, t_k, api, sg100) == pytest.approx(
        rs_scf * SCF_STB, rel=1e-12
    )


# =============================================================================
# Gas viscosity (Lee et al. 1966)
# =============================================================================
def test_mu_gas_lee_form():
    """[ASSERTED] Lee-Gonzalez-Eakin (1966) SPE-1340, hand-computed."""
    t_k, sg = 350.0, 0.7
    rho = pvt.rho_gas(20e6, 350.0, 0.86, sg)
    t_r = t_k * 1.8
    m = 28.9647 * sg
    rho_g_cm3 = rho * 1e-3
    k = (9.4 + 0.02 * m) * t_r ** 1.5 / (209.0 + 19.0 * m + t_r)
    x = 3.5 + 986.0 / t_r + 0.01 * m
    y = 2.4 - 0.2 * x
    mu_cp = 1e-4 * k * math.exp(x * rho_g_cm3 ** y)
    assert pvt.mu_gas_lee(t_k, rho, sg) == pytest.approx(mu_cp * 1e-3, rel=1e-12)


def test_mu_gas_lee_magnitude():
    """[VALUE-VALIDATED] Natural-gas viscosity is ~0.01-0.03 cp."""
    rho = pvt.rho_gas(20e6, 350.0, 0.86, 0.7)
    mu = pvt.mu_gas_lee(350.0, rho, 0.7)
    assert 1e-5 < mu < 3e-5


# =============================================================================
# Oil viscosity (Beggs-Robinson dead + live; V&B undersaturated)
# =============================================================================
def test_mu_oil_dead_beggs_robinson_form():
    """[ASSERTED] Beggs-Robinson (1975) SPE-5434 dead-oil, hand-computed."""
    t_k, api = (150 + 459.67) / 1.8, 30.0
    t_f = _k_to_f(t_k)
    z = 3.0324 - 0.02023 * api
    x = 10.0 ** z * t_f ** (-1.163)
    mu_cp = 10.0 ** x - 1.0
    assert pvt.mu_oil_dead_beggs_robinson(t_k, api) == pytest.approx(
        mu_cp * 1e-3, rel=1e-12
    )


def test_mu_oil_saturated_beggs_robinson_form():
    """[ASSERTED] Beggs-Robinson (1975) live saturated-oil, hand-computed."""
    mu_dead, rs = 5.0e-3, 130.0
    rs_scf = rs / SCF_STB
    a = 10.715 * (rs_scf + 100.0) ** (-0.515)
    b = 5.44 * (rs_scf + 150.0) ** (-0.338)
    mu_cp = a * (mu_dead * 1e3) ** b
    assert pvt.mu_oil_saturated_beggs_robinson(mu_dead, rs) == pytest.approx(
        mu_cp * 1e-3, rel=1e-12
    )


def test_mu_oil_saturated_lowers_dead():
    """[VALUE-VALIDATED] Dissolved gas lowers oil viscosity (live < dead)."""
    assert pvt.mu_oil_saturated_beggs_robinson(5.0e-3, 130.0) < 5.0e-3


def test_mu_oil_undersaturated_vazquez_beggs():
    """[ASSERTED/VALUE] V&B (1980) undersaturated: μ=μ_ob at Pb, rises above."""
    mu_pb, pb = 1.0e-3, 25e6
    assert pvt.mu_oil_undersaturated_vazquez_beggs(mu_pb, pb, pb) == pytest.approx(
        mu_pb
    )
    # above Pb viscosity increases
    assert pvt.mu_oil_undersaturated_vazquez_beggs(mu_pb, 40e6, pb) > mu_pb


# =============================================================================
# Water properties (McCain 1990)
# =============================================================================
def test_bw_mccain_form():
    """[ASSERTED] McCain (1990) ch.16 water FVF, hand-computed."""
    p_pa, t_k = 20e6, 350.0
    p_psia = p_pa / PSI
    t_f = _k_to_f(t_k)
    dvwt = -1.0001e-2 + 1.33391e-4 * t_f + 5.50654e-7 * t_f ** 2
    dvwp = (
        -1.95301e-9 * p_psia * t_f - 1.72834e-13 * p_psia ** 2 * t_f
        - 3.58922e-7 * p_psia - 2.25341e-10 * p_psia ** 2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = pvt.bw_mccain(p_pa, t_k)
    assert got == pytest.approx((1.0 + dvwt) * (1.0 + dvwp), rel=1e-12)


def test_bw_mccain_magnitude():
    """[VALUE-VALIDATED] Water FVF is ~1.0-1.05 rm3/sm3."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bw = pvt.bw_mccain(20e6, 350.0)
    assert 1.0 < bw < 1.06


def test_mu_water_mccain_form():
    """[ASSERTED] McCain (1990) ch.16 atmospheric water viscosity, hand-computed."""
    t_k, sal = 350.0, 0.05        # 5 wt%
    t_f = _k_to_f(t_k)
    s = sal * 100.0
    a = 109.574 - 8.40564 * s + 0.313314 * s ** 2 + 8.72213e-3 * s ** 3
    b = (
        -1.12166 + 2.63951e-2 * s - 6.79461e-4 * s ** 2
        - 5.47119e-5 * s ** 3 + 1.55586e-6 * s ** 4
    )
    mu_cp = a * t_f ** b
    assert pvt.mu_water_mccain(t_k, sal) == pytest.approx(mu_cp * 1e-3, rel=1e-12)


def test_mu_water_mccain_salinity_raises_viscosity():
    """[VALUE-VALIDATED] Brine is more viscous than fresh water at fixed T."""
    fresh = pvt.mu_water_mccain(350.0, 0.0)
    brine = pvt.mu_water_mccain(350.0, 0.15)
    assert brine > fresh


def test_mu_water_pressure_mccain_form():
    """[ASSERTED] McCain (1990) pressure factor, hand-computed; factor ~1 at 1 atm."""
    mu_atm, p_pa = 0.5e-3, 30e6
    p_psia = p_pa / PSI
    factor = 0.9994 + 4.0295e-5 * p_psia + 3.1062e-9 * p_psia ** 2
    assert pvt.mu_water_pressure_mccain(mu_atm, p_pa) == pytest.approx(
        mu_atm * factor, rel=1e-12
    )
    # near 1 atm the factor is ~1
    assert pvt.mu_water_pressure_mccain(mu_atm, 101325.0) == pytest.approx(
        mu_atm, rel=1e-3
    )


def test_rho_water_fresh_standard():
    """[VALUE-VALIDATED] Fresh water at 60 °F/1 atm (Bw≈1) ≈ 999 kg/m3.

    McCain's standard-condition brine density ρ_sc = 62.368 lb/ft3 ≈ 999 kg/m3;
    at 60 °F the FVF Bw ≈ 1, so ρ_water ≈ ρ_sc. (At higher T the water expands,
    e.g. ~990 kg/m3 at 100 °F — physically correct, not an anchor.)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = pvt.rho_water(101325.0, (60 + 459.67) / 1.8, 0.0)
    assert rho == pytest.approx(999.0, abs=5.0)


def test_rho_water_brine_denser():
    """[VALUE-VALIDATED] Brine is denser than fresh water at fixed P,T."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = pvt.rho_water(20e6, 350.0, 0.0)
        brine = pvt.rho_water(20e6, 350.0, 0.15)
    assert brine > fresh


# =============================================================================
# Oil density (definitional) — VALUE-VALIDATED
# =============================================================================
def test_rho_oil_dead_water():
    """[VALUE-VALIDATED] Dead 10 °API oil (γo=1), Rs=0, Bo=1 → water density."""
    rho = pvt.rho_oil(0.0, 1.0, 10.0, 0.7)
    assert rho == pytest.approx(62.4 * LBFT3, rel=1e-9)   # ≈ 999.5 kg/m3


def test_rho_oil_form():
    """[FORM-VERIFIED] Bellarby (2009) Eq 5.1 mass balance, hand-computed."""
    rs, bo, api, sg = 130.0, 1.35, 35.0, 0.75
    rs_scf = rs / SCF_STB
    sg_oil = 141.5 / (api + 131.5)
    rho_lbft3 = (62.4 * sg_oil + 0.0136 * rs_scf * sg) / bo
    assert pvt.rho_oil(rs, bo, api, sg) == pytest.approx(rho_lbft3 * LBFT3, rel=1e-9)


def test_rho_oil_magnitude():
    """[VALUE-VALIDATED] Live 35 °API oil density is physically 600-900 kg/m3."""
    rho = pvt.rho_oil(130.0, 1.35, 35.0, 0.75)
    assert 600.0 < rho < 900.0


# =============================================================================
# Validity bands — WARN, not clamp, not raise
# =============================================================================
def test_rs_standing_warns_out_of_band_but_returns():
    """[BAND] Standing Rs warns (does not raise) outside API band, still returns."""
    with pytest.warns(UserWarning):
        val = pvt.rs_standing(20e6, 350.0, 70.0, 0.75)   # API 70 > 63.8
    assert val > 0.0


def test_mu_oil_dead_warns_out_of_band():
    """[BAND] Beggs-Robinson dead-oil warns outside its T/API band."""
    with pytest.warns(UserWarning):
        pvt.mu_oil_dead_beggs_robinson((60 + 459.67) / 1.8, 30.0)   # 60 °F < 70


def test_z_dranchuk_warns_out_of_band():
    """[BAND] DAK warns outside its Ppr/Tpr band, still returns finite Z."""
    t_pc_k, p_pc_pa = pvt.pseudo_critical_sutton(0.7)
    with pytest.warns(UserWarning):
        z = pvt.z_dranchuk_abou_kassem(1.0e4, 350.0, t_pc_k, p_pc_pa)
    assert math.isfinite(z)


def test_mu_gas_lee_warns_out_of_band():
    """[BAND] Lee gas-viscosity warns below its 100 °F floor."""
    with pytest.warns(UserWarning):
        pvt.mu_gas_lee((80 + 459.67) / 1.8, 100.0, 0.7)   # 80 °F < 100
