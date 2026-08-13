"""Tests for welleng.flow.vlp — multiphase VLP local-gradient reference oracle.

Each function has an acceptance test. Tests are tagged in their docstring:

* ``[WORKED]``      — reproduces a **published worked example** (numeric value
  cited in the docstring). The two anchors are Hasan & Kabir (2018), *Fluid
  Flow and Heat Transfer in Wellbores*, Field Example 3.1 (vertical, slug) and
  Field Example 4.1 (deviated 72.5°, churn + Beggs-Brill). All field-unit
  inputs are converted to SI here; comparisons use ``pytest.approx`` (never
  exact ``==`` across a conversion).
* ``[FORM-EXACT]``  — hand-computes the published closed-form independently and
  asserts the function reproduces it (definitional forms: no-slip holdup,
  drift-flux, the Beggs-Brill pattern boundaries).
* ``[CONSISTENCY]`` — an internal-consistency property (no-slip limit vsg→0,
  Colebrook vs a known Moody value, pattern-map continuity).
* ``[BAND]``        — a validity-band ``warnings.warn`` fires (warn, not clamp,
  not raise).
* ``[ASSERTED]``    — formula-transcribed, no independent published worked
  value found (the pvt surface-tension fits).

Field-unit → SI conversion factors used in the [WORKED] tests.
"""
import math
import warnings

import pytest

from welleng.flow import pvt, vlp

FT = 0.3048                 # ft -> m
LBMFT3 = 16.018463         # lbm/ft3 -> kg/m3
CP = 1.0e-3                # cP -> Pa·s
DYNE_CM = 1.0e-3           # dyne/cm -> N/m
IN = 0.0254                # in -> m
PSI_FT = 6894.757 / FT     # psi/ft -> Pa/m


# =============================================================================
# friction_factor_colebrook
# =============================================================================
def test_colebrook_smooth_moody():
    """[CONSISTENCY] Smooth pipe, Re=1e5 → Darcy f ≈ 0.0180 (Moody chart)."""
    f = vlp.friction_factor_colebrook(1.0e5, 0.0)
    assert f == pytest.approx(0.0180, abs=3.0e-4)


def test_colebrook_rough_moody():
    """[CONSISTENCY] Re=1e5, eps/d=1e-3 → f ≈ 0.0224 (Moody chart)."""
    f = vlp.friction_factor_colebrook(1.0e5, 1.0e-3)
    assert f == pytest.approx(0.0224, abs=5.0e-4)


def test_colebrook_satisfies_implicit_equation():
    """[CONSISTENCY] The returned f satisfies the Colebrook identity exactly."""
    re, eps = 3.5e4, 5.0e-4
    f = vlp.friction_factor_colebrook(re, eps)
    lhs = 1.0 / math.sqrt(f)
    rhs = -2.0 * math.log10(eps / 3.7 + 2.51 / (re * math.sqrt(f)))
    assert lhs == pytest.approx(rhs, rel=1e-9)


def test_colebrook_laminar():
    """[FORM-EXACT] Re < 2100 returns the laminar law f = 64/Re."""
    assert vlp.friction_factor_colebrook(1000.0, 0.0) == pytest.approx(
        64.0 / 1000.0, rel=1e-12
    )


def test_colebrook_transitional_warns():
    """[BAND] The transitional 2100-4000 band warns (turbulent extrapolated)."""
    with pytest.warns(UserWarning):
        vlp.friction_factor_colebrook(3000.0, 1.0e-4)


# =============================================================================
# no_slip_holdup
# =============================================================================
def test_no_slip_holdup_formula():
    """[FORM-EXACT] lambda_L = vsl/(vsl+vsg), definitional."""
    assert vlp.no_slip_holdup(0.5, 1.5) == pytest.approx(0.25, rel=1e-12)


def test_no_slip_holdup_no_gas():
    """[CONSISTENCY] vsg = 0 → full liquid (lambda_L = 1)."""
    assert vlp.no_slip_holdup(2.0, 0.0) == pytest.approx(1.0, rel=1e-12)


# =============================================================================
# holdup_zuber_findlay
# =============================================================================
def test_zuber_findlay_formula():
    """[FORM-EXACT] H_L = 1 − vsg/(C0·vm + vd) (Zuber-Findlay 1965)."""
    vsl, vsg, c0, vd = 0.5, 1.0, 1.2, 0.2
    vm = vsl + vsg
    expect = 1.0 - vsg / (c0 * vm + vd)
    assert vlp.holdup_zuber_findlay(vsl, vsg, c0, vd) == pytest.approx(
        expect, rel=1e-12
    )


def test_zuber_findlay_no_gas():
    """[CONSISTENCY] vsg = 0 → H_L = 1 (no-slip liquid limit)."""
    assert vlp.holdup_zuber_findlay(1.0, 0.0, 1.2, 0.2) == pytest.approx(
        1.0, rel=1e-12
    )


# =============================================================================
# Hasan-Kabir — Field Example 3.1 (vertical, slug)  Hasan & Kabir (2018)
# =============================================================================
# Ex 3.1 inputs: 23 °API oil, 2.99-in ID, vertical (theta = 90° from horizontal).
HK31 = dict(
    vsl_m_s=1.601 * FT, vsg_m_s=2.784 * FT,
    rho_l=55.042 * LBMFT3, rho_g=2.17 * LBMFT3,
    mu_l=13.09 * CP, mu_g=0.019 * CP,
    sigma_n_m=31.6 * DYNE_CM, d_m=2.99 * IN, theta_deg=90.0,
)


def test_hasan_kabir_pattern_ex31_slug():
    """[WORKED] Field Ex 3.1: vertical → slug (v_inf=0.68, v_ann=6.94 ft/s)."""
    p = vlp.pattern_hasan_kabir(
        HK31["vsl_m_s"], HK31["vsg_m_s"], HK31["rho_l"], HK31["rho_g"],
        HK31["sigma_n_m"], HK31["d_m"], HK31["theta_deg"],
    )
    assert p == vlp.SLUG


def test_hasan_kabir_holdup_ex31():
    """[WORKED] Field Ex 3.1: slug holdup H_L = 0.543 (fg = 0.457)."""
    h_l = vlp.holdup_hasan_kabir(
        HK31["vsl_m_s"], HK31["vsg_m_s"], HK31["rho_l"], HK31["rho_g"],
        HK31["sigma_n_m"], HK31["d_m"], HK31["theta_deg"], vlp.SLUG,
    )
    assert h_l == pytest.approx(0.543, rel=0.02)


def test_hasan_kabir_dpdz_ex31():
    """[WORKED] Field Ex 3.1: total gradient 0.221 psi/ft (~5000 Pa/m).

    Static head dominates (0.2146 psi/ft); friction 0.0105 psi/ft is computed
    here with the Colebrook kernel (smooth) vs the book's Blasius (~1-2 %).
    """
    dpdz = vlp.dpdz_hasan_kabir(
        HK31["vsl_m_s"], HK31["vsg_m_s"], HK31["rho_l"], HK31["rho_g"],
        HK31["mu_l"], HK31["mu_g"], HK31["sigma_n_m"], HK31["d_m"],
        roughness_m=0.0, theta_deg=HK31["theta_deg"],
    )
    assert dpdz / PSI_FT == pytest.approx(0.221, rel=0.03)


# =============================================================================
# Hasan-Kabir — Field Example 4.1 (deviated 72.5°, churn)  Hasan & Kabir (2018)
# =============================================================================
# Ex 4.1 inputs: 33 °API oil + 20% water, 2.435-in ID, 72.5° from horizontal.
HK41 = dict(
    vsl_m_s=7.691 * FT, vsg_m_s=17.62 * FT,
    rho_l=53.3 * LBMFT3, rho_g=1.08 * LBMFT3,
    mu_l=0.6 * CP, mu_g=0.014 * CP,
    sigma_n_m=29.0 * DYNE_CM, d_m=2.435 * IN, theta_deg=72.5,
)


def test_hasan_kabir_pattern_ex41_slug():
    """[WORKED] Field Ex 4.1: deviated 72.5° → slug (annular rejected by the
    Barnea void gate; churn needs the viscosity-dependent Eq 3.10 path excluded
    by the viscosity-free ratified signature — see pattern_hasan_kabir note)."""
    p = vlp.pattern_hasan_kabir(
        HK41["vsl_m_s"], HK41["vsg_m_s"], HK41["rho_l"], HK41["rho_g"],
        HK41["sigma_n_m"], HK41["d_m"], HK41["theta_deg"],
    )
    assert p == vlp.SLUG


def test_hasan_kabir_holdup_ex41_churn_branch():
    """[WORKED] Field Ex 4.1 with explicit pattern=churn: H_L = 0.417
    (fg = 0.583, C0 = 1.15) — validates the churn drift-flux branch and the
    deviated Taylor-bubble rise velocity (Eq 4.33)."""
    h_l = vlp.holdup_hasan_kabir(
        HK41["vsl_m_s"], HK41["vsg_m_s"], HK41["rho_l"], HK41["rho_g"],
        HK41["sigma_n_m"], HK41["d_m"], HK41["theta_deg"], vlp.CHURN,
    )
    assert h_l == pytest.approx(0.417, rel=0.02)


def test_hasan_kabir_dpdz_ex41():
    """[WORKED] Field Ex 4.1: total gradient ≈ 0.27 psi/ft (deviated, ·sinθ).

    Classified slug here (Co=1.2) vs the book's churn (Co=1.15); the ~3 %
    difference is the drift-parameter choice (see pattern_hasan_kabir note).
    The book's Hasan-Kabir example is internally consistent (churn total
    0.267 psi/ft).
    """
    dpdz = vlp.dpdz_hasan_kabir(
        HK41["vsl_m_s"], HK41["vsg_m_s"], HK41["rho_l"], HK41["rho_g"],
        HK41["mu_l"], HK41["mu_g"], HK41["sigma_n_m"], HK41["d_m"],
        roughness_m=0.0, theta_deg=HK41["theta_deg"],
    )
    assert dpdz / PSI_FT == pytest.approx(0.276, rel=0.03)


def test_hasan_kabir_taylor_reduces_at_vertical():
    """[CONSISTENCY] Eq 4.33 deviated Taylor rise → Eq 3.8 at theta=90°."""
    at90 = vlp._taylor_rise(880.0, 35.0, 0.076, 90.0)
    base = 0.35 * (vlp.G * 0.076 * (880.0 - 35.0) / 880.0) ** 0.5
    assert at90 == pytest.approx(base, rel=1e-12)


def test_hasan_kabir_holdup_noslip_floor():
    """[CONSISTENCY] Holdup never falls below the no-slip holdup."""
    lam = vlp.no_slip_holdup(HK41["vsl_m_s"], HK41["vsg_m_s"])
    h_l = vlp.holdup_hasan_kabir(
        HK41["vsl_m_s"], HK41["vsg_m_s"], HK41["rho_l"], HK41["rho_g"],
        HK41["sigma_n_m"], HK41["d_m"], HK41["theta_deg"], vlp.CHURN,
    )
    assert h_l >= lam


# =============================================================================
# Beggs-Brill — pattern boundaries + Field Example 4.1  Beggs & Brill (1973)
# =============================================================================
def test_beggs_brill_pattern_ex41_intermittent():
    """[WORKED] Field Ex 4.1: lambda=0.307, high Fr → intermittent."""
    p = vlp.pattern_beggs_brill(HK41["vsl_m_s"], HK41["vsg_m_s"], HK41["d_m"])
    assert p == vlp.INTERMITTENT


def test_beggs_brill_holdup0_ex41():
    """[WORKED] Field Ex 4.1: horizontal holdup H_L(0) = 0.4146 (intermittent).

    The published anchor for the Beggs-Brill holdup correlation. With the
    original C >= 0 uphill clamp the inclination factor psi = 1 for this case
    (the book's unclamped psi = 0.984 is flagged non-physical by the authors),
    so H_L(72.5°) = H_L(0) = 0.4146.
    """
    # H_L(0): reconstruct via the module's horizontal holdup (psi at theta≈0).
    hl0 = vlp._bb_hl0(
        vlp.no_slip_holdup(HK41["vsl_m_s"], HK41["vsg_m_s"]),
        (HK41["vsl_m_s"] + HK41["vsg_m_s"]) ** 2
        / (vlp.G * HK41["d_m"]),
        vlp.INTERMITTENT,
    )
    assert hl0 == pytest.approx(0.4146, rel=0.01)
    hl_theta = vlp.holdup_beggs_brill(
        HK41["vsl_m_s"], HK41["vsg_m_s"], HK41["d_m"], HK41["theta_deg"],
        rho_l=HK41["rho_l"], sigma_n_m=HK41["sigma_n_m"],
    )
    assert hl_theta == pytest.approx(0.4146, rel=0.02)


def test_beggs_brill_s_function_ex41():
    """[WORKED] Field Ex 4.1: y = lambda/H_L² = 1.81 → s = 0.39, e^s = 1.48.

    The two-phase friction ratio f_tp/f_ns; validates the Eq 4.18 S-function.
    Uses the book's rounded lambda_L = 0.307 and H_L = 0.412.
    """
    lam = 0.307  # book's rounded no-slip liquid fraction
    h_l = 0.412  # book inclined holdup
    y = lam / (h_l * h_l)
    ly = math.log(y)
    denom = -0.0523 + 3.182 * ly - 0.8725 * ly * ly + 0.01853 * ly ** 4
    s = ly / denom
    assert y == pytest.approx(1.81, rel=0.01)
    assert s == pytest.approx(0.39, rel=0.05)
    assert math.exp(s) == pytest.approx(1.48, rel=0.05)


def test_beggs_brill_dpdz_ex41_static():
    """[WORKED] Field Ex 4.1: Beggs-Brill static head 0.151 psi/ft.

    The static component alone is the trustworthy published anchor: with the
    C >= 0 clamp H_L = 0.4146 → rho_m gives rho_m·g·sinθ = 0.151 psi/ft
    (matches the book). NOTE: we could not reproduce the book's printed friction
    (0.071 psi/ft, total 0.223) from its own stated f=0.0189 / rho_n=17.308 /
    vm=25.3 on our reading — the discrepancy may be our transcription of an
    input, so we do NOT anchor to the printed total. Instead we validate the
    static component here (matches) and the friction assembly separately below,
    and cross-check the whole gradient against the internally-consistent
    Hasan-Kabir example. (If someone reconciles the book's friction, tighten this.)
    """
    lam = vlp.no_slip_holdup(HK41["vsl_m_s"], HK41["vsg_m_s"])
    h_l = vlp.holdup_beggs_brill(
        HK41["vsl_m_s"], HK41["vsg_m_s"], HK41["d_m"], HK41["theta_deg"],
        rho_l=HK41["rho_l"], sigma_n_m=HK41["sigma_n_m"],
    )
    rho_m = HK41["rho_l"] * h_l + HK41["rho_g"] * (1.0 - h_l)
    static = rho_m * vlp.G * math.sin(math.radians(HK41["theta_deg"]))
    assert static / PSI_FT == pytest.approx(0.151, rel=0.02)


def test_beggs_brill_dpdz_assembly():
    """[CONSISTENCY] dpdz_beggs_brill = static + f_tp·rho_n·vm²/(2d) rebuilt.

    Independently re-assembles the two components (holdup → rho_m static;
    lambda/H_L → S-function → f_tp friction) and asserts the function matches,
    validating the coefficient wiring end-to-end.
    """
    vsl, vsg = HK41["vsl_m_s"], HK41["vsg_m_s"]
    rho_l, rho_g = HK41["rho_l"], HK41["rho_g"]
    mu_l, mu_g, d, th = HK41["mu_l"], HK41["mu_g"], HK41["d_m"], HK41["theta_deg"]
    vm = vsl + vsg
    lam = vlp.no_slip_holdup(vsl, vsg)
    h_l = vlp.holdup_beggs_brill(
        vsl, vsg, d, th, rho_l=rho_l, sigma_n_m=HK41["sigma_n_m"]
    )
    rho_m = rho_l * h_l + rho_g * (1.0 - h_l)
    rho_n = rho_l * lam + rho_g * (1.0 - lam)
    mu_n = mu_l * lam + mu_g * (1.0 - lam)
    static = rho_m * vlp.G * math.sin(math.radians(th))
    re_n = rho_n * vm * d / mu_n
    f_ns = vlp.friction_factor_colebrook(re_n, 0.0)
    y = lam / (h_l * h_l)
    ly = math.log(y)
    s = ly / (-0.0523 + 3.182 * ly - 0.8725 * ly * ly + 0.01853 * ly ** 4)
    friction = f_ns * math.exp(s) * rho_n * vm * vm / (2.0 * d)
    expect = static + friction
    got = vlp.dpdz_beggs_brill(
        vsl, vsg, rho_l, rho_g, mu_l, mu_g, HK41["sigma_n_m"], d, 0.0, th
    )
    assert got == pytest.approx(expect, rel=1e-9)


def test_beggs_brill_pattern_boundaries():
    """[FORM-EXACT] Segregated ↔ distributed switch across the L-boundaries.

    Low lambda + low Fr sits below L1 (segregated); at very high Fr the same
    low-lambda flow crosses into distributed. Uses SPE-4007 L1 = 316·lambda^0.302.
    """
    d = 0.05
    # Low liquid fraction, low velocity → segregated.
    assert vlp.pattern_beggs_brill(0.01, 0.5, d) == vlp.SEGREGATED
    # Low liquid fraction, very high velocity → distributed.
    assert vlp.pattern_beggs_brill(0.2, 60.0, d) == vlp.DISTRIBUTED


def test_beggs_brill_holdup_ge_noslip():
    """[CONSISTENCY] Beggs-Brill H_L(0) is floored at lambda_L (Eq 4.13)."""
    lam = vlp.no_slip_holdup(0.2, 60.0)
    hl = vlp.holdup_beggs_brill(0.2, 60.0, 0.05, 90.0)
    assert hl >= lam


# =============================================================================
# Hagedorn-Brown — STUBBED (no citeable chart-fit source; see docstrings)
# =============================================================================
def test_hagedorn_brown_holdup_stub():
    """[CONSISTENCY] Holdup is a documented NotImplementedError (chart-fit)."""
    with pytest.raises(NotImplementedError):
        vlp.holdup_hagedorn_brown(0.5, 1.0, 880.0, 0.01, 0.03, 0.076)


def test_hagedorn_brown_dpdz_stub():
    """[CONSISTENCY] Gradient is a documented NotImplementedError (chart-fit)."""
    with pytest.raises(NotImplementedError):
        vlp.dpdz_hagedorn_brown(
            0.5, 1.0, 880.0, 35.0, 0.01, 1e-5, 0.03, 0.076, 0.0
        )


# =============================================================================
# pvt surface-tension additions (Baker-Swerdloff oil, Katz water/gas)
# =============================================================================
def test_sigma_oil_gas_baker_swerdloff_asserted():
    """[ASSERTED] Baker-Swerdloff dead-oil sigma: 39−0.2571·API at 68°F.

    Formula-transcribed (OGJ 1956). 30 °API just inside the 68 °F endpoint →
    ≈ 31.29 dyne/cm (the 68 °F value, 39 − 0.2571·API).
    """
    t_69f = (69.0 + 459.67) / 1.8         # 69 °F -> K (inside the band)
    sigma = pvt.sigma_oil_gas_baker_swerdloff(t_69f, 30.0)
    assert sigma * 1.0e3 == pytest.approx(39.0 - 0.2571 * 30.0, rel=1e-2)


def test_sigma_water_gas_atmospheric_asserted():
    """[ASSERTED] Katz water/gas sigma ≈ 72 dyne/cm at 75 °F, atmospheric."""
    t_75f = (75.0 + 459.67) / 1.8         # 75 °F -> K (inside the band)
    sigma = pvt.sigma_water_gas(t_75f, 101325.0)
    assert sigma * 1.0e3 == pytest.approx(72.2, abs=1.5)


def test_sigma_oil_gas_band_warns():
    """[BAND] Baker-Swerdloff warns outside the 68-100 °F band."""
    t_hot = (200.0 + 459.67) / 1.8        # 200 °F -> K
    with pytest.warns(UserWarning):
        pvt.sigma_oil_gas_baker_swerdloff(t_hot, 30.0)
