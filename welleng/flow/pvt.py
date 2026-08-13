"""Black-oil PVT scalar reference oracle — classic correlations, as-published.

This module implements the standard black-oil PVT correlations (solution
gas-oil ratio, bubble-point pressure, oil/gas/water formation volume factors,
densities, viscosities and the real-gas Z-factor) as **scalar reference
forms** — pure ``float`` in, ``float`` out. It is welleng's open-core
*correctness oracle*: the vectorised / GPU forms that live in the commercial
layer are parity-gated against these functions (bit-identical, or a documented
tolerance where an implicit solve differs in its Newton tail).

Validation status (2026-08-13). Validated two ways: (1) exact-vs-published-formula
unit tests, and (2) a real-data gate against the operator-tuned Volve Eclipse PVT
tables (two fluid regions), run by welleng-production. The volumetric/Z chain
(Rs/Bo/Bg, Standing & Vazquez-Beggs, Hall-Yarborough & Dranchuk-Abou-Kassem, Bw)
lands within the ±5-8% correlation-class band vs the field PVT, with DAK and HY
mutually agreeing to ~0.2%; viscosities (Beggs-Robinson, Lee, McCain water) carry
wider, documented correlation-accuracy bands. No coefficient/unit errors surfaced.
So the earlier per-function ``[ASSERTED]`` (formula-transcribed, not yet field-
checked) tags are now superseded by that real-data check for the volumetric/Z set.

Design contract
---------------
* **Pure float in / float out.** No arrays, no fast paths — that is the
  private/api layer's job.
* **SI at the interface, classic as-published internally.** The public
  signature is strict SI: pressure **Pa (absolute)**, temperature **K
  (absolute)**, density **kg/m3**, viscosity **Pa·s**, solution GOR
  **sm3/sm3**, FVF **rm3/sm3**; API gravity is a plain float (°API), gas
  specific gravity ``gas_sg`` is relative to air (air = 1.0). Each classic is
  evaluated in its *published field units* (psia / °R / °F / cp / scf-STB) and
  converted SI↔field at the seam. The field↔SI conversions **reuse
  :mod:`welleng.units`** (one registry for the whole library); only the
  oilfield volume-*ratio* factor scf/STB→sm3/sm3, which pint does not carry,
  is defined locally (see :data:`SCF_STB_TO_SM3_SM3`).
* **Validity bands warn, never clamp, never raise.** Each function emits a
  :func:`warnings.warn` when evaluated outside its published band (band stated
  in the docstring) and returns the extrapolated value — matching the
  :mod:`welleng.kick_tolerance.gas_z` house pattern.
* **Citation in every docstring:** source + specific equation/table.

Standard conditions
-------------------
SI standard **15 °C (288.15 K) / 101.325 kPa** (matches Volve / Eclipse
METRIC). All ``sm3`` are at these conditions.

⚠️ **The scf→sm3 basis is a temperature basis, not a pure volume factor.**
A standard cubic foot (scf) is defined at **60 °F**, a standard cubic metre
(sm3) at **15 °C** — these are *different* reference temperatures (60 °F =
15.56 °C). :data:`SCF_STB_TO_SM3_SM3` is the pure *geometric* volume ratio
(ft³/bbl). For a **ratio** quantity (Rs = gas-vol / oil-vol, both taken to
standard) the 60 °F↔15 °C basis difference **largely cancels** — the gas and
oil standard volumes both shift with the reference-temperature change — so Rs
in sm3/sm3 uses this geometric factor directly. For an **absolute** standard
gas volume (e.g. Bg), the basis difference does **not** cancel; the definitional
:func:`bg` therefore fixes its reference at the SI 15 °C / 101.325 kPa state
explicitly rather than routing through scf.

References
----------
Standing, M. B. (1947). *Volumetric and Phase Behavior of Oil Field
    Hydrocarbon Systems*; the 22-sample California solution-GOR / bubble-point
    / FVF correlations. (Forms cross-checked against Bellarby (2009) Eqs 5.11,
    5.12, 5.13, 5.1.)
Vazquez, M. & Beggs, H. D. (1980). *Correlations for Fluid Physical Property
    Prediction.* JPT 32(6): 968-970 (SPE-6719).
Beggs, H. D. & Robinson, J. R. (1975). *Estimating the Viscosity of Crude Oil
    Systems.* JPT 27(9): 1140-1141 (SPE-5434).
Lee, A. L., Gonzalez, M. H. & Eakin, B. E. (1966). *The Viscosity of Natural
    Gases.* JPT 18(8): 997-1000 (SPE-1340).
Hall, K. R. & Yarborough, L. (1973). *A new equation of state for Z-factor
    calculations.* Oil & Gas Journal 71(25): 82-92.
Dranchuk, P. M. & Abou-Kassem, J. H. (1975). *Calculation of Z Factors for
    Natural Gases Using Equations of State.* JCPT 14(3): 34-36.
Sutton, R. P. (1985). *Compressibility Factors for High-Molecular-Weight
    Reservoir Gases.* SPE-14265.
McCain, W. D. (1990). *The Properties of Petroleum Fluids*, 2nd ed.,
    PennWell — water FVF / viscosity / density correlations (ch. 16).
Bellarby, J. (2009). *Well Completion Design*, Elsevier — §5.1 black-oil
    models (cross-check reference for the Standing forms).
Baker, O. & Swerdloff, W. (1956). *Calculations of Surface Tension-3:
    Calculations of Surface Tension Parachor Values.* Oil & Gas Journal 43(12)
    — dead-oil/gas surface tension vs T and API.
Katz, D. L. et al. (1959). *Handbook of Natural Gas Engineering*, McGraw-Hill
    — the water/gas interfacial-tension fit (also attributed to Ramey).
"""
from __future__ import annotations

import math
import warnings

from ..units import Units

# Single boundary converter (pint at setup, arithmetic at runtime); reuses the
# one welleng registry — NOT a parallel units module. All field↔SI seam
# conversions route through this.
_U = Units()

# --- SI seam helpers (field <-> SI via welleng.units) ------------------------
#: 1 psi in Pa (exact pint value) — cached for the compressibility 1/psi->1/Pa.
_PSI_IN_PA: float = _U.convert(1.0, "psi", "pascal")


def _pa_to_psi(p_pa: float) -> float:
    return _U.convert(p_pa, "pascal", "psi")


def _psi_to_pa(p_psia: float) -> float:
    return _U.convert(p_psia, "psi", "pascal")


def _k_to_rankine(t_k: float) -> float:
    return _U.convert(t_k, "kelvin", "degR")


def _k_to_f(t_k: float) -> float:
    """Absolute K -> °F (via °R; °R and K share absolute zero)."""
    return _U.convert(t_k, "kelvin", "degR") - 459.67


def _cp_to_pas(mu_cp: float) -> float:
    return _U.convert(mu_cp, "cP", "pascal*second")


def _pas_to_cp(mu_pa_s: float) -> float:
    return _U.convert(mu_pa_s, "pascal*second", "cP")


# --- oilfield volume-ratio factor (NOT in pint; the one flow-local constant) --
#: scf/STB -> sm3/sm3, the pure geometric volume ratio ft³/bbl.
#: 1 scf = 1 ft³ = 0.028316846592 m³; 1 STB = 42 US gal = 0.158987294928 m³.
#: RATIO quantity — the 60 °F (scf) ↔ 15 °C (sm3) basis difference largely
#: cancels (see module docstring). = 0.178107606679…
SCF_STB_TO_SM3_SM3: float = 0.028316846592 / 0.158987294928

# --- physical / standard-condition constants ---------------------------------
M_AIR_G_PER_MOL: float = 28.9647          # dry-air molar mass [g/mol]
R_GAS: float = 8.314462618                # universal gas constant [J/(mol·K)]
P_SC_PA: float = 101325.0                 # SI standard pressure [Pa]
T_SC_K: float = 288.15                    # SI standard temperature 15 °C [K]


def _api_to_sg_oil(api: float) -> float:
    """Oil specific gravity (water = 1) from °API — Bellarby (2009) Eq 5.2."""
    return 141.5 / (api + 131.5)


# =============================================================================
# Pseudo-critical properties (inputs to the Z-factors)
# =============================================================================
# NB: the underlying gas-gravity fits live ONCE in kick_tolerance.gas_z — these
# wrap them with an SI seam rather than re-typing the coefficients.
from ..kick_tolerance.gas_z import (  # noqa: E402  (after seam helpers by design)
    hall_yarborough_z as _hy_z_field,
    standing_pseudo_criticals as _standing_pc_field,
    sutton_pseudo_criticals as _sutton_pc_field,
)


def pseudo_critical_sutton(gas_sg: float) -> tuple[float, float]:
    """Pseudo-critical (T_pc [K], P_pc [Pa]) from gas specific gravity.

    Sutton (1985), SPE-14265 — the wider-gravity fit (roughly ``gas_sg``
    0.57-1.68), preferred for heavier/associated gases::

        T_pc = 169.2 + 349.5·g − 74.0·g²   [°R]
        P_pc = 756.8 − 131.0·g −  3.6·g²   [psia]

    Classic as-published; ``g`` is relative to air (air = 1.0). No sour-gas
    (H2S/CO2 Wichert-Aziz) correction — pass explicit pseudo-criticals for
    sour gases. Coefficients sourced from
    :func:`welleng.kick_tolerance.gas_z.sutton_pseudo_criticals`.
    """
    t_pc_r, p_pc_psia = _sutton_pc_field(gas_sg)
    return _U.convert(t_pc_r, "degR", "kelvin"), _psi_to_pa(p_pc_psia)


def pseudo_critical_standing(gas_sg: float) -> tuple[float, float]:
    """Pseudo-critical (T_pc [K], P_pc [Pa]) from gas specific gravity.

    Standing (1977) sweet-gas curves — the simpler classic (valid ``gas_sg``
    ~0.55-1.0)::

        T_pc = 168 + 325·g − 12.5·g²   [°R]
        P_pc = 677 +  15·g − 37.5·g²   [psia]

    Classic as-published; ``g`` relative to air. No sour-gas correction.
    Coefficients sourced from
    :func:`welleng.kick_tolerance.gas_z.standing_pseudo_criticals`.
    """
    t_pc_r, p_pc_psia = _standing_pc_field(gas_sg)
    return _U.convert(t_pc_r, "degR", "kelvin"), _psi_to_pa(p_pc_psia)


# =============================================================================
# Gas Z-factor
# =============================================================================
def z_hall_yarborough(
    p_pa: float, t_k: float, t_pc_k: float, p_pc_pa: float
) -> float:
    """Real-gas Z-factor [-]. Hall & Yarborough (1973), OGJ 71(25): 82-92.

    Implicit reduced-density solve (Newton with analytic derivative). This is
    an **SI seam over the EXISTING clean-room kernel**
    :func:`welleng.kick_tolerance.gas_z.hall_yarborough_z` — there is exactly
    one Hall-Yarborough kernel in welleng and it is unchanged (byte-identical;
    the kick-tolerance suite is its numeric guard). The seam only converts
    Pa→psia, K→°R for the pseudo-criticals and state.

    Band: 0.1 ≤ Ppr ≤ 24, 1.15 ≤ Tpr ≤ 3.0 — the kernel warns (does not clamp)
    outside it. Classic as-published.
    """
    return _hy_z_field(
        _pa_to_psi(p_pa),
        _k_to_rankine(t_k),
        _k_to_rankine(t_pc_k),
        _pa_to_psi(p_pc_pa),
    )


# Dranchuk & Abou-Kassem (1975) 11-constant Starling-EOS fit of Standing-Katz.
_DAK_A = (
    0.3265, -1.0700, -0.5339, 0.01569, -0.05165,
    0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210,
)
_DAK_MAX_ITER = 100
_DAK_TOL = 1.0e-12


def z_dranchuk_abou_kassem(
    p_pa: float, t_k: float, t_pc_k: float, p_pc_pa: float
) -> float:
    """Real-gas Z-factor [-]. Dranchuk & Abou-Kassem (1975), JCPT 14(3): 34-36.

    The 11-constant Starling-equation-of-state fit of the Standing-Katz chart,
    solved for reduced density ``rr`` by Newton-Raphson (analytic derivative)::

        Z = 1 + C1·rr + C2·rr² − C3·rr⁵
              + A10(1 + A11·rr²)(rr²/Tpr³)·exp(−A11·rr²)
        rr = 0.27·Ppr / (Z·Tpr)

    with the published constants A1..A11. Reduced quantities (Tpr = T/Tpc,
    Ppr = P/Ppc) are ratios, so no field-unit seam is needed. Band:
    0.2 ≤ Ppr ≤ 30, 1.0 < Tpr ≤ 3.0 (warn outside). Classic as-published.
    """
    tpr = t_k / t_pc_k
    ppr = p_pa / p_pc_pa
    if not (0.2 <= ppr <= 30.0 and 1.0 < tpr <= 3.0):
        warnings.warn(
            f"Dranchuk-Abou-Kassem Z evaluated outside its validity band "
            f"(Ppr={ppr:.3g} in [0.2, 30], Tpr={tpr:.3g} in (1.0, 3.0]); "
            f"the returned Z is extrapolated.",
            stacklevel=2,
        )
    a = _DAK_A
    ti = 1.0 / tpr
    c1 = a[0] + a[1] * ti + a[2] * ti ** 3 + a[3] * ti ** 4 + a[4] * ti ** 5
    c2 = a[5] + a[6] * ti + a[7] * ti ** 2
    c3 = a[8] * (a[6] * ti + a[7] * ti ** 2)
    a10, a11 = a[9], a[10]
    tpr3 = tpr ** 3

    # Newton on F(rr) = Z_eos(rr) − 0.27·Ppr/(rr·Tpr) = 0.
    rr = 0.27 * ppr / tpr            # seed at Z = 1
    for _ in range(_DAK_MAX_ITER):
        e = math.exp(-a11 * rr * rr)
        c4 = a10 * (1.0 + a11 * rr * rr) * (rr * rr / tpr3) * e
        z_eos = 1.0 + c1 * rr + c2 * rr ** 2 - c3 * rr ** 5 + c4
        z_def = 0.27 * ppr / (rr * tpr)
        f = z_eos - z_def
        if abs(f) < _DAK_TOL:
            break
        dc4 = (a10 / tpr3) * e * (
            2.0 * rr + 2.0 * a11 * rr ** 3 - 2.0 * a11 * a11 * rr ** 5
        )
        df = (
            c1 + 2.0 * c2 * rr - 5.0 * c3 * rr ** 4 + dc4
            + 0.27 * ppr / (rr * rr * tpr)
        )
        rr -= f / df
        if rr <= 0.0:
            rr = 1.0e-8
    return 0.27 * ppr / (rr * tpr)


# =============================================================================
# Gas properties (definitional — real-gas law)
# =============================================================================
def bg(p_pa: float, t_k: float, z: float) -> float:
    """Gas formation volume factor [rm3/sm3]. Definitional (real-gas law).

    Bg = V_res / V_std = Z·T·P_sc / (P·T_sc·Z_sc), with the SI standard state
    15 °C / 101.325 kPa and Z_sc = 1 (documented). At standard state Bg = 1.
    The reference is fixed at the SI standard directly (not via scf) because an
    absolute standard gas volume does NOT carry the 60 °F↔15 °C cancellation.
    """
    return z * t_k * P_SC_PA / (p_pa * T_SC_K)


def rho_gas(p_pa: float, t_k: float, z: float, gas_sg: float) -> float:
    """Gas density [kg/m3]. Definitional real-gas law ρ = P·M/(Z·R·T).

    Apparent molar mass M = 28.9647·gas_sg g/mol (air molar mass × specific
    gravity). At 15 °C / 101.325 kPa, Z = 1, gas_sg = 1 this returns the
    standard density of air ≈ 1.225 kg/m3.
    """
    m_kg_per_mol = M_AIR_G_PER_MOL * gas_sg * 1.0e-3
    return p_pa * m_kg_per_mol / (z * R_GAS * t_k)


def mu_gas_lee(t_k: float, rho_gas_kg_m3: float, gas_sg: float) -> float:
    """Gas viscosity [Pa·s]. Lee, Gonzalez & Eakin (1966), SPE-1340.

    Empirical function of temperature, gas density and molar mass::

        K = (9.4 + 0.02·M)·T^1.5 / (209 + 19·M + T)
        X = 3.5 + 986/T + 0.01·M
        Y = 2.4 − 0.2·X
        μ_g = 1e-4 · K · exp(X · ρ^Y)          [cp]

    with T in °R, M = 28.9647·gas_sg [lb/lbmol], and ρ the gas density in
    **g/cm3**. No non-hydrocarbon (N2/CO2/H2S) corrections; accuracy degrades
    at high specific gravity. Band: ~100-8000 psia, 100-340 °F (warn on T
    outside). Classic as-published.
    """
    t_r = _k_to_rankine(t_k)
    t_f = t_r - 459.67
    if not (100.0 <= t_f <= 340.0):
        warnings.warn(
            f"Lee gas-viscosity evaluated outside its validity band "
            f"(T={t_f:.4g} °F in [100, 340]); the returned μ is extrapolated.",
            stacklevel=2,
        )
    m = M_AIR_G_PER_MOL * gas_sg              # apparent molar mass [lb/lbmol]
    rho_g_cm3 = rho_gas_kg_m3 * 1.0e-3        # kg/m3 -> g/cm3 (SI prefix)
    k = (9.4 + 0.02 * m) * t_r ** 1.5 / (209.0 + 19.0 * m + t_r)
    x = 3.5 + 986.0 / t_r + 0.01 * m
    y = 2.4 - 0.2 * x
    mu_cp = 1.0e-4 * k * math.exp(x * rho_g_cm3 ** y)
    return _cp_to_pas(mu_cp)


# =============================================================================
# Solution GOR + bubble point
# =============================================================================
def rs_standing(p_pa: float, t_k: float, api: float, gas_sg: float) -> float:
    """Solution GOR [sm3/sm3]. Standing (1947); Bellarby (2009) Eq 5.11.

        Rs = γg·[(p/18.2 + 1.4)·10^(0.0125·API − 0.00091·T)]^1.2048

    p in psia, T in °F, Rs in scf/STB (converted to sm3/sm3). Saturated form:
    returns Rs at the given p; a caller caps it at Rs(Pb) above the bubble
    point. 22 California crudes; band API 16.5-63.8, Pb ≤ 48.3 MPa (warn
    outside). Classic as-published (original 1947 form).
    """
    if not (16.5 <= api <= 63.8 and p_pa <= 48.3e6):
        warnings.warn(
            f"Standing Rs evaluated outside its validity band "
            f"(API={api:.4g} in [16.5, 63.8], p={p_pa / 1e6:.4g} MPa ≤ 48.3); "
            f"the returned Rs is extrapolated.",
            stacklevel=2,
        )
    p_psia = _pa_to_psi(p_pa)
    t_f = _k_to_f(t_k)
    rs_scf = gas_sg * (
        (p_psia / 18.2 + 1.4) * 10.0 ** (0.0125 * api - 0.00091 * t_f)
    ) ** 1.2048
    return rs_scf * SCF_STB_TO_SM3_SM3


def pb_standing(
    rs_sm3_sm3: float, t_k: float, api: float, gas_sg: float
) -> float:
    """Bubble-point pressure [Pa]. Standing (1947) inverted; Bellarby Eq 5.12.

        Pb = 18.2·[(Rsb/γg)^0.83·10^(0.00091·T − 0.0125·API) − 1.4]   [psia]

    Rsb is the solution GOR at (or above) the bubble point, T in °F. Same band
    as :func:`rs_standing`. Classic as-published (original 1947 form); the
    exact inverse of :func:`rs_standing`.
    """
    if not (16.5 <= api <= 63.8):
        warnings.warn(
            f"Standing Pb evaluated outside its validity band "
            f"(API={api:.4g} in [16.5, 63.8]); the returned Pb is extrapolated.",
            stacklevel=2,
        )
    rs_scf = rs_sm3_sm3 / SCF_STB_TO_SM3_SM3
    t_f = _k_to_f(t_k)
    pb_psia = 18.2 * (
        (rs_scf / gas_sg) ** 0.83 * 10.0 ** (0.00091 * t_f - 0.0125 * api)
        - 1.4
    )
    return _psi_to_pa(pb_psia)


# Vazquez & Beggs (1980) two-class coefficient sets (≤30 / >30 °API).
_VB_RS = {
    "le30": (0.0362, 1.0937, 25.7240),
    "gt30": (0.0178, 1.1870, 23.9310),
}


def gas_sg_sep100_vazquez_beggs(
    gas_sg: float, api: float, p_sep_pa: float, t_sep_k: float
) -> float:
    """Separator gas-SG normalised to a 100-psig reference. V&B (1980) Eq 2.

        γgs = γg·[1 + 5.912e-5·API·T_sep·log10(p_sep / 114.7)]

    T_sep in °F, p_sep in psia (114.7 psia = 100 psig). Required by the V&B
    Rs / Bo / co forms, which are fitted on the 100-psig-referenced gravity.
    Classic as-published.
    """
    p_sep_psia = _pa_to_psi(p_sep_pa)
    t_sep_f = _k_to_f(t_sep_k)
    return gas_sg * (
        1.0 + 5.912e-5 * api * t_sep_f * math.log10(p_sep_psia / 114.7)
    )


def rs_vazquez_beggs(
    p_pa: float, t_k: float, api: float, gas_sg_100: float
) -> float:
    """Solution GOR [sm3/sm3]. Vazquez & Beggs (1980), SPE-6719, Eq 1.

        Rs = C1·γgs·p^C2·exp(C3·API / T)

    p in psia, T in °R, γgs the 100-psig-referenced gas SG (see
    :func:`gas_sg_sep100_vazquez_beggs`), Rs in scf/STB. Two coefficient
    classes: API ≤ 30 → (0.0362, 1.0937, 25.7240); API > 30 → (0.0178,
    1.1870, 23.9310). Classic as-published.
    """
    c1, c2, c3 = _VB_RS["le30"] if api <= 30.0 else _VB_RS["gt30"]
    p_psia = _pa_to_psi(p_pa)
    t_r = _k_to_rankine(t_k)
    rs_scf = c1 * gas_sg_100 * p_psia ** c2 * math.exp(c3 * api / t_r)
    return rs_scf * SCF_STB_TO_SM3_SM3


# =============================================================================
# Oil FVF + compressibility
# =============================================================================
def bo_standing(
    rs_sm3_sm3: float, t_k: float, api: float, gas_sg: float
) -> float:
    """Oil FVF [rm3/sm3], saturated. Standing (1947); Bellarby (2009) Eq 5.13.

        Bo = 0.9759 + 0.000120·[Rs·(γg/γo)^0.5 + 1.25·T]^1.2

    Rs in scf/STB, T in °F, γo the oil SG (from API). Bo in bbl/STB, which
    equals rm3/sm3 numerically (reservoir and stock-tank barrels convert with
    the same 0.158987 m3 factor). Classic as-published.
    """
    rs_scf = rs_sm3_sm3 / SCF_STB_TO_SM3_SM3
    t_f = _k_to_f(t_k)
    sg_oil = _api_to_sg_oil(api)
    return 0.9759 + 0.000120 * (
        rs_scf * (gas_sg / sg_oil) ** 0.5 + 1.25 * t_f
    ) ** 1.2


# Vazquez & Beggs (1980) Bo two-class coefficient sets.
_VB_BO = {
    "le30": (4.677e-4, 1.751e-5, -1.811e-8),
    "gt30": (4.670e-4, 1.100e-5, 1.337e-9),
}


def bo_vazquez_beggs(
    rs_sm3_sm3: float, t_k: float, api: float, gas_sg_100: float
) -> float:
    """Oil FVF [rm3/sm3], saturated. Vazquez & Beggs (1980), SPE-6719, Eq 3.

        Bo = 1 + C1·Rs + C2·(T − 60)·(API/γgs) + C3·Rs·(T − 60)·(API/γgs)

    Rs in scf/STB, T in °F, γgs the 100-psig gas SG. Two coefficient classes:
    API ≤ 30 → (4.677e-4, 1.751e-5, −1.811e-8); API > 30 → (4.670e-4,
    1.100e-5, 1.337e-9). Bo in bbl/STB = rm3/sm3. Classic as-published.
    """
    c1, c2, c3 = _VB_BO["le30"] if api <= 30.0 else _VB_BO["gt30"]
    rs_scf = rs_sm3_sm3 / SCF_STB_TO_SM3_SM3
    dt = _k_to_f(t_k) - 60.0
    ratio = api / gas_sg_100
    return 1.0 + c1 * rs_scf + c2 * dt * ratio + c3 * rs_scf * dt * ratio


def co_vazquez_beggs(
    rs_sm3_sm3: float, t_k: float, api: float, gas_sg_100: float, p_pa: float
) -> float:
    """Undersaturated oil compressibility co [1/Pa]. V&B (1980), Eq 4.

        co = (−1433 + 5·Rs + 17.2·T − 1180·γgs + 12.61·API) / (1e5·p)   [1/psi]

    Rs in scf/STB, T in °F, p in psia, γgs the 100-psig gas SG; result
    converted 1/psi → 1/Pa. Classic as-published.
    """
    rs_scf = rs_sm3_sm3 / SCF_STB_TO_SM3_SM3
    t_f = _k_to_f(t_k)
    p_psia = _pa_to_psi(p_pa)
    co_psi = (
        -1433.0 + 5.0 * rs_scf + 17.2 * t_f - 1180.0 * gas_sg_100
        + 12.61 * api
    ) / (1.0e5 * p_psia)
    return co_psi / _PSI_IN_PA


def bo_undersaturated(
    bo_pb: float, co_pa: float, p_pa: float, pb_pa: float
) -> float:
    """Oil FVF [rm3/sm3] above the bubble point. Definitional.

        Bo = Bob·exp(co·(Pb − P))

    from the isothermal-compressibility definition (Bellarby Eq 5.14 integrated
    at constant co); co from :func:`co_vazquez_beggs`. At P = Pb, Bo = Bob.
    Pure SI (co in 1/Pa, pressures in Pa).
    """
    return bo_pb * math.exp(co_pa * (pb_pa - p_pa))


# =============================================================================
# Oil viscosity
# =============================================================================
def mu_oil_dead_beggs_robinson(t_k: float, api: float) -> float:
    """Dead-oil viscosity [Pa·s]. Beggs & Robinson (1975), SPE-5434, Eq 2.

        Z = 3.0324 − 0.02023·API
        Y = 10^Z
        X = Y · T^(−1.163)
        μ_od = 10^X − 1                 [cp]

    T in °F. Band: 70-295 °F (≈21-146 °C), API 16-58 (warn outside). Classic
    as-published (the 10^x − 1 form).
    """
    t_f = _k_to_f(t_k)
    if not (70.0 <= t_f <= 295.0 and 16.0 <= api <= 58.0):
        warnings.warn(
            f"Beggs-Robinson dead-oil viscosity outside its validity band "
            f"(T={t_f:.4g} °F in [70, 295], API={api:.4g} in [16, 58]); "
            f"the returned μ is extrapolated.",
            stacklevel=2,
        )
    z = 3.0324 - 0.02023 * api
    y = 10.0 ** z
    x = y * t_f ** (-1.163)
    mu_od_cp = 10.0 ** x - 1.0
    return _cp_to_pas(mu_od_cp)


def mu_oil_saturated_beggs_robinson(
    mu_dead_pa_s: float, rs_sm3_sm3: float
) -> float:
    """Live saturated-oil viscosity [Pa·s]. Beggs & Robinson (1975) Eq 3.

        A = 10.715·(Rs + 100)^(−0.515)
        B = 5.44·(Rs + 150)^(−0.338)
        μ_ob = A · μ_od^B               [cp]

    Rs in scf/STB, μ_od the dead-oil viscosity. Classic as-published.
    """
    rs_scf = rs_sm3_sm3 / SCF_STB_TO_SM3_SM3
    mu_od_cp = _pas_to_cp(mu_dead_pa_s)
    a = 10.715 * (rs_scf + 100.0) ** (-0.515)
    b = 5.44 * (rs_scf + 150.0) ** (-0.338)
    return _cp_to_pas(a * mu_od_cp ** b)


def mu_oil_undersaturated_vazquez_beggs(
    mu_pb_pa_s: float, p_pa: float, pb_pa: float
) -> float:
    """Undersaturated oil viscosity [Pa·s]. Vazquez & Beggs (1980), Eq 5.

        m = 2.6·p^1.187·exp(−11.513 − 8.98e-5·p)
        μ = μ_ob·(p / Pb)^m

    p, Pb in psia; μ_ob the viscosity at the bubble point. The (p/Pb) ratio is
    dimensionless, so μ is scaled directly in Pa·s. At P = Pb, μ = μ_ob.
    Classic as-published.
    """
    p_psia = _pa_to_psi(p_pa)
    pb_psia = _pa_to_psi(pb_pa)
    m = 2.6 * p_psia ** 1.187 * math.exp(-11.513 - 8.98e-5 * p_psia)
    return mu_pb_pa_s * (p_psia / pb_psia) ** m


# =============================================================================
# Water properties (McCain 1990, ch. 16)
# =============================================================================
def bw_mccain(p_pa: float, t_k: float) -> float:
    """Water FVF [rm3/sm3], gas-free. McCain (1990), ch. 16.

        ΔVwT = −1.0001e-2 + 1.33391e-4·T + 5.50654e-7·T²
        ΔVwP = −1.95301e-9·p·T − 1.72834e-13·p²·T
               − 3.58922e-7·p − 2.25341e-10·p²
        Bw = (1 + ΔVwT)·(1 + ΔVwP)

    T in °F, p in psia. Band: ~T 100-260 °F, p to 5000 psia (warn outside).
    Bw in bbl/STB = rm3/sm3. Classic as-published.
    """
    p_psia = _pa_to_psi(p_pa)
    t_f = _k_to_f(t_k)
    if not (100.0 <= t_f <= 260.0 and p_psia <= 5000.0):
        warnings.warn(
            f"McCain Bw evaluated outside its validity band "
            f"(T={t_f:.4g} °F in [100, 260], p={p_psia:.4g} psia ≤ 5000); "
            f"the returned Bw is extrapolated.",
            stacklevel=2,
        )
    dvwt = -1.0001e-2 + 1.33391e-4 * t_f + 5.50654e-7 * t_f ** 2
    dvwp = (
        -1.95301e-9 * p_psia * t_f - 1.72834e-13 * p_psia ** 2 * t_f
        - 3.58922e-7 * p_psia - 2.25341e-10 * p_psia ** 2
    )
    return (1.0 + dvwt) * (1.0 + dvwp)


def mu_water_mccain(t_k: float, salinity_wt_frac: float) -> float:
    """Water viscosity [Pa·s] at ATMOSPHERIC pressure. McCain (1990), ch. 16.

        A = 109.574 − 8.40564·S + 0.313314·S² + 8.72213e-3·S³
        B = −1.12166 + 2.63951e-2·S − 6.79461e-4·S² − 5.47119e-5·S³
              + 1.55586e-6·S⁴
        μ_w(1 atm) = A · T^B            [cp]

    T in °F; **S is salinity in weight PERCENT** (S = salinity_wt_frac × 100,
    so ``salinity_wt_frac`` = 0.05 → 5 wt%) — pinned to weight fraction to
    avoid the ppm/fraction trap. Apply :func:`mu_water_pressure_mccain` for the
    reservoir-pressure adjustment. Band: T 100-400 °F, S 0-26 wt% (warn
    outside). Classic as-published.
    """
    t_f = _k_to_f(t_k)
    s = salinity_wt_frac * 100.0
    if not (100.0 <= t_f <= 400.0 and 0.0 <= s <= 26.0):
        warnings.warn(
            f"McCain water-viscosity outside its validity band "
            f"(T={t_f:.4g} °F in [100, 400], S={s:.4g} wt% in [0, 26]); "
            f"the returned μ is extrapolated.",
            stacklevel=2,
        )
    a = 109.574 - 8.40564 * s + 0.313314 * s ** 2 + 8.72213e-3 * s ** 3
    b = (
        -1.12166 + 2.63951e-2 * s - 6.79461e-4 * s ** 2
        - 5.47119e-5 * s ** 3 + 1.55586e-6 * s ** 4
    )
    return _cp_to_pas(a * t_f ** b)


def mu_water_pressure_mccain(mu_atm_pa_s: float, p_pa: float) -> float:
    """Pressure adjustment of water viscosity [Pa·s]. McCain (1990), ch. 16.

        μ_w / μ_w(1 atm) = 0.9994 + 4.0295e-5·p + 3.1062e-9·p²

    p in psia; multiplies the atmospheric viscosity from
    :func:`mu_water_mccain`. Band: p to ~10000 psia (warn outside). Classic
    as-published.
    """
    p_psia = _pa_to_psi(p_pa)
    if not (p_psia <= 10000.0):
        warnings.warn(
            f"McCain water-viscosity pressure factor outside its band "
            f"(p={p_psia:.4g} psia ≤ 10000); the factor is extrapolated.",
            stacklevel=2,
        )
    factor = 0.9994 + 4.0295e-5 * p_psia + 3.1062e-9 * p_psia ** 2
    return mu_atm_pa_s * factor


def rho_water(p_pa: float, t_k: float, salinity_wt_frac: float) -> float:
    """Brine density [kg/m3]. McCain (1990) route: ρ = ρ_sc(S) / Bw.

        ρ_w,sc = 62.368 + 0.438603·S + 1.60074e-3·S²   [lb/ft3]

    S = salinity in wt% (= salinity_wt_frac × 100). ρ_sc is the standard-
    condition brine density; the in-situ density is that shrunk by the water
    FVF :func:`bw_mccain`. Definitional on top of McCain's Bw.
    """
    s = salinity_wt_frac * 100.0
    rho_sc_lbft3 = 62.368 + 0.438603 * s + 1.60074e-3 * s ** 2
    rho_sc = _U.convert(rho_sc_lbft3, "pound/foot**3", "kilogram/meter**3")
    return rho_sc / bw_mccain(p_pa, t_k)


# =============================================================================
# Oil density (definitional closer)
# =============================================================================
def rho_oil(
    rs_sm3_sm3: float, bo_rm3_sm3: float, api: float, gas_sg: float
) -> float:
    """Live-oil density [kg/m3] by mass balance. Bellarby (2009) Eq 5.1.

        ρ_o = (62.4·γo + 0.0136·Rs·γg) / Bo            [lb/ft3]

    γo the oil SG (from API), Rs in scf/STB, γg the gas SG. The numerator is
    the stock-tank-oil mass plus the dissolved-gas mass per barrel; dividing by
    Bo gives the reservoir density. Result converted lb/ft3 → kg/m3.
    Definitional (McCain mass-balance form).
    """
    sg_oil = _api_to_sg_oil(api)
    rs_scf = rs_sm3_sm3 / SCF_STB_TO_SM3_SM3
    rho_lbft3 = (62.4 * sg_oil + 0.0136 * rs_scf * gas_sg) / bo_rm3_sm3
    return _U.convert(rho_lbft3, "pound/foot**3", "kilogram/meter**3")


# =============================================================================
# Gas/liquid surface tension (VLP inputs)
# =============================================================================
#: 1 dyne/cm in N/m (the surface-tension seam; = 1e-3 exactly).
_DYNE_CM_TO_N_M: float = 1.0e-3


def sigma_oil_gas_baker_swerdloff(t_k: float, api: float) -> float:
    """Dead-oil/gas surface tension [N/m]. Baker & Swerdloff (1956).

        sigma(68 °F)  = 39.0 - 0.2571·API          [dyne/cm]
        sigma(100 °F) = 37.5 - 0.2571·API          [dyne/cm]

    linearly interpolated in temperature between 68 °F and 100 °F (held flat
    outside that range). Returns the **dead-oil** (atmospheric) value in N/m.
    The standard live-oil **pressure de-rating** — Baker & Swerdloff's
    ``C = 1 - 0.024·p^0.45`` (p in psia), ``sigma_live = C·sigma_dead`` — is the
    caller's to apply where the local pressure is known (it is not an argument
    to this dead-oil form). T in °F for the correlation; band 68-100 °F (warn
    outside). Classic as-published (OGJ, 1956; reproduced in Beggs, 1991).
    """
    t_f = _k_to_f(t_k)
    if not (68.0 <= t_f <= 100.0):
        warnings.warn(
            f"Baker-Swerdloff surface tension outside its validity band "
            f"(T={t_f:.4g} °F in [68, 100]); the value is held at the nearest "
            f"endpoint / extrapolated.",
            stacklevel=2,
        )
    s68 = 39.0 - 0.2571 * api
    s100 = 37.5 - 0.2571 * api
    if t_f <= 68.0:
        sigma_dyne = s68
    elif t_f >= 100.0:
        sigma_dyne = s100
    else:
        sigma_dyne = s68 - (t_f - 68.0) * (s68 - s100) / (100.0 - 68.0)
    return sigma_dyne * _DYNE_CM_TO_N_M


def sigma_water_gas(t_k: float, p_pa: float) -> float:
    """Water/gas interfacial tension [N/m]. Katz et al. (1959) / Ramey fit.

        sigma(74 °F)  = 75.0 - 1.108·p^0.349        [dyne/cm]
        sigma(280 °F) = 53.0 - 0.1048·p^0.637       [dyne/cm]

    p in psia, linearly interpolated in temperature between 74 °F and 280 °F
    (held flat outside). The classic published water/gas IFT fit (Katz et al.,
    *Handbook of Natural Gas Engineering*, 1959; also attributed to Ramey). At
    atmospheric pressure and 74 °F this returns ≈ 72 dyne/cm, the surface
    tension of water. Band T 74-280 °F, p to ~5000 psia (warn outside). Returns
    N/m. Classic as-published.
    """
    t_f = _k_to_f(t_k)
    p_psia = _pa_to_psi(p_pa)
    if not (74.0 <= t_f <= 280.0 and p_psia <= 5000.0):
        warnings.warn(
            f"Water/gas surface tension outside its validity band "
            f"(T={t_f:.4g} °F in [74, 280], p={p_psia:.4g} psia ≤ 5000); "
            f"the value is held/extrapolated.",
            stacklevel=2,
        )
    s74 = 75.0 - 1.108 * p_psia ** 0.349
    s280 = 53.0 - 0.1048 * p_psia ** 0.637
    if t_f <= 74.0:
        sigma_dyne = s74
    elif t_f >= 280.0:
        sigma_dyne = s280
    else:
        sigma_dyne = s74 - (t_f - 74.0) * (s74 - s280) / (280.0 - 74.0)
    return sigma_dyne * _DYNE_CM_TO_N_M
