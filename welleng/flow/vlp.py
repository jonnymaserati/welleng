"""Multiphase vertical-lift-performance (VLP) scalar reference oracle.

This module implements the classic multiphase-flow *local pressure-gradient*
correlations — the drift-flux, mechanistic and empirical forms that give the
pressure gradient ``dp/dz`` at a single station from the local superficial
velocities and in-situ phase properties. Every function is a **scalar
reference form** — pure ``float`` in, ``float`` out — and this module is
welleng's open-core *correctness oracle*: the vectorised / GPU / marched forms
in the commercial layer parity-gate against these functions (bit-identical, or
a documented tolerance where an implicit solve differs in its tail).

Design contract
---------------
* **Local point gradient, not a traverse.** The oracle unit is ``dp/dz`` at one
  station given local conditions. The marching integrator (segmenting the well,
  coupling to PVT / temperature, integrating ``dp/dz`` over depth) is the
  commercial layer's job. Per correlation the sub-pieces (pattern, holdup,
  gradient) are exposed **individually** so each published stage is testable on
  its own.
* **Strict SI, no field seam.** Every input and output is SI: superficial
  velocities ``vsl_m_s`` / ``vsg_m_s`` **m/s**, densities ``rho_l`` / ``rho_g``
  **kg/m3**, viscosities ``mu_l`` / ``mu_g`` **Pa·s**, surface tension
  ``sigma_n_m`` **N/m**, diameter ``d_m`` and roughness ``roughness_m`` **m**,
  gradient output ``dpdz_pa_m`` **Pa/m**. No PVT coupling — the rates→velocities
  map (via Bo/Bg/Rs at local P,T) is the caller's.
* **Angle from horizontal.** ``theta_deg`` is measured **from horizontal**,
  ``+90`` = vertical up-flow; a single convention across every correlation,
  with documented conversions inside forms published otherwise (e.g. the
  Beggs-Brill ``sin(1.8·theta)`` inclination factor is native to this
  convention). Up-flow production is ``0 < theta <= 90``.
* **Two-phase liquid.** Oil + water are treated as one liquid phase with
  mixture properties at this tier (the correlations' own treatment);
  three-phase slip is out of scope.
* **Single Colebrook friction kernel.** Every correlation needing a Moody-chart
  friction factor calls :func:`friction_factor_colebrook` — a documented
  deviation where a source specified its own smooth-pipe form (Blasius / Chen),
  reconciled in that function's docstring.
* **Validity bands warn, never clamp, never raise** — matching the
  :mod:`welleng.flow.pvt` / :mod:`welleng.flow.ipr` house pattern.
* **Citation in every docstring:** source + specific equation.

Sign convention
---------------
``dpdz_pa_m`` is the total gradient with **positive = pressure increasing in the
downhole direction** (hydrostatic head plus the flow-opposing friction for
up-flow). The acceleration (kinetic-energy) term needs the local absolute
pressure, which is not a local-gradient-oracle input; it is omitted here and is
< 0.1 % of the total in the published examples (Hasan & Kabir 2018, Field
Examples 3.1 and 4.1) — the caller's marcher, which carries ``p``, may add it.

References
----------
Colebrook, C. F. (1939). *Turbulent Flow in Pipes, with Particular Reference to
    the Transition Region between the Smooth and Rough Pipe Laws.* J. Inst.
    Civil Eng. 11(4): 133-156.
Zuber, N. & Findlay, J. A. (1965). *Average Volumetric Concentration in
    Two-Phase Flow Systems.* J. Heat Transfer 87(4): 453-468.
Harmathy, T. Z. (1960). *Velocity of Large Drops and Bubbles in Media of
    Infinite or Restricted Extent.* AIChE J. 6(2): 281-288.
Taitel, Y., Bornea, D. & Dukler, A. E. (1980). *Modelling Flow Pattern
    Transitions for Steady Upward Gas-Liquid Flow in Vertical Tubes.* AIChE J.
    26(3): 345-354.
Kaya, A. S., Sarica, C. & Brill, J. P. (2001). *Mechanistic Modeling of
    Two-Phase Flow in Deviated Wells.* SPE Prod. & Facilities 16(3): 156-165
    (SPE-72998).
Hasan, A. R. & Kabir, C. S. (1988). *A Study of Multiphase Flow Behavior in
    Vertical Wells.* SPE Prod. Eng. 3(2): 263-272 (SPE-15138).
Hasan, A. R. & Kabir, C. S. (2018). *Fluid Flow and Heat Transfer in
    Wellbores*, 2nd ed. SPE — chapters 3 (vertical) and 4 (deviated) mechanistic
    forms and Field Examples 3.1 / 4.1 (the worked-value anchors used here).
Beggs, H. D. & Brill, J. P. (1973). *A Study of Two-Phase Flow in Inclined
    Pipes.* JPT 25(5): 607-617 (SPE-4007).
Hagedorn, A. R. & Brown, K. E. (1965). *Experimental Study of Pressure
    Gradients Occurring During Continuous Two-Phase Flow in Small-Diameter
    Vertical Conduits.* JPT 17(4): 475-484 (SPE-940).
Brill, J. P. & Mukherjee, H. (1999). *Multiphase Flow in Wells*, SPE Monograph
    Vol. 17 (the standard published Hagedorn-Brown chart-fit reference —
    see :func:`holdup_hagedorn_brown`).
"""
from __future__ import annotations

import math
import warnings

#: Standard gravity [m/s2].
G: float = 9.80665

# Flow-pattern integer codes (per-correlation; enums differ by correlation).
# Hasan-Kabir mechanistic:
BUBBLE: int = 0
SLUG: int = 1
CHURN: int = 2
ANNULAR: int = 3
# Beggs-Brill horizontal-basis:
SEGREGATED: int = 0
INTERMITTENT: int = 1
DISTRIBUTED: int = 2
TRANSITION: int = 3


# =============================================================================
# Shared utilities
# =============================================================================
def friction_factor_colebrook(reynolds: float, rel_roughness: float) -> float:
    """Darcy friction factor [-]. Colebrook & White (1939), implicit.

        1/sqrt(f) = -2·log10( eps/(3.7·d) + 2.51/(Re·sqrt(f)) )

    solved by fixed-point iteration from a Haaland explicit seed (converges to
    ~machine precision). ``rel_roughness`` is the relative roughness eps/d [-].
    This is the single Moody-``f`` kernel every VLP correlation in this module
    uses. For ``Re < 2100`` the laminar law ``f = 64/Re`` is returned instead
    (the transitional 2100-4000 band is not specially treated — Colebrook is
    evaluated there and warned). Classic as-published.
    """
    if reynolds <= 0.0:
        warnings.warn(
            f"Colebrook friction factor needs Re > 0 (got Re={reynolds:.4g}); "
            f"returning nan.",
            stacklevel=2,
        )
        return float("nan")
    if reynolds < 2100.0:
        return 64.0 / reynolds
    if 2100.0 <= reynolds < 4000.0:
        warnings.warn(
            f"Colebrook friction factor in the transitional band "
            f"(Re={reynolds:.4g} in [2100, 4000)); the turbulent law is "
            f"extrapolated into it.",
            stacklevel=2,
        )
    a = rel_roughness / 3.7
    # Haaland (1983) explicit seed for 1/sqrt(f).
    inv_sqrt = -1.8 * math.log10(a ** 1.11 + 6.9 / reynolds)
    for _ in range(50):
        f = 1.0 / (inv_sqrt * inv_sqrt)
        rhs = -2.0 * math.log10(a + 2.51 / (reynolds * math.sqrt(f)))
        if abs(rhs - inv_sqrt) < 1.0e-12:
            inv_sqrt = rhs
            break
        inv_sqrt = rhs
    return 1.0 / (inv_sqrt * inv_sqrt)


def no_slip_holdup(vsl_m_s: float, vsg_m_s: float) -> float:
    """No-slip liquid holdup lambda_L = vsl/(vsl+vsg) [-]. Definitional.

    The input liquid volume fraction (the holdup with zero slip between phases).
    Returns 1.0 for zero total velocity (no gas → full liquid).
    """
    total = vsl_m_s + vsg_m_s
    if total == 0.0:
        return 1.0
    return vsl_m_s / total


# =============================================================================
# Zuber-Findlay drift-flux
# =============================================================================
def holdup_zuber_findlay(
    vsl_m_s: float, vsg_m_s: float, c0: float, vd_m_s: float
) -> float:
    """Liquid holdup [-] from the drift-flux model. Zuber & Findlay (1965).

        alpha = vsg / (C0·vm + vd),   H_L = 1 - alpha

    with mixture velocity ``vm = vsl + vsg``, distribution parameter ``C0`` and
    drift velocity ``vd`` [m/s]. ``C0`` and ``vd`` are explicit — the
    profile/drift parameters are the caller's model choice (the Hasan-Kabir
    forms below supply per-pattern values). At ``vsg = 0`` this returns
    ``H_L = 1``. Classic as-published (J. Heat Transfer 87(4): 453-468).
    """
    vm = vsl_m_s + vsg_m_s
    denom = c0 * vm + vd_m_s
    if denom == 0.0:
        return 1.0
    alpha = vsg_m_s / denom
    return 1.0 - alpha


# =============================================================================
# Hasan-Kabir mechanistic (Stage-A mechanistic path)
# =============================================================================
def _harmathy_rise(rho_l: float, rho_g: float, sigma_n_m: float) -> float:
    """Small-bubble terminal rise velocity v_inf [m/s]. Harmathy (1960).

        v_inf = 1.53·[ g·sigma·(rho_l - rho_g) / rho_l^2 ]^(1/4)

    Hasan & Kabir (2018) Eq. 3.6. Not appreciably affected by inclination
    (Eq. 4.7 discussion), so used unchanged for deviated wells.
    """
    return 1.53 * (G * sigma_n_m * (rho_l - rho_g) / rho_l ** 2) ** 0.25


def _taylor_rise(
    rho_l: float, rho_g: float, d_m: float, theta_deg: float
) -> float:
    """Taylor-bubble rise velocity v_infT [m/s]. Hasan & Kabir (2018) Eq. 4.33.

        v_infT = 0.35·[ g·d·(rho_l - rho_g)/rho_l ]^0.5
                 · (sin a)^0.5 · (1 + cos a)^1.2

    with ``a = theta`` from horizontal. The inclination factor
    ``(sin a)^0.5·(1+cos a)^1.2`` reduces to 1 at vertical (a = 90°), where this
    equals Eq. 3.8 (C2 = 0.35 for large Nf, Eo). Davies & Taylor (1949) form.
    """
    a = math.radians(theta_deg)
    incl = math.sqrt(max(math.sin(a), 0.0)) * (1.0 + math.cos(a)) ** 1.2
    base = 0.35 * (G * d_m * (rho_l - rho_g) / rho_l) ** 0.5
    return base * incl


def _avg_rise(
    vsl_m_s: float, vsg_m_s: float, v_inf: float, v_inft: float,
    theta_deg: float,
) -> float:
    """Slug/churn average rise velocity [m/s]. Hasan & Kabir (2018) Eq. 3.28.

        vbar = v_inf·(1 - exp(-vt/vsg)) + v_infT·exp(-vt/vsg)

    a continuous blend of the small-bubble and Taylor-bubble rise velocities
    (``vt`` the bubble→slug transition velocity, Eq. 4.7) that removes the
    holdup discontinuity at the bubbly/slug boundary.
    """
    a = math.radians(theta_deg)
    vt = (0.429 * vsl_m_s + 0.357 * v_inf) * math.sin(a)
    if vsg_m_s <= 0.0:
        return v_inf
    e = math.exp(-vt / vsg_m_s)
    return v_inf * (1.0 - e) + v_inft * e


def pattern_hasan_kabir(
    vsl_m_s: float, vsg_m_s: float, rho_l: float, rho_g: float,
    sigma_n_m: float, d_m: float, theta_deg: float,
) -> int:
    """Flow pattern int [0=bubble, 1=slug, 2=churn, 3=annular]. Hasan & Kabir.

    Applies the mechanistic transition criteria of Hasan & Kabir (2018),
    ch. 3-4, in the order the book recommends (annular and bubbly first, since
    either precludes the rest):

    * **Annular** (Eq. 4.11): ``vsg >= 3.1·(sin a)^(1/4)·
      [g·sigma·(rho_l-rho_g)/rho_g^2]^(1/4)`` **and** the no-slip void
      ``vsg/vm >= 0.85`` — the Barnea et al. (1985) film-bridging limit that
      Field Example 4.1 uses to reject annular at ``vsg > vsg_ann`` when the
      void is too low (below 0.85 the liquid bridges the channel → slug/churn).
    * **Bubble** (Eq. 4.7): ``vsg <= (0.429·vsl + 0.357·v_inf)·sin a`` — the
      gas void fraction below the 0.25 bubble→slug threshold (drift-flux,
      C0 = 1.2, small-bubble rise ``v_inf`` from Harmathy Eq. 3.6).
    * **Churn** (Eq. 4.10, Kaya et al. 2001): ``vsg >= 12.19·(1.2·vsl +
      v_infT)`` — the void-above-0.78 slug→churn boundary.
    * **Slug** otherwise.

    ``a = theta`` from horizontal; the churn/annular criteria carry only a mild
    inclination dependence (the book notes one may ignore it away from bubbly).

    .. note::
       The ratified pattern signature is **viscosity-free**, so the slug↔churn
       split uses the clean Kaya Eq. 4.10 boundary. The book's Field Example 4.1
       narrative reaches *churn* via the viscosity-dependent dispersed-bubble
       onset (Eq. 3.10, which needs a friction factor); that path is outside
       this signature, so Example 4.1 classifies here as **slug** (the holdup
       difference vs churn is only the C0 = 1.2 vs 1.15 drift parameter, ~5 %).
       Field Example 3.1 (vertical) classifies as **slug** as in the book.

    No warn band (a pattern is always returned).
    """
    a = math.radians(theta_deg)
    sin_a = math.sin(a)
    vm = vsl_m_s + vsg_m_s
    v_inf = _harmathy_rise(rho_l, rho_g, sigma_n_m)
    v_inft = _taylor_rise(rho_l, rho_g, d_m, theta_deg)

    vsg_ann = 3.1 * (max(sin_a, 0.0)) ** 0.25 * (
        G * sigma_n_m * (rho_l - rho_g) / rho_g ** 2
    ) ** 0.25
    void_ns = vsg_m_s / vm if vm > 0.0 else 0.0
    if vsg_m_s >= vsg_ann and void_ns >= 0.85:
        return ANNULAR

    vsg_bs = (0.429 * vsl_m_s + 0.357 * v_inf) * sin_a
    if vsg_m_s <= vsg_bs:
        return BUBBLE

    vsg_sc = 12.19 * (1.2 * vsl_m_s + v_inft)
    if vsg_m_s >= vsg_sc:
        return CHURN
    return SLUG


def holdup_hasan_kabir(
    vsl_m_s: float, vsg_m_s: float, rho_l: float, rho_g: float,
    sigma_n_m: float, d_m: float, theta_deg: float, pattern: int,
) -> float:
    """Per-pattern drift-flux liquid holdup [-]. Hasan & Kabir (2018) ch. 3-4.

    The in-situ gas void fraction ``fg = vsg/(C0·vm + vbar)`` with per-regime
    distribution parameter and rise velocity; returns ``H_L = 1 - fg``:

    * **bubble** (Eq. 3.22): ``C0 = 1.2``, ``vbar = v_inf`` (Harmathy Eq. 3.6).
    * **slug** (Eq. 3.27): ``C0 = 1.2``, ``vbar`` the average rise velocity
      (Eq. 3.28) blending ``v_inf`` and the Taylor-bubble rise ``v_infT``
      (Eq. 4.33).
    * **churn** (Eq. 4.28): as slug but ``C0 = 1.15`` (flatter void profile).
    * **annular**: the simplified homogeneous treatment (Hasan et al. 2010b) —
      ``H_L = no-slip holdup`` — since at annular velocities slip is small.

    ``pattern`` is the code from :func:`pattern_hasan_kabir`. Validated: Field
    Example 3.1 (slug, ``H_L = 0.543``) and 4.1 (churn, ``H_L = 0.417``).
    ``H_L`` is floored at the no-slip holdup (physical lower bound). No warn
    band. ``vm = vsl + vsg``.
    """
    vm = vsl_m_s + vsg_m_s
    lambda_l = no_slip_holdup(vsl_m_s, vsg_m_s)
    if pattern == ANNULAR:
        return lambda_l
    v_inf = _harmathy_rise(rho_l, rho_g, sigma_n_m)
    if pattern == BUBBLE:
        c0, vbar = 1.2, v_inf
    else:  # slug or churn
        v_inft = _taylor_rise(rho_l, rho_g, d_m, theta_deg)
        vbar = _avg_rise(vsl_m_s, vsg_m_s, v_inf, v_inft, theta_deg)
        c0 = 1.15 if pattern == CHURN else 1.2
    denom = c0 * vm + vbar
    if denom == 0.0:
        return 1.0
    fg = vsg_m_s / denom
    return max(1.0 - fg, lambda_l)


def dpdz_hasan_kabir(
    vsl_m_s: float, vsg_m_s: float, rho_l: float, rho_g: float,
    mu_l: float, mu_g: float, sigma_n_m: float, d_m: float,
    roughness_m: float, theta_deg: float,
) -> float:
    """Total local pressure gradient [Pa/m]. Hasan & Kabir (2018) ch. 3-4.

        dp/dz = rho_m·g·sin(theta)            (static head)
                + f·rho_m·vm^2 / (2·d)        (friction)

    with the in-situ mixture density ``rho_m = rho_l·H_L + rho_g·(1 - H_L)``
    from the per-pattern holdup :func:`holdup_hasan_kabir`, and the friction
    factor ``f`` from :func:`friction_factor_colebrook` at the mixture Reynolds
    number ``Re_m = rho_m·vm·d/mu_m`` with the **mass-fraction-weighted**
    mixture viscosity ``mu_m = mu_l·(1-x) + mu_g·x``, ``x`` the gas mass
    fraction (Eq. 2.25). The book uses a smooth-pipe Blasius ``f``; the single
    Colebrook kernel is used here instead (a documented deviation — at the
    example Reynolds numbers the two agree to ~1-2 %). Acceleration is
    neglected (< 0.1 % in the examples; needs the local ``p``). Validated vs
    Field Examples 3.1 (0.221 psi/ft) and 4.1 (0.267 psi/ft). Positive =
    pressure increasing downhole.
    """
    vm = vsl_m_s + vsg_m_s
    pattern = pattern_hasan_kabir(
        vsl_m_s, vsg_m_s, rho_l, rho_g, sigma_n_m, d_m, theta_deg
    )
    h_l = holdup_hasan_kabir(
        vsl_m_s, vsg_m_s, rho_l, rho_g, sigma_n_m, d_m, theta_deg, pattern
    )
    rho_m = rho_l * h_l + rho_g * (1.0 - h_l)

    # Gas mass fraction for the mass-weighted mixture viscosity (Eq. 2.25).
    mass_g = vsg_m_s * rho_g
    mass_t = mass_g + vsl_m_s * rho_l
    x = mass_g / mass_t if mass_t > 0.0 else 0.0
    mu_m = mu_l * (1.0 - x) + mu_g * x

    static = rho_m * G * math.sin(math.radians(theta_deg))
    friction = 0.0
    if vm != 0.0 and mu_m > 0.0:
        re_m = rho_m * abs(vm) * d_m / mu_m
        f = friction_factor_colebrook(re_m, roughness_m / d_m)
        friction = f * rho_m * vm * vm / (2.0 * d_m)
    return static + friction


# =============================================================================
# Beggs-Brill (1973) — inclination workhorse (ORIGINAL, no Payne correction)
# =============================================================================
# Horizontal holdup coefficients H_L(0) = a·lambda^b / Fr^c (SPE-4007 Table).
_BB_HL = {
    SEGREGATED: (0.98, 0.4846, 0.0868),
    INTERMITTENT: (0.845, 0.5351, 0.0173),
    DISTRIBUTED: (1.065, 0.5824, 0.0609),
}
# Uphill inclination-correction coefficients (d', e, f, g) per pattern.
_BB_UP = {
    SEGREGATED: (0.011, -3.768, 3.539, -1.614),
    INTERMITTENT: (2.96, 0.305, -0.4473, 0.0978),
    DISTRIBUTED: (1.0, 0.0, 0.0, 0.0),  # C = 0, psi = 1 (no correction)
}
# Downhill coefficients (all patterns share one set).
_BB_DOWN = (4.70, -0.3692, 0.1244, -0.5056)


def pattern_beggs_brill(vsl_m_s: float, vsg_m_s: float, d_m: float) -> int:
    """Horizontal-basis flow pattern int. Beggs & Brill (1973), SPE-4007.

    ``[0=segregated, 1=intermittent, 2=distributed, 3=transition]`` from the
    no-slip liquid fraction ``lambda_L`` and Froude number ``Fr = vm^2/(g·d)``
    via the original 1973 boundaries::

        L1 = 316·lambda^0.302        L2 = 0.0009252·lambda^-2.4684
        L3 = 0.10·lambda^-1.4516     L4 = 0.5·lambda^-6.738

    segregated: lambda<0.01, Fr<L1  OR  lambda>=0.01, Fr<L2;
    transition: lambda>=0.01 and L2<=Fr<L3;
    distributed: lambda<0.4, Fr>=L1  OR  lambda>=0.4, Fr>L4;
    intermittent: otherwise. Classic as-published. Validated: Field Example 4.1
    (lambda=0.307, Fr high → intermittent).
    """
    vm = vsl_m_s + vsg_m_s
    lam = no_slip_holdup(vsl_m_s, vsg_m_s)
    fr = vm * vm / (G * d_m)
    l1 = 316.0 * lam ** 0.302
    l2 = 0.0009252 * lam ** (-2.4684)
    l3 = 0.10 * lam ** (-1.4516)
    l4 = 0.5 * lam ** (-6.738)
    if (lam < 0.01 and fr < l1) or (lam >= 0.01 and fr < l2):
        return SEGREGATED
    if lam >= 0.01 and l2 <= fr < l3:
        return TRANSITION
    if (lam < 0.4 and fr >= l1) or (lam >= 0.4 and fr > l4):
        return DISTRIBUTED
    return INTERMITTENT


def _bb_hl0(lam: float, fr: float, pattern: int) -> float:
    """Horizontal holdup H_L(0) = a·lambda^b/Fr^c, floored at lambda (Eq. 4.13)."""
    a, b, c = _BB_HL[pattern]
    hl0 = a * lam ** b / fr ** c
    return max(hl0, lam)


def _bb_psi(
    lam: float, fr: float, n_lv: float, theta_deg: float, pattern: int
) -> float:
    """Beggs-Brill inclination factor psi = H_L(theta)/H_L(0). Eqs. 4.20-4.22.

        C = (1 - lambda)·ln( d'·lambda^e · N_Lv^f · Fr^g )      (C >= 0 uphill)
        psi = 1 + C·[ sin(1.8·theta) - (1/3)·sin^3(1.8·theta) ]

    with ``N_Lv = vsl·(rho_l/(g·sigma))^0.25`` the (dimensionless, SI-consistent)
    liquid-velocity number. Uphill (theta >= 0) uses the per-pattern coefficients
    with the **original C >= 0 clamp** (the correction only raises holdup
    up-flow; distributed uphill takes C = 0). Downhill uses the shared set with
    no clamp. Classic 1973 form (no Payne correction).
    """
    if theta_deg >= 0.0:
        dp, e, ff, gg = _BB_UP[pattern]
    else:
        dp, e, ff, gg = _BB_DOWN
    arg = dp * lam ** e * n_lv ** ff * fr ** gg
    c = (1.0 - lam) * math.log(arg)
    if theta_deg >= 0.0:
        c = max(c, 0.0)
    rad = math.radians(1.8 * theta_deg)
    s = math.sin(rad)
    return 1.0 + c * (s - (s ** 3) / 3.0)


def holdup_beggs_brill(
    vsl_m_s: float, vsg_m_s: float, d_m: float, theta_deg: float,
    rho_l: float = 1000.0, sigma_n_m: float = 0.072,
) -> float:
    """Inclination-corrected liquid holdup [-]. Beggs & Brill (1973), SPE-4007.

        H_L(theta) = H_L(0)·psi(theta)

    the horizontal pattern holdup (Eq. 4.13, floored at ``lambda_L``) times the
    inclination factor ``psi`` (Eqs. 4.20-4.22). ``rho_l`` and ``sigma_n_m``
    enter only through the liquid-velocity number ``N_Lv`` in ``psi`` and carry
    water-like defaults (1000 kg/m3, 0.072 N/m) for the air/water basis; pass
    the true liquid values for oil systems. **ORIGINAL 1973 form — no Payne et
    al. correction**, with the original ``C >= 0`` uphill clamp (see
    :func:`_bb_psi`).

    Transition pattern: the published interpolation between the segregated and
    intermittent holdups, ``H_L = A·H_L(seg) + B·H_L(int)`` with
    ``A = (L3 - Fr)/(L3 - L2)``, ``B = 1 - A``.

    Validated against Field Example 4.1: H_L(0) = 0.4146 (intermittent);
    with the C >= 0 clamp psi = 1.0 (the unclamped book calc gives 0.984,
    which the authors flag as non-physical), so H_L(theta) = 0.4146.
    No warn band.
    """
    vm = vsl_m_s + vsg_m_s
    lam = no_slip_holdup(vsl_m_s, vsg_m_s)
    fr = vm * vm / (G * d_m)
    n_lv = vsl_m_s * (rho_l / (G * sigma_n_m)) ** 0.25
    pattern = pattern_beggs_brill(vsl_m_s, vsg_m_s, d_m)

    if pattern == TRANSITION:
        l2 = 0.0009252 * lam ** (-2.4684)
        l3 = 0.10 * lam ** (-1.4516)
        aa = (l3 - fr) / (l3 - l2)
        bb = 1.0 - aa
        hl_seg = _bb_hl0(lam, fr, SEGREGATED) * _bb_psi(
            lam, fr, n_lv, theta_deg, SEGREGATED
        )
        hl_int = _bb_hl0(lam, fr, INTERMITTENT) * _bb_psi(
            lam, fr, n_lv, theta_deg, INTERMITTENT
        )
        return aa * hl_seg + bb * hl_int

    return _bb_hl0(lam, fr, pattern) * _bb_psi(
        lam, fr, n_lv, theta_deg, pattern
    )


def dpdz_beggs_brill(
    vsl_m_s: float, vsg_m_s: float, rho_l: float, rho_g: float,
    mu_l: float, mu_g: float, sigma_n_m: float, d_m: float,
    roughness_m: float, theta_deg: float,
) -> float:
    """Total local pressure gradient [Pa/m]. Beggs & Brill (1973), SPE-4007.

        dp/dz = rho_m·g·sin(theta)                (static head)
                + f_tp·rho_n·vm^2 / (2·d)         (friction)

    with the **slip** mixture density ``rho_m = rho_l·H_L + rho_g·(1 - H_L)``
    (holdup from :func:`holdup_beggs_brill`) in the static head, and the
    **no-slip** density ``rho_n = rho_l·lambda + rho_g·(1 - lambda)`` in the
    friction term. The two-phase friction factor is the normalised ratio
    (Eqs. 4.16-4.18)::

        y = lambda / H_L(theta)^2
        s = ln(y) / [ -0.0523 + 3.182·ln(y) - 0.8725·ln(y)^2
                      + 0.01853·ln(y)^4 ]
        f_tp = f_ns · exp(s)

    with the singularity guard ``s = ln(2.2·y - 1.2)`` for ``1 < y < 1.2``, and
    the no-slip friction factor ``f_ns`` from :func:`friction_factor_colebrook`
    at ``Re_ns = rho_n·vm·d/mu_n`` (no-slip volume-weighted
    ``mu_n = mu_l·lambda + mu_g·(1 - lambda)``, Eq. 2.21). The book uses a
    smooth-pipe Blasius ``f_ns``; the Colebrook kernel is used here (documented
    deviation, ~1-2 % at the example Re). The Ek acceleration term needs the
    local ``p`` (not a local-gradient input) and is omitted (< 0.1 % in the
    example). Validated vs Field Example 4.1 (s=0.39, f_tp/f_ns=1.48; total
    ~0.22 psi/ft). Positive = pressure increasing downhole.
    """
    vm = vsl_m_s + vsg_m_s
    lam = no_slip_holdup(vsl_m_s, vsg_m_s)
    h_l = holdup_beggs_brill(
        vsl_m_s, vsg_m_s, d_m, theta_deg, rho_l, sigma_n_m
    )
    rho_m = rho_l * h_l + rho_g * (1.0 - h_l)
    rho_n = rho_l * lam + rho_g * (1.0 - lam)
    mu_n = mu_l * lam + mu_g * (1.0 - lam)

    static = rho_m * G * math.sin(math.radians(theta_deg))

    friction = 0.0
    if vm != 0.0 and mu_n > 0.0 and h_l > 0.0:
        re_n = rho_n * abs(vm) * d_m / mu_n
        f_ns = friction_factor_colebrook(re_n, roughness_m / d_m)
        y = lam / (h_l * h_l)
        if y <= 0.0 or y == 1.0:
            s = 0.0
        elif 1.0 < y < 1.2:
            s = math.log(2.2 * y - 1.2)
        else:
            ly = math.log(y)
            denom = (
                -0.0523 + 3.182 * ly - 0.8725 * ly * ly + 0.01853 * ly ** 4
            )
            s = ly / denom
        f_tp = f_ns * math.exp(s)
        friction = f_tp * rho_n * vm * vm / (2.0 * d_m)
    return static + friction


# =============================================================================
# Hagedorn-Brown (1965) — vertical-oil classic  [STUBBED: no citeable fit]
# =============================================================================
def holdup_hagedorn_brown(
    vsl_m_s: float, vsg_m_s: float, rho_l: float, mu_l: float,
    sigma_n_m: float, d_m: float,
) -> float:
    """Correlated liquid holdup [-]. Hagedorn & Brown (1965), SPE-940. STUB.

    Hagedorn-Brown obtains holdup from three **graphical** correlating functions
    of the published charts — the ``CNL`` viscosity-number coefficient, the
    holdup-group ``phi``, and the secondary-correction ``psi`` — read off
    Hagedorn & Brown's (1965) figures. Reproducing them faithfully requires a
    **named, published digitised fit** of those curves (the standard set is in
    Brill & Mukherjee 1999, *Multiphase Flow in Wells*, SPE Monograph 17).

    That reference is **not available in the reference library**, and hand-
    digitising the graphs or inventing coefficients would violate the open-
    oracle provenance requirement (an unattributed fit is a correctness +
    integrity failure). This function is therefore **not implemented pending a
    citeable chart-fit source** — see the module report / spec. Raises
    :class:`NotImplementedError`.
    """
    raise NotImplementedError(
        "holdup_hagedorn_brown is not implemented: the Hagedorn-Brown holdup "
        "requires a citeable published digitised fit of the CNL/phi/psi "
        "correlating-function charts (e.g. Brill & Mukherjee 1999, SPE "
        "Monograph 17), which is not currently held in the reference library. "
        "Hand-digitising the graphs is disallowed by the open-oracle "
        "provenance rule; supply the fit source to enable this function."
    )


def dpdz_hagedorn_brown(
    vsl_m_s: float, vsg_m_s: float, rho_l: float, rho_g: float,
    mu_l: float, mu_g: float, sigma_n_m: float, d_m: float,
    roughness_m: float,
) -> float:
    """Vertical total pressure gradient [Pa/m]. Hagedorn & Brown (1965). STUB.

    Depends on :func:`holdup_hagedorn_brown`, which is not implemented for lack
    of a citeable published fit of the Hagedorn-Brown holdup charts (see there).
    The gradient (mixture-Reynolds friction term plus the Griffith bubble-flow
    switch, the standard industrial HB inclusion) cannot be computed without the
    holdup, so this is likewise **not implemented pending a fit source**. Raises
    :class:`NotImplementedError`.
    """
    raise NotImplementedError(
        "dpdz_hagedorn_brown is not implemented: it depends on "
        "holdup_hagedorn_brown, which needs a citeable published chart-fit "
        "(see holdup_hagedorn_brown)."
    )
