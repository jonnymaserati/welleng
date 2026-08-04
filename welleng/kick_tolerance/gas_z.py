"""Clean-room Hall & Yarborough (1973) real-gas Z-factor backend.

Implements the Hall-Yarborough compressibility-factor correlation from the
public source:

    Hall, K. R. & Yarborough, L. (1973). "A new equation of state for
    Z-factor calculations." Oil & Gas Journal 71(25): 82-92.

The correlation is a public, widely-published equation of state. It is
implemented here from the standard public formulation only -- no external,
third-party, or private data source is referenced.

Formulation
-----------
Given pseudo-reduced temperature/pressure

    Tpr = T / Tpc ,   Ppr = P / Ppc ,   t = 1 / Tpr ,

the reduced density ``y`` solves the implicit Hall-Yarborough equation

    F(y) = -A
           + (y + y^2 + y^3 - y^4) / (1 - y)^3
           - (14.76 t - 9.76 t^2 + 4.58 t^3) y^2
           + (90.7 t - 242.2 t^2 + 42.4 t^3) y^(2.18 + 2.82 t)
         = 0 ,

with the leading constant

    A = 0.06125 Ppr t exp[-1.2 (1 - t)^2] .

``F(y) = 0`` is solved by Newton-Raphson (analytic derivative). The Z-factor
follows as

    Z = A / y = 0.06125 Ppr t exp[-1.2 (1 - t)^2] / y .

The correlation is valid roughly for 0.1 <= Ppr <= ~24 and 1.15 <= Tpr <= 3.0.

Tier 0 gas: pure methane
------------------------
For Tier 0 the influx is treated as pure methane (gas gravity ~0.5539). The
pseudo-criticals are methane's critical constants (equivalently the
single-component Standing-Katz values):

    Tpc = 343.0 degR ,   Ppc = 667.0 psia ,   M = 16.043 lbm/lbmol .

Units
-----
Pressure psia; temperature degR; density returned in ppg (mud-weight
equivalent) using the field-unit gas law rho = P M / (Z R T) with the universal
gas constant R = 10.732 psia.ft^3 / (lbmol.degR) and 7.4805 gal/ft^3.
"""

from __future__ import annotations

import math
import warnings

# --- Public methane (Tier 0) constants --------------------------------------
METHANE_GAS_GRAVITY = 0.5539            # air = 1.0
METHANE_TPC_RANKINE = 343.0            # methane pseudo-critical temperature [degR]
METHANE_PPC_PSIA = 667.0              # methane pseudo-critical pressure    [psia]
METHANE_M_LBM_PER_LBMOL = 16.043       # methane molar mass                 [lbm/lbmol]

# --- Field-unit gas-law constants -------------------------------------------
R_FIELD = 10.732                       # gas constant [psia.ft^3 / (lbmol.degR)]
GAL_PER_FT3 = 7.4805                    # US gallons per cubic foot

_MAX_ITER = 100
_TOL = 1.0e-12


def _hy_residual(y: float, a: float, t: float) -> float:
    """Hall-Yarborough residual F(y) (root is the reduced density)."""
    return (
        -a
        + (y + y ** 2 + y ** 3 - y ** 4) / (1.0 - y) ** 3
        - (14.76 * t - 9.76 * t ** 2 + 4.58 * t ** 3) * y ** 2
        + (90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3) * y ** (2.18 + 2.82 * t)
    )


def _hy_residual_derivative(y: float, t: float) -> float:
    """Analytic dF/dy for the Newton-Raphson step."""
    return (
        (1.0 + 4.0 * y + 4.0 * y ** 2 - 4.0 * y ** 3 + y ** 4) / (1.0 - y) ** 4
        - (29.52 * t - 19.52 * t ** 2 + 9.16 * t ** 3) * y
        + (2.18 + 2.82 * t)
        * (90.7 * t - 242.2 * t ** 2 + 42.4 * t ** 3)
        * y ** (1.18 + 2.82 * t)
    )


def reduced_density(
    p_psia: float,
    t_rankine: float,
    t_pc_rankine: float = METHANE_TPC_RANKINE,
    p_pc_psia: float = METHANE_PPC_PSIA,
    y0: float = 1.0e-3,
) -> float:
    """Solve the Hall-Yarborough implicit equation for reduced density ``y``.

    Newton-Raphson from the seed ``y0`` (default a small positive value); ``y`` is
    confined to (0, 1). ``y0`` lets a caller WARM-START from a nearby prior solve
    (consecutive sub-steps in a gas-column integration have close pressures, so the
    previous ``y`` converges in ~2 iterations instead of ~5) -- the converged root
    is identical to 1e-12 regardless of the seed.

    The residual and its analytic derivative are inlined here and the
    temperature-dependent coefficient groups are precomputed once (not per Newton
    iteration): this is the same arithmetic as :func:`_hy_residual` /
    :func:`_hy_residual_derivative` but avoids ~5 Python function calls and repeated
    power evaluations per solve, which dominate the migration-engine hot path.

    Raises
    ------
    ValueError
        If inputs are non-physical or the iteration fails to converge.
    """
    if p_psia <= 0.0 or t_rankine <= 0.0:
        raise ValueError("pressure and temperature must be positive")

    t = t_pc_rankine / t_rankine          # t = 1 / Tpr
    ppr = p_psia / p_pc_psia
    # Validity-band guard: the H-Y correlation holds for ~0.1<=Ppr<=24, 1.15<=Tpr<=3.0.
    # Warn (once per call-site, per the warnings filter) on silent extrapolation -- e.g.
    # a cold shallow station in a migration temperature profile. The check is 4 float
    # comparisons; the extrapolated Z is still returned (warning, not error).
    tpr = 1.0 / t
    if not (0.1 <= ppr <= 24.0 and 1.15 <= tpr <= 3.0):
        warnings.warn(
            f"Hall-Yarborough Z evaluated outside its validity band "
            f"(Ppr={ppr:.3g} in [0.1, 24], Tpr={tpr:.3g} in [1.15, 3.0]); "
            f"the returned Z is extrapolated.",
            stacklevel=2,
        )
    a = 0.06125 * ppr * t * math.exp(-1.2 * (1.0 - t) ** 2)

    # t-dependent coefficient groups -- constant across Newton iterations.
    t2 = t * t
    t3 = t2 * t
    c_res = 14.76 * t - 9.76 * t2 + 4.58 * t3          # residual y^2 coefficient
    c_pow = 90.7 * t - 242.2 * t2 + 42.4 * t3          # residual/derivative power coef
    expo = 2.18 + 2.82 * t                             # residual power exponent
    c_der = 29.52 * t - 19.52 * t2 + 9.16 * t3         # derivative linear coefficient
    dexpo = 1.18 + 2.82 * t                            # derivative power exponent

    y = y0
    for _ in range(_MAX_ITER):
        omy = 1.0 - y
        y2 = y * y
        y3 = y2 * y
        y4 = y3 * y
        f = (
            -a
            + (y + y2 + y3 - y4) / omy ** 3
            - c_res * y2
            + c_pow * y ** expo
        )
        if abs(f) < _TOL:
            return y
        df = (
            (1.0 + 4.0 * y + 4.0 * y2 - 4.0 * y3 + y4) / omy ** 4
            - c_der * y
            + expo * c_pow * y ** dexpo
        )
        y -= f / df
        if y <= 0.0:
            y = 1.0e-8
        elif y >= 1.0:
            y = 1.0 - 1.0e-8
    raise ValueError(
        f"Hall-Yarborough Newton-Raphson did not converge "
        f"(P={p_psia} psia, T={t_rankine} degR)"
    )


M_AIR_LBM_PER_LBMOL = 28.9647   # dry air molar mass [lbm/lbmol]


def standing_pseudo_criticals(gas_gravity: float) -> tuple:
    """Natural-gas pseudo-criticals from gas gravity — Standing (1977)::

        T_pc = 168 + 325.g - 12.5.g^2   [degR]
        P_pc = 677 + 15.0.g - 37.5.g^2  [psia]

    ``gas_gravity`` is relative to AIR (air = 1.0), i.e. ``M_gas / 28.9647`` — a
    COMPOSITION, not a density. Pure methane is 0.5539.

    **Why this matters and is not a refinement.** Molar mass alone scales density but
    says nothing about Z; the Hall & Yarborough correlation needs pseudo-criticals.
    Using methane's for a heavier gas is internally inconsistent, and not by a little:
    at 5400 psi, the temperature at which a 0.686-gravity gas reaches 2.00 ppg is
    **179.9 degF on methane's pseudo-criticals and 194.6 degF on Standing's** — a
    14.7 degF error, in a diagnostic whose entire purpose is to expose an implausible
    temperature.

    Valid for sweet natural gases roughly 0.55-1.0. Sour gases (H2S, CO2) need a
    Wichert-Aziz correction, which is NOT applied here — pass explicit pseudo-criticals
    for those.
    """
    g = float(gas_gravity)
    return (168.0 + 325.0 * g - 12.5 * g * g,
            677.0 + 15.0 * g - 37.5 * g * g)


def hall_yarborough_z(
    p_psia: float,
    t_rankine: float,
    t_pc_rankine: float = METHANE_TPC_RANKINE,
    p_pc_psia: float = METHANE_PPC_PSIA,
    y0: float = 1.0e-3,
) -> float:
    """Real-gas Z-factor by the Hall & Yarborough (1973) correlation.

    Parameters
    ----------
    p_psia
        Absolute pressure [psia].
    t_rankine
        Absolute temperature [degR].
    t_pc_rankine, p_pc_psia
        Pseudo-critical temperature [degR] and pressure [psia]. Default to the
        Tier-0 pure-methane values.

    Returns
    -------
    float
        Compressibility factor Z = 0.06125 Ppr t exp[-1.2 (1 - t)^2] / y.
    """
    t = t_pc_rankine / t_rankine
    ppr = p_psia / p_pc_psia
    a = 0.06125 * ppr * t * math.exp(-1.2 * (1.0 - t) ** 2)
    y = reduced_density(p_psia, t_rankine, t_pc_rankine, p_pc_psia, y0)
    return a / y


def hall_yarborough_z_and_y(
    p_psia: float,
    t_rankine: float,
    t_pc_rankine: float = METHANE_TPC_RANKINE,
    p_pc_psia: float = METHANE_PPC_PSIA,
    y0: float = 1.0e-3,
) -> tuple:
    """Z factor AND the reduced density ``y`` that produced it.

    ``y`` is returned so a caller stepping through nearby pressures (a gas-column
    integration) can pass it as the ``y0`` warm-start of the next solve, cutting
    Newton iterations without changing the converged result.
    """
    t = t_pc_rankine / t_rankine
    ppr = p_psia / p_pc_psia
    a = 0.06125 * ppr * t * math.exp(-1.2 * (1.0 - t) ** 2)
    y = reduced_density(p_psia, t_rankine, t_pc_rankine, p_pc_psia, y0)
    return a / y, y


def gas_density_ppg(
    p_psia: float,
    t_rankine: float,
    z: float,
    molar_mass_lbm: float = METHANE_M_LBM_PER_LBMOL,
) -> float:
    """Real-gas density in ppg (mud-weight equivalent).

    rho = P M / (Z R T)  [lbm/ft^3], converted to ppg via 7.4805 gal/ft^3.
    """
    rho_lbm_per_ft3 = p_psia * molar_mass_lbm / (z * R_FIELD * t_rankine)
    return rho_lbm_per_ft3 / GAL_PER_FT3


def methane_properties(p_psia: float, t_rankine: float) -> tuple[float, float]:
    """Convenience: (Z, density_ppg) for pure methane at (P, T).

    Returns
    -------
    (z, rho_ppg)
        Z-factor [-] and gas density [ppg] for Tier-0 methane.
    """
    z = hall_yarborough_z(p_psia, t_rankine)
    rho = gas_density_ppg(p_psia, t_rankine, z)
    return z, rho


def sutton_pseudo_criticals(gas_gravity: float) -> tuple:
    """Natural-gas pseudo-criticals from gas gravity — Sutton (1985)::

        T_pc = 169.2 + 349.5.g - 74.0.g^2   [degR]
        P_pc = 756.8 - 131.0.g -  3.6.g^2   [psia]

    An alternative to :func:`standing_pseudo_criticals` fitted to a wider gravity
    range; preferred for heavier/associated gases (roughly ``g`` 0.57-1.68). Like
    Standing's, it carries no sour-gas (H2S/CO2) correction -- pass explicit
    pseudo-criticals (Wichert-Aziz corrected) for sour gases.
    """
    g = float(gas_gravity)
    return (169.2 + 349.5 * g - 74.0 * g * g,
            756.8 - 131.0 * g - 3.6 * g * g)


_PC_CORRELATIONS = {
    "standing": standing_pseudo_criticals,
    "sutton": sutton_pseudo_criticals,
}


def gas_z(
    p_psia: float,
    t_rankine: float,
    gas_gravity: float = None,
    composition: dict = None,
    t_pc_rankine: float = None,
    p_pc_psia: float = None,
    method: str = "auto",
    pc_correlation: str = "standing",
) -> float:
    """Unified **classic** real-gas Z-factor — Hall-Yarborough or CoolProp EOS.

    A single entry point that switches between the two classic/public backends:

    * **Hall & Yarborough** (fast correlation) for a hydrocarbon gas specified by
      ``gas_gravity`` (or explicit ``t_pc_rankine``/``p_pc_psia``);
    * **CoolProp** real-EOS (accurate) for a ``composition`` mole-fraction dict
      (mixtures / non-hydrocarbons like CO2 / N2 / H2S).

    Parameters
    ----------
    p_psia, t_rankine : float
        Absolute pressure [psia] and temperature [degR].
    gas_gravity : float, optional
        Gas gravity relative to air; pseudo-criticals are derived via
        ``pc_correlation``. Ignored if ``t_pc_rankine``/``p_pc_psia`` are given.
    composition : dict, optional
        Mole-fraction composition (e.g. ``{"Methane": 0.9, "CO2": 0.1}``) -> the
        CoolProp path.
    t_pc_rankine, p_pc_psia : float, optional
        Explicit pseudo-criticals for the Hall-Yarborough path (overrides
        ``gas_gravity``). Default to Tier-0 methane if nothing is supplied.
    method : {"auto", "hall_yarborough", "coolprop"}
        ``"auto"`` picks CoolProp when a ``composition`` is given, else
        Hall-Yarborough. An explicit choice forces the backend.
    pc_correlation : {"standing", "sutton"}
        Gas-gravity -> pseudo-critical correlation for the Hall-Yarborough path.

    Returns
    -------
    float
        The real-gas Z-factor [-].

    Notes
    -----
    CoolProp is an optional extra. If the CoolProp path is requested (or auto-
    selected) but CoolProp is not installed, this **falls back to Hall-Yarborough**
    when a ``gas_gravity``/pseudo-criticals are available (so the correlation path
    always works), and raises only if it has nothing to fall back on. This is the
    classic/public oracle; the fast analytical (Papay) form lives in welleng-api.
    """
    if method not in ("auto", "hall_yarborough", "coolprop"):
        raise ValueError(f"unknown method: {method!r}")
    if pc_correlation not in _PC_CORRELATIONS:
        raise ValueError(f"unknown pc_correlation: {pc_correlation!r}")

    resolved = method
    if resolved == "auto":
        resolved = "coolprop" if composition else "hall_yarborough"

    def _hy() -> float:
        if t_pc_rankine is not None and p_pc_psia is not None:
            tpc, ppc = t_pc_rankine, p_pc_psia
        elif gas_gravity is not None:
            tpc, ppc = _PC_CORRELATIONS[pc_correlation](gas_gravity)
        else:
            tpc, ppc = METHANE_TPC_RANKINE, METHANE_PPC_PSIA
        return hall_yarborough_z(p_psia, t_rankine, tpc, ppc)

    if resolved == "hall_yarborough":
        return _hy()

    # CoolProp path
    if composition is None:
        raise ValueError("method='coolprop' requires a composition dict")
    from .gas_z_coolprop import fluid_z_density
    try:
        z, _ = fluid_z_density(composition, p_psia, t_rankine)
        return z
    except ImportError:
        if gas_gravity is not None or (
            t_pc_rankine is not None and p_pc_psia is not None
        ):
            return _hy()
        raise ImportError(
            "CoolProp is required for the composition path and is not installed; "
            "install the optional extra, or pass gas_gravity/pseudo-criticals to "
            "fall back to Hall-Yarborough"
        )
