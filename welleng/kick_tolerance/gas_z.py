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
) -> float:
    """Solve the Hall-Yarborough implicit equation for reduced density ``y``.

    Newton-Raphson from a small positive seed; ``y`` is confined to (0, 1).

    Raises
    ------
    ValueError
        If inputs are non-physical or the iteration fails to converge.
    """
    if p_psia <= 0.0 or t_rankine <= 0.0:
        raise ValueError("pressure and temperature must be positive")

    t = t_pc_rankine / t_rankine          # t = 1 / Tpr
    ppr = p_psia / p_pc_psia
    a = 0.06125 * ppr * t * math.exp(-1.2 * (1.0 - t) ** 2)

    y = 1.0e-3
    for _ in range(_MAX_ITER):
        f = _hy_residual(y, a, t)
        if abs(f) < _TOL:
            return y
        dy = f / _hy_residual_derivative(y, t)
        y -= dy
        if y <= 0.0:
            y = 1.0e-8
        elif y >= 1.0:
            y = 1.0 - 1.0e-8
    raise ValueError(
        f"Hall-Yarborough Newton-Raphson did not converge "
        f"(P={p_psia} psia, T={t_rankine} degR)"
    )


def hall_yarborough_z(
    p_psia: float,
    t_rankine: float,
    t_pc_rankine: float = METHANE_TPC_RANKINE,
    p_pc_psia: float = METHANE_PPC_PSIA,
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
    y = reduced_density(p_psia, t_rankine, t_pc_rankine, p_pc_psia)
    return a / y


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
