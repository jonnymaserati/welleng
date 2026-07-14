"""CoolProp real-EOS gas-property backend for kick tolerance (mixtures, CO2 / CCUS).

Optional -- requires the ``coolprop`` extra (``pip install 'welleng[kick]'``). The
Tier-0 default backend (:mod:`welleng.kick_tolerance.gas_z`) is a clean-room
Hall & Yarborough (1973) correlation for *pure methane*; that correlation cannot
represent CO2 + impurity influx. This module computes the real-gas Z-factor and
density for an arbitrary gas COMPOSITION (mole fractions) via CoolProp's HEOS
reference EOS (GERG-2008 for natural-gas mixtures), enabling CCUS / CO2 kicks.

CoolProp is MIT-licensed; use the native HEOS backend only (never the paid NIST
REFPROP backend). Cite: Bell, I. H. et al. (2014), Ind. Eng. Chem. Res. 53(6):2498.
"""
from __future__ import annotations

from typing import Dict, Tuple

# --- unit conversions (oilfield <-> SI) -------------------------------------
PSIA_TO_PA = 6894.757293168        # 1 psia -> Pa
RANKINE_TO_KELVIN = 5.0 / 9.0      # degR -> K
KG_M3_PER_PPG = 119.8264273        # 1 ppg (lb/US-gal) -> kg/m^3

# friendly component names -> CoolProp fluid names
_ALIASES = {
    "methane": "Methane", "c1": "Methane",
    "co2": "CarbonDioxide", "carbondioxide": "CarbonDioxide",
    "nitrogen": "Nitrogen", "n2": "Nitrogen",
    "ethane": "Ethane", "c2": "Ethane",
    "propane": "Propane", "c3": "Propane",
    "h2s": "HydrogenSulfide", "hydrogensulfide": "HydrogenSulfide",
    "water": "Water", "h2o": "Water",
}


def _coolprop_fluid_string(composition: Dict[str, float]) -> str:
    """Build a CoolProp fluid identifier from mole-fraction composition.

    Single component -> the plain CoolProp name (e.g. ``"CarbonDioxide"``).
    Mixture -> ``"HEOS::A[x]&B[y]"`` with fractions normalised to sum to 1.
    """
    if not composition:
        raise ValueError("composition is empty")
    total = float(sum(composition.values()))
    if total <= 0:
        raise ValueError("composition mole fractions must sum to > 0")
    resolved = [(_ALIASES.get(str(n).strip().lower(), n), f / total)
                for n, f in composition.items()]
    if len(resolved) == 1:
        return resolved[0][0]
    return "HEOS::" + "&".join(f"{name}[{frac:.10g}]" for name, frac in resolved)


def fluid_z_density(
    composition: Dict[str, float], p_psia: float, t_rankine: float
) -> Tuple[float, float]:
    """Real-gas (Z, density_ppg) for a gas ``composition`` at (P, T) via CoolProp.

    Parameters
    ----------
    composition : dict
        Component mole fractions, e.g. ``{"Methane": 0.9, "CO2": 0.1}`` (names are
        case-insensitive; common aliases like ``CO2``/``N2``/``H2S`` are mapped).
    p_psia, t_rankine : float
        Pressure [psia] and absolute temperature [degR].

    Returns
    -------
    (z, rho_ppg)
        Compressibility factor [-] and gas density [ppg mud-weight equivalent].
    """
    try:
        from CoolProp.CoolProp import PropsSI
    except ImportError as exc:  # optional dependency
        raise ImportError(
            "The CoolProp mixture gas backend requires the optional 'coolprop' "
            "dependency. Install it with: pip install 'welleng[kick]'."
        ) from exc

    fluid = _coolprop_fluid_string(composition)
    p_pa = p_psia * PSIA_TO_PA
    t_k = t_rankine * RANKINE_TO_KELVIN
    z = PropsSI("Z", "P", p_pa, "T", t_k, fluid)
    rho_kg_m3 = PropsSI("D", "P", p_pa, "T", t_k, fluid)
    return float(z), float(rho_kg_m3 / KG_M3_PER_PPG)
