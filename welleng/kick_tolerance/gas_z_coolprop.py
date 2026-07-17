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

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# --- unit conversions (oilfield <-> SI) -------------------------------------
PSIA_TO_PA = 6894.757293168        # 1 psia -> Pa
RANKINE_TO_KELVIN = 5.0 / 9.0      # degR -> K
KG_M3_PER_PPG = 119.8264273        # 1 ppg (lb/US-gal) -> kg/m^3

_CACHE_DIR = Path.home() / ".cache" / "welleng" / "ztables"

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


# Curated influx-fluid shortlist for a picker (label + engine-consumable mole-
# fraction composition, in CoolProp canonical names). The single source of truth
# for the API's /kick/fluids endpoint -- a thin serializer over this, NOT a table
# maintained API-side. The full fluid catalogue is CoolProp's own
# ``get_global_param_string("FluidsList")`` (enumerate live), not curated here.
# Natural-gas / sour compositions are REPRESENTATIVE EXAMPLES (so labelled), not
# authoritative standards. Pure methane routes to the free Hall-Yarborough path;
# any other composition uses the CoolProp real-EOS backend.
FLUID_PRESETS = (
    {"label": "Methane (CH4)", "composition": {"Methane": 1.0}},
    {"label": "Carbon dioxide (CO2)", "composition": {"CarbonDioxide": 1.0}},
    {"label": "Nitrogen (N2)", "composition": {"Nitrogen": 1.0}},
    {"label": "Natural gas (example blend)",
     "composition": {"Methane": 0.90, "Ethane": 0.05, "Propane": 0.02,
                     "Nitrogen": 0.02, "CarbonDioxide": 0.01}},
    {"label": "Sour gas (example)",
     "composition": {"Methane": 0.85, "HydrogenSulfide": 0.10,
                     "CarbonDioxide": 0.05}},
)


def fluid_presets() -> list:
    """Curated influx-fluid shortlist: ``[{label, composition}, ...]``.

    ``composition`` is a mole-fraction dict in CoolProp canonical names, consumed
    as-is by ``KickInputs.fluid`` / ``analytical_kick_tolerance(gas_composition=)``.
    The natural-gas / sour entries are representative examples, not standards. For
    the full fluid list use CoolProp's ``get_global_param_string("FluidsList")``.
    """
    return [{"label": p["label"], "composition": dict(p["composition"])}
            for p in FLUID_PRESETS]


def fluid_aliases() -> Dict[str, str]:
    """Friendly component name -> CoolProp fluid name (e.g. ``co2`` ->
    ``CarbonDioxide``). Case-insensitive keys; the same map used to resolve a
    composition before it reaches CoolProp."""
    return dict(_ALIASES)


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


class ZTable:
    """Precomputed real-gas ``Z(P,T)`` and density surface for a fixed gas
    composition -- the fast path for the CoolProp backend.

    A single CoolProp GERG-2008 mixture flash costs ~15 ms, so calling it inside
    the ~700-evaluation kick-tolerance solve is ~12 s / solve (unusable). Instead
    grid the (P, T) box ONCE (Z and density are smooth), then bilinearly
    interpolate in the hot loop (~2 us/lookup, interp error ~1e-6 vs CoolProp).
    The grid is disk-cached per (composition, box, resolution), so a project's
    fixed composition is built once and reused forever; pure methane stays on the
    fast Hall-Yarborough backend and does not need this.

    Parameters
    ----------
    composition : dict
        Mole fractions (same as :func:`fluid_z_density`).
    p_psia_range, t_rankine_range : (lo, hi)
        The pressure [psia] and temperature [degR] box to tabulate. Pad it beyond
        the case's surface..bottom-hole range so lookups never extrapolate.
    n_p, n_t : int
        Grid resolution (default 40 x 10 -- Z is smooth, so this is ample).
    """

    def __init__(self, composition: Dict[str, float],
                 p_psia_range: Tuple[float, float], t_rankine_range: Tuple[float, float],
                 n_p: int = 40, n_t: int = 10, cache: bool = True):
        self.p_grid = np.linspace(float(p_psia_range[0]), float(p_psia_range[1]), n_p)
        self.t_grid = np.linspace(float(t_rankine_range[0]), float(t_rankine_range[1]), n_t)
        key = hashlib.sha1(json.dumps(
            [sorted(composition.items()), self.p_grid.tolist(), self.t_grid.tolist()],
            sort_keys=True).encode()).hexdigest()[:16]
        cache_file = _CACHE_DIR / f"{key}.npz"
        if cache and cache_file.exists():
            d = np.load(cache_file)
            self._Z, self._RHO = d["Z"], d["RHO"]
        else:
            self._Z, self._RHO = self._build(composition)
            if cache:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez(cache_file, Z=self._Z, RHO=self._RHO)

    def _build(self, composition):
        from CoolProp.CoolProp import AbstractState, PT_INPUTS
        resolved = [(_ALIASES.get(str(n).strip().lower(), n), f)
                    for n, f in composition.items()]
        tot = sum(f for _, f in resolved)
        st = AbstractState("HEOS", "&".join(n for n, _ in resolved))
        st.set_mole_fractions([f / tot for _, f in resolved])
        Z = np.empty((self.t_grid.size, self.p_grid.size))
        RHO = np.empty_like(Z)
        for i, T in enumerate(self.t_grid):
            for j, P in enumerate(self.p_grid):
                st.update(PT_INPUTS, P * PSIA_TO_PA, T * RANKINE_TO_KELVIN)
                Z[i, j] = st.compressibility_factor()
                RHO[i, j] = st.rhomass() / KG_M3_PER_PPG      # ppg
        return Z, RHO

    def _bilinear(self, grid, p, t):
        # clamp to the box (no extrapolation), then bilinear on the two grids
        pv = np.clip(p, self.p_grid[0], self.p_grid[-1])
        tv = np.clip(t, self.t_grid[0], self.t_grid[-1])
        jp = np.clip(np.searchsorted(self.p_grid, pv) - 1, 0, self.p_grid.size - 2)
        it = np.clip(np.searchsorted(self.t_grid, tv) - 1, 0, self.t_grid.size - 2)
        p0, p1 = self.p_grid[jp], self.p_grid[jp + 1]
        t0, t1 = self.t_grid[it], self.t_grid[it + 1]
        fp = (pv - p0) / (p1 - p0)
        ft = (tv - t0) / (t1 - t0)
        g00 = grid[it, jp]; g01 = grid[it, jp + 1]
        g10 = grid[it + 1, jp]; g11 = grid[it + 1, jp + 1]
        return ((1 - ft) * ((1 - fp) * g00 + fp * g01)
                + ft * ((1 - fp) * g10 + fp * g11))

    def z(self, p_psia, t_rankine):
        """Interpolated compressibility factor Z(P, T)."""
        return self._bilinear(self._Z, np.asarray(p_psia, float), np.asarray(t_rankine, float))

    def rho_ppg(self, p_psia, t_rankine):
        """Interpolated gas density [ppg] at (P, T)."""
        return self._bilinear(self._RHO, np.asarray(p_psia, float), np.asarray(t_rankine, float))
