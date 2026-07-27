"""Gas-migration kick-tolerance engine (single migrating bubble, whole open hole).

WHAT THIS IS
------------
The Tier-0 single-bubble kick-tolerance check (``kick_tolerance.py``, transcribed
from SPE-208788-PA Appendix A) tests the tolerable influx at ONE station -- the
casing shoe. This engine is an **extension BEYOND that static NOGEPA single-shoe
method**: it tracks the single gas bubble as it MIGRATES up the annulus and
requires the imposed pressure to stay inside the pore/fracture (PP-FP) window at
EVERY exposed open-hole depth, at every migration step.

It is a **defensible assembly of referenced pieces** -- no single public source
packages the whole thing, so each piece is cited where it is used:

  * Hold BHP constant at the kill (circulating) bottom-hole pressure --
    Driller's-method well control, API RP 59 (Recommended Practice for Well
    Control Operations) 2nd ed. Sec. 4.10; SPE-202426 (managed-pressure /
    well-control constant-BHP kill).
  * March the single bubble up the annulus, expanding by the real-gas law --
    single-bubble gas migration, API RP 59 Sec. 4.8.7.3 (gas expansion on the
    way up); the closed-form single-bubble treatment of SPE-208788-PA.
  * Keep every EXPOSED open-hole formation between pore and fracture pressure --
    the barrier envelope, API RP 59 Sec. 12.5 (do not underbalance a permeable
    zone; do not exceed formation breakdown). Cased intervals are protected and
    are not checked.
  * Flag when the bubble grows longer than the open hole / BHA it must migrate
    past -- SPE-140113 (bubble-length vs. open-hole-length limit).
  * NOGEPA-50 (Netherlands well-control standard) frames the single-shoe kick
    tolerance this engine extends to the whole exposed open hole.

Modelling stance (safe-side, explicit)
--------------------------------------
  * SINGLE BUBBLE, no slip / no dispersion: one contiguous gas interval of
    conserved mass. Treating the influx as a single coherent bubble is the
    conservative (safe-side) idealisation -- a dispersed/percolating influx
    imposes LESS peak pressure at any shallow depth than one coherent light
    column, so the single bubble bounds the fracture risk.
  * Real-gas expansion via Boyle + Z + T from the bottom-hole state:
        V(P,T,Z) = V_bh * (P_bh * Z * T) / (P * Z_bh * T_bh).
    Z(P) is taken from the clean-room Hall & Yarborough (1973) methane backend
    (``gas_z.py``), so the bubble grows correctly as it rises. This is the
    dominant migration effect and it is modelled pressure-dependently.
  * GAS-COLUMN HYDROSTATIC uses the LOCAL, pressure- AND temperature-dependent
    gas density,
    rho_gas(P, d) = rho_gas_bh * (P * Z_bh * T_bh) / (P_bh * Z(P, T(d)) * T(d)),
    which lightens up-hole as P falls. The temperature enters via ``temp_profile``
    (default ``None`` = ISOTHERMAL at the bottom-hole T, so T cancels and this is
    exactly the previous rho_gas_bh * (P * Z_bh) / (P_bh * Z(P))); a supplied
    profile (``linear_temp_profile`` two-point gradient, or a full (tvd, T) table)
    makes Z and density track the true T(depth). A lighter up-hole column means a
    SMALLER
    pressure drop across the gas and hence a HIGHER imposed pressure at and above
    the gas top (and the shoe) -- the safe-side direction for a fracture barrier.
    ``pressure_at_depth`` / ``migrate`` expose ``gas_density_mode``:
    ``"conservative"`` (DEFAULT) holds the gas-TOP (lightest) density constant
    over the whole gas column -- the highest-pressure, safe-side BOUND;
    ``"exact"`` integrates the true local density -- the correct value, which
    lies just below the bound. Using the bottom-hole density everywhere would be
    non-conservative and is deliberately NOT an option.
  * OIL-BASED-MUD gas SOLUBILITY is OUT OF SCOPE -- the influx is treated as a
    free-gas bubble throughout (a water-based-mud / free-gas assumption). In OBM
    a dissolved influx breaks out shallow and expands abruptly; that is a
    separate model.
  * TVD-based, capacity-per-section. MD / deviation coupling (the true annular
    length a given TVD interval spans in a deviated well) is a LATER refinement;
    here each section carries an annular capacity [bbl/ft] against TVD directly.

Units (oilfield)
----------------
Depths TVD [ft]; densities [ppg]; pressures [psi]; temperature [degR]; annular
capacity [bbl/ft]; gas volume [bbl]. Gravitational constant
g = 0.0521 psi.ppg^-1.ft^-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

import numpy as np

try:  # package-relative import (brief spec: from .gas_z import ...)
    from .gas_z import hall_yarborough_z, hall_yarborough_z_and_y, gas_density_ppg
except ImportError:  # flat-script / pytest-from-directory execution
    from gas_z import hall_yarborough_z, hall_yarborough_z_and_y, gas_density_ppg

# --- Constant (public, oilfield units) --------------------------------------
G_PSI_PER_PPG_FT = 0.0521  # gravitational constant g [psi.ppg^-1.ft^-1]

# Below this pressure the Hall-Yarborough correlation leaves its validity band
# (Ppr >~ 0.1); near-atmospheric gas is ~ideal, so we fall back to Z = 1.
_HY_MIN_PSI = 100.0

# A ppg profile is either a callable tvd->ppg, or a (tvd_array, ppg_array) table.
ProfileLike = Union[Callable[[float], float], "tuple[Sequence[float], Sequence[float]]"]

# A temperature profile is a callable tvd->T[degR], a (tvd_array, T_rankine_array)
# table, or None (ISOTHERMAL at the bottom-hole temperature -- see below).
TempProfileLike = Union[
    None, Callable[[float], float], "tuple[Sequence[float], Sequence[float]]"
]


# ============================================================================
# Geometry
# ============================================================================
@dataclass
class WellSection:
    """One annular section, described against TVD.

    Parameters
    ----------
    top_tvd, bottom_tvd
        Section extent in true vertical depth [ft] (top shallower than bottom).
    annular_capacity_bbl_per_ft
        Annular capacity of the section [bbl/ft] -- the gas length in this
        section is volume / this capacity. Capacity changes discretely at
        section boundaries (e.g. casing shoe, BHA top).
    is_open_hole
        True for an exposed open-hole formation (subject to the PP-FP envelope
        check); False for a cased/protected interval (not checked).
    top_md, bottom_md
        Optional along-hole extent of the same section [ft]. Annular capacity
        is a volume per unit of ALONG-HOLE length, so in a deviated well the
        volume held between two TVDs is ``capacity * dMD``, not
        ``capacity * dTVD``. Supply these (from a survey) and the section
        reports :attr:`capacity_per_tvd_ft` accordingly. Leave them ``None``
        and the section is treated as vertical (``dMD == dTVD``), which is
        the pre-0.27 behaviour exactly.
    burst_pressure_psi
        Optional ALLOWABLE internal pressure for a cased section [psi] -- the
        published minimum internal yield pressure already reduced by a design
        factor. Supply it and the imposed pressure is checked against it over
        the cased interval, alongside the pore/fracture check in open hole;
        leave it ``None`` (the default) and cased intervals are unchecked, as
        they were before 0.27.

        INDICATIVE ONLY -- this is not a casing design tool. It credits NO
        external backup (the full internal pressure is resisted by the pipe,
        which is the safe-side worst case), and it accounts for no axial load,
        bending, temperature derating, wear or connection rating. A real burst
        design is a differential calculation against a backup profile over the
        load cases. Use this to notice that the casing may bind before the
        formation does, not to size a string.

    Pressure is a function of TVD and volume is a function of MD; a section
    carries both extents so the two integrals stay in their own domains. The
    ratio ``dMD/dTVD`` is the section-mean ``sec(inc)``, so a section should be
    short enough that inclination is near-constant across it -- build them with
    :func:`welleng.kick_tolerance.geometry.sections_from_architecture`, which
    splits at the union of geometry changes and survey stations.
    """

    top_tvd: float
    bottom_tvd: float
    annular_capacity_bbl_per_ft: float
    is_open_hole: bool
    top_md: float | None = None
    bottom_md: float | None = None
    burst_pressure_psi: float | None = None

    @property
    def md_extent(self) -> float:
        """Along-hole length of the section [ft]; the TVD extent if unset."""
        if self.top_md is None or self.bottom_md is None:
            return self.bottom_tvd - self.top_tvd
        return self.bottom_md - self.top_md

    @property
    def capacity_per_tvd_ft(self) -> float:
        """Annular capacity per foot of TVD [bbl/ft].

        ``annular_capacity_bbl_per_ft * (dMD / dTVD)`` -- the along-hole
        capacity scaled by the section-mean ``sec(inc)``. Equals the raw
        capacity for a vertical section. Every volume in the TVD-domain
        engines is ``this * dTVD``.

        Raises
        ------
        ValueError
            If the section has zero TVD extent (a horizontal section holds
            volume across no TVD at all, which the TVD-domain formulation
            cannot represent).
        """
        d_tvd = self.bottom_tvd - self.top_tvd
        if d_tvd <= 0.0:
            raise ValueError(
                "WellSection has zero TVD extent: a horizontal section holds "
                "volume over no TVD and cannot be expressed as a capacity per "
                "foot of TVD. Split the well above the horizontal, or use the "
                "MD-domain march."
            )
        return self.annular_capacity_bbl_per_ft * self.md_extent / d_tvd


# ============================================================================
# Profiles (PP / FP)  -- callable OR (tvd, ppg) table
# ============================================================================
def _as_ppg_callable(profile: ProfileLike) -> Callable[[np.ndarray], np.ndarray]:
    """Return a vectorised tvd->ppg callable from a callable or a (tvd, ppg) table."""
    if callable(profile):
        return lambda d: np.asarray(profile(d), dtype=float)
    tvd_arr, ppg_arr = profile
    tvd = np.asarray(tvd_arr, dtype=float)
    ppg = np.asarray(ppg_arr, dtype=float)
    order = np.argsort(tvd)
    tvd, ppg = tvd[order], ppg[order]
    return lambda d: np.interp(np.asarray(d, dtype=float), tvd, ppg)


def _profile_breakpoints(profile: ProfileLike) -> list:
    """TVD breakpoints of a PP/FP profile -- the depths where its gradient can turn.

    A ``(tvd, ppg)`` table returns its tvd knots; a callable has no discrete
    breakpoints (returns ``[]``), so fast mode falls back to its per-section grid.
    Used to anchor fast-mode check depths / bubble positions to the interfaces where
    the binding constraint can change.
    """
    if callable(profile):
        return []
    tvd_arr, _ = profile
    return [float(x) for x in np.asarray(tvd_arr, dtype=float)]


def ppg_to_psi(rho_ppg: np.ndarray, depth_ft: np.ndarray) -> np.ndarray:
    """Gradient pressure of a mud-weight-equivalent column: g * ppg * TVD [psi]."""
    return G_PSI_PER_PPG_FT * np.asarray(rho_ppg) * np.asarray(depth_ft)


# ============================================================================
# Temperature profile (non-isothermal gas) -- callable OR (tvd, degR) table
# ============================================================================
def _as_temp_callable(
    profile: TempProfileLike, t_default: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a vectorised tvd->T[degR] callable.

    Same coercion pattern as the PP/FP profiles, with one extra case:

      * ``None`` -> ISOTHERMAL: a constant equal to ``t_default`` (the bottom-hole
        temperature ``T_bh_rankine``). This reproduces the previous isothermal
        behaviour EXACTLY -- the local temperature equals the bottom-hole
        temperature everywhere, so the T-ratio in the gas density is exactly 1.
      * a callable ``tvd -> T_rankine``.
      * a ``(tvd_array, T_rankine_array)`` table (numpy-interpolated; a full /
        field profile -- the ADVANCED case).
    """
    if profile is None:
        t = float(t_default)
        # Isothermal: return the scalar for scalar/0-d input (the hot-loop case,
        # called ~1e7 times as float(temp_fn(depth))) instead of allocating a fresh
        # np.full array each time; still broadcast for genuine array input.
        return lambda d: t if np.ndim(d) == 0 else np.full(np.shape(d), t)
    if callable(profile):
        return lambda d: np.asarray(profile(d), dtype=float)
    tvd_arr, t_arr = profile
    tvd = np.asarray(tvd_arr, dtype=float)
    tt = np.asarray(t_arr, dtype=float)
    order = np.argsort(tvd)
    tvd, tt = tvd[order], tt[order]
    return lambda d: np.interp(np.asarray(d, dtype=float), tvd, tt)


def linear_temp_profile(
    shoe_tvd: float,
    shoe_temp_rankine: float,
    td_tvd: float,
    td_temp_rankine: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Two-point (shoe + TD) linear temperature gradient -- the BASIC case.

    Returns a callable ``tvd -> T_rankine`` for a straight-line geothermal
    gradient anchored at the casing shoe and at TD. Extrapolated linearly outside
    ``[shoe_tvd, td_tvd]`` (so the gradient continues to surface, TVD=0, as the
    bubble rises). For a full / field temperature survey pass a
    ``(tvd_array, T_rankine_array)`` table instead (the ADVANCED case).

    Parameters
    ----------
    shoe_tvd, td_tvd
        Anchor depths [ft TVD]; ``td_tvd`` must differ from ``shoe_tvd``.
    shoe_temp_rankine, td_temp_rankine
        Temperatures at those depths [degR]. A geothermal gradient has
        ``td_temp_rankine > shoe_temp_rankine`` (hotter with depth).
    """
    shoe_tvd = float(shoe_tvd)
    td_tvd = float(td_tvd)
    if td_tvd == shoe_tvd:
        raise ValueError("linear_temp_profile: shoe_tvd and td_tvd must differ")
    t_shoe = float(shoe_temp_rankine)
    t_td = float(td_temp_rankine)
    slope = (t_td - t_shoe) / (td_tvd - shoe_tvd)

    def _profile(d):
        d = np.asarray(d, dtype=float)
        return t_shoe + slope * (d - shoe_tvd)

    return _profile


# ============================================================================
# Pressure profile (public -- reused by the migration loop AND by callers/tests)
# ============================================================================
def pressure_at_depth(
    depth_ft: Union[float, np.ndarray],
    *,
    gas_top_tvd: float,
    gas_bottom_tvd: float,
    bottom_tvd: float,
    bhp_psi: float,
    rho_mud_ppg: float,
    gas_bh,
    gas_density_mode: str = "conservative",
    temp_profile: TempProfileLike = None,
    geothermal: TempProfileLike = None,
    g: float = G_PSI_PER_PPG_FT,
    n_sub: int = 40,
    z_fn=None,
) -> Union[float, np.ndarray]:
    """Imposed pressure at ``depth_ft`` for BHP held constant at the bottom.

    ``z_fn`` (optional): a real-gas Z provider ``(P_psi, T_rankine) -> Z`` -- e.g. a
    precomputed CoolProp :class:`~welleng.kick_tolerance.gas_z_coolprop.ZTable` for a
    mixture / CO2 / CCUS influx. When ``None`` (default) the clean-room
    Hall-Yarborough methane backend is used with a warm-started Newton (behaviour
    unchanged, bit-for-bit).

    Marching UP from the bottom, hydrostatic is removed: a mud gradient outside
    the gas interval and a gas gradient inside [gas_top, gas_bottom]. The local
    gas density is NON-ISOTHERMAL::

        rho_gas(P, d) = rho_gas_bh * (P * Z_bh * T_bh) / (P_bh * Z(P, T(d)) * T(d))

    with ``Z(P, T(d))`` from the clean-room Hall & Yarborough (1973) methane
    backend at the LOCAL temperature ``T(d)``. ``gas_bh = (P_bh, T_bh_rankine,
    Z_bh, rho_gas_bh)``. When ``temp_profile`` is ``None`` the local temperature
    equals ``T_bh`` everywhere, T cancels, and this reduces EXACTLY to the
    previous isothermal density ``rho_gas_bh * (P * Z_bh) / (P_bh * Z(P))``.
    Vectorised over ``depth_ft``.

    Parameters
    ----------
    temp_profile : None | callable | (tvd_array, T_rankine_array)
        Temperature profile T(d) [degR]. ``None`` (DEFAULT) is ISOTHERMAL at the
        bottom-hole temperature ``T_bh_rankine`` (from ``gas_bh``) -- reproduces
        the previous behaviour exactly. A callable ``tvd -> T_rankine`` or a
        ``(tvd, T_rankine)`` table sets a depth-varying temperature (see
        ``linear_temp_profile`` for the basic two-point gradient). Hotter gas
        up-hole is lighter (lower density) -> a HIGHER pressure at/above the gas.
    geothermal : None | callable | (tvd_array, T_rankine_array)
        The field geothermal gradient, used as the DEFAULT temperature when
        ``temp_profile`` is not given. Resolution is three-tier: an explicit
        ``temp_profile`` wins; else ``geothermal`` (the go-to default when a
        gradient is known); else isothermal at ``T_bh_rankine``.
    gas_density_mode : {"conservative", "exact"}
        Gas-column density treatment (each evaluated at the local T(d)):

        * ``"conservative"`` (DEFAULT, safe-side) -- use the gas density at the
          gas TOP (the lowest pressure -> lightest gas) as a CONSTANT for the
          whole gas column. The lightest-possible column gives the smallest
          pressure drop across the gas and hence the HIGHEST pressure at and
          above the gas top (and the shoe). This is the safe-side bound for a
          fracture barrier: it never under-states the shallow loading. Slightly
          over-conservative vs. the true integrated column.
        * ``"exact"`` -- integrate the TRUE pressure-dependent local density down
          the column (forward Euler). This is the technically-correct / true-
          average result; it lies BELOW the conservative bound.

        Ordering at any depth at/above the gas:
        ``P(conservative) >= P(exact) >= P(bottom-hole-constant)``.
    n_sub
        Sub-steps for the exact forward-Euler integration down the gas interval
        (also used to seed the gas-top pressure for the conservative mode).
    """
    if gas_density_mode not in ("conservative", "exact"):
        raise ValueError(
            f"gas_density_mode must be 'conservative' or 'exact', got {gas_density_mode!r}"
        )
    P_bh, T_bh_r, Z_bh, rho_gas_bh = gas_bh
    if temp_profile is None:
        temp_profile = geothermal          # geothermal is the default when supplied
    temp_fn = _as_temp_callable(temp_profile, T_bh_r)  # still None -> isothermal at T_bh
    d = np.asarray(depth_ft, dtype=float)
    scalar = d.ndim == 0

    # Pressure at the gas bottom: mud column below the bubble.
    P_gb = bhp_psi - g * rho_mud_ppg * (bottom_tvd - gas_bottom_tvd)

    gas_len = gas_bottom_tvd - gas_top_tvd
    if gas_len <= 1e-9:
        z_asc = np.array([gas_top_tvd])
        P_asc = np.array([P_gb])
        P_gt = P_gb
    else:
        # EXACT pass: integrate UP from gas_bottom to gas_top with the local
        # (pressure-dependent) density. Forward Euler; density at the sub-step
        # base. This also yields the gas-top pressure that seeds "conservative".
        n = max(2, int(n_sub))
        zs = np.linspace(gas_bottom_tvd, gas_top_tvd, n + 1)  # depth descending
        Ps = np.empty(n + 1)
        Ps[0] = P_gb
        y_seed = 1.0e-3                                  # warm-start the H-Y Newton
        for k in range(n):
            dz = zs[k] - zs[k + 1]                       # +ve (going up)
            Pk = max(Ps[k], 1.0)
            Tk = float(temp_fn(zs[k]))                    # local T at sub-step base
            if z_fn is None:
                Zk, y_seed = _z_and_y_at(Pk, Tk, y_seed)  # seed next solve from this y
            else:
                Zk = float(z_fn(Pk, Tk))                  # real-gas provider (CoolProp)
            # rho(P,d) = rho_bh * P*Z_bh*T_bh / (P_bh*Z(P,T(d))*T(d)); the trailing
            # (T_bh/Tk) is EXACTLY 1.0 when isothermal -> old value bit-for-bit.
            rho_local = rho_gas_bh * Pk * Z_bh / (P_bh * Zk) * (T_bh_r / Tk)
            Ps[k + 1] = Ps[k] - rho_local * g * dz
        P_gt_exact = Ps[-1]

        if gas_density_mode == "exact":
            z_asc = zs[::-1]
            P_asc = Ps[::-1]
            P_gt = P_gt_exact
        else:  # "conservative": constant gas-TOP (lightest) density
            # rho at the gas-top pressure; a couple of refresh iterations make
            # rho_top self-consistent with the updated (lighter) gas-top P.
            P_top = max(P_gt_exact, 1.0)
            T_top = float(temp_fn(gas_top_tvd))    # local T at the gas top
            y_top = 1.0e-3                          # warm-start across the 3 refreshes
            for _ in range(3):
                if z_fn is None:
                    Z_top, y_top = _z_and_y_at(P_top, T_top, y_top)
                else:
                    Z_top = float(z_fn(P_top, T_top))
                # (T_bh/T_top) is EXACTLY 1.0 when isothermal -> old value bit-for-bit.
                rho_top = rho_gas_bh * P_top * Z_bh / (P_bh * Z_top) * (T_bh_r / T_top)
                P_top = max(P_gb - rho_top * g * gas_len, 1.0)
            P_gt = P_gb - rho_top * g * gas_len  # >= exact P_gt (lighter column)
            # Linear profile inside the gas at the constant rho_top.
            z_asc = np.array([gas_top_tvd, gas_bottom_tvd])
            P_asc = np.array([P_gt, P_gb])

    out = np.where(
        d >= gas_bottom_tvd,
        bhp_psi - g * rho_mud_ppg * (bottom_tvd - d),        # below the gas
        np.where(
            d <= gas_top_tvd,
            P_gt - g * rho_mud_ppg * (gas_top_tvd - d),      # above the gas
            np.interp(d, z_asc, P_asc),                      # inside the gas
        ),
    )
    return float(out) if scalar else out


# ============================================================================
# Result dataclasses  (the `steps` list IS the animation trajectory)
# ============================================================================
@dataclass
class MigrationStep:
    """One migration step -- one frame of the animation trajectory."""

    gas_top_tvd: float          # shallow boundary of the bubble [ft]
    gas_bottom_tvd: float       # deep boundary of the bubble     [ft]
    gas_length_ft: float        # gas_bottom - gas_top            [ft]
    min_fp_margin_psi: float    # min over exposed depths of FP(d) - P(d) [psi]
    binding_tvd: float          # exposed depth achieving that minimum    [ft]
    p_at_binding_psi: float     # imposed pressure at binding_tvd          [psi]
    # Shut-in gauge readings for THIS bubble position (well-control kill sheet):
    sidp_psi: float             # shut-in drill-pipe pressure [psi]
    sicp_psi: float             # shut-in casing (annulus) pressure [psi]
    #  SIDP = BHP - g.rho_mud.bottom_tvd -- drill pipe full of MUD (influx in the
    #    annulus, not the string: the standard kick-in-annulus assumption). It is
    #    position-INDEPENDENT, so it is constant across the walk; out of scope if
    #    the influx entered the drill string.
    #  SICP = imposed pressure evaluated at surface (depth 0) for this step -- the
    #    existing annulus profile at the top. Per step it equals
    #    SIDP + (g_mud - g_gas).h_gas with the CURRENT gas length/density. Under
    #    constant BHP the bubble EXPANDS as it rises (h_gas grows, rho_gas falls),
    #    so SICP RISES up the walk -- the steps are a SICP schedule, not a flat
    #    line. IDEAL gas (Z=1, isothermal) expands by Boyle only; REAL gas adds the
    #    Z(P,T) correction -- the two give different schedules (the differentiator),
    #    and agree at the initial (deepest) bubble position where no expansion has
    #    happened yet (the well-control single-bubble hand-calc value).


@dataclass
class MigrationResult:
    """Outcome of a bubble migration sweep.

    ``steps`` is the ordered per-step animation trajectory (bottom -> surface).
    The scalar fields summarise the worst (binding) point over the whole sweep.
    """

    steps: list                 # list[MigrationStep] -- animation data
    within_envelope: bool       # PP(d) <= P(d) <= FP(d) at every exposed depth+step
    min_fp_margin_psi: float    # min over all steps of FP - P (binding constraint)
    binding_tvd: float          # exposed depth of the global minimum          [ft]
    binding_step: int           # index into `steps` of the global minimum
    bha_length_exceeded: bool   # bubble length ever exceeded the open hole it passes

    # Internal context (bottom-hole state + mud), so callers/tests can recompute
    # the pressure profile for any step via `pressure_at_depth`. Not part of the
    # animation payload.
    _ctx: dict = field(default_factory=dict, repr=False)


# ============================================================================
# Bottom-hole gas state resolution
# ============================================================================
def _resolve_bh_state(gas_bh_state, bhp_psi):
    """Return (P_bh, T_bh_rankine, Z_bh, rho_gas_ppg).

    ``gas_bh_state`` = (P_bh, T_bh_rankine, Z_bh, rho_gas_ppg). ``P_bh`` may be
    None (defaults to ``bhp_psi``); ``Z_bh`` / ``rho_gas_ppg`` may be None and
    are then COMPUTED by the clean-room Hall & Yarborough (1973) methane backend
    -- keeping the engine importable without CoolProp. ``T_bh_rankine`` is
    required (no temperature profile is otherwise supplied).
    """
    P_bh, T_bh_r, Z_bh, rho_gas = gas_bh_state
    if T_bh_r is None:
        raise ValueError("gas_bh_state must supply T_bh_rankine (bottom-hole temperature)")
    if P_bh is None:
        P_bh = bhp_psi
    if Z_bh is None:
        Z_bh = hall_yarborough_z(P_bh, T_bh_r)
    if rho_gas is None:
        rho_gas = gas_density_ppg(P_bh, T_bh_r, Z_bh)
    return float(P_bh), float(T_bh_r), float(Z_bh), float(rho_gas)


def _z_at(p_psi: float, t_rankine: float) -> float:
    """Methane Z at (P, T); ~ideal (Z=1) below the Hall-Yarborough validity band."""
    if p_psi < _HY_MIN_PSI:
        return 1.0
    try:
        return hall_yarborough_z(p_psi, t_rankine)
    except ValueError:
        return 1.0


def _z_and_y_at(p_psi: float, t_rankine: float, y0: float) -> tuple:
    """``(Z, y)`` at (P, T), warm-started from ``y0``. Same Z=1 fallback as
    :func:`_z_at` below the H-Y validity band (``y0`` passed straight through so the
    next nearby solve still gets a sensible seed)."""
    if p_psi < _HY_MIN_PSI:
        return 1.0, y0
    try:
        return hall_yarborough_z_and_y(p_psi, t_rankine, y0=y0)
    except ValueError:
        return 1.0, y0


# ============================================================================
# Filling the expanded volume into the annulus (section capacities)
# ============================================================================
def _fill_down(gas_top: float, volume_bbl: float, sections_sorted, bottom_tvd: float):
    """Place ``volume_bbl`` of gas from ``gas_top`` DOWNWARD, section by section.

    Capacity changes discretely per section, so the gas length is accumulated
    across the sections it spans. Returns (gas_bottom_tvd, gas_length_ft). If the
    volume cannot fit above the well bottom it is clamped at the bottom (the
    deficit surfaces via the BHA-length flag downstream).
    """
    remaining = volume_bbl
    d = gas_top
    for sec in sections_sorted:  # shallow -> deep
        if sec.bottom_tvd <= d:
            continue  # entirely above the gas top
        seg_top = max(d, sec.top_tvd)
        seg_bottom = min(sec.bottom_tvd, bottom_tvd)
        seg_len = seg_bottom - seg_top
        if seg_len <= 0.0:
            continue
        cap = sec.capacity_per_tvd_ft
        vol_avail = cap * seg_len
        if remaining <= vol_avail:
            d = seg_top + remaining / cap
            remaining = 0.0
            break
        remaining -= vol_avail
        d = seg_bottom
    gas_bottom = min(d, bottom_tvd)
    return gas_bottom, gas_bottom - gas_top


def _fill_up(gas_bottom: float, volume_bbl: float, sections_sorted, top_limit: float = 0.0):
    """Place ``volume_bbl`` of gas from ``gas_bottom`` UPWARD, section by section.

    Mirror of :func:`_fill_down`. Capacity changes discretely per section, so a
    bubble that exceeds one interval's annular volume SPILLS into the next
    interval up (general -- not just the bottom section). Returns the gas-top
    TVD (clamped at ``top_limit``). Used to seed the march's deepest gas-top
    position (bubble bottom at TD); a bottom-section-only ``V/cap`` estimate
    over-lengthens the bubble and starts the march above binding interfaces.
    """
    remaining = volume_bbl
    d = gas_bottom
    for sec in reversed(sections_sorted):  # deep -> shallow
        if sec.top_tvd >= d:
            continue  # entirely below the gas bottom
        seg_bottom = min(sec.bottom_tvd, d)
        seg_top = max(sec.top_tvd, top_limit)
        seg_len = seg_bottom - seg_top
        if seg_len <= 0.0:
            continue
        cap = sec.capacity_per_tvd_ft
        vol_avail = cap * seg_len
        if remaining <= vol_avail:
            d = seg_bottom - remaining / cap
            remaining = 0.0
            break
        remaining -= vol_avail
        d = seg_top
    return max(d, top_limit)


# ============================================================================
# The engine
# ============================================================================
def migrate(
    sections: Sequence[WellSection],
    pp: ProfileLike,
    fp: ProfileLike,
    *,
    bhp_psi: float,
    influx_bbl_bh: float,
    rho_mud_ppg: float,
    gas_bh_state,
    gas_density_mode: str = "conservative",
    temp_profile: TempProfileLike = None,
    geothermal: TempProfileLike = None,
    n_steps: int = 100,
    mode: str = "thorough",
    ideal_gas: bool = False,
) -> MigrationResult:
    """March a single gas bubble up the annulus under constant BHP.

    Holds the bottom-hole pressure constant at the kill value (Driller's method;
    API RP 59 Sec. 4.10; SPE-202426) and steps the bubble top from the bottom to
    surface. At each step the gas expands by the real-gas law (Boyle + Z + T from
    the bottom-hole state; API RP 59 Sec. 4.8.7.3), its length follows from the
    annular capacity of the section(s) it occupies, and the imposed pressure is
    checked against the pore/fracture window ``PP(d) <= P(d) <= FP(d)`` at every
    EXPOSED open-hole depth (barrier envelope; API RP 59 Sec. 12.5). Cased
    intervals are protected and not checked.

    This is an EXTENSION BEYOND the static NOGEPA single-shoe kick-tolerance
    method (NOGEPA-50) -- a defensible assembly of the cited pieces. The single
    coherent bubble is the safe-side idealisation; OBM gas solubility is out of
    scope. See the module docstring.

    Parameters
    ----------
    sections
        Annular sections (``WellSection``) covering surface (0) to bottom.
    pp, fp
        Pore and fracture pressure profiles [ppg], each a callable tvd->ppg OR a
        (tvd_array, ppg_array) table (numpy-interpolated).
    bhp_psi
        Constant bottom-hole pressure held during the kill [psi].
    influx_bbl_bh
        Influx volume at bottom-hole conditions V_bh [bbl].
    rho_mud_ppg
        Mud density [ppg].
    gas_bh_state
        (P_bh, T_bh_rankine, Z_bh, rho_gas_ppg). ``P_bh`` None -> ``bhp_psi``;
        ``Z_bh`` / ``rho_gas_ppg`` None -> computed by the Hall-Yarborough methane
        backend (``gas_z.py``); ``T_bh_rankine`` required.
    gas_density_mode : {"conservative", "exact"}
        Gas-column density treatment, passed through to ``pressure_at_depth``.
        Default ``"conservative"`` (gas-TOP lightest density, safe-side bound);
        ``"exact"`` integrates the true pressure-dependent local density. See
        ``pressure_at_depth``.
    temp_profile : None | callable | (tvd_array, T_rankine_array)
        Temperature profile T(d) [degR], passed through to ``pressure_at_depth``
        AND used in the Boyle expansion (the gas expands at the LOCAL temperature
        at its top, not the bottom-hole temperature). ``None`` (DEFAULT) is
        ISOTHERMAL at ``T_bh_rankine`` -- reproduces the previous result exactly.
        See ``linear_temp_profile`` for the basic two-point gradient.
    geothermal : None | callable | (tvd_array, T_rankine_array)
        Field geothermal gradient, the DEFAULT temperature when ``temp_profile``
        is not given (explicit ``temp_profile`` > ``geothermal`` > isothermal).
    n_steps
        Number of migration steps (bottom -> surface).

    Returns
    -------
    MigrationResult
        ``.steps`` is the per-step animation trajectory; scalar fields summarise
        the binding (worst) constraint over the whole sweep.
    """
    sections_sorted = sorted(sections, key=lambda s: s.top_tvd)
    bottom_tvd = max(s.bottom_tvd for s in sections_sorted)

    P_bh, T_bh_r, Z_bh, rho_gas_ppg = _resolve_bh_state(gas_bh_state, bhp_psi)
    gas_bh = (P_bh, T_bh_r, Z_bh, rho_gas_ppg)  # bottom-hole anchor for rho_gas(P)
    # Ideal gas (Z=1, isothermal): forces Z=1 everywhere (expansion + column) and
    # isothermal at T_bh. The bubble STILL expands (Boyle), so SICP still rises up
    # the walk -- ideal vs real differ in the expansion Z (Boyle vs Boyle+Z), i.e.
    # the SICP schedule shape, and agree at the initial bubble position. Real gas
    # (default) keeps the Hall-Yarborough Z + any temperature profile.
    z_ideal = (lambda _p, _t: 1.0) if ideal_gas else None
    if ideal_gas:
        temp_profile = None                # isothermal at T_bh
    elif temp_profile is None:
        temp_profile = geothermal          # geothermal is the default when supplied
    temp_fn = _as_temp_callable(temp_profile, T_bh_r)  # still None -> isothermal at T_bh

    pp_fn = _as_ppg_callable(pp)
    fp_fn = _as_ppg_callable(fp)

    if mode not in ("fast", "thorough"):
        raise ValueError(f"mode must be 'fast' or 'thorough', got {mode!r}")
    # Grid resolution. THOROUGH: a fine per-section grid (definitive check). FAST:
    # anchor to the INTERFACES only -- section boundaries (BHA / shoe / hole changes)
    # and PP/FP breakpoints -- where the binding constraint can turn, plus a light
    # per-section fill. The envelope is smooth between interfaces, so this defines it
    # well enough for the API / GUI at a fraction of the cost. A survey that spawns
    # many fine sections is bounded by the per-section counts below.
    per_sec, n_march = (51, n_steps) if mode == "thorough" else (5, 16)

    # Exposed open-hole check depths: per-section grid + interfaces always included.
    interfaces = _profile_breakpoints(pp) + _profile_breakpoints(fp)
    exposed = []
    for sec in sections_sorted:
        if sec.is_open_hole:
            exposed.append(np.linspace(sec.top_tvd, sec.bottom_tvd, per_sec))
            exposed.append(np.array(
                [b for b in interfaces if sec.top_tvd <= b <= sec.bottom_tvd]
            ))
    if not exposed:
        raise ValueError("no open-hole sections: nothing to check")
    exposed_depths = np.unique(np.concatenate([e for e in exposed if e.size]))
    pp_psi = ppg_to_psi(pp_fn(exposed_depths), exposed_depths)
    fp_psi = ppg_to_psi(fp_fn(exposed_depths), exposed_depths)

    open_hole_length = sum(
        s.bottom_tvd - s.top_tvd for s in sections_sorted if s.is_open_hole
    )

    # Deepest gas-top position: bubble BOTTOM pinned at TD, filled UP across
    # sections (spills interval-to-interval -- NOT a bottom-section-only
    # V/cap estimate, which over-lengthens the bubble and starts the march
    # above binding interfaces, silently skipping the worst gas positions on a
    # tight/BHA bottom section => non-conservative max_influx). See _fill_up.
    gas_top_start = _fill_up(bottom_tvd, influx_bbl_bh, sections_sorted)

    if mode == "thorough":
        gas_top_march = np.linspace(gas_top_start, 0.0, n_march)  # n_march == n_steps
    else:  # fast: coarse march, anchored to the interfaces the gas top crosses
        anchors = [b for b in interfaces if 0.0 <= b <= gas_top_start]
        for s in sections_sorted:
            for edge in (s.top_tvd, s.bottom_tvd):
                if 0.0 <= edge <= gas_top_start:
                    anchors.append(edge)
        march = np.concatenate([
            np.linspace(gas_top_start, 0.0, n_march), np.array(anchors, dtype=float)
        ])
        gas_top_march = np.unique(march)[::-1]  # descending: bottom -> surface

    steps: list = []
    bha_flag = False
    global_min = np.inf
    global_binding_tvd = np.nan
    global_binding_step = 0
    all_within = True

    # SIDP -- drill pipe full of mud, influx in the annulus (kick-in-annulus).
    # Position-independent, so constant across the whole walk: BHP minus the mud
    # hydrostatic over the full TVD to the bit.
    sidp_psi = bhp_psi - G_PSI_PER_PPG_FT * rho_mud_ppg * bottom_tvd

    for i, gas_top in enumerate(gas_top_march):
        # Fixed-point on the representative bubble pressure P_rep (at the gas
        # top -- the lowest pressure / largest, safe-side bubble). Boyle uses Z(P)
        # so the bubble expands correctly as it rises.
        P_rep = bhp_psi - G_PSI_PER_PPG_FT * rho_mud_ppg * (bottom_tvd - gas_top)
        P_rep = max(P_rep, 1.0)
        T_local = float(temp_fn(gas_top))  # local T at the representative (gas-top) depth
        gas_bottom, gas_len = gas_top, 0.0
        for _ in range(100):
            Z = 1.0 if ideal_gas else _z_at(P_rep, T_local)
            # V(P,T,Z) = V_bh * (P_bh*Z*T_local) / (P*Z_bh*T_bh); T_local=T_bh -> old.
            V = influx_bbl_bh * (P_bh * Z * T_local) / (P_rep * Z_bh * T_bh_r)
            gas_bottom, gas_len = _fill_down(gas_top, V, sections_sorted, bottom_tvd)
            P_new = float(pressure_at_depth(
                gas_top, gas_top_tvd=gas_top, gas_bottom_tvd=gas_bottom,
                bottom_tvd=bottom_tvd, bhp_psi=bhp_psi,
                rho_mud_ppg=rho_mud_ppg, gas_bh=gas_bh,
                gas_density_mode=gas_density_mode, temp_profile=temp_profile,
                n_sub=20, z_fn=z_ideal,
            ))
            P_new = max(P_new, 1.0)
            if abs(P_new - P_rep) < 1e-4:
                P_rep = P_new
                break
            P_rep = 0.5 * (P_rep + P_new)  # damped fixed point

        if gas_len > open_hole_length:
            bha_flag = True

        # Envelope check at every exposed open-hole depth for this step.
        P = pressure_at_depth(
            exposed_depths, gas_top_tvd=gas_top, gas_bottom_tvd=gas_bottom,
            bottom_tvd=bottom_tvd, bhp_psi=bhp_psi,
            rho_mud_ppg=rho_mud_ppg, gas_bh=gas_bh,
            gas_density_mode=gas_density_mode, temp_profile=temp_profile,
            z_fn=z_ideal,
        )
        fp_margin = fp_psi - P          # >= 0 required (no breakdown)
        pp_margin = P - pp_psi          # >= 0 required (no further influx)
        j = int(np.argmin(fp_margin))
        step_min = float(fp_margin[j])
        step_binding_tvd = float(exposed_depths[j])
        step_p_binding = float(P[j])

        # SICP = the same imposed annulus profile evaluated at surface (depth 0)
        # for this bubble position (no new physics -- the existing profile at the
        # top). Ideal gas -> flat across the walk; real gas -> varies as the
        # bubble expands. SIDP is position-independent (computed once, above).
        sicp = float(pressure_at_depth(
            0.0, gas_top_tvd=gas_top, gas_bottom_tvd=gas_bottom,
            bottom_tvd=bottom_tvd, bhp_psi=bhp_psi,
            rho_mud_ppg=rho_mud_ppg, gas_bh=gas_bh,
            gas_density_mode=gas_density_mode, temp_profile=temp_profile,
            z_fn=z_ideal,
        ))

        if not (np.all(fp_margin >= 0.0) and np.all(pp_margin >= 0.0)):
            all_within = False

        steps.append(MigrationStep(
            gas_top_tvd=float(gas_top),
            gas_bottom_tvd=float(gas_bottom),
            gas_length_ft=float(gas_len),
            min_fp_margin_psi=step_min,
            binding_tvd=step_binding_tvd,
            p_at_binding_psi=step_p_binding,
            sidp_psi=sidp_psi,
            sicp_psi=sicp,
        ))

        if step_min < global_min:
            global_min = step_min
            global_binding_tvd = step_binding_tvd
            global_binding_step = i

    return MigrationResult(
        steps=steps,
        within_envelope=bool(all_within),
        min_fp_margin_psi=float(global_min),
        binding_tvd=float(global_binding_tvd),
        binding_step=int(global_binding_step),
        bha_length_exceeded=bool(bha_flag),
        _ctx=dict(
            bottom_tvd=bottom_tvd, bhp_psi=bhp_psi,
            rho_mud_ppg=rho_mud_ppg, rho_gas_ppg=rho_gas_ppg,
            gas_bh=gas_bh, gas_density_mode=gas_density_mode,
            temp_profile=temp_profile,
            P_bh=P_bh, T_bh_rankine=T_bh_r, Z_bh=Z_bh,
            open_hole_length=open_hole_length,
        ),
    )


# ============================================================================
# Inverse: MAX influx that can be circulated out (the migration kick tolerance)
# ============================================================================
@dataclass
class KickToleranceResult:
    """Migration kick tolerance V* AND where/why it is limited.

    ``max_influx_bbl`` is the largest influx that can be circulated out. The
    binding fields describe the breach that limits it. ``binding_tvd`` is the
    governing depth -- which may be a WEAK FORMATION deeper than the casing shoe,
    NOT the shoe assumed by the static single-shoe check: the migration checks
    every exposed depth at every bubble position, so it reports the true limit.
    """

    max_influx_bbl: float        # V* -- max influx that can be circulated out  [bbl]
    binding_tvd: float           # governing (breach) depth                     [ft]
    binding_step: int            # migration step (animation frame) at the limit
    limited_by: str              # "fracture" | "open_hole_capacity" | "cap"
    #                              "open_hole_capacity" (with open_hole_unconstrained=True):
    #                              the shoe holds to full open-hole displacement, so the
    #                              OPEN HOLE does not constrain the kick tolerance at the
    #                              provided fracture pressure. max_influx_bbl is then the
    #                              full open-hole gas CAPACITY, NOT the kick tolerance --
    #                              the governing limit lies beyond what is assessed here
    #                              (this check stops at the shoe) and is NOT determined.
    min_fp_margin_psi: float     # FP margin at the breach (~0 when fracture-limited)
    open_hole_unconstrained: bool = False   # True when the open hole does not constrain
    #                              the KT at the provided (uncertain) fracture pressure --
    #                              the shoe holds through full open-hole displacement.
    #                              max_influx_bbl is then the full open-hole gas capacity,
    #                              NOT a fracture limit and NOT the kick tolerance. This is
    #                              NOT "unlimited": limits are simply not assessed beyond
    #                              the open hole -- above the shoe (e.g. casing burst as the
    #                              gas reaches surface) is a separate casing-design check,
    #                              and sub-shoe leak-off into permeable formations is not
    #                              modelled. A to-surface casing-burst assessment is a
    #                              documented follow-up (task) -- not applied here.


def max_influx_circulated(
    sections: Sequence[WellSection],
    pp: "ProfileLike",
    fp: "ProfileLike",
    *,
    bhp_psi: float,
    rho_mud_ppg: float,
    gas_bh_state,
    gas_density_mode: str = "conservative",
    temp_profile: TempProfileLike = None,
    geothermal: TempProfileLike = None,
    n_steps: int = 100,
    mode: str = "thorough",
    v_cap_bbl: float = 500.0,
    tol_bbl: float = 0.1,
    max_iter: int = 60,
) -> KickToleranceResult:
    """MAX bottom-hole influx that can be CIRCULATED OUT within the PP-FP envelope
    over the whole migration -- the migration-form kick tolerance, WITH where/why
    it breaches. INVERSE of :func:`migrate` (which checks a GIVEN influx).

    An influx is tolerable when the migration keeps ``within_envelope`` True AND the
    bubble fits the open hole (``bha_length_exceeded`` False). Both tighten monotonically
    with influx, so the tolerable set is ``[0, V*]``; bisect for ``V*``. If ``V*`` is set
    by the bubble outgrowing the open hole while the shoe still holds, the result is the
    full-open-hole-displacement influx flagged ``limited_by="open_hole_capacity"`` /
    ``open_hole_unconstrained=True`` -- NOT "unlimited": the open hole does not constrain
    the KT at the provided (uncertain) fracture pressure. The governing limit lies beyond
    what is assessed here (this check stops at the shoe) and is NOT determined.
    Otherwise report the binding depth/step of the breach
    just above it. Generally <= the static single-shoe max (A-7/A-8): the entire
    circulation path is checked, catching **deeper weak zones and the BHA limit**.

    .. note::
       The gas-top march is seeded from the deepest position (bubble BOTTOM at TD,
       filled UP across sections via :func:`_fill_up`) -- NOT a bottom-section-only
       ``V/cap`` estimate, which over-lengthened the bubble on a tight/BHA bottom
       section, started the march above the binding interfaces and silently
       over-estimated ``V*`` (a NON-conservative result; fixed 2026-07-14). The
       margin is monotone in influx once the worst gas position is actually visited.

       Residual: the uniform (and fast-mode) march can still slightly UNDER-sample a
       NARROW breakpoint of ``P(gas_top)`` -- e.g. where the gas BOTTOM crosses a
       capacity discontinuity and fills the tight section to TD -- so ``V*`` may be
       marginally non-conservative near such a breakpoint. The exact/conservative
       value comes from the analytical solver evaluated at the COMPLETE breakpoint
       set (gas-top- AND gas-bottom-at-boundary + deepest position); see
       the welleng kick-tolerance design notes (not shipped).
    """
    if temp_profile is None:
        temp_profile = geothermal          # geothermal is the default when supplied

    # Physical ceiling: the exposed open-hole annular volume. An influx cannot
    # exceed it as a single bubble below the shoe. If the whole exposed hole can be
    # displaced to gas without breaching the fracture envelope, the OPEN HOLE does not
    # constrain the KT at the provided FP (not "unlimited" -- the limit lies beyond
    # what is assessed here); see the open_hole_unconstrained handling below.
    v_hole = sum(
        s.capacity_per_tvd_ft * (s.bottom_tvd - s.top_tvd)
        for s in sections if s.is_open_hole
    )
    v_ceiling = min(v_hole, v_cap_bbl)               # never search beyond the hole

    def _run(v_bbl: float) -> MigrationResult:
        return migrate(
            sections, pp, fp, bhp_psi=bhp_psi, influx_bbl_bh=v_bbl,
            rho_mud_ppg=rho_mud_ppg, gas_bh_state=gas_bh_state,
            gas_density_mode=gas_density_mode, temp_profile=temp_profile,
            n_steps=n_steps, mode=mode,
        )

    def _tolerable(r: MigrationResult) -> bool:
        return r.within_envelope and not r.bha_length_exceeded

    def _result(vstar, r, limited_by, open_hole_unconstrained=False) -> "KickToleranceResult":
        return KickToleranceResult(
            max_influx_bbl=float(vstar),
            binding_tvd=float(r.binding_tvd),
            binding_step=int(r.binding_step),
            limited_by=limited_by,
            min_fp_margin_psi=float(r.min_fp_margin_psi),
            open_hole_unconstrained=open_hole_unconstrained,
        )

    r_tol = _run(tol_bbl)
    if not _tolerable(r_tol):                       # even a tiny influx breaches
        return _result(0.0, r_tol,
                       "bha_length" if r_tol.bha_length_exceeded else "fracture")

    r_ceil = _run(v_ceiling)
    if _tolerable(r_ceil):                           # whole searchable volume tolerable
        return _result(v_ceiling, r_ceil, "cap")     # (fracture holds AND bubble fits)

    lo, hi = tol_bbl, v_ceiling                      # lo tolerable, hi not
    for _ in range(max_iter):
        if hi - lo <= tol_bbl:
            break
        mid = 0.5 * (lo + hi)
        if _tolerable(_run(mid)):
            lo = mid
        else:
            hi = mid
    breach = _run(hi)                                # first non-tolerable influx
    if breach.bha_length_exceeded and breach.within_envelope:
        # The bubble outgrew the open hole while the SHOE STILL HELD: the gas-top-at-
        # shoe fracture worst case is unreachable (by the time the bubble tail clears
        # TD its top is above the shoe), so the OPEN HOLE does not constrain the KT at
        # the provided fracture pressure. NOT "unlimited" and we do NOT claim what the
        # limit is: this check stops at the shoe. Limits beyond it -- above-shoe (e.g.
        # casing burst to surface) and sub-shoe leak-off into permeable formations --
        # are not assessed, and FP is itself uncertain. ``lo`` = full open-hole gas
        # capacity (NOT the kick tolerance). [JJ, 2026-07-16]
        return _result(lo, breach, "open_hole_capacity", open_hole_unconstrained=True)
    return _result(lo, breach, "fracture")
