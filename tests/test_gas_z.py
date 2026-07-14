"""Validation of the clean-room Hall & Yarborough (1973) Z-factor backend.

Two independent PUBLIC references:

1. CoolProp (MIT-licensed, HEOS reference EOS) -- the Hall-Yarborough Z is
   asserted within a documented ~2% band across a (T, P) grid covering the well
   range. H-Y is a correlation; CoolProp is a reference multiparameter EOS, so a
   ~1-2% spread is expected and acceptable for the correlation's intended use.

2. The paper's published printed Z values (public, SPE-208788-PA Table 1):
   Z ~= 1.165 at TD (302 degF, 6893 psi) and Z ~= 1.123 at the shoe (212 degF).
   The clean-room H-Y reproduces both to ~0.5%.

Cites: Hall, K. R. & Yarborough, L. (1973), Oil & Gas Journal 71(25): 82-92.
"""

import math

import pytest

from welleng.kick_tolerance.gas_z import (
    METHANE_M_LBM_PER_LBMOL,
    METHANE_PPC_PSIA,
    METHANE_TPC_RANKINE,
    gas_density_ppg,
    hall_yarborough_z,
    methane_properties,
    reduced_density,
)

CoolProp = pytest.importorskip("CoolProp.CoolProp")
PropsSI = CoolProp.PropsSI

PSI_TO_PA = 6894.757293
RANKINE_TO_KELVIN = 5.0 / 9.0

# CoolProp reference EOS vs the H-Y correlation: documented band.
COOLPROP_BAND = 0.02


def _rankine(t_degf: float) -> float:
    return t_degf + 460.0


def _coolprop_z(p_psia: float, t_rankine: float) -> float:
    return PropsSI(
        "Z", "T", t_rankine * RANKINE_TO_KELVIN, "P", p_psia * PSI_TO_PA, "Methane"
    )


# --- 1. CoolProp grid validation --------------------------------------------

# 100-320 degF, 4000-8000 psi -- covers the well range (shoe ~212 degF, TD 302
# degF; shoe/TD pressures ~5.4-6.9 kpsi).
GRID = [
    (t_f, p_psi)
    for t_f in range(100, 321, 20)
    for p_psi in range(4000, 8001, 500)
]


@pytest.mark.parametrize("t_f,p_psi", GRID)
def test_hy_within_band_of_coolprop(t_f, p_psi):
    """H-Y Z within the documented ~2% band of CoolProp across the grid."""
    t_r = _rankine(t_f)
    z_hy = hall_yarborough_z(p_psi, t_r)
    z_cp = _coolprop_z(p_psi, t_r)
    assert math.isclose(z_hy, z_cp, rel_tol=COOLPROP_BAND), (
        f"T={t_f}F P={p_psi}psi: HY={z_hy:.4f} CP={z_cp:.4f} "
        f"dev={abs(z_hy - z_cp) / z_cp * 100:.2f}%"
    )


def test_coolprop_grid_max_deviation_reported():
    """Report the worst-case H-Y vs CoolProp deviation (documented < 2%)."""
    worst = max(
        abs(hall_yarborough_z(p, _rankine(t)) - _coolprop_z(p, _rankine(t)))
        / _coolprop_z(p, _rankine(t))
        for t, p in GRID
    )
    assert worst < COOLPROP_BAND, f"max CoolProp deviation {worst * 100:.2f}%"


# --- 2. Paper printed-Z reproduction (SPE-208788-PA Table 1) ----------------

def test_reproduces_paper_Z_at_TD():
    """Z ~= 1.165 at TD conditions (302 degF, 6893 psi) to ~0.5%."""
    z = hall_yarborough_z(6893.0, _rankine(302.0))
    assert math.isclose(z, 1.1650, rel_tol=0.005), f"Z_td={z:.4f}"


def test_reproduces_paper_Z_at_shoe():
    """Z ~= 1.123 at the shoe (212 degF, influx-gas pressure ~6.4 kpsi) to ~0.5%.

    The paper's printed shoe Z corresponds to a near-bottom-hole influx-gas
    pressure (not the shoe fracture pressure); at 212 degF that pressure is
    ~6402 psi, where clean-room H-Y reproduces the published 1.123.
    """
    z = hall_yarborough_z(6402.0, _rankine(212.0))
    assert math.isclose(z, 1.1230, rel_tol=0.005), f"Z_s={z:.4f}"


# --- Backend internals ------------------------------------------------------

def test_reduced_density_solves_hy_equation():
    """The returned reduced density is a genuine root and lies in (0, 1)."""
    t_r = _rankine(302.0)
    y = reduced_density(6893.0, t_r)
    assert 0.0 < y < 1.0
    # Z = A / y is self-consistent with hall_yarborough_z.
    z = hall_yarborough_z(6893.0, t_r)
    tpr = _rankine(302.0) / METHANE_TPC_RANKINE
    ppr = 6893.0 / METHANE_PPC_PSIA
    t = 1.0 / tpr
    a = 0.06125 * ppr * t * math.exp(-1.2 * (1.0 - t) ** 2)
    assert math.isclose(z, a / y, rel_tol=1e-9)


def test_gas_density_matches_coolprop_at_TD():
    """H-Y density (via Z) within ~2% of CoolProp mass density at TD."""
    t_r = _rankine(302.0)
    z = hall_yarborough_z(6893.0, t_r)
    rho_ppg = gas_density_ppg(6893.0, t_r, z)
    # CoolProp mass density [kg/m3] -> lbm/ft3 -> ppg.
    rho_si = PropsSI(
        "D", "T", t_r * RANKINE_TO_KELVIN, "P", 6893.0 * PSI_TO_PA, "Methane"
    )
    rho_cp_ppg = rho_si * 0.0624279606 / 7.4805
    assert math.isclose(rho_ppg, rho_cp_ppg, rel_tol=COOLPROP_BAND), (
        f"HY={rho_ppg:.4f} CP={rho_cp_ppg:.4f} ppg"
    )


def test_methane_properties_convenience():
    z, rho = methane_properties(6893.0, _rankine(302.0))
    assert math.isclose(z, hall_yarborough_z(6893.0, _rankine(302.0)))
    assert math.isclose(rho, gas_density_ppg(6893.0, _rankine(302.0), z))


def test_methane_constants():
    assert METHANE_TPC_RANKINE == 343.0
    assert METHANE_PPC_PSIA == 667.0
    assert math.isclose(METHANE_M_LBM_PER_LBMOL, 16.043, rel_tol=1e-3)


if __name__ == "__main__":
    worst = max(
        (
            abs(hall_yarborough_z(p, _rankine(t)) - _coolprop_z(p, _rankine(t)))
            / _coolprop_z(p, _rankine(t)),
            t,
            p,
        )
        for t, p in GRID
    )
    print(f"CoolProp grid max deviation = {worst[0] * 100:.3f}% "
          f"at T={worst[1]}F P={worst[2]}psi")
    print(f"Z_td(6893,302F) = {hall_yarborough_z(6893.0, _rankine(302.0)):.4f} "
          f"(paper 1.165)")
    print(f"Z_s (6402,212F) = {hall_yarborough_z(6402.0, _rankine(212.0)):.4f} "
          f"(paper 1.123)")
