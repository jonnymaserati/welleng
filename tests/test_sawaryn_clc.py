"""Validate the Sawaryn (2021, SPE-204111-PA) analytical CLC solver against the
paper's own worked examples.

Ground truth = Sawaryn's Example 2 (Section "Examples"): same kickoff/target as
Example 1, R1=1250 ft, R2=1750 ft → tangent-length roots
β = (1072.6, 1630.2, 1789.95, 2356.9) ft, principal β=1072.6, with subtended
arc angles α1=13.953°, α2=13.109°.
"""

import numpy as np
import pytest

from welleng.sawaryn_clc import (
    tangent, _scalars, eq15, solve_beta, subtended_angles,
)

# Example 1/2 geometry (SPE-204111-PA)
P1 = np.array([8000.0, 8000.0, 6000.0])
P4 = np.array([9500.0, 8800.0, 6500.0])
T1 = tangent(75.0, 15.0)
T4 = tangent(85.0, 30.0)


def _ex2_scalars():
    return _scalars(P1, T1, P4, T4)


def test_scalars_match_example1():
    # Paper Example 1: ψ²=3.14e6, η1=1728.93, η4=1736.15, η14=252.95, μ=0.95202
    psi2, eta1, eta4, eta14, mu = _ex2_scalars()
    assert psi2 == pytest.approx(3.14e6, rel=1e-4)
    assert eta1 == pytest.approx(1728.93, abs=0.02)
    assert eta4 == pytest.approx(1736.15, abs=0.02)
    assert eta14 == pytest.approx(252.95, abs=0.02)
    assert mu == pytest.approx(0.95202, abs=1e-4)


def test_eq15_roots_reproduce_example2():
    # Sawaryn's four real positive roots, reproduced from Eq. 15.
    # NOTE: the printed factored Eq. 15 reproduces the roots to <1% (principal
    # ~0.6%); exact β comes from forward-verification (separate refinement step).
    psi2, eta1, eta4, eta14, mu = _ex2_scalars()
    roots = solve_beta(psi2, eta1, eta4, eta14, mu, 1250.0, 1750.0)
    expected = [1072.6, 1630.2, 1789.95, 2356.9]
    assert len(roots) == 4
    for got, exp in zip(roots, expected):
        assert got == pytest.approx(exp, rel=0.01)  # within 1%


def test_back_substitution_is_exact_at_principal():
    # Eqs. 18-25: at the TRUE principal β=1072.6, the angles must be exact.
    psi2, eta1, eta4, eta14, mu = _ex2_scalars()
    a1s, a2s = subtended_angles(1072.6, psi2, eta1, eta4, eta14, mu, 1250.0, 1750.0)
    a1_deg = [np.degrees(a) for a in a1s]
    a2_deg = [np.degrees(a) for a in a2s]
    assert any(a == pytest.approx(13.953, abs=0.01) for a in a1_deg)
    assert any(a == pytest.approx(13.109, abs=0.01) for a in a2_deg)
