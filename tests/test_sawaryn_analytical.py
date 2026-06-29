"""Validate the Sawaryn (2021, SPE-204111-PA) analytical CLC solver against the
paper's own worked Example 2.

Same kickoff/target as Example 1, R1=1250 ft, R2=1750 ft → tangent-length roots
β = (1072.6, 1630.2, 1789.95, 2356.9) ft, principal β=1072.6 with subtended arc
angles α1=13.953°, α2=13.109°.

The solver is forward-verified (Eqs. 11-13 + 18-25); it does NOT use the printed
Eq. 15, which carries a transcription/print error (see ``test_eq15_is_trapped``).
"""

import numpy as np
import pytest

from welleng.sawaryn_analytical import (
    tangent, _scalars, forward, subtended_angles, solve_clc_analytical, eq15,
    solve_clc_resultant,
)

P1 = np.array([8000.0, 8000.0, 6000.0])
P4 = np.array([9500.0, 8800.0, 6500.0])
T1 = tangent(75.0, 15.0)
T4 = tangent(85.0, 30.0)
R1, R2 = 1250.0, 1750.0


def test_scalars_match_example1():
    # Paper Example 1: ψ²=3.14e6, η1=1728.93, η4=1736.15, η14=252.95, μ=0.95202
    psi2, eta1, eta4, eta14, mu = _scalars(P1, T1, P4, T4)
    assert psi2 == pytest.approx(3.14e6, rel=1e-4)
    assert eta1 == pytest.approx(1728.93, abs=0.02)
    assert eta4 == pytest.approx(1736.15, abs=0.02)
    assert eta14 == pytest.approx(252.95, abs=0.02)
    assert mu == pytest.approx(0.95202, abs=1e-4)


def test_forward_model_reproduces_example2():
    # Eqs. 11-13 from the known principal solution must give the η invariants.
    _, eta1, eta4, eta14, mu = _scalars(P1, T1, P4, T4)
    f = forward(np.radians(13.953), np.radians(13.109), 1072.6, mu, R1, R2)
    assert f[0] == pytest.approx(eta1, abs=0.05)
    assert f[1] == pytest.approx(eta4, abs=0.05)
    assert f[2] == pytest.approx(eta14, abs=0.05)


def test_back_substitution_is_exact_at_principal():
    # Eqs. 18-25: at the true principal β=1072.6, the angles must be exact.
    psi2, eta1, eta4, eta14, mu = _scalars(P1, T1, P4, T4)
    a1s, a2s = subtended_angles(1072.6, psi2, eta1, eta4, eta14, mu, R1, R2)
    assert any(np.degrees(a) == pytest.approx(13.953, abs=0.01) for a in a1s)
    assert any(np.degrees(a) == pytest.approx(13.109, abs=0.01) for a in a2s)


def test_solve_clc_reproduces_example2_exactly():
    # The full forward-verified solver: all four roots + the principal angles.
    sols = solve_clc_analytical(P1, T1, P4, T4, R1, R2)
    betas = sorted(s['beta'] for s in sols)
    expected = [1072.6, 1630.2, 1789.95, 2356.9]
    assert len(betas) == 4
    for got, exp in zip(betas, expected):
        assert got == pytest.approx(exp, abs=0.2)   # exact to printed precision
    principal = min(sols, key=lambda s: s['beta'])
    assert np.degrees(principal['alpha1']) == pytest.approx(13.953, abs=0.01)
    assert np.degrees(principal['alpha2']) == pytest.approx(13.109, abs=0.01)
    assert principal['residual'] < 1e-3


def test_solve_clc_resultant_is_complete_on_example2():
    # The resultant solver is complete *by construction* (every valid solution's
    # beta is a root of the eliminated polynomial); spurious roots filtered by
    # forward-verification. Recovers all four Example-2 roots exactly.
    pytest.importorskip("flint")  # optional 'analytical' extra (python-flint)
    sols = solve_clc_resultant(P1, T1, P4, T4, R1, R2)
    betas = sorted(s['beta'] for s in sols)
    expected = [1072.6, 1630.2, 1789.95, 2356.9]
    assert len(betas) == 4
    for got, exp in zip(betas, expected):
        assert got == pytest.approx(exp, abs=0.2)
    principal = min(sols, key=lambda s: s['beta'])
    assert np.degrees(principal['alpha1']) == pytest.approx(13.953, abs=0.01)
    assert np.degrees(principal['alpha2']) == pytest.approx(13.109, abs=0.01)
    assert all(s['residual'] < 1e-6 for s in sols)


def test_eq15_is_trapped():
    # Document Sawaryn's trap: the printed Eq. 15 does NOT vanish at the true
    # principal root (the forward-verified solver is exact there instead).
    psi2, eta1, eta4, eta14, mu = _scalars(P1, T1, P4, T4)
    scale = abs(eq15(500.0, psi2, eta1, eta4, eta14, mu, R1, R2))
    assert abs(eq15(1072.6, psi2, eta1, eta4, eta14, mu, R1, R2)) > 1e-3 * scale
