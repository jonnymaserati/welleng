"""Tests for welleng.flow.ipr — inflow-performance scalar reference oracle.

Each function has an acceptance test. Tests are tagged in their docstring:

* ``[FORM-EXACT]``    — the test hand-computes the published closed-form
  equation independently and asserts the function reproduces it to machine
  precision (validates coefficients + the algebra). Vogel's dimensionless
  curve IS its formula, so a hand value needs no chart digitisation.
* ``[CONSISTENCY]``   — an internal-consistency property that must hold to
  machine precision by construction: a branch continuity/slope match, the
  Jones B=0 → Darcy identity, a Fetkovich proportionality, or an inverse
  round-trip of a test-point back-out.
* ``[BAND]``          — the validity-band ``warnings.warn`` fires (warn, not
  clamp, not raise) outside the published band.
"""
import math
import warnings

import pytest

from welleng.flow import ipr


# =============================================================================
# darcy_linear + pi_from_test
# =============================================================================
def test_darcy_linear_formula():
    """[FORM-EXACT] q = J·(pr − pwf), definitional PI (SPE straight-line)."""
    pr, pwf, j = 3.0e7, 2.0e7, 5.0e-9
    assert ipr.darcy_linear(pr, pwf, j) == pytest.approx(
        j * (pr - pwf), rel=1e-12
    )


def test_darcy_linear_injection_negative():
    """[FORM-EXACT] Negative drawdown (pwf > pr) returns negative rate."""
    assert ipr.darcy_linear(2.0e7, 3.0e7, 5.0e-9) < 0.0


def test_pi_from_test_roundtrip():
    """[CONSISTENCY] pi_from_test inverts darcy_linear: recover J, get q back."""
    pr, pwf, j = 3.0e7, 1.8e7, 4.2e-9
    q = ipr.darcy_linear(pr, pwf, j)
    j_back = ipr.pi_from_test(q, pr, pwf)
    assert j_back == pytest.approx(j, rel=1e-12)
    assert ipr.darcy_linear(pr, pwf, j_back) == pytest.approx(q, rel=1e-12)


# =============================================================================
# vogel + vogel_q_max  (Vogel 1968, SPE-1476)
# =============================================================================
def test_vogel_formula():
    """[FORM-EXACT] q/q_max = 1 − 0.2·x − 0.8·x², x = pwf/pr (SPE-1476)."""
    pr, pwf, q_max = 2.5e7, 1.0e7, 3.0e-3
    x = pwf / pr
    expect = q_max * (1.0 - 0.2 * x - 0.8 * x * x)
    assert ipr.vogel(pr, pwf, q_max) == pytest.approx(expect, rel=1e-12)


def test_vogel_endpoints():
    """[FORM-EXACT] pwf=0 → q_max (AOF); pwf=pr → q=0 (SPE-1476 endpoints)."""
    pr, q_max = 2.5e7, 3.0e-3
    assert ipr.vogel(pr, 0.0, q_max) == pytest.approx(q_max, rel=1e-12)
    assert ipr.vogel(pr, pr, q_max) == pytest.approx(0.0, abs=1e-15)


def test_vogel_q_max_roundtrip():
    """[CONSISTENCY] vogel_q_max inverts vogel: recover q_max, get q back."""
    pr, pwf, q_max = 2.5e7, 1.2e7, 2.4e-3
    q = ipr.vogel(pr, pwf, q_max)
    q_max_back = ipr.vogel_q_max(q, pr, pwf)
    assert q_max_back == pytest.approx(q_max, rel=1e-12)
    assert ipr.vogel(pr, pwf, q_max_back) == pytest.approx(q, rel=1e-12)


def test_vogel_band_warns_outside_domain():
    """[BAND] pwf/pr outside [0, 1] warns (does not raise, does not clamp)."""
    with pytest.warns(UserWarning, match="producing domain"):
        val = ipr.vogel(2.0e7, 3.0e7, 3.0e-3)  # pwf > pr
    assert math.isfinite(val)


# =============================================================================
# standing_composite  (Standing 1971)
# =============================================================================
def test_standing_composite_above_pb_is_linear():
    """[FORM-EXACT] pwf >= Pb branch is the straight line J·(pr − pwf)."""
    pr, pwf, pb, j = 3.0e7, 2.5e7, 2.0e7, 5.0e-9
    assert ipr.standing_composite(pr, pwf, pb, j) == pytest.approx(
        j * (pr - pwf), rel=1e-12
    )


def test_standing_composite_below_pb_formula():
    """[FORM-EXACT] pwf < Pb branch = q_b + (J·Pb/1.8)·Vogel (Standing 1971)."""
    pr, pwf, pb, j = 3.0e7, 1.0e7, 2.0e7, 5.0e-9
    q_b = j * (pr - pb)
    x = pwf / pb
    expect = q_b + (j * pb / 1.8) * (1.0 - 0.2 * x - 0.8 * x * x)
    assert ipr.standing_composite(pr, pwf, pb, j) == pytest.approx(
        expect, rel=1e-12
    )


def test_standing_composite_continuous_at_pb():
    """[CONSISTENCY] Both branches agree at pwf = Pb (value-continuous)."""
    pr, pb, j = 3.0e7, 2.0e7, 5.0e-9
    eps = 1.0e-3  # Pa either side of Pb
    lo = ipr.standing_composite(pr, pb - eps, pb, j)
    hi = ipr.standing_composite(pr, pb + eps, pb, j)
    at = ipr.standing_composite(pr, pb, pb, j)
    assert lo == pytest.approx(at, rel=1e-9)
    assert hi == pytest.approx(at, rel=1e-9)


def test_standing_composite_slope_continuous_at_pb():
    """[CONSISTENCY] dq/dpwf matches across Pb (slope-continuous by design).

    The one-sided derivatives AT Pb must both equal −J. The upper branch is
    linear so its slope is exactly −J; the lower (Vogel) branch is exactly
    quadratic, so a second-order backward stencil at Pb evaluates its boundary
    slope exactly (no truncation) — both sides recover −J to machine precision.
    """
    pr, pb, j = 3.0e7, 2.0e7, 5.0e-9
    h = 1.0e6  # Pa step; stencils are exact for linear/quadratic branches
    f = ipr.standing_composite
    # upper (linear) branch slope at Pb — exact
    slope_above = (f(pr, pb + 2 * h, pb, j) - f(pr, pb + h, pb, j)) / h
    # lower (Vogel) branch slope at Pb — 2nd-order backward stencil, exact for
    # a quadratic; f(pb) is value-continuous so lies on the Vogel polynomial
    slope_below = (
        3.0 * f(pr, pb, pb, j)
        - 4.0 * f(pr, pb - h, pb, j)
        + f(pr, pb - 2 * h, pb, j)
    ) / (2.0 * h)
    assert slope_above == pytest.approx(-j, rel=1e-9)
    assert slope_below == pytest.approx(-j, rel=1e-9)
    assert slope_below == pytest.approx(slope_above, rel=1e-9)


def test_standing_composite_band_warns_pr_below_pb():
    """[BAND] pr < Pb warns (saturated reservoir — vogel() applies)."""
    with pytest.warns(UserWarning, match="pr >= Pb"):
        ipr.standing_composite(1.5e7, 1.0e7, 2.0e7, 5.0e-9)


# =============================================================================
# fetkovich + fetkovich_from_tests  (Fetkovich 1973, SPE-4529)
# =============================================================================
def test_fetkovich_formula():
    """[FORM-EXACT] q = C·(pr² − pwf²)^n (SPE-4529 backpressure form)."""
    pr, pwf, c, n = 3.0e7, 2.0e7, 1.0e-19, 0.85
    expect = c * (pr * pr - pwf * pwf) ** n
    assert ipr.fetkovich(pr, pwf, c, n) == pytest.approx(expect, rel=1e-12)


def test_fetkovich_n1_proportional_to_dp2():
    """[CONSISTENCY] n=1 → q ∝ (pr² − pwf²): ratio equals the dp² ratio."""
    pr, c = 3.0e7, 1.0e-19
    q1 = ipr.fetkovich(pr, 2.0e7, c, 1.0)
    q2 = ipr.fetkovich(pr, 1.0e7, c, 1.0)
    dp1 = pr * pr - 2.0e7 ** 2
    dp2 = pr * pr - 1.0e7 ** 2
    assert q1 / q2 == pytest.approx(dp1 / dp2, rel=1e-12)


def test_fetkovich_from_tests_roundtrip():
    """[CONSISTENCY] (C, n) from two points reproduces both rates."""
    pr, c, n = 3.0e7, 3.3e-20, 0.78
    pwf1, pwf2 = 2.4e7, 1.2e7
    q1 = ipr.fetkovich(pr, pwf1, c, n)
    q2 = ipr.fetkovich(pr, pwf2, c, n)
    c_back, n_back = ipr.fetkovich_from_tests(q1, pwf1, q2, pwf2, pr)
    assert c_back == pytest.approx(c, rel=1e-12)
    assert n_back == pytest.approx(n, rel=1e-12)
    assert ipr.fetkovich(pr, pwf1, c_back, n_back) == pytest.approx(
        q1, rel=1e-12
    )
    assert ipr.fetkovich(pr, pwf2, c_back, n_back) == pytest.approx(
        q2, rel=1e-12
    )


def test_fetkovich_band_warns_on_n():
    """[BAND] n outside [0.5, 1.0] warns (does not raise)."""
    with pytest.warns(UserWarning, match="Fetkovich exponent"):
        ipr.fetkovich(3.0e7, 2.0e7, 1.0e-19, 1.3)


# =============================================================================
# jones  (Jones, Blount & Glaze 1976, SPE-6133)
# =============================================================================
def test_jones_satisfies_quadratic():
    """[FORM-EXACT] The returned q satisfies pr − pwf = A·q + B·q² (SPE-6133)."""
    pr, pwf, a, b = 3.0e7, 2.0e7, 2.0e9, 5.0e11
    q = ipr.jones(pr, pwf, a, b)
    dp = pr - pwf
    assert a * q + b * q * q == pytest.approx(dp, rel=1e-12)


def test_jones_b_zero_equals_darcy():
    """[CONSISTENCY] B=0 degrades to Darcy: jones == darcy_linear(J=1/A).

    The rationalised root q = 2Δp/(A + sqrt(A²)) reduces to Δp/A with no
    epsilon switch. Asserted with approx (rel=1e-12) rather than exact ``==``
    because ``sqrt(A*A)`` is not guaranteed bit-identical to ``A`` (≤ 1 ulp).
    """
    pr, pwf, a = 3.0e7, 2.0e7, 2.0e9
    q_jones = ipr.jones(pr, pwf, a, 0.0)
    q_darcy = ipr.darcy_linear(pr, pwf, 1.0 / a)
    assert q_jones == pytest.approx(q_darcy, rel=1e-12)


def test_jones_stable_root_matches_classic_quadratic():
    """[CONSISTENCY] Rationalised root equals the classic [−A+√(A²+4BΔp)]/2B."""
    pr, pwf, a, b = 3.0e7, 2.0e7, 2.0e9, 5.0e11
    dp = pr - pwf
    classic = (-a + math.sqrt(a * a + 4.0 * b * dp)) / (2.0 * b)
    assert ipr.jones(pr, pwf, a, b) == pytest.approx(classic, rel=1e-12)


def test_no_band_warning_inside_valid_domain():
    """[BAND] No warning is emitted anywhere inside the valid domain."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ipr.darcy_linear(3.0e7, 2.0e7, 5.0e-9)
        ipr.vogel(2.5e7, 1.0e7, 3.0e-3)
        ipr.standing_composite(3.0e7, 1.0e7, 2.0e7, 5.0e-9)
        ipr.fetkovich(3.0e7, 2.0e7, 1.0e-19, 0.85)
        ipr.jones(3.0e7, 2.0e7, 2.0e9, 5.0e11)
