"""MahalanobisClearance quadratic-form fast path (perf/mahalanobis-solve).

The combined-metric quadratic form dp^T S^{-1} dp was computed via a full
eigendecomposition per station. When sigma_pa > 0 (operational default 0.5), S
is strictly positive-definite, so a direct linear solve gives the identical form
~3x faster; the eigh path is retained only for the sigma_pa == 0 degenerate case
(a zero-variance direction must read as +inf = "clear"). These tests pin: the
solve path equals the eigh path for SPD S, the fallback keeps the +inf
semantics, and the end-to-end SF is unchanged vs a forced-eigh reference.
"""
import numpy as np
import pytest

from welleng.survey import Survey, SurveyHeader
from welleng.clearance import MahalanobisClearance


def _survey(n, shift):
    md = np.linspace(0, 30 * (n - 1), n)
    inc = np.clip(np.linspace(0, 90, n), 0, 60)
    azi = (np.linspace(0, 120, n) + shift) % 360
    return Survey(md=md, inc=inc, azi=azi, header=SurveyHeader(),
                  error_model="ISCWSA MWD Rev5.11")


def _clearance(sigma_pa=0.5, n=60):
    return MahalanobisClearance(_survey(n, 0.0), _survey(n, 8.0),
                                sigma_pa=sigma_pa)


def _eigh_quad_form(S, dp):
    """The previous (eigendecomposition) implementation, verbatim, as reference."""
    vals, vecs = np.linalg.eigh(S)
    proj = np.einsum('...ji,...j->...i', vecs, dp)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(vals > 0, proj ** 2 / vals, np.inf)
    terms = np.where(np.isclose(proj, 0.0), 0.0, terms)
    return np.sum(terms, axis=-1)


def test_default_sigma_pa_uses_fast_path():
    assert _clearance().sigma_pa == 0.5  # > 0 -> solve path


def test_solve_path_equals_eigh_for_spd():
    c = _clearance()  # sigma_pa > 0 -> fast solve path
    rng = np.random.default_rng(0)
    A = rng.normal(size=(400, 3, 3))
    S = np.einsum('oij,okj->oik', A, A) + 0.5 * np.eye(3)   # SPD
    dp = rng.normal(size=(400, 3))
    fast = c._quad_form_inv(S, dp)
    ref = _eigh_quad_form(S, dp)
    assert np.allclose(fast, ref, rtol=1e-9, atol=1e-12)


def test_scalar_and_batched_agree():
    c = _clearance()
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3))
    S = A @ A.T + 0.5 * np.eye(3)
    dp = rng.normal(size=3)
    scalar = float(c._quad_form_inv(S, dp))
    batched = c._quad_form_inv(S[None], dp[None])[0]
    assert np.isclose(scalar, batched, rtol=1e-12)


def test_fallback_preserves_inf_clear_direction():
    # sigma_pa == 0 -> eigh fallback. A rank-deficient S (zero variance along z):
    # a dp component in that zero-variance direction is infinitely many sigma away
    # => +inf ("clear"); a dp lying in the range space is finite.
    c = _clearance(sigma_pa=0.0)
    S = np.diag([1.0, 1.0, 0.0])
    assert np.isinf(c._quad_form_inv(S, np.array([0.0, 0.0, 1.0])))   # null dir
    assert np.isclose(c._quad_form_inv(S, np.array([2.0, 0.0, 0.0])), 4.0)  # in-plane


def test_end_to_end_sf_unchanged_vs_eigh():
    n = 120
    new = MahalanobisClearance(_survey(n, 0.0), _survey(n, 8.0))
    sf_new = np.asarray(new.sf, float)

    orig = MahalanobisClearance._quad_form_inv
    try:
        MahalanobisClearance._quad_form_inv = staticmethod(
            lambda S, dp: _eigh_quad_form(S, dp)
        )
        old = MahalanobisClearance(_survey(n, 0.0), _survey(n, 8.0))
        sf_old = np.asarray(old.sf, float)
    finally:
        MahalanobisClearance._quad_form_inv = orig

    assert np.array_equal(np.isfinite(sf_new), np.isfinite(sf_old))  # inf mask
    m = np.isfinite(sf_new)
    assert np.allclose(sf_new[m], sf_old[m], rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
