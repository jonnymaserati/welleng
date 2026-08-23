"""Tests for welleng.combination.fuse_covariances (overlapping-survey BLUE)."""
import numpy as np
import pytest

from welleng.combination import fuse_covariances


def _spd(seed, scale=1.0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(3, 3))
    return (M @ M.T + 3 * np.eye(3)) * scale


# -- correctness properties -------------------------------------------------

def test_independent_equal_inputs_halve_variance():
    S = _spd(1)[None]                      # (1,3,3)
    r = fuse_covariances(S, S)             # C = 0
    # BLUE of two equal independent estimates: Sigma_c = Sigma/2
    np.testing.assert_allclose(r.cov_fused[0], S[0] / 2, rtol=1e-10)
    np.testing.assert_allclose(r.reduction_factor[0], np.sqrt(2), rtol=1e-10)


def test_fully_shared_does_not_reduce():
    # C = A = B: the two surveys are the same realisation -> no information gained
    S = _spd(2)[None]
    r = fuse_covariances(S, S, cross=S)
    np.testing.assert_allclose(r.cov_fused[0], S[0], rtol=1e-8)
    np.testing.assert_allclose(r.reduction_factor[0], 1.0, rtol=1e-6)


def test_partial_share_between_the_two_limits():
    A = _spd(3)[None]
    B = _spd(4)[None]
    full = fuse_covariances(A, B)                     # C=0
    half = fuse_covariances(A, B, cross=0.5 * np.minimum(A, B))
    # sharing reduces the achievable reduction: partial sits between
    # independent (max reduction) and fully-shared (none)
    assert full.sigma_fused[0] <= half.sigma_fused[0] + 1e-9
    assert half.sigma_fused[0] <= min(full.sigma_a[0], full.sigma_b[0]) + 1e-9


def test_shared_vs_independent_report_differently():
    # opposite conditions must give different results
    A, B = _spd(5)[None], _spd(6)[None]
    indep = fuse_covariances(A, B)
    shared = fuse_covariances(A, B, cross=0.9 * np.minimum(A, B))
    assert not np.allclose(indep.cov_fused, shared.cov_fused)


def test_fused_never_worse_than_either_input():
    rng = np.random.default_rng(7)
    A = np.stack([_spd(i) for i in range(10)])
    B = np.stack([_spd(100 + i, scale=rng.uniform(0.3, 3)) for i in range(10)])
    r = fuse_covariances(A, B)
    assert np.all(r.sigma_fused <= r.sigma_a + 1e-9)
    assert np.all(r.sigma_fused <= r.sigma_b + 1e-9)
    assert np.all(r.reduction_factor >= 1.0 - 1e-9)


# -- MC oracle (the definitive gate) ----------------------------------------

def test_mc_oracle_reproduces_fused_covariance_with_cross():
    """Draw SHARED sources once (applied to both surveys) + independent
    sources separately; fuse per realisation; the empirical covariance of the
    fused estimate must match the analytical fused covariance."""
    rng = np.random.default_rng(2024)
    Cs = _spd(11, scale=0.4)      # shared (common-mode) covariance
    Da = _spd(12, scale=1.0)      # independent, survey A
    Db = _spd(13, scale=0.6)      # independent, survey B
    A = Cs + Da                   # Sigma_A = shared + indep_A
    B = Cs + Db
    C = Cs                        # cross-cov = cov(shared)

    N = 300_000
    Ls, La, Lb = (np.linalg.cholesky(m) for m in (Cs, Da, Db))
    shared = rng.normal(size=(N, 3)) @ Ls.T
    xa = shared + rng.normal(size=(N, 3)) @ La.T
    xb = shared + rng.normal(size=(N, 3)) @ Lb.T

    r = fuse_covariances(A[None], B[None], cross=C[None],
                         pos_a=xa[:1], pos_b=xb[:1])  # analytical cov only
    # fuse every realisation with the (constant) gain
    S = A + B - C - C.T
    gain = (A - C) @ np.linalg.inv(S)
    xc = xa + (gain @ (xb - xa).T).T
    emp = np.cov(xc.T)

    np.testing.assert_allclose(emp, r.cov_fused[0], rtol=0.03, atol=1e-3)
    # and it must beat each input (shared part remains, independent averaged)
    assert np.sqrt(np.linalg.eigvalsh(r.cov_fused[0]).max()) < \
        np.sqrt(np.linalg.eigvalsh(A).max())


def test_position_fusion_shape_and_bounds():
    A, B = _spd(8)[None], _spd(9)[None]
    r = fuse_covariances(A, B, pos_a=np.array([[1., 2., 3.]]),
                         pos_b=np.array([[1.2, 1.8, 3.1]]))
    assert r.pos_fused.shape == (1, 3)
    # fused position lies between the two inputs component-wise
    lo = np.minimum([1., 2., 3.], [1.2, 1.8, 3.1])
    hi = np.maximum([1., 2., 3.], [1.2, 1.8, 3.1])
    assert np.all(r.pos_fused[0] >= lo - 1e-9) and np.all(r.pos_fused[0] <= hi + 1e-9)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fuse_covariances(_spd(1)[None], np.zeros((2, 3, 3)))
