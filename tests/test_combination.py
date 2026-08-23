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


def test_independent_oracle_stacked_gls():
    """Cross-check against a fully INDEPENDENT derivation: stacked generalized
    least squares. Two estimates of x with stacked measurement model
    z = [x_A; x_B] = [I; I] x + noise, noise covariance R = [[A, C],[C^T, B]].
    GLS: Sigma_c = (H^T R^-1 H)^-1, x_c = Sigma_c H^T R^-1 z. No Kalman gain
    anywhere -> catches a wrong gain, which the MC (same gain) cannot."""
    for seed in (20, 21, 22):
        A, B = _spd(seed), _spd(seed + 50, scale=0.7)
        C = 0.3 * np.minimum(A, B)
        R = np.block([[A, C], [C.T, B]])
        H = np.vstack([np.eye(3), np.eye(3)])          # (6,3)
        Rinv = np.linalg.inv(R)
        cov_gls = np.linalg.inv(H.T @ Rinv @ H)        # (3,3)

        r = fuse_covariances(A[None], B[None], cross=C[None])
        np.testing.assert_allclose(r.cov_fused[0], cov_gls, rtol=1e-8, atol=1e-10)

        # position too: fused estimate must equal the GLS estimate
        xa, xb = np.array([1., 2., 3.]), np.array([1.3, 1.7, 3.4])
        z = np.concatenate([xa, xb])
        x_gls = cov_gls @ H.T @ Rinv @ z
        rp = fuse_covariances(A[None], B[None], cross=C[None],
                              pos_a=xa[None], pos_b=xb[None])
        np.testing.assert_allclose(rp.pos_fused[0], x_gls, rtol=1e-8, atol=1e-10)


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


def test_combine_surveys_continuous_via_cov_nev_at():
    """combine_surveys evaluates each survey's analytical interior covariance
    (cov_nev_at) at arbitrary (off-station) MDs, then BLUE-fuses -- the
    continuous combination. Two independent copies of one survey must fuse to
    sigma/sqrt(2), and match a manual cov_nev_at + fuse_covariances."""
    import welleng as we
    from welleng.combination import combine_surveys

    sh = we.survey.SurveyHeader(
        name="t", azi_reference="grid", latitude=58.0, longitude=2.0,
        b_total=50000.0, dip=72.0, declination=1.0)
    md = np.linspace(0, 2000, 41)
    inc = np.linspace(0, 40, 41)
    azi = np.full(41, 30.0)
    s = we.survey.Survey(md=md, inc=inc, azi=azi, header=sh,
                         error_model="MWD+SRGM", deg=True)

    q = 0.5 * (md[10:14] + md[11:15])          # off-station interior MDs
    r = combine_surveys(s, s, q)               # two independent copies

    # independent equal inputs -> sqrt(2) reduction at every interior MD
    np.testing.assert_allclose(r.reduction_factor, np.sqrt(2), rtol=1e-6)
    # matches the manual composition (cov_nev_at -> fuse_covariances)
    covq = np.stack([np.asarray(s.err.cov_nev_at(float(m))) for m in q])
    np.testing.assert_allclose(r.cov_fused, covq / 2, rtol=1e-8)


def test_combine_surveys_requires_error_model():
    import welleng as we
    sh = we.survey.SurveyHeader(name="t", azi_reference="grid")
    s = we.survey.Survey(md=np.array([0., 100.]), inc=np.array([0., 5.]),
                         azi=np.array([0., 10.]), header=sh, deg=True)
    from welleng.combination import combine_surveys
    with pytest.raises(ValueError):
        combine_surveys(s, s, [50.0])


def _two_surveys(model_t="MWD+SRGM", model_r="MWD+SRGM+SAG", n=40):
    import welleng as we
    sh = we.survey.SurveyHeader(
        name="t", azi_reference="grid", latitude=58.0, longitude=2.0,
        b_total=50000.0, dip=72.0, declination=1.0)
    md = np.linspace(0, 3000, n)
    inc = np.linspace(0, 50, n)
    azi = np.linspace(20, 70, n)
    tgt = we.survey.Survey(md=md, inc=inc, azi=azi, header=sh,
                           error_model=model_t, deg=True)
    ref = we.survey.Survey(md=md, inc=inc, azi=azi, header=sh,
                           error_model=model_r, deg=True)
    return tgt, ref, n


def test_forward_carry_reduces_and_never_worse():
    from welleng.combination import carry_systematic_forward
    tgt, ref, n = _two_surveys()
    fc = carry_systematic_forward(tgt, ref, np.arange(2, 22), np.arange(24, n))
    assert np.all(fc.sigma_carried <= fc.sigma_nominal + 1e-9)
    assert np.all(fc.reduction_factor >= 1.0 - 1e-9)
    for C in fc.cov_carried:                       # conditioned cov stays PSD
        assert np.linalg.eigvalsh(C).min() > -1e-9


def test_forward_carry_mc_gate():
    """Independent MC: simulate true target/reference errors, form the overlap
    observation, estimate the deep error with the BLUE (Schur) gain; the
    empirical residual covariance must match cov_carried."""
    from welleng.combination import carry_systematic_forward, _correlated_stacks
    tgt, ref, n = _two_surveys()
    oi, di = np.arange(2, 22), np.array([n - 1])
    fc = carry_systematic_forward(tgt, ref, oi, di, obs_subsample=10)
    Am, Rm, _ = _correlated_stacks(tgt)
    Ag, Rg, _ = _correlated_stacks(ref)
    ois = oi[np.linspace(0, len(oi) - 1, 10).round().astype(int)]

    def stack(A, idx):
        return np.concatenate([A[:, k, :].T for k in idx], axis=0)

    def blk(mats):
        o = np.zeros((3 * len(mats), 3 * len(mats)))
        for i, m in enumerate(mats):
            o[3 * i:3 * i + 3, 3 * i:3 * i + 3] = m
        return o
    Hm, Hg = stack(Am, ois), stack(Ag, ois)
    covz = Hm @ Hm.T + Hg @ Hg.T + blk([Rg[k] for k in ois]) + blk([Rm[k] for k in ois])
    k = di[0]
    Ak = Am[:, k, :].T
    K = (Ak @ Hm.T) @ np.linalg.inv(covz)          # BLUE (Schur) gain

    rng = np.random.default_rng(3)
    N = 60000
    epsm = rng.normal(size=(N, Am.shape[0]))
    epsg = rng.normal(size=(N, Ag.shape[0]))
    Lm = {kk: np.linalg.cholesky(Rm[kk] + 1e-12 * np.eye(3)) for kk in list(ois) + [k]}
    Lg = {kk: np.linalg.cholesky(Rg[kk] + 1e-12 * np.eye(3)) for kk in ois}
    z = epsm @ Hm.T - epsg @ Hg.T
    for i, kk in enumerate(ois):
        z[:, 3 * i:3 * i + 3] += (rng.normal(size=(N, 3)) @ Lm[kk].T
                                  - rng.normal(size=(N, 3)) @ Lg[kk].T)
    x_deep_true = epsm @ Ak.T + rng.normal(size=(N, 3)) @ Lm[k].T
    residual = x_deep_true - z @ K.T               # truth - BLUE estimate
    emp = np.cov(residual.T)
    np.testing.assert_allclose(emp, fc.cov_carried[0], rtol=0.05, atol=1e-3)


def test_forward_carry_persist_subset_reduces_less():
    """Carrying only the persisting class ('global'/declination) reduces no more
    than carrying every correlated source, and is still a genuine reduction."""
    from welleng.combination import carry_systematic_forward
    tgt, ref, n = _two_surveys()
    oi, di = np.arange(2, 22), np.arange(24, n)
    allc = carry_systematic_forward(tgt, ref, oi, di)                # persist=None
    glob = carry_systematic_forward(tgt, ref, oi, di, persist="global")
    assert np.all(glob.sigma_carried >= allc.sigma_carried - 1e-9)
    assert np.all(glob.reduction_factor <= allc.reduction_factor + 1e-9)
    assert np.all(glob.reduction_factor >= 1.0 - 1e-9)


def test_forward_carry_persistence_mc_gate():
    """Bomb-proof the persistence model: when the systematic RE-REALISES between
    overlap and deep (a different tool), only the 'global' (declination) source
    persists. persist='global' must then match the empirical residual — the
    non-global part does NOT reduce."""
    from welleng.combination import carry_systematic_forward, _correlated_stacks
    tgt, ref, n = _two_surveys()
    oi, di = np.arange(2, 22), np.array([n - 1])
    fc = carry_systematic_forward(tgt, ref, oi, di, persist="global",
                                  obs_subsample=10)
    Am, Rm, prop = _correlated_stacks(tgt)
    Ag, Rg, _ = _correlated_stacks(ref)
    ois = oi[np.linspace(0, len(oi) - 1, 10).round().astype(int)]
    g = np.array([p == "global" for p in prop])          # persisting
    s = ~g                                                # re-realising

    def stack(A, idx):
        return np.concatenate([A[:, k, :].T for k in idx], axis=0)

    def blk(mats):
        o = np.zeros((3 * len(mats), 3 * len(mats)))
        for i, m in enumerate(mats):
            o[3 * i:3 * i + 3, 3 * i:3 * i + 3] = m
        return o
    Hg_glob, Hg_sys = stack(Am[g], ois), stack(Am[s], ois)
    Href = stack(Ag, ois)
    covz = (Hg_glob @ Hg_glob.T + Hg_sys @ Hg_sys.T + Href @ Href.T
            + blk([Rg[k] for k in ois]) + blk([Rm[k] for k in ois]))
    k = di[0]
    Akg, Aks = Am[g][:, k, :].T, Am[s][:, k, :].T
    K = (Akg @ Hg_glob.T) @ np.linalg.inv(covz)          # gain via GLOBAL only

    rng = np.random.default_rng(11)
    N = 80000
    eg = rng.normal(size=(N, g.sum()))                   # global: SHARED
    es_o = rng.normal(size=(N, s.sum()))                 # systematic overlap
    es_d = rng.normal(size=(N, s.sum()))                 # systematic deep (RE-REALISED)
    er = rng.normal(size=(N, Ag.shape[0]))               # reference
    Lm = {kk: np.linalg.cholesky(Rm[kk] + 1e-12 * np.eye(3)) for kk in list(ois) + [k]}
    Lg = {kk: np.linalg.cholesky(Rg[kk] + 1e-12 * np.eye(3)) for kk in ois}
    z = eg @ Hg_glob.T + es_o @ Hg_sys.T - er @ Href.T
    for i, kk in enumerate(ois):
        z[:, 3 * i:3 * i + 3] += (rng.normal(size=(N, 3)) @ Lm[kk].T
                                  - rng.normal(size=(N, 3)) @ Lg[kk].T)
    deep_true = eg @ Akg.T + es_d @ Aks.T + rng.normal(size=(N, 3)) @ Lm[k].T
    resid = deep_true - z @ K.T
    np.testing.assert_allclose(np.cov(resid.T), fc.cov_carried[0], rtol=0.06, atol=1e-3)
