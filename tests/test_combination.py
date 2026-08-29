"""Tests for welleng.combination.fuse_covariances (overlapping-survey BLUE)."""
import numpy as np
import welleng as we
import pytest

from welleng.combination import fuse_covariances, combine_surveys


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


def test_fuse_covariances_rejects_nonfinite_input():
    # a NaN covariance (e.g. from a negative inclination poisoning the error
    # model) must raise a clear error naming the station, not a cryptic
    # eigvalsh "did not converge" from the PSD clip.
    a = np.tile(np.eye(3), (5, 1, 1))
    b = a.copy()
    a[2, 0, 0] = np.nan
    with pytest.raises(ValueError, match=r"non-finite.*index.*\[2\]"):
        fuse_covariances(a, b)


def test_survey_normalises_negative_inclination():
    # a negative inclination is the same direction as |inc| at azi+180; Survey
    # normalises it, so the covariance is finite (not NaN) and the geometry
    # matches the explicitly-normalised survey.
    h = we.survey.SurveyHeader(name="t", azi_reference="grid", latitude=58.0,
                               b_total=50000.0, dip=72.0, declination=1.0)
    md = np.array([0.0, 30.0, 60.0, 90.0])
    inc_neg = np.array([0.0, -5.0, 10.0, 20.0])       # a negative station
    azi = np.array([45.0, 45.0, 45.0, 45.0])
    s = we.survey.Survey(md=md, inc=inc_neg, azi=azi,
                         header=h, error_model="MWD+SRGM", deg=True)
    # equivalent survey with the negative station written as +inc, azi+180
    s_eq = we.survey.Survey(
        md=md, inc=np.array([0.0, 5.0, 10.0, 20.0]),
        azi=np.array([45.0, 225.0, 45.0, 45.0]),
        header=h, error_model="MWD+SRGM", deg=True,
    )
    assert np.isfinite(s.cov_nev).all()               # no NaN poisoning
    assert np.allclose(np.c_[s.n, s.e, s.tvd],
                       np.c_[s_eq.n, s_eq.e, s_eq.tvd], atol=1e-9)


def test_combine_surveys_return_trajectory_roundtrip():
    # combine two surveys of one well and reconstruct the fused trajectory;
    # forward min-curve on (mds, inc, azi) must reproduce the fused positions.
    h = we.survey.SurveyHeader(name="t", azi_reference="grid", latitude=58.0,
                               b_total=50000.0, dip=72.0, declination=1.0)
    md = np.arange(0.0, 2000.0 + 1, 30.0)
    inc = np.clip((md - 300) / 1500 * 60, 1.0, 60.0)
    azi = np.full(md.size, 30.0)
    rng = np.random.default_rng(3)
    A = we.survey.Survey(md=md, inc=inc + rng.normal(0, 0.1, md.size),
                         azi=azi + rng.normal(0, 0.2, md.size),
                         header=h, error_model="MWD+SRGM", deg=True)
    B = we.survey.Survey(md=md, inc=inc + rng.normal(0, 0.1, md.size),
                         azi=azi + rng.normal(0, 0.2, md.size),
                         header=h, error_model="GYRO-NS-CT", deg=True)
    mds = np.arange(300.0, 2000.0 + 1, 30.0)          # skip the md=0 tie station
    r = combine_surveys(A, B, mds, return_trajectory=True)
    assert r.inc is not None and r.azi is not None and r.pos_fused is not None
    rec = we.survey.Survey(md=mds, inc=r.inc, azi=r.azi, header=h, deg=True)
    rec_pos = np.c_[rec.n, rec.e, rec.tvd]
    # compare shape (origin-independent): displacement from the first station.
    # MD is held exact; the single-arc reconstruction carries a small position
    # residual (below the metre-scale EOU) where the BLUE path won't sit on one
    # arc per leg.
    assert np.allclose(rec.md, mds)                    # MD held exact
    assert np.allclose(rec_pos - rec_pos[0],
                       r.pos_fused - r.pos_fused[0], atol=0.05)


def _dia_surveys(seed=5, n=120, xcl_representation="dia"):
    h = we.survey.SurveyHeader(name="t", azi_reference="grid", latitude=58.0,
                               b_total=50000.0, dip=72.0, declination=1.0,
                               xcl_representation=xcl_representation)
    md = np.linspace(0, 2000, n)
    inc = np.clip((md - 300) / 1500 * 60, 1.0, 60.0)
    azi = 30 + np.clip((md - 500) / 1500, 0, 1) * 40
    rng = np.random.default_rng(seed)
    A = we.survey.Survey(md=md, inc=inc + rng.normal(0, 0.15, n),
                         azi=azi + rng.normal(0, 0.3, n),
                         header=h, error_model="MWD+SRGM", deg=True)
    B = we.survey.Survey(md=md, inc=inc + rng.normal(0, 0.15, n),
                         azi=azi + rng.normal(0, 0.3, n),
                         header=h, error_model="GYRO-NS-CT", deg=True)
    return h, A, B, md[md >= 300]


def test_cov_dia_xcl_representation_is_clean_dia_error():
    # XCL is a real course-length ANGLE error (Codling SPE-187249). Under
    # xcl_representation="dia" the error model folds it into the correct inc/azi
    # component -> a physical DIA covariance (a few tenths of a degree). Under the
    # default "nev_direct" the XCL e_DIA lands in the wrong column and inflates
    # raw sig_inc to several degrees (must be excluded there).
    _, A_dia, _, mds = _dia_surveys(xcl_representation="dia")
    _, A_nd, _, _ = _dia_surveys(xcl_representation="nev_direct")
    m = float(mds[len(mds) // 2])
    sig_dia = np.degrees(A_dia.err.cov_dia_at(m)[1, 1] ** 0.5)          # XCL included
    sig_nd_full = np.degrees(A_nd.err.cov_dia_at(m)[1, 1] ** 0.5)
    sig_nd_trim = np.degrees(
        A_nd.err.cov_dia_at(m, exclude=("XCLA", "XCLH"))[1, 1] ** 0.5)
    assert sig_dia < 1.5                  # 'dia' recast: clean, modest angle error
    assert sig_nd_full > 3 * sig_nd_trim  # 'nev_direct': XCL mislabelled, inflated


def test_combine_surveys_dia_space_physical_and_agrees():
    # DIA-space fusion returns the MIA path DIRECTLY: min-curve-valid by
    # construction (0 chord>dmd, no reconstruction) and agreeing with the
    # NEV-position BLUE within the EOU. xcl_representation="dia" -> XCL is INCLUDED
    # (a clean course-length angle error), consistent with cov_nev_at.
    h, A, B, mds = _dia_surveys(xcl_representation="dia")
    dia = combine_surveys(A, B, mds, return_trajectory=True, space="dia")
    nev = combine_surveys(A, B, mds, return_trajectory=True, space="nev")
    assert dia.md is not None and dia.cov_dia is not None
    assert np.allclose(dia.md, mds)                         # MD held exact
    # the returned (md,inc,azi) IS the path -> rebuilding it reproduces pos_fused
    rec = we.survey.Survey(md=dia.md, inc=dia.inc, azi=dia.azi, header=h, deg=True)
    pos = np.c_[rec.n, rec.e, rec.tvd]
    assert np.allclose(pos, dia.pos_fused, atol=1e-6)       # MIA is the output
    # physical: every leg chord <= dmd (min-curve by construction)
    dmd = np.diff(mds)
    chord = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    assert (chord <= dmd + 1e-9).all()
    # agrees with the NEV-position BLUE within the EOU (origin-independent)
    eou = dia.sigma_fused.max()
    d = np.linalg.norm((dia.pos_fused - dia.pos_fused[0])
                       - (nev.pos_fused - nev.pos_fused[0]), axis=1)
    assert d.max() < 0.10 * eou
    # fused DIA angle uncertainty is physical (XCL folded in as a clean angle error)
    assert np.degrees(dia.cov_dia[len(mds) // 2, 1, 1] ** 0.5) < 1.5


def test_combine_surveys_dia_warns_without_dia_representation():
    # nev_direct surveys: XCL has no clean measurement-space e_DIA, so DIA fusion
    # must WARN and exclude it (still returns a physical survey).
    h, A, B, mds = _dia_surveys(xcl_representation="nev_direct")
    with pytest.warns(RuntimeWarning, match="xcl_representation='dia'"):
        dia = combine_surveys(A, B, mds, return_trajectory=True, space="dia")
    dmd = np.diff(mds)
    chord = np.linalg.norm(np.diff(dia.pos_fused, axis=0), axis=1)
    assert (chord <= dmd + 1e-9).all()


def test_combine_surveys_space_invalid_raises():
    _, A, B, mds = _dia_surveys()
    with pytest.raises(ValueError, match="space must be"):
        combine_surveys(A, B, mds, space="xyz")


def test_covariance_block_diagonal_matches_cov_nevs():
    # the covariated 'solid' block: diagonal (3,3) blocks == per-station cov_NEVs
    from welleng.combination import covariance_block
    _, A, _, _ = _dia_surveys(n=40)
    B = covariance_block(A)
    n = len(A.md)
    diag = np.stack([B[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(n)])
    assert np.allclose(diag, A.err.errors.cov_NEVs, atol=1e-9)
    # PSD + symmetric
    assert np.allclose(B, B.T)
    assert np.linalg.eigvalsh(B).min() > -1e-8


def test_fuse_covariated_reduces_and_lifts_downhole():
    # conditioning an MWD run on a gyro observation reduces the whole well
    # (single-station-lifts-all), strongest deep where the systematic accumulates.
    from welleng.combination import fuse_covariated
    h, A, B, _ = _dia_surveys(n=40)
    obs_md = float(A.md[20])
    r = fuse_covariated(A, B, [obs_md])
    assert r["cov_block"].shape == (120, 120)
    assert np.linalg.eigvalsh(r["cov_block"]).min() > -1e-8      # PSD
    assert (r["reduction_factor"] >= 1.0 - 1e-9).all()           # never worse
    assert r["sigma_post"][20] < r["sigma_prior"][20]           # observed station down
    # reduction propagates beyond the observed station (off-diagonal reach)
    assert r["reduction_factor"][39] > 1.0 + 1e-6


def test_fuse_covariated_mc_gate():
    # MC oracle: empirical residual covariance after conditioning == Kalman posterior.
    from welleng.combination import fuse_covariated, _CORR_MODES
    h, A, B, _ = _dia_surveys(n=24)
    n = len(A.md)
    corr = [np.asarray(v.sigma_e_NEV, float).reshape(-1)
            for v in A.err.errors.errors.values() if v.propagation in _CORR_MODES]
    randcov = [sum(np.asarray(v.cov_NEV, float)[i]
                   for v in A.err.errors.errors.values()
                   if v.propagation not in _CORR_MODES) for i in range(n)]
    Lr = [np.linalg.cholesky(c + 1e-15 * np.eye(3)) for c in randcov]
    s = 15
    obs_md = float(A.md[s])
    r = fuse_covariated(A, B, [obs_md])
    P = r["cov_prior"]
    Rg = np.asarray(B.err.cov_nev_at(obs_md), float)
    H = np.zeros((3, 3 * n))
    H[:, 3 * s:3 * s + 3] = np.eye(3)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rg)
    Lg = np.linalg.cholesky(Rg + 1e-12 * np.eye(3))
    rng = np.random.default_rng(0)
    M = 20000
    resid = np.zeros((M, 3 * n))
    for m in range(M):
        x = sum(rng.standard_normal() * sig for sig in corr)
        for i in range(n):
            x[3 * i:3 * i + 3] += Lr[i] @ rng.standard_normal(3)
        z = x[3 * s:3 * s + 3] + Lg @ rng.standard_normal(3)
        resid[m] = x - K @ z
    Pemp = np.cov(resid.T)
    for i in (5, 15, 23):
        bl = slice(3 * i, 3 * i + 3)
        e = np.sqrt(max(np.linalg.eigvalsh(Pemp[bl, bl])[-1], 0))
        k = np.sqrt(max(np.linalg.eigvalsh(r["cov_block"][bl, bl])[-1], 0))
        assert abs(e / k - 1.0) < 0.05           # within MC noise at 20k


def test_covariance_block_at_continuous_surface():
    # the continuous covariated surface: exact interior diagonals (cov_nev_at) +
    # cross-station off-diagonals, queryable at any MD, station-consistent.
    from welleng.combination import covariance_block, covariance_block_at
    _, A, _, _ = _dia_surveys(n=40)
    q = np.array([417.3, 933.8, 1541.0, 1888.2])          # off-station
    Bq = covariance_block_at(A, q)
    # diagonal blocks == the arc-faithful interior covariance
    for i in range(len(q)):
        assert np.allclose(Bq[3*i:3*i+3, 3*i:3*i+3],
                           A.err.cov_nev_at(float(q[i])), atol=1e-12)
    assert np.allclose(Bq, Bq.T)                          # symmetric
    assert np.linalg.eigvalsh(Bq).min() > -1e-6           # PSD
    # at survey stations, off-diagonals match the station covariated block
    sm = A.md[[10, 20, 30]]
    Bs = covariance_block_at(A, sm)
    Bfull = covariance_block(A)
    assert np.allclose(Bs[0:3, 3:6],
                       Bfull[3*10:3*10+3, 3*20:3*20+3], atol=1e-9)
    # correlation carried at interior points
    assert np.linalg.norm(Bq[0:3, 9:12]) > 0.0


def test_combine_surveys_commutative_ab_equals_ba():
    # the combination must be order-independent: fusing A with B equals fusing B
    # with A at any query MD. This holds because both surveys are evaluated at the
    # SAME query MD via the continuous cov_nev_at surface (a grid-resample of one
    # onto the other's stations would NOT be symmetric). Consistency guard.
    _, A, B, _ = _dia_surveys(n=60)
    mds = np.array([717.3, 1123.8, 1541.0])           # off-station interior
    ab = combine_surveys(A, B, mds)
    ba = combine_surveys(B, A, mds)
    assert np.allclose(ab.cov_fused, ba.cov_fused, atol=1e-10)
    # and the continuous covariated surface is symmetric station-to-station:
    # C(m_i, m_j) == C(m_j, m_i)^T
    from welleng.combination import covariance_block_at
    Bq = covariance_block_at(A, mds)
    assert np.allclose(Bq[0:3, 3:6], Bq[3:6, 0:3].T, atol=1e-12)


# -- pillar-2 term swap: replace a toolcode term with a correction covariance --

def test_covariance_block_exclude_removes_named_term():
    from welleng.combination import covariance_block, _CORR_MODES
    _, A, _, _ = _dia_surveys(n=30)
    full = covariance_block(A)
    ex = covariance_block(A, exclude="SAGE")
    sage = A.err.errors.errors["SAGE"]
    assert sage.propagation in _CORR_MODES                       # systematic
    sig = np.asarray(sage.sigma_e_NEV, float).reshape(-1)
    np.testing.assert_allclose(full - ex, np.outer(sig, sig), atol=1e-12)


def test_covariance_block_exclude_unknown_raises():
    from welleng.combination import covariance_block
    _, A, _, _ = _dia_surveys(n=10)
    with pytest.raises(KeyError, match="not in the error model"):
        covariance_block(A, exclude="NOPE")


def test_propagate_dia_axis_reproduces_systematic_source():
    # the injected-block propagator IS the model's own systematic operator:
    # propagating a rank-1 inc cov outer(u) == the toolcode term's outer(sigma)
    from welleng.combination import _propagate_dia_axis_cov
    _, A, _, _ = _dia_surveys(n=30)
    sage = A.err.errors.errors["SAGE"]
    u = np.asarray(sage.e_DIA, float)[:, 1]                       # its inc realisation
    blk = _propagate_dia_axis_cov(A, np.outer(u, u), "inc")
    sig = np.asarray(sage.sigma_e_NEV, float).reshape(-1)
    np.testing.assert_allclose(blk, np.outer(sig, sig), atol=1e-12)


def test_swap_covariated_term_roundtrip_restores_toolcode_term():
    # exclude SAGE then inject its OWN implied inc covariance -> full block back
    from welleng.combination import covariance_block, swap_covariated_term
    _, A, _, _ = _dia_surveys(n=30)
    full = covariance_block(A)
    u = np.asarray(A.err.errors.errors["SAGE"].e_DIA, float)[:, 1]
    swapped = swap_covariated_term(A, "SAGE", np.outer(u, u), axis="inc")
    np.testing.assert_allclose(swapped, full, atol=1e-12)


def test_swap_covariated_term_deterministic_and_zero_limits():
    # None or zero injection == the excluded block (perfect / no-uncertainty corr)
    from welleng.combination import covariance_block, swap_covariated_term
    _, A, _, _ = _dia_surveys(n=20)
    n = len(A.md)
    ex = covariance_block(A, exclude="SAGE")
    np.testing.assert_allclose(swap_covariated_term(A, "SAGE", None), ex, atol=0)
    np.testing.assert_allclose(
        swap_covariated_term(A, "SAGE", np.zeros((n, n)), axis="inc"),
        ex, atol=1e-12)


def test_swap_covariated_term_psd_and_smaller_than_blanket():
    # a realistic small sag residual (< the blanket SAGE) -> valid PSD block with
    # LESS vertical uncertainty than the uncorrected toolcode term
    from welleng.combination import covariance_block, swap_covariated_term
    _, A, _, _ = _dia_surveys(n=30)
    n = len(A.md)
    sage = A.err.errors.errors["SAGE"]
    u = np.asarray(sage.e_DIA, float)[:, 1]
    Csag = 0.09 * np.outer(u, u)                     # 0.3^2: a 30%-of-blanket residual
    full = covariance_block(A)
    sw = swap_covariated_term(A, "SAGE", Csag, axis="inc")
    assert np.allclose(sw, sw.T)
    assert np.linalg.eigvalsh(sw).min() > -1e-8
    v_full = np.trace(full[3 * (n - 1):3 * n, 3 * (n - 1):3 * n])
    v_sw = np.trace(sw[3 * (n - 1):3 * n, 3 * (n - 1):3 * n])
    assert v_sw < v_full                             # correction tightens the EOU


def test_swap_covariated_term_bad_axis_and_shape():
    from welleng.combination import swap_covariated_term
    _, A, _, _ = _dia_surveys(n=15)
    n = len(A.md)
    with pytest.raises(ValueError, match="axis must be"):
        swap_covariated_term(A, "SAGE", np.zeros((n, n)), axis="tvd")
    with pytest.raises(ValueError, match="axis_cov must be"):
        swap_covariated_term(A, "SAGE", np.zeros((n + 1, n + 1)), axis="inc")
