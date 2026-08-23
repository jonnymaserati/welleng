"""Combining two overlapping surveys of the SAME well into one best estimate.

This is the *fusion* (uncertainty-reducing) counterpart to
:mod:`welleng.conditioning`, which computes the two-*well* difference
covariance for anti-collision. Here two surveys (e.g. MWD + a gyro run)
cover an overlapping measured-depth interval of ONE well; their independent
error sources let a weighted combination beat *either* survey.

The estimator is the **Best Linear Unbiased Estimator (BLUE)** for two
correlated estimates of the same quantity -- equivalently the weighted
least-squares / inverse-covariance weighting described by Chia et al. (2003),
Ledroz et al. (2016) and Bang (SPE-195621). It is NOT a Kalman filter: there
is no time/sequence, just two estimates of one position. (Kalman is the right
tool for the *forward-carry* extension -- a persistent systematic state
propagated into a later section -- which is deliberately out of scope here.)

Given two per-station covariances ``Sigma_A``, ``Sigma_B`` and the
cross-covariance ``C = cov(x_A, x_B)`` from their SHARED error sources:

    Sigma_c = Sigma_A - (Sigma_A - C)(Sigma_A + Sigma_B - C - C^T)^-1 (Sigma_A - C)^T
    x_c     = x_A     + (Sigma_A - C)(Sigma_A + Sigma_B - C - C^T)^-1 (x_B - x_A)

Key property (the correctness anchor): error components that are **fully
shared** (common-mode) between the two surveys -- captured in ``C`` -- do NOT
reduce. Only the independent part is averaged down. With ``C = 0`` (fully
independent -- e.g. every Volve survey tool is flagged ``correlate="N"``) this
collapses to the classic information-add ``(Sigma_A^-1 + Sigma_B^-1)^-1``.

References
----------
- Chia, C.R. et al. (2003) -- rigorous weighted averaging of overlapping surveys.
- Ledroz, A. et al. (2016) -- combined-IPM construction (constant weights).
- Bang, J. (2019) SPE-195621 -- inclination/azimuth-dependent weights.
- Torkildsen et al. (2004) SPE-90408 -- simultaneous propagation captures the
  correlation between two tools (otherwise the covariance is underestimated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FusedSurvey:
    """Result of BLUE-combining two overlapping surveys, per station.

    Attributes
    ----------
    cov_fused : (n, 3, 3) ndarray
        The fused covariance -- never larger than either input in any
        direction (equal only where the two surveys are fully correlated).
    cov_a, cov_b : (n, 3, 3) ndarray
        The input covariances, echoed for comparison.
    sigma_a, sigma_b, sigma_fused : (n,) ndarray
        ``sqrt(max eigenvalue)`` -- worst-direction 1sigma per station.
    reduction_factor : (n,) ndarray
        ``min(sigma_a, sigma_b) / sigma_fused`` -- how much the fusion
        tightens the better of the two inputs (>= 1; 1 = no gain).
    pos_fused : (n, 3) ndarray or None
        The fused position estimate, if positions were supplied.
    """

    cov_fused: NDArray[np.float64]
    cov_a: NDArray[np.float64]
    cov_b: NDArray[np.float64]
    sigma_a: NDArray[np.float64]
    sigma_b: NDArray[np.float64]
    sigma_fused: NDArray[np.float64]
    reduction_factor: NDArray[np.float64]
    pos_fused: NDArray[np.float64] | None = None


def _psd(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrise and clip to positive-semidefinite (float-drift defence)."""
    cov = 0.5 * (cov + cov.swapaxes(-1, -2))
    w = np.linalg.eigvalsh(cov)
    if (w < -1e-9).any():
        import warnings
        warnings.warn(
            "combined covariance had negative eigenvalues -- clipping to PSD "
            "(float drift / ill-conditioned conditioning at a strongly "
            "observed station)",
            RuntimeWarning,
        )
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 0.0, None)
        cov = np.einsum("...ij,...j,...kj->...ik", vecs, vals, vecs)
    return cov


def fuse_covariances(
    cov_a: NDArray[np.float64],
    cov_b: NDArray[np.float64],
    *,
    cross: NDArray[np.float64] | None = None,
    pos_a: NDArray[np.float64] | None = None,
    pos_b: NDArray[np.float64] | None = None,
) -> FusedSurvey:
    """BLUE-combine two overlapping surveys' per-station covariances.

    Parameters
    ----------
    cov_a, cov_b : (n, 3, 3) ndarray
        Per-station NEV covariance of each survey, aligned to the SAME
        stations (interpolate both to a common measured depth first).
    cross : (n, 3, 3) ndarray, optional
        Cross-covariance ``C = cov(x_A, x_B)`` from the error sources shared
        by both surveys (global terms realised identically; shared systematic
        terms if the same tool/reference). Default ``None`` == 0 == fully
        independent (the ``correlate="N"`` case). Fully-shared components in
        ``C`` are preserved (do not reduce).
    pos_a, pos_b : (n, 3) ndarray, optional
        NEV positions of each survey; if both given, the fused position is
        returned in :attr:`FusedSurvey.pos_fused`.

    Returns
    -------
    FusedSurvey
    """
    A = np.asarray(cov_a, dtype=float).reshape(-1, 3, 3)
    B = np.asarray(cov_b, dtype=float).reshape(-1, 3, 3)
    if A.shape != B.shape:
        raise ValueError(
            f"cov_a {A.shape} and cov_b {B.shape} must match (align to "
            "common stations first)"
        )
    n = A.shape[0]
    if cross is None:
        C = np.zeros_like(A)
    else:
        C = np.asarray(cross, dtype=float).reshape(-1, 3, 3)
        if C.shape != A.shape:
            raise ValueError(f"cross {C.shape} must match cov_a {A.shape}")

    # BLUE with correlated estimates (Bar-Shalom fusion).
    S = A + B - C - C.swapaxes(-1, -2)          # innovation covariance
    K = A - C                                    # gain numerator (Sigma_A - C)
    # inv is ~7x faster than pinv and correct for every VALID input: S is
    # positive-definite unless a direction is FULLY shared (C spans the whole
    # of it), which makes it exactly singular. Fall back to pinv on that edge
    # -> gain 0 in the shared subspace -> that direction does not reduce, the
    # correct BLUE limit.
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)                 # fully-shared singular edge
    gain = K @ Sinv                              # (n,3,3)
    cov_fused = _psd(A - gain @ K.swapaxes(-1, -2))

    def _sig(cov):
        return np.sqrt(np.clip(np.linalg.eigvalsh(cov).max(axis=-1), 0.0, None))

    sigma_a, sigma_b = _sig(A), _sig(B)
    sigma_f = _sig(cov_fused)
    with np.errstate(divide="ignore", invalid="ignore"):
        red = np.where(sigma_f > 0, np.minimum(sigma_a, sigma_b) / sigma_f, np.nan)

    pos_fused = None
    if pos_a is not None and pos_b is not None:
        xa = np.asarray(pos_a, dtype=float).reshape(n, 3)
        xb = np.asarray(pos_b, dtype=float).reshape(n, 3)
        pos_fused = xa + np.einsum("...ij,...j->...i", gain, (xb - xa))

    return FusedSurvey(
        cov_fused=cov_fused, cov_a=A, cov_b=B,
        sigma_a=sigma_a, sigma_b=sigma_b, sigma_fused=sigma_f,
        reduction_factor=red, pos_fused=pos_fused,
    )


def combine_surveys(
    survey_a,
    survey_b,
    mds,
    *,
    cross=None,
) -> FusedSurvey:
    """BLUE-combine two overlapping surveys of the same well at arbitrary MDs.

    Evaluates each survey's covariance at every ``md`` via the analytical
    arc-faithful interior covariance :meth:`welleng.error.ErrorModel.cov_nev_at`
    -- the correct way to bring both surveys onto a common measured depth
    (analytical, not the linear covariance interpolation that under-reports
    near doglegs). Fusion of the resulting covariances is the same BLUE as
    :func:`fuse_covariances`, so the combination is *continuous*: query it at
    any MD in the overlap, on or off a survey station.

    Parameters
    ----------
    survey_a, survey_b : welleng.survey.Survey
        Two surveys of the SAME well over an overlapping MD interval, each with
        an error model applied (so ``survey.err`` is set).
    mds : float or array-like
        Measured depth(s) at which to evaluate and combine. Must lie within
        both surveys' MD range.
    cross : (n, 3, 3) ndarray, optional
        Cross-covariance from shared error sources at each MD. Default None == 0
        (fully independent -- the ``correlate="N"`` case). Fully-shared
        components do not reduce.

    Returns
    -------
    FusedSurvey
    """
    mds = np.atleast_1d(np.asarray(mds, dtype=float))
    for name, s in (("survey_a", survey_a), ("survey_b", survey_b)):
        if getattr(s, "err", None) is None:
            raise ValueError(f"{name} has no error model applied (survey.err is None)")
    cov_a = np.stack([np.asarray(survey_a.err.cov_nev_at(float(m))) for m in mds])
    cov_b = np.stack([np.asarray(survey_b.err.cov_nev_at(float(m))) for m in mds])
    return fuse_covariances(cov_a, cov_b, cross=cross)


@dataclass(frozen=True)
class ForwardCarry:
    """Result of carrying a reference-calibrated systematic into deep stations.

    Attributes
    ----------
    cov_nominal, cov_carried : (m, 3, 3) ndarray
        Per-deep-station covariance before / after carrying the overlap
        calibration. ``cov_carried`` <= ``cov_nominal`` (systematic reduced;
        random unchanged).
    sigma_nominal, sigma_carried : (m,) ndarray
        Worst-direction 1sigma per deep station.
    reduction_factor : (m,) ndarray
        ``sigma_nominal / sigma_carried`` (>= 1).
    """

    cov_nominal: NDArray[np.float64]
    cov_carried: NDArray[np.float64]
    sigma_nominal: NDArray[np.float64]
    sigma_carried: NDArray[np.float64]
    reduction_factor: NDArray[np.float64]


def _correlated_stacks(survey):
    """(A, cov_random, prop) for a survey: A (S, n, 3) = per correlated-source
    sigma-scaled NEV running sum; cov_random (n, 3, 3); prop (S,) = each
    source's propagation mode ('global'|'systematic'|'well'). Position error at
    station k is ``A[:, k, :].T @ eps + N(0, cov_random[k])``, eps ~ N(0, I)."""
    if getattr(survey, "err", None) is None:
        raise ValueError("survey has no error model applied (survey.err is None)")
    s = survey.err._interior_stacks()
    keep = ~s["random"]
    return s["sigma_e_NEV"][keep], s["cov_random"], s["propagation"][keep]


def carry_systematic_forward(
    target,
    reference,
    overlap_idx,
    deep_idx,
    *,
    persist=None,
    obs_subsample=12,
):
    """Carry a reference-calibrated systematic into the target's deep stations.

    An independent ``reference`` survey (e.g. a gyro) over an overlap constrains
    the ``target`` survey's (e.g. MWD) SYSTEMATIC error realisation; because that
    realisation persists down-leg, the constraint reduces the target covariance
    at DEEP stations the reference never reached. This is aided-inertial-
    navigation / Kalman conditioning (adopt, don't reinvent; cf. ISCWSA-49 2019
    gyro+MWD statistical estimation), in the joint-covariance (Schur) form:

        overlap obs z = target - reference; the cross-covariance from the
        target's SHARED systematic sources is cov(X_deep, z) = A_deep Hm^T;
        conditioned deep cov = cov(X_deep) - cov(X_deep, z) cov(z)^-1 cov(z,
        X_deep), with cov(z) = Hm Hm^T + reference cov + target random.

    (Woodbury-equivalent to the source-space information form
    P=(I+H^T R^-1 H)^-1, but inverts the well-conditioned cov(z) rather than the
    ill-conditioned R.) Observability is geometry-dependent: the overlap must
    excite the systematic terms that matter at depth; validated by MC (tests).

    Parameters
    ----------
    target, reference : Survey
        Same station set, each with an error model. ``reference`` is the
        independent, more-accurate survey over the overlap.
    overlap_idx : array-like of int
        Station indices the reference covers (the calibration interval).
    deep_idx : array-like of int
        Target-only station indices to carry the calibration to.
    persist : None | str | iterable of str, optional
        Which propagation classes persist between the overlap and the deep
        section (and therefore carry). ``None`` (default) carries every
        correlated source. Physically: ``'global'`` (geomagnetic reference /
        declination) persists across BHA AND tool changes -- the assumption-free
        choice; ``'systematic'`` (sensor bias/scale/misalignment) persists only
        if the SAME MWD tool is reused; a source that re-realises (new tool for
        that class) is independent between deep and overlap and does not carry.
        e.g. ``persist='global'`` for the unconditional declination carry;
        ``persist=('global', 'systematic')`` when the tool is reused.
    obs_subsample : int
        Cap on the number of overlap observation stations used (keeps the
        conditioning block modest); evenly spaced. Default 12.

    Returns
    -------
    ForwardCarry
    """
    overlap_idx = np.asarray(overlap_idx, dtype=int)
    deep_idx = np.asarray(deep_idx, dtype=int)
    Am, Rm, prop_m = _correlated_stacks(target)
    Ag, Rg, _ = _correlated_stacks(reference)

    oi = overlap_idx
    if obs_subsample and len(oi) > obs_subsample:
        oi = oi[np.linspace(0, len(oi) - 1, obs_subsample).round().astype(int)]

    def _stack(A, idx):                       # (3*len(idx), S)
        return np.concatenate([A[:, k, :].T for k in idx], axis=0)

    def _blkdiag(mats):
        out = np.zeros((3 * len(mats), 3 * len(mats)))
        for i, m in enumerate(mats):
            out[3 * i:3 * i + 3, 3 * i:3 * i + 3] = m
        return out

    Hm = _stack(Am, oi)                        # (3no, S_target)
    Hg = _stack(Ag, oi)                        # (3no, S_reference)
    # observation noise: reference full covariance over the overlap (its
    # systematic correlates across stations) + the target's random.
    R = (Hg @ Hg.T
         + _blkdiag([Rg[k] for k in oi])
         + _blkdiag([Rm[k] for k in oi]))
    # Schur-complement (joint-covariance) conditioning. Equivalent by Woodbury
    # to the source-space information form P=(I+H^T R^-1 H)^-1, but inverts the
    # well-conditioned full obs covariance covz = Hm Hm^T + R instead of the
    # ill-conditioned R (the reference systematic makes R near-rank-deficient) --
    # the source-space form was ~6% off here from that ill-conditioning.
    covz = Hm @ Hm.T + R                        # full covariance of the overlap obs
    # pinv-fallback: a zero-covariance overlap station (tie at md=0) is singular
    # and carries no information; pinv handles it.
    try:
        covz_inv = np.linalg.inv(covz)
    except np.linalg.LinAlgError:
        covz_inv = np.linalg.pinv(covz)

    Ad = Am[:, deep_idx, :]                     # (S, m, 3) ALL correlated
    cov_nom = np.einsum("ski,skj->kij", Ad, Ad) + Rm[deep_idx]
    # Only PERSISTING sources are correlated between deep and overlap, so only
    # they enter the cross-covariance (and thus reduce the deep cov). 'global'
    # (geomag/declination) persists across BHA AND tool; 'systematic' (sensor)
    # persists only if the same tool is reused; sources that re-realise (a new
    # tool/BHA) have independent deep/overlap draws -> zero cross -> no carry.
    if persist is None:
        pmask = np.ones(Am.shape[0], dtype=bool)
    else:
        want = {persist} if isinstance(persist, str) else set(persist)
        pmask = np.array([p in want for p in prop_m], dtype=bool)
    Adp = Am[pmask][:, deep_idx, :]             # persisting sources, deep
    Hmp = _stack(Am[pmask], oi)                 # persisting sources, overlap
    # cross[k] = cov(X_deep_k, z) via persisting sources = A_deep^p (Hm^p)^T ;
    # conditioned = nom - cross covz^-1 cross^T
    cross = np.einsum("smi,ps->mip", Adp, Hmp)  # (m, 3, 3no)
    reduction = np.einsum("mip,pq,mjq->mij", cross, covz_inv, cross)
    cov_carried = _psd(cov_nom - reduction)    # symmetrise + clip float drift

    def _sig(C):
        return np.sqrt(np.clip(np.linalg.eigvalsh(C).max(axis=-1), 0.0, None))

    sn, sc = _sig(cov_nom), _sig(cov_carried)
    with np.errstate(divide="ignore", invalid="ignore"):
        red = np.where(sc > 0, sn / sc, np.nan)
    return ForwardCarry(
        cov_nominal=cov_nom, cov_carried=cov_carried,
        sigma_nominal=sn, sigma_carried=sc, reduction_factor=red,
    )
