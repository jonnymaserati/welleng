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

from dataclasses import dataclass, replace

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
    innovation_mahalanobis, innovation_flag : (n,) ndarray or None
        Set when positions are supplied. ``innovation_mahalanobis`` is the
        Mahalanobis distance of the overlap innovation ``d = x_B - x_A`` under
        the innovation covariance ``S = A + B - C - C^T`` -- i.e.
        ``sqrt(d^T S^-1 d)`` per station. Under the null (the two surveys agree
        within their stated uncertainty) ``d`` is zero-mean Gaussian with
        covariance ``S``, so ``d^T S^-1 d`` is chi-square with 3 dof.
        ``innovation_flag`` is ``True`` where it exceeds ``qc_chi2`` -- a station
        whose two surveys disagree by more than their combined uncertainty
        allows, i.e. a bad station or an under-stated input covariance. Because
        the covariance-EOU conditioning is joint, one such station propagates
        through the off-diagonals and biases the whole run, so it must be caught
        BEFORE conditioning, not after.
    md, inc, azi : (n,) ndarray or None
        The fused best-estimate trajectory as a minimum-curvature listing --
        set only when a trajectory was requested (``combine_surveys(...,
        return_trajectory=True)``). ``(md, inc, azi)`` is drawable by
        conventional trajectory software: the query MDs with the single-arc
        (reflection) reconstruction of the fused NEV positions. MD is held
        exact (it is measured); the reconstruction carries a small position
        residual against the raw BLUE points (typically cm, well below the
        metre-scale position uncertainty), accumulating to a closure residual
        at the segment end. **Every station is CALCULATED, not measured** -- the
        fused path is a derived best estimate, so the whole listing is a
        calculated product (no measured/calculated split per station).
        DLS-faithful: the reconstructed dogleg severity tracks the input surveys.

        .. note::
           This trajectory reconstruction is a **concept pending field
           validation** -- verified on synthetic and Volve data, not yet across
           a range of real wells. The single-arc form is exact when a leg is one
           min-curve arc; its residual grows with the curvature CHANGE across a
           leg (build/turn transitions), so it is trajectory-shape dependent. For
           a typical build (2-3 deg/30 m) it is ~1% of the EOU at standard ~30 m
           spacing, within a couple of per cent to ~120 m, and becomes a large
           fraction only beyond ~300 m spacing (an under-sampled survey anyway);
           the remedy there is to sample the continuous fusion on a finer MD grid
           (calculated inner stations). Where the fused overlap is interior (its
           last station is not the well's total depth), stitching to a
           single-source continuation should absorb the closure residual in the
           final section (truncate/extend the last leg's chord so the segment
           ends at the correct MD).
    """

    cov_fused: NDArray[np.float64]      # (n,3,3) fused BLUE covariance, NEV, m^2
    cov_a: NDArray[np.float64]          # (n,3,3) input survey A covariance, echoed
    cov_b: NDArray[np.float64]          # (n,3,3) input survey B covariance, echoed
    sigma_a: NDArray[np.float64]        # (n,) worst-direction 1sigma of A (m)
    sigma_b: NDArray[np.float64]        # (n,) worst-direction 1sigma of B (m)
    sigma_fused: NDArray[np.float64]    # (n,) worst-direction 1sigma of the fusion (m)
    reduction_factor: NDArray[np.float64]  # (n,) min(sigma_a,sigma_b)/sigma_fused (>=1)
    pos_fused: NDArray[np.float64] | None = None  # (n,3) fused NEV pos, if given
    md: NDArray[np.float64] | None = None   # (k,) fused trajectory measured depth
    inc: NDArray[np.float64] | None = None  # (k,) fused trajectory inclination (deg)
    azi: NDArray[np.float64] | None = None  # (k,) fused trajectory azimuth (deg)
    cov_dia: NDArray[np.float64] | None = None  # (n,3,3) fused DIA cov (space="dia")
    innovation_mahalanobis: NDArray[np.float64] | None = None  # (n,) sqrt(d' S^-1 d)
    innovation_flag: NDArray[np.bool_] | None = None  # (n,) overlap-QC gate tripped


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
    qc_chi2: float = 16.266,
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
        returned in :attr:`FusedSurvey.pos_fused`, and the overlap innovation is
        QC-gated (see ``qc_chi2`` and :attr:`FusedSurvey.innovation_flag`).
    qc_chi2 : float, default 16.266
        Chi-square(3) threshold for the overlap-innovation QC gate (default is
        the 99.9th percentile; use 11.345 for 99%, 7.815 for 95%). A station
        whose innovation ``d^T S^-1 d`` exceeds this is flagged in
        :attr:`FusedSurvey.innovation_flag` and a warning is raised. Only active
        when ``pos_a`` and ``pos_b`` are both supplied.

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

    # Guard non-finite input up front: a NaN/inf covariance (e.g. from an
    # invalid negative/out-of-range inclination that NaNs the error model and
    # poisons the systematic running sum) otherwise surfaces only as a cryptic
    # "Eigenvalues did not converge" from the PSD clip below. Fail clearly,
    # naming the offending stations, so the cause is the survey, not the fuse.
    for name, M in (("cov_a", A), ("cov_b", B), ("cross", C)):
        if not np.isfinite(M).all():
            bad = np.where(~np.isfinite(M.reshape(len(M), -1)).all(axis=1))[0]
            raise ValueError(
                f"{name} has non-finite values at station index(es) "
                f"{bad.tolist()[:20]} -- the input covariance is invalid. A "
                "common cause is a negative or out-of-range inclination in the "
                "source survey, which NaNs the error model and propagates down "
                "the systematic running sum. Fix the survey/error model first."
            )

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
    innov_mahal = None
    innov_flag = None
    if pos_a is not None and pos_b is not None:
        xa = np.asarray(pos_a, dtype=float).reshape(n, 3)
        xb = np.asarray(pos_b, dtype=float).reshape(n, 3)
        d = xb - xa                                  # overlap innovation
        pos_fused = xa + np.einsum("...ij,...j->...i", gain, d)
        # Innovation QC BEFORE the conditioning is trusted: d^T S^-1 d is
        # chi-square(3) under the null. A station above qc_chi2 is inconsistent
        # beyond its combined uncertainty; because the covariance-EOU update is
        # joint, one bad station biases the whole run through the off-diagonals,
        # so it is gated here rather than left to the caller.
        m2 = np.einsum("...i,...ij,...j->...", d, Sinv, d)
        innov_mahal = np.sqrt(np.clip(m2, 0.0, None))
        innov_flag = m2 > qc_chi2
        if innov_flag.any():
            import warnings
            bad = np.where(innov_flag)[0]
            warnings.warn(
                f"overlap innovation exceeds the QC gate (chi-square(3) > "
                f"{qc_chi2:g}) at station index(es) {bad.tolist()[:20]} -- the "
                "two surveys disagree by more than their combined uncertainty "
                "(a bad station, or an under-stated input covariance). Review "
                "before trusting the joint conditioning, which propagates a bad "
                "station through the whole run.",
                RuntimeWarning,
            )

    return FusedSurvey(
        cov_fused=cov_fused, cov_a=A, cov_b=B,
        sigma_a=sigma_a, sigma_b=sigma_b, sigma_fused=sigma_f,
        reduction_factor=red, pos_fused=pos_fused,
        innovation_mahalanobis=innov_mahal, innovation_flag=innov_flag,
    )


def _interp_at(survey, mds):
    """Vectorised (inc_rad, azi_grid_rad, pos_nev) at ``mds`` -- one min-curve
    interpolation over the whole survey, not a per-md ``interpolate_md`` loop
    (~500x faster, bit-identical). ``mds`` must lie within the survey range."""
    r = survey.interpolate_mds(np.asarray(mds, dtype=float))
    idx = np.searchsorted(r.md, mds)
    if not np.allclose(r.md[idx], mds):
        # fall back to nearest (defensive: interpolate_mds inserts the exact mds)
        idx = np.abs(r.md[:, None] - np.asarray(mds)[None, :]).argmin(axis=0)
    pos = np.column_stack([r.n, r.e, r.tvd])[idx]
    return r.inc_rad[idx], r.azi_grid_rad[idx], pos


def _fuse_positions(survey_a, survey_b, mds, cross):
    """Fused NEV covariance + BLUE position at each md (helper)."""
    cov_a = np.stack([np.asarray(survey_a.err.cov_nev_at(float(m))) for m in mds])
    cov_b = np.stack([np.asarray(survey_b.err.cov_nev_at(float(m))) for m in mds])
    pos_a = _interp_at(survey_a, mds)[2]
    pos_b = _interp_at(survey_b, mds)[2]
    return fuse_covariances(cov_a, cov_b, cross=cross, pos_a=pos_a, pos_b=pos_b)


# XCL course-length terms (XCLA/XCLH, Codling SPE-187249): a REAL angular
# (course-length) uncertainty, NOT non-physical -- and there is a proper way to
# express them as DIA errors. With ``SurveyHeader.xcl_representation == "dia"`` the
# error model recasts XCL as an effective inc/azi angle error (``_xcl_dia``,
# Codling Eq. 1 = 0.167*DL*course-length; reproduces the NEV-direct STATION
# covariance to machine precision), so ``cov_DIA`` carries them in the correct
# inc/azi component and they are INCLUDED in the fusion -- consistent with
# ``cov_nev_at`` (which also includes XCL). Only under the default "nev_direct"
# representation is the XCL ``e_DIA`` not a clean measurement error (it lands in
# the wrong column, inflating raw sig_inc to several degrees); there it must be
# excluded. See projects/specs/2026-08-26_dia_space_survey_fusion.md.
_XCL_TERMS = ("XCLA", "XCLH")


def _fuse_dia(survey_a, survey_b, mds, exclude):
    """Fuse (inc, azi) per station in DIA space by BLUE on the angle covariance.

    Returns (inc_f rad, azi_f grid rad, cov_dia (n,3,3) with the fused (inc,azi)
    block; md variance 0 -- measured depth is shared, not fused).
    """
    iA, aA, _ = _interp_at(survey_a, mds)      # inc, azi (grid) rad -- vectorised
    iB, aB, _ = _interp_at(survey_b, mds)
    Ca = np.stack([np.asarray(survey_a.err.cov_dia_at(float(m), exclude=exclude))
                   for m in mds])[:, 1:, 1:]          # (n,2,2) (inc,azi) block
    Cb = np.stack([np.asarray(survey_b.err.cov_dia_at(float(m), exclude=exclude))
                   for m in mds])[:, 1:, 1:]
    xa = np.stack([iA, aA], axis=1)
    xb = np.stack([iB, aB], axis=1)
    # unwrap azimuth relative to A so the fusion is not corrupted by the 0/2pi seam
    daz = (xb[:, 1] - xa[:, 1] + np.pi) % (2 * np.pi) - np.pi
    xb = xb.copy()
    xb[:, 1] = xa[:, 1] + daz
    Cai = np.linalg.inv(Ca)
    Cbi = np.linalg.inv(Cb)
    Cf = np.linalg.inv(Cai + Cbi)                     # fused angle covariance
    xf = (np.einsum("nij,njk,nk->ni", Cf, Cai, xa)
          + np.einsum("nij,njk,nk->ni", Cf, Cbi, xb))
    cov_dia = np.zeros((len(mds), 3, 3))
    cov_dia[:, 1:, 1:] = Cf
    return xf[:, 0], xf[:, 1] % (2 * np.pi), cov_dia


def combine_surveys(
    survey_a,
    survey_b,
    mds,
    *,
    cross=None,
    return_trajectory=False,
    space="nev",
    exclude_dia=None,
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
    return_trajectory : bool, default False
        Also reconstruct the fused best-estimate *trajectory* -- the BLUE
        position at each MD, recovered to (md, inc, azi) and returned in
        :attr:`FusedSurvey.md`, ``.inc``, ``.azi`` (with ``.pos_fused``), a
        min-curve listing of the combined path. The tie-in tangent is taken from
        ``survey_a`` at the first MD; ``mds`` must be sorted ascending. This is
        the single-arc (reflection) reconstruction at the query MDs: MD is held
        exact (measured), DLS-faithful, carrying a small position residual
        against the raw BLUE points (typically cm, below the metre-scale
        position uncertainty). See the note on :attr:`FusedSurvey.md` -- this
        reconstruction is a concept pending field validation.
    space : {"nev", "dia"}, default "nev"
        The space the fusion is performed in.

        - ``"nev"`` -- BLUE on the NEV position covariance (as above). Optimises
          the position estimate; a trajectory (``return_trajectory``) is then
          *reconstructed* from the fused positions, carrying a small residual.
        - ``"dia"`` -- BLUE on the DIA (measured-depth, inclination, azimuth)
          covariance: fuse the **angles** per station (MD is the shared index),
          giving the fused (md, inc, azi) listing *directly*. The path is
          minimum-curvature-valid by construction (no reconstruction, no
          chord>MD residual), and agrees with the NEV fusion within the EOU.
          ``.md/.inc/.azi`` and ``.pos_fused`` are the DIA-fused survey;
          ``.cov_fused`` is the NEV position EOU (the reported ellipsoid);
          ``.cov_dia`` is the fused angle covariance. This is the principled
          space for producing a fused *survey* -- see
          ``projects/specs/2026-08-26_dia_space_survey_fusion.md`` (concept
          pending field validation). The rigorous ``C_nev = V C_dia V^T``
          propagation and the correlated (single-station-lifts-all) form are
          follow-on work.
    exclude_dia : iterable of str, optional (``space="dia"`` only)
        Error-source codes omitted from the DIA fusion weighting. Default
        ``None`` = **auto**: the XCL course-length terms are INCLUDED when both
        surveys use ``SurveyHeader.xcl_representation == "dia"`` (then they are a
        clean angular error -- Codling SPE-187249, folded into the right inc/azi
        component), and EXCLUDED with a warning otherwise (under the default
        "nev_direct" representation the XCL ``e_DIA`` is not a clean measurement
        error). Pass an explicit iterable to override -- e.g. ``()`` to force the
        full budget, or a wider set to also drop MSA / sag-correction /
        non-physical-misalignment terms once identified (an expert judgment).

    Returns
    -------
    FusedSurvey
    """
    mds = np.atleast_1d(np.asarray(mds, dtype=float))
    for name, s in (("survey_a", survey_a), ("survey_b", survey_b)):
        if getattr(s, "err", None) is None:
            raise ValueError(f"{name} has no error model applied (survey.err is None)")
    if space not in ("nev", "dia"):
        raise ValueError(f"space must be 'nev' or 'dia', got {space!r}")

    if space == "dia":
        from .survey import Survey
        if exclude_dia is None:                    # auto XCL handling
            both_dia = all(
                getattr(s.header, "xcl_representation", "nev_direct") == "dia"
                for s in (survey_a, survey_b)
            )
            if both_dia:
                exclude_dia = ()                   # XCL is a clean DIA error -> include
            else:
                import warnings
                warnings.warn(
                    "DIA fusion: surveys not built with "
                    "SurveyHeader(xcl_representation='dia'); the XCL course-length "
                    "terms (XCLA/XCLH) have no clean measurement-space e_DIA under "
                    "'nev_direct' and are EXCLUDED from the fusion. Rebuild the "
                    "surveys with xcl_representation='dia' to include the "
                    "course-length uncertainty properly.",
                    RuntimeWarning,
                )
                exclude_dia = _XCL_TERMS
        inc_f, azi_f, cov_dia = _fuse_dia(survey_a, survey_b, mds, exclude_dia)
        base = _fuse_positions(survey_a, survey_b, mds, cross)   # NEV EOU + sigmas
        if not return_trajectory:
            return replace(base, cov_dia=cov_dia)
        rec = Survey(md=mds, inc=np.degrees(inc_f), azi=np.degrees(azi_f),
                     header=survey_a.header, deg=True)
        pos = np.column_stack([rec.n, rec.e, rec.tvd])
        return replace(base, md=mds, inc=np.degrees(inc_f), azi=np.degrees(azi_f),
                       pos_fused=pos, cov_dia=cov_dia)

    if not return_trajectory:
        cov_a = np.stack([np.asarray(survey_a.err.cov_nev_at(float(m))) for m in mds])
        cov_b = np.stack([np.asarray(survey_b.err.cov_nev_at(float(m))) for m in mds])
        return fuse_covariances(cov_a, cov_b, cross=cross)

    from .utils import survey_from_positions

    fused = _fuse_positions(survey_a, survey_b, mds, cross)
    tie = np.asarray(survey_a.interpolate_md(float(mds[0])).vec_nev, dtype=float)
    _, inc0, azi0 = survey_from_positions(fused.pos_fused, tie, mds=mds)
    return replace(fused, md=mds, inc=inc0, azi=azi0)


_CORR_MODES = ("systematic", "global", "well", "within_pad")


def covariance_block(survey, exclude=None):
    """The full-well "solid" NEV covariance block ``C_well`` (3n, 3n), SI m^2.

    The **covariated ellipse of uncertainty**: instead of ``n`` independent
    per-station ellipsoids, the single block matrix whose diagonal (3, 3) blocks
    are the per-station covariance (equal to ``survey.err.cov_NEVs``) and whose
    off-diagonal blocks are the cross-station correlation from the shared
    (systematic / global / well / within_pad) error sources. This is the object
    that lets a constraint at one station propagate to the whole well.

    The covariated method of Bulychenkov (2026), *Covariance EOU: A Joint-Covariance
    Framework for Wellbore Positioning and Multisensor Data Fusion*
    (v1, doi:10.5281/zenodo.22148756, his Eq. 4) -- a matrix expansion of the
    conventional ISCWSA / Williamson (SPE 67616) per-station calculation
    (``C_well = sum_e J_e C_e J_e^T``). This is welleng's open implementation of
    that method. Each correlated source contributes ``outer(sigma_e_NEV)`` (one
    realisation, fully correlated across the stations it spans); each random source
    contributes its per-station ``cov_NEV`` on the diagonal only.

    Parameters
    ----------
    survey : welleng.survey.Survey
        A survey with an error model applied (``survey.err`` set).
    exclude : str or iterable of str, optional
        Error-source name(s) to omit from the block -- used by the pillar-2
        term swap (:func:`swap_covariated_term`) to drop a toolcode term (e.g.
        the blanket sag term ``"vsag"`` / ``SAGE``) before an external
        correction-uncertainty covariance is injected in its place.

    Returns
    -------
    numpy.ndarray
        (3n, 3n) symmetric PSD block covariance in NEV, SI m^2. The (i, j) block
        ``[3i:3i+3, 3j:3j+3]`` is ``cov(pos_i, pos_j)``.

    Notes
    -----
    Materialises the full (3n, 3n) matrix, so cost grows with the square of
    the number of stations.
    """
    if getattr(survey, "err", None) is None:
        raise ValueError("survey has no error model applied (survey.err is None)")
    errs = survey.err.errors.errors
    excl = frozenset(
        (exclude,) if isinstance(exclude, str) else (exclude or ())
    )
    if excl - set(errs):
        raise KeyError(
            f"exclude names not in the error model: {sorted(excl - set(errs))}"
        )
    n = len(survey.md)
    block = np.zeros((3 * n, 3 * n))
    for name, v in errs.items():
        if name in excl:
            continue
        if v.propagation in _CORR_MODES:
            sig = np.asarray(v.sigma_e_NEV, dtype=float).reshape(-1)
            block += np.outer(sig, sig)             # correlated: cross-station
        else:
            cn = np.asarray(v.cov_NEV, dtype=float)
            for i in range(n):
                block[3 * i:3 * i + 3, 3 * i:3 * i + 3] += cn[i]   # random: diag
    return 0.5 * (block + block.T)                  # symmetrise float drift


_DIA_AXIS = {"depth": 0, "inc": 1, "azi": 2}


def _propagate_dia_axis_cov(survey, axis_cov, axis):
    """NEV block (3n, 3n) of a per-station DIA-*axis* correction covariance.

    A correction-uncertainty covariance (e.g. C_sag over the inclination
    correction) lives in one DIA axis and is cross-station correlated. Propagate
    it to NEV through the error model's OWN exact systematic operator -- the
    balanced-tangential, whole-well-correlated ``e_DIA -> sigma_e_NEV`` map
    (``_sigma_e_NEV_systematic``, the same one every systematic toolcode term
    uses) -- so the injected block is consistent with the toolcode term it
    replaces, to machine precision. ``axis_cov`` is (n, n); it is symmetrised and
    eigen-decomposed, and each non-negative eigen-realisation is pushed through
    the operator as a systematic inc/depth/azi DIA error and accumulated as
    ``outer(sigma)`` (its own correlated realisation).
    """
    m = survey.err
    n = len(survey.md)
    col = _DIA_AXIS[axis]
    A = np.asarray(axis_cov, dtype=float)
    if A.shape != (n, n):
        raise ValueError(
            f"axis_cov must be ({n}, {n}) for this survey, got {A.shape}"
        )
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    tol = max(1e-18, 1e-12 * float(w.max(initial=0.0)))
    block = np.zeros((3 * n, 3 * n))
    for lam, u in zip(w, V.T):
        if lam <= tol:                       # drop null / negative eigen-noise
            continue
        e_dia = np.zeros((n, 3))
        e_dia[:, col] = np.sqrt(lam) * u
        sig = m._sigma_e_NEV_systematic(
            m._e_NEV(e_dia), m._e_NEV_star(e_dia)
        ).reshape(-1)
        block += np.outer(sig, sig)
    return 0.5 * (block + block.T)


def swap_covariated_term(survey, exclude, injected_dia_cov, *, axis="inc"):
    """Covariated NEV block with a toolcode term REPLACED by a correction's
    residual-uncertainty covariance -- the pillar-2 term swap.

    When a survey correction is applied (BHA sag, a depth-stretch model, an MSA
    solve), the blanket toolcode term that *assumed* a generic correction is no
    longer right: it must be replaced by the residual uncertainty of the ACTUAL
    correction for this BHA / hole. This returns::

        covariance_block(survey, exclude=exclude) + propagate(injected_dia_cov)

    i.e. ``C_well`` with the named toolcode term(s) dropped and the external
    correction-uncertainty covariance (from
    :func:`welleng.interpretation.correction_covariance_mc`, in the given DIA
    axis) injected in their place, propagated to NEV through the model's own
    systematic operator so the swap is consistent to machine precision.

    Never keep the toolcode term AND add the correction covariance (double
    count) nor drop both (under-count): correct the survey, ``exclude`` the term,
    inject its residual. C_msa reuses this exact hook (``axis="inc"``/``"azi"``).

    Parameters
    ----------
    survey : welleng.survey.Survey
        Survey with an error model applied.
    exclude : str or iterable of str
        Toolcode source name(s) the correction supersedes (e.g. ``"vsag"``).
    injected_dia_cov : (n, n) array_like or None
        The correction's residual covariance in ``axis`` space, one entry per
        survey station. ``None`` returns the excluded block unchanged (the
        deterministic limit -- a perfect, zero-uncertainty correction).
    axis : {"inc", "depth", "azi"}, default "inc"
        Which DIA axis the correction acts on (sag -> ``"inc"``).

    Returns
    -------
    numpy.ndarray
        (3n, 3n) symmetric NEV block. The pillar-2 realistic-error models this
        consumes are Bulychenkov (2026), §3, v1, doi:10.5281/zenodo.22148756.
    """
    base = covariance_block(survey, exclude=exclude)
    if injected_dia_cov is None:
        return base
    if axis not in _DIA_AXIS:
        raise ValueError(f"axis must be one of {sorted(_DIA_AXIS)}, got {axis!r}")
    swapped = base + _propagate_dia_axis_cov(survey, injected_dia_cov, axis)
    return 0.5 * (swapped + swapped.T)


def covariance_block_at(survey, mds):
    """The **continuous** covariated covariance block at arbitrary measured depths.

    Bulychenkov's integration operator (2026, Eqs. 25-28, v1,
    doi:10.5281/zenodo.22148756) already propagates the
    covariance to points of interest along the well; this couples that off-station
    covariated structure to welleng's arc-faithful interior covariance for accuracy:

    - the diagonal (i, i) blocks are the **analytical arc-faithful interior**
      covariance :meth:`welleng.error.ErrorModel.cov_nev_at` -- exact at any MD,
      on or off a survey station (avoids the ~25% dogleg under-report a full-leg /
      linear interior incurs mid-leg); and
    - the off-diagonal (i, j) blocks are the **cross-station correlation** of the
      covariated EOU (:func:`covariance_block`), evaluated at the query MDs.

    The arc-faithful interior diagonal is welleng's accuracy refinement; pairing it
    with the covariated block's cross-station structure gives a covariated covariance
    **queryable at any MD** -- which makes the continuous combination of two surveys
    on different station grids consistent (it is evaluated at the same query MD for
    both, so the combination is order-independent) and accurate off-station.

    Parameters
    ----------
    survey : welleng.survey.Survey
        Survey with an error model applied.
    mds : array-like
        Query measured depths (within the survey range).

    Returns
    -------
    numpy.ndarray
        (3k, 3k) symmetric block covariance at the k query MDs.

    Notes
    -----
    Diagonal blocks are exact (``cov_nev_at``). Off-diagonal blocks use the
    correlated sources' cumulative ``sigma_e_NEV`` interpolated to the query MDs
    (first-order interior for the cross terms; exact at survey stations).
    """
    if getattr(survey, "err", None) is None:
        raise ValueError("survey has no error model applied (survey.err is None)")
    mds = np.atleast_1d(np.asarray(mds, dtype=float))
    k = len(mds)
    smd = np.asarray(survey.md, dtype=float)
    errs = survey.err.errors.errors
    block = np.zeros((3 * k, 3 * k))
    # off-diagonals from the correlated sources' interior cumulative sigma_e_NEV
    for v in errs.values():
        if v.propagation in _CORR_MODES:
            sig = np.asarray(v.sigma_e_NEV, dtype=float)          # (n, 3)
            si = np.column_stack([np.interp(mds, smd, sig[:, c]) for c in range(3)])
            flat = si.reshape(-1)                                 # (3k,)
            block += np.outer(flat, flat)
    # exact interior diagonal blocks (overwrite the interpolated i==i term)
    for i in range(k):
        block[3 * i:3 * i + 3, 3 * i:3 * i + 3] = np.asarray(
            survey.err.cov_nev_at(float(mds[i])), dtype=float)
    return 0.5 * (block + block.T)


def fuse_covariated(survey_target, survey_obs, obs_mds):
    """Condition a survey's covariated EOU on another survey's positions -- the
    covariated MWD-Gyro fusion (single-station-lifts-the-whole-well).

    The MWD-gyro fusion of Bulychenkov (2026, Eqs. 29-30, v1,
    doi:10.5281/zenodo.22148756). Builds the target's full-well block covariance
    (:func:`covariance_block`) and conditions it, in the maximum-likelihood / GLS
    sense (the standard update, SPE 85111 in block form), on the observing survey's
    position estimates at ``obs_mds``. Because
    the target's shared systematic (declination, reused-tool bias) is one
    realisation across every station it spans, the reduction **propagates through
    all correlated stations** -- not only the observed ones -- so a gyro over a
    short interval tightens the whole MWD run (strongest where the systematic has
    accumulated, i.e. deep). This is the general form that subsumes
    :func:`carry_systematic_forward` (the forward-only slice).

    Parameters
    ----------
    survey_target : welleng.survey.Survey
        The survey being improved (e.g. the MWD run), full well.
    survey_obs : welleng.survey.Survey
        The observing survey (e.g. a gyro), providing position constraints.
    obs_mds : array-like
        Measured depths (within both surveys) at which ``survey_obs`` observes.

    Returns
    -------
    dict
        ``{"cov_block": (3n,3n) posterior block, "cov_prior": (3n,3n) prior,
        "sigma_prior": (n,), "sigma_post": (n,), "reduction_factor": (n,)}`` --
        ``sigma_*`` are worst-direction 1sigma per target station.

    Notes
    -----
    The observation is treated as independent of the target (the ``correlate="N"``
    case -- true for MWD vs gyro, which share no declination error). Where the two
    genuinely share a systematic, a cross-covariance term is needed (not modelled
    here). **Gyro-model caveat:** the gain scales with the gyro's real accuracy
    advantage; the standard OWSG gyro model is deliberately conservative, so the
    lift is small unless a representative (vendor / IFR) gyro model is used.
    """
    obs_mds = np.atleast_1d(np.asarray(obs_mds, dtype=float))
    n = len(survey_target.md)
    P = covariance_block(survey_target)                     # prior (3n,3n)
    # observation selection H (3k, 3n): the target station nearest each obs md
    tmd = np.asarray(survey_target.md, dtype=float)
    idx = np.abs(tmd[:, None] - obs_mds[None, :]).argmin(axis=0)
    k = len(idx)
    H = np.zeros((3 * k, 3 * n))
    for r, i in enumerate(idx):
        H[3 * r:3 * r + 3, 3 * i:3 * i + 3] = np.eye(3)
    # observation covariance R: the obs survey's OWN correlated block over the
    # observed stations -- NOT block-diagonal. The gyro's systematic correlates
    # its own stations, so treating multiple observations as independent
    # over-counts the information (inflates the reduction). Take the sub-block of
    # its covariated EOU at the stations nearest obs_mds.
    omd = np.asarray(survey_obs.md, dtype=float)
    oidx = np.abs(omd[:, None] - obs_mds[None, :]).argmin(axis=0)
    Bobs = covariance_block(survey_obs)
    sel = np.concatenate([[3 * i, 3 * i + 1, 3 * i + 2] for i in oidx])
    R = Bobs[np.ix_(sel, sel)]
    HP = H @ P
    gain = HP.T @ np.linalg.inv(HP @ H.T + R)               # (3n, 3k)
    Ppost = P - gain @ HP
    Ppost = 0.5 * (Ppost + Ppost.T)

    def _sig(M):
        return np.array([
            np.sqrt(max(np.linalg.eigvalsh(
                M[3 * i:3 * i + 3, 3 * i:3 * i + 3])[-1], 0.0))
            for i in range(n)
        ])

    sp, sq = _sig(P), _sig(Ppost)
    with np.errstate(divide="ignore", invalid="ignore"):
        rf = np.where(sq > 0, sp / sq, 1.0)
    return {"cov_block": Ppost, "cov_prior": P,
            "sigma_prior": sp, "sigma_post": sq, "reduction_factor": rf}


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

    cov_nominal: NDArray[np.float64]    # (m,3,3) deep cov BEFORE carry, NEV, m^2
    cov_carried: NDArray[np.float64]    # (m,3,3) deep cov AFTER carry (<= nominal)
    sigma_nominal: NDArray[np.float64]  # (m,) worst-dir 1sigma before carry (m)
    sigma_carried: NDArray[np.float64]  # (m,) worst-dir 1sigma after carry (m)
    reduction_factor: NDArray[np.float64]  # (m,) sigma_nominal/sigma_carried (>=1)


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
