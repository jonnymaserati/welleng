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
            "fused covariance had negative eigenvalues -- clipping to PSD; "
            "check the cross-covariance is consistent with the inputs "
            "(C must satisfy [[A, C],[C^T, B]] >= 0)",
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
    # pseudo-inverse, not inverse: when a direction is FULLY shared the
    # innovation covariance is singular there (the two surveys agree exactly);
    # pinv gives gain 0 in that subspace -> that direction does not reduce,
    # which is the correct BLUE limit. Equals inv when S is non-singular.
    Sinv = np.linalg.pinv(S)
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
