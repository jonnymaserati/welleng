"""Shared-error conditioning of two wells' combined covariance.

The standard ISCWSA pair calculation assumes the two wells' position-error
vectors are entirely independent — combined covariance ``Σ_A + Σ_B``. This
is correct between geographically and temporally separated wells, where the
underlying error sources (magnetic declination, geomagnetic reference
field, sensor biases, …) really do realise independently. It is *not*
correct for two wells drilled from the same platform within a short
time window, where:

- **Global error sources** (magnetic declination model errors,
  BGGM/IFR field model errors, true-vs-grid azimuth correction) are
  realised *identically* in both wells. They share one realisation.
- **Systematic error sources** (sensor biases, tool misalignment, …)
  are independent if the two wells used different tools, but
  *identical* if they shared one MWD service per platform run. The
  honest assumption depends on the operator and the campaign.
- **Random error sources** (per-station survey noise) are always
  independent.

When global and (optionally) systematic terms are shared, they
*cancel* in the position difference:

    cov(X_A − X_B)
        = cov_random_A + cov_random_B
        + cov_systematic_A + cov_systematic_B   (if independent)
        + (cov_global_A − cov_global_B)         (zero if shared)

i.e. the difference covariance can be substantially smaller than the
naive sum, which translates into a substantially lower probability
of collision than the naive ISCWSA calculation reports.

This module provides helpers to construct the *correct* combined
covariance under different assumptions about which error components
are shared between the two wells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ShareMode = Literal[
    "all_independent", "globals_shared", "globals_and_systematic_shared"
]


@dataclass(frozen=True)
class CombinedCovariance:
    """Result of combining two wells' covariances under a sharing assumption.

    Attributes
    ----------
    cov_combined : (n, 3, 3) ndarray
        The covariance of ``X_A − X_B`` per station after
        accounting for the share-mode.
    cov_naive : (n, 3, 3) ndarray
        The naive ``Σ_A + Σ_B`` for comparison.
    sigma_naive : (n,) ndarray
        ``sqrt(max eigenvalue)`` of cov_naive — naive 1σ in
        worst direction, per station.
    sigma_combined : (n,) ndarray
        Same for cov_combined.
    reduction_factor : (n,) ndarray
        ``sigma_naive / sigma_combined`` per station — how much
        the share-mode tightens the combined uncertainty.
    """

    cov_combined: NDArray[np.float64]
    cov_naive: NDArray[np.float64]
    sigma_naive: NDArray[np.float64]
    sigma_combined: NDArray[np.float64]
    reduction_factor: NDArray[np.float64]


def combine_covariances(
    cov_total_a: NDArray[np.float64],
    cov_total_b: NDArray[np.float64],
    *,
    cov_global_a: NDArray[np.float64] | None = None,
    cov_global_b: NDArray[np.float64] | None = None,
    cov_systematic_a: NDArray[np.float64] | None = None,
    cov_systematic_b: NDArray[np.float64] | None = None,
    share_mode: ShareMode = "globals_shared",
) -> CombinedCovariance:
    """Combine two wells' per-station covariances under a share-mode.

    Parameters
    ----------
    cov_total_a, cov_total_b : (n, 3, 3) ndarray
        Total covariance per station for each well, as produced by
        the ISCWSA error model.
    cov_global_a, cov_global_b : (n, 3, 3) ndarray, optional
        Global-error component per well (magnetic declination, BGGM /
        IFR field model errors). Required when ``share_mode`` includes
        global cancellation. The ISCWSA model in welleng exposes this
        as ``Survey.cov_nev_global``.
    cov_systematic_a, cov_systematic_b : (n, 3, 3) ndarray, optional
        Systematic-error component per well (sensor biases, tool
        misalignment). Required when share_mode is
        ``'globals_and_systematic_shared'``. ISCWSA exposes this as
        ``Survey.cov_nev_systematic``.
    share_mode : {'all_independent', 'globals_shared', \
'globals_and_systematic_shared'}
        Which components are realised identically in the two wells:

        - ``'all_independent'``: classical naive
          ``Σ_A + Σ_B`` (no cancellation). Equivalent to passing
          neither global nor systematic arrays.
        - ``'globals_shared'`` (default): global error sources
          realise identically; their contribution cancels in the
          difference covariance. This is the operationally honest
          default for two wells from the same platform.
        - ``'globals_and_systematic_shared'``: both global and
          systematic terms cancel. Stronger assumption — only
          appropriate if the same MWD service / tool was used on
          both wells.

    Returns
    -------
    CombinedCovariance
    """
    cov_a = np.asarray(cov_total_a, dtype=float).reshape(-1, 3, 3)
    cov_b = np.asarray(cov_total_b, dtype=float).reshape(-1, 3, 3)
    if cov_a.shape != cov_b.shape:
        raise ValueError(
            f"cov_total_a {cov_a.shape} and cov_total_b {cov_b.shape} "
            "must have the same shape (per-station alignment)"
        )

    cov_naive = cov_a + cov_b

    if share_mode == "all_independent":
        cov_combined = cov_naive.copy()
    elif share_mode == "globals_shared":
        if cov_global_a is None or cov_global_b is None:
            raise ValueError("globals_shared requires cov_global_a and _b")
        gA = np.asarray(cov_global_a, dtype=float).reshape(-1, 3, 3)
        gB = np.asarray(cov_global_b, dtype=float).reshape(-1, 3, 3)
        # If the global realisation is identical, the contribution to
        # cov(X_A - X_B) is (gA + gB) - 2*shared_global. With identical
        # gA == gB == g_shared, that's (g + g) - 2*g = 0. We don't
        # require gA == gB exactly because sample-time differences
        # along the well allow small variation; we subtract the
        # arithmetic mean of the two as a robust shared estimate.
        shared = 0.5 * (gA + gB)
        cov_combined = cov_naive - 2.0 * shared
    elif share_mode == "globals_and_systematic_shared":
        if any(c is None for c in (cov_global_a, cov_global_b,
                                   cov_systematic_a, cov_systematic_b)):
            raise ValueError(
                "globals_and_systematic_shared requires cov_global_* "
                "and cov_systematic_* for both wells"
            )
        gA = np.asarray(cov_global_a, dtype=float).reshape(-1, 3, 3)
        gB = np.asarray(cov_global_b, dtype=float).reshape(-1, 3, 3)
        sA = np.asarray(cov_systematic_a, dtype=float).reshape(-1, 3, 3)
        sB = np.asarray(cov_systematic_b, dtype=float).reshape(-1, 3, 3)
        shared = 0.5 * (gA + gB) + 0.5 * (sA + sB)
        cov_combined = cov_naive - 2.0 * shared
    else:
        raise ValueError(f"unknown share_mode {share_mode!r}")

    # Symmetrise + clip to PSD as defence against floating-point drift.
    cov_combined = 0.5 * (cov_combined + cov_combined.swapaxes(-1, -2))
    eig = np.linalg.eigvalsh(cov_combined)
    if (eig < -1e-9).any():
        # Strong sign that the share-mode is not consistent with the
        # supplied components (e.g. globals_shared but Σ_global > Σ_total).
        # Project onto PSD by clipping eigenvalues; warn quietly.
        import warnings
        warnings.warn(
            "combined covariance had negative eigenvalues — clipping to "
            "PSD; check share-mode assumption against your error model",
            RuntimeWarning,
        )
        eigvals, eigvecs = np.linalg.eigh(cov_combined)
        eigvals = np.clip(eigvals, 0.0, None)
        cov_combined = np.einsum(
            "...ij,...j,...kj->...ik", eigvecs, eigvals, eigvecs
        )

    sigma_naive = np.sqrt(np.linalg.eigvalsh(cov_naive).max(axis=-1))
    sigma_combined = np.sqrt(np.linalg.eigvalsh(cov_combined).max(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        reduction = np.where(
            sigma_combined > 0, sigma_naive / sigma_combined, np.nan
        )

    return CombinedCovariance(
        cov_combined=cov_combined,
        cov_naive=cov_naive,
        sigma_naive=sigma_naive,
        sigma_combined=sigma_combined,
        reduction_factor=reduction,
    )
