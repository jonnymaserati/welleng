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

A shared source is one realisation ``g_k`` acting on both wells through each
well's own weights: its position error is ``v_Ak g_k`` in A and ``v_Bk g_k`` in
B, where ``v`` is the source's signed position-error vector at one standard
deviation (Williamson, 2000, SPE-67616-PA, Eq. A-24, with the cross-well
correlation one for the shared sources and zero for the rest). So

    cov(X_A − X_B) = Σ_A + Σ_B − Σ_k (v_Ak v_Bkᵀ + v_Bk v_Akᵀ)

summed over the shared sources. The shared contribution is
``(v_Ak − v_Bk)(v_Ak − v_Bk)ᵀ``: it cancels only where the two wells weight the
source alike (same attitude and departure). Where they differ it can exceed the
independent sum: two wells departing in opposite azimuths see one declination
error rotate them apart, which doubles their relative error rather than
cancelling it. Covariances alone cannot give the cross term, since a covariance
loses each source's sign, so sharing takes the signed per-source vectors
(:func:`source_vectors`).

This module provides helpers to construct the combined covariance under
different assumptions about which error sources are shared between the two
wells.
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
        ``sigma_naive / sigma_combined`` per station: above 1 where sharing
        tightens the combined uncertainty, below 1 where it widens it.
    """

    cov_combined: NDArray[np.float64]
    cov_naive: NDArray[np.float64]
    sigma_naive: NDArray[np.float64]
    sigma_combined: NDArray[np.float64]
    reduction_factor: NDArray[np.float64]


def source_vectors(survey_a, survey_b, propagation: str):
    """Signed per-source position-error vectors of the sources two wells share.

    Parameters
    ----------
    survey_a, survey_b : welleng.survey.Survey
        Error-modelled surveys (``error_model`` set).
    propagation : {'global', 'systematic'}
        Which sources to return.

    Returns
    -------
    codes : tuple of str
        The error-source codes of that propagation present in both wells, in
        one order.
    vec_a, vec_b : (n_a, k, 3) and (n_b, k, 3) ndarray
        Each source's position-error vector at one standard deviation, in NEV,
        in metres, at every station of each well; ``k`` = ``len(codes)``. A
        source's covariance in a well is ``v vᵀ``.

    Raises
    ------
    ValueError
        If either survey carries no error model.
    """
    vecs = []
    for name, s in (("survey_a", survey_a), ("survey_b", survey_b)):
        err = getattr(s, "err", None)
        if err is None:
            raise ValueError(
                f"{name} has no per-source error vectors: it carries no error "
                "model (a composed survey keeps covariance buckets only), so a "
                "shared source's signed vector in it is not known")
        vecs.append({k: np.asarray(e.sigma_e_NEV, dtype=float)
                     for k, e in err.errors.errors.items()
                     if e.propagation == propagation})
    codes = tuple(k for k in vecs[0] if k in vecs[1])
    out = []
    for v, s in zip(vecs, (survey_a, survey_b)):
        n = len(s.md)
        out.append(np.stack([v[k] for k in codes], axis=1) if codes
                   else np.zeros((n, 0, 3)))
    return codes, out[0], out[1]


def _cross(vec_a, vec_b, n: int, name: str) -> NDArray[np.float64]:
    va = np.asarray(vec_a, dtype=float)
    vb = np.asarray(vec_b, dtype=float)
    va = va.reshape(-1, va.shape[-2], 3) if va.ndim >= 2 else va
    vb = vb.reshape(-1, vb.shape[-2], 3) if vb.ndim >= 2 else vb
    if va.ndim != 3 or va.shape != vb.shape or va.shape[0] != n:
        raise ValueError(
            f"{name}_a and {name}_b must both be (n, k, 3), one row per station "
            f"of the covariances and the same sources in the same order; got "
            f"{va.shape} and {vb.shape} for n = {n}")
    return np.einsum("nki,nkj->nij", va, vb)


def combine_covariances(
    cov_total_a: NDArray[np.float64],
    cov_total_b: NDArray[np.float64],
    *,
    vec_global_a: NDArray[np.float64] | None = None,
    vec_global_b: NDArray[np.float64] | None = None,
    vec_systematic_a: NDArray[np.float64] | None = None,
    vec_systematic_b: NDArray[np.float64] | None = None,
    share_mode: ShareMode = "globals_shared",
    cov_global_a: NDArray[np.float64] | None = None,
    cov_global_b: NDArray[np.float64] | None = None,
    cov_systematic_a: NDArray[np.float64] | None = None,
    cov_systematic_b: NDArray[np.float64] | None = None,
) -> CombinedCovariance:
    """Combine two wells' per-station covariances under a share-mode.

    Parameters
    ----------
    cov_total_a, cov_total_b : (n, 3, 3) ndarray
        Total covariance per station for each well, in metres squared, the
        stations paired row by row.
    vec_global_a, vec_global_b : (n, k, 3) ndarray, optional
        Signed position-error vectors, in metres, of the ``k`` global sources
        the two wells share, at the paired stations, in one source order
        (:func:`source_vectors`). Required when ``share_mode`` shares globals.
    vec_systematic_a, vec_systematic_b : (n, k, 3) ndarray, optional
        The same for the shared systematic sources. Required when
        ``share_mode`` is ``'globals_and_systematic_shared'``.
    share_mode : {'all_independent', 'globals_shared', \
'globals_and_systematic_shared'}
        Which sources are realised identically in the two wells:

        - ``'all_independent'``: ``Σ_A + Σ_B``.
        - ``'globals_shared'`` (default): global sources (geomagnetic
          reference, depth stretch) are one realisation for both wells.
        - ``'globals_and_systematic_shared'``: systematic sources are one
          realisation too; appropriate only where both wells were surveyed
          by the same tool run.

        A shared source contributes ``(v_A − v_B)(v_A − v_B)ᵀ``, which is
        smaller than the independent ``v_A v_Aᵀ + v_B v_Bᵀ`` where the wells
        weight it alike and larger where they weight it oppositely, so sharing
        can widen the combined covariance as well as tighten it.
    cov_global_a, cov_global_b, cov_systematic_a, cov_systematic_b : optional
        Refused: a covariance has lost each source's sign, so it cannot give
        the cross term; pass the ``vec_*`` arguments instead.

    Returns
    -------
    CombinedCovariance

    Raises
    ------
    ValueError
        On unequal covariance shapes, a ``cov_*`` component, a shared mode
        without its vectors, vectors of the wrong shape, or an unknown mode.
    """
    cov_a = np.asarray(cov_total_a, dtype=float).reshape(-1, 3, 3)
    cov_b = np.asarray(cov_total_b, dtype=float).reshape(-1, 3, 3)
    if cov_a.shape != cov_b.shape:
        raise ValueError(
            f"cov_total_a {cov_a.shape} and cov_total_b {cov_b.shape} "
            "must have the same shape (per-station alignment)"
        )
    if any(c is not None for c in (cov_global_a, cov_global_b,
                                   cov_systematic_a, cov_systematic_b)):
        raise ValueError(
            "cov_global_* / cov_systematic_* are not used: a shared source's "
            "contribution needs its signed vector in each well, which a "
            "covariance has lost; pass vec_global_* / vec_systematic_* "
            "(welleng.conditioning.source_vectors)"
        )

    n = cov_a.shape[0]
    cov_naive = cov_a + cov_b

    if share_mode == "all_independent":
        cov_combined = cov_naive.copy()
    elif share_mode in ("globals_shared", "globals_and_systematic_shared"):
        if vec_global_a is None or vec_global_b is None:
            raise ValueError(f"{share_mode} requires vec_global_a and _b")
        cross = _cross(vec_global_a, vec_global_b, n, "vec_global")
        if share_mode == "globals_and_systematic_shared":
            if vec_systematic_a is None or vec_systematic_b is None:
                raise ValueError(
                    "globals_and_systematic_shared requires vec_systematic_a "
                    "and _b as well as vec_global_a and _b"
                )
            cross = cross + _cross(vec_systematic_a, vec_systematic_b, n,
                                   "vec_systematic")
        cov_combined = cov_naive - cross - cross.swapaxes(-1, -2)
    else:
        raise ValueError(f"unknown share_mode {share_mode!r}")

    # The exact form is the covariance of A's error less B's, so it is positive
    # semi-definite whenever the vectors belong to the totals given; a negative
    # eigenvalue past rounding means they do not, and no result is given.
    cov_combined = 0.5 * (cov_combined + cov_combined.swapaxes(-1, -2))
    eig = np.linalg.eigvalsh(cov_combined)
    scale = np.maximum(np.linalg.eigvalsh(cov_naive).max(axis=-1), np.finfo(float).tiny)
    if (eig.min(axis=-1) < -1e-9 * scale).any():
        raise ValueError(
            "the combined covariance is not positive semi-definite: the shared "
            "sources' vectors are not those of the total covariances given"
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
