"""Correction-uncertainty covariance by Monte-Carlo input propagation.

A survey *correction* (BHA sag, a depth stretch model, an MSA correction) is a
deterministic model of uncertain inputs. The **residual uncertainty left after
correcting** is the covariance of the correction over those input
uncertainties -- propagated here by plain Monte-Carlo: draw perturbed inputs,
re-evaluate the correction, take the covariance of the resulting correction
vectors. Within one run the draws are single realisations, so the returned
matrix carries the full **cross-station correlation** (a biased stabiliser OD
mis-corrects every station the same way) -- exactly the block a covariated
error model consumes.

The correction model is **injected** as a callable, so this module stays a
generic, model-agnostic propagation utility: welleng-core does not depend on
any specific mechanics implementation (e.g. a BHA-sag beam model living in a
consumer package plugs in at the call site). Standard uncertainty propagation
(JCGM 101 / GUM Supplement 1 -- Monte Carlo method).

Example (sketch)::

    def draw(rng):
        g = replace(base_geom,
                    stabilizer_od=base_geom.stabilizer_od + rng.normal(0, od_sig),
                    mud_weight=base_geom.mud_weight + rng.normal(0, mw_sig))
        return lambda: sag_correction(g, survey)     # consumer's model

    res = correction_covariance_mc(draw, n_draws=1000, rng=rng)
    # res.covariance -> (n_stations, n_stations) inc-correction covariance
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CorrectionUncertainty", "correction_covariance_mc"]


@dataclass(frozen=True)
class CorrectionUncertainty:
    """Monte-Carlo correction-uncertainty result.

    Attributes
    ----------
    mean : (n,) ndarray
        Mean correction over the draws (the best-estimate correction under
        input uncertainty; generally ~ the nominal correction).
    covariance : (n, n) ndarray
        Covariance of the correction across draws -- the residual uncertainty
        the correction leaves behind, WITH cross-station correlation (each draw
        is one realisation of the inputs over the whole run).
    std : (n,) ndarray
        Per-station 1-sigma (sqrt of the diagonal).
    n_draws : int
        Draws used.
    se_std : (n,) ndarray
        Monte-Carlo standard error of ``std`` (~ ``std / sqrt(2 (n-1))``) --
        the convergence handle: enlarge ``n_draws`` until ``se_std`` is small
        against the sigma you are quoting.
    """

    mean: np.ndarray
    covariance: np.ndarray
    std: np.ndarray
    n_draws: int
    se_std: np.ndarray


def correction_covariance_mc(draw, n_draws=1000, rng=None):
    """Covariance of a deterministic correction under uncertain inputs (MC).

    Parameters
    ----------
    draw : callable
        ``draw(rng) -> callable() -> (n,) array_like``: draws ONE realisation
        of the uncertain inputs (applying each input's own distribution and
        propagation character -- e.g. a systematic stabiliser-OD offset drawn
        once per run vs a per-station random enlargement drawn per station
        *inside* the closure) and returns a zero-argument callable evaluating
        the correction for that realisation. The correction model itself is
        whatever the closure wraps -- this function never imports it.
    n_draws : int, default 1000
        Monte-Carlo draws. The result carries ``se_std`` to judge convergence.
    rng : numpy.random.Generator, optional
        Source of randomness (default ``numpy.random.default_rng()``).

    Returns
    -------
    CorrectionUncertainty

    Notes
    -----
    - **Zero input uncertainty => zero covariance** (the deterministic limit)
      -- kept as a regression gate.
    - The full (n, n) matrix is returned (not just the diagonal) because the
      within-run input realisation correlates the stations; a covariated error
      model consumes exactly that block. SCALAR REFERENCE -- batched/GPU forms
      belong to welleng-api.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_draws < 2:
        raise ValueError(f"n_draws must be >= 2, got {n_draws}")
    first = np.asarray(draw(rng)(), dtype=float)
    if first.ndim != 1:
        raise ValueError(
            f"correction must return a 1-D per-station array, got shape "
            f"{first.shape}"
        )
    n = first.shape[0]
    X = np.empty((n_draws, n))
    X[0] = first
    for k in range(1, n_draws):
        xk = np.asarray(draw(rng)(), dtype=float)
        if xk.shape != (n,):
            raise ValueError(
                f"draw {k}: correction shape {xk.shape} != ({n},) -- the "
                "station set must be identical across draws"
            )
        X[k] = xk
    mean = X.mean(axis=0)
    dev = X - mean
    cov = (dev.T @ dev) / (n_draws - 1)
    cov = 0.5 * (cov + cov.T)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    se = std / np.sqrt(2.0 * (n_draws - 1))
    return CorrectionUncertainty(
        mean=mean, covariance=cov, std=std, n_draws=n_draws, se_std=se
    )
