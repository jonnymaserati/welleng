"""Multi-station analysis (MSA): estimate the tool's actual sensor errors.

Given many survey stations, MSA estimates the constant per-axis sensor *bias*
and *scale-factor* errors of a triad (magnetometer or accelerometer) from the
redundancy that every station's measured total field must equal the reference
(Ekseth et al., SPE-133417, multistation test). This is the inverse of the
forward error model: instead of assuming the model's error magnitudes, it
measures them -- so a survey's *actual* performance can be checked against the
error model applied to it (its EOU is only valid if the actual errors are within
the model).

Closed form
-----------
The constraint ``|B_true| = B_ref`` with ``B_true = (B_meas - b) / (1 + s)``
linearises (small errors) to a *linear* equation in the six unknowns
``p = [bx, by, bz, sx, sy, sz]`` per station::

    2 [Bx, By, Bz, Bx^2, By^2, Bz^2] . p = |B_meas|^2 - B_ref^2

Stacking the stations gives ``A p = d`` solved in closed form by ordinary least
squares, ``p = (A^T A)^-1 A^T d`` -- one matrix solve, no iteration. The
estimate covariance ``(A^T A)^-1 A^T diag(sigma_d^2) A (A^T A)^-1`` is available
analytically and yields the per-component *estimability* and the correlation
matrix, which gate poorly-observed components.

Estimability / the geometry gate
--------------------------------
MSA estimability is governed by the survey's *directional* (azimuth / toolface)
diversity, not its inclination. In particular the *axial* sensor error is
intrinsically weakly observed unless the well changes direction substantially;
in the industry it is handled by a separate axial (drillstring) interference
correction and in-field referencing. This module therefore returns the estimate
*with* its analytical standard errors and a per-component ``estimable`` verdict,
rather than a blanket, possibly-meaningless, error vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .qc import GeomagReference

__all__ = ["MSAResult", "estimate_sensor_errors", "apply_sensor_errors"]

_AXES = ("x", "y", "z")


@dataclass
class MSAResult:
    """Estimated tool sensor errors and their estimability.

    Attributes
    ----------
    bias : ndarray, shape (3,)
        Estimated per-axis bias (sensor units).
    scale : ndarray, shape (3,)
        Estimated per-axis fractional scale-factor error.
    covariance : ndarray, shape (6, 6)
        Estimate covariance for ``[bx, by, bz, sx, sy, sz]``.
    std : ndarray, shape (6,)
        Standard errors (sqrt of the covariance diagonal).
    correlation : ndarray, shape (6, 6)
        Correlation matrix of the estimate.
    estimable : ndarray of bool, shape (6,)
        Per-component verdict: False where the standard error exceeds the
        estimability threshold (that component is not reliably observed by this
        survey's geometry).
    n_stations : int
        Number of stations used.
    condition_number : float
        Condition number of ``A^T A`` (large => ill-conditioned geometry).
    """

    bias: np.ndarray
    scale: np.ndarray
    covariance: np.ndarray
    std: np.ndarray
    correlation: np.ndarray
    estimable: np.ndarray
    n_stations: int
    condition_number: float

    @property
    def axial_estimable(self) -> bool:
        """Whether the axial (z) bias and scale are both reliably observed."""
        return bool(self.estimable[2] and self.estimable[5])


def estimate_sensor_errors(
    triad,
    ref: GeomagReference,
    *,
    sensor: str = "mag",
    noise: Optional[float] = None,
    bias_estimability: float = 500.0,
    scale_estimability: float = 0.01,
    min_stations: int = 10,
) -> MSAResult:
    """Closed-form multi-station estimate of tool bias + scale errors.

    Parameters
    ----------
    triad : array_like, shape (n, 3)
        Per-station triad readings (magnetometer if ``sensor='mag'``, else
        accelerometer), same units as the matching reference total.
    ref : GeomagReference
        Reference field; ``ref.b_total`` (mag) or ``ref.g_total`` (accel) is the
        target total.
    sensor : {'mag', 'accel'}, default 'mag'
        Which triad is supplied (selects the reference total).
    noise : float, optional
        Per-axis measurement noise standard deviation (same units), used to
        propagate the estimate covariance. If None, a unit noise is used so the
        *relative* estimability and the correlation matrix are still meaningful
        but the absolute standard errors are only proportional.
    bias_estimability, scale_estimability : float
        Standard-error thresholds above which a bias / scale component is judged
        not estimable by this geometry.
    min_stations : int, default 10
        Minimum independent stations required (SPE-133417); fewer raises.

    Returns
    -------
    MSAResult
    """
    if sensor not in ("mag", "accel"):
        raise ValueError(f"sensor must be 'mag' or 'accel', got {sensor!r}")
    B = np.atleast_2d(np.asarray(triad, dtype=float))
    n = len(B)
    if n < min_stations:
        raise ValueError(
            f"MSA needs at least {min_stations} independent stations, got {n}"
        )
    ref_total = ref.b_total if sensor == "mag" else ref.g_total

    # linearised design matrix  A p = d
    A = np.empty((n, 6))
    A[:, 0:3] = 2.0 * B
    A[:, 3:6] = 2.0 * B * B
    d = np.sum(B * B, axis=1) - ref_total**2

    AtA = A.T @ A
    AtA_inv = np.linalg.inv(AtA)
    p = AtA_inv @ (A.T @ d)

    # analytical estimate covariance via linearised noise propagation on d:
    #   d_i = |B_i|^2 - ref^2  =>  var(d_i) ~ (2 |B_i| sigma)^2   (dominant term)
    sigma = 1.0 if noise is None else float(noise)
    sig_d = 2.0 * np.linalg.norm(B, axis=1) * sigma
    cov = AtA_inv @ (A.T @ (sig_d[:, None] ** 2 * A)) @ AtA_inv
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)

    thresholds = np.array([bias_estimability] * 3 + [scale_estimability] * 3)
    estimable = std < thresholds

    return MSAResult(
        bias=p[:3],
        scale=p[3:],
        covariance=cov,
        std=std,
        correlation=corr,
        estimable=estimable,
        n_stations=n,
        condition_number=float(np.linalg.cond(AtA)),
    )


def apply_sensor_errors(triad, result, *, only_estimable=False):
    """Correct triad readings for the MSA-estimated bias and scale errors.

    Closes the MSA loop: :func:`estimate_sensor_errors` measures the sensor
    errors, this removes them, ``B_true = (B_meas - b) / (1 + s)`` (the model of
    the closed form above), so the corrected triad can be re-run through
    :func:`~welleng.interpretation.forward.sensor_to_survey` for a corrected
    survey. The correction is opt-in — MSA is often used to *flag* out-of-model
    tool performance, not to auto-correct raw sensors.

    Parameters
    ----------
    triad : array_like, shape (n, 3)
        Per-station triad readings (same sensor/units as passed to
        :func:`estimate_sensor_errors`).
    result : MSAResult
        The estimate to apply.
    only_estimable : bool, default False
        If True, apply a component's bias/scale only where
        ``result.estimable`` is set (poorly-observed components — typically the
        axial term — are left uncorrected rather than corrected by a meaningless
        estimate). If False, apply all six.

    Returns
    -------
    numpy.ndarray
        (n, 3) corrected triad.
    """
    B = np.atleast_2d(np.asarray(triad, dtype=float))
    bias = np.asarray(result.bias, dtype=float).copy()
    scale = np.asarray(result.scale, dtype=float).copy()
    if only_estimable:
        est = np.asarray(result.estimable, dtype=bool)
        bias[~est[:3]] = 0.0
        scale[~est[3:]] = 0.0
    return (B - bias) / (1.0 + scale)
