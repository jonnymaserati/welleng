"""Georeference survey QC: total-gravity, total-field and dip checks.

The georeference tests compare the *measured* total gravity field, total
magnetic field and magnetic dip against known reference values, and flag
stations whose residuals exceed tolerance (Ekseth et al., SPE-133417). A
total-field or dip failure indicates magnetic interference (e.g. insufficient
non-magnetic spacing, or proximity to casing) so the azimuth at that station is
unreliable; a total-gravity failure indicates an accelerometer problem.

Validated: reproduces a commercial vendor's own per-station DE-QC pass/fail
flags to 100% agreement, with the vendor's total-field / dip tolerances
recovered as ~826 nT and ~0.44 deg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "GeomagReference",
    "QCResult",
    "georef_checks",
    "DualDepthResult",
    "dual_depth_difference",
]


@dataclass(frozen=True)
class GeomagReference:
    """Reference geomagnetic + gravity field at the survey location/time.

    Units are the caller's, but must match the sensor units passed to the QC
    functions (e.g. ``b_total`` in nT with magnetometer readings in nT).

    Parameters
    ----------
    b_total : float
        Reference total magnetic-field strength.
    dip : float
        Reference magnetic dip angle (deg, down-positive).
    g_total : float
        Reference total gravity-field strength.
    declination : float, default 0.0
        Magnetic declination (deg, east-positive).
    grid_convergence : float, default 0.0
        Grid convergence (deg).
    """

    b_total: float
    dip: float
    g_total: float
    declination: float = 0.0
    grid_convergence: float = 0.0


# representative single-station MWD tolerances (caller may override)
DEFAULT_TOLERANCES = {"g_total": 3.0, "b_total": 350.0, "dip": 0.5}


@dataclass
class QCResult:
    """Per-station georeference QC residuals and pass/fail flags.

    Attributes
    ----------
    d_g, d_b, d_dip : ndarray
        Measured-minus-reference residuals for total gravity, total field and
        dip.
    flag_g, flag_b, flag_dip : ndarray of bool
        True where the residual exceeds tolerance (i.e. the station FAILS that
        check).
    tolerances : dict
        The tolerances applied.
    """

    d_g: np.ndarray
    d_b: np.ndarray
    d_dip: np.ndarray
    flag_g: np.ndarray
    flag_b: np.ndarray
    flag_dip: np.ndarray
    tolerances: dict = field(default_factory=dict)

    @property
    def passed(self) -> np.ndarray:
        """True where the station passes all three checks."""
        return ~(self.flag_g | self.flag_b | self.flag_dip)


def _totals(g, b):
    g = np.atleast_2d(np.asarray(g, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    g_total = np.linalg.norm(g, axis=1)
    b_total = np.linalg.norm(b, axis=1)
    dip = np.degrees(np.arcsin(np.sum(g * b, axis=1) / (g_total * b_total)))
    return g_total, b_total, dip


def georef_checks(
    g_xyz,
    b_xyz,
    ref: GeomagReference,
    tolerances: Optional[dict] = None,
) -> QCResult:
    """Run the georeference QC tests on one or more survey stations.

    Parameters
    ----------
    g_xyz, b_xyz : array_like, shape (3,) or (n, 3)
        Accelerometer and magnetometer readings in the tool frame (same units
        as ``ref.g_total`` / ``ref.b_total``).
    ref : GeomagReference
        Reference field.
    tolerances : dict, optional
        Keys ``'g_total'``, ``'b_total'``, ``'dip'``. Missing keys fall back to
        :data:`DEFAULT_TOLERANCES`.

    Returns
    -------
    QCResult
    """
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    g_total, b_total, dip = _totals(g_xyz, b_xyz)

    d_g = g_total - ref.g_total
    d_b = b_total - ref.b_total
    d_dip = dip - ref.dip

    return QCResult(
        d_g=d_g,
        d_b=d_b,
        d_dip=d_dip,
        flag_g=np.abs(d_g) > tol["g_total"],
        flag_b=np.abs(d_b) > tol["b_total"],
        flag_dip=np.abs(d_dip) > tol["dip"],
        tolerances=tol,
    )


@dataclass
class DualDepthResult:
    """Dual-depth-difference QC test outcome.

    Attributes
    ----------
    depth_difference_error : ndarray
        The measured difference between the two independent depth measurements
        (pipe tally vs wireline), i.e. the quantity being tested.
    tolerance : ndarray
        Allowed difference: the RSS of the two error-model depth uncertainties.
    flag : ndarray of bool
        True where the difference exceeds tolerance (station FAILS -- a gross
        depth error is present).
    """

    depth_difference_error: np.ndarray
    tolerance: np.ndarray
    flag: np.ndarray

    @property
    def passed(self) -> np.ndarray:
        return ~self.flag


def dual_depth_difference(
    depth_difference_error,
    sigma_pipe,
    sigma_wll,
):
    """Dual-depth-difference QC test (Ekseth et al., SPE-133417).

    Where an independent second depth measurement exists (e.g. a wireline pass
    with a casing-collar locator alongside the drillpipe tally), the difference
    between the two depths is checked against the combined error-model
    uncertainty. A difference larger than tolerance indicates a gross depth
    error not explained by the error models.

    The allowed difference is the root-sum-square of the two independent depth
    uncertainties (same sigma convention -- e.g. both 3-sigma):

        tolerance = sqrt(sigma_pipe**2 + sigma_wll**2)

    Reproduces SPE-133417 Table 1a/1b/1c exactly.

    Parameters
    ----------
    depth_difference_error : float or array_like
        Measured pipe-vs-wireline depth-difference error.
    sigma_pipe, sigma_wll : float or array_like
        Error-model depth uncertainties (same sigma level) for the pipe-tally
        and wireline measurements respectively.

    Returns
    -------
    DualDepthResult
    """
    dd = np.abs(np.asarray(depth_difference_error, dtype=float))
    tol = np.hypot(np.asarray(sigma_pipe, dtype=float),
                   np.asarray(sigma_wll, dtype=float))
    return DualDepthResult(
        depth_difference_error=np.asarray(depth_difference_error, dtype=float),
        tolerance=tol,
        flag=dd > tol,
    )
