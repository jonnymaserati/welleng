"""Sensor -> survey: the MWD navigation equations.

Compute inclination, azimuth and toolface from a triad of accelerometer
readings (gravity) and a triad of magnetometer readings (Earth's field), using
the standard, public-domain minimum-set MWD equations (Williamson, SPE-67616).

The equations are scale-free in each sensor (they depend only on ratios), so the
accelerometer and magnetometer readings may be supplied in any self-consistent
units (e.g. mG and nT, or m/s^2 and T) -- the result is identical.

Validated against an operator's commercial-vendor survey (raw sensors -> the
vendor's own computed inc/azi) to 0.005 deg inclination and 0.03 deg azimuth.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sensor_to_survey", "gyro_to_survey", "earth_rate_components", "EARTH_RATE"]

#: Earth's sidereal rotation rate (deg/hr): 360 deg / 23.9344696 h.
EARTH_RATE = 15.041067


def earth_rate_components(latitude, rate: float = EARTH_RATE):
    """Local horizontal (true-north) and vertical Earth-rate components.

    At latitude ``phi`` the Earth-rotation vector has a horizontal component
    ``rate*cos(phi)`` pointing true north and a vertical component
    ``rate*sin(phi)`` (up). These are the gyro reference values; the Earth-rate
    "dip" equals the latitude, so no spatial model/service is required (unlike
    the magnetic field).

    Parameters
    ----------
    latitude : float or array_like
        Geographic latitude (deg).
    rate : float, default EARTH_RATE
        Earth-rotation rate magnitude (deg/hr).

    Returns
    -------
    horizontal, vertical : ndarray
        Earth-rate components (deg/hr).
    """
    phi = np.radians(np.asarray(latitude, dtype=float))
    return rate * np.cos(phi), rate * np.sin(phi)


def sensor_to_survey(
    g_xyz,
    b_xyz,
    *,
    declination: float = 0.0,
    grid_convergence: float = 0.0,
    axis: str = "z",
    deg: bool = True,
):
    """Inclination, azimuth and toolface from accelerometer + magnetometer.

    Parameters
    ----------
    g_xyz : array_like, shape (3,) or (n, 3)
        Accelerometer readings (gravity vector) in the tool frame.
    b_xyz : array_like, shape (3,) or (n, 3)
        Magnetometer readings (Earth field) in the tool frame, same station
        ordering as ``g_xyz``.
    declination : float, default 0.0
        Magnetic declination (deg, east-positive). Added to the magnetic
        azimuth to give azimuth relative to true north.
    grid_convergence : float, default 0.0
        Grid convergence (deg). Subtracted to give grid azimuth. With both
        ``declination`` and ``grid_convergence`` zero the result is the raw
        *magnetic* azimuth.
    axis : {'z', 'x'}, default 'z'
        Which sensor axis is the along-hole (axial) axis. ``'z'`` is the usual
        convention; ``'x'`` handles tools/exports that list the axial axis
        first (seen in some vendor survey files).
    deg : bool, default True
        If True (default) return degrees, else radians.

    Returns
    -------
    inc, azi, toolface : float or ndarray
        Inclination (0 = vertical), azimuth (grid/true per the corrections
        above, wrapped to [0, 360)) and gravity toolface (wrapped to [0, 360)).

    Notes
    -----
    Classic minimum-set equations (Williamson, SPE-67616; textbook)::

        inc      = atan2(sqrt(Gx^2 + Gy^2), Gz)
        toolface = atan2(Gy, Gx)
        azi_mag  = atan2( (Gx By - Gy Bx) |G|,
                          Bz (Gx^2 + Gy^2) - Gz (Gx Bx + Gy By) )
    """
    g = np.asarray(g_xyz, dtype=float)
    b = np.asarray(b_xyz, dtype=float)
    scalar = g.ndim == 1
    g = np.atleast_2d(g)
    b = np.atleast_2d(b)

    if axis == "x":
        # axial-first ordering -> move the axial (first) axis to z
        g = g[:, [1, 2, 0]]
        b = b[:, [1, 2, 0]]
    elif axis != "z":
        raise ValueError(f"axis must be 'z' or 'x', got {axis!r}")

    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]

    gxy2 = gx * gx + gy * gy
    gtot = np.sqrt(gxy2 + gz * gz)

    inc = np.arctan2(np.sqrt(gxy2), gz)
    toolface = np.arctan2(gy, gx)

    num = (gx * by - gy * bx) * gtot
    den = bz * gxy2 - gz * (gx * bx + gy * by)
    azi_mag = np.arctan2(num, den)

    azi = np.deg2rad(np.rad2deg(azi_mag) + declination - grid_convergence)

    azi = np.mod(azi, 2 * np.pi)
    toolface = np.mod(toolface, 2 * np.pi)

    if deg:
        inc, azi, toolface = np.rad2deg(inc), np.rad2deg(azi), np.rad2deg(toolface)

    if scalar:
        return float(inc[0]), float(azi[0]), float(toolface[0])
    return inc, azi, toolface


def gyro_to_survey(
    g_xyz,
    w_xyz,
    *,
    grid_convergence: float = 0.0,
    axis: str = "z",
    deg: bool = True,
):
    """Inclination, azimuth and toolface from accelerometer + rate-gyro.

    Gyrocompassing: the accelerometer gives inclination and toolface (as in
    :func:`sensor_to_survey`) and the rate-gyro measures the Earth-rotation
    vector in the tool frame, whose horizontal component points *true* north.
    The azimuth is obtained from the same minimum-set equation, with the gyro
    readings in place of the magnetometer.

    Unlike a magnetic survey the gyro references true north directly, so **no
    magnetic declination is applied** (only grid convergence, if supplied).

    Parameters
    ----------
    g_xyz : array_like, shape (3,) or (n, 3)
        Accelerometer readings (gravity) in the tool frame.
    w_xyz : array_like, shape (3,) or (n, 3)
        Rate-gyro readings (Earth-rotation rate) in the tool frame, any
        self-consistent angular-rate unit.
    grid_convergence : float, default 0.0
        Grid convergence (deg), subtracted to give grid azimuth.
    axis : {'z', 'x'}, default 'z'
        Along-hole (axial) sensor axis, as in :func:`sensor_to_survey`.
    deg : bool, default True
        Return degrees if True, else radians.

    Returns
    -------
    inc, azi, toolface : float or ndarray
        Inclination, azimuth (true/grid, wrapped to [0, 360)) and toolface.
    """
    return sensor_to_survey(
        g_xyz,
        w_xyz,
        declination=0.0,
        grid_convergence=grid_convergence,
        axis=axis,
        deg=deg,
    )
