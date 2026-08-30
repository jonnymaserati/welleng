import re
from typing import Annotated, Literal, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


def get_dogleg(inc1, azi1, inc2, azi2):
    """
    Compute the dogleg angle between two survey stations (vectorised).

    Uses the numerically stable Haversine form to avoid arccos precision loss
    at small angles.

    Parameters
    ----------
    inc1, azi1: float or array — inclination / azimuth at station 1 (radians)
    inc2, azi2: float or array — inclination / azimuth at station 2 (radians)

    Returns
    -------
    dogleg: float or array — dogleg angle in radians
    """
    # Clamp the haversine argument to [0, 1]: it is a sum of squares that is
    # mathematically <= 1, but FP rounding can push it a few ulps past 1.0 at
    # the antiparallel edge (a horizontal 180 deg turn: inc1==inc2==90, azi
    # differing by 180), where sqrt of >1 -> arcsin NaN. The clamp makes that
    # exact-pi U-turn return pi instead of NaN (welleng #307). No effect on any
    # in-domain value.
    return 2.0 * np.arcsin(np.sqrt(np.clip(
        np.sin((inc2 - inc1) / 2) ** 2
        + np.sin(inc1) * np.sin(inc2) * np.sin((azi2 - azi1) / 2) ** 2,
        0.0, 1.0,
    )))


def get_rf(dogleg):
    """
    Compute the ratio factor (RF) for minimum curvature (vectorised).

    Returns 1.0 where dogleg is 0 (limit of the function as dogleg → 0).

    Parameters
    ----------
    dogleg: float or array — dogleg angle(s) in radians

    Returns
    -------
    rf: float or array — ratio factor(s)
    """
    dogleg = np.asarray(dogleg, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rf = np.where(dogleg == 0.0, 1.0, 2.0 / dogleg * np.tan(dogleg / 2))
    return float(rf) if rf.ndim == 0 else rf


def arc_step(v1, v2, theta, dmd, x):
    """Minimum-curvature arc kernel -- THE single home (welleng #308).

    Every min-curve arc evaluation in the library routes through this one
    function so the derivation lives ONCE: ``MinCurve.interpolate`` (interior
    MD query), :func:`min_curve_step` (station construction, ``x = dmd``) and
    :func:`welleng.connector.interpolate_curve` (dense arc render) are all
    callers, not re-derivations of it.

    Parameters
    ----------
    v1, v2 : (n, 3) array
        Start / end UNIT tangents of a min-curve leg, in ANY consistent
        orthonormal basis -- the result comes back in that SAME basis (the
        kernel is coordinate-agnostic; callers pass [E, N, V] or [N, E, V] and
        read it back in kind).
    theta : (n,) array
        Leg dogleg (radians).
    dmd : (n,) array
        Leg length.
    x : (n,) array
        Arc length from the ``v1`` station to the query point
        (``0 <= x <= dmd``); ``x = dmd`` gives the far station.

    Returns
    -------
    disp, tangent : (n, 3) arrays
        Local displacement from the ``v1`` station to the query point, and the
        unit tangent there.

    Notes
    -----
    Position is the half-angle form -- the symbolic-identity rewrite of the
    canonical ``R[sin(phi) v1 + (1-cos(phi)) u]`` that pairs each vector sum
    with ITS OWN denominator (``|v1+v2| = 2cos(theta/2)``,
    ``|v1-v2| = 2sin(theta/2)``) so neither ratio amplifies as ``theta -> pi``
    (~0.1 ulp near pi vs tens of ulp for the ``/sin`` and ``rf = (2/phi)
    tan(phi/2)`` forms). It is applied for ``theta`` in ``[0, pi]`` (the
    ``get_dogleg`` range). The tangent is the Rodrigues ``u``-form
    ``cos(phi) v1 + sin(phi) u`` with ``u = (v2 - cos(theta) v1)/sin(theta)``:
    a one-time ``1/sin`` set-up with no per-query amplifier, and -- unlike the
    normalise-the-derivative shortcut -- it keeps the correct sign for
    ``theta > pi`` (the long-way arcs ``interpolate_curve`` renders). The exact
    antiparallel turn (``theta = pi``, arc plane undetermined) is left as the
    start tangent.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    theta = np.asarray(theta, dtype=float)
    dmd = np.asarray(dmd, dtype=float)
    x = np.asarray(x, dtype=float)
    # Broadcast-clean over ANY leading batch shape: theta/dmd/x are (...) and
    # v1/v2 are (..., 3). np.where over guarded intermediates (no boolean
    # masking / reshape) so N-D batched callers -- e.g. cov/dense-sweep surfaces
    # feeding (n_leg, n_query, 3) -- work exactly like the 1-D case (welleng
    # #308 regression, api-caught: the old inline forms were broadcast-clean).
    s = np.sin(theta)
    curved = (theta >= 1e-14) & (np.abs(s) >= 1e-12)   # (...) plane well-defined
    th = np.where(curved, theta, 1.0)                  # guarded denominators
    sin_th = np.where(curved, s, 1.0)
    dmd_safe = np.where(dmd == 0.0, 1.0, dmd)
    phi = x * theta / dmd_safe                         # partial dogleg
    R = dmd / th
    cw = curved[..., None]
    # Position: half-angle where curved, else the straight chord x*v1.
    hc = np.cos((th - phi) / 2) / np.cos(th / 2)
    hs = np.sin((th - phi) / 2) / np.sin(th / 2)
    disp_curved = (R * np.sin(phi / 2))[..., None] * (
        (v1 + v2) * hc[..., None] + (v1 - v2) * hs[..., None]
    )
    disp = np.where(cw, disp_curved, x[..., None] * v1)
    # Tangent: Rodrigues u-form where curved, else the start tangent v1.
    u = (v2 - np.cos(th)[..., None] * v1) / sin_th[..., None]
    tang_curved = np.cos(phi)[..., None] * v1 + np.sin(phi)[..., None] * u
    tangent = np.where(cw, tang_curved, v1)
    return disp, tangent


def min_curve_step(delta_md, inc1, azi1, inc2, azi2, rf=None):
    """
    Compute position increments using minimum curvature (vectorised).

    Delegates the geometry to the single arc kernel :func:`arc_step` (welleng
    #308) evaluated at the far station (``x = dmd``), so a station position can
    no longer diverge from ``MinCurve.interpolate`` / ``interpolate_curve``.

    Parameters
    ----------
    delta_md: (n,) array — measured-depth increments
    inc1, azi1: (n,) arrays — start inclination / azimuth (radians)
    inc2, azi2: (n,) arrays — end inclination / azimuth (radians)
    rf: ignored — retained for backward compatibility (the arc kernel no longer
        needs a precomputed ratio factor).

    Returns
    -------
    deltas: (n, 3) array — position increments in [N, E, V] order
    """
    scalar = np.ndim(inc1) == 0
    inc1, azi1 = np.atleast_1d(inc1), np.atleast_1d(azi1)
    inc2, azi2 = np.atleast_1d(inc2), np.atleast_1d(azi2)
    dmd = np.atleast_1d(np.asarray(delta_md, dtype=float))
    si1, ci1 = np.sin(inc1), np.cos(inc1)
    si2, ci2 = np.sin(inc2), np.cos(inc2)
    ca1, sa1 = np.cos(azi1), np.sin(azi1)
    ca2, sa2 = np.cos(azi2), np.sin(azi2)
    v1 = np.stack((si1 * ca1, si1 * sa1, ci1), axis=-1)   # [N, E, V]
    v2 = np.stack((si2 * ca2, si2 * sa2, ci2), axis=-1)
    theta = get_dogleg(inc1, azi1, inc2, azi2)
    disp, _ = arc_step(v1, v2, theta, dmd, dmd)
    return disp[0] if scalar else disp


def _horizontal_tangent_delta(u1, u2, alpha):
    """Subtended angle at which a min-curve arc's tangent becomes horizontal
    (its TVD turning point), or ``None`` if not in the open interval
    ``(0, alpha)``. Vertical specialisation of Sawaryn & Thorogood (2005,
    SPE-84246-PA, Eq. 31): the tangent's vertical component (SLERP of the end
    tangents) vanishes where ``sin(alpha-d) u1 + sin(d) u2 = 0`` -> ``tan(d) =
    -sin(alpha) u1 / (u2 - cos(alpha) u1)``. ``u1``/``u2`` are the start/end
    unit-tangent vertical components; a dogleg is at most ``pi`` so at most one
    such point exists.
    """
    p_term = np.sin(alpha) * u1
    q_term = u2 - np.cos(alpha) * u1
    if abs(p_term) < 1e-15 and abs(q_term) < 1e-15:
        return None
    base = np.arctan2(-p_term, q_term)
    for cand in (base, base + np.pi, base - np.pi):
        if 1e-12 < cand < alpha - 1e-12:
            return cand
    return None


def _arc_tvd_crossings(u1, u2, alpha, delta_md, dvert):
    """Subtended angles in ``[0, alpha]`` at which a min-curve arc reaches a
    target TVD. Closed-form *Interpolation at a Plane* of Sawaryn & Thorogood
    (2005, SPE-84246-PA), Eqs. 25-27 + Eq. 1, specialised to a horizontal target
    plane. ``u1``/``u2`` are the start/end unit-tangent vertical components,
    ``alpha`` the dogleg, ``delta_md`` the arc length, ``dvert`` the target TVD
    minus the arc-start TVD. Returns the 0, 1 or 2 real roots (discriminant
    guarded).
    """
    a = u1 * np.sin(alpha)
    b = u1 * np.cos(alpha) - u2
    c = dvert * alpha * np.sin(alpha) / delta_md + b
    disc = a * a + b * b - c * c
    if disc < -1e-12:
        return []
    disc = max(disc, 0.0)
    root = disc ** 0.5
    out = []
    for sign in (1.0, -1.0):
        d = 2.0 * np.arctan2(a + sign * root, b + c)
        d %= (2 * np.pi)
        if d > alpha:
            if abs(d - 2 * np.pi) < 1e-7:
                d = 0.0
            else:
                continue
        out.append(min(max(d, 0.0), alpha))
        if root == 0.0:
            break
    return out


class MinCurve:
    def __init__(
        self,
        md,
        inc,
        azi,
    ):
        """
        Generate LOCAL geometric data from a well bore survey.

        Positions (``poss``) are in local coordinates relative to the origin;
        MinCurve is azimuth-reference agnostic (the caller knows which reference
        ``azi`` is in) and holds no surface/start position or datum state -- that
        belongs to the owning ``Survey``, which applies the start offset, grid
        scale factor and NEV interpretation to interpret this local geometry.

        Parameters
        ----------
        md: list or 1d array of floats
            Measured depth along well path from a datum.
        inc: list or 1d array of floats
            Well path inclination (relative to z/tvd axis where 0
            indicates down), in radians.
        azi: list or 1d array of floats
            Well path azimuth (relative to y/North axis),
            in radians.

        Notes
        -----
        MinCurve is units-agnostic: ``md`` may be in any length unit and the
        geometry is all ratios/angles. Dogleg severity (which needs a per-unit
        coefficient) is the :meth:`dls` method, into which the caller injects the
        coefficient for its units.
        """

        self.md = md
        survey_length = len(self.md)
        assert survey_length > 1, "Survey must have at least two rows"

        self.inc = inc
        self.azi = azi

        inc = np.array(inc)
        azi = np.array(azi)
        # Per-station unit tangents are constants; cache them once so
        # interpolate() doesn't recompute get_vec on every query (welleng #307).
        self._tangents = get_vec(inc, azi, deg=False)
        inc_1, inc_2 = inc[:-1], inc[1:]
        azi_1, azi_2 = azi[:-1], azi[1:]

        self.delta_md = np.diff(self.md, prepend=0)

        self.dogleg = np.zeros(survey_length)
        self.dogleg[1:] = get_dogleg(inc_1, azi_1, inc_2, azi_2)

        self.rf = np.ones(survey_length)
        self.rf[1:] = get_rf(self.dogleg[1:])

        # Curvature radius per station (arc length / dogleg = delta_md / dogleg),
        # inf where the section is straight. This is the convention-free geometric
        # quantity MinCurve operates in: its length unit simply follows md, and it
        # carries no dogleg-severity convention. DLS (deg/30m or deg/100ft) is the
        # caller's units-aware derivation from this radius (see Survey / utils
        # dls_from_radius).
        self.curve_radius = np.full(survey_length, np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.curve_radius[1:] = np.where(
                self.dogleg[1:] > 0.0,
                self.delta_md[1:] / self.dogleg[1:],
                np.inf,
            )

        # compute all three coordinate deltas in a single trig pass
        deltas = min_curve_step(
            self.delta_md[1:], inc_1, azi_1, inc_2, azi_2, self.rf[1:]
        )
        self.delta_y = np.zeros(survey_length); self.delta_y[1:] = deltas[:, 0]
        self.delta_x = np.zeros(survey_length); self.delta_x[1:] = deltas[:, 1]
        self.delta_z = np.zeros(survey_length); self.delta_z[1:] = deltas[:, 2]

        # cumulate the coordinates -> LOCAL positions (origin-relative). The
        # caller applies any start/surface offset; MinCurve holds no datum state.
        # column_stack + cumsum directly; the previous np.vstack(...) wrapper was
        # pure overhead (atleast_2d + a per-row stack dispatcher) on the hot path.
        self.poss = np.cumsum(
            np.column_stack((self.delta_x, self.delta_y, self.delta_z)), axis=0
        )

    def interpolate(self, md, angles=False):
        """Minimum-curvature position at arbitrary measured depth(s).

        Interpolates along the min-curve ARC between the bracketing stations
        (closed-form half-angle position + its analytic tangent, welleng #308)
        -- **never a straight chord**. This is the light MD->TVD (and
        n/e) path for shoes / formation tops etc.: no ``Survey``, no covariance,
        unit-agnostic (the result's length unit follows ``md``).

        Parameters
        ----------
        md : float or array_like
            Measured depth(s) to interpolate at.

        Returns
        -------
        numpy.ndarray
            LOCAL ``(East, North, TVD)`` position relative to station 0 (same
            column order as :attr:`poss`) -- shape ``(3,)`` for scalar ``md``
            else ``(n, 3)``. The caller adds any datum/start offset (MinCurve
            holds no datum). ``md`` outside the survey range yields ``nan``.

        Notes
        -----
        NEVER linear-interpolate a trajectory. Agrees with
        :meth:`welleng.survey.Survey.interpolate_md` to sub-ulp for doglegs up
        to ~2 rad; near ``pi`` this half-angle form is the better-conditioned of
        the two (~0.1 ulp vs tens for the balanced-tangential node path).
        """
        scalar = np.ndim(md) == 0
        q = np.atleast_1d(np.asarray(md, dtype=float))
        mds = np.asarray(self.md, dtype=float)
        in_range = (q >= mds[0]) & (q <= mds[-1])
        idx = np.clip(np.searchsorted(mds, q, side="left") - 1, 0, len(mds) - 2)
        x = q - mds[idx]
        DL = self.dogleg[idx + 1]
        dmd = self.delta_md[idx + 1]
        # Single arc kernel (welleng #308): local displacement + query tangent
        # from the bracketing station, in the tangents' [E, N, V] basis.
        disp, t_query = arc_step(
            self._tangents[idx], self._tangents[idx + 1], DL, dmd, x
        )
        pos = self.poss[idx].astype(float) + disp
        pos[~in_range] = np.nan
        if angles:
            # inc/azi only for the angles=True return -- one get_angles on the
            # query tangent, not a per-position round-trip.
            ang = get_angles(t_query)
            inc_i = np.where(in_range, ang[:, 0], np.nan)
            azi_i = np.where(in_range, ang[:, 1], np.nan)
            if scalar:
                return pos[0], float(inc_i[0]), float(azi_i[0])
            return pos, inc_i, azi_i
        return pos[0] if scalar else pos

    def tvd_turning_points(self):
        """Measured depths where the path's TVD turns (passes horizontal).

        Between consecutive turning points TVD is monotonic in MD, so these are
        where a TVD-domain treatment must be cut to stay single-valued. Closed
        form -- Sawaryn & Thorogood (2005, SPE-84246-PA) *Turning Point* (Eq. 31)
        per minimum-curvature leg (see :func:`_horizontal_tangent_delta`). MDs
        are in ``self.md``'s units; empty if TVD is monotonic throughout.
        """
        u = np.cos(np.asarray(self.inc, dtype=float))   # vertical tangent comp.
        mds = []
        for i in range(len(self.md) - 1):
            alpha = self.dogleg[i + 1]
            dmd = self.delta_md[i + 1]
            if dmd == 0 or np.isnan(alpha) or alpha <= 1e-9:
                continue
            d_tp = _horizontal_tangent_delta(u[i], u[i + 1], alpha)
            if d_tp is not None:
                mds.append(self.md[i] + d_tp / alpha * dmd)
        return np.array(sorted(mds), dtype=float)

    def interpolate_tvd(self, tvd):
        """Measured depth(s) where the path reaches a target (local) TVD.

        The INVERSE of position-at-md. Reversal-robust: does NOT assume
        monotonic TVD -- each min-curve arc is split at its turning point into
        monotonic spans and every crossing is solved in closed form (Sawaryn &
        Thorogood 2005, SPE-84246-PA, *Interpolation at a Plane*, Eqs. 25-27 +
        Eq. 1; see :func:`_arc_tvd_crossings`). Returns all crossing MDs, sorted.

        ``tvd`` is in the LOCAL frame (relative to station 0, like the TVD column
        of :attr:`poss`); ``Survey`` layers its datum on top. Empty if the target
        is never reached.
        """
        z = self.poss[:, 2]
        u = np.cos(np.asarray(self.inc, dtype=float))
        tol_md, tol_ang = 1e-6, 1e-9
        crossings = []
        for i in range(len(self.md) - 1):
            alpha = self.dogleg[i + 1]
            dmd = self.delta_md[i + 1]
            if dmd == 0:
                continue
            v1, v2 = z[i], z[i + 1]
            if np.isnan(alpha) or alpha <= tol_ang:      # straight: linear in MD
                dv = v2 - v1
                if abs(dv) <= tol_ang:
                    if abs(tvd - v1) <= tol_md:
                        crossings.append(self.md[i])
                    continue
                frac = (tvd - v1) / dv
                if -1e-9 <= frac <= 1 + 1e-9:
                    crossings.append(self.md[i] + min(max(frac, 0.0), 1.0) * dmd)
                continue
            u1, u2 = u[i], u[i + 1]
            d_tp = _horizontal_tangent_delta(u1, u2, alpha)
            breaks = [0.0, alpha] if d_tp is None else [0.0, d_tp, alpha]
            for da, db in zip(breaks[:-1], breaks[1:]):
                va = v1 if da == 0.0 else self.interpolate(
                    self.md[i] + da / alpha * dmd)[2]
                vb = v2 if db == alpha else self.interpolate(
                    self.md[i] + db / alpha * dmd)[2]
                lo, hi = (va, vb) if va <= vb else (vb, va)
                if not (lo - tol_md <= tvd <= hi + tol_md):
                    continue
                for d in _arc_tvd_crossings(u1, u2, alpha, dmd, tvd - v1):
                    if da - 1e-7 <= d <= db + 1e-7:
                        crossings.append(
                            self.md[i] + min(max(d / alpha * dmd, 0.0), dmd))
        crossings.sort()
        out, last = [], None
        for md in crossings:
            if last is not None and abs(md - last) <= tol_md:
                continue
            out.append(md)
            last = md
        return np.array(out, dtype=float)


def get_vec(inc, azi, nev=False, r=1, deg=True):
    """
    Convert inc and azi into a vector.

    Parameters
    ----------
    inc: array of n floats
        Inclination relative to the z-axis (up)
    azi: array of n floats
        Azimuth relative to the y-axis
    r: float or array of n floats
        Scalar to return a scaled vector

    Returns
    -------
    vec: arraylike
        An (n,3) array of vectors
    """
    if deg:
        inc_rad, azi_rad = np.radians(np.array([inc, azi]))
    else:
        inc_rad = inc
        azi_rad = azi
    y = r * np.sin(inc_rad) * np.cos(azi_rad)
    x = r * np.sin(inc_rad) * np.sin(azi_rad)
    z = r * np.cos(inc_rad)

    if nev:
        vec = np.column_stack([y, x, z])
    else:
        vec = np.column_stack([x, y, z])

    return vec / np.linalg.norm(vec, axis=-1).reshape(-1, 1)


def get_nev(
    pos, start_xyz=np.array([0., 0., 0.]), start_nev=np.array([0., 0., 0.])
):
    """
    Convert [x, y, z] coordinates to [n, e, tvd] coordinates.

    Parameters
    ----------
    pos: (n,3) array of floats
        Array of [x, y, z] coordinates
    start_xyz: (,3) array of floats
        The datum of the [x, y, z] cooardinates
    start_nev: (,3) array of floats
        The datum of the [n, e, tvd] coordinates

    Returns
    -------
        An (n,3) array of [n, e, tvd] coordinates.
    """
    # e, n, v = (
    #     np.array([pos]).reshape(-1,3) - np.array([start_xyz])
    # ).T
    e, n, v = (
        np.array([pos]).reshape(-1, 3) - np.array([start_xyz])
    ).T

    return np.column_stack([n, e, v]) + np.array([start_nev])


def get_xyz(pos, start_xyz=[0., 0., 0.], start_nev=[0., 0., 0.]):
    y, x, z = (
        np.array([pos]).reshape(-1, 3) - np.array([start_nev])
    ).T

    return np.column_stack([x, y, z]) + np.array([start_xyz])


def _get_angles(vec):
    xy = vec[:, 0] ** 2 + vec[:, 1] ** 2
    inc = np.arctan2(np.sqrt(xy), vec[:, 2])  # for elevation angle defined from Z-axis down
    azi = (np.arctan2(vec[:, 0], vec[:, 1]) + (2 * np.pi)) % (2 * np.pi)

    return np.stack((inc, azi), axis=1)


def arc_inc_azi_extrema(vec_a, vec_b, dogleg, vertical_eps=1e-4):
    """Exact inclination + azimuth extrema over minimum-curvature arcs.

    A minimum-curvature segment is a planar circular arc whose unit tangent
    sweeps ``t(theta) = vec_a * cos(theta) + u * sin(theta)`` for
    ``theta in [0, dogleg]``, where ``u`` is the in-plane unit vector
    perpendicular to ``vec_a`` (so ``t(0) = vec_a``, ``t(dogleg) = vec_b``).
    All inputs/outputs are in the NEV (north, east, tvd-down) frame.

    Two closed-form results (verified against dense sampling):

    - **Inclination** ``inc = acos(t_V)`` with ``t_V = A cos + B sin``
      (``A = vec_a_V``, ``B = u_V``); its extrema are at the arc ends plus the
      interior critical points ``theta = phi (+/- pi, + 2pi)`` that fall in
      ``[0, dogleg]``, ``phi = atan2(B, A)``. Exact, <=6 evaluations/arc.
    - **Azimuth** ``azi = atan2(t_E, t_N)`` is **strictly monotonic** along any
      circular arc: ``d(azi)/dtheta`` numerator ``= vec_a_N u_E - vec_a_E u_N``
      is constant (the ``cos^2 + sin^2`` cross-terms cancel identically). So its
      extrema are the two ENDPOINTS, swept in direction ``sign(K)``; the total
      signed swing can exceed 2*pi (arc covers all azimuths).

    Parameters
    ----------
    vec_a, vec_b : (n, 3) array — unit start/end tangents (NEV).
    dogleg : (n,) array — subtended (dogleg) angle of each arc, radians.
    vertical_eps : float — arcs whose minimum inclination is below this (radians)
        pass through vertical, where azimuth is singular; ``passes_vertical`` is
        flagged and the azimuth span should be treated as full-wrap by callers.

    Notes
    -----
    At ``dogleg`` exactly ``pi`` (antiparallel tangents, ``vec_b = -vec_a``) the
    arc plane -- hence ``u`` -- is not recoverable from ``vec_a``/``vec_b`` alone;
    such arcs are treated as degenerate (constant inc/azi, zero swing). This is a
    measure-zero case for real min-curvature/CLC arcs; near-``pi`` is exact.

    Returns
    -------
    dict with (n,)-arrays: ``inc_min``, ``inc_max`` (radians); ``azi_start``,
    ``azi_end`` (radians, in (-pi, pi]); ``azi_swing`` (signed total azimuth
    change, radians; ``abs >= 2*pi`` => all azimuths covered);
    ``passes_vertical`` (bool).
    """
    vec_a = np.atleast_2d(np.asarray(vec_a, dtype=float))
    vec_b = np.atleast_2d(np.asarray(vec_b, dtype=float))
    dogleg = np.atleast_1d(np.asarray(dogleg, dtype=float))

    # In-plane unit perpendicular u such that t(theta)=vec_a*cos+u*sin reaches
    # vec_b at theta=dogleg: u = (vec_b - cos(dogleg) vec_a) / sin(dogleg). The
    # SIGN of sin(dogleg) is what orients the sweep for REFLEX arcs (dogleg > pi,
    # as the CLC solver produces) -- a norm-only recovery would flip it.
    dot = np.einsum('ij,ij->i', vec_a, vec_b)          # = cos(dogleg)
    sin_dl = np.sin(dogleg)
    w = vec_b - dot[:, None] * vec_a
    nw = np.linalg.norm(w, axis=1)
    # degenerate when sin(dogleg)~0: dogleg ~ 0 (straight) or ~ pi (antiparallel
    # tangents, u unrecoverable from vec_a/vec_b alone).
    straight = (np.abs(dogleg) < 1e-9) | (nw < 1e-12)
    safe_nw = np.where(nw < 1e-12, 1.0, nw)
    u = np.sign(np.where(sin_dl == 0.0, 1.0, sin_dl))[:, None] * (w / safe_nw[:, None])

    # --- inclination extrema (critical points of t_V) ---
    A, B = vec_a[:, 2], u[:, 2]
    phi = np.arctan2(B, A)
    cand = np.stack([
        np.zeros_like(dogleg), dogleg,
        phi, phi + np.pi, phi - np.pi, phi + 2.0 * np.pi
    ], axis=1)
    in_range = (cand >= 0.0) & (cand <= dogleg[:, None])
    tV = A[:, None] * np.cos(cand) + B[:, None] * np.sin(cand)
    tV = np.where(in_range, tV, np.nan)
    tV_min = np.nanmin(tV, axis=1)
    tV_max = np.nanmax(tV, axis=1)
    inc_min = np.arccos(np.clip(tV_max, -1.0, 1.0))
    inc_max = np.arccos(np.clip(tV_min, -1.0, 1.0))
    inc_a = np.arccos(np.clip(vec_a[:, 2], -1.0, 1.0))
    inc_min = np.where(straight, inc_a, inc_min)
    inc_max = np.where(straight, inc_a, inc_max)

    # --- azimuth: monotonic; endpoints + signed swing ---
    azi_start = np.arctan2(vec_a[:, 1], vec_a[:, 0])
    azi_end = np.arctan2(vec_b[:, 1], vec_b[:, 0])
    K = vec_a[:, 0] * u[:, 1] - vec_a[:, 1] * u[:, 0]  # t_N u_E - t_E u_N, constant

    # Signed swing = continuous (unwrapped) atan2(t_E, t_N) over [0, dogleg].
    # atan2 wraps by 2*pi each time the projection crosses the -N axis
    # (t_E = 0 while t_N < 0). Count those crossings analytically: zeros of
    # t_E(theta) = vec_a_E cos + u_E sin are theta0 + k*pi, theta0 = atan2(-vec_a_E, u_E).
    cE, sE = vec_a[:, 1], u[:, 1]
    cN, sN = vec_a[:, 0], u[:, 0]
    theta0 = np.arctan2(-cE, sE)
    n = len(dogleg)
    wraps = np.zeros(n)
    kmax = int(np.ceil(np.nanmax(dogleg) / np.pi)) + 2 if n else 0
    for k in range(-1, kmax + 1):
        th = theta0 + k * np.pi
        hit = (th > 1e-12) & (th < dogleg - 1e-12)
        tN_here = cN * np.cos(th) + sN * np.sin(th)
        cross = hit & (tN_here < 0.0)
        wraps += np.where(cross, np.sign(K), 0.0)
    azi_swing = (azi_end - azi_start) + 2.0 * np.pi * wraps
    # snap toward the monotonic direction when the raw endpoint diff disagrees
    disagree = (np.abs(azi_swing) > 1e-9) & (np.sign(azi_swing) != np.sign(K)) \
        & (np.abs(K) > 1e-12)
    azi_swing = np.where(disagree, azi_swing + np.sign(K) * 2.0 * np.pi, azi_swing)
    azi_swing = np.where(straight, 0.0, azi_swing)

    passes_vertical = inc_min < vertical_eps

    return {
        'inc_min': inc_min, 'inc_max': inc_max,
        'azi_start': azi_start, 'azi_end': azi_end,
        'azi_swing': azi_swing, 'passes_vertical': passes_vertical,
    }


def get_angles(
    vec: Annotated[NDArray, Literal["N", 3]], nev: bool = False
):
    '''
    Determines the inclination and azimuth from a vector.

    Parameters
    ----------
    vec: (n,3) array of floats
    nev: boolean (default: False)
        Indicates if the vector is in (x,y,z) or (n,e,v) coordinates.

    Returns
    -------
    [inc, azi]: (n,2) array of floats
        A numpy array of incs and axis in radians

    '''
    # make sure it's a unit vector
    vec = vec / np.linalg.norm(vec, axis=-1).reshape(-1, 1)
    vec = vec.reshape(-1, 3)

    # if it's nev then need to do the shuffle
    if nev:
        y, x, z = vec.T
        vec = np.column_stack([x, y, z])

    return _get_angles(vec)


def survey_from_positions(nev, tie_vec, deg=True, mds=None):
    """Reconstruct a minimum-curvature survey (md, inc, azi) from NEV positions.

    Analytical inverse of the minimum-curvature method with a fixed tie-in
    tangent. On each leg the minimum-curvature chord bisects the two station
    tangents, so the end tangent is the start tangent *reflected about the
    chord direction*::

        t_{i+1} = 2 (c . t_i) c - t_i

    with the tie-in tangent seeding the march; measured depth accumulates as the
    arc length ``L * alpha / sin(alpha)`` (``alpha`` = angle between the leg's
    start tangent and its chord ``c``). Closed form, O(n), no iteration.

    Exact (to machine precision) when the position path is representable as a
    minimum-curvature arc chain -- the case for a *fused* best-estimate
    trajectory (its stations are a small perturbation of two arc chains). A path
    that bends more than a single arc per leg can support needs the
    curve-hold-curve / added-node refinement (:mod:`welleng.connector`); this is
    the closed-form first solution, not the only one.

    Parameters
    ----------
    nev : (n, 3) array_like
        Station positions in (north, east, vertical), metres.
    tie_vec : (3,) array_like
        Tie-in unit tangent at the first station, in NEV. Fixed -- it pins the
        otherwise free march.
    deg : bool, default True
        Return inc/azi in degrees, else radians.
    mds : (n,) array_like, optional
        Return these measured depths instead of the reconstructed arc-length
        MD. The reconstructed MD is self-consistent with the returned inc/azi;
        supplying the input surveys' MDs instead carries a small residual
        (typically sub-mm per 1000 m) where the fused path length differs from
        the originals.

    Returns
    -------
    md, inc, azi : (n,) ndarrays
    """
    nev = np.asarray(nev, dtype=float).reshape(-1, 3)
    n = len(nev)
    t = np.zeros((n, 3))
    t[0] = np.asarray(tie_vec, dtype=float).reshape(3)
    t[0] /= np.linalg.norm(t[0])
    md = np.zeros(n)
    for i in range(n - 1):
        d = nev[i + 1] - nev[i]
        length = np.linalg.norm(d)
        if length < 1e-9:                       # coincident stations
            t[i + 1] = t[i]
            continue
        c = d / length
        refl = 2.0 * (c @ t[i]) * c - t[i]      # reflect t_i about the chord
        norm = np.linalg.norm(refl)
        t[i + 1] = refl / norm if norm > 1e-12 else t[i]
        alpha = np.arccos(np.clip(c @ t[i], -1.0, 1.0))
        md[i + 1] = md[i] + (
            length * alpha / np.sin(alpha) if alpha > 1e-9 else length
        )
    ang = get_angles(t, nev=True)
    inc, azi = ang[:, 0], ang[:, 1]
    if deg:
        inc, azi = np.degrees(inc), np.degrees(azi)
    if mds is not None:
        md = np.asarray(mds, dtype=float).reshape(-1)
    return md, inc, azi


def _get_transform(inc, azi):
    ci, si = np.cos(inc), np.sin(inc)
    ca, sa = np.cos(azi), np.sin(azi)
    z = np.zeros_like(inc)
    trans = np.stack([
        np.stack([ci * ca,  ci * sa, -si], axis=-1),
        np.stack([-sa,       ca,      z  ], axis=-1),
        np.stack([si * ca,  si * sa,  ci ], axis=-1),
    ], axis=1)

    return trans


def get_transform(
    survey
):
    """
    Determine the transform for transforming between NEV and HLA coordinate
    systems.

    Parameters
    ----------
    survey: (n,3) array of floats
        The [md, inc, azi] survey listing array.

    Returns
    -------
    transform: (n,3,3) array of floats
    """
    survey = survey.reshape(-1, 3)
    inc = np.array(survey[:, 1])
    azi = np.array(survey[:, 2])

    return _get_transform(inc, azi)


def NEV_to_HLA(
    survey: Annotated[NDArray, Literal["N", 3]],
    NEV: Union[
        Annotated[NDArray, Literal["N", 3]],
        Annotated[NDArray, Literal["N", 3, 3]]
    ],
    cov: bool = True
) -> Union[
        Annotated[NDArray, Literal['N, 3']],
        Annotated[NDArray, Literal['N, 3, 3']]
]:
    """
    Transform from NEV to HLA coordinate system.

    Parameters
    ----------
    survey: (n,3) array of floats
        The [md, inc, azi] survey listing array.
    NEV: (n,3) or (n,3,3) array of floats
        The NEV coordinates or covariance matrices.
    cov: boolean
        If cov is True then a (n,3,3) array of covariance matrices
        is expected, else a (n,3) array of coordinates.

    Returns
    -------
    HLAs: NDArray
        Either a transformed (n,3) array of HLA coordinates or an
        (n,3,3) array of HLA covariance matrices.
    """

    trans = get_transform(survey)

    if cov:
        # HLA_cov = trans @ NEV_cov @ trans.T  (batched over n stations)
        return trans @ NEV @ trans.swapaxes(-1, -2)

    else:
        NEV = NEV.reshape(-1, 3)
        return np.einsum('...k,...jk', NEV, trans)


def HLA_to_NEV(survey, HLA, cov=True, trans=None):
    if trans is None:
        trans = get_transform(survey)

    if cov:
        # NEV_cov = trans.T @ HLA_cov @ trans  (batched over n stations)
        return trans.swapaxes(-1, -2) @ HLA @ trans

    else:
        shape = HLA.shape
        return (HLA.reshape(shape[0], -1, 3) @ trans).reshape(shape)


def get_sigmas(cov, long=False):
    """
    Extracts the sigma values of a covariance matrix along the principle axii.

    Parameters
    ----------
    cov: (n,3,3) array of floats

    Returns
    -------
    arr: (n,3) array of floats
    """

    assert cov.shape[-2:] == (3, 3), "Cov is the wrong shape"

    cov = cov.reshape(-1, 3, 3)

    aa, ab, ac = cov[:, :, 0].T
    ab, bb, bc = cov[:, :, 1].T
    ac, bc, cc = cov[:, :, 2].T

    if long:
        return (aa, bb, cc, ab, ac, bc)
    else:
        return (np.sqrt(aa), np.sqrt(bb), np.sqrt(cc))


def get_unit_vec(vec):
    vec = vec / np.linalg.norm(vec)

    return vec


def linear_convert(data, factor):
    flag = False
    if not isinstance(data, list):
        flag = True
        data = [data]
    converted = [d * factor if d is not None else None for d in data]
    if flag:
        return converted[0]
    else:
        return converted


def make_cov(a, b, c, long=False):
    # a, b, c = np.sqrt(np.array([a, b, c]))
    if long:
        cov = np.array([
            [a * a, a * b, a * c],
            [a * b, b * b, b * c],
            [a * c, b * c, c * c]
        ])

    else:
        cov = np.array([
            [a * a, np.zeros_like(a), np.zeros_like(a)],
            [np.zeros_like(a), b * b, np.zeros_like(a)],
            [np.zeros_like(a), np.zeros_like(a), c * c]
        ])

    return cov.T


def make_long_cov(arr):
    """
    Build a (n, 3, 3) covariance matrix from the 6 unique upper-triangle
    elements per station.

    Parameters
    ----------
    arr: (n, 6) array — columns [aa, ab, ac, bb, bc, cc]

    Returns
    -------
    cov: (n, 3, 3) array
    """
    aa, ab, ac, bb, bc, cc = np.array(arr).T
    return np.stack([
        np.stack([aa, ab, ac], axis=-1),
        np.stack([ab, bb, bc], axis=-1),
        np.stack([ac, bc, cc], axis=-1),
    ], axis=1)


def dls_from_radius(radius):
    """
    Returns the dls in degrees from a radius.
    """
    if isinstance(radius, np.ndarray):
        circumference = np.full_like(radius, np.inf)
        circumference = np.where(
            radius != 0,
            2 * np.pi * radius,
            circumference
        )
    else:
        if radius == 0:
            return np.inf
        circumference = 2 * np.pi * radius
    dls = 360 / circumference * 30

    return dls


def radius_from_dls(dls):
    """
    Returns the radius in meters from a DLS in deg/30m.
    """
    if isinstance(dls, np.ndarray):
        circumference = np.full_like(dls, np.inf)
        circumference = np.where(
            dls != 0,
            (30 / dls) * 360,
            circumference
        )
    else:
        if dls == 0:
            return np.inf
        circumference = (30 / dls) * 360
    radius = circumference / (2 * np.pi)

    return radius


def cov_from_vec(arr):
    """
    Returns a (n, 3, 3) covariance matrix from an (n, 3) array via outer product.

    Parameters
    ----------
    arr: (n, 3) array
        Array of vector components.

    Returns
    -------
    (n, 3, 3) array
    """
    arr = np.array(arr)
    return arr[:, :, None] * arr[:, None, :]


def errors_from_cov(cov, data=False):
    """
    Parameters
    ----------
    cov: (n, 3, 3) array
        The error covariance matrices.
    data: bool (default: False)
        If True returns a dictionary, else returns a list.
    """
    nn, ne, nv, _, ee, ev, _, _, vv = (
        cov.reshape(-1, 9).T
    )

    if data:
        return {
            i: {
                'nn': _nn, 'ne': _ne, 'nv': _nv,
                'ee': _ee, 'ev': _ev, 'vv': _vv
            }
            for i, (_nn, _ne, _nv, _ee, _ev, _vv)
            in enumerate(zip(nn, ne, nv, ee, ev, vv))
        }

    return np.array([nn, ne, nv, ee, ev, vv]).T


def _get_arc_pos_and_vec(dogleg, radius):
    pos = np.array([
        np.cos(dogleg),
        0.,
        np.sin(dogleg)
    ]) * radius
    pos[0] = radius - pos[0]

    vec = np.array([
        np.sin(dogleg),
        0.,
        np.cos(dogleg)
    ])
    return (pos, vec)


class Arc:
    def __init__(self, dogleg, radius):
        """
        Generates a generic arc that can be transformed with a specific pos
        and vec via a transform method. The arc is initialized at a local
        origin and kicks off down and to the north (assuming an NEV coordinate
        system).

        Parameters
        ----------
        dogleg: float
            The sweep angle of the arc in radians.
        radius: float
            The radius of the arc in meters.

        Returns
        -------
        arc: Arc object
        """
        self.dogleg = dogleg
        self.radius = radius
        self.delta_md = dogleg * radius

        self.pos, self.vec = _get_arc_pos_and_vec(dogleg, radius)


    def transform(self, toolface, pos=None, vec=None, target=False):
        """
        Transforms an Arc to a position and orientation.

        Parameters
        ----------
        pos: (,3) array
        The desired position to transform the arc.
        vec: (,3) array
            The orientation unit vector to transform the arc.
        target: bool
            If true, returned arc vector is reversed.

        Returns
        -------
        tuple (pos_new, vec_new)
        pos_new: (,3) array
            The position at the end of the arc post transform.
        vec_new: (,3) array
            The unit vector at the end of the arc post transform.
        """
        if vec is None:
            vec = np.array([0., 0., 1.])
        if target:
            vec *= -1
        inc, azi = get_angles(vec, nev=True).reshape(2)
        angles = [
            toolface,
            inc,
            azi
        ]
        r = R.from_euler('zyz', angles, degrees=False)

        pos_new, vec_new = r.apply(np.vstack((self.pos, self.vec)))

        # make sure vec_new is a unit vector:
        vec_new = get_unit_vec(vec_new)

        if pos is not None:
            pos_new += pos
        if target:
            vec_new *= -1

        return (pos_new, vec_new)


def get_arc(
    dogleg, radius, toolface, pos=None, vec=None, target=False
) -> tuple:
    """Creates an Arc instance and transforms it to the desired position
    and orientation.

    Parameters
    ----------
    dogleg: float
        The swept angle of the arc (arc angle) in radians.
    radius: float
        The radius of the arc (in meters).
    toolface: float
        The toolface angle in radians (relative to the high side) to rotate the
        arc at the desired position and orientation.
    pos: (,3) array
        The desired position to transform the arc.
    vec: (,3) array
        The orientation unit vector to transform the arc.
    target: bool
        If true, returned arc vector is reversed.

    Returns
    -------
    tuple of (pos_new, vec_new, arc.delta_md)
    pos_new: (,3) array
        The position at the end of the arc post transform.
    vec_new: (,3) array
        The unit vector at the end of the arc post transform.
    arc.delta_md: int
        The arc length of the arc.
    """
    arc = Arc(dogleg, radius)
    pos_new, vec_new = arc.transform(toolface, pos, vec, target)

    return (pos_new, vec_new, arc.delta_md)


def annular_volume(od: float, id: float = None, length: float = None):
    """
    Calculate an annular volume.

    If no ``id`` is provided then circular volume is calculated. If no
    ``length`` is provided, then the unit volume is calculated (i.e. the
    area).

    Units are assumed consistent across input parameters, i.e. the
    calculation is dimensionless.

    Parameters
    ----------
    od: float
        The outer diameter.
    id: float | None, optional
        The inner diameter, default is 0.
    length : float | None, optional
        The length of the annulus.

    Returns
    -------
    annular_volume: float
        The (unit) volume of the annulus or cylinder.

    Examples
    --------
    In the following example we calculate annular volume along a 1,000 meter
    section length of 9 5/8" casing inside 12 1/4" hole.

    >>> from welleng.utils import annular_volume
    >>> from welleng.units import ureg
    >>> av = annular_volume(
    ...     od=ureg('12.25 inch').to('meters),
    ...     id=ureg(f'{9+5/8} inch').to('meter'),
    ...     length=ureg('1000 meter')
    ... )
    >>> print(av)
    29.096093526301622 meter ** 3
    """
    length = 1 if length is None else length
    id = 0 if id is None else id
    annular_unit_volume = (np.pi * (od**2 - id**2)) / 4
    annular_volume = annular_unit_volume * length

    return annular_volume


def _decimal2dms(decimal: tuple, ndigits: int = None) -> tuple:
    try:
        _decimal, direction = decimal
    except (TypeError, ValueError):
        _decimal = decimal[0] if isinstance(decimal, np.ndarray) else decimal
        direction = None
    _decimal = float(_decimal)
    minutes, seconds = divmod(abs(_decimal) * 3600, 60)
    _, minutes = divmod(minutes, 60)

    return np.array([
        int(_decimal),
        int(minutes),
        seconds if ndigits is None else round(seconds, ndigits)
    ]) if direction is None else np.array([
        int(_decimal),
        int(minutes),
        seconds if ndigits is None else round(seconds, ndigits),
        direction
    ], dtype=object)


def decimal2dms(decimal: tuple | NDArray, ndigits: int = None) -> tuple | NDArray:
    """Converts a decimal lat, lon to degrees, minutes and seconds.

    Parameters
    ----------
    decimal : tuple | arraylike
        A tuple of (lat, direction) or (lon, direction) or arraylike of
        ((lat, direction), (lon, direction)) coordinates.
    ndigits: int (default is None)
        If specified, rounds the seconds decimal to the desired number of
        digits.

    Returns
    -------
    dms: arraylike
        An array of (degrees, minutes, seconds, direction).

    Examples
    --------
    If you want to convert the lat/lon coordinates for Den Haag from decimals
    to degrees, minutes and seconds:

    >>> LAT, LON = [(52.078663, 'N'), (4.288788, 'E')]
    >>> dms = decimal2dms((LAT, LON), ndigits=6)
    >>> print(dms)
    [[52 4 43.1868 'N']
     [4 17 19.6368 'E']]
    """
    flag = False
    _decimal = np.array(decimal)
    if _decimal.dtype == np.float64:
        _decimal = _decimal.reshape((-1, 1))
        flag = True
    try:
        dms = np.apply_along_axis(_decimal2dms, -1, _decimal, ndigits)
    except np.exceptions.AxisError:
        dms = _decimal2dms(_decimal, ndigits)

    if dms.shape == (4,):
        return tuple(dms)
    else:
        return dms.reshape((-1, 3)) if flag else dms


def _dms2decimal(dms: NDArray, ndigits: int = None) -> NDArray:
    try:
        degrees, minutes, seconds, direction = dms
    except ValueError:
        degrees, minutes, seconds = dms
        direction = None

    decimal = abs(degrees) + minutes / 60 + seconds / 3600

    return np.array([
        np.copysign(
            decimal if ndigits is None else round(decimal, ndigits),
            degrees
        )
    ]) if direction is None else np.array([
        np.copysign(
            decimal if ndigits is None else round(decimal, ndigits),
            degrees
        ),
        direction
    ], dtype=object)


def dms2decimal(dms: tuple | NDArray, ndigits: int = None) -> NDArray:
    """Converts a degrees, minutes and seconds lat, lon to decimals.

    Parameters
    ----------
    dms : tuple | arraylike
        A tuple or arraylike of (degrees, minutes, seconds, direction) lat
        and/or lon or arraylike of lat, lon coordinates.
    ndigits: int (default is None)
        If specified, rounds the decimal to the desired number of digits.

    Returns
    -------
    degrees: arraylike
        A tuple or array of lats and/or longs in decimals.

    Examples
    --------
    If you want to convert the lat/lon coordinates for Den Haag from degrees,
    minutes and seconds to decimals:

    >>> LAT, LON = (52, 4, 43.1868, 'N'), (4, 17, 19.6368, 'E')
    >>> decimal = dms2decimal((LAT, LON), ndigits=6)
    >>> print(decimal)
    [[52.078663 'N']
     [4.288788 'E']]
    """
    result = np.apply_along_axis(
        _dms2decimal, -1, np.array(dms, dtype=object), ndigits
    )

    if result.shape == ():
        return float(result)
    elif result.shape == (1,):
        return float(result[0])
    elif result.shape[-1] == 1:
        return result.reshape(-1)
    else:
        return result


def pprint_dms(dms, symbols: bool = True, return_data: bool = False):
    """Pretty prints a (decimal, minutes, seconds) tuple or list.

    Parameters
    ----------
    dms: tuple | list
        An x or y or northing or easting (degree, minute, second).
    symbols: bool (default: True)
        Whether to print symbols for (deg, min, sec).
    return_data: bool (default: False)
        If True then will return the string rather than print it.
    """
    if symbols:
        try:
            deg, min, sec = dms
            text = f"{deg}\N{DEGREE SIGN}, {min}', {sec}\""
        except ValueError:
            deg, min, sec, _ = dms
            text = f"{deg}\N{DEGREE SIGN}, {min}', {sec}\" {_}"

    else:
        try:
            deg, min, sec = dms
            text = f"{deg} deg, {min} min, {sec} sec"
        except ValueError:
            deg, min, sec, _ = dms
            text = f"{deg} deg, {min} min, {sec} sec {_}"

    if return_data:
        return text
    else:
        print(text)


def dms_from_string(text):
    """Extracts the values from a string dms x or y or northing or easting.
    """
    pattern = re.compile(r'(\d+)\s*(?:°|deg)?,\s*(\d+)\s*(?:\'|min)?,\s*(\d+(?:\.\d+)?)\s*(sec)?\s*.*?(\S+)?$', re.IGNORECASE)
    matches = pattern.findall(text)

    if matches:
        deg, min, sec_str = matches[0][:3]
        sec = float(sec_str)
        final_data = matches[0][-1] if matches[0][-1] else None

        if final_data:
            return (int(deg), int(min), sec, final_data)
        else:
            return (int(deg), int(min), sec)

    else:
        return


def make_clc_path(
    toolface1, dogleg1, distance, toolface2, dogleg2,
    pos0=None, vec0=None, radius=1.0
):
    """Generate a curve-hold-curve (CLC) path from arc parameters.

    Builds the path in three steps: first arc, straight hold, second arc.
    Useful for constructing known-geometry test cases and for quickly
    prototyping CLC trajectories.

    Parameters
    ----------
    toolface1: float
        Toolface angle for the first curve in radians.
    dogleg1: float
        Sweep angle (dogleg) for the first curve in radians.
    distance: float
        Length of the straight hold section (same units as radius).
    toolface2: float
        Toolface angle for the second curve in radians.
    dogleg2: float
        Sweep angle (dogleg) for the second curve in radians.
    pos0: (3,) array-like, optional
        Start position [N, E, V]. Defaults to [0, 0, 0].
    vec0: (3,) array-like, optional
        Start direction unit vector. Defaults to [0, 0, 1] (pointing down).
    radius: float, optional
        Arc radius for both curves. Defaults to 1.0.

    Returns
    -------
    dict with keys:
        pos1, vec1  – end of first arc
        dist_curve1 – arc length of first curve
        pos2, vec2  – end of hold section / start of second arc
        pos3, vec3  – end of second arc
        dist_curve2 – arc length of second curve
    """
    pos0 = np.array([0., 0., 0.]) if pos0 is None else np.asarray(pos0, dtype=float)
    vec0 = np.array([0., 0., 1.]) if vec0 is None else np.asarray(vec0, dtype=float)

    pos1, vec1, dist_curve1 = get_arc(dogleg1, radius, toolface1, pos=pos0, vec=vec0)
    pos2 = pos1 + vec1 * distance
    vec2 = vec1.copy()
    pos3, vec3, dist_curve2 = get_arc(dogleg2, radius, toolface2, pos=pos2, vec=vec2)

    return dict(
        pos1=pos1, vec1=vec1, dist_curve1=dist_curve1,
        pos2=pos2, vec2=vec2,
        pos3=pos3, vec3=vec3, dist_curve2=dist_curve2,
    )


def get_toolface(pos1: NDArray, vec1: NDArray, pos2: NDArray) -> NDArray:
    """Returns the toolface(s) of offset position(s) relative to reference
    positions and vectors.  Accepts either single (3,) arrays or batches of
    (n, 3) arrays; all three arguments must have the same leading dimension.

    Parameters
    ----------
    pos1: ndarray, shape (3,) or (n, 3)
        The reference NEV coordinate(s), e.g. current location.
    vec1: ndarray, shape (3,) or (n, 3)
        The reference NEV unit vector(s), e.g. current direction.
    pos2: ndarray, shape (3,) or (n, 3)
        The offset NEV coordinate(s), e.g. a target position.

    Returns
    -------
    toolface: float or ndarray
        The toolface(s) in radians [0, 2π) to pos2 from pos1 along vec1.
        Returns a scalar float when single (3,) inputs are given.
    """
    pos1 = np.atleast_2d(pos1)
    vec1 = np.atleast_2d(vec1)
    pos2 = np.atleast_2d(pos2)

    angles = np.flip(get_angles(vec1, nev=True), axis=1)
    r = R.from_euler('zy', angles * -1, degrees=False)
    pos = r.apply(pos2 - pos1)
    result = np.arctan2(*np.flip(pos[:, :-1], axis=1).T) % (2 * np.pi)
    return float(result[0]) if result.size == 1 else result


def get_toolface_fast(pos1: NDArray, vec1: NDArray, pos2: NDArray) -> float:
    """Returns the toolface of a single offset position using a direct
    closed-form expression — approximately 12× faster than ``get_toolface``
    for scalar inputs.

    Suitable when pos1, vec1 and pos2 are all individual (3,) arrays.
    For batch use, prefer the vectorised ``get_toolface``.

    Parameters
    ----------
    pos1: array-like, shape (3,)
        The reference NEV coordinate, e.g. current location.
    vec1: array-like, shape (3,)
        The reference NEV unit vector, e.g. current direction.
    pos2: array-like, shape (3,)
        The offset NEV coordinate, e.g. a target position.

    Returns
    -------
    toolface: float
        The toolface in radians [0, 2π) to pos2 from pos1 along vec1.
    """
    n1, e1, v1 = pos1
    n2, e2, v2 = pos2
    vn1, ve1, vv1 = vec1

    azimuth_vec1 = np.arctan2(ve1, vn1) % (2 * np.pi)
    cos_azi = np.cos(azimuth_vec1)
    sin_azi = np.sin(azimuth_vec1)

    numerator = (e2 - e1) * cos_azi + (n1 - n2) * sin_azi
    horiz_mag = np.sqrt(ve1 ** 2 + vn1 ** 2)
    denominator = (
        vv1 * (e2 - e1) * sin_azi
        + vv1 * (n2 - n1) * cos_azi
        + (v1 - v2) * horiz_mag
    ) / np.sqrt(ve1 ** 2 + vn1 ** 2 + vv1 ** 2)

    return np.arctan2(numerator, denominator) % (2 * np.pi)




def best_fit_rotation_2d(a, b, weights=None):
    """Closed-form 2D rotation angle that best aligns vectors ``a`` onto ``b``.

    Solves the 2D orthogonal-Procrustes (Wahba) problem: the rotation angle
    ``theta`` minimising ``sum_i w_i |R(theta) a_i - b_i|^2`` has the exact
    closed form

        theta = atan2( sum_i w_i (a_i x b_i),  sum_i w_i (a_i . b_i) )

    where ``a_i x b_i = a_ix b_iy - a_iy b_ix`` (the 2D cross product) and
    ``a_i . b_i`` is the dot product. No iteration; a single ``atan2``.

    With ``weights=None`` the vectors enter at their own magnitude, so longer
    vectors dominate the fit -- useful when ``a``/``b`` are displacement steps
    and the longer steps are the more reliable direction estimates. Pass unit
    vectors (or explicit ``weights``) to weight every pair equally.

    In directional surveying this recovers the single rotation between two sets
    of directions -- e.g. the grid-convergence / reference offset between a
    survey's stated azimuths and the direction of its position steps, or between
    two surveys of the same well.

    Parameters
    ----------
    a, b : (n, 2) array_like
        Two sets of 2D vectors, paired by row. ``b_i`` is ``a_i`` rotated
        (plus noise). Any consistent axis convention works (e.g. columns
        ``(north, east)``); the result is the rotation from ``a`` to ``b``.
    weights : (n,) array_like, optional
        Per-pair weights. Default ``None`` weights by vector magnitude.

    Returns
    -------
    float
        The best-fit rotation angle in radians, in ``(-pi, pi]``.

    Notes
    -----
    Derivation: minimising ``sum w_i |R a_i - b_i|^2`` over rotations is
    maximising ``sum w_i b_i^T R a_i``. Writing ``R(theta)`` and differentiating
    gives ``tan(theta) = sum w_i (a_i x b_i) / sum w_i (a_i . b_i)``, resolved to
    the correct quadrant by ``atan2``. This is the 2D case of Kabsch/Wahba.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("a and b must be matching (n, 2) arrays")
    w = 1.0 if weights is None else np.asarray(weights, dtype=float)[:, None]
    cross = np.sum(w * (a[:, 0:1] * b[:, 1:2] - a[:, 1:2] * b[:, 0:1]))
    dot = np.sum(w * (a[:, 0:1] * b[:, 0:1] + a[:, 1:2] * b[:, 1:2]))
    return float(np.arctan2(cross, dot))
