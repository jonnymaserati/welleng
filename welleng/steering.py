"""
Map between the *design* (minimum-curvature) trajectory and the *drilled /
adjusted* (maximum-curvature) trajectory — the "error domain" of the
survey-interval directional bias.

Motivation
----------
A directional well is planned as a smooth minimum-curvature path (the **design**
plan), but the well that is actually drilled carries the survey-interval
directional bias: between stations the bit follows a higher-curvature bow, so the
real path deflects from the plan (principally a systematic **shallowing** of TVD).
``Survey.maximum_curvature`` (see ``welleng.survey``) is the exact forward
construction of that bowed path — the **adjusted** plan. The paper *"The
geometric survey-interval error"* develops the design-phase correction that spends
this bias at the drawing board.

This module supplies the reusable, bidirectional **domain map** between the two
trajectories, so that while drilling the adjusted plan you can switch to the
design plan and read where you are on it — and so that a formation top picked on
the drilled path maps to its prognosed depth on the design plan.

The maths
---------
``Survey.maximum_curvature`` **preserves the measured depth at every control
station** (``adjusted.md[::2] == design.md``) and moves only the *position*: at a
matched MD the adjusted path sits at a different (shallower) TVD. The domain map
is therefore a **same-measured-depth comparison**, not a solver:

Let ``D`` be the design plan and ``A`` the adjusted plan, both parameterised by
measured depth ``s``. The **error-domain displacement** is

    e(s) = A.pos(s) - D.pos(s)

zero at the surface and growing down-hole. Decomposed in the target trajectory's
tangent frame ``(t, h, l)`` — along-hole ``t``, high-side ``h`` (direction of
increasing inclination), lateral ``l = t x h`` — it gives the familiar steering
triad (along-track / high-side / lateral offset).

For a **formation top** encountered while drilling at adjusted MD ``s_top`` with
actual TVD ``z = A.tvd(s_top)`` there are two complementary residuals:

* same-MD (how wrong was the plan's depth at this drilled MD):
  ``dz = z - D.tvd(s_top)``
* same-TVD (how much extra MD was drilled to reach the formation):
  ``dmd = s_top - {s : D.tvd(s) = z}``

Both are recovered by :func:`formation_prognosis`.

Scope
-----
Pure geometry — no error model. Works for any pair of surveys sharing a datum:
design vs its ``maximum_curvature`` adjusted plan (exact shared control stations),
or a design plan vs a real drilled survey (shared at the tie-in). The along/high-
side/lateral decomposition is the standard borehole toolface frame and is not
novel; the framed contribution is the design<->error-domain mapping and the
top-residual-as-calibration-signal (see ``docs/dev/MAXCURVE_DOMAIN_MAPPING.md``).
"""
from collections import namedtuple

import numpy as np

__all__ = ["DomainPoint", "TopPrognosis", "project_to", "formation_prognosis"]


DomainPoint = namedtuple(
    "DomainPoint",
    [
        "md", "inc", "azi", "pos_nev", "tvd",
        "offset_nev", "offset_along", "offset_highside", "offset_lateral",
    ],
)

TopPrognosis = namedtuple(
    "TopPrognosis",
    [
        "name", "md_actual", "tvd_actual",
        "md_prognosed", "tvd_prognosed", "tvd_residual", "md_residual",
    ],
)


def _toolface_frame(vec_nev):
    """Return (t, h, l): along-hole, high-side and lateral unit vectors in the
    (n, e, tvd-down) frame for a tangent ``vec_nev``.

    High side is the component of the *upward* vertical perpendicular to the
    tangent (the direction of increasing inclination). For a vertical hole the
    high side is undefined; ``h`` and ``l`` are returned as ``nan`` there.
    """
    t = np.asarray(vec_nev, dtype=float)
    t = t / np.linalg.norm(t)
    up = np.array([0.0, 0.0, -1.0])          # upward vertical (tvd is +down)
    h = up - np.dot(up, t) * t
    n = np.linalg.norm(h)
    if n < 1e-9:                              # (near-)vertical: high side undefined
        nan3 = np.full(3, np.nan)
        return t, nan3, nan3
    h = h / n
    l = np.cross(t, h)
    return t, h, l


def project_to(source, target, md):
    """Map a station at measured depth ``md`` on ``source`` to the corresponding
    station on ``target``, and decompose the offset in ``target``'s tangent frame.

    "Corresponding" is **same measured depth** — the natural correspondence
    because :meth:`~welleng.survey.Survey.maximum_curvature` preserves MD at the
    control stations (see module docstring). Use it to switch a point between the
    design and adjusted ("error domain") trajectories: pass the drilled/adjusted
    survey as ``source`` and the design plan as ``target`` to read where you are
    on the plan, or vice versa.

    Parameters
    ----------
    source : welleng.survey.Survey
        The trajectory the query point lives on.
    target : welleng.survey.Survey
        The trajectory to map onto (its station at the same MD is returned).
    md : float
        Measured depth of the query point (metres or feet, matching the surveys).

    Returns
    -------
    DomainPoint
        ``md``, ``inc``/``azi`` (deg) and ``pos_nev``/``tvd`` of the target
        station, plus ``offset_nev = source.pos(md) - target.pos(md)`` and its
        along-hole / high-side / lateral components in the target tangent frame
        (high-side/lateral are ``nan`` where the target is vertical).

    Examples
    --------
    >>> import numpy as np, welleng as we
    >>> design = we.survey.Survey(
    ...     md=[0, 300, 900, 1800, 2400, 3000.], inc=[0, 0, 40, 40, 80, 80.],
    ...     azi=np.zeros(6), header=we.survey.SurveyHeader(name="d"))
    >>> adjusted = design.maximum_curvature(dls_noise=1.0)
    >>> p = we.steering.project_to(adjusted, design, 3000.)
    >>> bool(p.offset_highside > 0)   # adjusted sits high (shallow) of the plan
    True
    """
    ns = source.interpolate_md(md)
    nt = target.interpolate_md(md)
    if ns is None or nt is None:
        raise ValueError(
            f"md {md} is outside the range of both surveys "
            f"(source {source.md[0]}-{source.md[-1]}, "
            f"target {target.md[0]}-{target.md[-1]})"
        )
    pos_s = np.asarray(ns.pos_nev, dtype=float)
    pos_t = np.asarray(nt.pos_nev, dtype=float)
    offset = pos_s - pos_t

    t, h, l = _toolface_frame(nt.vec_nev)
    return DomainPoint(
        md=float(md),
        inc=float(np.degrees(nt.inc_rad)),
        azi=float(np.degrees(nt.azi_rad)),
        pos_nev=pos_t,
        tvd=float(pos_t[2]),
        offset_nev=offset,
        offset_along=float(np.dot(offset, t)),
        offset_highside=float(np.dot(offset, h)),
        offset_lateral=float(np.dot(offset, l)),
    )


def _md_at_tvd(survey, tvd):
    """Shallowest measured depth at which ``survey`` reaches ``tvd`` (linear in
    TVD between the two bracketing stations). Returns ``None`` if ``tvd`` is not
    reached. For a non-monotone TVD profile (build-then-drop) the shallowest
    crossing is returned; callers wanting a deeper crossing should slice first.
    """
    z = np.asarray(survey.tvd, dtype=float)
    md = np.asarray(survey.md, dtype=float)
    for i in range(len(z) - 1):
        z0, z1 = z[i], z[i + 1]
        lo, hi = min(z0, z1), max(z0, z1)
        if lo <= tvd <= hi:
            if z1 == z0:
                return float(md[i])
            f = (tvd - z0) / (z1 - z0)
            return float(md[i] + f * (md[i + 1] - md[i]))
    return None


def formation_prognosis(design, actual, tops):
    """Map formation tops picked on the ``actual`` (drilled/adjusted) trajectory
    back to their prognosed depths on the ``design`` plan.

    For each top penetrated at an actual measured depth, returns both residuals:
    the **TVD error** at that drilled MD (``tvd_residual``, negative = the well is
    shallower than planned) and the **extra MD** drilled to reach the formation's
    TVD (``md_residual``). See the module docstring for the maths.

    Parameters
    ----------
    design : welleng.survey.Survey
        The planned trajectory carrying the depth prognosis.
    actual : welleng.survey.Survey
        The drilled (or predicted maximum-curvature) trajectory on which the tops
        were picked.
    tops : iterable of (float, str)
        ``(md_actual, name)`` pairs — the measured depth on ``actual`` at which
        each named top was penetrated.

    Returns
    -------
    list of TopPrognosis
        Per top: actual MD/TVD, prognosed MD/TVD, and the two residuals. A top
        whose actual TVD the design never reaches has ``md_prognosed`` and
        ``md_residual`` as ``nan``.

    Examples
    --------
    >>> import numpy as np, welleng as we
    >>> design = we.survey.Survey(
    ...     md=[0, 300, 900, 1800, 2400, 3000.], inc=[0, 0, 40, 40, 80, 80.],
    ...     azi=np.zeros(6), header=we.survey.SurveyHeader(name="d"))
    >>> actual = design.maximum_curvature(dls_noise=1.0)
    >>> res = we.steering.formation_prognosis(design, actual, [(1800., "Top A")])
    >>> res[0].name, bool(res[0].tvd_residual < 0)   # shallower than prognosis
    ('Top A', True)
    """
    out = []
    for md_actual, name in tops:
        na = actual.interpolate_md(md_actual)
        if na is None:
            raise ValueError(
                f"top '{name}' md {md_actual} is outside the actual survey "
                f"({actual.md[0]}-{actual.md[-1]})"
            )
        tvd_actual = float(na.pos_nev[2])

        nd = design.interpolate_md(md_actual)
        tvd_prognosed = float(nd.pos_nev[2]) if nd is not None else float("nan")

        md_prognosed = _md_at_tvd(design, tvd_actual)
        if md_prognosed is None:
            md_prognosed = float("nan")
            md_residual = float("nan")
        else:
            md_residual = float(md_actual) - md_prognosed

        out.append(TopPrognosis(
            name=name,
            md_actual=float(md_actual),
            tvd_actual=tvd_actual,
            md_prognosed=md_prognosed,
            tvd_prognosed=tvd_prognosed,
            tvd_residual=tvd_actual - tvd_prognosed,
            md_residual=md_residual,
        ))
    return out
