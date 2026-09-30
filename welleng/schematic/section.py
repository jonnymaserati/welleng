"""Section-view generator.

The well is drawn along its real trajectory in the vertical-section plane
(VS @ azimuth, TVD down), to scale in both directions. Diameters are exaggerated
by one factor for the whole drawing, applied along the normal to the path.

Drawing rules
-------------
- **Exaggeration** is automatic or set by the caller. Automatic: the widest drawn
  feature (open hole, or a string where it is wider) spans ``frac`` of the plot
  extent, capped at ``0.8 * E_max``.
- **E_max** is the largest exaggeration at which no offset edge folds back on
  itself. For a minimum-curvature leg of radius ``R = dMD / dogleg`` projected on
  the view plane, the tightest radius as drawn is ``R (n . p)^2`` (the projection
  of a circle is an ellipse, tightest radius ``b^2 / a``), where ``n`` is the
  normal to the leg's plane and ``p`` the normal to the view plane. E_max is the
  minimum over legs of that radius divided by the widest drawn radius on the leg.
  A caller's value above E_max is drawn as given and reported in
  ``Drawing.section_info``; nothing is marked on the drawing. A leg whose plane is
  seen edge-on (``n . p`` near 0) is reported as needing a plan view.
- **Open hole** is a light trace at the bit radius, continuous: a step along the
  normal where the bit size changes, and from a string's wall to the hole where a
  hole section starts below a string that has no hole around it.
- **A string with no hole section over its length** (a driven conductor) has no
  hole trace, no cement and no shoe triangle.
- **Cement** fills from the string's OD to the next wall outward (the next string's
  ID, or the hole) over each recorded interval; the top of each interval is a line
  across the annulus along the normal.
- **Annulus fill** is off by default; given a colour, it fills the uncemented
  annular space only.
- **Shoe triangles**: base along the normal at the shoe, outside the wall, 0.8 of
  the annular gap there; height 1.6 times the base.
- **Sea and mudline** are drawn outside the outermost string only, from the well's
  datum elevation and water depth when both are recorded.
- **Callouts** (string, shoe MD/TVD, inclination, weight, grade, top of cement)
  are placed off the drawing and inside the plot frame, clear of each other, with
  a leader to the shoe.

No renderer imports -- emits :mod:`welleng.schematic.drawing` entities only.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .depth import DepthResolver
from .drawing import (
    Drawing,
    Hatch,
    Line,
    Polygon,
    Polyline,
    Rect,
    Style,
    Text,
    ViewTransform,
)
from .models import WellSchematic

_IN2M = 0.0254

L_FRAME = "FRAME"
L_SEA = "SEA"
L_HOLE = "HOLE"
L_FLUID = "FLUID"
L_PATH = "PATH"
L_CASING = "CASING"
L_CEMENT = "CEMENT"
L_PLUG = "PLUG"
L_COMPLETION = "COMPLETION"
L_SHOE = "SHOE"
L_ANNOTATION = "ANNOTATION"

_FRAME = Style(color="#9a9a9a", lineweight=0.2)
_SEA = Style(color="#d7ebf7", lineweight=0.0, fill="#d7ebf7")
_MUDLINE = Style(color="#8a7a5a", lineweight=0.35)
_HOLE = Style(color="#b4b4b4", lineweight=0.3)
_PATH = Style(color="#999999", lineweight=0.2, linestyle="dashed")
_STEEL = Style(color="#222222", lineweight=0.2, fill="#555555")
_CEMENT = Style(color="#c9c2ae", lineweight=0.0, fill="#c9c2ae")
_TOC = Style(color="#6b6454", lineweight=0.35)
_PLUG = Style(color="#6b5d2f", lineweight=0.2, fill="#cdbf94")
_TUBING = Style(color="#1565c0", lineweight=0.4)
_BLACK = Style(color="#000000", lineweight=0.2, fill="#000000")
_LABEL = Style(color="#111111", lineweight=0.2)
_LEADER = Style(color="#8a8a8a", lineweight=0.15)

#: Fraction of the plot extent spanned by the widest drawn feature (automatic).
AUTO_FRACTION = 0.10
#: Automatic exaggeration never exceeds this fraction of E_max.
FOLD_MARGIN = 0.8
#: |n . p| below which a curved leg is reported as seen edge-on.
EDGE_ON = 0.2

_TEXT_H = 2.0  # callout text height, paper mm
_CHAR_W = 0.6  # glyph width / height, for sizing a callout box
_LINE_H = 1.35  # line pitch / height


def build_section(
    schematic: WellSchematic,
    azimuth: Optional[float] = None,
    step: float = 10.0,
    default_radial_scale: Optional[float] = None,
    resolver: Optional[DepthResolver] = None,
    *,
    exaggeration: Optional[float] = None,
    frac: float = AUTO_FRACTION,
    hole: str = "trace",
    annulus_fill: Optional[str] = None,
    cement: bool = True,
    callouts: bool = True,
    target_w: float = 170.0,
    target_h: float = 250.0,
    margin: float = 12.0,
) -> Drawing:
    """Build a section-view :class:`Drawing` for the primary bore.

    Parameters
    ----------
    schematic : WellSchematic
    azimuth : float, optional
        Vertical-section azimuth, degrees; default the median survey azimuth.
    step : float
        MD step (m) of the depth resolver's grid.
    default_radial_scale : float, optional
        Deprecated: use ``exaggeration``. If given, it is used as the exaggeration.
    resolver : DepthResolver, optional
    exaggeration : float, optional
        Diameter exaggeration. ``None`` (default) chooses it automatically (see the
        module docstring). A value is used as given; if it exceeds E_max the
        folding legs are reported in ``Drawing.section_info["folds"]``.
    frac : float
        Automatic mode: fraction of the plot extent spanned by the widest feature.
    hole : ``"trace"`` or ``"none"``
        Draw the open hole as a light trace at the bit radius, or not at all.
    annulus_fill : str, optional
        A colour: fill the uncemented annular space with it. ``None``: no fill.
    cement : bool
        Draw recorded cement.
    callouts : bool
        Draw shoe callouts.
    target_w, target_h, margin : float
        Paper size (mm) the drawing is fitted to, and the margin inside it.

    Returns
    -------
    Drawing
        ``section_info`` carries ``exaggeration``, ``mode`` (``"auto"`` or
        ``"user"``), ``e_max``, ``folds`` (legs where the drawn edge folds: MD
        interval and that leg's limit) and ``edge_on`` (MD intervals seen edge-on).
    """
    if default_radial_scale is not None:
        warnings.warn(
            "build_section(default_radial_scale=...) is deprecated; use "
            "exaggeration=...",
            DeprecationWarning,
            stacklevel=2,
        )
        if exaggeration is None:
            exaggeration = float(default_radial_scale)
    if hole not in ("trace", "none"):
        raise ValueError(f"hole must be 'trace' or 'none', got {hole!r}")

    bore = schematic.primary
    well = schematic.well
    if resolver is None:
        resolver = DepthResolver(bore.survey, step=step, name=well.name)
    sv = bore.survey
    if azimuth is None:
        azimuth = float(np.median(sv.azi))
    a_vs = np.radians(azimuth)
    md0, md1 = float(resolver.md[0]), float(resolver.md[-1])

    casings = sorted(bore.drawable_casings, key=lambda c: -c.od_in)
    holes = sorted(bore.hole_sections, key=lambda h: h.top_md)

    def hole_r(m):
        """Hole radius (m) at ``m``, or None where no hole section was recorded."""
        for h in holes:
            if h.top_md <= m <= h.base_md:
                return h.bit_in * _IN2M / 2.0
        return None

    def driven(c):
        """No hole section anywhere over the string's length."""
        return not any(h.top_md < c.shoe_md and h.base_md > c.top_md for h in holes)

    def outer_wall(c, m):
        """Radius (m) of the next wall outward from string ``c`` at ``m``."""
        cand = [
            o.id_at(m) * _IN2M / 2.0
            for o in casings
            if o.od_at(m) > c.od_at(m) and o.steel_at(m)
        ]
        if cand:
            return min(cand)
        r = hole_r(m)
        return r if r is not None else c.od_at(m) * _IN2M / 2.0

    # --- exaggeration and its fold limit -----------------------------------------
    md_s = np.asarray(sv.md, float)
    inc_s, azi_s = (
        np.radians(np.asarray(sv.inc, float)),
        np.radians(np.asarray(sv.azi, float)),
    )
    t_s = np.stack(
        (np.sin(inc_s) * np.cos(azi_s), np.sin(inc_s) * np.sin(azi_s), np.cos(inc_s)),
        axis=-1,
    )
    p_view = np.array([-np.sin(a_vs), np.cos(a_vs), 0.0])  # normal to the view plane

    def r_max(a, b):

        """Widest drawn radius (m) on MD a..b: the hole, or a string where wider."""
        rs = [h.bit_in * _IN2M / 2.0 for h in holes if h.top_md < b and h.base_md > a]
        rs += [c.od_in * _IN2M / 2.0 for c in casings if c.top_md < b and c.shoe_md > a]
        return max(rs) if rs else 0.0

    legs = []  # (a, b, limit, n.p)
    for i in range(md_s.size - 1):
        cr = np.cross(t_s[i], t_s[i + 1])
        s = float(np.linalg.norm(cr))
        dl = float(np.arctan2(s, float(np.dot(t_s[i], t_s[i + 1]))))
        if dl < 1e-9:
            continue
        rho = (md_s[i + 1] - md_s[i]) / dl * (float(np.dot(cr / s, p_view)) ** 2)
        r = r_max(md_s[i], md_s[i + 1])
        lim = rho / r if r > 0 else np.inf
        legs.append(
            (
                float(md_s[i]),
                float(md_s[i + 1]),
                lim,
                abs(float(np.dot(cr / s, p_view))),
            )
        )
    e_max = min([lg[2] for lg in legs], default=np.inf)

    grid = np.asarray(resolver.md, float)
    vs_g = np.asarray(resolver.vs(grid, azimuth), float)
    tvd_g = np.asarray(resolver.tvd_at(grid), float)
    extent = max(float(np.ptp(vs_g)), float(np.ptp(tvd_g)), 1.0)
    widest = r_max(md0, md1) or 0.1
    if exaggeration is None:
        E = min(frac * extent / (2.0 * widest), FOLD_MARGIN * e_max)
        mode = "auto"
    else:
        E = float(exaggeration)
        mode = "user"
    folds = [(a, b, lim) for a, b, lim, _ in legs if E >= lim]
    edge_on = [(a, b) for a, b, _, np_ in legs if np_ < EDGE_ON]

    # --- geometry helpers: position and normal at MD, exact minimum curvature ----
    def frame(m):
        """Position (VS, TVD), unit normal and unit tangent at MD(s) ``m``."""
        m = np.atleast_1d(np.asarray(m, float))
        P = np.column_stack((resolver.vs(m, azimuth), resolver.tvd_at(m)))
        inc, azi = resolver.inc_azi_at(m)
        inc, azi = np.radians(inc), np.radians(azi)
        t = np.column_stack((np.sin(inc) * np.cos(azi - a_vs), np.cos(inc)))
        n = np.linalg.norm(t, axis=1, keepdims=True)
        t = np.where(n > 1e-12, t / np.where(n > 1e-12, n, 1.0), [0.0, 1.0])
        return P, np.column_stack((t[:, 1], -t[:, 0])), t

    def radii(r, mm):

        """Radius (m) at each MD in ``mm``: a constant or a function of MD."""
        return (
            np.array([r(m) for m in mm]) if callable(r) else np.full(mm.shape, float(r))
        )

    def samples(a, b):

        """MDs from ``a`` to ``b`` at half the resolver step."""
        return np.linspace(a, b, max(2, int((b - a) / max(step / 2.0, 1.0)) + 2))

    def band(dwg, a, b, r_lo, r_hi, layer, style, kind=Polygon):

        """A band between two radii from MD ``a`` to ``b``, one entity per side."""
        if b <= a:
            return
        mm = samples(a, b)
        P, N, _ = frame(mm)
        rl, rh = radii(r_lo, mm) * E, radii(r_hi, mm) * E
        for sg in (-1.0, 1.0):
            lo = P + sg * rl[:, None] * N
            hi = P + sg * rh[:, None] * N
            pts = [tuple(p) for p in np.vstack((lo, hi[::-1]))]
            dwg.add(
                kind(pts, layer=layer, style=style)
                if kind is Polygon
                else Hatch(pts, pattern="solid", layer=layer, style=style)
            )

    def trace(dwg, a, b, r, layer, style):

        """A line at radius ``r`` from MD ``a`` to ``b``, one per side."""
        if b <= a:
            return
        mm = samples(a, b)
        P, N, _ = frame(mm)
        rr = radii(r, mm) * E
        for sg in (-1.0, 1.0):
            dwg.add(
                Polyline(
                    [tuple(p) for p in P + sg * rr[:, None] * N],
                    layer=layer,
                    style=style,
                )
            )

    def across(dwg, m, r_lo, r_hi, layer, style):

        """A line across radii ``r_lo``..``r_hi`` along the normal at MD ``m``."""
        P, N, _ = frame(m)
        for sg in (-1.0, 1.0):
            a = P[0] + sg * r_lo * E * N[0]
            b = P[0] + sg * r_hi * E * N[0]
            dwg.add(Line(tuple(a), tuple(b), layer=layer, style=style))

    dwg = Drawing(name=f"{well.name}_section")
    dwg.h_unit_label = "VS m"
    dwg.v_unit_label = "TVD m"
    for layer in (
        L_FRAME,
        L_SEA,
        L_HOLE,
        L_FLUID,
        L_CEMENT,
        L_PATH,
        L_CASING,
        L_PLUG,
        L_COMPLETION,
        L_SHOE,
        L_ANNOTATION,
    ):
        dwg.add_layer(layer)

    # --- sea and mudline: outside the outermost string only -----------------------
    seabed_md = None
    sea = None
    if (
        well.datum_elevation_m is not None
        and well.water_depth_m is not None
        and well.datum_elevation_m >= 0.0
    ):
        msl, seabed = (
            float(well.datum_elevation_m),
            float(well.datum_elevation_m + well.water_depth_m),
        )
        crossings = resolver.md_at_tvd(seabed)
        seabed_md = crossings[0] if crossings else None
        if seabed_md is not None:
            sea = (msl, seabed)

    # open hole: light trace, continuous across bit-size changes and string shoes
    if hole == "trace":
        for k, h in enumerate(holes):
            top = h.top_md if seabed_md is None else max(h.top_md, seabed_md)
            r1 = h.bit_in * _IN2M / 2.0
            trace(dwg, top, h.base_md, r1, L_HOLE, _HOLE)
            above = next(
                (h0 for h0 in holes[:k] if abs(h0.base_md - h.top_md) < 1e-6), None
            )
            if above is not None:
                r0 = above.bit_in * _IN2M / 2.0
                across(dwg, h.top_md, min(r0, r1), max(r0, r1), L_HOLE, _HOLE)
                continue
            host = [c for c in casings if abs(c.shoe_md - h.top_md) < 1e-6]
            if host and h.top_md > md0:
                r0 = max(c.id_at(c.shoe_md) for c in host) * _IN2M / 2.0
                across(dwg, h.top_md, min(r0, r1), max(r0, r1), L_HOLE, _HOLE)

    dwg.add(Polyline(list(zip(vs_g, tvd_g)), layer=L_PATH, style=_PATH))

    # --- strings: annular fill, cement, steel, shoes -----------------------------
    fill = (
        Style(color=annulus_fill, lineweight=0.0, fill=annulus_fill)
        if annulus_fill
        else None
    )
    anchors = []
    for c in casings:
        walls_only = driven(c)  # no hole around it: no cement, no shoe
        ro = lambda m, c=c: c.od_at(m) * _IN2M / 2.0  # noqa: E731
        ow = lambda m, c=c: outer_wall(c, m)  # noqa: E731
        cem = c.cement_intervals() if (cement and not walls_only) else []
        if fill is not None and not walls_only:
            for a, b in c.steel_intervals():
                inside_other = any(o.od_in > c.od_in and o.steel_at(a) for o in casings)
                start = a if (seabed_md is None or inside_other) else max(a, seabed_md)
                cuts = [
                    (max(start, t), min(b, u)) for t, u in cem if u > start and t < b
                ]
                pieces, cur = [], start
                for t, u in sorted(cuts):
                    if t > cur:
                        pieces.append((cur, t))
                    cur = max(cur, u)
                if cur < b:
                    pieces.append((cur, b))
                for a2, b2 in pieces:
                    band(dwg, a2, b2, ro, ow, L_FLUID, fill)
        for t, u in cem:
            band(dwg, t, u, ro, ow, L_CEMENT, _CEMENT, kind=Hatch)
            across(dwg, t, ro(t), ow(t), L_CEMENT, _TOC)
        for a, b, od, id_ in c.steel_profile():
            band(dwg, a, b, id_ * _IN2M / 2.0, od * _IN2M / 2.0, L_CASING, _STEEL)
        if c.shoe_remains and not walls_only:
            m = c.shoe_md
            gap = max(outer_wall(c, m) - ro(m), 0.0)
            w = 0.8 * gap * E
            if w > 0.0:
                P, N, T = frame(m)
                for sg in (-1.0, 1.0):
                    base_in = P[0] + sg * ro(m) * E * N[0]
                    dwg.add(
                        Polygon(
                            [
                                tuple(base_in),
                                tuple(base_in + sg * w * N[0]),
                                tuple(base_in - 1.6 * w * T[0]),
                            ],
                            layer=L_SHOE,
                            style=_BLACK,
                        )
                    )
        anchors.append(c)

    # --- plugs and completion ------------------------------------------------------
    def bore_r(m, od=None):
        """Bore radius (m) at ``m``: the innermost string there (optionally by OD)."""
        cand = [
            c.id_at(m)
            for c in casings
            if c.steel_at(m) and (od is None or c.has_od(od))
        ]
        return (min(cand) if cand else 6.0) * _IN2M / 2.0

    for pl in bore.cement_plugs:
        mid = (pl.top_md + pl.base_md) / 2.0
        band(dwg, pl.top_md, pl.base_md, 0.0, bore_r(mid), L_PLUG, _PLUG, kind=Hatch)
    for mp in bore.mechanical_plugs:
        r = bore_r(mp.md, getattr(mp, "casing_od_in", None))
        band(
            dwg,
            max(md0, mp.md - 3.0),
            min(md1, mp.md + 3.0),
            0.0,
            r,
            L_PLUG,
            Style(color="#111111", lineweight=0.3, fill="#4a4a4a"),
        )
    for item in bore.completion:
        if item.type != "tubing" or item.od_in is None:
            continue
        trace(
            dwg,
            item.top_md,
            item.base_md,
            item.od_in * _IN2M / 2.0,
            L_COMPLETION,
            _TUBING,
        )

    # --- fit to paper from the geometry, then the frame -----------------------------
    xmin, ymin, xmax, ymax = dwg.bounds()
    if sea is not None:
        ymin = min(ymin, sea[0])
    pad = 0.06 * max(xmax - xmin, ymax - ymin, 1.0)
    fx0, fx1, fy0, fy1 = xmin - pad, xmax + pad, ymin - pad, ymax + pad
    scale = min(
        (target_w - 2 * margin) / (fx1 - fx0), (target_h - 2 * margin) / (fy1 - fy0)
    )
    dwg.transform = ViewTransform(
        h_scale=scale, v_scale=scale, x0=(fx0 + fx1) / 2.0, y0=fy0, flip_y=True
    )
    dwg.add(Rect((fx0, fy0), fx1 - fx0, fy1 - fy0, layer=L_FRAME, style=_FRAME))

    if sea is not None:
        msl, seabed = sea
        mm = samples(md0, seabed_md)
        P, N, _ = frame(mm)
        r_out = (
            np.array(
                [
                    max(
                        [c.od_at(m) * _IN2M / 2.0 for c in casings if c.steel_at(m)]
                        or [0.0]
                    )
                    for m in mm
                ]
            )
            * E
        )
        left = P - r_out[:, None] * np.abs(N)
        right = P + r_out[:, None] * np.abs(N)
        wet = P[:, 1] >= msl
        if wet.sum() >= 2:
            L, R = left[wet], right[wet]
            dwg.add(
                Polygon(
                    [(fx0, L[0, 1]), *map(tuple, L), (fx0, L[-1, 1])],
                    layer=L_SEA,
                    style=_SEA,
                )
            )
            dwg.add(
                Polygon(
                    [(fx1, R[0, 1]), *map(tuple, R), (fx1, R[-1, 1])],
                    layer=L_SEA,
                    style=_SEA,
                )
            )
            dwg.add(
                Line(
                    (fx0, seabed),
                    (float(L[-1, 0]), seabed),
                    layer=L_SEA,
                    style=_MUDLINE,
                )
            )
            dwg.add(
                Line(
                    (float(R[-1, 0]), seabed),
                    (fx1, seabed),
                    layer=L_SEA,
                    style=_MUDLINE,
                )
            )
        dwg.entities.sort(key=lambda e: 0 if getattr(e, "layer", "") == L_SEA else 1)

    if callouts and anchors:
        _place_callouts(
            dwg,
            anchors,
            frame,
            E,
            scale,
            (fx0, fx1, fy0, fy1),
            resolver,
            grid,
            vs_g,
            tvd_g,
            r_max,
        )

    note = (
        f"x{E:.0f} ({mode}; fold limit x{e_max:.0f})"
        if np.isfinite(e_max)
        else f"x{E:.0f} ({mode})"
    )
    dwg.set_title_block(
        title=well.name,
        view=f"Section view @ {azimuth:.0f} deg",
        exaggeration=note,
        depth_scale="TVD to scale",
    )
    if folds:
        dwg.set_title_block(
            folds=", ".join(
                f"{a:.0f}-{b:.0f} m (limit x{lim:.0f})" for a, b, lim in folds
            )
        )
    dwg.section_info = {
        "exaggeration": E,
        "mode": mode,
        "e_max": e_max,
        "folds": folds,
        "edge_on": edge_on,
    }
    return dwg


def _place_callouts(
    dwg, casings, frame, E, scale, box, resolver, grid, vs_g, tvd_g, r_max
):
    """Shoe callouts: off the drawing, inside the frame, clear of each other, with
    a leader to the shoe. Forces from the drawing, the other callouts and the frame
    edges; where they cancel, the nearest free position around the anchor."""
    fx0, fx1, fy0, fy1 = box
    h = _TEXT_H / scale  # text height in world units
    pts = []
    for m in grid:
        P, N, _ = frame(m)
        r = r_max(max(grid[0], m - 1.0), min(grid[-1], m + 1.0)) * E
        pts.extend(P[0] + f * r * N[0] for f in np.linspace(-1.0, 1.0, 9))
    geom = np.array(pts)
    W = max(fx1 - fx0, fy1 - fy0)

    items = []
    for c in casings:
        P, N, _ = frame(c.shoe_md)
        inc, _ = resolver.inc_azi_at(c.shoe_md)
        lines = [
            f"{c.name} shoe",
            f"{c.shoe_md:.0f} mMD / {P[0, 1]:.0f} mTVD",
            f"inc {float(inc):.1f} deg",
        ]
        spec = " ".join(
            s
            for s in (
                f"{c.nominal_weight_ppf:g} ppf" if c.nominal_weight_ppf else "",
                c.grade or "",
            )
            if s
        )
        if spec:
            lines.append(spec)
        if c.toc_md is not None:
            lines.append(f"TOC {c.toc_md:.0f} mMD")
        w = max(len(s) for s in lines) * _CHAR_W * h
        hh = len(lines) * _LINE_H * h
        side = N[0] if N[0, 0] >= 0 else -N[0]
        anchor = P[0] + c.od_at(c.shoe_md) * _IN2M / 2.0 * E * side
        centre = anchor + side * (0.06 * W + w / 2.0)
        items.append([lines, anchor, centre, w, hh])

    def bbox(it):

        """Callout box (x0, x1, y0, y1) in world units."""
        _, _, (cx, cy), w, hh = it
        return cx - w / 2, cx + w / 2, cy - hh / 2, cy + hh / 2

    mg = 0.01 * W

    def clash(i, x0, x1, y0, y1):

        """True if a box leaves the frame, meets the drawing or another callout."""
        if x0 < fx0 or x1 > fx1 or y0 < fy0 or y1 > fy1:
            return True
        if np.any(
            (geom[:, 0] > x0 - mg)
            & (geom[:, 0] < x1 + mg)
            & (geom[:, 1] > y0 - mg)
            & (geom[:, 1] < y1 + mg)
        ):
            return True
        for j, other in enumerate(items):
            if j != i:
                a0, a1, b0, b1 = bbox(other)
                if x0 < a1 and x1 > a0 and y0 < b1 and y1 > b0:
                    return True
        return False

    stepw = 0.01 * W
    for _ in range(300):
        moved = False
        for i, it in enumerate(items):
            x0, x1, y0, y1 = bbox(it)
            f = np.zeros(2)
            hit = geom[
                (geom[:, 0] > x0 - mg)
                & (geom[:, 0] < x1 + mg)
                & (geom[:, 1] > y0 - mg)
                & (geom[:, 1] < y1 + mg)
            ]
            if len(hit):
                d = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) - hit.mean(axis=0)
                f += d / (np.linalg.norm(d) + 1e-12)
            for j, other in enumerate(items):
                if j != i:
                    a0, a1, b0, b1 = bbox(other)
                    if x0 < a1 and x1 > a0 and y0 < b1 and y1 > b0:
                        f[1] += 1.0 if (y0 + y1) > (b0 + b1) else -1.0
            f[0] += 2.0 * (x0 < fx0) - 2.0 * (x1 > fx1)
            f[1] += 2.0 * (y0 < fy0) - 2.0 * (y1 > fy1)
            if f.any():
                it[2] = it[2] + f * stepw
                moved = True
        if not moved:
            break
    for i, it in enumerate(items):
        if not clash(i, *bbox(it)):
            continue
        _, anchor, _, w, hh = it
        for rad in np.linspace(0.03, 0.6, 30) * W:
            done = False
            for ang in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
                cx, cy = anchor[0] + rad * np.cos(ang), anchor[1] + rad * np.sin(ang)
                if not clash(i, cx - w / 2, cx + w / 2, cy - hh / 2, cy + hh / 2):
                    it[2] = np.array([cx, cy])
                    done = True
                    break
            if done:
                break

    for lines, anchor, (cx, cy), w, hh in items:
        x_left = cx - w / 2
        # leader to the nearest point of the callout box, so it never crosses the text
        ex = min(max(anchor[0], cx - w / 2), cx + w / 2)
        ey = min(max(anchor[1], cy - hh / 2), cy + hh / 2)
        dwg.add(Line(tuple(anchor), (ex, ey), layer=L_ANNOTATION, style=_LEADER))
        top = cy - hh / 2 + _LINE_H * h * 0.8
        for k, s in enumerate(lines):
            dwg.add(
                Text(
                    (x_left, top + k * _LINE_H * h),
                    s,
                    height=_TEXT_H,
                    layer=L_ANNOTATION,
                    style=_LABEL,
                )
            )
