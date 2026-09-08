"""Section-view generator.

Components follow the real trajectory in the vertical-section (VS @ azimuth)
vs TVD plane: casing/cement/plug/tubing are drawn as ribbons offset normal to
the local path tangent, radius exaggerated with the same **per-section**
:class:`RadialScale` used by the column view, and casings terminate in the
standard black-triangle shoe. Vertical (TVD) is uniform/to-scale.

No renderer imports -- emits :mod:`welleng.schematic.drawing` entities only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .depth import DepthResolver, radial_scale_for
from .drawing import Drawing, Hatch, Polygon, Polyline, Style, Text, ViewTransform
from .models import WellSchematic

_IN2M = 0.0254

L_PATH = "PATH"
L_CASING = "CASING"
L_CEMENT = "CEMENT"
L_PLUG = "PLUG"
L_COMPLETION = "COMPLETION"
L_SHOE = "SHOE"
L_ANNOTATION = "ANNOTATION"

_PATH = Style(color="#999999", lineweight=0.25, linestyle="dashed")
_STEEL = Style(color="#222222", lineweight=0.5)
_CEMENT = Style(color="#7a6f4a", lineweight=0.2, fill="#d8cfae")
_PLUG = Style(color="#6b5d2f", lineweight=0.2, fill="#cdbf94")
_TUBING = Style(color="#1565c0", lineweight=0.5)
_BLACK = Style(color="#000000", lineweight=0.2, fill="#000000")
_LABEL = Style(color="#111111", lineweight=0.2)


def build_section(
    schematic: WellSchematic,
    azimuth: Optional[float] = None,
    step: float = 10.0,
    default_radial_scale: float = 40.0,
    resolver: Optional[DepthResolver] = None,
) -> Drawing:
    """Build a section-view :class:`Drawing` for the primary bore."""
    bore = schematic.primary
    if resolver is None:
        resolver = DepthResolver(bore.survey, step=step, name=schematic.well.name)
    if azimuth is None:
        azimuth = float(np.median(bore.survey.azi))
    radial = radial_scale_for(bore, resolver, mode="MD", default=default_radial_scale)

    md_grid = resolver.md
    vs = resolver.vs(md_grid, azimuth)
    tvd = resolver.tvd

    def P(m):
        return np.array([np.interp(m, md_grid, vs), np.interp(m, md_grid, tvd)])

    def normal(m):
        d = P(min(md_grid[-1], m + 2.5)) - P(max(md_grid[0], m - 2.5))
        norm = np.linalg.norm(d)
        t = d / norm if norm > 1e-9 else np.array([0.0, 1.0])
        return np.array([-t[1], t[0]]), t

    def ribbon(top, base, od_in):
        n = max(4, int((base - top) / 20))
        mm = np.linspace(top, base, n)
        up, dn = [], []
        for m in mm:
            p = P(m)
            nrm, _ = normal(m)
            w = od_in * _IN2M * radial.at(m) / 2.0
            up.append(tuple(p + nrm * w))
            dn.append(tuple(p - nrm * w))
        return up + dn[::-1]

    dwg = Drawing(name=f"{schematic.well.name}_section")
    dwg.h_unit_label = "VS m"
    dwg.v_unit_label = "TVD m"
    for layer in (L_PATH, L_CEMENT, L_CASING, L_PLUG, L_COMPLETION,
                  L_SHOE, L_ANNOTATION):
        dwg.add_layer(layer)

    # trajectory
    dwg.add(Polyline([tuple(P(m)) for m in md_grid], layer=L_PATH, style=_PATH))

    # cement annuli then casing steel outlines
    for c in sorted(bore.casings, key=lambda c: -c.od_in):
        dwg.add(Hatch(ribbon(c.toc_md, c.shoe_md, c.od_in),
                      pattern="cement", layer=L_CEMENT, style=_CEMENT))
    for c in bore.casings:
        dwg.add(Polygon(ribbon(c.top_md, c.shoe_md, c.od_in),
                        layer=L_CASING,
                        style=Style(color="#222222", lineweight=0.5, fill=None)))
        _shoe(dwg, P, normal, radial, c.shoe_md, c.od_in)
        p = P(c.shoe_md)
        nrm, _ = normal(c.shoe_md)
        w = c.od_in * _IN2M * radial.at(c.shoe_md) / 2.0
        dwg.add(Text(tuple(p + nrm * w * 1.2), c.name, height=2.0,
                     layer=L_ANNOTATION, style=_LABEL))

    # plugs (bore-width ribbon)
    for pl in bore.cement_plugs:
        mid = (pl.top_md + pl.base_md) / 2.0
        cand = [c.id_in for c in bore.casings if c.top_md <= mid <= c.shoe_md]
        bore_id = min(cand) if cand else 6.0
        dwg.add(Hatch(ribbon(pl.top_md, pl.base_md, bore_id),
                      pattern="plug", layer=L_PLUG, style=_PLUG))
        dwg.add(Text(tuple(P(mid)), pl.name, height=2.0,
                     layer=L_ANNOTATION, style=Style(color="#6b5d2f")))

    # completion tubing follows the path
    for item in bore.completion:
        if item.type != "tubing":
            continue
        n = max(4, int((item.base_md - item.top_md) / 20))
        mm = np.linspace(item.top_md, item.base_md, n)
        for sign in (-1, 1):
            pts = []
            for m in mm:
                p = P(m)
                nrm, _ = normal(m)
                w = item.od_in * _IN2M * radial.at(m) / 2.0
                pts.append(tuple(p + sign * nrm * w))
            dwg.add(Polyline(pts, layer=L_COMPLETION, style=_TUBING))

    dwg.set_title_block(
        title=schematic.well.name,
        view=f"Section view @ {azimuth:.0f} deg",
        radial=_scale_note(radial),
        depth_scale="TVD uniform (to scale)",
    )
    _fit_equal(dwg, target_w=170.0, target_h=250.0, margin=12.0)
    return dwg


def _shoe(dwg, P, normal, radial, shoe_md, od_in) -> None:
    """Black triangle casing shoe on both sides, oriented along the path."""
    p = P(shoe_md)
    nrm, tan = normal(shoe_md)
    w = od_in * _IN2M * radial.at(shoe_md) / 2.0
    length = od_in * _IN2M * radial.at(shoe_md) * 0.9
    for sign in (-1, 1):
        base = p + sign * nrm * w
        apex = base + tan * length
        shoulder = p + sign * nrm * w * 0.4
        dwg.add(Polygon([tuple(base), tuple(apex), tuple(shoulder)],
                        layer=L_SHOE, style=_BLACK))


def _scale_note(radial) -> str:
    lo, hi = min(radial.scales), max(radial.scales)
    return f"x{hi:g}" if lo == hi else f"x{hi:g} (shallow) -> x{lo:g} (deep)"


def _fit_equal(dwg: Drawing, target_w: float, target_h: float, margin: float) -> None:
    """Equal-aspect fit (section view is a true VS/TVD plane)."""
    xmin, ymin, xmax, ymax = dwg.bounds()
    w = max(xmax - xmin, 1e-6)
    h = max(ymax - ymin, 1e-6)
    scale = min((target_w - 2 * margin) / w, (target_h - 2 * margin) / h)
    dwg.transform = ViewTransform(
        h_scale=scale, v_scale=scale,
        x0=(xmin + xmax) / 2.0, y0=ymin, flip_y=True,
    )
