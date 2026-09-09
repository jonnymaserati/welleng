"""Composable shared-depth tracks.

Stackable vertical tracks that all render against one shared depth axis:
:class:`DepthTrack` (the ruler), :class:`SchematicTrack` (nested casings),
:class:`LithologyTrack` (litho bands), :class:`IntervalsTrack` (seal/flow
bands) and :class:`PPFPTrack` (pore/frac **pressure** curves). Each track
occupies a horizontal band ``[x0, x0+width]`` in paper mm and maps its own
internal quantity across that band; the vertical (depth) axis is shared and
uniform.

No renderer imports -- emits :mod:`welleng.schematic.drawing` entities only.
"""
from __future__ import annotations

from dataclasses import dataclass

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
)
from .models import WellSchematic

_GRID = Style(color="#dddddd", lineweight=0.15)
_LABEL = Style(color="#111111", lineweight=0.2)
_STEEL = Style(color="#222222", lineweight=0.3, fill="#3f3f3f")
_CEMENT = Style(color="#7a6f4a", lineweight=0.2, fill="#d8cfae")
_PLUG = Style(color="#6b5d2f", lineweight=0.2, fill="#cdbf94")
_TUBING = Style(color="#1565c0", lineweight=0.4)
_BLACK = Style(color="#000000", lineweight=0.2, fill="#000000")
_PORE = Style(color="#1565c0", lineweight=0.45)
_FRAC = Style(color="#c62828", lineweight=0.45)
_FILL = Style(color="#e6e6e6", lineweight=0.0, fill="#ececec")


@dataclass
class DepthLayout:
    """Shared depth<->paper mapping handed to every track."""

    mode: str
    resolver: DepthResolver
    v_scale: float          # paper mm per depth-metre
    ymax: float             # max depth (metres) in ``mode``

    def y(self, md) -> float:
        return float(self.resolver.depth(md, self.mode)) * self.v_scale

    @property
    def bottom(self) -> float:
        return self.ymax * self.v_scale


class Track:
    """Base track: a titled band of fixed paper width (mm)."""

    title = "track"
    width = 30.0

    def build(self, dwg: Drawing, layout: DepthLayout, x0: float) -> None:
        raise NotImplementedError

    def _header(self, dwg: Drawing, x0: float) -> None:
        dwg.add(Text((x0 + self.width / 2.0, -4.0), self.title, height=2.2,
                     ha="center", va="bottom", layer="ANNOTATION", style=_LABEL))


class DepthTrack(Track):
    title = "depth"
    width = 14.0

    def build(self, dwg, layout, x0):
        dwg.add_layer("GRID")
        dwg.add_layer("ANNOTATION")
        self._header(dwg, x0)
        dwg.add(Text((x0 + self.width / 2.0, layout.bottom + 6.0),
                     f"{layout.mode} (m)", height=2.0, ha="center", va="top",
                     layer="ANNOTATION", style=_LABEL))
        step = _nice_step(layout.ymax)
        depth = 0.0
        while depth <= layout.ymax + 1e-6:
            y = depth * layout.v_scale
            dwg.add(Line((x0, y), (x0 + self.width, y), layer="GRID", style=_GRID))
            dwg.add(Text((x0 + self.width - 1.0, y), f"{depth:.0f}", height=1.8,
                         ha="right", va="center", layer="ANNOTATION", style=_LABEL))
            depth += step


class SchematicTrack(Track):
    title = "schematic\n(radius exagg.)"
    width = 66.0

    def __init__(self, schematic: WellSchematic, width: float = 66.0):
        self.schematic = schematic
        self.width = width

    def build(self, dwg, layout, x0):
        for layer in ("CEMENT", "CASING", "PLUG", "COMPLETION", "SHOE", "ANNOTATION"):
            dwg.add_layer(layer)
        self._header(dwg, x0)
        bore = self.schematic.primary
        cx = x0 + self.width / 2.0
        max_r = max([c.od_in / 2.0 for c in bore.casings]
                    + [h.bit_in / 2.0 for h in bore.hole_sections] + [1.0])
        sx = (self.width / 2.0 * 0.9) / max_r    # mm per inch-radius (band fit)

        def rx(r_in, sign):
            return cx + sign * r_in * sx

        # cement annuli
        ordered = sorted(bore.casings, key=lambda c: -c.od_in)
        for i, c in enumerate(ordered):
            if c.toc_md is None:
                continue      # no cement RECORDED: draw nothing, claim nothing
            r_out = c.od_in / 2.0
            r_in = ordered[i - 1].id_in / 2.0 if i > 0 else max_r
            lo, hi = sorted((r_out, r_in))
            yt, yb = layout.y(c.toc_md), layout.y(c.shoe_md)
            for sign in (-1, 1):
                a, b = rx(lo, sign), rx(hi, sign)
                dwg.add(Hatch([(a, yt), (b, yt), (b, yb), (a, yb)],
                              pattern="cement", layer="CEMENT", style=_CEMENT))
        # casing steel + shoes
        shoe_h = layout.bottom * 0.012
        for c in bore.casings:
            r_out, r_in = c.od_in / 2.0, c.id_in / 2.0
            yt, yb = layout.y(c.top_md), layout.y(c.shoe_md)
            for sign in (-1, 1):
                a, b = rx(r_in, sign), rx(r_out, sign)
                dwg.add(Rect((min(a, b), yt), abs(b - a), yb - yt,
                             layer="CASING", style=_STEEL))
                sxo = rx(r_out, sign)
                dwg.add(Polygon([(sxo - shoe_h, yb), (sxo + shoe_h, yb),
                                 (sxo + sign * shoe_h, yb + shoe_h)],
                                layer="SHOE", style=_BLACK))
            dwg.add(Text((rx(r_out, 1) + 1.0, yt + 3.0), c.name, height=1.7,
                         va="top", layer="ANNOTATION", style=_LABEL))
        # plugs
        for p in bore.cement_plugs:
            mid = (p.top_md + p.base_md) / 2.0
            cand = [c.id_in / 2.0 for c in bore.casings if c.top_md <= mid <= c.shoe_md]
            r_in = min(cand) if cand else 3.0
            yt, yb = layout.y(p.top_md), layout.y(p.base_md)
            dwg.add(Hatch([(rx(r_in, -1), yt), (rx(r_in, 1), yt),
                           (rx(r_in, 1), yb), (rx(r_in, -1), yb)],
                          pattern="plug", layer="PLUG", style=_PLUG))
            dwg.add(Text((rx(r_in, -1) - 1.0, (yt + yb) / 2.0), p.name, height=1.7,
                         ha="right", va="center", layer="ANNOTATION",
                         style=Style(color="#6b5d2f")))
        # completion tubing
        for item in bore.completion:
            if item.type != "tubing":
                continue
            r = item.od_in / 2.0
            yt, yb = layout.y(item.top_md), layout.y(item.base_md)
            for sign in (-1, 1):
                dwg.add(Line((rx(r, sign), yt), (rx(r, sign), yb),
                             layer="COMPLETION", style=_TUBING))


class LithologyTrack(Track):
    title = "litho"
    width = 16.0

    def __init__(self, schematic: WellSchematic, width: float = 16.0):
        self.schematic = schematic
        self.width = width

    def build(self, dwg, layout, x0):
        dwg.add_layer("LITHO")
        dwg.add_layer("ANNOTATION")
        self._header(dwg, x0)
        forms = self.schematic.formations
        for a, b in zip(forms[:-1], forms[1:]):
            yt, yb = layout.y(a.top_md), layout.y(b.top_md)
            dwg.add(Rect((x0, yt), self.width, yb - yt, layer="LITHO",
                         style=Style(color="#808080", lineweight=0.15, fill=a.color)))
            if a.name:
                dwg.add(Text((x0 + self.width / 2.0, (yt + yb) / 2.0), a.name,
                             height=1.7, ha="center", va="center", rotation=90.0,
                             layer="ANNOTATION", style=_LABEL))


class IntervalsTrack(Track):
    title = "seal/flow"
    width = 12.0

    def __init__(self, schematic: WellSchematic, width: float = 12.0):
        self.schematic = schematic
        self.width = width

    def build(self, dwg, layout, x0):
        dwg.add_layer("INTERVALS")
        dwg.add_layer("ANNOTATION")
        self._header(dwg, x0)
        forms = self.schematic.formations
        for a, b in zip(forms[:-1], forms[1:]):
            if not (a.seal or a.flow):
                continue
            yt, yb = layout.y(a.top_md), layout.y(b.top_md)
            color = "#5e35b1" if a.seal else "#e65100"
            label = "SEAL" if a.seal else "FLOW"
            dwg.add(Rect((x0, yt), self.width, yb - yt, layer="INTERVALS",
                         style=Style(color=color, lineweight=0.2, fill=color,
                                     opacity=0.35)))
            dwg.add(Text((x0 + self.width / 2.0, (yt + yb) / 2.0), label,
                         height=1.6, ha="center", va="center", rotation=90.0,
                         layer="ANNOTATION", style=Style(color=color)))


class PPFPTrack(Track):
    title = "PP / FP"
    width = 44.0

    def __init__(self, schematic: WellSchematic, width: float = 44.0):
        self.schematic = schematic
        self.width = width

    def build(self, dwg, layout, x0):
        dwg.add_layer("PPFP")
        dwg.add_layer("ANNOTATION")
        self._header(dwg, x0)
        pp = self.schematic.pressures
        if pp is None:
            return
        pmin = min(min(pp.pore), min(pp.frac))
        pmax = max(max(pp.pore), max(pp.frac))
        span = max(pmax - pmin, 1e-6)
        pad = span * 0.08
        pmin, pmax = pmin - pad, pmax + pad
        span = pmax - pmin

        def px(p):
            return x0 + (p - pmin) / span * self.width

        ys = [layout.y(m) for m in pp.md]
        pore = [(px(p), y) for p, y in zip(pp.pore, ys)]
        frac = [(px(p), y) for p, y in zip(pp.frac, ys)]
        dwg.add(Polygon(pore + frac[::-1], layer="PPFP", style=_FILL))
        dwg.add(Polyline(pore, layer="PPFP", style=_PORE))
        dwg.add(Polyline(frac, layer="PPFP", style=_FRAC))
        dwg.add(Text((x0 + self.width / 2.0, layout.bottom + 6.0),
                     f"pressure ({pp.unit})", height=2.0, ha="center", va="top",
                     layer="ANNOTATION", style=_LABEL))
        dwg.add(Text((px(pp.pore[-1]), ys[-1] + 2.0), "pore", height=1.6,
                     ha="center", va="top", layer="ANNOTATION", style=_PORE))
        dwg.add(Text((px(pp.frac[-1]), ys[-1] + 2.0), "frac", height=1.6,
                     ha="center", va="top", layer="ANNOTATION", style=_FRAC))


def _nice_step(span: float) -> float:
    raw = span / 8.0
    for s in (50, 100, 250, 500, 1000, 2000):
        if raw <= s:
            return float(s)
    return 5000.0
