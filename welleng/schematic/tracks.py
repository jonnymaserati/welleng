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
    SymbolRef,
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
_LEADER = Style(color="#9a9a9a", lineweight=0.12)
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


def _remap(entities, kx: float, ky: float, x0: float):
    """Re-express world-coordinate entities in a track's paper band.

    ``x`` (exaggerated inch-radius, centred on 0) scales by ``kx`` about the
    band centre; ``y`` (depth in the layout's domain) scales by ``ky``. Text
    heights and line weights are already paper-mm and are NOT touched.
    """
    def pt(p):
        return (x0 + p[0] * kx, p[1] * ky)

    out = []
    for e in entities:
        if isinstance(e, Line):
            out.append(Line(pt(e.start), pt(e.end), layer=e.layer, style=e.style))
        elif isinstance(e, Polyline):
            out.append(Polyline([pt(q) for q in e.points], closed=e.closed,
                                layer=e.layer, style=e.style))
        elif isinstance(e, Polygon):
            out.append(Polygon([pt(q) for q in e.points], layer=e.layer,
                               style=e.style))
        elif isinstance(e, Hatch):
            out.append(Hatch([pt(q) for q in e.boundary], pattern=e.pattern,
                             layer=e.layer, style=e.style))
        elif isinstance(e, Rect):
            c = pt(e.corner)
            out.append(Rect(c, e.width * kx, e.height * ky, layer=e.layer,
                            style=e.style))
        elif isinstance(e, SymbolRef):
            # sx/sy are in world units, so they scale with their axis.
            out.append(SymbolRef(e.name, pt(e.position), e.sx * kx, e.sy * ky,
                                 e.rotation, e.layer))
        elif isinstance(e, Text):
            out.append(Text(pt(e.position), e.text, height=e.height,
                            rotation=e.rotation, ha=e.ha, va=e.va,
                            layer=e.layer, style=e.style))
    return out


class SchematicTrack(Track):
    """The column schematic, rendered into a track band.

    ⚠️ **This delegates to :func:`welleng.schematic.column.build_column`; it
    does NOT draw the well itself.** It used to, and the copy diverged: while
    ``column.py`` gained flat-grey cement, at-gauge open hole, wall-anchored
    shoes, formation bands, annulus fluids, perforations, liner hangers and
    combination strings, this track still drew hatched cement and a shoe whose
    height was a fraction of WELL DEPTH -- the exact defect corrected in
    ``column.py`` months earlier. Nothing failed; the composite sheet simply
    disagreed with the column view of the same well, and a state doc recorded
    the split as closed because only two of the three renderers were checked.

    One renderer means every future correction reaches this track for free,
    which is the only version of this that stays true.
    """

    title = "schematic\n(radius exagg.)"
    width = 66.0

    #: Fraction of the band reserved each side for names. The well is drawn
    #: into what is left. Without it a name has nowhere to go: the geometry
    #: fills the band, so anchoring on the wall prints across the casing and
    #: anchoring at the edge prints across whatever reaches the edge.
    GUTTER = 0.17

    def __init__(self, schematic: WellSchematic, width: float = 66.0,
                 rock: bool = False):
        self.schematic = schematic
        self.width = width
        self.rock = rock

    def build(self, dwg, layout, x0):
        from .column import _free_slot, build_column

        col = build_column(self.schematic, mode=layout.mode, bare=True,
                           rock=self.rock)
        for layer in col.layers:
            dwg.add_layer(layer)
        for name, sym in col.symbols.items():
            dwg.define_symbol(sym)
        self._header(dwg, x0)

        xmin, _ymin, xmax, _ymax = col.bounds()
        half = max(abs(xmin), abs(xmax), 1e-9)
        draw_w = self.width * (1.0 - 2.0 * self.GUTTER)
        kx = (draw_w / 2.0) / half
        ky = layout.v_scale
        dwg.extend(_remap(col.entities, kx, ky, x0 + self.width / 2.0))

        # Names in the reserved gutters, at the SHOE depth, de-collided with
        # the same outward search column.py uses. Every string starts at
        # surface, so top-anchored names print on one line at depth 0; and two
        # shoes a few metres apart print on top of each other unless something
        # moves them.
        bore = self.schematic.primary
        min_gap = layout.bottom * 0.024
        for side, rows in (
            (1, [(c.shoe_md, c.name, _LABEL) for c in bore.casings]),
            (-1, [((p.top_md + p.base_md) / 2.0, p.name,
                   Style(color="#5c5c5c")) for p in bore.cement_plugs]),
        ):
            taken: list = []
            gx = (x0 + self.width - 0.5) if side > 0 else (x0 + 0.5)
            for depth, text, style in sorted(rows):
                y0 = layout.y(depth)
                y = _free_slot(y0, taken, min_gap, layout.bottom)
                taken.append(y)
                if abs(y - y0) > 1e-9:
                    dwg.add(Line((gx, y), (gx - side * 1.5, y0),
                                 layer="ANNOTATION", style=_LEADER))
                dwg.add(Text((gx, y), text, height=1.7,
                             ha="right" if side > 0 else "left", va="center",
                             layer="ANNOTATION", style=style))


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
        for a, top, base in self.schematic.formation_bands():
            yt, yb = layout.y(top), layout.y(base)
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
        for a, top, base in self.schematic.formation_bands():
            if not (a.seal or a.flow):
                continue
            yt, yb = layout.y(top), layout.y(base)
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
