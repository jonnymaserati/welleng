"""Renderer-agnostic CAD drawing model.

A :class:`Drawing` is a bag of geometric entities on named :class:`Layer` s,
expressed in **real-world coordinates**, plus a :class:`ViewTransform` that
maps world coordinates to *paper millimetres* with **independent horizontal
and vertical scale** (radius exaggeration = H-scale != V-scale). Backends
(:mod:`welleng.schematic.backends`) consume this model and apply the transform
at render time.

Convention: after ``transform.apply`` every coordinate is in paper mm. Text
heights and line weights are already in paper mm and are NOT transformed.

No renderer imports here -- pure data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]


# --------------------------------------------------------------------------
# per-section radial (horizontal) exaggeration
# --------------------------------------------------------------------------
@dataclass
class RadialScale:
    """Piecewise-constant radial exaggeration as a function of depth.

    ``breaks[k]`` is the top depth of section ``k`` (in whatever depth domain
    the caller uses -- MD or TVD) and ``scales[k]`` its exaggeration. The
    depth (vertical) scale is handled separately and stays uniform; only this
    radial multiplier steps at section boundaries, producing the intentional
    inward "jog". Enforced **monotonic non-increasing** so every step tapers
    inward with depth.
    """

    breaks: List[float]
    scales: List[float]

    def __post_init__(self) -> None:
        if len(self.breaks) != len(self.scales):
            raise ValueError("breaks and scales must be the same length")
        order = sorted(range(len(self.breaks)), key=lambda i: self.breaks[i])
        self.breaks = [self.breaks[i] for i in order]
        self.scales = [self.scales[i] for i in order]
        for a, b in zip(self.scales, self.scales[1:]):
            if b > a:
                raise ValueError(
                    "radial scale must be monotonic non-increasing with depth"
                )

    @classmethod
    def uniform(cls, scale: float, top: float = 0.0) -> "RadialScale":
        return cls(breaks=[top], scales=[scale])

    def at(self, depth: float) -> float:
        """The radial multiplier active at ``depth`` (piecewise-constant)."""
        s = self.scales[0]
        for b, sc in zip(self.breaks, self.scales):
            if depth >= b:
                s = sc
            else:
                break
        return s

    def boundaries_within(self, d_top: float, d_base: float) -> List[float]:
        """Section-top depths strictly inside ``(d_top, d_base)``, ascending."""
        return [b for b in self.breaks if d_top < b < d_base]


# --------------------------------------------------------------------------
# style + layers
# --------------------------------------------------------------------------
@dataclass
class Style:
    """Visual style.

    ``color``/``fill`` are hex or CSS names; a ``None`` fill means no fill.
    """

    color: str = "#000000"
    lineweight: float = 0.25          # paper mm
    fill: Optional[str] = None        # fill colour or None (open)
    linestyle: str = "solid"          # solid | dashed | dotted
    opacity: float = 1.0


@dataclass
class Layer:
    name: str
    color: str = "#000000"
    visible: bool = True


# --------------------------------------------------------------------------
# entities (geometry in world coordinates; style in paper units)
# --------------------------------------------------------------------------
@dataclass
class Line:
    start: Point
    end: Point
    layer: str = "0"
    style: Style = field(default_factory=Style)


@dataclass
class Polyline:
    points: List[Point]
    closed: bool = False
    layer: str = "0"
    style: Style = field(default_factory=Style)


@dataclass
class Rect:
    """Axis-aligned rectangle by two opposite corners (world coords)."""

    corner: Point
    width: float
    height: float
    layer: str = "0"
    style: Style = field(default_factory=Style)

    def as_points(self) -> List[Point]:
        x, y = self.corner
        w, h = self.width, self.height
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


@dataclass
class Polygon:
    points: List[Point]
    layer: str = "0"
    style: Style = field(default_factory=Style)


@dataclass
class Hatch:
    """Filled boundary with a named fill pattern (cement, plug, ...)."""

    boundary: List[Point]
    pattern: str = "cement"
    layer: str = "0"
    style: Style = field(default_factory=Style)


@dataclass
class Text:
    position: Point
    text: str
    height: float = 2.5               # paper mm
    rotation: float = 0.0             # degrees, ccw
    ha: str = "left"                  # left | center | right
    va: str = "baseline"             # top | center | baseline | bottom
    layer: str = "0"
    style: Style = field(default_factory=Style)


@dataclass
class Symbol:
    """A reusable block definition: named list of local-coordinate entities."""

    name: str
    entities: List[object] = field(default_factory=list)


@dataclass
class SymbolRef:
    """An instance (INSERT) of a :class:`Symbol` at a world position."""

    name: str
    position: Point
    sx: float = 1.0                   # x scale (negative = mirror)
    sy: float = 1.0
    rotation: float = 0.0             # degrees, ccw
    layer: str = "0"


# --------------------------------------------------------------------------
# view transform
# --------------------------------------------------------------------------
@dataclass
class ViewTransform:
    """World -> paper-mm map with independent H/V scale.

    ``h_scale`` / ``v_scale`` are paper mm per world x / y unit. ``flip_y``
    sends increasing depth downward (negative paper-y).
    """

    h_scale: float = 1.0
    v_scale: float = 1.0
    x0: float = 0.0
    y0: float = 0.0
    flip_y: bool = True

    def apply(self, x: float, y: float) -> Point:
        px = (x - self.x0) * self.h_scale
        py = (y - self.y0) * self.v_scale
        if self.flip_y:
            py = -py
        return (px, py)

    def apply_many(self, pts) -> List[Point]:
        return [self.apply(x, y) for x, y in pts]


# --------------------------------------------------------------------------
# drawing container
# --------------------------------------------------------------------------
@dataclass
class Drawing:
    """A CAD document: layers, entities, symbol definitions, and a transform."""

    name: str = "well_schematic"
    units: str = "mm"                                  # paper units after transform
    transform: ViewTransform = field(default_factory=ViewTransform)
    layers: Dict[str, Layer] = field(default_factory=dict)
    entities: List[object] = field(default_factory=list)
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    title_block: Dict[str, str] = field(default_factory=dict)
    h_unit_label: str = "m"                            # for the horizontal scale bar
    v_unit_label: str = "m"

    def __post_init__(self) -> None:
        self.add_layer("0")

    # --- layers ------------------------------------------------------------
    def add_layer(self, name: str, color: str = "#000000",
                  visible: bool = True) -> Layer:
        layer = self.layers.get(name)
        if layer is None:
            layer = Layer(name=name, color=color, visible=visible)
            self.layers[name] = layer
        return layer

    # --- entities ----------------------------------------------------------
    def add(self, entity) -> object:
        layer = getattr(entity, "layer", None)
        if layer is not None:
            self.add_layer(layer)
        self.entities.append(entity)
        return entity

    def extend(self, entities) -> None:
        for e in entities:
            self.add(e)

    # --- symbols -----------------------------------------------------------
    def define_symbol(self, symbol: Symbol) -> Symbol:
        self.symbols[symbol.name] = symbol
        return symbol

    def place_symbol(
        self,
        name: str,
        position: Point,
        sx: float = 1.0,
        sy: float = 1.0,
        rotation: float = 0.0,
        layer: str = "0",
    ) -> SymbolRef:
        ref = SymbolRef(name, position, sx, sy, rotation, layer)
        return self.add(ref)

    # --- annotations: title block + scale bars -----------------------------
    def set_title_block(self, **fields) -> None:
        self.title_block.update({k: str(v) for k, v in fields.items()})

    def visible_layers(self) -> List[str]:
        return [n for n, layer in self.layers.items() if layer.visible]

    def bounds(self):
        """World-coordinate bounding box (xmin, ymin, xmax, ymax) of entities."""
        xs: List[float] = []
        ys: List[float] = []

        def acc(pts):
            for x, y in pts:
                xs.append(x)
                ys.append(y)

        for e in self.entities:
            if isinstance(e, Line):
                acc([e.start, e.end])
            elif isinstance(e, (Polyline, Polygon)):
                acc(e.points)
            elif isinstance(e, Rect):
                acc(e.as_points())
            elif isinstance(e, Hatch):
                acc(e.boundary)
            elif isinstance(e, (Text, SymbolRef)):
                acc([e.position])
        if not xs:
            return (0.0, 0.0, 1.0, 1.0)
        return (min(xs), min(ys), max(xs), max(ys))
