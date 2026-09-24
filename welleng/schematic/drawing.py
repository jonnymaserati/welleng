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

    breaks: List[float]              # section-top depths; sorted ascending on init
    scales: List[float]              # radial multiplier per section, matching breaks

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
        """A single-section scale: ``scale`` applies at every depth."""
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
    opacity: float = 1.0              # 0 (transparent) .. 1 (opaque)


@dataclass
class Layer:
    """A named layer that entities reference by name."""

    name: str                         # key in ``Drawing.layers``
    color: str = "#000000"
    visible: bool = True              # listed by ``Drawing.visible_layers`` when True


# --------------------------------------------------------------------------
# entities (geometry in world coordinates; style in paper units)
# --------------------------------------------------------------------------
@dataclass
class Line:
    """Straight segment between two points (world coords)."""

    start: Point                      # (x, y) world coords
    end: Point                        # (x, y) world coords
    layer: str = "0"                  # layer name
    style: Style = field(default_factory=Style)  # stroke style


@dataclass
class Polyline:
    """Connected line segments through ``points`` (world coords)."""

    points: List[Point]               # vertices, (x, y) world coords
    closed: bool = False              # True joins the last vertex back to the first
    layer: str = "0"                  # layer name
    style: Style = field(default_factory=Style)  # stroke/fill style


@dataclass
class Rect:
    """Axis-aligned rectangle by one corner, a width and a height (world coords)."""

    corner: Point                     # (x, y) corner, world coords
    width: float                      # extent along +x from ``corner``, world units
    height: float                     # extent along +y from ``corner``, world units
    layer: str = "0"                  # layer name
    style: Style = field(default_factory=Style)  # stroke/fill style

    def as_points(self) -> List[Point]:
        """The four corners, starting at ``corner``, as world-coordinate points."""
        x, y = self.corner
        w, h = self.width, self.height
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


@dataclass
class Polygon:
    """Closed polygon through ``points`` (world coords)."""

    points: List[Point]               # vertices, (x, y) world coords
    layer: str = "0"                  # layer name
    style: Style = field(default_factory=Style)  # stroke/fill style


@dataclass
class Hatch:
    """Filled boundary with a named fill pattern (cement, plug, ...)."""

    boundary: List[Point]             # closed outline, (x, y) world coords
    pattern: str = "cement"           # fill pattern name: cement | plug | solid
    layer: str = "0"                  # layer name
    style: Style = field(default_factory=Style)  # stroke/fill style


@dataclass
class Text:
    """A text label anchored at a world position; sizes are in paper mm."""

    position: Point                   # anchor (x, y), world coords
    text: str                         # label string
    height: float = 2.5               # paper mm
    rotation: float = 0.0             # degrees, ccw
    ha: str = "left"                  # left | center | right
    va: str = "baseline"             # top | center | baseline | bottom
    layer: str = "0"                  # layer name
    style: Style = field(default_factory=Style)  # text colour via ``style.color``


@dataclass
class Symbol:
    """A reusable block definition: named list of local-coordinate entities."""

    name: str                         # key in ``Drawing.symbols``
    entities: List[object] = field(default_factory=list)  # symbol-local coords


@dataclass
class SymbolRef:
    """An instance (INSERT) of a :class:`Symbol` at a world position."""

    name: str                         # name of the placed :class:`Symbol`
    position: Point                   # insertion point (x, y), world coords
    sx: float = 1.0                   # x scale (negative = mirror)
    sy: float = 1.0                   # y scale (negative = mirror)
    rotation: float = 0.0             # degrees, ccw
    layer: str = "0"                  # layer name


# --------------------------------------------------------------------------
# view transform
# --------------------------------------------------------------------------
@dataclass
class ViewTransform:
    """World -> paper-mm map with independent H/V scale.

    ``h_scale`` / ``v_scale`` are paper mm per world x / y unit. ``flip_y``
    sends increasing depth downward (negative paper-y).
    """

    h_scale: float = 1.0              # paper mm per world x unit
    v_scale: float = 1.0              # paper mm per world y unit
    x0: float = 0.0                   # world x mapped to paper x = 0
    y0: float = 0.0                   # world y mapped to paper y = 0
    flip_y: bool = True               # True negates paper y (depth plots downward)

    def apply(self, x: float, y: float) -> Point:
        """Map one world point ``(x, y)`` to paper mm."""
        px = (x - self.x0) * self.h_scale
        py = (y - self.y0) * self.v_scale
        if self.flip_y:
            py = -py
        return (px, py)

    def apply_many(self, pts) -> List[Point]:
        """Map an iterable of world ``(x, y)`` points to paper mm."""
        return [self.apply(x, y) for x, y in pts]


# --------------------------------------------------------------------------
# drawing container
# --------------------------------------------------------------------------
@dataclass
class Drawing:
    """A CAD document: layers, entities, symbol definitions, and a transform."""

    name: str = "well_schematic"     # drawing name
    units: str = "mm"                                  # paper units after transform
    transform: ViewTransform = field(default_factory=ViewTransform)  # world -> paper mm
    layers: Dict[str, Layer] = field(default_factory=dict)  # by name; always has "0"
    entities: List[object] = field(default_factory=list)  # entities in insertion order
    symbols: Dict[str, Symbol] = field(default_factory=dict)  # definitions by name
    title_block: Dict[str, str] = field(default_factory=dict)  # label -> value
    h_unit_label: str = "m"                            # for the horizontal scale bar
    v_unit_label: str = "m"                            # for the vertical scale bar

    def __post_init__(self) -> None:
        self.add_layer("0")

    # --- layers ------------------------------------------------------------
    def add_layer(self, name: str, color: str = "#000000",
                  visible: bool = True) -> Layer:
        """Return layer ``name``, creating it if absent; existing layers unchanged."""
        layer = self.layers.get(name)
        if layer is None:
            layer = Layer(name=name, color=color, visible=visible)
            self.layers[name] = layer
        return layer

    # --- entities ----------------------------------------------------------
    def add(self, entity) -> object:
        """Append ``entity``, creating its layer if needed; returns the entity."""
        layer = getattr(entity, "layer", None)
        if layer is not None:
            self.add_layer(layer)
        self.entities.append(entity)
        return entity

    def extend(self, entities) -> None:
        """:meth:`add` each entity in turn."""
        for e in entities:
            self.add(e)

    # --- symbols -----------------------------------------------------------
    def define_symbol(self, symbol: Symbol) -> Symbol:
        """Register ``symbol`` under its name, replacing any existing definition."""
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
        """Add a :class:`SymbolRef` instance of symbol ``name`` and return it."""
        ref = SymbolRef(name, position, sx, sy, rotation, layer)
        return self.add(ref)

    # --- annotations: title block + scale bars -----------------------------
    def set_title_block(self, **fields) -> None:
        """Merge keyword fields into ``title_block``, values converted to ``str``."""
        self.title_block.update({k: str(v) for k, v in fields.items()})

    def visible_layers(self) -> List[str]:
        """Names of layers whose ``visible`` flag is True."""
        return [n for n, layer in self.layers.items() if layer.visible]

    def bounds(self):
        """World-coordinate bounding box (xmin, ymin, xmax, ymax) of entities."""
        xs: List[float] = []
        ys: List[float] = []

        def acc(pts):
            """Collect the x and y of each point into the running lists."""
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
