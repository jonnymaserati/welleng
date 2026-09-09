"""welleng.schematic
--------------------

A scalable 2D CAD drawing generator for well schematics (plug-and-abandonment
focus). Survey-driven, OSDU-aligned lean JSON input, a renderer-agnostic
drawing model, and multiple export backends (DXF / SVG / PDF / matplotlib).

Pipeline::

    WellSchematic (pydantic input, models.py)
        -> DepthResolver (survey-driven MD<->TVD, depth.py)
        -> build_column / build_section / WellFigure  (generators)
        -> Drawing  (renderer-agnostic model, drawing.py)
        -> to_dxf / to_svg / to_pdf / to_matplotlib  (backends.py)

Requires the ``schematic`` extra (``pydantic``, and ``ezdxf`` for DXF export).
"""
from __future__ import annotations

from .backends import (
    to_dxf,
    to_matplotlib,
    to_pdf,
    to_png,
    to_svg,
)
from .column import build_column
from .depth import DepthResolver, radial_scale_for
from .drawing import (
    Drawing,
    Hatch,
    Layer,
    Line,
    Polygon,
    Polyline,
    RadialScale,
    Rect,
    Style,
    Symbol,
    SymbolRef,
    Text,
    ViewTransform,
)
from .figure import WellFigure
from .models import (
    Casing,
    AnnulusFluid,
    Perforation,
    CementPlug,
    CompletionItem,
    Formation,
    HoleSection,
    PressureProfile,
    SurveyRef,
    Tubular,
    TubularSection,
    Well,
    WellSchematic,
    Wellbore,
)
from .section import build_section
from .symbols import register_standard_symbols
from .tracks import (
    DepthLayout,
    DepthTrack,
    IntervalsTrack,
    LithologyTrack,
    PPFPTrack,
    SchematicTrack,
    Track,
)

__all__ = [
    # models
    "Well", "Wellbore", "WellSchematic", "SurveyRef", "HoleSection",
    "Tubular", "TubularSection", "Casing", "CementPlug", "AnnulusFluid",
    "Perforation", "CompletionItem", "Formation", "PressureProfile",
    # drawing model
    "Drawing", "Layer", "ViewTransform", "RadialScale", "Style", "Symbol",
    "SymbolRef", "Line", "Polyline", "Rect", "Polygon", "Hatch", "Text",
    # depth
    "DepthResolver", "radial_scale_for",
    # generators
    "build_column", "build_section", "WellFigure",
    # tracks
    "Track", "DepthLayout", "DepthTrack", "SchematicTrack", "LithologyTrack",
    "IntervalsTrack", "PPFPTrack",
    # symbols + backends
    "register_standard_symbols",
    "to_dxf", "to_svg", "to_pdf", "to_png", "to_matplotlib",
]
