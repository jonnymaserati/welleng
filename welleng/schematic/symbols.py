"""Reusable symbol factories.

Each factory returns a :class:`~welleng.schematic.drawing.Symbol` -- a named
block of *local-coordinate* entities -- that a backend can realise as a DXF
BLOCK (and INSERT) or flatten. Local coordinates live in a nominal unit box;
callers scale/rotate/translate them via ``Drawing.place_symbol``.

Extend by writing another factory that returns a ``Symbol`` and registering it
in :func:`register_standard_symbols`.

No renderer imports -- pure geometry.
"""
from __future__ import annotations

from typing import Dict

from .drawing import Drawing, Polygon, Rect, Style, Symbol

BLACK = Style(color="#000000", lineweight=0.2, fill="#000000")
_RED = Style(color="#b71c1c", lineweight=0.2, fill="#c62828")
_OPEN = Style(color="#000000", lineweight=0.3, fill=None)

# Symbol names -- import these constants rather than hard-coding strings.
CASING_SHOE = "casing_shoe"
PACKER = "packer"
SCSSV = "scsssv"
NIPPLE = "nipple"


def casing_shoe() -> Symbol:
    """Filled right triangle sitting OUTSIDE the casing wall at the shoe.

    Unit space: origin at the shoe depth on the casing OD, ``+x`` outward
    (mirror with a negative ``sx`` for the other side), ``+y`` deeper. The
    vertical leg runs UP the casing from the shoe, the base is at the shoe
    depth and flares outward, apex uppermost -- i.e. the conventional shoe
    wedge, which reads as belonging to the string rather than straddling it.
    """
    tri = Polygon(points=[(0.0, -1.0), (0.0, 0.0), (1.0, 0.0)], style=BLACK)
    return Symbol(name=CASING_SHOE, entities=[tri])


def packer() -> Symbol:
    """Filled block packer, unit box centred on origin."""
    body = Rect(corner=(-0.5, -0.5), width=1.0, height=1.0, style=BLACK)
    return Symbol(name=PACKER, entities=[body])


def scsssv() -> Symbol:
    """Sub-surface safety valve: red filled triangle (flapper), apex at +y."""
    tri = Polygon(points=[(-0.5, 0.0), (0.5, 0.0), (0.0, 1.0)], style=_RED)
    return Symbol(name=SCSSV, entities=[tri])


def nipple() -> Symbol:
    """Landing nipple: small open box."""
    body = Rect(corner=(-0.5, -0.3), width=1.0, height=0.6, style=_OPEN)
    return Symbol(name=NIPPLE, entities=[body])


_FACTORIES = {
    CASING_SHOE: casing_shoe,
    PACKER: packer,
    SCSSV: scsssv,
    NIPPLE: nipple,
}


def register_standard_symbols(drawing: Drawing) -> Dict[str, Symbol]:
    """Define every standard symbol on ``drawing`` (idempotent)."""
    for name, factory in _FACTORIES.items():
        if name not in drawing.symbols:
            drawing.define_symbol(factory())
    return drawing.symbols
