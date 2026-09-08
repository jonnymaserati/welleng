"""Export backends: one :class:`Drawing` -> DXF / SVG / PDF / matplotlib.

This is the ONLY module that imports rendering libraries, and it does so
lazily (inside each function) so that importing :mod:`welleng.schematic`
never requires ``ezdxf`` or ``matplotlib``. The drawing model stays fully
renderer-agnostic.

The transform is applied here: every entity's world coordinates are mapped to
paper millimetres via ``drawing.transform``; symbol references are flattened
(non-DXF backends) or realised as blocks (DXF).
"""
from __future__ import annotations

import math
from typing import List

from .drawing import (
    Drawing,
    Hatch,
    Line,
    Polygon,
    Polyline,
    Rect,
    Style,
    SymbolRef,
    Text,
)

_MM_PER_PT = 2.834645669  # points per mm (for matplotlib sizing)

_PATTERN_MPL = {"cement": "xxx", "plug": "....", "solid": None}
_PATTERN_DXF = {"cement": "ANSI37", "plug": "ANSI31", "solid": "SOLID"}
_LS_MPL = {"solid": "-", "dashed": "--", "dotted": ":"}


# --------------------------------------------------------------------------
# flattening (shared by matplotlib + svg)
# --------------------------------------------------------------------------
def _hex_rgb(color: str):
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _shape(e):
    """Return (points, closed, filled, style) for a simple shape, else None."""
    if isinstance(e, Line):
        return [e.start, e.end], False, False, e.style
    if isinstance(e, Polyline):
        return list(e.points), e.closed, False, e.style
    if isinstance(e, Rect):
        return e.as_points(), True, e.style.fill is not None, e.style
    if isinstance(e, Polygon):
        return list(e.points), True, e.style.fill is not None, e.style
    return None


def _place(pt, ref: SymbolRef, transform):
    lx, ly = pt
    x, y = lx * ref.sx, ly * ref.sy
    th = math.radians(ref.rotation)
    c, s = math.cos(th), math.sin(th)
    wx = ref.position[0] + (x * c - y * s)
    wy = ref.position[1] + (x * s + y * c)
    return transform.apply(wx, wy)


def _flatten(drawing: Drawing) -> List[dict]:
    """Flatten entities to paper-mm primitives: poly / hatch / text dicts."""
    T = drawing.transform
    out: List[dict] = []
    for e in drawing.entities:
        if isinstance(e, SymbolRef):
            sym = drawing.symbols.get(e.name)
            if sym is None:
                continue
            for se in sym.entities:
                shp = _shape(se)
                if shp is None:
                    continue
                pts, closed, filled, style = shp
                out.append(dict(kind="poly", pts=[_place(p, e, T) for p in pts],
                                closed=closed, filled=filled, style=style,
                                layer=e.layer))
        elif isinstance(e, Text):
            out.append(dict(kind="text", pos=T.apply(*e.position), text=e.text,
                            height=e.height, rotation=e.rotation, ha=e.ha,
                            va=e.va, style=e.style, layer=e.layer))
        elif isinstance(e, Hatch):
            out.append(dict(kind="hatch", pts=T.apply_many(e.boundary),
                            pattern=e.pattern, style=e.style, layer=e.layer))
        else:
            shp = _shape(e)
            if shp is None:
                continue
            pts, closed, filled, style = shp
            out.append(dict(kind="poly", pts=T.apply_many(pts), closed=closed,
                            filled=filled, style=style, layer=e.layer))
    return out


def _bbox(prims):
    xs, ys = [], []
    for pr in prims:
        if pr["kind"] in ("poly", "hatch"):
            for x, y in pr["pts"]:
                xs.append(x)
                ys.append(y)
        else:
            xs.append(pr["pos"][0])
            ys.append(pr["pos"][1])
    if not xs:
        return (0.0, 0.0, 1.0, 1.0)
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------
# matplotlib (preview) + pdf/png via matplotlib
# --------------------------------------------------------------------------
def to_matplotlib(drawing: Drawing, ax=None):
    """Render onto a matplotlib Axes (creating a Figure if needed). Returns fig."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8.27, 11.69))
    fig = ax.figure

    for pr in _flatten(drawing):
        st: Style = pr["style"]
        if pr["kind"] == "text":
            x, y = pr["pos"]
            ax.text(x, y, pr["text"], fontsize=pr["height"] * _MM_PER_PT,
                    rotation=pr["rotation"], ha=pr["ha"],
                    va=("baseline" if pr["va"] == "baseline" else pr["va"]),
                    color=st.color, zorder=6)
            continue
        xs = [p[0] for p in pr["pts"]]
        ys = [p[1] for p in pr["pts"]]
        lw = max(st.lineweight, 0.05) * _MM_PER_PT
        if pr["kind"] == "hatch":
            ax.fill(xs, ys, facecolor=st.fill or "none", edgecolor=st.color,
                    hatch=_PATTERN_MPL.get(pr["pattern"]), linewidth=lw,
                    alpha=st.opacity, zorder=3)
        elif pr["filled"]:
            ax.fill(xs, ys, facecolor=st.fill, edgecolor=st.color,
                    linewidth=lw, alpha=st.opacity, zorder=4)
        else:
            if pr["closed"]:
                xs, ys = xs + xs[:1], ys + ys[:1]
            ax.plot(xs, ys, color=st.color, linewidth=lw,
                    linestyle=_LS_MPL.get(st.linestyle, "-"), zorder=5)

    ax.set_aspect("equal")
    ax.axis("off")
    _mpl_title_block(ax, drawing)
    return fig


def _mpl_title_block(ax, drawing: Drawing) -> None:
    if not drawing.title_block:
        return
    lines = [f"{k}: {v}" for k, v in drawing.title_block.items()]
    ax.text(0.01, 0.005, "   ".join(lines), transform=ax.transAxes,
            fontsize=6, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", lw=0.5))


def to_pdf(drawing: Drawing, path: str, sheet: str = "A4") -> str:
    """Render to a PDF sheet (A4/A3) with the title block."""
    import matplotlib.pyplot as plt

    sizes = {"A4": (8.27, 11.69), "A3": (11.69, 16.54)}
    fig, ax = plt.subplots(figsize=sizes.get(sheet.upper(), sizes["A4"]))
    to_matplotlib(drawing, ax=ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def to_png(drawing: Drawing, path: str, dpi: int = 150) -> str:
    import matplotlib.pyplot as plt

    fig = to_matplotlib(drawing)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# native SVG
# --------------------------------------------------------------------------
def to_svg(drawing: Drawing, path: str, margin: float = 8.0) -> str:
    """Native SVG writer (no matplotlib). Units are millimetres."""
    prims = _flatten(drawing)
    xmin, ymin, xmax, ymax = _bbox(prims)
    # SVG y increases downward; our paper y is negative-for-deeper, so flip.
    def sx(x):
        return x - xmin + margin

    def sy(y):
        return (-y) - (-ymax) + margin

    w = (xmax - xmin) + 2 * margin
    h = (ymax - ymin) + 2 * margin
    body: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.1f}mm" '
        f'height="{h:.1f}mm" viewBox="0 0 {w:.1f} {h:.1f}">',
        _svg_defs(),
        f'<rect x="0" y="0" width="{w:.1f}" height="{h:.1f}" fill="white"/>',
    ]
    for pr in prims:
        st: Style = pr["style"]
        if pr["kind"] == "text":
            x, y = sx(pr["pos"][0]), sy(pr["pos"][1])
            anchor = {"left": "start", "center": "middle",
                      "right": "end"}.get(pr["ha"], "start")
            rot = (f' transform="rotate({-pr["rotation"]:.2f} '
                   f'{x:.2f} {y:.2f})"') if pr["rotation"] else ""
            body.append(
                f'<text x="{x:.2f}" y="{y:.2f}" font-size="{pr["height"]:.2f}" '
                f'text-anchor="{anchor}" fill="{st.color}"{rot}>'
                f'{_esc(pr["text"])}</text>'
            )
            continue
        pts = " ".join(f"{sx(px):.2f},{sy(py):.2f}" for px, py in pr["pts"])
        lw = max(st.lineweight, 0.05)
        if pr["kind"] == "hatch":
            fill = f"url(#pat_{pr['pattern']})"
            body.append(f'<polygon points="{pts}" fill="{fill}" stroke="{st.color}" '
                        f'stroke-width="{lw:.2f}"/>')
        elif pr["filled"]:
            body.append(f'<polygon points="{pts}" fill="{st.fill}" stroke="{st.color}" '
                        f'stroke-width="{lw:.2f}" fill-opacity="{st.opacity:.2f}"/>')
        else:
            dash = ' stroke-dasharray="3,2"' if st.linestyle == "dashed" else (
                ' stroke-dasharray="1,2"' if st.linestyle == "dotted" else "")
            tag = "polygon" if pr["closed"] else "polyline"
            body.append(f'<{tag} points="{pts}" fill="none" stroke="{st.color}" '
                        f'stroke-width="{lw:.2f}"{dash}/>')
    _svg_title_block(body, drawing, w, h)
    body.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(body))
    return path


def _svg_defs() -> str:
    return (
        '<defs>'
        '<pattern id="pat_cement" width="3" height="3" patternUnits="userSpaceOnUse">'
        '<path d="M0,3 L3,0 M-1,1 L1,-1 M2,4 L4,2" '
        'stroke="#7a6f4a" stroke-width="0.3"/>'
        '<rect width="3" height="3" fill="#d8cfae" fill-opacity="0.5"/></pattern>'
        '<pattern id="pat_plug" width="2.5" height="2.5" patternUnits="userSpaceOnUse">'
        '<rect width="2.5" height="2.5" fill="#cdbf94"/>'
        '<circle cx="1.25" cy="1.25" r="0.3" fill="#6b5d2f"/></pattern>'
        '</defs>'
    )


def _svg_title_block(body: List[str], drawing: Drawing, w: float, h: float) -> None:
    if not drawing.title_block:
        return
    lines = [f"{k}: {v}" for k, v in drawing.title_block.items()]
    y = h - 2.0 - 3.0 * (len(lines) - 1)
    body.append(f'<rect x="1" y="{y - 3:.1f}" width="{min(w - 2, 90):.1f}" '
                f'height="{3 * len(lines) + 2:.1f}" fill="white" '
                f'stroke="#888" stroke-width="0.2"/>')
    for line in lines:
        body.append(
            f'<text x="2.5" y="{y:.1f}" font-size="2.2" '
            f'fill="#111">{_esc(line)}</text>'
        )
        y += 3.0


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("\n", " "))


# --------------------------------------------------------------------------
# DXF (ezdxf) -- layers -> DXF layers, symbols -> BLOCKs, hatch -> DXF hatch
# --------------------------------------------------------------------------
def to_dxf(drawing: Drawing, path: str) -> str:
    """Export to DXF. Requires the ``schematic`` extra (``ezdxf``)."""
    try:
        import ezdxf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "to_dxf requires ezdxf -- install the schematic extra: "
            "pip install 'welleng[schematic]'"
        ) from exc

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    T = drawing.transform

    # layers
    for name, layer in drawing.layers.items():
        if name == "0":
            continue
        dl = doc.layers.add(name)
        try:
            dl.rgb = _hex_rgb(layer.color)
        except Exception:
            pass

    # symbol blocks (local coordinates)
    for name, sym in drawing.symbols.items():
        blk = doc.blocks.new(name=name)
        for se in sym.entities:
            shp = _shape(se)
            if shp is None:
                continue
            pts, closed, filled, style = shp
            _dxf_shape(blk, pts, closed, filled, style, "0")

    # entities
    for e in drawing.entities:
        if isinstance(e, SymbolRef):
            insert = T.apply(*e.position)
            ys = e.sy * T.v_scale * (-1 if T.flip_y else 1)
            msp.add_blockref(e.name, insert, dxfattribs={
                "layer": e.layer,
                "xscale": e.sx * T.h_scale,
                "yscale": ys,
                "rotation": e.rotation,
            })
        elif isinstance(e, Text):
            _dxf_text(msp, T.apply(*e.position), e)
        elif isinstance(e, Hatch):
            pts = T.apply_many(e.boundary)
            outline = msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": e.layer})
            _set_rgb(outline, e.style.color)
            hatch = msp.add_hatch(dxfattribs={"layer": e.layer})
            _set_rgb(hatch, e.style.fill or e.style.color)
            hatch.set_pattern_fill(_PATTERN_DXF.get(e.pattern, "ANSI31"), scale=2.0)
            hatch.paths.add_polyline_path(pts, is_closed=True)
        else:
            shp = _shape(e)
            if shp is None:
                continue
            pts, closed, filled, style = shp
            _dxf_shape(msp, T.apply_many(pts), closed, filled, style, e.layer)

    doc.saveas(path)
    return path


def _dxf_shape(target, pts, closed, filled, style: Style, layer: str) -> None:
    pl = target.add_lwpolyline(pts, close=(closed or filled),
                               dxfattribs={"layer": layer})
    _set_rgb(pl, style.color)
    if filled and style.fill:
        hatch = target.add_hatch(dxfattribs={"layer": layer})
        _set_rgb(hatch, style.fill)
        hatch.paths.add_polyline_path(pts, is_closed=True)


def _dxf_text(msp, pos, e: Text) -> None:
    attach = _ATTACH.get((e.va, e.ha), 5)
    m = msp.add_mtext(e.text, dxfattribs={"layer": e.layer, "char_height": e.height})
    m.dxf.attachment_point = attach
    m.set_location(insert=pos, rotation=e.rotation)
    _set_rgb(m, e.style.color)


# MTEXT attachment points: (va, ha) -> code 1..9
_ATTACH = {
    ("top", "left"): 1, ("top", "center"): 2, ("top", "right"): 3,
    ("center", "left"): 4, ("center", "center"): 5, ("center", "right"): 6,
    ("bottom", "left"): 7, ("bottom", "center"): 8, ("bottom", "right"): 9,
    ("baseline", "left"): 7, ("baseline", "center"): 8, ("baseline", "right"): 9,
}


def _set_rgb(entity, color: str) -> None:
    try:
        entity.rgb = _hex_rgb(color)
    except Exception:  # pragma: no cover
        pass
