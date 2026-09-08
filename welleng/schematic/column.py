"""Column-schematic generator.

Produces a symbolic straight-well :class:`Drawing`: a vertical depth axis with
nested casings/liners (steel walls od..id), triangle shoes, cement in the
annuli (toc->shoe), cement plugs (bore fill) and completion symbols. The
horizontal axis is *radius* with a **per-section** exaggeration (see
:class:`RadialScale`): a wall spanning several hole sections is emitted as a
stepped polyline that jogs inward at each boundary. The vertical (depth) axis
is uniform and to-scale.

No renderer imports -- emits :mod:`welleng.schematic.drawing` entities only.
"""
from __future__ import annotations

from typing import List, Optional

from .depth import DepthResolver, radial_scale_for
from .drawing import (
    Drawing,
    Hatch,
    Line,
    Polygon,
    Polyline,
    RadialScale,
    Style,
    Text,
    ViewTransform,
)
from .models import WellSchematic
from .symbols import CASING_SHOE, NIPPLE, SCSSV, register_standard_symbols

# layers
L_GRID = "GRID"
L_HOLE = "HOLE"
L_CASING = "CASING"
L_CEMENT = "CEMENT"
L_FLUID = "FLUID"
L_PLUG = "PLUG"
L_COMPLETION = "COMPLETION"
L_SHOE = "SHOE"
L_ANNOTATION = "ANNOTATION"

# Shoe wedge height as a multiple of its width, and a DIRECT multiple: the
# renderer maps this drawing's world coordinates near-isotropically (measured
# 4.03 px per x-unit against 3.62 px per y-metre, i.e. 1 x-unit ~ 1.1 m on
# paper), so a height in metres set from a width in x-units lands at roughly
# that ratio on the page.
#
# Two earlier attempts got this wrong in both directions by scaling through
# (depth extent / radial extent): that factor is ~3.5 for a typical well and
# has nothing to do with the paper aspect, which gave a 343 m spike one way
# and a flat 0.13-aspect sliver the other. Measure the renderer, do not infer
# it from the data extents.
SHOE_ASPECT = 0.6

# Packer seal height, on the same normalised basis. A real packer element is a
# couple of metres and invisible at well scale, so a schematic exaggerates it --
# but it must still read as an ANNULUS SEAL, not a bar across the well.
PACKER_ASPECT = 0.35

# Valve (SSSV / landing nipple) body height, same normalised basis.
VALVE_ASPECT = 0.30

# styles
_STEEL = Style(color="#222222", lineweight=0.35, fill="#3f3f3f")
_HOLEWALL = Style(color="#b0b0b0", lineweight=0.2)
_CEMENT = Style(color="#8a8a8a", lineweight=0.2, fill="#bdbdbd")
_PLUG = Style(color="#6f6f6f", lineweight=0.25, fill="#a6a6a6")
_TUBING = Style(color="#1565c0", lineweight=0.45)
_GRID = Style(color="#cccccc", lineweight=0.15, linestyle="dotted")
_LABEL = Style(color="#111111", lineweight=0.2)
_CEMENT_LABEL = Style(color="#5c5c5c", lineweight=0.2)
_FLUID_LABEL = Style(color="#37627a", lineweight=0.2)
_PACKER = Style(color="#111111", lineweight=0.25, fill="#1a1a1a")
_SSSV = Style(color="#8e1b1b", lineweight=0.3, fill="#f2dede")
_SSSV_FLAP = Style(color="#8e1b1b", lineweight=0.25, fill="#a52121")
_NIPPLE = Style(color="#333333", lineweight=0.3, fill=None)

# Fallback annulus-fluid fills, keyed on the fluid NAME and used only when the
# fluid carries no explicit colour. Deliberately pale: annulus fluid is
# background to steel and cement and must never out-read a barrier.
_FLUID_FILLS = {
    "brine": "#cfe3f2", "seawater": "#cfe3f2", "water": "#cfe3f2",
    "mud": "#d8cdb4", "obm": "#d3c2a4", "wbm": "#cfd8c0",
    "packer": "#e2d6ef", "diesel": "#efe3c2", "base oil": "#efe3c2",
    "gas": "#f6e2e2", "nitrogen": "#f6e2e2",
}
_FLUID_DEFAULT = "#e6eef3"


def _fluid_fill(f) -> str:
    """Explicit colour wins; else match on the name; else a neutral default."""
    if getattr(f, "colour", None):
        return f.colour
    nm = (getattr(f, "name", "") or "").lower()
    for key, col in _FLUID_FILLS.items():
        if key in nm:
            return col
    return _FLUID_DEFAULT


def _wall(r: float, d_top: float, d_base: float, radial: RadialScale, sign: int) -> List:
    """Stepped world-coordinate points for a wall at physical radius ``r``.

    x = ``sign * r * radial.at(depth)`` (exaggerated inches); the radius steps
    at each section boundary within ``(d_top, d_base)`` producing the jog.
    """
    depths = [d_top] + radial.boundaries_within(d_top, d_base) + [d_base]
    pts: List = []
    for a, b in zip(depths, depths[1:]):
        x = sign * r * radial.at((a + b) / 2.0)
        pts.append((x, a))
        pts.append((x, b))
    return pts


def _band(r_out: float, r_in: float, d_top: float, d_base: float,
          radial: RadialScale, sign: int) -> List:
    """Closed polygon points for an annular band between two radii (one side)."""
    outer = _wall(r_out, d_top, d_base, radial, sign)
    inner = _wall(r_in, d_top, d_base, radial, sign)
    return outer + inner[::-1]


def _annulus_outer_r(md: float, inner, casings, hole) -> float:
    """Outer boundary of ``inner``'s annulus AT ``md``, as a radius in inches.

    The next-outer casing ID where one is actually PRESENT at that depth,
    otherwise the drilled hole. Taking the next-outer casing ID regardless of
    depth overshoots wherever that casing has already ended -- a 20in annulus
    below a 30in conductor shoe is bounded by the 26in HOLE, not the 30in ID,
    and drawing to the ID puts cement and fluid an inch into rock.
    """
    cands = [c.id_in / 2.0 for c in casings
             if c.od_in > inner.od_in and c.top_md <= md <= c.shoe_md]
    cands += [h.bit_in / 2.0 for h in hole if h.top_md <= md <= h.base_md]
    return min(cands) if cands else inner.od_in / 2.0 + 1.0


def _annulus_segments(md_top: float, md_base: float, casings, hole):
    """MD sub-intervals over which the annulus outer boundary is constant."""
    edges = {md_top, md_base}
    for c in casings:
        for m in (c.top_md, c.shoe_md):
            if md_top < m < md_base:
                edges.add(m)
    for h in hole:
        for m in (h.top_md, h.base_md):
            if md_top < m < md_base:
                edges.add(m)
    ordered = sorted(edges)
    return list(zip(ordered, ordered[1:]))


def build_column(
    schematic: WellSchematic,
    mode: str = "MD",
    step: float = 10.0,
    default_radial_scale: float = 40.0,
    resolver: Optional[DepthResolver] = None,
) -> Drawing:
    """Build a column-schematic :class:`Drawing` for the primary bore."""
    bore = schematic.primary
    if resolver is None:
        resolver = DepthResolver(bore.survey, step=step, name=schematic.well.name)
    radial = radial_scale_for(bore, resolver, mode=mode, default=default_radial_scale)

    def d(md):
        return float(resolver.depth(md, mode))

    dwg = Drawing(name=f"{schematic.well.name}_column_{mode}")
    dwg.h_unit_label = "in (exagg.)"
    dwg.v_unit_label = mode + " m"
    for layer in (L_GRID, L_HOLE, L_FLUID, L_CASING, L_CEMENT, L_PLUG,
                  L_COMPLETION, L_SHOE, L_ANNOTATION):
        dwg.add_layer(layer)
    register_standard_symbols(dwg)

    ymax = resolver.max_depth(mode)
    casings = bore.casings
    hole = bore.hole_sections
    max_bit = max((h.bit_in for h in hole), default=30.0)

    # --- open hole walls (per section, naturally stepped) ------------------
    for h in hole:
        s = radial.at((d(h.top_md) + d(h.base_md)) / 2.0)
        r = h.bit_in / 2.0
        for sign in (-1, 1):
            dwg.add(Line((sign * r * s, d(h.top_md)), (sign * r * s, d(h.base_md)),
                         layer=L_HOLE, style=_HOLEWALL))

    # --- annulus fluids (drawn BEFORE cement so cement paints over them) ---
    ordered = sorted(casings, key=lambda c: -c.od_in)
    for f in getattr(bore, "annulus_fluids", []) or []:
        idx = next((i for i, c in enumerate(ordered)
                    if abs(c.od_in - f.inside_od_in) < 1e-6), None)
        if idx is None or f.base_md <= f.top_md:
            continue                      # names an annulus that is not there
        inner = ordered[idx]
        r_in = inner.od_in / 2.0
        fill = Style(color=_fluid_fill(f), lineweight=0.0, fill=_fluid_fill(f))
        for a, b in _annulus_segments(f.top_md, f.base_md, casings, hole):
            r_out = _annulus_outer_r((a + b) / 2.0, inner, casings, hole)
            lo, hi = sorted((r_in, r_out))
            if hi - lo <= 1e-9:
                continue
            for sign in (-1, 1):
                dwg.add(Polygon(_band(hi, lo, d(a), d(b), radial, sign),
                                layer=L_FLUID, style=fill))
        lo, hi = sorted((r_in, _annulus_outer_r(
            (f.top_md + f.base_md) / 2.0, inner, casings, hole)))
        # Label INSIDE its own annulus, rotated. Nested annuli commonly share a
        # top (all open to surface), so labelling at mid-depth outside the
        # string stacks every label at the same place; each annulus has a
        # distinct RADIUS, so that is what separates them.
        y_lbl = d((f.top_md + f.base_md) / 2.0)
        s_lbl = radial.at(y_lbl)
        label = f.name + (f" ({f.density_sg:g} sg)" if f.density_sg else "")
        dwg.add(Text(((lo + hi) / 2.0 * s_lbl, y_lbl), label, height=1.7,
                     rotation=90.0, ha="center", va="center",
                     layer=L_ANNOTATION, style=_FLUID_LABEL))

    # --- cement in annuli (toc -> shoe) ------------------------------------
    for c in ordered:
        r_in = c.od_in / 2.0
        for a, b in _annulus_segments(c.toc_md, c.shoe_md, casings, hole):
            r_out = _annulus_outer_r((a + b) / 2.0, c, casings, hole)
            lo, hi = sorted((r_in, r_out))
            if hi - lo <= 1e-9:
                continue
            for sign in (-1, 1):
                dwg.add(Hatch(_band(hi, lo, d(a), d(b), radial, sign),
                              pattern="solid", layer=L_CEMENT, style=_CEMENT))

    # --- casing steel walls + shoes ----------------------------------------
    # Shoe glyph proportion. Height must be tied to the WIDTH, not to total
    # well depth: width is in exaggerated-radial units and depth in metres, so
    # a depth-fraction height makes the wedge stretch as the well gets deeper.
    # Scaling by (depth extent / radial extent) puts both in the same
    # normalised space, and SHOE_ASPECT then sets the shape once for every
    # string and every well.
    x_extent = max_bit / 2.0 * radial.at(0.0)
    _norm = (ymax / x_extent) if x_extent else 1.0
    for c in casings:
        r_out, r_in = c.od_in / 2.0, c.id_in / 2.0
        for sign in (-1, 1):
            dwg.add(Polygon(_band(r_out, r_in, d(c.top_md), d(c.shoe_md), radial, sign),
                            layer=L_CASING, style=_STEEL))
            # shoe wedge at the setting depth: anchored ON the casing OD and
            # flared OUTWARD (negative sx mirrors it for the left side), so it
            # reads as part of the string rather than straddling the wall.
            # Anchor the shoe on the LAST DRAWN WALL POINT, not on an
            # independent radial.at(shoe_md). _wall() scales each segment by
            # the radial factor at the segment MIDPOINT, whereas the shoe depth
            # sits exactly ON a section boundary and resolves to the next
            # section's factor -- two sources for one position, which opened a
            # visible gap between the shoe and the casing it belongs to.
            x_out = _wall(r_out, d(c.top_md), d(c.shoe_md), radial, sign)[-1][0]
            x_in = _wall(r_in, d(c.top_md), d(c.shoe_md), radial, sign)[-1][0]
            # Width from the DRAWN WALL THICKNESS, not the OD: the OD is
            # multiplied by the radial exaggeration, so an OD-proportional shoe
            # makes a 30in conductor's wedge grotesque while a 7in liner's
            # vanishes. A few wall thicknesses reads consistently at any size.
            shoe_w = max(abs(x_out - x_in) * 2.2, abs(x_out) * 0.04)
            shoe_h = shoe_w * SHOE_ASPECT
            dwg.place_symbol(CASING_SHOE, (x_out, d(c.shoe_md)),
                             sx=sign * shoe_w, sy=shoe_h, layer=L_SHOE)
        s = radial.at(d(c.top_md))
        dwg.add(Text((r_out * s * 1.05, d(c.top_md) + ymax * 0.006), c.name,
                     height=2.0, va="top", layer=L_ANNOTATION, style=_LABEL))

    # --- cement plugs (bore fill) ------------------------------------------
    for p in bore.cement_plugs:
        mid = (p.top_md + p.base_md) / 2.0
        candidates = [c.id_in / 2.0 for c in casings if c.top_md <= mid <= c.shoe_md]
        r_in = min(candidates) if candidates else 3.0
        band = _wall(r_in, d(p.top_md), d(p.base_md), radial, 1) \
            + _wall(r_in, d(p.top_md), d(p.base_md), radial, -1)[::-1]
        dwg.add(Hatch(band, pattern="solid", layer=L_PLUG, style=_PLUG))
        s = radial.at(d(mid))
        dwg.add(Text((-r_in * s * 1.1, d(mid)), p.name, height=2.0,
                     ha="right", va="center", layer=L_ANNOTATION, style=_CEMENT_LABEL))

    # --- completion --------------------------------------------------------
    for item in bore.completion:
        if item.type == "tubing":
            r = item.od_in / 2.0
            for sign in (-1, 1):
                dwg.add(Polyline(_wall(r, d(item.top_md), d(item.base_md), radial, sign),
                                 layer=L_COMPLETION, style=_TUBING))
        else:
            y = d(item.md)
            s = radial.at(y)
            width = item.od_in * s
            if item.type == "packer":
                # A packer SEALS THE ANNULUS: it spans from the tubing OD out
                # to its own sealing OD (the casing ID it sets against), and
                # nothing of it belongs in the bore or beyond the casing wall.
                # It was previously placed CENTRED on its od_in radius with a
                # half-width of the same size, so it reached from half that
                # radius to one and a half times it -- straddling the casing
                # wall and finishing in the cement -- and stood ymax*0.01
                # (tens of metres) tall.
                # A packer can only seal against the string it is SET IN, so
                # the outer radius is clamped to the ID of the innermost casing
                # or liner present at that depth. Trusting item.od_in alone put
                # a packer whose od_in was the 9-5/8in casing ID out past a 7in
                # liner's wall and into the cement, because the liner -- not
                # the casing -- is what it actually sets in down there.
                r_ids = [c.id_in / 2.0 for c in casings
                         if c.top_md <= item.md <= c.shoe_md]
                r_host = min(r_ids) if r_ids else item.od_in / 2.0
                r_seal = min(item.od_in / 2.0, r_host)
                r_tbg = next(
                    (t.od_in / 2.0 for t in bore.completion
                     if t.type == "tubing"
                     and (t.top_md or 0.0) <= item.md <= (t.base_md or item.md)),
                    0.0,
                )
                lo, hi = sorted((r_tbg, r_seal))
                # Height: a seal, not a section of well. Proportional to the
                # annulus it fills, but hard-capped so a wide annulus cannot
                # produce a block tens of metres tall.
                h = min(abs(hi - lo) * s * _norm * PACKER_ASPECT, ymax * 0.008)
                h = max(h, ymax * 0.002)
                for sign in (-1, 1):
                    dwg.add(Polygon(
                        _band(hi, lo, y - h / 2.0, y + h / 2.0, radial, sign),
                        layer=L_COMPLETION, style=_PACKER))
            elif item.type in ("sssv", "nipple"):
                # Mount the valve IN the tubing it belongs to. Previously it was
                # a bare glyph at x=0, ymax*0.012 (tens of metres) tall, sized
                # off its own od_in -- so it floated in the bore and read as an
                # arrow rather than a valve in a string.
                r_tbg = next(
                    (t.od_in / 2.0 for t in bore.completion
                     if t.type == "tubing"
                     and (t.top_md or 0.0) <= item.md <= (t.base_md or item.md)),
                    item.od_in / 2.0,
                )
                w = r_tbg * s
                hv = min(2.0 * w * _norm * VALVE_ASPECT, ymax * 0.006)
                hv = max(hv, ymax * 0.002)
                # valve body: a short outlined section of the string
                dwg.add(Polygon(
                    [(-w, y - hv / 2.0), (w, y - hv / 2.0),
                     (w, y + hv / 2.0), (-w, y + hv / 2.0)],
                    layer=L_COMPLETION,
                    style=_SSSV if item.type == "sssv" else _NIPPLE))
                if item.type == "sssv":
                    # closed flapper: hinged at one wall, swung across the bore
                    dwg.add(Polygon(
                        [(-w, y - hv / 2.0), (w, y + hv * 0.10),
                         (w * 0.72, y + hv / 2.0), (-w, y + hv * 0.10)],
                        layer=L_COMPLETION, style=_SSSV_FLAP))
            if item.name:
                dwg.add(Text((width * 0.6 + 4.0, y), item.name, height=2.0,
                             va="center", layer=L_ANNOTATION, style=_TUBING))

    # --- depth grid + ruler ------------------------------------------------
    _add_depth_ruler(dwg, ymax, radial, max_bit)

    dwg.set_title_block(
        title=schematic.well.name,
        view=f"Column schematic ({mode})",
        radial=_scale_note(radial),
        depth_scale="uniform (to scale)",
    )
    _fit_transform(dwg, target_w=170.0, target_h=250.0, margin=12.0)
    return dwg


def _add_depth_ruler(dwg: Drawing, ymax: float, radial: RadialScale, max_bit: float) -> None:
    """Horizontal depth grid lines + a left-hand depth-label ruler."""
    x_extent = max_bit / 2.0 * radial.at(0.0) * 1.1
    step = _nice_step(ymax)
    depth = 0.0
    while depth <= ymax + 1e-6:
        dwg.add(Line((-x_extent, depth), (x_extent, depth),
                     layer=L_GRID, style=_GRID))
        dwg.add(Text((-x_extent * 1.02, depth), f"{depth:.0f}", height=2.0,
                     ha="right", va="center", layer=L_ANNOTATION, style=_LABEL))
        depth += step


def _nice_step(span: float) -> float:
    raw = span / 8.0
    for s in (50, 100, 250, 500, 1000, 2000):
        if raw <= s:
            return float(s)
    return 5000.0


def _scale_note(radial: RadialScale) -> str:
    lo, hi = min(radial.scales), max(radial.scales)
    return f"x{hi:g}" if lo == hi else f"x{hi:g} (shallow) -> x{lo:g} (deep)"


def _fit_transform(dwg: Drawing, target_w: float, target_h: float, margin: float) -> None:
    """Set an independent-H/V transform that fits the entities to the sheet."""
    xmin, ymin, xmax, ymax = dwg.bounds()
    w = max(xmax - xmin, 1e-6)
    h = max(ymax - ymin, 1e-6)
    dwg.transform = ViewTransform(
        h_scale=(target_w - 2 * margin) / w,
        v_scale=(target_h - 2 * margin) / h,
        x0=(xmin + xmax) / 2.0,
        y0=ymin,
        flip_y=True,
    )
