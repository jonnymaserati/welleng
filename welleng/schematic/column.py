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
from .symbols import CASING_SHOE, register_standard_symbols

# layers
L_GRID = "GRID"
L_HOLE = "HOLE"
L_ROCK = "ROCK"
L_CASING = "CASING"
L_CEMENT = "CEMENT"
L_FLUID = "FLUID"
L_PLUG = "PLUG"
L_COMPLETION = "COMPLETION"
L_SHOE = "SHOE"
L_HANGER = "HANGER"
L_PERF = "PERF"
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

# Liner-hanger block height, as a multiple of the annulus width it spans.
HANGER_ASPECT = 1.25

# styles
_STEEL = Style(color="#222222", lineweight=0.35, fill="#3f3f3f")
_HOLEWALL = Style(color="#b0b0b0", lineweight=0.2)
# Rock outside the hole wall. The formation is BACKGROUND to the well, so its
# colour is blended toward light grey before use: a raw seal or reservoir
# colour (saturated purple, saturated yellow) out-reads the wellbore itself,
# which inverts what the drawing is about.
_ROCK_EDGE = "#9e9e9e"
_ROCK_MUTE = 0.62          # fraction of the way to _ROCK_GROUND
_ROCK_GROUND = (0.96, 0.96, 0.95)


def _mute(colour: str, f: float = _ROCK_MUTE) -> str:
    """Blend a hex colour toward a near-white ground, for background fills."""
    c = colour.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        rgb = tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return colour
    out = tuple(v + (g - v) * f for v, g in zip(rgb, _ROCK_GROUND))
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in out)
_CEMENT = Style(color="#8a8a8a", lineweight=0.2, fill="#bdbdbd")
_PLUG = Style(color="#6f6f6f", lineweight=0.25, fill="#a6a6a6")
_TUBING = Style(color="#1565c0", lineweight=0.45)
_GRID = Style(color="#cccccc", lineweight=0.15, linestyle="dotted")
_LABEL = Style(color="#111111", lineweight=0.2)
_LEADER = Style(color="#9a9a9a", lineweight=0.12)
_CEMENT_LABEL = Style(color="#5c5c5c", lineweight=0.2)
_FLUID_LABEL = Style(color="#37627a", lineweight=0.2)
_PACKER = Style(color="#111111", lineweight=0.25, fill="#1a1a1a")
_SSSV = Style(color="#8e1b1b", lineweight=0.3, fill="#f2dede")
_SSSV_FLAP = Style(color="#8e1b1b", lineweight=0.25, fill="#a52121")
_NIPPLE = Style(color="#333333", lineweight=0.3, fill=None)
_HANGER = Style(color="#1a1a1a", lineweight=0.3, fill="#8c8c8c")
_PERF = Style(color="#111111", lineweight=0.45)

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


def _wall(r: float, d_top: float, d_base: float, radial: RadialScale,
          sign: int) -> List:
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
    r_inner = inner.od_at(md) / 2.0
    cands = [c.id_at(md) / 2.0 for c in casings
             if c.od_at(md) > inner.od_at(md) and c.top_md <= md <= c.shoe_md]
    cands += [h.bit_in / 2.0 for h in hole if h.top_md <= md <= h.base_md]
    return min(cands) if cands else r_inner + 1.0


def _free_slot(y: float, taken, min_gap: float, ymax: float) -> float:
    """Nearest depth to ``y`` clearing every entry in ``taken`` by ``min_gap``.

    Searched OUTWARD in both directions rather than only downward: a label
    whose true depth sits just above an immovable one (a depth-ruler number)
    has to move up, and a one-way nudge can only push it further into the
    thing it is colliding with.
    """
    for k in range(0, 80):
        cands = (y,) if k == 0 else (y - k * min_gap * 0.5,
                                     y + k * min_gap * 0.5)
        for cand in cands:
            if 0.0 <= cand <= ymax and all(
                    abs(t - cand) >= min_gap for t in taken):
                return cand
    return y


def _place_gutter_labels(dwg, requests, x_gutter: float, ymax: float,
                         min_gap_frac: float = 0.022, reserved=()) -> None:
    """Lay labels out in a side gutter with leader lines, de-collided.

    ``requests`` are ``(depth, anchor_x, text, style)``. Labels are placed at
    ``x_gutter`` on the anchor's side, moved to the nearest free depth so none
    overlaps its neighbour, and joined to their feature by a thin leader.

    Placing a label AT its feature does not work on a well schematic: every
    string starts at surface, so top-anchored labels all land on the same depth
    and pile up, and an annulus band is routinely narrower than its own fluid
    name. Separating them in a gutter is what the reference drawings do.

    ``reserved`` are depths on the LEFT already occupied by the depth-ruler
    numbers. They are immovable and share that gutter, so they have to be part
    of the same collision problem -- de-collided among themselves the callouts
    still land on top of the ruler.
    """
    if not requests:
        return
    min_gap = ymax * min_gap_frac
    for side in (1, -1):
        rows = sorted((r for r in requests if (r[1] >= 0) == (side > 0)),
                      key=lambda r: r[0])
        taken: List[float] = [float(y) for y in reserved] if side < 0 else []
        for depth, anchor_x, text, style in rows:
            y = _free_slot(depth, taken, min_gap, ymax)
            taken.append(y)
            gx = side * abs(x_gutter)
            dwg.add(Line((anchor_x, depth), (gx, y), layer=L_ANNOTATION,
                         style=_LEADER))
            dwg.add(Text((gx + side * ymax * 0.001, y), text, height=2.0,
                         ha="left" if side > 0 else "right", va="center",
                         layer=L_ANNOTATION, style=style))


def _annulus_segments(md_top: float, md_base: float, casings, hole):
    """MD sub-intervals over which the annulus outer boundary is constant."""
    edges = {md_top, md_base}
    for c in casings:
        # crossovers included: a combination string's own diameter step moves
        # the annulus wall as surely as another string's shoe does.
        for m in (c.top_md, c.shoe_md, *c.crossovers()):
            if md_top < m < md_base:
                edges.add(m)
    for h in hole:
        for m in (h.top_md, h.base_md):
            if md_top < m < md_base:
                edges.add(m)
    ordered = sorted(edges)
    return list(zip(ordered, ordered[1:]))


def _rock_inner_r(md: float, casings, hole) -> float:
    """Radius (inches) where rock begins at ``md``: the drilled hole, or the
    widest string where no hole section is recorded."""
    holes = [h.bit_in / 2.0 for h in hole if h.top_md <= md <= h.base_md]
    if holes:
        return max(holes)
    ods = [c.od_at(md) / 2.0 for c in casings if c.top_md <= md <= c.shoe_md]
    return max(ods) if ods else 0.0


def _draw_rock(dwg, schematic, casings, hole, radial, d, ymax, max_bit) -> None:
    """Fill formation bands from the hole wall out to the drawing edge.

    The references all show rock against the hole; without it there is no
    visual difference between "cemented annulus" and "formation", which is the
    distinction a reader most needs. Bands step with the hole, so the rock
    boundary follows the wellbore rather than being a straight edge.

    A formation with no colour is left UNFILLED rather than given one -- an
    unidentified interval should look unidentified (same rule as the lithology
    column).
    """
    bands = schematic.formation_bands()
    if not bands:
        return
    # stop short of the label gutter (placed at *1.06) so annotation never
    # sits on top of the rock
    x_edge = max_bit / 2.0 * radial.at(0.0) * 1.0
    for f, top, base in bands:
        colour = (f.color or "").strip()
        if not colour or colour.lower() in ("#ffffff", "white"):
            continue                # unidentified: leave it blank
        style = Style(color=_ROCK_EDGE, lineweight=0.1, fill=_mute(colour))
        for a, b in _annulus_segments(top, base, casings, hole):
            r_in = _rock_inner_r((a + b) / 2.0, casings, hole)
            da, db = d(a), d(b)
            for sign in (-1, 1):
                inner = _wall(r_in, da, db, radial, sign)
                pts = inner + [(sign * x_edge, db), (sign * x_edge, da)]
                dwg.add(Polygon(pts, layer=L_ROCK, style=style))


def _draw_liner_hangers(dwg, casings, radial, d, x_scale_ref) -> None:
    """Hanger block in the annulus at a HUNG string's top (``top_md`` > 0).

    A liner is not a casing that happens to start deep: it hangs off the string
    above, and the hanger is where the load transfers. Without it a liner reads
    as a string that simply begins in mid-air. Box-with-X, matching the
    plumbing view so the two do not disagree.
    """
    for c in casings:
        if c.top_md <= 1e-6:
            continue                        # run from surface: not hung
        hosts = [h.id_at(c.top_md) / 2.0 for h in casings
                 if h.od_at(c.top_md) > c.od_at(c.top_md)
                 and h.top_md <= c.top_md <= h.shoe_md]
        if not hosts:
            continue                        # nothing to hang from
        y = d(c.top_md)
        s = radial.at(y)
        r_out, r_host = c.od_at(c.top_md) / 2.0, min(hosts)
        lo, hi = sorted((r_out, r_host))
        # square-ish on paper: height in metres ~ the annulus width in x-units
        h = max(abs(hi - lo) * s * HANGER_ASPECT, 1e-6)
        for sign in (-1, 1):
            x0, x1 = sign * lo * s, sign * hi * s
            y0, y1 = y, y + h
            dwg.add(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                            layer=L_HANGER, style=_HANGER))
            dwg.add(Line((x0, y0), (x1, y1), layer=L_HANGER, style=_HANGER))
            dwg.add(Line((x0, y1), (x1, y0), layer=L_HANGER, style=_HANGER))


def _draw_perforations(dwg, bore, casings, hole, radial, d) -> None:
    """Perforation marks crossing the shot casing wall into the formation.

    Drawn as a ladder of ticks rather than a filled band: perforations are
    discrete holes through the wall, and a band would read as an interval of
    missing casing.
    """
    for pf in getattr(bore, "perforations", []) or []:
        if pf.base_md <= pf.top_md:
            continue
        mid = (pf.top_md + pf.base_md) / 2.0
        present = [c for c in casings if c.top_md <= mid <= c.shoe_md]
        if pf.casing_od_in is not None:
            shot = next((c for c in present if c.has_od(pf.casing_od_in)), None)
        else:
            shot = min(present, key=lambda c: c.od_at(mid)) if present else None
        if shot is None:
            continue                        # names a string that is not there
        # tick spacing from the interval, capped so a long zone stays legible
        n = max(2, min(int((pf.base_md - pf.top_md) / 8.0), 40))
        for k in range(n + 1):
            md = pf.top_md + (pf.base_md - pf.top_md) * k / n
            y = d(md)
            s = radial.at(y)
            r_in = shot.id_at(md) / 2.0
            r_far = _annulus_outer_r(md, shot, casings, hole)
            for sign in (-1, 1):
                dwg.add(Line((sign * r_in * s, y), (sign * r_far * s, y),
                             layer=L_PERF, style=_PERF))


def build_column(
    schematic: WellSchematic,
    mode: str = "MD",
    step: float = 10.0,
    default_radial_scale: float = 40.0,
    resolver: Optional[DepthResolver] = None,
    bare: bool = False,
    rock: bool = True,
) -> Drawing:
    """Build a column-schematic :class:`Drawing` for the primary bore.

    ``bare=True`` omits the sheet furniture -- gutter callouts, depth ruler,
    title block and the fit-to-sheet transform -- and leaves the geometry in
    world coordinates (exaggerated inch-radius by depth). That is what lets
    :class:`~welleng.schematic.tracks.SchematicTrack` render THROUGH this
    function instead of reimplementing it: one renderer, one set of
    conventions, and the composite figure inherits every correction made here
    without anyone having to remember to port it.

    ``rock=False`` omits the formation background. It is context for a view
    that stands alone; in a composite that already carries a lithology track it
    is a SECOND lithology column beside the first, and two columns of rock
    colour side by side invite the reader to reconcile them.
    """
    bore = schematic.primary
    if resolver is None:
        resolver = DepthResolver(bore.survey, step=step, name=schematic.well.name)
    radial = radial_scale_for(bore, resolver, mode=mode, default=default_radial_scale)

    def d(md):
        return float(resolver.depth(md, mode))

    dwg = Drawing(name=f"{schematic.well.name}_column_{mode}")
    dwg.h_unit_label = "in (exagg.)"
    dwg.v_unit_label = mode + " m"
    for layer in (L_GRID, L_ROCK, L_HOLE, L_FLUID, L_CASING, L_CEMENT, L_PLUG,
                  L_COMPLETION, L_SHOE, L_HANGER, L_PERF, L_ANNOTATION):
        dwg.add_layer(layer)
    register_standard_symbols(dwg)

    ymax = resolver.max_depth(mode)
    labels: List = []            # (depth, anchor_x, text, style) -> gutter
    casings = bore.casings
    hole = bore.hole_sections
    # Reference width for the drawing edge, ruler and gutter. Casing ODs count,
    # not just bit sizes: a DRIVEN conductor is never drilled, so its OD IS the
    # widest geometry in the well and a bit-only extent clips it.
    max_bit = max([h.bit_in for h in hole] + [c.od_in for c in casings]
                  + [30.0])

    # --- formation (rock) OUTSIDE the hole wall, drawn first ---------------
    if rock:
        _draw_rock(dwg, schematic, casings, hole, radial, d, ymax, max_bit)

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
        # matched on ANY of the string's diameters: a combination string
        # presents more than one wall, and the caller names a wall.
        inner = next((c for c in ordered if c.has_od(f.inside_od_in)), None)
        if inner is None or f.base_md <= f.top_md:
            continue                      # names an annulus that is not there
        fill = Style(color=_fluid_fill(f), lineweight=0.0, fill=_fluid_fill(f))
        for a, b in _annulus_segments(f.top_md, f.base_md, casings, hole):
            mid_seg = (a + b) / 2.0
            r_in = inner.od_at(mid_seg) / 2.0
            r_out = _annulus_outer_r(mid_seg, inner, casings, hole)
            lo, hi = sorted((r_in, r_out))
            if hi - lo <= 1e-9:
                continue
            for sign in (-1, 1):
                dwg.add(Polygon(_band(hi, lo, d(a), d(b), radial, sign),
                                layer=L_FLUID, style=fill))
        f_mid = (f.top_md + f.base_md) / 2.0
        lo, hi = sorted((inner.od_at(f_mid) / 2.0,
                         _annulus_outer_r(f_mid, inner, casings, hole)))
        # Label INSIDE its own annulus, rotated. Nested annuli commonly share a
        # top (all open to surface), so labelling at mid-depth outside the
        # string stacks every label at the same place; each annulus has a
        # distinct RADIUS, so that is what separates them.
        y_lbl = d((f.top_md + f.base_md) / 2.0)
        s_lbl = radial.at(y_lbl)
        label = f.name + (f" ({f.density_sg:g} sg)" if f.density_sg else "")
        labels.append((y_lbl, (lo + hi) / 2.0 * s_lbl, label, _FLUID_LABEL))

    # --- cement in annuli (toc -> shoe) ------------------------------------
    for c in ordered:
        if c.toc_md is None:
            continue          # no cement RECORDED: draw nothing, claim nothing
        for a, b in _annulus_segments(c.toc_md, c.shoe_md, casings, hole):
            mid_seg = (a + b) / 2.0
            r_in = c.od_at(mid_seg) / 2.0
            r_out = _annulus_outer_r(mid_seg, c, casings, hole)
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
        # Steel per DIAMETER, one shoe per STRING. A combination string is one
        # string on one hanger; drawing a shoe at each diameter change asserts
        # the string ends there and the annulus opens below it, both false.
        prof = c.profile()
        for seg_top, seg_base, seg_od, seg_id in prof:
            for sign in (-1, 1):
                dwg.add(Polygon(
                    _band(seg_od / 2.0, seg_id / 2.0,
                          d(seg_top), d(seg_base), radial, sign),
                    layer=L_CASING, style=_STEEL))
        shoe_top, _shoe_base, shoe_od, shoe_id = prof[-1]
        r_out, r_in = shoe_od / 2.0, shoe_id / 2.0
        if not getattr(c, "has_shoe", True):
            # Screens and junk terminate; they have no shoe, and a shoe is a
            # BARRIER-RELEVANT symbol -- drawing one asserts a barrier that is
            # not there. Label off the string end instead.
            s_end = radial.at(d(c.shoe_md))
            labels.append((d(c.shoe_md), r_out * s_end, c.name, _LABEL))
            continue
        for sign in (-1, 1):
            # shoe wedge at the setting depth: anchored ON the casing OD and
            # flared OUTWARD (negative sx mirrors it for the left side), so it
            # reads as part of the string rather than straddling the wall.
            # Anchor the shoe on the LAST DRAWN WALL POINT, not on an
            # independent radial.at(shoe_md). _wall() scales each segment by
            # the radial factor at the segment MIDPOINT, whereas the shoe depth
            # sits exactly ON a section boundary and resolves to the next
            # section's factor -- two sources for one position, which opened a
            # visible gap between the shoe and the casing it belongs to.
            x_out = _wall(r_out, d(shoe_top), d(c.shoe_md), radial, sign)[-1][0]
            x_in = _wall(r_in, d(shoe_top), d(c.shoe_md), radial, sign)[-1][0]
            # Width from the DRAWN WALL THICKNESS, not the OD: the OD is
            # multiplied by the radial exaggeration, so an OD-proportional shoe
            # makes a 30in conductor's wedge grotesque while a 7in liner's
            # vanishes. A few wall thicknesses reads consistently at any size.
            shoe_w = max(abs(x_out - x_in) * 2.2, abs(x_out) * 0.04)
            # Clamped to the room between the string and the label gutter. The
            # WIDEST string sits ON the drawing edge (a driven conductor is at
            # the edge by definition), so an unclamped wedge grew straight past
            # the rock and printed over the string's own name.
            room = x_extent * 1.06 - abs(x_out)
            if room > 0.0:
                shoe_w = min(shoe_w, room)
            shoe_h = shoe_w * SHOE_ASPECT
            dwg.place_symbol(CASING_SHOE, (x_out, d(c.shoe_md)),
                             sx=sign * shoe_w, sy=shoe_h, layer=L_SHOE)
        # anchor on the SHOE, not the top: every string starts at surface, so
        # top-anchored names all collide at depth 0.
        s_shoe = radial.at(d(c.shoe_md))
        labels.append((d(c.shoe_md), r_out * s_shoe, c.name, _LABEL))

    # --- cement plugs (bore fill) ------------------------------------------
    for p in bore.cement_plugs:
        mid = (p.top_md + p.base_md) / 2.0
        candidates = [c.id_at(mid) / 2.0 for c in casings
                      if c.top_md <= mid <= c.shoe_md]
        r_in = min(candidates) if candidates else 3.0
        band = _wall(r_in, d(p.top_md), d(p.base_md), radial, 1) \
            + _wall(r_in, d(p.top_md), d(p.base_md), radial, -1)[::-1]
        dwg.add(Hatch(band, pattern="solid", layer=L_PLUG, style=_PLUG))
        s = radial.at(d(mid))
        labels.append((d(mid), -r_in * s, p.name, _CEMENT_LABEL))

    # --- completion --------------------------------------------------------
    # Tubing runs whose depths meet are ONE string. Drawn as independent
    # polylines at their own radius, a tapered string (5-1/2in, then 4-1/2in,
    # then an ESP section) rendered as pairs of floating lines with visible
    # gaps, which reads as a discontinuity in the completion. The runs already
    # share depths, so the step between the two radii is drawn explicitly.
    tubing = sorted((t for t in bore.completion if t.type == "tubing"),
                    key=lambda t: (t.top_md or 0.0))
    for i, item in enumerate(tubing):
        r = item.od_in / 2.0
        for sign in (-1, 1):
            dwg.add(Polyline(_wall(r, d(item.top_md), d(item.base_md),
                                   radial, sign),
                             layer=L_COMPLETION, style=_TUBING))
        nxt = tubing[i + 1] if i + 1 < len(tubing) else None
        if nxt is None or item.base_md is None:
            continue
        if abs((nxt.top_md or 0.0) - item.base_md) > 1e-6:
            continue                        # a real gap: do not close it
        y = d(item.base_md)
        s = radial.at(y)
        r_next = nxt.od_in / 2.0
        if abs(r_next - r) <= 1e-9:
            continue                        # same size: nothing to step
        for sign in (-1, 1):
            dwg.add(Polyline([(sign * r * s, y), (sign * r_next * s, y)],
                             layer=L_COMPLETION, style=_TUBING))

    for item in bore.completion:
        if item.type != "tubing":
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
                r_ids = [c.id_at(item.md) / 2.0 for c in casings
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
                labels.append((y, width * 0.5, item.name, _TUBING))

    # --- liner hangers + perforations --------------------------------------
    # Drawn LAST of the geometry: both sit in the annulus or across a wall,
    # so anything filling that space (annulus cement, a plug, a packer) must
    # already be down or it paints them out. Insertion order is the z-order
    # in every backend, so it has to agree with the layer list above.
    _draw_liner_hangers(dwg, casings, radial, d, max_bit)
    _draw_perforations(dwg, bore, casings, hole, radial, d)

    # --- annotations in side gutters, de-collided --------------------------
    if bare:
        return dwg

    # One source for the ruler depths: the callout de-collider has to know
    # where the depth numbers are, and the ruler has to draw them there.
    ruler_depths = _ruler_depths(ymax)
    _place_gutter_labels(dwg, labels,
                         x_gutter=max_bit / 2.0 * radial.at(0.0) * 1.06,
                         ymax=ymax, reserved=ruler_depths)

    # --- depth grid + ruler ------------------------------------------------
    _add_depth_ruler(dwg, ruler_depths, radial, max_bit)

    dwg.set_title_block(
        title=schematic.well.name,
        view=f"Column schematic ({mode})",
        radial=_scale_note(radial),
        depth_scale="uniform (to scale)",
    )
    _fit_transform(dwg, target_w=170.0, target_h=250.0, margin=12.0)
    return dwg


def _ruler_depths(ymax: float) -> List[float]:
    """Depths carrying a grid line and a ruler number."""
    step = _nice_step(ymax)
    out, depth = [], 0.0
    while depth <= ymax + 1e-6:
        out.append(depth)
        depth += step
    return out


def _add_depth_ruler(dwg: Drawing, depths, radial: RadialScale,
                     max_bit: float) -> None:
    """Horizontal depth grid lines + a left-hand depth-label ruler."""
    x_extent = max_bit / 2.0 * radial.at(0.0) * 1.1
    for depth in depths:
        dwg.add(Line((-x_extent, depth), (x_extent, depth),
                     layer=L_GRID, style=_GRID))
        dwg.add(Text((-x_extent * 1.02, depth), f"{depth:.0f}", height=2.0,
                     ha="right", va="center", layer=L_ANNOTATION, style=_LABEL))


def _nice_step(span: float) -> float:
    raw = span / 8.0
    for s in (50, 100, 250, 500, 1000, 2000):
        if raw <= s:
            return float(s)
    return 5000.0


def _scale_note(radial: RadialScale) -> str:
    lo, hi = min(radial.scales), max(radial.scales)
    return f"x{hi:g}" if lo == hi else f"x{hi:g} (shallow) -> x{lo:g} (deep)"


def _fit_transform(dwg: Drawing, target_w: float, target_h: float,
                   margin: float) -> None:
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
