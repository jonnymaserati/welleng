"""Plumbing / system-P&A view: a multi-bore well drawn as the COMBINATION of
its wellpaths (parent + sidetrack(s)) so a whole connected system can be shown
verifiably abandoned -- not each wellpath in isolation.

The insight: a plumbing diagram is just N wellpaths that share geometry above
their divergence. Each :class:`~welleng.schematic.models.Wellbore` carries its
own trajectory + tubulars + plugs
a child bore coincides with its parent above
``kickoff_md`` and diverges below. Nothing here is bespoke to a particular well
-- the whole scene is derived from a :class:`~welleng.schematic.models.WellSchematic`
(i.e. from the OSDU-lean JSON schema).

Geometry primitive (transferred from the validated prototype, now generic):

* Every bore's centreline comes from its **survey** via minimum curvature,
  projected onto a single **vertical-section azimuth** so all bores lie in one
  plane: ``x = DepthResolver.vs(md, vs_azi)``, ``y = depth(md)`` (MD or TVD).
* Drawing is done in ONE **equal-aspect** (x, y) metre space, so a perpendicular
  offset is geometrically true everywhere including round the build -- radii are
  exaggerated (``exag``) to be visible, departure is real.
* :class:`Centreline` exposes ``perp(md, r, side)`` = centreline offset ``r``
  along the TRUE normal
  ALL geometry (casing walls, hole wall, annular
  cement, shoe tri, plug, liner hanger) is built from it. One ``perp``-swept
  polygon draws a straight+curved element in a single go -- never a box laid
  over a fill, never a double-draw.

This renderer is intentionally matplotlib-direct (like the composite tracks)
folding it onto the renderer-agnostic :mod:`~welleng.schematic.drawing` model is
a later step. Assumes each ``Wellbore.survey`` is a full tie-on from surface
(absolute N/E), so a child's vertical section is continuous with its parent's.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .depth import DepthResolver
from .models import Casing, Wellbore, WellSchematic

_IN2M = 0.0254


# --------------------------------------------------------------------------
# centreline: real survey -> equal-aspect (x, y) metre space + true normal
# --------------------------------------------------------------------------
class Centreline:
    """A bore's centreline in the shared vertical-section plane.

    ``x`` = vertical-section departure (m) on ``vs_azi``
    ``y`` = plotting depth
    (m) in ``mode``. ``exag`` multiplies tubular RADII only (departure is real),
    so nested strings are visible
    equal aspect keeps every offset perpendicular.
    """

    def __init__(
        self,
        resolver: DepthResolver,
        vs_azi: float,
        mode: str = "MD",
        exag: float = 45.0,
        step: float = 3.0,
    ) -> None:
        self.res = resolver
        self.vs_azi = float(vs_azi)
        self.mode = mode
        self.exag = float(exag)
        md0, md1 = float(resolver.md[0]), float(resolver.md[-1])
        self.grid = np.arange(md0, md1 + step, step)
        self._x = resolver.vs(self.grid, vs_azi)
        self._y = resolver.depth(self.grid, mode)

    def x(self, md):
        return np.interp(md, self.grid, self._x)

    def y(self, md):
        return np.interp(md, self.grid, self._y)

    def rdraw(self, r_in: float) -> float:
        """Inches of true radius -> exaggerated metres in the drawing."""
        return r_in * _IN2M * self.exag

    def _tangent(self, md):
        md = np.atleast_1d(np.asarray(md, float))
        if md.size == 1:
            e = 1.0
            dx = self.x(md + e) - self.x(md - e)
            dy = self.y(md + e) - self.y(md - e)
        else:
            dx = np.gradient(self.x(md))
            dy = np.gradient(self.y(md))
        return np.atleast_1d(dx), np.atleast_1d(dy)

    def ndir(self, md):
        """Unit normal (perpendicular) in equal-aspect space."""
        dx, dy = self._tangent(md)
        L = np.hypot(dx, dy)
        L[L == 0] = 1.0
        return dy / L, -dx / L

    def tdir(self, md):
        """Unit down-hole tangent."""
        dx, dy = self._tangent(md)
        L = np.hypot(dx, dy)
        L[L == 0] = 1.0
        return dx / L, dy / L

    def perp(self, md, r_draw, side):
        """Centreline offset ``side*r_draw`` (metres) along the true normal."""
        md = np.atleast_1d(np.asarray(md, float))
        nx, ny = self.ndir(md)
        return self.x(md) + side * r_draw * nx, self.y(md) + side * r_draw * ny


# --------------------------------------------------------------------------
# telescoping-radius helpers (generic over a bore + its ancestors' strings)
# --------------------------------------------------------------------------
def _hole_r_in(md: float, bore: Wellbore) -> Optional[float]:
    for s in bore.hole_sections:
        if s.top_md <= md <= s.base_md:
            return s.bit_in / 2.0
    return None


def _casings(bore: Wellbore, ancestors: Sequence[Wellbore]) -> List[Casing]:
    out: List[Casing] = []
    for b in list(ancestors) + [bore]:
        out.extend(b.casings)
    return out


def _bore_r_in(md: float, cas: Sequence[Casing],
               hole_in: Optional[float]) -> Optional[float]:
    ids = [c.id_in / 2.0 for c in cas if c.top_md <= md <= c.shoe_md]
    if ids:
        return min(ids)
    return hole_in


def _outer_edge_in(
    c: Casing, md: float, cas: Sequence[Casing], hole_in: Optional[float]
) -> Tuple[float, bool]:
    """Outer edge of casing ``c``'s annular cement (inches) + is-open-hole flag.

    min(next-outer casing ID, drilled hole).
    """
    ids = [d.id_in / 2.0 for d in cas
           if d.od_in > c.od_in and d.top_md <= md <= d.shoe_md]
    hr = hole_in if hole_in is not None else (min(ids) if ids else c.od_in / 2.0 + 1.0)
    if ids and min(ids) < hr:
        return min(ids), False           # confined by steel
    return hr, True                      # against rock


# --------------------------------------------------------------------------
# element drawers (all off Centreline.perp)
# --------------------------------------------------------------------------
_STEEL = dict(facecolor="0.2", edgecolor="k", lw=0.4, zorder=3)
_CEMENT = dict(facecolor="0.72", edgecolor="none", zorder=1)
_PLUG = dict(facecolor="0.72", edgecolor="0.4", lw=0.4, zorder=2.5)

# Fallback fills keyed on the fluid NAME, used only when a fluid carries no
# explicit colour. Deliberately washed-out: an annulus fluid is background to
# the steel and cement, and must never out-read a barrier.
_FLUID_FILLS = {
    "brine": "#cfe3f2", "seawater": "#cfe3f2", "water": "#cfe3f2",
    "mud": "#d8cdb4", "obm": "#d3c2a4", "wbm": "#cfd8c0",
    "packer": "#e2d6ef", "diesel": "#efe3c2", "base oil": "#efe3c2",
    "gas": "#f6e2e2", "nitrogen": "#f6e2e2", "inhibited": "#cfe3f2",
}
_FLUID_DEFAULT = "#e6eef3"


def _fluid_fill(f) -> str:
    """Explicit colour wins
    else match the name
    else a neutral default."""
    if getattr(f, "colour", None):
        return f.colour
    nm = (getattr(f, "name", "") or "").lower()
    for key, col in _FLUID_FILLS.items():
        if key in nm:
            return col
    return _FLUID_DEFAULT


def _draw_annulus_fluids(ax, cl: Centreline, bore: Wellbore):
    """Fill each annulus fluid between its casing OD and the annulus outer wall.

    Uses the same outer-edge resolution as the cement (next-outer casing ID, or
    the drilled hole where nothing confines it), so a fluid column and the
    cement below it line up exactly rather than being drawn to different walls.
    """
    cas = list(bore.casings)
    for f in getattr(bore, "annulus_fluids", []) or []:
        inner = next((c for c in cas if abs(c.od_in - f.inside_od_in) < 1e-6), None)
        if inner is None:                    # names an annulus that is not there
            continue
        top, base = float(f.top_md), float(f.base_md)
        if base <= top:
            continue
        md = np.linspace(top, base, 160)
        ri = cl.rdraw(inner.od_in / 2.0)
        oR = np.empty(md.size)
        for i, m in enumerate(md):
            edge_in, _open = _outer_edge_in(inner, m, cas, _hole_r_in(m, bore))
            oR[i] = cl.rdraw(edge_in)
        _band(ax, cl, md, ri, oR,
              facecolor=_fluid_fill(f), edgecolor="none", zorder=0.8)


def _band(ax, cl: Centreline, md, ri_draw, ro_draw, **kw):
    """Two-side strip between inner/outer offsets (annulus / steel wall)."""
    ro_arr = np.broadcast_to(ro_draw, md.shape) if np.ndim(ro_draw) else None
    for s in (-1, 1):
        xi, yi = cl.perp(md, ri_draw, s)
        if ro_arr is None:
            xo, yo = cl.perp(md, ro_draw, s)
        else:
            nx, ny = cl.ndir(md)
            xo, yo = cl.x(md) + s * ro_arr * nx, cl.y(md) + s * ro_arr * ny
        ax.fill(np.r_[xi, xo[::-1]], np.r_[yi, yo[::-1]], **kw)


def _draw_hole(ax, cl: Centreline, bore: Wellbore, md0: float, md1: float):
    """Telescoping drilled-hole wall, drawn at the nominal gauge radius.

    The wall is a clean line at the drilled radius. It is deliberately NOT
    roughened: a drawn wobble is decorative, implies washout detail the
    schematic does not hold, and any real caliper would be plotted as data.
    """
    md = np.arange(md0, md1 + 0.001, 3.0)
    rr = np.array([cl.rdraw(_hole_r_in(m, bore) or 0.0) for m in md])
    nx, ny = cl.ndir(md)
    x, y = cl.x(md), cl.y(md)
    for s in (-1, 1):
        d = s * rr
        ax.plot(x + d * nx, y + d * ny, color="0.55", lw=0.7, zorder=0.6)
    # rounded bottom: a smooth half-ellipse spanning the hole, depth = radius
    rb = cl.rdraw((_hole_r_in(md1 - 1, bore) or 0.0))
    nxb, nyb = cl.ndir(md1)
    txb, tyb = cl.tdir(md1)
    nxb, nyb, txb, tyb = nxb[0], nyb[0], txb[0], tyb[0]
    x0, y0 = float(cl.x(md1)), float(cl.y(md1))
    th = np.linspace(-np.pi / 2, np.pi / 2, 60)
    ax.plot(x0 + rb * np.sin(th) * nxb + rb * np.cos(th) * txb,
            y0 + rb * np.sin(th) * nyb + rb * np.cos(th) * tyb,
            color="0.55", lw=0.7, zorder=0.6)


def _draw_casing(ax, cl: Centreline, c: Casing, cas: Sequence[Casing], bore: Wellbore):
    ro, ri = cl.rdraw(c.od_in / 2), cl.rdraw(c.id_in / 2)
    md = np.linspace(c.top_md, c.shoe_md, 240)
    # annular cement toc -> shoe, casing OD -> outer edge (hole wall in open hole)
    mdc = md[md >= c.toc_md]
    if mdc.size:
        oR = np.empty(mdc.size)
        for i, m in enumerate(mdc):
            edge_in, open_hole = _outer_edge_in(c, m, cas, _hole_r_in(m, bore))
            e = cl.rdraw(edge_in)
            oR[i] = e
        _band(ax, cl, mdc, ro, oR, **_CEMENT)
    # steel wall
    _band(ax, cl, md, ri, ro, **_STEEL)
    # right-angle black shoe tri: vertical leg UP the outer wall, apex OUTWARD
    nx, ny = cl.ndir(c.shoe_md)
    tx, ty = cl.tdir(c.shoe_md)
    nx, ny, tx, ty = nx[0], ny[0], tx[0], ty[0]
    x0, y0 = float(cl.x(c.shoe_md)), float(cl.y(c.shoe_md))
    hh = 0.35 * abs(cl.y(c.shoe_md) - cl.y(c.top_md))
    hh = float(min(hh, cl.rdraw(2.5) * 6))
    w = cl.rdraw(min(0.30 * c.od_in / 2, 2.0))
    for s in (-1, 1):
        ox, oy = x0 + s * ro * nx, y0 + s * ro * ny
        ax.fill([ox, ox - tx * hh, ox + s * w * nx],
                [oy, oy - ty * hh, oy + s * w * ny], facecolor="k", zorder=4)
    lx, ly = cl.perp(c.shoe_md, ro + cl.rdraw(2.0), 1)
    ax.text(float(lx[0]), float(ly[0]), c.name, fontsize=5.2, va="center", zorder=6)


def _draw_liner_hanger(ax, cl: Centreline, c: Casing,
                       cas: Sequence[Casing], start_md: float):
    """Box-with-X in the annulus each side, at a hung liner's top (top_md > start)."""
    if c.top_md <= start_md + 1e-6:
        return
    hosts = [d.id_in / 2.0 for d in cas
             if d.od_in > c.od_in and d.top_md <= c.top_md <= d.shoe_md]
    if not hosts:
        return
    ro, hri = cl.rdraw(c.od_in / 2), cl.rdraw(min(hosts))
    # square-ish: drawing height = annulus width
    ty = cl.tdir(c.top_md)[1][0]
    hh_md = (hri - ro) / max(abs(ty), 1e-3)
    md = np.linspace(c.top_md, c.top_md + hh_md, 6)
    for s in (-1, 1):
        xi, yi = cl.perp(md, ro, s)
        xo, yo = cl.perp(md, hri, s)
        ax.fill(np.r_[xi, xo[::-1]], np.r_[yi, yo[::-1]],
                facecolor="white", edgecolor="k", lw=0.8, zorder=5)
        ax.plot([xi[0], xo[-1]], [yi[0], yo[-1]], color="k", lw=0.6, zorder=6)
        ax.plot([xo[0], xi[-1]], [yo[0], yi[-1]], color="k", lw=0.6, zorder=6)


def _draw_plug(ax, cl: Centreline, plug, cas: Sequence[Casing], bore: Wellbore):
    """Grey solid plug, single wall-to-wall polygon, diameter = casing ID it sits in."""
    md = np.linspace(plug.top_md, plug.base_md, 120)
    rr = np.array([cl.rdraw(_bore_r_in(m, cas, _hole_r_in(m, bore)) or 0.0)
                   for m in md])
    nx, ny = cl.ndir(md)
    xL = cl.x(md) - rr * nx
    yL = cl.y(md) - rr * ny
    xR = cl.x(md) + rr * nx
    yR = cl.y(md) + rr * ny
    ax.fill(np.r_[xL, xR[::-1]], np.r_[yL, yR[::-1]], **_PLUG)
    lr = cl.rdraw(
        _bore_r_in(plug.base_md, cas, _hole_r_in(plug.base_md, bore)) or 0.0)
    lx, ly = cl.perp(plug.base_md, lr + cl.rdraw(1.5), 1)
    ax.text(float(lx[0]), float(cl.y((plug.top_md + plug.base_md) / 2)),
            plug.name, fontsize=5.2, va="center", color="0.35")


# --------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------
def _default_vs_azi(resolvers: Dict[str, DepthResolver]) -> float:
    """VS azimuth = direction of max horizontal departure across all bores."""
    best = (0.0, 0.0)  # (departure, azi)
    for res in resolvers.values():
        n, e, _ = res.pos(res.md[-1])
        dep = float(np.hypot(n, e))
        if dep > best[0]:
            best = (dep, float(np.degrees(np.arctan2(e, n))))
    return best[1]


def render_plumbing(
    schematic: WellSchematic,
    ax=None,
    vs_azi: Optional[float] = None,
    mode: str = "MD",
    exag: float = 45.0,
):
    """Draw a system-P&A plumbing diagram for a (possibly multilateral) well.

    Each bore is drawn from its ``kickoff_md`` (root from surface) to TD
    the
    shared trunk is drawn once by the root, so the whole connected system reads
    as one. Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 9.6))

    bores = {b.id: b for b in schematic.wellbores}
    resolvers = {bid: DepthResolver(b.survey, name=bid) for bid, b in bores.items()}
    if vs_azi is None:
        vs_azi = _default_vs_azi(resolvers)

    def ancestors(b: Wellbore) -> List[Wellbore]:
        chain: List[Wellbore] = []
        p = b.parent_id
        while p is not None and p in bores:
            chain.insert(0, bores[p])
            p = bores[p].parent_id
        return chain

    ymax = 0.0
    # root(s) first, then children (draw order = trunk behind)
    order = sorted(bores.values(), key=lambda b: b.kickoff_md)
    for b in order:
        cl = Centreline(resolvers[b.id], vs_azi, mode=mode, exag=exag)
        anc = ancestors(b)
        cas = _casings(b, anc)
        start = b.kickoff_md if b.parent_id else float(resolvers[b.id].md[0])
        td = float(resolvers[b.id].md[-1])
        ymax = max(ymax, float(cl.y(td)))
        _draw_hole(ax, cl, b, start, td)
        # fluids first: they sit behind cement and steel (zorder 0.8 < 1 < 3),
        # so a cemented interval always reads as cement even where a fluid
        # column is declared over the same depths.
        _draw_annulus_fluids(ax, cl, b)
        for c in b.casings:
            _draw_casing(ax, cl, c, cas, b)
            _draw_liner_hanger(ax, cl, c, cas, start)
        for plug in b.cement_plugs:
            _draw_plug(ax, cl, plug, cas, b)

    ax.set_aspect("equal")
    ax.set_ylim(ymax * 1.05, -ymax * 0.06)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel(f"{mode} {schematic.well.depth_reference} (m)")
    ax.grid(axis="y", color="0.93", lw=0.4)
    return ax
