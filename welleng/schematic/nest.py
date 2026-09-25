"""Multi-bore "nest" view: a parent bore and its sidetracks in swim lanes.

Each bore of a :class:`~welleng.schematic.models.WellSchematic` is drawn in its
own vertical lane; a branch leaves its parent's lane through a connector (a
quarter turn, a horizontal run and a quarter turn down) at its
:attr:`~welleng.schematic.models.Wellbore.branch_md`. Lane offsets are
DIAGRAMMATIC -- they show branch identity, not distance -- and the figure says
so. For the real geometry of the bores in one vertical-section plane, use
:func:`~welleng.schematic.plumbing.render_plumbing`.

One depth axis, three modes (:data:`DEPTH_MODES`), every y on the figure
through one :class:`DepthMap`:

* ``"tvd"`` -- y is TVDSS from minimum curvature on each bore's survey
  (:class:`~welleng.schematic.depth.DepthResolver`). The reference the MD
  views are checked against.
* ``"md"`` -- y is MD. A connector CONSUMES MD: every point of it at height y
  is that MD on the branch, so it is kept short.
* ``"md-paused"`` -- the MD clock stops at each branch point. The connector is
  drawn in a PAUSE, a band of page that carries no depth, bounded by break
  lines (ISO 128) with the depth spine broken across it. Pipe width no longer
  trades against connector length, and nothing of the parent is drawn on the
  bend.

What a reader takes at face value is kept true: every tick is the true depth at
its height and none sits inside a pause; nothing that carries a depth is drawn
inside a pause; a curve is broken at every pause (:meth:`DepthMap.trace`).
Anything the model holds but the view does not draw, and every inference made
to draw it, is stated in a caveat footer (:attr:`NestView.caveats`).

Packer and hanger boxes span from the inner string to the host casing's wall.
The inner string is, in order: the recorded
:attr:`~welleng.schematic.models.CompletionItem.inner_string`; the smallest
string passing through the depth; a string whose top lies within
:data:`HANG_TOL_M` below it (the string it hangs); the nearest tubing above it,
marked [INFERRED]. A packer with no recorded OD is drawn to its host casing,
and says so.

Like :func:`~welleng.schematic.plumbing.render_plumbing` this view draws
directly with matplotlib (imported when called); it needs the ``[all]`` extra.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .depth import DepthResolver
from .models import WellSchematic

#: The depth-axis modes of :func:`render_nest` and :class:`DepthMap`.
DEPTH_MODES = ("tvd", "md", "md-paused")

#: A string whose top lies within this many metres below a packer is taken to
#: be the string that packer hangs, never its host.
HANG_TOL_M = 5.0

_PALETTE = ("#8a8a8a", "#1f77b4", "#c0392b", "#2ca02c", "#9467bd", "#8c564b",
            "#e377c2")


# --------------------------------------------------------------------------
# depth map: (bore, md) -> page y, with pauses
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Void:
    """A pause: a band of page where depth does not advance.

    A branch connector is drawn in it. Keyed on the branch point, so the map
    stays a function of depth alone.
    """

    bore: str        # the bore the branch LEAVES (the parent)
    md: float        # the branch point, MD on that bore
    height: float    # page units (same unit as y); must be positive
    label: str = ""  # the branch's label


class DepthMap:
    """``(bore, md) -> y`` for one drawing, in one of :data:`DEPTH_MODES`.

    ``y = d + sum(height of every void shallower than d)``, where ``d`` is the
    DOMAIN value: MD, or TVD from the caller's ``tvd(bore, md)``. Voids are
    keyed on the domain value of their branch point on the parent, so y is
    monotone in depth and the inverse is exact outside the voids.

    In TVD mode a lateral that climbs back above a void's TVD is drawn above it
    again: the map does not hide a TVD reversal.

    The map holds no survey maths; ``tvd`` must be minimum curvature on the
    bore's survey (as :class:`~welleng.schematic.depth.DepthResolver`).
    """

    def __init__(self, mode: str, *,
                 tvd: Optional[Callable[[str, float], float]] = None,
                 voids: Sequence[Void] = ()):
        if mode not in DEPTH_MODES:
            raise ValueError(
                f"depth mode {mode!r} is not one of {', '.join(DEPTH_MODES)}")
        if mode == "tvd" and tvd is None:
            raise ValueError(
                "TVD mode needs the caller's tvd(bore, md) -- minimum "
                "curvature on the bore's survey; nothing is assumed in its "
                "place")
        if mode == "md" and voids:
            raise ValueError(
                "plain MD has no voids -- a connector consumes MD there; use "
                "'md-paused'")
        for v in voids:
            if not v.height > 0.0:
                raise ValueError(
                    f"void at {v.bore} {v.md:.2f} m has height {v.height!r}: "
                    "a void must have positive height")
        self.mode = mode
        self._tvd = tvd
        keyed = sorted(((self.domain(v.bore, v.md), v) for v in voids),
                       key=lambda p: p[0])
        for (d0, v0), (d1, v1) in zip(keyed[:-1], keyed[1:]):
            if abs(d1 - d0) < 1e-9:
                raise ValueError(
                    f"two voids at the same depth ({v0.bore} {v0.md:.2f} m, "
                    f"{v1.bore} {v1.md:.2f} m): their connectors would be "
                    "drawn on top of each other")
        self._d = tuple(d for d, _ in keyed)
        self.voids = tuple(v for _, v in keyed)

    @property
    def paused(self) -> bool:
        """Whether connectors are drawn in voids (True) or consume depth."""
        return self.mode != "md"

    @property
    def depths(self) -> Tuple[float, ...]:
        """Domain value of each void, ascending, parallel to :attr:`voids`."""
        return self._d

    def domain(self, bore: str, md: float) -> float:
        """The depth the axis is labelled in: MD, or TVD from ``tvd``."""
        if self.mode == "tvd":
            return float(self._tvd(bore, md))
        return float(md)

    def y_of_domain(self, d: float, side: str = "lo") -> float:
        """Page y for domain value ``d``.

        At a void's own depth, ``side='lo'`` is the TOP of the void and
        ``'hi'`` its bottom; both are that depth, and the band between them is
        no depth at all.
        """
        if side not in ("lo", "hi"):
            raise ValueError("side must be 'lo' or 'hi'")
        shift = 0.0
        for dv, v in zip(self._d, self.voids):
            if dv < d - 1e-9 or (side == "hi" and abs(dv - d) <= 1e-9):
                shift += v.height
        return d + shift

    def y(self, bore: str, md: float, side: str = "lo") -> float:
        """Page y for ``md`` on ``bore``; ``side`` as :meth:`y_of_domain`."""
        return self.y_of_domain(self.domain(bore, md), side)

    def domain_at(self, y: float) -> Optional[float]:
        """The depth drawn at page ``y``; None strictly inside a void."""
        shift = 0.0
        for dv, v in zip(self._d, self.voids):
            top = dv + shift
            if y < top - 1e-9:
                break
            if y < top + v.height - 1e-9:
                return None if y > top + 1e-9 else dv
            shift += v.height
        return y - shift

    def spans(self) -> List[Tuple[float, float, Void]]:
        """``[(y_top, y_bottom, void), ...]``, shallowest first."""
        return [(self.y_of_domain(d, "lo"), self.y_of_domain(d, "hi"), v)
                for d, v in zip(self._d, self.voids)]

    def ticks(self, step: float, lo: float, hi: float
              ) -> List[Tuple[float, float]]:
        """``[(y, depth), ...]`` at multiples of ``step`` over ``[lo, hi]``.

        The label is the TRUE depth at y; none falls inside a void (a tick at
        a void's own depth sits on its top edge).
        """
        if not step > 0.0:
            raise ValueError("tick step must be positive")
        k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
        return [(self.y_of_domain(k * step), k * step)
                for k in range(k0, k1 + 1)]

    def trace(self, bore: str, md: Sequence[float], values: Sequence[float]):
        """``(y, values)`` for a curve sampled at ``md`` on ``bore``.

        A NaN break is inserted wherever consecutive samples straddle a void,
        so no line is drawn through page the curve was never evaluated in. A
        sample AT a void's depth is drawn on both edges (both are that depth)
        with the break between them.
        """
        nan = float("nan")
        ys, vs, prev = [], [], None
        for m, val in zip(md, values):
            d = self.domain(bore, m)
            if prev is not None and any(min(prev, d) < dv < max(prev, d)
                                        for dv in self._d):
                ys.append(nan)
                vs.append(nan)
            if any(abs(d - dv) <= 1e-9 for dv in self._d):
                ys += [self.y_of_domain(d, "lo"), nan,
                       self.y_of_domain(d, "hi")]
                vs += [val, nan, val]
            else:
                ys.append(self.y_of_domain(d))
                vs.append(val)
            prev = d
        return ys, vs


def paused_extent(extent0: float, page: float, radii: Sequence[float]) -> float:
    """Page extent (y units) once every connector is drawn in its own void.

    A connector of turn radius ``R`` (x units) is ``2 R dpx`` tall, where
    ``dpx = E / page`` is set by the extent ``E`` the voids add to. Solving
    ``E = E0 + 2 sum(R) E / page`` for ``E`` gives::

        E = E0 * page / (page - 2 * sum(R))

    ``extent0`` is the extent with no voids; ``radii`` are the connectors whose
    voids lie above the deepest point drawn. Refuses ``2 sum(R) >= page``: the
    voids alone would fill the page.
    """
    two_r = 2.0 * float(sum(radii))
    if not page > two_r:
        raise ValueError(
            f"connectors of total radius {two_r / 2.0:.4g} need {two_r:.4g} "
            f"of a page {page:.4g} tall: the voids alone would fill it")
    return float(extent0) * page / (page - two_r)


# --------------------------------------------------------------------------
# connector geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LaneChange:
    """A connector: quarter turn, horizontal run, quarter turn down.

    Coordinates are ``(x, z)`` with ``z = z0 + depth_per_x * u``, where
    ``(x, u)`` is the isotropic paper frame the geometry is built in; with
    ``depth_per_x`` set from the axes' aspect, the turns are circles on paper.
    """

    points: tuple    # centreline ((x, z), ...)
    radius: float    # turn radius, x units
    straight: float  # horizontal run between the turns, x units
    depth: float     # depth consumed, z units: always 2 * radius * depth_per_x
    _paper: tuple    # ((x, u, tx, tu), ...): point + unit tangent, paper frame
    _z0: float       # z of the start
    _k: float        # depth_per_x

    def walls(self, od: float):
        """The two walls of a bore of width ``od`` (x units) on this path."""
        if od <= 0.0:
            raise ValueError("od must be positive: a bore with no width is not "
                             "a bore")
        h = 0.5 * od
        a, b = [], []
        for x, u, tx, tu in self._paper:
            nx, nu = tu, -tx                  # unit normal in the paper frame
            a.append((x + h * nx, self._z0 + self._k * (u + h * nu)))
            b.append((x - h * nx, self._z0 + self._k * (u - h * nu)))
        return tuple(a), tuple(b)


def lane_change(x0: float, z0: float, offset: float, *, radius: float,
                depth_per_x: float = 1.0, n: int = 45) -> LaneChange:
    """Connector from ``(x0, z0)`` heading down to
    ``(x0 + offset, z0 + 2 radius depth_per_x)`` heading down again.

    Two quarter turns of ``radius`` joined by a horizontal run of
    ``offset - 2 radius``. The depth consumed is ``2 radius`` whatever the
    offset, so lanes can be spread apart without the connector taking more
    depth. Refuses ``offset < 2 radius``: the two turns alone move that far
    across.
    """
    R, k = float(radius), float(depth_per_x)
    if R <= 0.0:
        raise ValueError("radius must be positive")
    if offset < 2.0 * R - 1e-12:
        raise ValueError(
            f"offset {offset:.4g} is less than two turn radii ({2.0 * R:.4g}): "
            "the quarter turns alone move that far across")
    paper = []
    for i in range(n):                                   # turn 1: down -> across
        t = 0.5 * math.pi * i / (n - 1)
        paper.append((x0 + R - R * math.cos(t), R * math.sin(t),
                      math.sin(t), math.cos(t)))
    paper.append((x0 + offset - R, R, 1.0, 0.0))         # the horizontal run
    for i in range(1, n):                                # turn 2: across -> down
        t = 0.5 * math.pi * i / (n - 1)
        paper.append((x0 + offset - R + R * math.sin(t),
                      2.0 * R - R * math.cos(t), math.cos(t), math.sin(t)))
    pts = tuple((x, z0 + k * u) for x, u, _, _ in paper)
    return LaneChange(points=pts, radius=R, straight=offset - 2.0 * R,
                      depth=2.0 * R * k, _paper=tuple(paper), _z0=float(z0),
                      _k=k)


# --------------------------------------------------------------------------
# the view
# --------------------------------------------------------------------------
@dataclass
class NestView:
    """What :func:`render_nest` drew, for a caller to save and to check.

    Path nodes are ``(x, y, tx, ty, md)``: paper position, unit tangent in the
    isotropic paper frame, and the MD on that bore.
    """

    fig: object                     # matplotlib Figure
    ax: object                      # its Axes
    mode: str                       # one of DEPTH_MODES
    depth_map: DepthMap             # every y on the figure came from this
    paths: Dict[str, list]          # bore id -> centreline nodes, surface to TD
    connectors: Dict[str, list]     # child id -> its connector's nodes
    branch_md: Dict[str, float]     # child id -> MD on its parent where it leaves
    connector_md: Dict[str, float]  # child id -> MD its connector spans (0 when paused)
    parent: Dict[str, Optional[str]]  # bore id -> parent id
    packer_boxes: list              # (bore, md, y_top, y_bottom) per packer drawn
    break_lines: list               # the break-line artists
    well_artists: set               # every artist of the well itself
    dpx: float                      # page y per x unit (circles on paper)
    scale: float                    # x units per inch of diameter
    lane_x: Dict[str, float]        # bore id -> its lane's x
    caveats: List[str]              # what is not drawn, and every inference
    occupied: Callable              # (y, tol, gap=, pad=) -> x intervals of the well

    def save(self, stem: str, formats: Sequence[str] = ("png", "svg", "pdf"),
             dpi: int = 170) -> None:
        """Save the figure as ``stem.<fmt>`` for each format."""
        for ext in formats:
            self.fig.savefig(f"{stem}.{ext}", **({"dpi": dpi} if ext == "png"
                                                 else {}))


def _string_label(name: Optional[str]) -> str:
    """A string's name up to any parenthesised note."""
    return (name or "").split("(")[0].strip()


def render_nest(schematic: WellSchematic, *, mode: str = "md-paused",
                grid: bool = False, labels: Optional[Dict[str, str]] = None,
                extra_caveats: Sequence[str] = ()) -> NestView:
    """Draw the nest of ``schematic`` on the ``mode`` depth axis.

    Parameters
    ----------
    schematic : WellSchematic
        One :class:`~welleng.schematic.models.Wellbore` per bore, each with
        only its own rows, placed by ``parent_id`` and ``branch_md``. Parents
        must appear before their children.
    mode : str
        One of :data:`DEPTH_MODES`. ``"tvd"`` needs a survey on every bore
        that covers it below its branch point, and ``Well.datum_elevation_m``
        (TVDSS is not drawn on an assumed datum).
    grid : bool
        Draw faint lines at the labelled depth ticks, behind the drawing.
    labels : dict, optional
        Bore id -> short label. Default: the id less its nearest ancestor's
        id as a prefix.
    extra_caveats : sequence of str
        The caller's own reconciliations, printed with the view's.

    Returns
    -------
    NestView
        The figure and what was drawn. Nothing is saved; see
        :meth:`NestView.save`.

    Raises
    ------
    ValueError
        On an unknown mode, a bore with nothing to draw, a missing datum or
        survey coverage in TVD mode, or a branch point at or below the deepest
        point drawn in a paused mode.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    if mode not in DEPTH_MODES:
        raise ValueError(f"mode {mode!r} is not one of {', '.join(DEPTH_MODES)}")
    bore_of = {b.id: b for b in schematic.wellbores}
    names = [b.id for b in schematic.wellbores]
    parent = {b.id: b.parent_id for b in schematic.wellbores}
    for n in names:
        p = parent[n]
        if p is not None and (p not in bore_of or names.index(p) > names.index(n)):
            raise ValueError(f"{n}: parent {p!r} is not a bore listed before it")
    kop = {b.id: float(b.branch_md) for b in schematic.wellbores if b.parent_id}
    wellname = schematic.well.name
    col = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(names)}
    caveats: List[str] = list(extra_caveats)

    def _td(b):
        ends = ([h.base_md for h in b.hole_sections]
                + [c.shoe_md for c in b.drawable_casings])
        if not ends:
            raise ValueError(f"{b.id}: no hole section or placed string -- "
                             "nothing to draw and no TD")
        return max(ends)
    td = {b.id: _td(b) for b in schematic.wellbores}

    def _short(n):
        if labels and n in labels:
            return labels[n]
        p = parent[n]
        while p is not None:              # strip the nearest ancestor's id
            if n.startswith(p + "-"):
                return n[len(p) + 1:]
            p = parent[p]
        return n
    short = {n: _short(n) for n in names}

    # what the model holds that this view does not draw -- said, not dropped
    for b in schematic.wellbores:
        for c in b.unplaceable_casings:
            caveats.append(f"{short[b.id]} {c.name}: no shoe depth -- not drawn")
        if b.mechanical_plugs:
            caveats.append(f"{short[b.id]}: {len(b.mechanical_plugs)} "
                           "mechanical plug(s) not drawn in this view")
        if b.perforations:
            caveats.append(f"{short[b.id]}: perforations not drawn in this view")
        if b.annulus_fluids:
            caveats.append(f"{short[b.id]}: annulus fluids not drawn in this view")
        other = [c.name or c.type for c in b.completion
                 if c.type in ("sssv", "nipple")]
        if other:
            caveats.append(f"{short[b.id]}, not drawn in this view: "
                           + "; ".join(other))
        if b.window_md is not None and abs(b.window_md - b.kickoff_md) > 1.0:
            caveats.append(f"{short[b.id]} branches at its milled window "
                           f"{b.window_md:.0f} m (kick-off recorded "
                           f"{b.kickoff_md:.1f} m)")

    # ---- strings, per bore: (name, kind, od, steel intervals, cement) --------
    strings = {n: [(c.name, c.kind, c.od_in, c.top_md, c.shoe_md,
                    c.steel_intervals(), c.cement_intervals())
                   for c in sorted(bore_of[n].drawable_casings,
                                   key=lambda c: -c.od_in)] for n in names}

    def chain(n):
        """[(bore, deepest MD at which n's path is still in it)], n to root."""
        out, cur, limit = [], n, float("inf")
        while cur:
            out.append((cur, limit))
            if parent[cur]:
                limit = min(limit, kop[cur])
            cur = parent[cur]
        return out

    def host_strings(n, md):
        """(od, id) of casings and liners physically at ``md`` on n's path,
        excluding any whose top lies within HANG_TOL_M of it (a hung string)."""
        out = []
        for b, lim in chain(n):
            if md > lim:
                continue
            for c in bore_of[b].drawable_casings:
                if (c.kind in ("casing", "liner") and c.steel_at(md)
                        and abs(c.top_md - md) > HANG_TOL_M):
                    out.append((c.od_at(md), c.id_at(md)))
        return out

    def named_string_od(n, name):
        """OD (in) of the string called ``name`` on n's path, or None."""
        for b, _lim in chain(n):
            for c in bore_of[b].drawable_casings:
                if c.name == name:
                    return c.od_in
            for c in bore_of[b].completion:
                if c.type == "tubing" and c.name == name:
                    return c.od_in
        return None

    def packer_inner_od(n, pk, outer_od):
        """(inner OD, why-or-None), in the order the module docstring gives."""
        if pk.inner_string is not None:
            od = named_string_od(n, pk.inner_string)
            if od is not None:
                return od, None
            caveats.append(f"{pk.name or 'packer'} at {pk.md:.0f} m: its "
                           f"inner string {pk.inner_string!r} is not in the "
                           "model -- inferred instead")
        b = bore_of[n]
        cands = ([(c.od_in, c.top_md, c.shoe_md) for c in b.drawable_casings]
                 + [(c.od_in, c.top_md, c.base_md) for c in b.completion
                    if c.type == "tubing"])
        cands = [c for c in cands if c[0] < outer_od - 1e-6]
        through = [od for od, t, bs in cands if t < pk.md - 1e-6 and pk.md < bs]
        if through:
            return min(through), None
        hung = [(t - pk.md, od) for od, t, bs in cands
                if -1e-6 <= t - pk.md <= HANG_TOL_M]
        if hung:
            return min(hung)[1], None
        above = [(pk.md - c.base_md, c.od_in) for c in b.completion
                 if c.type == "tubing" and c.base_md <= pk.md]
        if above:
            gap, od = min(above)
            return od, (f"inner OD {od:.3f}in from the tubing ending "
                        f"{gap:.1f} m above it [INFERRED]")
        return None, None

    # ---- TVD: minimum curvature on each bore's survey -------------------------
    tvd_fn, datum, res = None, None, {}
    tie: Dict[str, float] = {}      # bore -> TVD offset tying it to its parent
    if mode == "tvd":
        datum = schematic.well.datum_elevation_m
        if datum is None:
            raise ValueError(f"{wellname}: no datum elevation on the well -- "
                             "TVDSS is not drawn on an assumed datum")
        res = {n: DepthResolver(bore_of[n].survey, name=n) for n in names}

        @functools.lru_cache(maxsize=None)
        def tvd_fn(n, md):
            """TVDSS (m) at ``md`` on bore n's path."""
            # above a branch point the bore IS its parent's hole, so its TVD
            # is the parent's -- not its own survey's, which ties in at its
            # own recorded kick-off and can sit a little off the parent
            while parent[n] and md <= kop[n]:
                n = parent[n]
            r = res[n]
            if not (r.md_min - 1e-6 <= md <= r.md_max + 1e-6):
                raise ValueError(
                    f"{n}: MD {md:.1f} m is outside its survey "
                    f"({r.md_min:.1f}-{r.md_max:.1f} m) -- no TVD is drawn "
                    "for it, and none is extrapolated")
            return (r.tvd_at(min(max(md, r.md_min), r.md_max)) - datum
                    + tie.get(n, 0.0))

        # A child's own survey can put the shared branch point a little off
        # its parent's (it ties in at its own kick-off). Tie it to the parent
        # there, so its depth has no step at the branch; say so past 1 cm.
        for n in names:                          # parents come first
            if parent[n]:
                tie[n] = tvd_fn(parent[n], kop[n]) - (
                    res[n].tvd_at(kop[n]) - datum)
                if abs(tie[n]) > 0.01:
                    side = "shallower" if tie[n] > 0 else "deeper"
                    caveats.append(
                        f"{short[n]}: its survey puts the branch point "
                        f"{abs(tie[n]) * 100:.1f} cm {side} than "
                        f"{short[parent[n]]}'s; its TVD is tied to "
                        f"{short[parent[n]]}'s there")

    # ---- figure, scales -----------------------------------------------------
    FIG_W, FIG_H = 9.6, 11.0
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    AX_L, AX_B, AX_W, AX_H = 0.20, 0.14, 0.58, 0.78      # room for the footer
    ax = fig.add_axes([AX_L, AX_B, AX_W, AX_H])
    X0, X1 = -0.6, 5.1              # the drawing sits against the depth axis
    page = (AX_H * FIG_H) * ((X1 - X0) / (AX_W * FIG_W))  # y extent / dpx
    # One turn radius per connector, sized to the widest hole through it, and
    # ONE radial scale for every diameter and every mode. The scale is set so
    # that in plain MD the widest connector spans D_ARC of MD with its widest
    # hole's half-width inside 0.95 of its turn radius.
    D_ARC = 120.0
    widest = {c: max((h.bit_in for h in bore_of[c].hole_sections), default=0.0)
              for c in kop}
    for c, w in widest.items():
        if w <= 0.0:
            raise ValueError(f"{c}: no hole section -- its connector has no "
                             "width to carry")
    PAD_FRAC = 0.10         # bottom of the page left for the vertical TD labels
    dpx_md = (max(td.values()) / (1.0 - PAD_FRAC)) / page
    S = (1.9 * (D_ARC / (2.0 * dpx_md)) / max(widest.values())) if kop else \
        1.9 * (D_ARC / (2.0 * dpx_md)) / max(
            (h.bit_in for b in schematic.wellbores for h in b.hole_sections),
            default=1.0)
    r_c = {c: widest[c] * S / 1.9 for c in kop}
    off = 2.0 * max(r_c.values(), default=0.0) + 0.35
    lane_x = {n: i * off for i, n in enumerate(names)}
    x_lbl = min(2.0 * off + 0.5, X1 - 1.2)

    def _extent(dmap):
        """Shallowest and deepest page y of any bore."""
        ys = [dmap.y(n, m) for n in names
              for m in np.append(np.arange(0.0, td[n], 5.0), td[n])]
        return min(ys), max(ys)

    if mode == "md" or not kop:
        dmap, dpx = DepthMap("tvd" if mode == "tvd" else "md", tvd=tvd_fn), \
            dpx_md
        if mode == "tvd":
            lo0, hi0 = _extent(dmap)
            dpx = (hi0 - lo0) / (1.0 - PAD_FRAC) / page
    else:
        # void height = the connector's own paper height, 2 R dpx, and dpx is
        # set by the page extent the voids add to (paused_extent). That form
        # assumes every void lies above the deepest point drawn -- checked.
        map0 = DepthMap("tvd" if mode == "tvd" else "md", tvd=tvd_fn)
        lo0, hi0 = _extent(map0)
        dpx = paused_extent(hi0 - lo0, page * (1.0 - PAD_FRAC),
                            [r_c[c] for c in kop]) / (1.0 - PAD_FRAC) / page
        dmap = DepthMap(mode, tvd=tvd_fn,
                        voids=[Void(parent[c], kop[c], 2.0 * r_c[c] * dpx,
                                    short[c]) for c in kop])
        if max(dmap.depths) >= hi0:
            raise ValueError("a branch point is at or below the deepest point "
                             "drawn -- the page scale does not hold; nothing "
                             "drawn")
    ytop, ybot = _extent(dmap)
    ybot = ytop + (ybot - ytop) / (1.0 - PAD_FRAC)
    ax.set_ylim(ybot, ytop)
    ax.set_xlim(X0, X1)
    # MD a connector consumes (plain MD only)
    depth_md = {c: (2.0 * r_c[c] * dpx if mode == "md" else 0.0) for c in kop}

    # ---- paths: nodes (x, y, tx, ty, md) -------------------------------------
    def lane(n, x, md0, md1, first=None):
        """A straight lane on bore n from md0 to md1, with the jump across
        every void it passes. ``first`` (a connector's end) replaces md0."""
        mds = [md0, md1]
        if mode == "tvd":
            mds += list(np.arange(np.ceil(md0), md1, 2.0))  # TVD is not linear
        for dv, v in zip(dmap.depths, dmap.voids):
            if mode != "tvd":
                if md0 < v.md <= md1:
                    mds.append(v.md)
                continue
            # every MD on n where its TVDSS is the void's (exact inverse;
            # a climbing lateral crosses more than once)
            for m in res[n].md_at_tvd(dv + datum - tie.get(n, 0.0)):
                if md0 - 1e-9 <= m <= md1 + 1e-9 and \
                        (parent[n] is None or m > kop[n]):
                    mds.append(m)
        out = [first] if first is not None else []
        # one node per MD: a grid node and an inverse root an ulp apart would
        # emit a void's edges twice and draw back up through it
        merged: List[float] = []
        for m in sorted(mds):
            if not merged or m - merged[-1] > 1e-6:
                merged.append(m)
        for m in merged:
            if first is not None and m <= md0 + 1e-9:
                continue
            d = dmap.domain(n, m)
            if any(abs(d - dv) <= 1e-6 for dv in dmap.depths):
                deeper = (mode != "tvd"
                          or dmap.domain(n, min(m + 0.5, md1)) >= d)
                for side in (("lo", "hi") if deeper else ("hi", "lo")):
                    out.append((x, dmap.y_of_domain(d, side), 0.0, 1.0, m))
            else:
                out.append((x, dmap.y_of_domain(d), 0.0, 1.0, m))
        return out

    conn: Dict[str, list] = {}

    def _at(P, s):
        i = min(int(s), len(P) - 2)
        t = s - i
        return tuple(a + t * (b - a) for a, b in zip(P[i], P[i + 1]))

    def s_lo(P, md):
        """First path position with MD >= md (the TOP of a void there)."""
        for i, q in enumerate(P):
            if q[4] >= md - 1e-9:
                if i == 0 or q[4] == P[i - 1][4]:
                    return float(i)
                return i - 1 + (md - P[i - 1][4]) / (q[4] - P[i - 1][4])
        return float(len(P) - 1)

    def s_hi(P, md):
        """Last path position with MD <= md (the BOTTOM of a void there)."""
        for i in range(len(P) - 1, -1, -1):
            if P[i][4] <= md + 1e-9:
                if i == len(P) - 1 or P[i + 1][4] == P[i][4]:
                    return float(i)
                return i + (md - P[i][4]) / (P[i + 1][4] - P[i][4])
        return 0.0

    def sub(P, s0, s1):
        """The nodes of P between path positions s0 and s1, ends included."""
        if s1 < s0:
            return []
        out = [_at(P, s0)]
        out += [P[i] for i in range(int(np.floor(s0)) + 1, int(np.ceil(s1)))]
        out.append(_at(P, s1))
        return out

    def path(n):
        """Bore n's centreline nodes from surface to TD: its parent's path to
        the branch point, the connector, then its own lane."""
        if parent[n] is None:
            return lane(n, lane_x[n], 0.0, td[n])
        p, k = parent[n], kop[n]
        P = path(p)
        up = sub(P, 0.0, s_lo(P, k))
        y0 = dmap.y(p, k, "lo")
        lc = lane_change(lane_x[p], y0, lane_x[n] - lane_x[p], radius=r_c[n],
                         depth_per_x=dpx)
        cn = [(x, y0 + dpx * u, tx, tu, k + (dpx * u if mode == "md" else 0.0))
              for x, u, tx, tu in lc._paper]
        conn[n] = cn
        return up[:-1] + cn + lane(n, lane_x[n], k + depth_md[n], td[n],
                                   first=cn[-1])

    paths = {n: path(n) for n in names}
    children = {n: [c for c in kop if parent[c] == n] for n in names}
    right: List[dict] = []                 # right-hand column labels, placed last

    def at_md(n, md):
        """The node of bore n's path at MD ``md`` (the top of a void there)."""
        return _at(paths[n], s_lo(paths[n], md))

    def walls(pts, width):
        """(left, right), each [(x, y, md), ...], offset along the normal in
        the isotropic paper frame."""
        h = 0.5 * width
        rgt = [(x + h * ty, y - dpx * h * tx, m) for x, y, tx, ty, m in pts]
        lft = [(x - h * ty, y + dpx * h * tx, m) for x, y, tx, ty, m in pts]
        return lft, rgt

    def split(poly, inside):
        """Pieces of a polyline outside ``inside``, cut AT the boundary by
        bisection along each segment that changes state (never by dropping
        points: a straight run is two points)."""
        pieces, cur = [], []
        for i, q in enumerate(poly):
            if i:
                a = poly[i - 1]
                if inside(a) != inside(q):
                    lo, hi = 0.0, 1.0
                    for _ in range(40):
                        t = 0.5 * (lo + hi)
                        m = tuple(u + t * (v - u) for u, v in zip(a, q))
                        if inside(m) == inside(a):
                            lo = t
                        else:
                            hi = t
                    edge = tuple(u + 0.5 * (lo + hi) * (v - u)
                                 for u, v in zip(a, q))
                    if inside(a):
                        cur = [edge]
                    else:
                        cur.append(edge)
                        pieces.append(cur)
                        cur = []
            if not inside(q):
                cur.append(q)
        pieces.append(cur)
        return [p for p in pieces if len(p) > 1]

    def s_at_y(P, y, md_lo, md_hi):
        """Path position where P crosses page y with MD in [md_lo, md_hi]."""
        for i in range(len(P) - 1):
            (ya, ma), (yb, mb) = (P[i][1], P[i][4]), (P[i + 1][1], P[i + 1][4])
            if min(ya, yb) - 1e-9 <= y <= max(ya, yb) + 1e-9 and yb != ya:
                t = (y - ya) / (yb - ya)
                if md_lo - 1e-6 <= ma + t * (mb - ma) <= md_hi + 1e-6:
                    return i + t
        return None

    def mouth(owner, width, child):
        """(s0, s1) on the owner's path across which its element of ``width``
        is open for ``child`` to leave: where the child's upper hole wall
        crosses the element's right wall, to where its lower wall does."""
        P, k = paths[owner], kop[child]
        xw = lane_x[owner] + 0.5 * width
        left, rgt = walls(conn[child], widest[child] * S)
        lo, hi = s_lo(P, k), s_hi(P, k + depth_md[child])
        y0 = next((y for x, y, _ in rgt if x >= xw), None)
        y1 = next((y for x, y, _ in left if x >= xw), None)
        k1 = k + depth_md[child]
        s0 = lo if (rgt[0][0] >= xw or y0 is None) else \
            (s_at_y(P, y0, k, k1) or lo)
        s1 = hi if y1 is None else (s_at_y(P, y1, k, k1) or hi)
        return max(s0, lo), s1

    def parent_open_below(n):
        """Whether the parent stays OPEN below n's branch point (no plug)."""
        p, k = parent[n], kop[n]
        return not any(pl.top_md <= k + 1.0 and pl.base_md > k
                       for pl in bore_of[p].cement_plugs)

    def width_at(n, md):
        """Width of the widest hole recorded at ``md`` in bore n's own hole."""
        bits = [h.bit_in for h in bore_of[n].hole_sections
                if h.top_md <= md <= h.base_md]
        return (max(bits) if bits else 0.0) * S

    def innermost_string_width(n, md):
        """Width (x units) of the smallest string at ``md`` in bore n, else
        its hole."""
        ods = [c.od_in for c in bore_of[n].drawable_casings if c.steel_at(md)]
        return min(ods) * S if ods else width_at(n, md)

    def plot(poly, **kw):
        """Draw a polyline of (x, y, ...) points on the axes."""
        ax.plot([q[0] for q in poly], [q[1] for q in poly], **kw)

    def draw_element(owner, top, base, width, kind="hole", **kw):
        """Walls of one hole section or string along its owner's path, the
        right wall gapped where a child leaves.

        ``kind="hole"``: at a branch the child's inner hole wall is dropped
        where it lies inside the parent -- an open-hole kick-off leaves no
        wall between the two holes. ``kind="string"``: steel is run after the
        kick-off and is continuous through the bend, except where the parent
        stays open below the branch: there the string's inner wall is opened
        across the parent's bore (a liner run back into the parent, giving
        access to the parent's hole below).
        """
        P = paths[owner]
        s0, s1 = s_lo(P, top), s_hi(P, base)
        pts = sub(P, s0, s1)
        left, _ = walls(pts, width)
        p = parent.get(owner)
        parts = [left]
        if p and top <= kop[owner] + 1e-6:
            k, k1 = kop[owner], kop[owner] + depth_md[owner]

            def in_bend(q):
                """Whether wall point q lies on this bore's connector, left of
                its own lane."""
                return k - 1e-6 <= q[2] <= k1 + 1e-6 and \
                    q[0] < lane_x[owner] - 1e-6
            y_top = dmap.y(p, k, "lo")
            inside = None
            if kind == "hole":
                xc = lane_x[p] + 0.5 * width_at(p, k)

                def inside(q):
                    """Whether q lies inside the parent's hole on the bend."""
                    return in_bend(q) and q[0] < xc - 1e-9
            elif parent_open_below(owner):
                wi = innermost_string_width(p, k)
                xl, xc = lane_x[p] - 0.5 * wi, lane_x[p] + 0.5 * wi

                def inside(q):
                    """Whether q lies across the parent's innermost string,
                    below the top of the void."""
                    return in_bend(q) and xl <= q[0] < xc - 1e-9 and \
                        q[1] > y_top + 1e-6
            if inside is not None:
                parts = split(left, inside)
                gx = [q for q in left if inside(q)]
                if kind == "string" and gx:
                    g = gx[len(gx) // 2]
                    right.append(dict(
                        y=g[1], anchor=g[:2], lines=2, size=6.3, arrow=True,
                        text=f"opening in the liner:\naccess to {short[p]} below"))
        for part in parts:
            plot(part, **kw)
        edges = [s0]
        for c in children[owner]:
            if top < kop[c] < base:
                edges += list(mouth(owner, width, c))
        edges.append(s1)
        for a, b in zip(edges[0::2], edges[1::2]):
            if b > a:
                plot(walls(sub(P, a, b), width)[1], **kw)
        return pts

    # ---- annulus geometry -----------------------------------------------------
    def hole_width(n, z):
        """Hole diameter (x units) at MD z on n's PATH: the parent's hole above
        n's branch point."""
        if parent[n] and z < kop[n]:
            return hole_width(parent[n], z)
        return width_at(n, z)

    def outer_width(n, z, od_in):
        """What bounds the annulus outside a string of ``od_in`` on n's path
        at MD z: the next string out physically there, else the hole."""
        ods = [c.od_at(z) for b, lim in chain(n) if z <= lim
               for c in bore_of[b].drawable_casings
               if c.steel_at(z) and c.od_in > od_in + 1e-6]
        return min(ods) * S if ods else hole_width(n, z)

    def cement_annulus(n, toc, shoe, od_in):
        """Grey annulus strips from ``toc`` to ``shoe`` along n's path, split
        wherever the outer boundary changes and opened across any branch
        mouth on the right-hand side."""
        P = paths[n]
        cuts = {toc}
        for b, _lim in chain(n):
            for _nm, _k, _od, top, sh, _st, _cm in strings[b]:
                cuts.update(v for v in (top, sh) if toc < v < shoe)
            for h in bore_of[b].hole_sections:
                cuts.update(v for v in (h.top_md, h.base_md) if toc < v < shoe)
        cuts.update(kop[c] for c in kop if toc < kop[c] < shoe)
        ss = {s_lo(P, v) for v in cuts} | {s_hi(P, shoe)}
        mouths = [mouth(n, od_in * S, c) for c in children[n]
                  if toc < kop[c] < shoe]
        ss.update(v for g in mouths for v in g)
        ss = sorted(ss)
        w_in = od_in * S
        for a, b in zip(ss[:-1], ss[1:]):
            if b - a < 1e-9:
                continue
            w_out = outer_width(n, _at(P, 0.5 * (a + b))[4], od_in)
            if w_out <= w_in + 1e-9:
                continue
            pts = sub(P, a, b)
            li, ri = walls(pts, w_in)
            lo, ro = walls(pts, w_out)
            in_mouth = any(g0 - 1e-6 <= a and b <= g1 + 1e-6
                           for g0, g1 in mouths)
            for inner, outer, side in ((li, lo, "L"), (ri, ro, "R")):
                if side == "R" and in_mouth:
                    continue
                ax.add_patch(plt.Polygon(
                    [q[:2] for q in inner + outer[::-1]], closed=True,
                    facecolor="#bdbdbd", edgecolor="none", zorder=2.5))

    def shoe_triangles(x, y, w, size=0.055):
        """Filled shoe triangles outside both walls at the shoe."""
        for sgn in (-1, 1):
            xw = x + sgn * 0.5 * w
            ax.add_patch(plt.Polygon(
                [(xw, y), (xw + sgn * size, y), (xw, y - size * dpx * 1.4)],
                closed=True, facecolor="#1a1a1a", edgecolor="none", zorder=7))

    # ---- pauses: drawn as engineering breaks ----------------------------------
    spine = blended_transform_factory(ax.transAxes, ax.transData)
    x_break_end = x_lbl - 0.15          # break lines stop short of the labels
    break_lines: list = []
    well: set = set()

    def occupied(y, tol, gap=0.25, pad=0.06):
        """x intervals the WELL occupies within +-tol of page y: every wall,
        string, fill and box that comes that close, clustered and padded. A
        band, not a crossing test: a connector's wall can run along a void
        edge without crossing it."""
        lo, hi = y - tol, y + tol
        xs = []
        for ln in ax.lines:
            if ln.get_transform() != ax.transData or ln not in well:
                continue
            x, yy = (np.asarray(v, float) for v in ln.get_data())
            for i in range(len(x) - 1):
                ya, yb, xa, xb = yy[i], yy[i + 1], x[i], x[i + 1]
                if max(ya, yb) < lo or min(ya, yb) > hi:
                    continue
                if ya == yb:
                    xs += [xa, xb]
                    continue
                ta = np.clip([(lo - ya) / (yb - ya), (hi - ya) / (yb - ya)],
                             0.0, 1.0)
                xs += [xa + t * (xb - xa) for t in ta]
        for pt in ax.patches:
            if pt not in well:
                continue
            v = ax.transData.inverted().transform(
                pt.get_transform().transform(pt.get_path().vertices))
            if v[:, 1].min() <= hi and v[:, 1].max() >= lo:
                xs += [v[:, 0].min(), v[:, 0].max()]
        out = []
        for x in sorted(xs):
            if out and x - out[-1][1] < gap:
                out[-1][1] = x
            else:
                out.append([x, x])
        return [(a - pad, b + pad) for a, b in out]

    def break_line(y, x0=X0, x1=x_break_end, pitch=0.9, amp=0.045):
        """A long break line at page y, drawn only through empty page."""
        free, cur = [], x0
        for a, b in occupied(y, tol=1.5 * amp * dpx):
            if b <= x0 or a >= x1:
                continue
            if a > cur:
                free.append((cur, a))
            cur = max(cur, b)
        if cur < x1:
            free.append((cur, x1))
        for a, b in free:
            xs, ys = [a], [y]
            for xc in np.arange(a + 0.5 * pitch, b - 0.2, pitch):
                for dx, dy in ((-0.03, 0.0), (-0.01, -amp * dpx),
                               (0.01, amp * dpx), (0.03, 0.0)):
                    xs.append(xc + dx)
                    ys.append(y + dy)
            xs.append(b)
            ys.append(y)
            break_lines.append(ax.plot(xs, ys, color="#555555", lw=0.6,
                                       zorder=6, solid_joinstyle="miter")[0])

    # the depth spine is broken across every pause, with // marks at its edges
    if dmap.spans():
        ax.spines["left"].set_visible(False)
        edges = [ytop] + [y for y0, y1, _v in dmap.spans() for y in (y0, y1)] \
            + [ybot]
        for a, b in zip(edges[0::2], edges[1::2]):
            ax.plot([0.0, 0.0], [a, b], transform=spine, color="black", lw=0.8,
                    clip_on=False, zorder=9)
    for y0, y1, _v in dmap.spans():
        for yb in (y0, y1):
            for dx in (-0.012, 0.004):
                ax.plot([dx, dx + 0.016], [yb + 0.004 * (ybot - ytop),
                                           yb - 0.004 * (ybot - ytop)],
                        transform=spine, color="#1a1a1a", lw=0.8,
                        clip_on=False, zorder=9)

    # ---- open hole, per bore, with a ledge at each section TD -----------------
    hole_col = {n: ("#5a4a3a" if parent[n] is None else col[n]) for n in names}
    for n in names:
        secs = sorted(((h.bit_in, h.top_md, h.base_md)
                       for h in bore_of[n].hole_sections), key=lambda r: r[1])
        if not secs:
            caveats.append(f"{short[n]}: no hole section recorded -- no hole "
                           "drawn")
            continue
        if parent[n] is None and secs[0][1] > 0.0:
            caveats.append(f"{short[n]}: no hole recorded above "
                           f"{secs[0][1]:.1f} m")
        drawn = []
        for i, (bit, top, base) in enumerate(secs):
            if i + 1 < len(secs) and base > secs[i + 1][1]:
                caveats.append(
                    f"{short[n]} {bit:.2f}in hole recorded to {base:.0f} m, "
                    f"overlapping the next section from {secs[i + 1][1]:.0f} m"
                    f" -- drawn to {secs[i + 1][1]:.0f} m")
                base = secs[i + 1][1]
            if i == 0 and parent[n] and top > kop[n]:
                top = kop[n]                         # drawn from the branch point
            draw_element(n, top, base, bit * S, color=hole_col[n], lw=1.0,
                         zorder=3)
            drawn.append((bit, top, base))
        for (b0, _t0, e0), (b1, t1, _e1) in zip(drawn[:-1], drawn[1:]):
            if abs(e0 - t1) < 1e-6 and b1 < b0:     # the ledge, both sides
                q = at_md(n, t1)
                for sgn in (-1, 1):
                    ax.plot([q[0] + sgn * 0.5 * b1 * S, q[0] + sgn * 0.5 * b0 * S],
                            [q[1], q[1]], color=hole_col[n], lw=1.0, zorder=3)
        q = at_md(n, td[n])
        ax.plot([q[0] - 0.5 * drawn[-1][0] * S, q[0] + 0.5 * drawn[-1][0] * S],
                [q[1]] * 2, color=hole_col[n], lw=1.0, zorder=3)

    # ---- cement plugs -----------------------------------------------------------
    # A plug a branch was drilled out of stops at the branch's inner hole wall,
    # not at the kick-off depth: drawn to the kick-off it would read as cement
    # isolating the branch. That face is drawn in the branch's colour.
    for n in names:
        for pl in bore_of[n].cement_plugs:
            top, base = pl.top_md, pl.base_md
            w = width_at(n, 0.5 * (top + base))
            xl, xr = lane_x[n] - 0.5 * w, lane_x[n] + 0.5 * w
            yb = dmap.y(n, base, "hi")
            face, fc = [], None
            for c in children[n]:
                if top - 1e-6 <= kop[c] <= base:
                    inner, _ = walls(conn[c], widest[c] * S)
                    face = [q[:2] for q in inner
                            if xl - 1e-9 <= q[0] <= xr + 1e-9]
                    fc = c
            if face:
                zr = face[-1][1]
                poly = [(xl, face[0][1])] + face + [(xr, zr), (xr, yb), (xl, yb)]
                ax.add_patch(plt.Polygon(poly, closed=True, facecolor="#bbbbbb",
                                         edgecolor="none", alpha=0.8, zorder=2))
                ax.plot([q[0] for q in face], [q[1] for q in face],
                        color=col[fc], lw=1.1, zorder=3)
            else:
                yt = dmap.y(n, top, "lo")
                ax.add_patch(plt.Rectangle((xl, yt), w, yb - yt,
                                           facecolor="#bbbbbb", edgecolor="none",
                                           alpha=0.8, zorder=2))
            ym = dmap.y(n, 0.5 * (top + base))
            right.append(dict(y=ym, anchor=(xr + 0.02, ym), lines=1, size=6.5,
                              color="#444444",
                              text=f"{short[n]} plugged back {top:.0f}-{base:.0f}"
                                   " mMD",
                              leader=dict(color="#bbbbbb", lw=0.5, ls=":")))

    # ---- strings ------------------------------------------------------------------
    # casing: solid, shoe triangles, cement. liner: solid, no shoe mark.
    # screen: dashed walls.
    for n in names:
        for nm, kind, od, _top, shoe, steel, cement in strings[n]:
            w = od * S
            style = dict(color="#2b2b2b", lw=1.2, zorder=5)
            if kind == "screen":
                style.update(ls=(0, (4, 2)))
            pts = None
            for a, b in steel:
                pts = draw_element(n, a, b, w, kind="string", **style)
            for a, b in cement:
                cement_annulus(n, a, b, od)
                qt = at_md(n, a)
                right.append(dict(
                    y=qt[1],
                    anchor=(qt[0] + 0.5 * outer_width(n, a + 0.5, od) + 0.02,
                            qt[1]),
                    lines=1, size=6.0, color="#555555",
                    text=f"TOC {a:.0f} mMD behind the {_string_label(nm)}",
                    leader=dict(color="#cccccc", lw=0.5, ls=":")))
            if not cement and kind == "casing":
                caveats.append(f"{short[n]} {nm}: no cement recorded -- none "
                               "drawn")
            if pts is None or not steel or steel[-1][1] < shoe - 1e-6:
                continue                     # no steel at the shoe to mark
            xe, ye = pts[-1][0], pts[-1][1]
            if kind == "casing":
                shoe_triangles(xe, ye, w)
                label = f"{_string_label(nm)}  shoe {shoe:.0f} mMD"
            elif kind == "screen":
                label = f"{short[n]} {nm} {steel[0][0]:.0f}-{shoe:.0f} mMD"
            else:
                continue                             # liner pieces unlabelled
            right.append(dict(y=ye, anchor=(xe + 0.5 * w + 0.07, ye), lines=1,
                              size=6.6, text=label,
                              leader=dict(color="#bbbbbb", lw=0.5, ls=":")))

    # ---- completion: tubing, packers, hangers ---------------------------------
    for n in names:
        for c in bore_of[n].completion:
            if c.type == "tubing":
                draw_element(n, c.top_md, c.base_md, c.od_in * S, kind="string",
                             color="#2b5c9e", lw=1.0, zorder=5)

    packer_boxes: list = []

    def xbox(n, md, inner_od, outer_od, label):
        """Packer / hanger boxes with an X, both sides of n's path at ``md``."""
        x, y = at_md(n, md)[:2]
        wi, wo = inner_od * S, outer_od * S
        hb = 0.5 * (wo - wi) * dpx * 1.15
        for sgn in (-1, 1):
            xa, xb = sorted((x + sgn * 0.5 * wi, x + sgn * 0.5 * wo))
            ax.add_patch(plt.Rectangle((xa, y), xb - xa, hb, facecolor="#ffffff",
                                       edgecolor="#1a1a1a", lw=0.9, zorder=8))
            ax.plot([xa, xb], [y, y + hb], color="#1a1a1a", lw=0.8, zorder=9)
            ax.plot([xa, xb], [y + hb, y], color="#1a1a1a", lw=0.8, zorder=9)
        packer_boxes.append((n, md, y, y + hb))
        right.append(dict(y=y + 0.5 * hb, anchor=(x + 0.5 * wo + 0.02, y + 0.5 * hb),
                          lines=1, size=6.3, text=label,
                          leader=dict(color="#999999", lw=0.5)))

    for n in names:
        for pk in [c for c in bore_of[n].completion if c.type == "packer"]:
            nm = _string_label(pk.name or "packer")[:34]
            host = host_strings(n, pk.md)
            if pk.od_in is None:
                if not host:
                    caveats.append(f"{short[n]} {nm} at {pk.md:.0f} m: no OD "
                                   "recorded and no host casing -- not drawn")
                    continue
                seal = min(i for _o, i in host)
            else:
                seal = pk.od_in
            outer = [o for o, _i in host if o >= seal - 1e-6]
            inner, why = packer_inner_od(n, pk, seal)
            if inner is None:
                caveats.append(f"{short[n]} {nm} at {pk.md:.0f} m: no string to "
                               "hang or seal on -- not drawn")
                continue
            if pk.od_in is None:
                caveats.append(f"{nm} at {pk.md:.0f} m: OD not recorded -- "
                               f"drawn to its host casing ID {seal:.3f}in "
                               "[DERIVED]")
            if why:
                caveats.append(f"{nm} at {pk.md:.0f} m: {why}")
            # the box meets the host casing's WALL as drawn (its OD)
            xbox(n, pk.md, inner, min(outer) if outer else seal,
                 f"{nm}  {pk.md:.0f} mMD ({short[n]})")

    # ---- branch labels: one per branch, in its colour ---------------------------
    for c, k in kop.items():
        xm, ym = conn[c][len(conn[c]) // 2][:2]
        right.append(dict(
            y=ym, anchor=(xm, ym), lines=2, size=6.6, color=col[c],
            text=f"{short[c]} leaves {short[parent[c]]}  {k:.0f} mMD"
                 + (f"\n{'TVD' if mode == 'tvd' else 'MD'} paused across the "
                    "connector" if dmap.paused
                    else f"\nconnector spans {depth_md[c]:.0f} m MD"),
            leader=dict(color=col[c], lw=0.5, ls=":")))

    # break lines last among the drawing, so every artist they must avoid
    # exists; the well is snapshotted first so later leaders are not the well
    well.update(ax.lines)
    well.update(ax.patches)
    for y0, y1, _v in dmap.spans():
        for yb in (y0, y1):
            break_line(yb)

    # ---- the label column: every label spread together ---------------------------
    px = (ybot - ytop) / (AX_H * FIG_H * 72.0)             # page y per point
    rows = sorted(right, key=lambda r: r["y"])
    half = [0.5 * r["lines"] * 1.3 * r["size"] * px for r in rows]
    ys: List[float] = []
    for i, r in enumerate(rows):
        ys.append(max(r["y"], ys[-1] + half[i - 1] + half[i]) if ys
                  else r["y"])
    over = ys[-1] + half[-1] - ybot if ys else 0.0
    if over > 0.0:                          # push the stack back up from the bottom
        ys[-1] -= over
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - half[i + 1] - half[i])
    for r, yl in zip(rows, ys):
        if r.get("arrow"):
            ax.annotate(r["text"], xy=r["anchor"], xytext=(x_lbl, yl),
                        fontsize=r["size"], va="center", color="#2b2b2b",
                        ha="left",
                        arrowprops=dict(arrowstyle="->", lw=0.7, color="#2b2b2b"))
            continue
        ax.plot([r["anchor"][0], x_lbl - 0.05], [r["anchor"][1], yl], zorder=1,
                **r["leader"])
        ax.text(x_lbl, yl, r["text"], fontsize=r["size"], va="center",
                color=r.get("color", "#2b2b2b"))

    # ---- TDs: vertical, centred in each bore's lane, below TD ---------------------
    for n in names:
        x, y = at_md(n, td[n])[:2]
        ax.text(x, y + 0.006 * (ybot - ytop), f"{short[n]}\nTD {td[n]:.0f} mMD",
                rotation=90, ha="center", va="top", fontsize=7.0, color=col[n],
                linespacing=1.1)

    # ---- the axis: every tick at true depth ---------------------------------------
    # labelled majors, semi-minors at half the step, minors at a tenth
    step = 250.0 if mode == "tvd" else 500.0
    d0, d1 = np.ceil(dmap.domain_at(ytop)), dmap.domain_at(ybot)
    ticks = dmap.ticks(step, d0, d1)
    ax.set_yticks([y for y, _ in ticks])
    ax.set_yticklabels([f"{d:.0f}" for _, d in ticks])

    def _on(d, k):
        return abs(d / k - round(d / k)) < 1e-9
    semi = [y for y, d in dmap.ticks(step / 2.0, d0, d1) if not _on(d, step)]
    ax.set_yticks([y for y, d in dmap.ticks(step / 10.0, d0, d1)
                   if not _on(d, step / 2.0)], minor=True)
    ax.tick_params(axis="y", which="major", length=6.0, width=0.8)
    ax.tick_params(axis="y", which="minor", length=2.0, width=0.5)
    semi_len = 4.0 / (AX_W * FIG_W * 72.0)          # 4 pt, as a fraction of axes
    for y in semi:
        ax.plot([-semi_len, 0.0], [y, y], transform=spine, color="black",
                lw=0.7, clip_on=False, zorder=9)
    if grid:
        for y, _d in ticks:
            ax.axhline(y, xmax=(x_lbl - 0.15 - X0) / (X1 - X0),
                       color="#ececec", lw=0.5, zorder=0)
    ref = schematic.well.depth_reference
    ax.set_ylabel({"md": f"MD along the bore (m {ref})",
                   "md-paused": f"MD along the bore (m {ref})",
                   "tvd": f"TVDSS (m; {ref} {datum or 0.0:.1f} m)"}[mode])
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_title({
        "md": f"{wellname} nest in MD\nlane offset DIAGRAMMATIC; one radial "
              f"scale for every diameter; each connector spans <= {D_ARC:.0f} "
              "m MD",
        "md-paused": f"{wellname} nest in MD, clock PAUSED at each branch\n"
                     "lane offset DIAGRAMMATIC; one radial scale for every "
                     "diameter; each connector drawn in a pause that carries "
                     "no MD",
        "tvd": f"{wellname} nest in TVDSS (minimum curvature)\nlane offset "
               "DIAGRAMMATIC; connectors drawn in pauses; a lateral overdraws "
               "itself where it climbs"}[mode], fontsize=8.5)
    if caveats:
        fig.text(0.02, 0.012, "   ·   ".join(caveats), fontsize=6.3,
                 color="#a33", wrap=True)
    return NestView(
        fig=fig, ax=ax, mode=mode, depth_map=dmap, paths=paths,
        connectors=conn, branch_md=kop, connector_md=depth_md, parent=parent,
        packer_boxes=packer_boxes, break_lines=break_lines, well_artists=well,
        dpx=dpx, scale=S, lane_x=lane_x, caveats=caveats, occupied=occupied)
