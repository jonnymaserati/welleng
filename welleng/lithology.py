"""Lithology / stratigraphy column rendering.

Draws a lithostratigraphic column against depth -- a flat colour per interval,
optionally overlaid with the FGDC lithologic pattern for that rock type -- and
composes it beside a log plot (:func:`welleng.exchange.las.plot_curves`) or a
well schematic.

Reference data
--------------
:func:`nl_groups` returns the Dutch (NL) lithostratigraphic groups shipped in
``welleng/data/nl_lithology_patterns.yaml``: group name, colour, and an FGDC
section-37 pattern code. Colours are a presentation choice; the pattern codes
are from FGDC-STD-013-2006 (US public domain).

Pattern images
--------------
The FGDC pattern PNGs are NOT vendored -- they are a separate CC0 repackaging
(``github.com/davenquinn/geologic-patterns``). Point :class:`FgdcPatterns` at
its ``assets/png`` directory, or set ``WELLENG_FGDC_PATTERNS``. Without them the
column still renders, in flat colour: patterns are an enhancement, never a
requirement (a caller with no assets must still get a usable figure).

``matplotlib`` and ``PIL`` are lazy imports behind the ``welleng[las]`` extra.

Example
-------
>>> from welleng.lithology import nl_groups, plot_lithology
>>> plot_lithology(nl_groups(), out="litho.png")          # doctest: +SKIP
"""
from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Iterable, Sequence

import numpy as np

_DATA = "nl_lithology_patterns.yaml"
ENV_PATTERNS = "WELLENG_FGDC_PATTERNS"

# Roughly how many pattern tiles should stack down the FULL height of the
# drawn axis. Tile height in depth units is therefore (visible span) / this,
# which keeps a pattern the same size on paper whatever the depth window --
# the point being that tiles must NOT be sized from interval thickness against
# a fixed depth constant, or zooming in stretches a single tile over the whole
# interval (a 200 m salt band at a 280 m window rendered as tall rectangles
# instead of cross-hatch). Taste, not standard.
TILES_PER_AXIS = 10.0


class LithologyError(Exception):
    """Lithology reference data or pattern assets could not be used."""


@dataclass
class Interval:
    """One lithostratigraphic interval. ``pattern`` is an FGDC section-37 code."""

    name: str
    top: float
    base: float
    colour: str = "#ffffff"
    pattern: int | None = None
    note: str = ""

    @property
    def thickness(self) -> float:
        return self.base - self.top


@lru_cache(maxsize=1)
def _reference() -> dict[str, Any]:
    import yaml                       # PyYAML is a core dependency
    return yaml.safe_load((files("welleng") / "data" / _DATA).read_text())


def pattern_names() -> dict[int, str]:
    """FGDC code -> the standard's own wording, for the codes we reference."""
    return dict(_reference()["patterns"])


def nl_groups() -> list[Interval]:
    """The Dutch lithostratigraphic groups, shallow to deep.

    Tops/bases are NOT included -- they are per-well. Each interval comes back
    with ``top``/``base`` of 0.0; assign them, or use :func:`intervals_from_tops`.
    """
    return [
        Interval(name=g["name"], top=0.0, base=0.0, colour=g["colour"],
                 pattern=g.get("pattern"), note=g.get("note", ""))
        for g in _reference()["groups"]
    ]


def intervals_from_tops(tops: Sequence[tuple[str, float]], base: float,
                        lookup: Iterable[Interval] | None = None) -> list[Interval]:
    """Build intervals from ``(name, top_depth)`` pairs plus a final ``base``.

    Colour and pattern are taken from ``lookup`` (default: the NL groups) by
    case-insensitive name match; an unmatched name renders in flat white, which
    is deliberate -- an unknown unit should look unknown, not be given a
    plausible colour it has not earned.
    """
    ref = {i.name.lower(): i for i in (lookup if lookup is not None else nl_groups())}
    ordered = sorted(tops, key=lambda t: t[1])
    out: list[Interval] = []
    for idx, (name, top) in enumerate(ordered):
        bot = ordered[idx + 1][1] if idx + 1 < len(ordered) else base
        match = ref.get(str(name).lower())
        out.append(Interval(
            name=str(name), top=float(top), base=float(bot),
            colour=match.colour if match else "#ffffff",
            pattern=match.pattern if match else None,
            note=match.note if match else "",
        ))
    return out


def nl_groups_by_code() -> dict[str, Interval]:
    """RGD group-rank code -> template :class:`Interval` (colour + pattern)."""
    return {
        g["rgd_group"]: Interval(
            name=g["name"], top=0.0, base=0.0, colour=g["colour"],
            pattern=g.get("pattern"), note=g.get("note", ""),
        )
        for g in _reference()["groups"] if g.get("rgd_group")
    }


def group_of(unit_id: str) -> str | None:
    """Roll an RGD unit code up to its GROUP-rank code.

    A formation or member code resolves through its ancestor chain (the last
    entry is the group); a code already at group rank returns itself. Returns
    None for a code the nomenclature does not know.
    """
    from .exchange import rgd_nomenclature as rgd
    info = rgd.resolve(str(unit_id))
    if not info:
        return None
    if info.get("rank") == "group":
        return info["code"]
    anc = info.get("ancestors") or []
    return anc[-1] if anc else None


def colour(unit_id: str) -> str | None:
    """Colour (hex) for ANY RGD unit code, rolled up to its group. Or ``None``.

    A member or formation inherits its group's colour rather than falling to
    grey, so a consumer plotting a Dutch column does not have to invent one.
    ``None`` for a code the nomenclature does not know, or a group the shipped
    table does not carry -- **deliberately not a fallback palette**: an
    invented colour that looks official is worse than an obviously absent one,
    exactly as with the OCR names.

    ⚠️ **These colours are welleng's presentation choice, NOT a published
    standard.** TNO's DINOloket stratigraphic nomenclature is the authority and
    is not yet ingested here (``docs/dev/NLOG_STRATIGRAPHY_NOMENCLATURE.md``).
    Say so on any figure that uses them.
    """
    g = group_of(unit_id)
    if g is None:
        return None
    hit = nl_groups_by_code().get(g)
    return hit.colour if hit else None


def pattern(unit_id: str) -> int | None:
    """FGDC pattern code for ANY RGD unit code, rolled up to its group.

    Same roll-up and same ``None`` policy as :func:`colour`.

    ⚠️ **One pattern per GROUP is a simplification** -- a group spans more than
    one rock type -- and an ornament carries meaning a colour does not: the
    box/cross-hatch patterns read as EVAPORITE, so ornamenting a carbonate with
    one says salt. Override per interval at formation or member rank where the
    distinction matters.
    """
    g = group_of(unit_id)
    if g is None:
        return None
    hit = nl_groups_by_code().get(g)
    return hit.pattern if hit else None


#: Lithology words that map unambiguously onto a shipped FGDC pattern. Matched
#: on WORD BOUNDARIES only, because some Dutch unit names are proper nouns that
#: contain a lithology word: *Buntsandstein* contains "sandstein" but the unit
#: is claystone-dominated, and a substring match draws it as clean sand.
_NAME_LITHOLOGY: dict[str, int] = {
    "conglomerate": 601, "gravel": 601,
    "sandstone": 607, "sand": 607,
    "silt": 616, "siltstone": 616,
    "claystone": 620, "clay": 620, "shale": 620, "mudstone": 620,
    "marl": 623, "marlstone": 623,          # FGDC 623 IS "calcareous shale or marl"
    "chalk": 626,
    "limestone": 627,
    "dolostone": 642, "dolomite": 642,
    "coal": 658, "lignite": 658,
    "gypsum": 667,
    "salt": 668, "halite": 668,
}

#: Words that name a rock the FGDC chart has no pattern for. Listed so such a
#: name is REFUSED rather than taking the nearest pattern: **anhydrite is not
#: gypsum**, and 667 is gypsum only.
_NAME_UNPATTERNED = ("anhydrite", "tuffite", "tuff", "bentonite")


def pattern_from_name(name: str | None) -> int | None:
    """FGDC pattern implied by a unit NAME, or ``None`` when it does not say.

    Word-boundary matched, and deliberately unwilling:

    * a name with **no** lithology word returns ``None`` -- "Ommelanden
      Formation" says nothing about the rock, so the caller keeps whatever it
      had.
    * a name with **two different** lithologies returns ``None``. Picking the
      first would be a coin toss on a barrier drawing.
    * a name whose rock the FGDC chart has no pattern for returns ``None``
      rather than the nearest one: **anhydrite is not gypsum**, and 667 is
      gypsum only.
    * ⚠️ **Substring matching is wrong here.** *Buntsandstein* contains
      "sandstein" and the Dutch unit logs 127 gAPI -- claystone-dominated. Word
      boundaries are what stop a proper noun being read as a description.

    This is a NAME heuristic, not a lithology record. It exists because the
    shipped table carries one pattern per GROUP, and a group spans more than
    one rock: it is how the Z4 Fringe SANDSTONE Member stops inheriting the
    Zechstein evaporite pattern and being drawn as salt on a barrier drawing.
    """
    if not name:
        return None
    words = set(re.findall(r"[a-z]+", str(name).lower()))
    if words & set(_NAME_UNPATTERNED):
        return None
    hits = {code for w, code in _NAME_LITHOLOGY.items() if w in words}
    return hits.pop() if len(hits) == 1 else None


def intervals_from_nlog(column, label: str = "formation") -> list[Interval]:
    """Colour an NLOG :class:`~welleng.exchange.nlog.StratColumn`.

    ``label``:

    ``"formation"`` (default)
        The unit's own name at formation/member rank, and its own lithology
        where the name says it -- so the Z4 Fringe **Sandstone** Member is
        drawn as sand rather than inheriting the Zechstein evaporite pattern.
        Colour stays the GROUP's, which is what makes a column read as
        stratigraphy rather than as a rock-type mosaic.
    ``"group"``
        Group name and group pattern throughout -- the coarse view.
    ``"code"``
        The raw RGD code as logged, for debugging. (``"unit"`` is accepted as
        the old spelling of this.)

    ⭐ **One pattern per GROUP is a simplification and it bites at the worst
    place.** The Rijnland Group is marl AND clay AND sandstone, and a well's
    cap rock is a claystone member inside it; the Zechstein is an evaporite
    group containing a sandstone reservoir. A consumer hit both, and had to
    hand-write an override table to stop a flow zone being shaded as salt on a
    barrier drawing. That override now happens here, from the name, and only
    where the name is unambiguous -- see :func:`pattern_from_name`.

    An unrecognised or unmatched unit renders flat white with no pattern -- an
    unknown unit should look unknown rather than be given a colour it has not
    earned. Intervals with no ``bottom_md`` are dropped, since a band needs a
    base; that is a data gap, not something to interpolate over.
    """
    from .exchange import rgd_nomenclature as rgd

    by_code = nl_groups_by_code()
    out: list[Interval] = []
    for s in getattr(column, "intervals", []):
        top, base = getattr(s, "top_md", None), getattr(s, "bottom_md", None)
        if top is None or base is None:
            continue
        unit = getattr(s, "unit_id", None)
        gcode = group_of(unit) if unit else None
        tmpl = by_code.get(gcode) if gcode else None
        pattern = tmpl.pattern if tmpl else None

        if label == "group":
            name = tmpl.name if tmpl else str(unit or "?")
        elif label in ("code", "unit"):
            name = str(unit or "?")
        else:
            info = rgd.resolve(str(unit)) if unit else None
            name = (info or {}).get("inherited_name") or (
                tmpl.name if tmpl else str(unit or "?"))
            # An inherited name belongs to the PARENT: say so, rather than
            # labelling a member with its formation's name as if it were one.
            if info and not info.get("name") and info.get("name_from"):
                name = f"{name} ({info.get('rank') or 'member'})"
            pattern = pattern_from_name(name) or pattern

        out.append(Interval(
            name=name, top=float(top), base=float(base),
            colour=tmpl.colour if tmpl else "#ffffff",
            pattern=pattern,
            note=f"{unit} -> {gcode}" if gcode else str(unit or ""),
        ))
    return out


class FgdcPatterns:
    """Resolves FGDC pattern codes to RGBA tiles from a CC0 PNG asset directory.

    The PNGs carry NATIVE transparency (background alpha 0, opaque line work),
    so the RGBA is used as-is. Do not derive alpha from luminance -- that path
    was tried and is wrong; it grays the fill and loses the line work.
    """

    def __init__(self, directory: str | os.PathLike | None = None):
        d = directory or os.environ.get(ENV_PATTERNS)
        self.directory = str(d) if d else None

    @property
    def available(self) -> bool:
        return bool(self.directory) and os.path.isdir(self.directory)

    def path(self, code: int) -> str | None:
        if not self.available:
            return None
        p = os.path.join(self.directory, f"{int(code)}.png")
        return p if os.path.exists(p) else None

    @lru_cache(maxsize=64)
    def _load(self, code: int):
        from PIL import Image                      # lazy: welleng[las]
        p = self.path(code)
        if p is None:
            return None
        img = np.asarray(Image.open(p).convert("RGBA"), dtype=float) / 255.0
        if img[..., 3].max() == 0:
            raise LithologyError(f"pattern {code} is fully transparent")
        return img

    def tile(self, code: int, thickness: float, tile_depth: float | None = None):
        """RGBA tile stack for an interval, or None if the asset is absent.

        ``tile_depth`` is the depth interval ONE tile should occupy; repeats are
        ``thickness / tile_depth``. Callers derive it from the visible depth
        window (see :data:`TILES_PER_AXIS`) so the pattern keeps a constant size
        on paper regardless of zoom.
        """
        img = self._load(int(code))
        if img is None:
            return None
        if not tile_depth or tile_depth <= 0:
            return np.tile(img, (1, 1, 1))
        reps = max(1, int(round(abs(thickness) / tile_depth)))
        return np.tile(img, (reps, 1, 1))


def plot_lithology(
    intervals: Sequence[Interval],
    ax=None,
    patterns: FgdcPatterns | str | None = None,
    depth_range: tuple[float, float] | None = None,
    label: bool = True,
    label_width: int = 12,
    out: str | None = None,
    title: str | None = "lithology",
):
    """Draw a lithology column (depth down the y axis).

    ``patterns`` may be an :class:`FgdcPatterns`, a directory, or None (flat
    colour). Returns the Axes.
    """
    try:
        import matplotlib
        if out is not None and ax is None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:                     # pragma: no cover
        raise LithologyError(
            "rendering needs matplotlib - install `welleng[las]`"
        ) from exc

    if not intervals:
        raise LithologyError("no intervals to draw")
    pats = patterns if isinstance(patterns, FgdcPatterns) else FgdcPatterns(patterns)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.0, 9.0))

    # Restrict to the drawn window and CLIP each band to it. Artists outside the
    # axes limits are not free: text is unclipped by default, and a savefig with
    # bbox_inches="tight" then grows the canvas to contain labels for intervals
    # nobody asked to see (measured: a 280 m log window produced a 7500 px
    # figure because a full-well column's labels were still artists).
    if depth_range is not None:
        lo, hi = min(depth_range), max(depth_range)
        drawn = [iv for iv in intervals if iv.base > lo and iv.top < hi]
        if not drawn:
            raise LithologyError("no intervals overlap the requested depth range")
    else:
        lo = min(i.top for i in intervals)
        hi = max(i.base for i in intervals)
        drawn = list(intervals)

    top, base = lo, hi
    # One tile occupies this much depth, derived from the VISIBLE window so the
    # pattern reads the same size whether the axis spans a whole well or 200 m.
    tile_depth = (hi - lo) / TILES_PER_AXIS if hi > lo else None
    for iv in drawn:
        t, b = max(iv.top, lo), min(iv.base, hi)
        if b <= t:
            continue
        ax.add_patch(Rectangle((0, t), 1, b - t, facecolor=iv.colour,
                               edgecolor="0.5", lw=0.3, zorder=1, clip_on=True))
        if iv.pattern is not None:
            tiled = pats.tile(iv.pattern, b - t, tile_depth)
            if tiled is not None:
                im = ax.imshow(tiled, extent=[0, 1, b, t], aspect="auto",
                               zorder=2, interpolation="nearest")
                im.set_clip_on(True)
        if label:
            # break_long_words=False: a unit name is a proper noun and
            # "Lower Buntsa / ndstein Formation" is not a wrap, it is a
            # different word. A name too long for the column overruns it
            # instead, which is visible and fixable by widening the track;
            # a silently mangled name is neither.
            ax.text(0.5, (t + b) / 2.0,
                    textwrap.fill(iv.name, label_width,
                                  break_long_words=False,
                                  break_on_hyphens=False),
                    fontsize=5.6,
                    va="center", ha="center", zorder=3, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=iv.colour,
                              edgecolor="none", alpha=0.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(base, top)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=8)
    if fig is not None:
        ax.set_ylabel("MD (m)")
        fig.tight_layout()
        if out is not None:
            fig.savefig(out, dpi=150, bbox_inches="tight")
    return ax


def plot_log_with_lithology(
    las,
    intervals: Sequence[Interval],
    tracks: Sequence[Sequence[str]] | None = None,
    depth_range: tuple[float, float] | None = None,
    patterns: FgdcPatterns | str | None = None,
    out: str | None = None,
    title: str | None = None,
):
    """Log tracks with a lithology column alongside, on a shared depth axis.

    ``las`` is a :class:`welleng.exchange.las.LasFile`. Returns the Figure.
    """
    try:
        import matplotlib
        if out is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:                     # pragma: no cover
        raise LithologyError(
            "rendering needs matplotlib - install `welleng[las]`"
        ) from exc
    from .exchange.las import _default_tracks, _is_log_scale, _robust_limits

    trk = [list(t) for t in (tracks if tracks is not None else _default_tracks(las))]
    depth = las.depth
    mask = np.isfinite(depth)
    if depth_range is not None:
        mask &= (depth >= depth_range[0]) & (depth <= depth_range[1])
    if not mask.any():
        raise LithologyError("no samples in the requested depth range")

    n = len(trk)
    fig, axes = plt.subplots(
        1, n + 1, sharey=True,
        figsize=(2.4 * n + 2.4, 9.5),
        gridspec_kw={"width_ratios": [1.0] * n + [0.55]},
    )
    axes = np.atleast_1d(axes)
    for ax, track in zip(axes[:n], trk):
        plotted = []
        for m in track:
            try:
                v = las.curve(m)
            except Exception:
                continue
            unit = las.units.get(m)
            ax.plot(v[mask], depth[mask], lw=0.6,
                    label=f"{m}{f' [{unit}]' if unit else ''}")
            plotted.append(v[mask])
        log = _is_log_scale(track)
        if log:
            ax.set_xscale("log")
        lims = _robust_limits(plotted, log)
        if lims is not None:
            ax.set_xlim(*lims)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)

    d = depth[mask]
    lo_w, hi_w = (depth_range if depth_range is not None
                  else (float(np.nanmin(d)), float(np.nanmax(d))))
    plot_lithology(intervals, ax=axes[n], patterns=patterns,
                   depth_range=(lo_w, hi_w), title="lithology")
    idx_unit = las.units.get(las.index_mnemonic)
    axes[0].set_ylabel(
        f"{las.index_mnemonic}{f' [{idx_unit}]' if idx_unit else ''}"
    )
    # The axes are shared, and plot_lithology has already set a deep-at-bottom
    # ylim spanning the WHOLE column. Re-set it to the plotted LOG window --
    # otherwise the log is squeezed into a sliver of a full-well column -- and
    # do NOT invert again here, which would flip depth back to increasing
    # upward.
    axes[0].set_ylim(hi_w, lo_w)
    fig.suptitle(title or las.well.get("WELL") or "log + lithology", fontsize=10)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
