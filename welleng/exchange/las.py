"""LAS well-log reader and track renderer.

Reading is delegated to ``lasio`` (LAS 1.2 / 2.0 / 3.0, wrapped lines, vendor
malformations) rather than reimplemented -- LAS in the wild carries decades of
vendor drift, unlike the frozen dBASE format read by
:mod:`welleng.exchange.winlog`. Both ``lasio`` and ``matplotlib`` are LAZY
imports behind the ``welleng[las]`` extra, so ``import welleng`` stays free of
either.

Accepts a path, raw bytes, or text -- so a log fetched straight from NLOG
(:meth:`welleng.exchange.nlog.NLOGClient.fetch_document`) can be parsed without
touching disk.

Example
-------
>>> from welleng.exchange.las import open_las, plot_curves
>>> las = open_las("path/to/log.las")            # doctest: +SKIP
>>> las.mnemonics()                              # doctest: +SKIP
>>> plot_curves(las, out="log.png")              # doctest: +SKIP
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

# Mnemonic families used only to pick sensible DEFAULT track grouping/scaling.
# A caller passing explicit ``tracks`` bypasses all of this.
_RESISTIVITY = ("RES", "RT", "ILD", "ILM", "LLD", "LLS", "SFL", "MSFL", "RXO",
                "RACE", "RPCE", "AT10", "AT30", "AT90", "RD", "RS")
_POROSITY = ("NPHI", "PHIN", "TNPH", "RHOB", "DEN", "DPHI", "PEF", "DPEF")
_GAMMA = ("GR", "GRD", "SGR", "CGR", "GRAFM", "GR1AX", "CAL", "CALI", "CALCX")


class LasError(Exception):
    """A LAS file could not be read."""


@dataclass
class LasFile:
    """A parsed LAS log. ``curves`` maps mnemonic -> 1-D array."""

    well: dict[str, Any] = field(default_factory=dict)
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    index_mnemonic: str = "DEPT"
    source: str | None = None

    # -- access ----------------------------------------------------------- #
    def mnemonics(self) -> list[str]:
        """Curve mnemonics, index first."""
        return list(self.curves)

    @property
    def depth(self) -> np.ndarray:
        return self.curves[self.index_mnemonic]

    def curve(self, mnemonic: str) -> np.ndarray:
        """One curve, case-insensitively."""
        for k in self.curves:
            if k.upper() == mnemonic.upper():
                return self.curves[k]
        raise LasError(
            f"no curve {mnemonic!r}; available: {', '.join(self.mnemonics())}"
        )

    def describe(self) -> str:
        """One line per curve: mnemonic, unit, description, % non-null."""
        out = []
        for m, v in self.curves.items():
            pct = 100.0 * float(np.isfinite(v).mean()) if v.size else 0.0
            out.append(
                f"{m:<12} {self.units.get(m, ''):<8} {pct:5.1f}% "
                f"{self.descriptions.get(m, '')}"
            )
        return "\n".join(out)


def open_las(source: str | bytes | os.PathLike, **kwargs: Any) -> LasFile:
    """Read a LAS file from a path, raw bytes, or text.

    ``kwargs`` pass through to ``lasio.read`` (e.g. ``ignore_header_errors=True``
    for a badly-formed vendor header).
    """
    try:
        import lasio                     # lazy: welleng[las]
    except ImportError as exc:           # pragma: no cover - env-dependent
        raise LasError(
            "reading LAS needs lasio - install `welleng[las]` (or `uv pip "
            "install lasio`)"
        ) from exc

    name = None
    if isinstance(source, bytes):
        buf: Any = io.StringIO(source.decode("latin-1", "ignore"))
    elif isinstance(source, str) and ("~" in source and "\n" in source):
        buf = io.StringIO(source)        # raw LAS text, not a path
    else:
        buf = os.fspath(source)
        name = str(buf)

    try:
        raw = lasio.read(buf, **kwargs)
    except Exception as exc:
        raise LasError(f"lasio could not read the LAS source: {exc}") from exc

    curves, units, descs = {}, {}, {}
    for c in raw.curves:
        curves[c.mnemonic] = np.asarray(c.data, dtype=float)
        units[c.mnemonic] = c.unit or ""
        descs[c.mnemonic] = c.descr or ""

    well = {}
    for item in raw.well:
        well[item.mnemonic] = item.value

    index = raw.curves[0].mnemonic if len(raw.curves) else "DEPT"
    return LasFile(well=well, curves=curves, units=units, descriptions=descs,
                   index_mnemonic=index, source=name)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _default_tracks(las: LasFile, max_tracks: int = 8) -> list[list[str]]:
    """Group curves into tracks: the conventional families first (gamma,
    resistivity, porosity), then remaining curves grouped BY UNIT.

    Grouping by unit is what stops a tool with several curves in the same
    measurement (three dB amplitudes on a cement log, say) from consuming a
    track each while genuinely different curves fall off the end of the plot.
    """
    mn = [m for m in las.mnemonics() if m != las.index_mnemonic]

    def _match(fams):
        return [m for m in mn if any(m.upper().startswith(f) for f in fams)]

    tracks, used = [], set()
    for fams in (_GAMMA, _RESISTIVITY, _POROSITY):
        grp = [m for m in _match(fams) if m not in used]
        if grp:
            tracks.append(grp)
            used.update(grp)

    by_unit: dict[str, list[str]] = {}
    for m in mn:
        if m in used:
            continue
        unit = (las.units.get(m) or "").upper()
        if not unit:
            tracks.append([m])          # no unit is NOT evidence of a shared
            continue                    # measurement -- give it its own track
        by_unit.setdefault(unit, []).append(m)
    for _unit, grp in by_unit.items():
        tracks.append(grp)

    return tracks[:max_tracks] or [mn[:1]]


def _robust_limits(values: Sequence[np.ndarray], log: bool) -> tuple | None:
    """X limits from the 1st-99th percentile, so a single spike cannot squash
    the scale of the data everyone actually wants to read."""
    finite = np.concatenate([v[np.isfinite(v)] for v in values if v.size]) \
        if values else np.array([])
    if log:
        finite = finite[finite > 0]
    if finite.size < 10:
        return None
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return None
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad) if not log else (max(lo / 1.5, 1e-6), hi * 1.5)


def _is_log_scale(track: Sequence[str]) -> bool:
    return all(any(m.upper().startswith(f) for f in _RESISTIVITY) for m in track)


def plot_curves(
    las: LasFile,
    tracks: Sequence[Sequence[str]] | None = None,
    depth_range: tuple[float, float] | None = None,
    out: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render a conventional multi-track log plot (depth down the Y axis).

    ``tracks`` is a list of tracks, each a list of mnemonics sharing an X axis;
    the default groups gamma/caliper, resistivity (log scale) and porosity.
    Saves to ``out`` when given and returns the matplotlib Figure.
    """
    try:
        import matplotlib                # lazy: welleng[las]
        if out is not None:
            matplotlib.use("Agg")        # headless-safe when writing a file
        import matplotlib.pyplot as plt
    except ImportError as exc:           # pragma: no cover - env-dependent
        raise LasError(
            "rendering LAS needs matplotlib - install `welleng[las]`"
        ) from exc

    tracks = [list(t) for t in (tracks if tracks is not None
                                else _default_tracks(las))]
    depth = las.depth
    mask = np.isfinite(depth)
    if depth_range is not None:
        mask &= (depth >= depth_range[0]) & (depth <= depth_range[1])
    if not mask.any():
        raise LasError("no samples in the requested depth range")

    n = len(tracks)
    fig, axes = plt.subplots(
        1, n, sharey=True,
        figsize=figsize or (2.6 * n + 1.2, 9.5),
    )
    axes = np.atleast_1d(axes)

    for ax, track in zip(axes, tracks):
        plotted = []
        for m in track:
            try:
                v = las.curve(m)
            except LasError:
                continue
            ax.plot(v[mask], depth[mask], lw=0.6,
                    label=f"{m}{f' [{las.units.get(m)}]' if las.units.get(m) else ''}")
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

    idx_unit = las.units.get(las.index_mnemonic)
    axes[0].set_ylabel(
        f"{las.index_mnemonic}{f' [{idx_unit}]' if idx_unit else ''}"
    )
    axes[0].invert_yaxis()               # depth increases downward
    fig.suptitle(title or las.well.get("WELL") or las.source or "LAS", fontsize=10)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=140, bbox_inches="tight")
    return fig
