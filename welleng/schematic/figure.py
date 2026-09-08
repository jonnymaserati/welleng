"""High-level composer: :class:`WellFigure`.

Stacks shared-depth tracks into one :class:`Drawing` and renders it through
any backend, in either ``MD`` or ``TVD`` depth mode.

No renderer imports at module scope -- backends are dispatched lazily.
"""
from __future__ import annotations

from typing import List, Optional

from .backends import to_dxf, to_matplotlib, to_pdf, to_png, to_svg
from .depth import DepthResolver
from .drawing import Drawing, ViewTransform
from .models import WellSchematic
from .tracks import (
    DepthLayout,
    DepthTrack,
    IntervalsTrack,
    LithologyTrack,
    PPFPTrack,
    SchematicTrack,
    Track,
)

_BACKENDS = {
    "dxf": to_dxf,
    "svg": to_svg,
    "pdf": to_pdf,
    "png": to_png,
    "matplotlib": to_matplotlib,
    "mpl": to_matplotlib,
}


class WellFigure:
    """Compose tracks against a shared depth axis and render to any backend."""

    def __init__(
        self,
        schematic: WellSchematic,
        mode: str = "MD",
        tracks: Optional[List[Track]] = None,
        target_height: float = 250.0,
        spacing: float = 6.0,
        step: float = 10.0,
    ):
        self.schematic = schematic
        self.mode = mode.upper()
        self.spacing = spacing
        self.target_height = target_height
        self.resolver = DepthResolver(schematic.primary.survey, step=step,
                                      name=schematic.well.name)
        self.tracks = tracks if tracks is not None else self._default_tracks()
        self._drawing: Optional[Drawing] = None

    def _default_tracks(self) -> List[Track]:
        tracks: List[Track] = [
            DepthTrack(),
            SchematicTrack(self.schematic),
            LithologyTrack(self.schematic),
            IntervalsTrack(self.schematic),
        ]
        if self.schematic.pressures is not None:
            tracks.append(PPFPTrack(self.schematic))
        return tracks

    def build(self) -> Drawing:
        """Build (and cache) the composite :class:`Drawing`."""
        ymax = self.resolver.max_depth(self.mode)
        v_scale = self.target_height / max(ymax, 1e-6)
        layout = DepthLayout(mode=self.mode, resolver=self.resolver,
                             v_scale=v_scale, ymax=ymax)
        dwg = Drawing(name=f"{self.schematic.well.name}_figure_{self.mode}")
        dwg.transform = ViewTransform(h_scale=1.0, v_scale=1.0, flip_y=True)
        x0 = 0.0
        for track in self.tracks:
            track.build(dwg, layout, x0)
            x0 += track.width + self.spacing
        dwg.set_title_block(
            title=self.schematic.well.name,
            view=f"Composite ({self.mode})",
            tracks=str(len(self.tracks)),
        )
        self._drawing = dwg
        return dwg

    @property
    def drawing(self) -> Drawing:
        if self._drawing is None:
            self.build()
        return self._drawing

    def render(self, backend: str, path: Optional[str] = None):
        """Render via ``backend`` in {dxf, svg, pdf, png, matplotlib}."""
        key = backend.lower()
        if key not in _BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; "
                f"choose from {sorted(_BACKENDS)}"
            )
        fn = _BACKENDS[key]
        if key in ("matplotlib", "mpl"):
            return fn(self.drawing)
        if path is None:
            raise ValueError(f"backend {backend!r} needs an output path")
        return fn(self.drawing, path)
