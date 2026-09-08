"""Survey-driven MD<->TVD depth resolver.

Builds a :class:`welleng.survey.Survey` from a lean :class:`SurveyRef` and
exposes interpolators for depth (MD or TVD), N/E/TVD position, and
vertical-section departure. Also builds a :class:`RadialScale` from a bore's
hole sections in the chosen depth domain (so section boundaries land in the
same coordinates the generators emit).

Imports numpy + welleng (both core deps) -- but NOT any renderer.
"""
from __future__ import annotations


import numpy as np

import welleng as we

from .drawing import RadialScale
from .models import SurveyRef, Wellbore


class DepthResolver:
    """Resolve depth relationships from a survey via minimum curvature."""

    def __init__(
        self,
        survey_ref: SurveyRef,
        step: float = 10.0,
        name: str = "schematic",
    ) -> None:
        md = np.asarray(survey_ref.md, dtype=float)
        self.step = float(step)
        # dense, uniform MD grid for smooth interpolation
        grid = np.arange(md.min(), md.max() + step, step)
        inc = np.interp(grid, md, survey_ref.inc)
        azi = np.interp(grid, md, survey_ref.azi)
        self.survey = we.survey.Survey(
            md=grid, inc=inc, azi=azi,
            header=we.survey.SurveyHeader(name=name),
        )
        self.md = grid
        self.tvd = np.asarray(self.survey.tvd, dtype=float)
        self.n = np.asarray(self.survey.n, dtype=float)
        self.e = np.asarray(self.survey.e, dtype=float)

    # --- scalar/array interpolators ---------------------------------------
    def tvd_at(self, md):
        return np.interp(md, self.md, self.tvd)

    def depth(self, md, mode: str = "MD"):
        """Return the plotting depth for ``md`` in ``'MD'`` or ``'TVD'`` mode."""
        if mode.upper() == "TVD":
            return self.tvd_at(md)
        return np.interp(md, self.md, self.md)  # identity, but clamps to range

    def pos(self, md):
        """(N, E, TVD) at ``md``."""
        return (
            np.interp(md, self.md, self.n),
            np.interp(md, self.md, self.e),
            np.interp(md, self.md, self.tvd),
        )

    def vs(self, md, azimuth_deg: float):
        """Vertical-section departure at ``md`` projected onto ``azimuth_deg``."""
        a = np.radians(azimuth_deg)
        n = np.interp(md, self.md, self.n)
        e = np.interp(md, self.md, self.e)
        return n * np.cos(a) + e * np.sin(a)

    @property
    def total_depth(self):
        return float(self.md[-1])

    def max_depth(self, mode: str = "MD") -> float:
        return float(self.depth(self.md[-1], mode))


def radial_scale_for(
    bore: Wellbore,
    resolver: DepthResolver,
    mode: str = "MD",
    default: float = 40.0,
) -> RadialScale:
    """Build a :class:`RadialScale` from a bore's hole sections.

    Section-top depths are expressed in the requested ``mode`` domain so the
    breakpoints coincide with the y-coordinates the generators emit. Sections
    without an explicit ``radial_scale`` fall back to ``default``.
    """
    sections = sorted(bore.hole_sections, key=lambda s: s.top_md)
    if not sections:
        return RadialScale.uniform(default)
    breaks = [float(resolver.depth(s.top_md, mode)) for s in sections]
    scales = [
        default if s.radial_scale is None else float(s.radial_scale)
        for s in sections
    ]
    return RadialScale(breaks=breaks, scales=scales)
