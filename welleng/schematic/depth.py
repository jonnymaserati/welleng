"""Survey-driven MD<->TVD depth resolver.

Builds a :class:`welleng.survey.Survey` from a lean :class:`SurveyRef` and
exposes interpolators for depth (MD or TVD), N/E/TVD position, and
vertical-section departure. Also builds a :class:`RadialScale` from a bore's
hole sections in the chosen depth domain (so section boundaries land in the
same coordinates the generators emit).

Imports numpy + welleng (both core deps) -- but NOT any renderer.
"""
from __future__ import annotations


import warnings

import numpy as np

import welleng as we

from .drawing import RadialScale
from .models import SurveyRef, Wellbore


#: Refuse, rather than warn, when asked for a depth outside the survey's own
#: MD range. See :meth:`DepthResolver._check_range` for why the default is a
#: warning and why that is a migration courtesy rather than a judgement.
STRICT_RANGE = False


class DepthResolver:
    """Resolve depth relationships from a survey via minimum curvature.

    ⚠️ **A survey does not necessarily start at surface.** A sidetrack's
    definitive survey begins at its TIE-IN, and two things follow that this
    class used to get silently wrong:

    1. Asked for an MD *above* the first station, ``np.interp`` returns the
       first value rather than declining -- so a shoe at 161 m came back as
       TVD 0.0, and one below the tie-in came back as its depth measured FROM
       THE TIE-IN. Out-of-range asks are now reported (see
       :data:`STRICT_RANGE`), and :attr:`md_min` / :attr:`md_max` /
       :meth:`covers` let a caller test before asking.
    2. Without a tie-in TVD the survey is built from zero, so EVERY TVD it
       yields -- in range as well as out -- is measured from the first station
       rather than from the datum. Pass ``SurveyRef.tie_in_tvd`` to fix that;
       a survey starting below MD 0 without one is reported.

    Neither is something this class can repair on its own: the trajectory
    above the first station belongs to the parent wellbore, which a resolver
    holding one ``SurveyRef`` has no handle on. Composing the lineage is the
    caller's job; saying so instead of returning a plausible number is this
    class's.
    """

    def __init__(
        self,
        survey_ref: SurveyRef,
        step: float = 10.0,
        name: str = "schematic",
    ) -> None:
        md = np.asarray(survey_ref.md, dtype=float)
        self.step = float(step)
        # Dense, uniform MD grid for smooth interpolation, CLIPPED to the last
        # station: `arange(lo, hi + step, step)` overshoots `hi` whenever the
        # span is not a whole number of steps, and np.interp clamps, so the
        # overshoot is a tail of fabricated hold section -- and md_max would
        # then overstate what the survey actually covers.
        grid = np.arange(md.min(), md.max() + step, step)
        grid = grid[grid <= md.max()]
        if grid[-1] < md.max():
            grid = np.append(grid, md.max())
        grid = np.unique(grid)
        # ⛔ NOT np.interp on inc/azi. They are ANGLES on a circular arc, and
        # azimuth wraps at 0/360, so a linear blend across the wrap swings the
        # tangent through the opposite heading. This module densified the
        # survey linearly and then handed the result to minimum curvature,
        # which made the whole trajectory a min-curve path through blended
        # angles rather than the min-curve interpolation of the real stations.
        # welleng.survey.interpolate_mds IS that interpolation -- core's own
        # primitive, ignored here.
        # The tie-in. A survey starting below MD 0 whose first station's TVD is
        # unrecorded cannot be referenced to the datum -- and the failure is
        # silent, because a run-relative TVD is a well-formed number.
        tie_in = survey_ref.tie_in_tvd
        self.tie_in_tvd = tie_in
        if tie_in is None and float(grid[0]) > 0.0:
            warnings.warn(
                f"survey starts at MD {float(grid[0]):.1f} m, not at surface, "
                "and SurveyRef.tie_in_tvd is not set: every TVD from this "
                "resolver is measured FROM THE FIRST STATION, not from the "
                "datum, and is short by the tie-in TVD. Set "
                "SurveyRef.tie_in_tvd to reference it to the datum.",
                stacklevel=2,
            )
        # NB pass start_nev only when there IS one: `start_nev=None` is not the
        # same as omitting it -- the header stores the None and the position
        # maths then indexes a 0-d array.
        kwargs = {} if tie_in is None else {"start_nev": [0.0, 0.0, float(tie_in)]}
        stations = we.survey.Survey(
            md=md,
            inc=np.asarray(survey_ref.inc, dtype=float),
            azi=np.asarray(survey_ref.azi, dtype=float),
            header=we.survey.SurveyHeader(name=name),
            **kwargs,
        )
        self._stations = stations           # the REAL survey; lookups use this
        self.survey = we.survey.interpolate_mds(stations, grid)
        # interpolate_mds merges the true stations into the grid, so take the
        # md array it actually produced rather than assuming it is `grid`.
        self.md = np.asarray(self.survey.md, dtype=float)
        self.tvd = np.asarray(self.survey.tvd, dtype=float)
        self.n = np.asarray(self.survey.n, dtype=float)
        self.e = np.asarray(self.survey.e, dtype=float)

    # --- what this survey actually covers ---------------------------------
    @property
    def md_min(self) -> float:
        """Shallowest MD this survey covers. Above it there is no trajectory."""
        return float(self.md[0])

    @property
    def md_max(self) -> float:
        """Deepest MD this survey covers."""
        return float(self.md[-1])

    def covers(self, md) -> bool:
        """Whether ``md`` (scalar or array) lies within the surveyed range."""
        a = np.asarray(md, dtype=float)
        return bool(np.all((a >= self.md_min) & (a <= self.md_max)))

    def _check_range(self, md, what: str):
        """Report an ask outside the surveyed interval.

        ``np.interp`` clamps, which for a depth lookup means returning a
        confident number for a depth the survey says nothing about -- the
        first station's value for anything above it, the last station's for
        anything below. Both are fabrications, and the shallow one is the
        dangerous shape: plausible, in range for the well, wrong.

        Warn by default rather than raise ONLY as a migration courtesy -- a
        renderer legitimately asks for a band edge a little past TD, and
        raising would take the whole drawing out for a cosmetic overrun. Set
        :data:`STRICT_RANGE` to refuse.
        """
        a = np.asarray(md, dtype=float)
        if a.size == 0 or (np.all(a >= self.md_min) and np.all(a <= self.md_max)):
            return
        lo, hi = float(np.min(a)), float(np.max(a))
        msg = (
            f"{what}: MD range [{lo:.1f}, {hi:.1f}] m falls outside the "
            f"survey's own [{self.md_min:.1f}, {self.md_max:.1f}] m. The "
            "value returned is the nearest station's, which for an MD above "
            "the first station is that station's depth and not the depth "
            "asked for. A survey that starts at a tie-in does not describe "
            "the well above it -- compose the parent wellbore's survey to "
            "cover that interval."
        )
        if STRICT_RANGE:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)

    # --- scalar/array interpolators ---------------------------------------
    # These delegate to minimum curvature on the REAL stations -- they are
    # EXACT, not interpolated off the dense grid.
    #
    # The dense grid was tempting: at a 10 m step its off-grid residual
    # measured 8.2 mm on a 480 m-station build-and-hold, which is far inside
    # any line width this module draws. That argument is the SAME one that let
    # the 14.4 m defect ship -- the residual scales with the CALLER's dogleg
    # and with `step`, neither of which this class controls, so "small on the
    # survey I happened to measure" says nothing about the next survey. The
    # grid stays for drawing a polyline; it is not what answers a question.
    def _at(self, md):
        """Exact minimum-curvature (N, E, TVD) at ``md``, scalar or array.

        NB ``interpolate_mds`` MERGES the real stations into whatever it is
        asked for, so the result is longer than the query and the requested
        rows have to be selected back out by MD -- it is not 1:1.
        """
        a = np.atleast_1d(np.asarray(md, dtype=float))
        a = np.clip(a, self.md_min, self.md_max)   # range already reported
        out = we.survey.interpolate_mds(self._stations, a)
        om = np.asarray(out.md, dtype=float)
        pos = np.asarray(out.pos_nev, dtype=float)
        idx = np.searchsorted(om, a)
        idx = np.clip(idx, 0, om.size - 1)
        # guard the float-equality assumption rather than trusting it
        off = np.abs(om[idx] - a)
        bad = off > 1e-6
        if np.any(bad):
            alt = np.clip(idx - 1, 0, om.size - 1)
            use_alt = np.abs(om[alt] - a) < off
            idx = np.where(use_alt, alt, idx)
        return pos[idx]

    def tvd_at(self, md):
        """Minimum-curvature TVD (m) at ``md``, scalar or array.

        Out-of-range MDs are reported (see :data:`STRICT_RANGE`) and clamped
        to the surveyed range.
        """
        self._check_range(md, "tvd_at")
        out = self._at(md)[:, 2]
        return out if np.ndim(md) else float(out[0])

    def depth(self, md, mode: str = "MD"):
        """Return the plotting depth for ``md`` in ``'MD'`` or ``'TVD'`` mode."""
        if mode.upper() == "TVD":
            return self.tvd_at(md)
        self._check_range(md, "depth")
        out = np.clip(np.asarray(md, dtype=float), self.md_min, self.md_max)
        return out if np.ndim(md) else float(out)

    def pos(self, md):
        """(N, E, TVD) at ``md``."""
        self._check_range(md, "pos")
        p = self._at(md)
        if np.ndim(md):
            return (p[:, 0], p[:, 1], p[:, 2])
        return (float(p[0, 0]), float(p[0, 1]), float(p[0, 2]))

    def vs(self, md, azimuth_deg: float):
        """Vertical-section departure at ``md`` projected onto ``azimuth_deg``."""
        self._check_range(md, "vs")
        a = np.radians(azimuth_deg)
        p = self._at(md)
        out = p[:, 0] * np.cos(a) + p[:, 1] * np.sin(a)
        return out if np.ndim(md) else float(out[0])

    @property
    def total_depth(self):
        """Deepest MD (m) of the survey; same value as :attr:`md_max`."""
        return float(self.md[-1])

    def max_depth(self, mode: str = "MD") -> float:
        """Plotting depth (m) of the deepest station in ``mode``."""
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
