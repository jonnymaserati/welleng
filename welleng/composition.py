"""Compose a single wellbore's survey sections into one tied survey.

A wellbore is rarely surveyed by a single continuous run of one tool. It is
drilled and surveyed in ordered *sections* (legs), each potentially using a
different survey tool / ISCWSA error model, and each tied on to the end of the
previous section. Naively concatenating the per-section covariances is wrong
in two opposite ways:

- **Restarting the systematic error at every section** understates the
  uncertainty of a section that is really one continuous survey (the
  systematic sensor biases keep accumulating *correlated* down the hole — they
  are one physical realisation, not a fresh draw per section).
- **Treating a genuine tool change as one continuous survey** overstates the
  correlation: a new tool's sensor biases are an *independent* realisation, so
  its systematic error must restart (while the accumulated *position*
  uncertainty of course carries forward).

``SurveyComposition`` ties the sections together with per-component-correct
covariance carry:

1. Consecutive sections sharing a tool (same ``error_model`` *and* ``tool_id``)
   are grouped into **one** :class:`~welleng.survey.Survey` so their systematic
   error accumulates correlated — exactly as if surveyed in one run.
2. At a tool change the next group is tied on at the previous group's end
   position, carrying the accumulated covariance. The new tool's *systematic*
   is independent (restarts); only the accumulated *position* covariance and
   the shared *global* geomagnetic terms carry across the tie.
3. Whether the *global* (declination / B-field, ``DECG`` / ``DBHG``) and
   *systematic* terms are shared across a given tie is controlled by a
   :data:`~welleng.conditioning.ShareMode`, defaulting to the ISCWSA Side-track
   Clearance RP (2022) recommendation: within one wellbore / campaign the
   global geomag terms stay *correlated* across the tie (same geomag date and
   model), while sensor systematic *resets* at a tool change. A section drilled
   in a different campaign (e.g. a side-track years later, different geomag
   secular variation) can be marked ``all_independent``.

The result is one unified :class:`~welleng.survey.Survey` whose total covariance
*and* per-component (``cov_nev_global`` / ``cov_nev_systematic`` /
``cov_nev_random``) breakdown are correctly composed. Preserving that breakdown
is what lets the relative-error / anti-collision framework
(:func:`welleng.conditioning.combine_covariances`) later difference two composed
wellbores with correct cancellation of shared global terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conditioning import ShareMode
from .survey import Survey, SurveyHeader

__all__ = ["SurveySection", "SurveyComposition"]

#: Default ISCWSA error model when a section does not name one.
DEFAULT_ERROR_MODEL = "ISCWSA MWD Rev5.11"


def _model_key(error_model) -> object:
    """Hashable identity for a section's error model.

    Named models are their string; prebuilt model DICTS (e.g. an EDM/COMPASS
    IPM from ``welleng.errors.edm_ipm``) are unhashable, so key on the
    object's identity — two sections share a tool run only when they carry
    the SAME model object, which is the correct correlation semantics for a
    tool-specific imported IPM.
    """
    return error_model if isinstance(error_model, str) else id(error_model)

#: Survey-date gap (years) beyond which an unspecified tie auto-defaults to
#: ``all_independent`` (geomag secular variation makes the global realisation
#: effectively independent between campaigns this far apart).
_DATE_INDEPENDENCE_YEARS = 2.0

_TIE_MD_TOL = 1e-6


@dataclass
class SurveySection:
    """One surveyed section (leg) of a wellbore, tied on to the previous one.

    Provide either raw ``md`` / ``inc`` / ``azi`` arrays (with ``deg`` and an
    optional ``header``) or an existing :class:`~welleng.survey.Survey` via
    ``survey``. The first station of each section must coincide (in measured
    depth) with the last station of the previous section — that shared station
    is the tie-on.

    Parameters
    ----------
    md, inc, azi : array_like, optional
        Section geometry. Ignored if ``survey`` is given.
    survey : welleng.survey.Survey, optional
        An existing survey to take geometry (and, if not overridden, the
        ``error_model`` / ``header``) from.
    deg : bool, default True
        Whether ``inc`` / ``azi`` are in degrees.
    error_model : str, optional
        ISCWSA error model name for this section's tool. Defaults to
        :data:`DEFAULT_ERROR_MODEL`.
    tool_id : str, optional
        Identifier of the physical survey tool / run. Consecutive sections with
        the *same* ``error_model`` and ``tool_id`` are treated as one
        continuous survey (systematic error stays correlated). A change of
        ``tool_id`` (or ``error_model``) marks a tool change / tie. ``None`` is
        treated as "the same unspecified tool continuing" — so if you never say
        the tool changed, it does not.
    survey_date : str, optional
        ``YYYY-MM-DD`` date the section was surveyed. Used only to auto-pick a
        default ``share_mode`` at the tie *before* this section.
    geomag_model : str, optional
        Name of the geomagnetic reference model (e.g. ``"BGGM2020"``). Used only
        to auto-pick a default ``share_mode``.
    share_mode : {'all_independent', 'globals_shared', \
'globals_and_systematic_shared'}, optional
        Explicit override for how the tie *before* this section shares error
        components with the previous group. If ``None``, auto-picked from the
        context keys (see :class:`SurveyComposition`).
    header : welleng.survey.SurveyHeader, optional
        Survey header (geomag field, dip, location) for this section's error
        model. Falls back to the composition-level default.
    """

    md: Optional[ArrayLike] = None
    inc: Optional[ArrayLike] = None
    azi: Optional[ArrayLike] = None
    survey: Optional[Survey] = None
    deg: bool = True
    error_model: Optional[str] = None
    tool_id: Optional[str] = None
    survey_date: Optional[str] = None
    geomag_model: Optional[str] = None
    share_mode: Optional[ShareMode] = None
    header: Optional[SurveyHeader] = None

    def __post_init__(self) -> None:
        if self.survey is not None:
            if not isinstance(self.survey, Survey):
                raise TypeError("`survey` must be a welleng.survey.Survey")
            if self.error_model is None:
                self.error_model = self.survey.error_model
            if self.header is None:
                self.header = self.survey.header
        elif self.md is None or self.inc is None or self.azi is None:
            raise ValueError(
                "SurveySection needs either `survey` or all of md/inc/azi"
            )


@dataclass
class _Group:
    """A run of consecutive same-tool sections, built as one Survey."""

    md: NDArray[np.float64]           # grid-radian geometry (tie station incl.)
    inc_rad: NDArray[np.float64]
    azi_grid_rad: NDArray[np.float64]
    error_model: str
    header: SurveyHeader
    share_mode: ShareMode             # tie BEFORE this group (ignored for [0])
    survey: Survey = field(default=None, repr=False)


class SurveyComposition:
    """Tie a wellbore's ordered survey sections into one continuous survey.

    Parameters
    ----------
    sections : sequence of SurveySection
        Ordered from shallow to deep. Each section ties on to the previous.
    header : welleng.survey.SurveyHeader, optional
        Default header applied to any section that does not carry its own.
    share_mode : ShareMode, optional
        Default share mode for every tie whose section does not set one and
        for which the context keys do not force a choice. Defaults to
        ``"globals_shared"`` (the ISCWSA side-track RP recommendation).

    Notes
    -----
    **Auto share-mode.** When a section's ``share_mode`` is ``None`` the tie
    before it is resolved from context: if the two sides name *different*
    ``geomag_model`` values, or their ``survey_date`` values differ by more
    than two years, the tie is ``all_independent`` (the global geomagnetic
    realisation has drifted / changed model and no longer cancels); otherwise
    the composition ``share_mode`` (default ``globals_shared``) is used.

    **Mixed error models across a shared tie.** The *shared* (globally- or
    systematically-correlated) component of a multi-group run is computed by
    building that run as a single survey using the run's first section's error
    model. This is exact when all groups in the run use the same error model
    (the common case). If they differ, the shared component uses the first
    model as the reference and a warning is issued.
    """

    def __init__(
        self,
        sections: Sequence[SurveySection],
        header: Optional[SurveyHeader] = None,
        share_mode: ShareMode = "globals_shared",
    ) -> None:
        sections = list(sections)
        if not sections:
            raise ValueError("SurveyComposition needs at least one section")
        self.sections = sections
        self.default_header = header
        self.default_share_mode = share_mode

        self._groups = self._build_groups()
        self._survey: Optional[Survey] = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def survey(self) -> Survey:
        """Return the unified, tied :class:`~welleng.survey.Survey`.

        The result carries the composed total covariance (``cov_nev`` /
        ``cov_hla``) and its per-component breakdown (``cov_nev_global`` /
        ``cov_nev_systematic`` / ``cov_nev_random``). It is cached.
        """
        if self._survey is None:
            self._survey = self._compose()
        return self._survey

    # ------------------------------------------------------------------ #
    # grouping
    # ------------------------------------------------------------------ #
    def _resolve_header(self, section: SurveySection) -> SurveyHeader:
        header = section.header or self.default_header
        if header is None:
            header = SurveyHeader()
        return header

    def _section_grid(self, section: SurveySection):
        """Normalise a section to (md, inc_rad, azi_grid_rad) + resolved meta."""
        header = self._resolve_header(section)
        error_model = section.error_model or DEFAULT_ERROR_MODEL
        if section.survey is not None:
            s = section.survey
            md = np.asarray(s.md, dtype=float)
            inc_rad = np.asarray(s.inc_rad, dtype=float)
            azi_grid_rad = np.asarray(s.azi_grid_rad, dtype=float)
        else:
            # Build a geometry-only survey to normalise angles to grid radians
            # regardless of the section's azi_reference / deg convention.
            geom = Survey(
                md=section.md, inc=section.inc, azi=section.azi,
                deg=section.deg, header=header, error_model=None,
            )
            md = np.asarray(geom.md, dtype=float)
            inc_rad = np.asarray(geom.inc_rad, dtype=float)
            azi_grid_rad = np.asarray(geom.azi_grid_rad, dtype=float)
        return md, inc_rad, azi_grid_rad, header, error_model

    def _build_groups(self) -> List[_Group]:
        groups: List[_Group] = []
        prev_key = None
        for section in self.sections:
            md, inc_rad, azi_grid_rad, header, error_model = (
                self._section_grid(section)
            )
            key = (_model_key(error_model), section.tool_id)
            same_tool = bool(groups) and key == prev_key

            if same_tool:
                g = groups[-1]
                self._assert_tie(g.md[-1], md[0])
                # drop the duplicate tie station of the appended section
                g.md = np.hstack((g.md, md[1:]))
                g.inc_rad = np.hstack((g.inc_rad, inc_rad[1:]))
                g.azi_grid_rad = np.hstack((g.azi_grid_rad, azi_grid_rad[1:]))
            else:
                if groups:
                    self._assert_tie(groups[-1].md[-1], md[0])
                share_mode = self._resolve_share_mode(section, groups)
                groups.append(_Group(
                    md=md, inc_rad=inc_rad, azi_grid_rad=azi_grid_rad,
                    error_model=error_model, header=header,
                    share_mode=share_mode,
                ))
            prev_key = key
        return groups

    @staticmethod
    def _assert_tie(md_prev_end: float, md_next_start: float) -> None:
        if abs(md_prev_end - md_next_start) > _TIE_MD_TOL:
            raise ValueError(
                f"section tie mismatch: previous section ends at MD "
                f"{md_prev_end:g} but next starts at MD {md_next_start:g}; "
                "each section must tie on at the previous section's last MD"
            )

    def _resolve_share_mode(
        self, section: SurveySection, groups
    ) -> ShareMode:
        if section.share_mode is not None:
            return section.share_mode
        if not groups:
            return self.default_share_mode  # no tie before the first group
        prev_section = self._last_section_of_group(len(groups) - 1)
        # Different geomag model -> independent globals.
        if (
            section.geomag_model is not None
            and prev_section.geomag_model is not None
            and section.geomag_model != prev_section.geomag_model
        ):
            return "all_independent"
        # Far-apart survey dates -> independent globals.
        d_new = _parse_date(section.survey_date)
        d_old = _parse_date(prev_section.survey_date)
        if d_new is not None and d_old is not None:
            years = abs((d_new - d_old).days) / 365.25
            if years > _DATE_INDEPENDENCE_YEARS:
                return "all_independent"
        return self.default_share_mode

    def _last_section_of_group(self, group_index: int) -> SurveySection:
        """The final input section that fed a given group (for context keys)."""
        # Re-walk the section list mirroring _build_groups' grouping to find the
        # last section belonging to `group_index`.
        gi = -1
        prev_key = None
        last = self.sections[0]
        for section in self.sections:
            error_model = section.error_model or DEFAULT_ERROR_MODEL
            key = (_model_key(error_model), section.tool_id)
            if not (gi >= 0 and key == prev_key):
                gi += 1
            if gi == group_index:
                last = section
            elif gi > group_index:
                break
            prev_key = key
        return last

    # ------------------------------------------------------------------ #
    # composition
    # ------------------------------------------------------------------ #
    def _group_survey(self, g: _Group, start_nev) -> Survey:
        # geometry-only (covariance components come from _run_component); this
        # is used purely to chain group start positions.
        if g.survey is None or not np.allclose(g.survey.start_nev, start_nev):
            g.survey = Survey(
                md=g.md, inc=g.inc_rad, azi=g.azi_grid_rad, deg=False,
                header=g.header, error_model=None,
                start_nev=np.asarray(start_nev, dtype=float),
            )
        return g.survey

    def _compose(self) -> Survey:
        groups = self._groups

        # 1. Chain group start positions (only TVD affects covariance, but we
        #    need continuous positions anyway).
        start_nev = np.array([0.0, 0.0, 0.0])
        group_surveys: List[Survey] = []
        start_nevs: List[NDArray[np.float64]] = []
        for g in groups:
            start_nevs.append(np.asarray(start_nev, dtype=float))
            s = self._group_survey(g, start_nev)
            group_surveys.append(s)
            start_nev = s.pos_nev[-1]

        # 2. Compose each error component across the groups with its own
        #    correlation rule.
        share_global = [
            g.share_mode in ("globals_shared", "globals_and_systematic_shared")
            for g in groups
        ]
        share_systematic = [
            g.share_mode == "globals_and_systematic_shared" for g in groups
        ]
        # Random noise accumulates continuously and depends only on the tool
        # *type* (error model), never on the tool instance — so it stays
        # correlated (one continuous run) across any tie that keeps the same
        # error model, breaking only at a genuine model change. Composing it
        # per-group instead would inject a small propagation-context artifact
        # at every tool change.
        share_random = [
            _model_key(groups[k].error_model)
            == _model_key(groups[k - 1].error_model)
            for k in range(len(groups))
        ]
        # k=0 has no tie before it.
        share_global[0] = False
        share_systematic[0] = False
        share_random[0] = False

        # 'well' terms (COMPASS tie W): systematic throughout the WELL —
        # ONE realisation chained across every tie (severed only by an
        # explicit all_independent campaign break), as the flag says. An
        # earlier empirical rejection of this rule was an artifact of the
        # non-telescoping chained evaluation this module used to have; with
        # the Williamson A-14-exact per-term path the chained form
        # reproduces the operator-stored covariances.
        share_well = list(share_global)

        # One propagation per (run, depth-datum) instead of one per component
        # -- see _run_component. Scoped to this call: cleared below so a
        # mutated composition can never read a stale run.
        self._run_cache = {}
        glob = self._compose_component(
            "cov_nev_global", share_global, start_nevs
        )
        syst = self._compose_component(
            "cov_nev_systematic", share_systematic, start_nevs
        )
        rand = self._compose_component(
            "cov_nev_random", share_random, start_nevs
        )
        well = self._compose_component(
            "cov_nev_well", share_well, start_nevs
        )
        self._run_cache = {}

        # 3. Stitch per-group arrays (drop duplicate tie station of each
        #    subsequent group) into unified per-station arrays.
        cov_global = self._stitch(glob)
        cov_systematic = self._stitch(syst)
        cov_random = self._stitch(rand)
        cov_well = self._stitch(well)

        # 3a. Re-bucket the non-cancellable global. The ``cov_nev_global``
        #     bucket is what cancels against another wellbore that shares the
        #     campaign's geomagnetic realisation. An ``all_independent`` tie
        #     marks a change of that realisation (different geomag date/model),
        #     so everything the well accumulates *after* the first such tie is
        #     no longer cancellable against the campaign baseline — its global
        #     contribution is folded into the (non-cancelling) systematic
        #     bucket. The accumulated baseline global is frozen as a constant
        #     offset downstream. This is total-preserving: it only moves
        #     covariance between buckets, never changes ``cov_nev``.
        cov_global, cov_systematic = self._rebucket_global(
            cov_global, cov_systematic, share_global
        )
        cov_nev = cov_global + cov_systematic + cov_random + cov_well

        # 4. Unified geometry -> one Survey with the composed covariance.
        md = self._stitch_1d([g.md for g in groups])
        inc = self._stitch_1d([g.inc_rad for g in groups])
        azi = self._stitch_1d([g.azi_grid_rad for g in groups])

        header = groups[0].header
        # Compose in the grid domain; force the unified header to match so the
        # supplied grid angles are interpreted consistently.
        if header.azi_reference != "grid":
            header = _grid_header(header)

        survey = Survey(
            md=md, inc=inc, azi=azi, deg=False, header=header,
            start_nev=group_surveys[0].start_nev,
            start_xyz=group_surveys[0].start_xyz,
            cov_nev=cov_nev, error_model=None,
        )
        survey.cov_nev_global = cov_global
        survey.cov_nev_systematic = cov_systematic
        survey.cov_nev_random = cov_random
        survey.cov_nev_well = cov_well
        return survey

    def _rebucket_global(self, cov_global, cov_systematic, share_global):
        """Fold post-``all_independent`` global into the systematic bucket.

        Returns ``(cov_global_cancellable, cov_systematic_adjusted)`` with an
        unchanged sum. See the caller for the rationale.
        """
        # first group tied on with a non-shared (all_independent) global
        break_group = next(
            (k for k in range(1, len(self._groups)) if not share_global[k]),
            None,
        )
        if break_group is None:
            return cov_global, cov_systematic  # everything cancellable

        # unified index of the last station still connected to the baseline
        # geomag realisation (the tie station into ``break_group``).
        n_pref = len(self._groups[0].md)
        for k in range(1, break_group):
            n_pref += len(self._groups[k].md) - 1
        b = n_pref - 1

        baseline = cov_global[b]
        cov_global_cancellable = cov_global.copy()
        cov_systematic_adjusted = cov_systematic.copy()
        # downstream of the break: freeze the baseline global, move the rest
        # (post-break global accumulation) into the systematic bucket.
        post = slice(b + 1, None)
        cov_systematic_adjusted[post] += cov_global[post] - baseline
        cov_global_cancellable[post] = baseline
        return cov_global_cancellable, cov_systematic_adjusted

    def _compose_component(self, attr, share, start_nevs):
        """Return a list of per-group composed arrays (each incl tie station 0).

        ``share[k]`` (k>=1) is True when this component continues *correlated*
        across the tie before group k. Correlated runs are computed by building
        the run as one survey; a carry (freeze + independent add) is applied
        across run boundaries.
        """
        n = len(self._groups)
        out: List[NDArray[np.float64]] = [None] * n
        carry = np.zeros((3, 3))
        k = 0
        while k < n:
            # maximal run [k..j] joined by shared ties
            j = k
            while j + 1 < n and share[j + 1]:
                j += 1
            run_comp = self._run_component(attr, k, j, start_nevs[k])
            run_composed = carry + run_comp

            # slice run back into per-group arrays (each incl its tie station)
            idx = 0
            for gi in range(k, j + 1):
                n_g = len(self._groups[gi].md)
                if gi == k:
                    out[gi] = run_composed[idx:idx + n_g]
                    idx += n_g
                else:
                    tie = run_composed[idx - 1][None]
                    out[gi] = np.vstack((tie, run_composed[idx:idx + n_g - 1]))
                    idx += n_g - 1
            carry = run_composed[-1]
            k = j + 1
        return out

    def _run_component(self, attr, k, j, start_nev):
        """Component of groups k..j computed as ONE correlated survey.

        A "ghost" continuation station (the first post-tie station of group
        ``j+1``, if any) is appended so the run's final station — the tie into
        the next group — is computed *with* its true following interval, exactly
        as it would be in a single unified survey. The ghost is then discarded.
        Several ISCWSA weight-function terms use the interval to the *next*
        station, so without this the tie station would be mis-computed.
        """
        groups = self._groups[k:j + 1]
        models = {_model_key(g.error_model) for g in groups}
        if len(models) > 1:
            # Different models sharing a realisation cannot be built as one
            # reference-model survey (that evaluates the wrong weights over
            # the other runs' spans and badly overstates the shared growth).
            # Chain per-term instead: each run with its OWN model, same-name
            # error sources continuing one realisation across the ties.
            return self._run_component_per_term(attr, k, j, start_nev)
        md = self._stitch_1d([g.md for g in groups])
        inc = self._stitch_1d([g.inc_rad for g in groups])
        azi = self._stitch_1d([g.azi_grid_rad for g in groups])
        n_real = len(md)
        if j + 1 < len(self._groups) and len(self._groups[j + 1].md) > 1:
            nxt = self._groups[j + 1]
            md = np.append(md, nxt.md[1])
            inc = np.append(inc, nxt.inc_rad[1])
            azi = np.append(azi, nxt.azi_grid_rad[1])
        header = groups[0].header
        # ONE propagation serves every component that shares this run's
        # geometry AND its depth datum. `attr` selects which covariance to read
        # off a build; it does not change the build -- the only thing that does
        # is the severed-run `_tmd_datum` override below. So a run needs at most
        # TWO propagations (with and without the override), not one per
        # component. Un-cached, a 2-section compose ran EIGHT full ErrorModel
        # propagations where 2-3 suffice, discarding three quarters of each
        # result (welleng-probcol's profile: 93% of their programme setup).
        severed = attr in ("cov_nev_systematic", "cov_nev_well") and k > 0
        cache = getattr(self, "_run_cache", None)
        if cache is None:
            cache = self._run_cache = {}
        key = (k, j, severed, tuple(np.asarray(start_nev, float).ravel()))
        cached = cache.get(key)
        if cached is not None:
            return getattr(cached, attr)[:n_real]
        if severed:
            # A severed run's own systematic/well realisations act on the
            # depth measured BY THAT RUN. COMPASS-formula terms weighted by
            # ``tmd`` (e.g. a dsf depth-scale error) must see run-relative
            # depth — depth-from-surface transferred at the tie, so the new
            # realisation carries none of it. (Named ISCWSA models bind
            # ``MD`` and are unaffected.)
            import copy
            header = copy.copy(header)
            header._tmd_datum = float(groups[0].md[0])
        run = Survey(
            md=md, inc=inc, azi=azi, deg=False, header=header,
            error_model=groups[0].error_model, start_nev=start_nev,
        )
        cache[key] = run
        return getattr(run, attr)[:n_real]

    #: composed-covariance attribute -> the term propagation mode it holds
    _ATTR_TO_PROPAGATION = {
        "cov_nev_global": "global",
        "cov_nev_systematic": "systematic",
        "cov_nev_well": "well",
    }

    def _run_component_per_term(self, attr, k, j, start_nev):
        """Shared component of a MULTI-MODEL run, chained per error source
        (Williamson SPE-67616-PA Eq. A-14: one realisation, e-vectors summed
        linearly across legs, each leg under its OWN model's weights).

        Implementation note — why full-geometry evaluation. A leg evaluated
        as a standalone survey zeroes its tie station, which DISCARDS that
        endpoint's telescoped contribution: for scale-type weights
        (e.g. depth-scale, ``w ~ tmd``) the two-sided station differentials
        (Eqs. A-4a/A-4b) cancel at interior stations, so the leg total is an
        ENDPOINT DIFFERENCE — drop the tie endpoint and every chained leg
        re-counts the full accumulated magnitude (measured: x2 per leg on a
        straight well, xN_legs on the composed covariance). So instead each
        leg's model is evaluated over the FULL run geometry (every station
        interior, telescoping intact) and the per-station sigma increments
        are mixed by the station's OWNING leg:

            v(s) = sum_{k<=s} (sigma_own(k)[k] - sigma_own(k)[k-1])

        which reduces to the engine's own accumulation when one model owns
        every station, and gives each tie interval one half from each of its
        two tools (the physically measured split). Covariance is the sum
        over source names of the outer product of ``v`` (independent
        sources; correlation one within a source across legs — Eq. A-14).

        Random components never take this path (``share_random`` requires an
        identical model).
        """
        prop = self._ATTR_TO_PROPAGATION[attr]
        groups = self._groups[k:j + 1]
        md = self._stitch_1d([g.md for g in groups])
        inc = self._stitch_1d([g.inc_rad for g in groups])
        azi = self._stitch_1d([g.azi_grid_rad for g in groups])
        n_real = len(md)
        # ghost continuation station (see _run_component)
        if j + 1 < len(self._groups) and len(self._groups[j + 1].md) > 1:
            nxt = self._groups[j + 1]
            md = np.append(md, nxt.md[1])
            inc = np.append(inc, nxt.inc_rad[1])
            azi = np.append(azi, nxt.azi_grid_rad[1])

        # owner[s] = index into `groups` of the leg that measured station s
        # (the tie station belongs to the EARLIER leg; its following interval
        # is split between the two tools by the two-sided differentials).
        owner = np.zeros(len(md), dtype=int)
        pos = len(groups[0].md)
        for gi in range(1, len(groups)):
            n_g = len(groups[gi].md)
            owner[pos:pos + n_g - 1] = gi
            pos += n_g - 1
        owner[pos:] = len(groups) - 1          # ghost rides the last leg

        # Per-model sigma profiles over the full geometry, per term name. The
        # Survey per group depends only on the geometry, header and model --
        # NOT on `attr`, which only selects which propagation mode to filter
        # for -- so the same builds serve every component that reaches this
        # path (global / systematic / well; random never does). Cached, or the
        # multi-model run re-propagates each leg three times over.
        cache = getattr(self, "_run_cache", None)
        if cache is None:
            cache = self._run_cache = {}
        key = ("per_term", k, j, tuple(np.asarray(start_nev, float).ravel()))
        runs = cache.get(key)
        if runs is None:
            runs = cache[key] = [
                Survey(
                    md=md, inc=inc, azi=azi, deg=False, header=g.header,
                    error_model=g.error_model, start_nev=start_nev,
                )
                for g in groups
            ]
        sigmas: List[dict] = [
            {
                name: np.asarray(term.sigma_e_NEV, dtype=float)
                for name, term in run.err.errors.errors.items()
                if term.propagation == prop
            }
            for run in runs
        ]

        names = sorted({n for s in sigmas for n in s})
        cov = np.zeros((n_real, 3, 3))
        zero = np.zeros((len(md), 3))
        for name in names:
            profiles = [s.get(name, zero) for s in sigmas]
            # mixed per-station increments, chained (one realisation)
            v = np.zeros((n_real, 3))
            for s_i in range(1, n_real):
                sig_own = profiles[owner[s_i]]
                v[s_i] = v[s_i - 1] + (sig_own[s_i] - sig_own[s_i - 1])
            cov += v[:, :, None] * v[:, None, :]
        return cov

    # ------------------------------------------------------------------ #
    # stitching helpers (drop the duplicate tie station of each next group)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stitch(arrays: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        parts = [arrays[0]] + [a[1:] for a in arrays[1:]]
        return np.concatenate(parts, axis=0)

    @staticmethod
    def _stitch_1d(arrays: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        parts = [arrays[0]] + [a[1:] for a in arrays[1:]]
        return np.concatenate(parts)


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _grid_header(header: SurveyHeader) -> SurveyHeader:
    import copy
    h = copy.copy(header)
    h.azi_reference = "grid"
    return h
