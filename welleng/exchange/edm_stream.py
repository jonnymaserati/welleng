"""Streaming Landmark EDM (COMPASS export) parser.

This module provides a memory-bounded, streaming alternative to the DOM-based
:class:`welleng.exchange.edm.EDM` class. A Landmark ``EDM``/COMPASS export is a
single, flat XML file (``<export><TOPLEVEL>...`` where every ``CD_*``/``TU_*``
element is a table row and *all* data lives in element attributes). These files
are routinely hundreds of megabytes (the public Volve export is ~211 MB), which
makes ``xml.etree.ElementTree.parse`` (full DOM) impractical.

:class:`EDMReader` performs a single ``iterparse`` sweep to build a lightweight,
typed index of the (small) metadata tables and the (larger) survey-station
tables, deliberately skipping the huge non-survey tables (pore/frac pressure,
real-time configuration, torque & drag, etc.). Surveys are then assembled
lazily and returned as typed :class:`WellboreSurvey` objects that carry per
station covariance and the resolved survey tool for each measured-depth
interval.

Design notes / assumptions
--------------------------
* **Units.** Stored depth/offset values carry no per-field unit-of-measure. The
  public Volve export is in **feet** (confirmed empirically), so the reader's
  ``source_units`` defaults to ``"feet"``. This is exposed as a constructor
  parameter -- it is *not* hard-coded globally. Angles are always degrees.
  Covariance is in ``source_units**2``.
* **Covariance axes.** ``CD_DEFINITIVE_SURVEY_STATION`` stores the total
  accumulated position covariance as ``covariance_{xx,xy,xz,yy,yz,zz}`` in a
  local Cartesian frame. Consistent with the validated ``examples/volve_wells``
  mapping, this module treats ``x -> East``, ``y -> North``, ``z -> Vertical``.
  :attr:`SurveyStation.covariance` is stored in that native ``(x, y, z)`` order;
  :meth:`WellboreSurvey.to_welleng` re-orders it to welleng's ``NEV`` frame.
* **Error models.** Survey tools are *classified* (see :func:`classify_tool`)
  and surfaced with their name/interval, but deliberately **not** mapped to an
  ISCWSA error model -- pre-ISCWSA COMPASS tool models were frequently bespoke.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

FEET_TO_METERS = 0.3048

# Tables materialised during the index sweep. Everything else is skipped.
_METADATA_TAGS = frozenset({
    "CD_PROJECT", "CD_SITE", "CD_WELL", "CD_WELLBORE", "CD_DATUM",
    "CD_SURVEY_TOOL", "CD_SURVEY_HEADER", "CD_DEFINITIVE_SURVEY_HEADER",
    "CD_SURVEY_PROGRAM",
})
_STATION_TAGS = frozenset({
    "CD_SURVEY_STATION", "CD_DEFINITIVE_SURVEY_STATION",
})
_WANTED_TAGS = _METADATA_TAGS | _STATION_TAGS


class ToolKind(str, Enum):
    """Best-effort classification of a COMPASS survey tool."""

    GYRO = "gyro"
    MWD = "mwd"
    DEFINITIVE = "definitive"
    INCLINATION_ONLY = "inclination_only"
    OTHER = "other"


# Keyword tables for :func:`classify_tool`. Order of the checks matters:
# "MWD gyro" tools are gyros, so gyro indicators are tested before magnetic ones.
_GYRO_KEYWORDS = (
    "gyro", "keeper", "wellbore surveyor", "inertial", "rigs",
    "north seek", "north-seek", "sdc",
)
_INC_ONLY_KEYWORDS = ("inclination only", "inc only", "inc-only")
_DEFINITIVE_KEYWORDS = ("definitive", "combined")
_MWD_KEYWORDS = ("mwd", "magnetic", "magn", "ems", "measurement while")


def classify_tool(
    name: Optional[str], description: Optional[str] = None
) -> ToolKind:
    """Classify a survey tool from its name and (optional) description.

    Pure function -- no I/O, no side effects. Matching is case-insensitive and
    considers the concatenation of ``name`` and ``description``.

    Parameters
    ----------
    name : str or None
        The ``tool_name`` attribute of a ``CD_SURVEY_TOOL`` row.
    description : str, optional
        The ``description`` attribute of the same row.

    Returns
    -------
    ToolKind

    Notes
    -----
    Gyro-while-drilling tools (e.g. ``"MWD gyro, Gyrodata, GWD70"``) are
    classified as :attr:`ToolKind.GYRO` -- the gyro sensor governs the error
    behaviour even though the tool is conveyed on an MWD string.
    """
    def _classify(text: Optional[str]) -> ToolKind:
        text = (text or "").lower().strip()
        if not text:
            return ToolKind.OTHER
        if any(k in text for k in _GYRO_KEYWORDS):
            return ToolKind.GYRO
        if any(k in text for k in _INC_ONLY_KEYWORDS):
            return ToolKind.INCLINATION_ONLY
        if any(k in text for k in _DEFINITIVE_KEYWORDS):
            return ToolKind.DEFINITIVE
        if any(k in text for k in _MWD_KEYWORDS):
            return ToolKind.MWD
        return ToolKind.OTHER

    # Classify the NAME on its own first; the description only breaks a tie
    # when the name is inconclusive. (A description like "Magnetic tools
    # without gyro-verification" must not turn a magnetic tool into a gyro.)
    kind = _classify(name)
    if kind is ToolKind.OTHER:
        kind = _classify(description)
    return kind


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------

@dataclass
class SurveyTool:
    """A ``CD_SURVEY_TOOL`` row (a survey tool definition)."""

    tool_id: str
    name: str
    description: str = ""
    kind: ToolKind = ToolKind.OTHER
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class Wellbore:
    """A ``CD_WELLBORE`` row."""

    wellbore_id: str
    name: str
    well_id: str
    parent_wellbore_id: Optional[str] = None
    well_legal_name: Optional[str] = None
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class SurveyHeader:
    """A survey header (``CD_SURVEY_HEADER`` or ``CD_DEFINITIVE_SURVEY_HEADER``).

    ``kind`` is ``"raw"`` for control-point survey headers and ``"definitive"``
    for the accumulated/blended definitive survey headers.
    """

    header_id: str
    wellbore_id: str
    kind: str  # "raw" | "definitive"
    phase: Optional[str] = None  # PLAN | ACTUAL | PROTOTYPE
    name: Optional[str] = None
    tie_survey_header_id: Optional[str] = None
    survey_tool_id: Optional[str] = None
    n_stations: int = 0
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class ProgramInterval:
    """A ``CD_SURVEY_PROGRAM`` row: a per-interval tool assignment.

    Joins a definitive survey header to a (raw) survey header + survey tool over
    the measured-depth range ``[md_top, md_base]``.
    """

    def_survey_header_id: Optional[str]
    survey_header_id: Optional[str]
    survey_tool_id: Optional[str]
    md_top: Optional[float]
    md_base: Optional[float]
    sequence_no: Optional[int]
    tool: Optional[SurveyTool] = None

    def contains(self, md: float) -> bool:
        top = -np.inf if self.md_top is None else self.md_top
        base = np.inf if self.md_base is None else self.md_base
        return top <= md <= base


@dataclass
class SurveyStation:
    """A single assembled survey station.

    ``covariance`` (when present) is a symmetric ``3x3`` array in the file's
    native ``(x, y, z)`` axes, i.e. ``x=East, y=North, z=Vertical`` (see the
    module docstring), in ``source_units**2``.
    """

    md: Optional[float]
    inc: Optional[float]
    azi: Optional[float]
    tvd: Optional[float] = None
    north: Optional[float] = None
    east: Optional[float] = None
    dls: Optional[float] = None
    sequence_no: Optional[int] = None
    covariance: Optional[np.ndarray] = None
    ellipse: Optional[Tuple[float, float, float]] = None  # (n, e, v)
    tool: Optional[SurveyTool] = None


@dataclass
class WellboreSurvey:
    """An assembled survey for a wellbore.

    Attributes
    ----------
    wellbore : Wellbore
    header : SurveyHeader
    stations : list of SurveyStation
    source_units : str
        Units of the stored depths/offsets (e.g. ``"feet"``).
    """

    wellbore: Wellbore
    header: SurveyHeader
    stations: List[SurveyStation]
    source_units: str = "feet"

    def __len__(self) -> int:
        return len(self.stations)

    @property
    def has_covariance(self) -> bool:
        return bool(self.stations) and all(
            s.covariance is not None for s in self.stations
        )

    def as_arrays(self) -> dict:
        """Return the survey as a dict of numpy arrays (native source units).

        Keys: ``md, inc, azi, tvd, north, east, dls, covariance,
        tool_name, tool_kind``. ``covariance`` is an ``(n, 3, 3)`` array (with
        ``nan`` blocks where a station lacks covariance), or ``None`` if no
        station has covariance.
        """
        def col(attr):
            return np.array(
                [getattr(s, attr) if getattr(s, attr) is not None else np.nan
                 for s in self.stations],
                dtype=float,
            )

        if any(s.covariance is not None for s in self.stations):
            cov = np.array([
                s.covariance if s.covariance is not None
                else np.full((3, 3), np.nan)
                for s in self.stations
            ])
        else:
            cov = None

        return {
            "md": col("md"),
            "inc": col("inc"),
            "azi": col("azi"),
            "tvd": col("tvd"),
            "north": col("north"),
            "east": col("east"),
            "dls": col("dls"),
            "covariance": cov,
            "tool_name": [
                s.tool.name if s.tool is not None else None
                for s in self.stations
            ],
            "tool_kind": [
                s.tool.kind if s.tool is not None else None
                for s in self.stations
            ],
        }

    def to_welleng(self, units: str = "meters", azi_reference: str = "grid"):
        """Build a :class:`welleng.survey.Survey` from this survey.

        Parameters
        ----------
        units : str, {"meters", "feet"}
            Target units. Depths/offsets are converted from ``source_units``.
        azi_reference : str
            Passed to the survey header (default ``"grid"`` -- COMPASS offsets
            are grid-referenced).

        Returns
        -------
        welleng.survey.Survey
        """
        from ..survey import Survey, SurveyHeader as WeSurveyHeader
        from ..utils import make_long_cov

        units = units.lower()
        if units not in ("meters", "feet"):
            raise ValueError("units must be 'meters' or 'feet'")
        src = self.source_units.lower()
        if src in ("feet", "ft") and units == "meters":
            length = FEET_TO_METERS
        elif src in ("meters", "m", "metres") and units == "feet":
            length = 1.0 / FEET_TO_METERS
        else:
            length = 1.0

        arrays = self.as_arrays()
        md = arrays["md"] * length
        tvd = arrays["tvd"] * length
        north = arrays["north"] * length
        east = arrays["east"] * length

        cov_nev = None
        if self.has_covariance:
            # native (x, y, z) with x=East, y=North, z=Vertical -> welleng NEV.
            # make_long_cov consumes columns [nn, ne, nv, ee, ev, vv].
            xx = np.array([s.covariance[0, 0] for s in self.stations])
            xy = np.array([s.covariance[0, 1] for s in self.stations])
            xz = np.array([s.covariance[0, 2] for s in self.stations])
            yy = np.array([s.covariance[1, 1] for s in self.stations])
            yz = np.array([s.covariance[1, 2] for s in self.stations])
            zz = np.array([s.covariance[2, 2] for s in self.stations])
            cov_nev = make_long_cov(
                np.array([yy, xy, yz, xx, xz, zz]).T
            ) * (length ** 2)

        header = WeSurveyHeader(
            name=self.header.name or self.header.header_id,
            azi_reference=azi_reference,
        )
        return Survey(
            md=md,
            inc=arrays["inc"],
            azi=arrays["azi"],
            n=north,
            e=east,
            tvd=tvd,
            header=header,
            cov_nev=cov_nev,
            unit=units,
        )


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------

def _f(attrib: Dict[str, str], key: str) -> Optional[float]:
    v = attrib.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(attrib: Dict[str, str], key: str) -> Optional[int]:
    f = _f(attrib, key)
    return None if f is None else int(f)


def _iter_rows(path: str) -> Iterator[Tuple[str, Dict[str, str]]]:
    """Yield ``(tag, attrib)`` for wanted rows with bounded memory.

    The EDM file is flat (``<export><TOPLEVEL><CD_.../>...``). ``iterparse``
    otherwise accumulates every parsed child under its parent element; we clear
    both the element and its parent on each ``end`` event so peak memory stays
    proportional to a single row rather than the whole file.
    """
    stack: List[ET.Element] = []
    context = ET.iterparse(path, events=("start", "end"))
    for event, elem in context:
        if event == "start":
            stack.append(elem)
            continue
        # end event
        stack.pop()
        if elem.tag in _WANTED_TAGS:
            yield elem.tag, dict(elem.attrib)
        elem.clear()
        # Drop already-parsed siblings held by the parent (TOPLEVEL) so it
        # doesn't grow to hold every row in the file.
        if stack:
            stack[-1].clear()


class EDMReader:
    """Streaming reader for a Landmark EDM/COMPASS XML export.

    Build with :meth:`open` (or the constructor). One ``iterparse`` sweep
    indexes the metadata + survey-station tables; surveys are assembled lazily
    via :meth:`survey`.

    Parameters
    ----------
    path : str
        Path to the EDM XML file.
    source_units : str, {"feet", "meters"}
        Units of the stored depth/offset values (default ``"feet"`` -- the
        Volve export). Angles are always degrees.
    """

    def __init__(self, path: str, source_units: str = "feet"):
        self.path = path
        self.source_units = source_units.lower()

        self.projects: Dict[str, Dict[str, str]] = {}
        self.sites: Dict[str, Dict[str, str]] = {}
        self.wells: Dict[str, Dict[str, str]] = {}
        self.wellbores: Dict[str, Wellbore] = {}
        self.datums: Dict[str, List[Dict[str, str]]] = {}  # by well_id
        self.tools: Dict[str, SurveyTool] = {}
        self.headers: Dict[str, SurveyHeader] = {}  # raw + definitive by id
        self.programs: Dict[str, List[ProgramInterval]] = {}  # by def hdr id

        # station records grouped by header id (parsed floats)
        self._raw_stations: Dict[str, List[dict]] = {}
        self._def_stations: Dict[str, List[dict]] = {}

        # name -> id lookups (built after sweep)
        self.wellbore_name_to_id: Dict[str, str] = {}

        self._index()

    # -- construction --------------------------------------------------------
    @classmethod
    def open(cls, path: str, source_units: str = "feet") -> "EDMReader":
        """Open and index an EDM file. Alias for the constructor."""
        return cls(path, source_units=source_units)

    def _index(self) -> None:
        for tag, a in _iter_rows(self.path):
            if tag == "CD_PROJECT":
                self.projects[a["project_id"]] = a
            elif tag == "CD_SITE":
                self.sites[a["site_id"]] = a
            elif tag == "CD_WELL":
                self.wells[a["well_id"]] = a
            elif tag == "CD_WELLBORE":
                self.wellbores[a["wellbore_id"]] = Wellbore(
                    wellbore_id=a["wellbore_id"],
                    name=a.get("wellbore_name", a["wellbore_id"]),
                    well_id=a.get("well_id", ""),
                    parent_wellbore_id=a.get("parent_wellbore_id"),
                    well_legal_name=a.get("well_legal_name"),
                    raw=a,
                )
            elif tag == "CD_DATUM":
                self.datums.setdefault(a.get("well_id", ""), []).append(a)
            elif tag == "CD_SURVEY_TOOL":
                self.tools[a["survey_tool_id"]] = SurveyTool(
                    tool_id=a["survey_tool_id"],
                    name=a.get("tool_name", ""),
                    description=a.get("description", ""),
                    kind=classify_tool(
                        a.get("tool_name"), a.get("description")
                    ),
                    raw=a,
                )
            elif tag == "CD_SURVEY_HEADER":
                self.headers[a["survey_header_id"]] = SurveyHeader(
                    header_id=a["survey_header_id"],
                    wellbore_id=a.get("wellbore_id", ""),
                    kind="raw",
                    phase=a.get("phase"),
                    name=a.get("survey_name"),
                    tie_survey_header_id=a.get("tie_survey_header_id"),
                    survey_tool_id=a.get("survey_tool_id"),
                    raw=a,
                )
            elif tag == "CD_DEFINITIVE_SURVEY_HEADER":
                self.headers[a["def_survey_header_id"]] = SurveyHeader(
                    header_id=a["def_survey_header_id"],
                    wellbore_id=a.get("wellbore_id", ""),
                    kind="definitive",
                    phase=a.get("phase"),
                    name=a.get("name"),
                    tie_survey_header_id=a.get("tie_survey_header_id"),
                    survey_tool_id=a.get("survey_tool_id"),
                    raw=a,
                )
            elif tag == "CD_SURVEY_PROGRAM":
                did = a.get("def_survey_header_id")
                self.programs.setdefault(did, []).append(ProgramInterval(
                    def_survey_header_id=did,
                    survey_header_id=a.get("survey_header_id"),
                    survey_tool_id=a.get("survey_tool_id"),
                    md_top=_f(a, "md_top"),
                    md_base=_f(a, "md_base"),
                    sequence_no=_i(a, "sequence_no"),
                ))
            elif tag == "CD_SURVEY_STATION":
                self._raw_stations.setdefault(
                    a["survey_header_id"], []
                ).append({
                    "md": _f(a, "md"), "inc": _f(a, "inclination"),
                    "azi": _f(a, "azimuth"), "tvd": _f(a, "tvd"),
                    "north": _f(a, "offset_north"),
                    "east": _f(a, "offset_east"),
                    "dls": _f(a, "dogleg_severity"),
                    "sequence_no": _i(a, "sequence_no"),
                })
            elif tag == "CD_DEFINITIVE_SURVEY_STATION":
                self._def_stations.setdefault(
                    a["def_survey_header_id"], []
                ).append({
                    "md": _f(a, "md"), "inc": _f(a, "inclination"),
                    "azi": _f(a, "azimuth"), "tvd": _f(a, "tvd"),
                    "north": _f(a, "offset_north"),
                    "east": _f(a, "offset_east"),
                    "dls": _f(a, "dogleg_severity"),
                    "sequence_no": _i(a, "sequence_no"),
                    "cov": (
                        _f(a, "covariance_xx"), _f(a, "covariance_xy"),
                        _f(a, "covariance_xz"), _f(a, "covariance_yy"),
                        _f(a, "covariance_yz"), _f(a, "covariance_zz"),
                    ),
                    "ellipse": (
                        _f(a, "ellipse_north"), _f(a, "ellipse_east"),
                        _f(a, "ellipse_vertical"),
                    ),
                })

        # finalise: station counts, name lookups, resolve program tools
        for hid, header in self.headers.items():
            if header.kind == "raw":
                header.n_stations = len(self._raw_stations.get(hid, []))
            else:
                header.n_stations = len(self._def_stations.get(hid, []))
        for wb in self.wellbores.values():
            self.wellbore_name_to_id[wb.name] = wb.wellbore_id
        for intervals in self.programs.values():
            for iv in intervals:
                if iv.survey_tool_id is not None:
                    iv.tool = self.tools.get(iv.survey_tool_id)

    # -- lookups -------------------------------------------------------------
    def _resolve_wellbore(self, wellbore) -> Wellbore:
        if isinstance(wellbore, Wellbore):
            return wellbore
        if wellbore in self.wellbores:
            return self.wellbores[wellbore]
        if wellbore in self.wellbore_name_to_id:
            return self.wellbores[self.wellbore_name_to_id[wellbore]]
        raise KeyError(f"unknown wellbore: {wellbore!r}")

    def find_wellbores(self, substring: str) -> List[Wellbore]:
        """Return wellbores whose name contains ``substring`` (case-insensitive)."""
        s = substring.lower()
        return [
            wb for wb in self.wellbores.values() if s in wb.name.lower()
        ]

    def survey_headers(
        self,
        wellbore,
        kind: str = "definitive",
        phase: Optional[str] = None,
    ) -> List[SurveyHeader]:
        """Return the survey headers for a wellbore, filtered by kind/phase.

        Sorted by station count (descending), so the first entry is the
        sensible default selection.
        """
        wb = self._resolve_wellbore(wellbore)
        out = [
            h for h in self.headers.values()
            if h.wellbore_id == wb.wellbore_id
            and h.kind == kind
            and (phase is None or h.phase == phase)
        ]
        out.sort(key=lambda h: h.n_stations, reverse=True)
        return out

    # -- survey assembly -----------------------------------------------------
    def survey(
        self,
        wellbore,
        kind: str = "definitive",
        phase: Optional[str] = "ACTUAL",
        header_id: Optional[str] = None,
    ) -> WellboreSurvey:
        """Assemble a typed :class:`WellboreSurvey`.

        Parameters
        ----------
        wellbore : str or Wellbore
            Wellbore id, wellbore name, or a :class:`Wellbore`.
        kind : str, {"definitive", "raw"}
            ``"definitive"`` returns the accumulated survey with covariance and
            per-interval tools resolved from ``CD_SURVEY_PROGRAM``; ``"raw"``
            returns the control-point survey.
        phase : str, optional
            ``PLAN`` / ``ACTUAL`` / ``PROTOTYPE``. Default ``"ACTUAL"``.
        header_id : str, optional
            Select a specific header. If omitted, the header with the most
            stations (matching ``kind``/``phase``) is used.

        Returns
        -------
        WellboreSurvey
        """
        wb = self._resolve_wellbore(wellbore)
        if header_id is not None:
            header = self.headers.get(header_id)
            if header is None:
                raise KeyError(f"unknown survey header: {header_id!r}")
        else:
            candidates = self.survey_headers(wb, kind=kind, phase=phase)
            if not candidates:
                raise LookupError(
                    f"no {kind} survey (phase={phase}) for wellbore "
                    f"{wb.name!r}"
                )
            header = candidates[0]

        if header.kind == "definitive":
            stations = self._assemble_definitive(header)
        else:
            stations = self._assemble_raw(header)

        return WellboreSurvey(
            wellbore=wb,
            header=header,
            stations=stations,
            source_units=self.source_units,
        )

    def _tool_for_md(
        self, intervals: List[ProgramInterval], md: Optional[float]
    ) -> Optional[SurveyTool]:
        if md is None or not intervals:
            return None
        for iv in intervals:
            if iv.contains(md):
                return iv.tool
        return None

    def _assemble_definitive(
        self, header: SurveyHeader
    ) -> List[SurveyStation]:
        records = sorted(
            self._def_stations.get(header.header_id, []),
            key=lambda r: (r["md"] if r["md"] is not None else np.inf),
        )
        intervals = sorted(
            self.programs.get(header.header_id, []),
            key=lambda iv: (
                iv.sequence_no if iv.sequence_no is not None else 0
            ),
        )
        stations = []
        for r in records:
            cov = None
            if all(c is not None for c in r["cov"]):
                xx, xy, xz, yy, yz, zz = r["cov"]
                cov = np.array([
                    [xx, xy, xz],
                    [xy, yy, yz],
                    [xz, yz, zz],
                ])
            stations.append(SurveyStation(
                md=r["md"], inc=r["inc"], azi=r["azi"], tvd=r["tvd"],
                north=r["north"], east=r["east"], dls=r["dls"],
                sequence_no=r["sequence_no"], covariance=cov,
                ellipse=r["ellipse"],
                tool=self._tool_for_md(intervals, r["md"]),
            ))
        return stations

    def _assemble_raw(self, header: SurveyHeader) -> List[SurveyStation]:
        records = sorted(
            self._raw_stations.get(header.header_id, []),
            key=lambda r: (r["md"] if r["md"] is not None else np.inf),
        )
        # ACTUAL raw headers may carry a single tool for the whole survey.
        tool = (
            self.tools.get(header.survey_tool_id)
            if header.survey_tool_id else None
        )
        return [
            SurveyStation(
                md=r["md"], inc=r["inc"], azi=r["azi"], tvd=r["tvd"],
                north=r["north"], east=r["east"], dls=r["dls"],
                sequence_no=r["sequence_no"], tool=tool,
            )
            for r in records
        ]

    # -- graph traversal -----------------------------------------------------
    def sidetrack_chain(self, wellbore) -> List[Wellbore]:
        """Return the parent-wellbore chain, root-first, ending at ``wellbore``.

        Follows ``parent_wellbore_id``. Uses a local accumulator (no mutable
        default argument) and guards against cycles.
        """
        wb = self._resolve_wellbore(wellbore)
        chain: List[Wellbore] = []
        seen = set()
        current: Optional[Wellbore] = wb
        while current is not None and current.wellbore_id not in seen:
            chain.append(current)
            seen.add(current.wellbore_id)
            pid = current.parent_wellbore_id
            current = self.wellbores.get(pid) if pid else None
        chain.reverse()
        return chain

    def tie_chain(self, header) -> List[SurveyHeader]:
        """Return the tie-on header chain, root-first, ending at ``header``.

        Follows ``tie_survey_header_id``. No mutable default argument; cycle
        guarded.
        """
        if isinstance(header, SurveyHeader):
            hdr: Optional[SurveyHeader] = header
        else:
            hdr = self.headers.get(header)
            if hdr is None:
                raise KeyError(f"unknown survey header: {header!r}")
        chain: List[SurveyHeader] = []
        seen = set()
        current = hdr
        while current is not None and current.header_id not in seen:
            chain.append(current)
            seen.add(current.header_id)
            tid = current.tie_survey_header_id
            current = self.headers.get(tid) if tid else None
        chain.reverse()
        return chain


def open_edm(path: str, source_units: str = "feet") -> EDMReader:
    """Convenience factory mirroring :meth:`EDMReader.open`."""
    return EDMReader.open(path, source_units=source_units)
