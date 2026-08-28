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
    "CD_SURVEY_PROGRAM", "DP_MAGNETIC",
})
_STATION_TAGS = frozenset({
    "CD_SURVEY_STATION", "CD_DEFINITIVE_SURVEY_STATION",
})
_WANTED_TAGS = _METADATA_TAGS | _STATION_TAGS

# Optional (opt-in via ``with_geopressure=True``) — geopressure / temperature
# prognoses, as-run geometry, formation tops, and the grade/material catalogue.
# Off by default: these are the large non-survey tables the light reader
# deliberately skips (Volve: ~76k geopressure rows). PRESSURE is canonical; the
# stored EMW is an RKB-datum view (do not derive PPFP logic from it).
_GEOPRESSURE_TAGS = frozenset({
    "CD_PORE_PRESSURE", "CD_PORE_PRESSURE_GROUP",
    "CD_FRAC_GRADIENT", "CD_FRAC_GRADIENT_GROUP",
    "CD_TEMP_GRADIENT", "CD_TEMP_GRADIENT_GROUP",
    "CD_HOLE_SECT", "CD_CASE", "CD_WELLBORE_FORMATION",
    "CD_GRADE", "CD_MATERIAL",
    "WP_FLUID_TEMP", "WP_FLUID_TEMP_FANN_DATA",
})

# Optional (opt-in via ``with_assemblies=True``) — the string catalogue:
# casing/tubing/liner/BHA assemblies and their components. Casing components
# additionally carry Landmark's STORED axial/burst/collapse ratings, returned
# verbatim as a validation oracle (welleng recomputes; it does not read them
# back as truth).
_ASSEMBLY_TAGS = frozenset({"CD_ASSEMBLY", "CD_ASSEMBLY_COMP"})

# Optional (opt-in via ``with_load_cases=True``) — the design load-case set:
# enabled load profiles + design factors (TU_LOAD_PROFILE), the per-depth load
# a string sees (TU_CUSTOM_LOAD_PROFILE), and the named loads + type/category
# enums (TU_LOAD_HEADER). Parameters are returned RAW -- their design semantics
# (which is a design factor, the load-case-code taxonomy) live in the casing
# design engine, not the reader.
_LOAD_CASE_TAGS = frozenset({
    "TU_LOAD_PROFILE", "TU_CUSTOM_LOAD_PROFILE", "TU_LOAD_HEADER",
    "TU_EPP_PARAMETERS", "TU_DLS_OVERRIDE",
})

# Optional (opt-in via ``with_tool_terms=True``) — the raw per-tool IPM error
# terms (Volve: ~836 rows). Off by default; ``errors.edm_ipm`` reads them into
# assembled models, this surfaces the raw term rows for a survey-program layer.
_TOOL_TERM_TAGS = frozenset({"DP_TOOL_TERM"})

# Human-readable schema — cryptic EDM table/field codes -> name + description.
# Exposed via :attr:`EDMReader.schema` so accessors, docs, and any downstream
# viewer are self-documenting rather than surfacing raw ``CD_*`` codes.
EDM_SCHEMA = {
    "CD_WELL": {
        "name": "Well",
        "description": "Well header (surface location, water depth).",
        "fields": {
            "well_common_name": "well name",
            "geo_offset_east": "surface easting (map)",
            "geo_offset_north": "surface northing (map)",
            "slot_ew": "slot east-west offset",
            "slot_ns": "slot north-south offset",
            "water_depth": "water depth",
        },
    },
    "CD_WELLBORE": {
        "name": "Wellbore",
        "description": "Wellbore / sidetrack (parent chain, kick-off).",
        "fields": {
            "wellbore_name": "wellbore name",
            "parent_wellbore_id": "parent wellbore",
            "ko_md": "kick-off MD",
        },
    },
    "CD_DATUM": {
        "name": "Depth datum",
        "description": "Per-rig, dated depth reference (RT/RKB elevation).",
        "fields": {
            "datum_name": "rig / datum name",
            "datum_elevation": "elevation above MSL",
            "date_last_modified": "established",
            "is_default": "default flag",
        },
    },
    "DP_MAGNETIC": {
        "name": "Geomagnetic reference",
        "description": "Per-wellbore reference field (the operator's own geomag "
                       "model) — feeds a SurveyHeader / interpretation reference.",
        "fields": {
            "field_strength": "total field strength (nT)",
            "dip_angle": "magnetic dip (deg)",
            "declination": "declination (deg)",
            "declination_date": "model date",
            "model_name": "geomag model (e.g. BGGM / IGRF)",
        },
    },
    "CD_PORE_PRESSURE": {
        "name": "Pore pressure",
        "description": "Pore-pressure prognosis point (pressure is canonical).",
        "fields": {
            "tvd": "TVD",
            "pore_pressure": "pressure (psi)",
            "pore_pressure_emw": "equiv. mud weight (ppg, RKB-datum view)",
            "is_permeable_zone": "permeable flag",
        },
    },
    "CD_FRAC_GRADIENT": {
        "name": "Fracture gradient",
        "description": "Fracture-pressure prognosis point.",
        "fields": {
            "tvd": "TVD",
            "frac_gradient_pressure": "pressure (psi)",
            "frac_gradient_emw": "equiv. mud weight (ppg, RKB-datum view)",
        },
    },
    "CD_TEMP_GRADIENT": {
        "name": "Temperature gradient",
        "description": "Geothermal temperature prognosis point (Volve degF).",
        "fields": {
            "tvd": "TVD",
            "temperature": "temperature (degF in Volve)",
        },
    },
    "CD_HOLE_SECT": {
        "name": "Hole section",
        "description": "As-run hole & casing section geometry.",
        "fields": {
            "sect_type_code": "CAS (cased) | OPEN",
            "od_casing": "casing OD",
            "id_casing": "casing ID",
            "id_drift": "drift ID",
            "hole_size": "hole size",
            "length": "section length",
            "md_shoe": "shoe MD",
            "catalog_key_desc": "catalogue description",
        },
    },
    "CD_CASE": {
        "name": "Case",
        "description": ("A design case/scenario; links a hole-section group "
                        "to a named case + phase."),
        "fields": {
            "case_name": "case name",
            "phase": "PLAN | ACTUAL | PROTOTYPE",
            "scenario_id": "scenario",
            "hole_sect_group_id": "hole-section group (geometry scenario)",
        },
    },
    "CD_WELLBORE_FORMATION": {
        "name": "Formation top",
        "description": "Prognosed formation top.",
        "fields": {
            "formation_name": "formation",
            "prognosed_md": "top MD",
            "prognosed_tvd": "top TVD",
        },
    },
    "CD_GRADE": {
        "name": "Pipe grade",
        "description": "Casing/tubing grade catalogue (yield, UTS).",
        "fields": {
            "grade": "grade name",
            "min_yield_stress": "min yield (field units)",
            "ultimate_tensile_strength": "UTS",
            "is_api": "API grade flag",
        },
    },
    "CD_MATERIAL": {
        "name": "Material",
        "description": "Pipe material catalogue (density, elastic props).",
        "fields": {
            "material": "material name",
            "density": "density (lb/ft3)",
            "youngs_modulus": "Young's modulus (field units)",
            "poissons_ratio": "Poisson's ratio",
            "expansion_coefficient": "thermal expansion coeff",
        },
    },
    "CD_ASSEMBLY": {
        "name": "Assembly",
        "description": "A string: casing / tubing / liner / BHA.",
        "fields": {
            "string_type": "Casing | Tubing | Liner | BHA | ...",
            "assembly_name": "string name",
            "assembly_size": "nominal size",
            "hole_size": "hole size",
            "md_assembly_base": "base MD",
            "tvd_assembly_base": "base TVD",
        },
    },
    "TU_LOAD_PROFILE": {
        "name": "Load profile",
        "description": ("An enabled design load profile + its parameters "
                        "(design factors, load inputs). Absent value = default."),
        "fields": {
            "profile_name": "load-profile name (e.g. BrstGasKickProfile)",
            "parameter_name": "parameter name (e.g. DesignPipeBurstFactor)",
            "parameter_value_num": "numeric value (absent -> app default)",
            "application_name": "originating app (e.g. StressCheck)",
        },
    },
    "TU_CUSTOM_LOAD_PROFILE": {
        "name": "Custom load profile",
        "description": "Per-depth load a string sees (the SF denominator).",
        "fields": {
            "measured_depth": "MD",
            "internal_pressure": "internal pressure",
            "external_pressure": "external pressure",
            "profile_name": "load-profile name",
        },
    },
    "TU_LOAD_HEADER": {
        "name": "Load header",
        "description": "A named load within a case, with type/category enums.",
        "fields": {
            "name": "load name (e.g. L.2.Pressure test the tubing)",
            "load_type": "load type code",
            "load_category": "0 burst | 1 collapse | 2 axial | 3 service",
            "sequence_no": "order within the case",
        },
    },
    "TU_EPP_PARAMETERS": {
        "name": "External-pressure params",
        "description": "Per-LOAD external-pressure profile parameters.",
        "fields": {
            "load_id": "the load these belong to (per-load, not per-case)",
            "name": "parameter name",
            "value_num": "numeric value",
        },
    },
    "TU_DLS_OVERRIDE": {
        "name": "Dogleg override",
        "description": ("An interval overriding the survey dogleg severity "
                        "(md_top may be below md_base -- use span)."),
        "fields": {
            "md_top": "interval top MD (not guaranteed shallower)",
            "md_base": "interval base MD",
            "dogleg_severity": "override DLS",
            "scenario_id": "scenario",
        },
    },
    "WP_FLUID_TEMP": {
        "name": "Fluid rheology",
        "description": "Temperature/pressure-resolved drilling-fluid rheology.",
        "fields": {
            "temperature": "temperature",
            "pressure": "pressure",
            "density": "density",
            "plastic_viscosity": "plastic viscosity",
            "yield_point": "yield point",
            "n_prime_parameter": "power-law n'",
            "k_prime_parameter": "power-law K'",
        },
    },
    "CD_ASSEMBLY_COMP": {
        "name": "Assembly component",
        "description": ("String component: geometry + material, plus Landmark's "
                        "stored axial/burst/collapse ratings on casing "
                        "components (a validation oracle -- welleng recomputes)."),
        "fields": {
            "sect_type_code": "DP | CAS | ... (section type)",
            "grade": "pipe grade",
            "od_body": "body OD (in)",
            "id_body": ("body ID (in) -- ⚠ can be a 3.000in collar-bore PLACEHOLDER "
                        "(detectable only by the same ID across DISTINCT ODs); "
                        "wrong on small pipe, overstates steel/weight"),
            "min_yield_stress": "min yield stress",
            "axial_rating": "stored pipe-body yield (klbf)",
            "pipe_pressure_burst": "stored burst rating (psi)",
            "pressure_collapse": "stored collapse rating (psi)",
            "critical_percent_collapse": "collapse derating INPUT (not a result)",
            "makeup_torque": "connection make-up torque",
            "catalog_key_desc": "description -> AssemblyComponent.description",
            "length": "component length (ft)",
            "approximate_weight": ("stated nominal unit weight lb/ft -- ⚠ UNVERIFIED; "
                                   "a tool-joint uplift convention can overstate it, "
                                   "cross-check against OD/ID geometry"),
            "od_connection": "connection OD (in)",
            "id_connection": "connection ID (in)",
            "length_tool_joint": "tool-joint length (ft)",
            "connection_name": "connection / thread name",
            "material / material_id / density / youngs_modulus / poissons_ratio":
                "material properties (for T&D / stress / sag consumers)",
        },
    },
}


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
    name: Optional[str], description: Optional[str] = None,
    remarks: Optional[str] = None,
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
    if kind is ToolKind.OTHER:
        kind = _classify(remarks)
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
    remarks: str = ""
    correlate: bool = False           # CD_SURVEY_TOOL.correlate 'Y'/'N' — whether
    #                                   this tool's errors correlate with another
    #                                   survey's (drives combined-survey weighting)
    is_range_limited: bool = False     # inclination-range gating applies
    inclination_range_min: Optional[float] = None   # deg
    inclination_range_max: Optional[float] = None   # deg
    kind: ToolKind = ToolKind.OTHER
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class ToolTerm:
    """A ``DP_TOOL_TERM`` row: one raw IPM error term of a survey tool.

    The raw per-tool error-model terms (opt-in ``with_tool_terms=True``). This is
    the un-normalised term dump; :func:`welleng.errors.edm_ipm.parse_edm_ipm`
    turns these into assembled error *models*.
    """

    survey_tool_id: str
    term_name: str
    sequence_no: Optional[int] = None
    # weight axis (mapping in errors.edm_ipm): e/d=depth, i/j=inclination,
    # a/b=azimuth, l=azimuth/sin(inc) (east-singularity), m=combined;
    # tie_type='n' rows are EDM intermediates, not error terms.
    vector_type: Optional[str] = None
    tie_type: Optional[str] = None      # s/r/g/w = Systematic/Random/Global/Well
    c_value: Optional[float] = None
    c_units: Optional[str] = None
    c_formula: Optional[str] = None
    c_range: Optional[int] = None       # 1 => the min/max activity window applies
    min_range: Optional[float] = None
    max_range: Optional[float] = None
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
class WellPosition:
    """A well's surface position (``CD_WELL``) — map offset + slot offset."""

    well_id: str
    name: str
    east: Optional[float]      # geo_offset_east (map easting)
    north: Optional[float]     # geo_offset_north (map northing)
    slot_ew: Optional[float]   # slot east-west offset
    slot_ns: Optional[float]   # slot north-south offset


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
    md_min: Optional[float] = None
    md_max: Optional[float] = None
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
class SurveyRun:
    """A per-interval survey-tool assignment: which tool ran over which
    measured-depth range, from ``CD_SURVEY_PROGRAM`` joined to its raw
    ``CD_SURVEY_HEADER`` (wellbore + survey name). The IPM survey-program layer a
    consumer builds a tool-runs table from."""

    wellbore_id: str
    survey_name: Optional[str]
    survey_tool_id: Optional[str]
    md_min: Optional[float]        # CD_SURVEY_PROGRAM md_top (the run's start MD)
    md_max: Optional[float]        # CD_SURVEY_PROGRAM md_base (the run's end MD)
    survey_header_id: Optional[str]
    def_survey_header_id: Optional[str]
    sequence_no: Optional[int]
    tool_name: Optional[str] = None
    tool_remarks: Optional[str] = None
    tool_kind: Optional[str] = None


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
            depth_unit=units,   # match the Survey's unit -- else its assert fires
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

@dataclass
class GeopressureProfile:
    """A phased pore/frac/temperature prognosis for one wellbore.

    ``value`` is **pressure in psi** for pore/frac (the canonical quantity) or
    **temperature** for a temp profile (Volve stores degF). ``emw`` (pore/frac
    only) is the stored equivalent mud weight in ppg — an **RKB-datum view**,
    for display only; never derive PPFP logic from it. ``tvd`` is in the
    it was the reader's ``source_units`` (feet on Volve).
    """

    kind: str                       # "pore" | "frac" | "temp"
    phase: Optional[str]            # PLAN | ACTUAL | PROTOTYPE
    name: str
    wellbore_id: str
    group_id: str
    update_date: Optional[str]
    tvd: np.ndarray
    value: np.ndarray
    emw: Optional[np.ndarray] = None
    permeable: Optional[np.ndarray] = None   # pore only: is_permeable_zone=="Y"/point
    # Gradient (linear) form: some groups store a slope on the group row instead
    # of a point list (temperature especially). When ``is_gradient_form``, the
    # tvd/value arrays hold the shallow anchors (surface, mudline) and the
    # geothermal below the reference is ``value_at(tvd) = reference_value +
    # gradient * (tvd - reference_tvd)``.
    is_gradient_form: bool = False
    gradient: Optional[float] = None        # per source-unit length
    reference_tvd: Optional[float] = None
    reference_value: Optional[float] = None
    surface_value: Optional[float] = None

    def value_at(self, tvd: float) -> Optional[float]:
        """Value at a TVD for a gradient-form profile (None if not gradient)."""
        if not self.is_gradient_form or self.gradient is None \
                or self.reference_tvd is None or self.reference_value is None:
            return None
        return self.reference_value + self.gradient * (tvd - self.reference_tvd)


@dataclass
class HoleSection:
    """A ``CD_HOLE_SECT`` row — one as-run hole/casing section."""

    wellbore_id: str
    sect_type_code: str             # "CAS" (cased) | "OPEN"
    od_casing: Optional[float]
    id_casing: Optional[float]
    id_drift: Optional[float]
    hole_size: Optional[float]
    length: Optional[float]
    md_shoe: Optional[float]
    group_id: str = ""              # hole_sect_group_id (the design scenario)
    description: str = ""           # catalog_key_desc
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class HoleSectionGroup:
    """A design scenario for a wellbore's geometry (``hole_sect_group_id``).

    Links (via ``CD_CASE``) to the case(s) that reference the group, so a caller
    can pick a scenario by name/phase instead of an opaque id.
    """

    group_id: str
    wellbore_id: str
    case_names: List[str]
    phases: List[str]
    n_sections: int


@dataclass
class Formation:
    """A ``CD_WELLBORE_FORMATION`` row — one prognosed formation top."""

    name: str
    wellbore_id: str
    phase: Optional[str]
    md: Optional[float]             # prognosed_md
    tvd: Optional[float]            # prognosed_tvd
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class Magnetics:
    """A ``DP_MAGNETIC`` row — the per-wellbore geomagnetic reference.

    The operator's own reference field (total intensity, dip and declination
    from the named model), used to build a :class:`~welleng.survey.SurveyHeader`
    (``b_total`` / ``dip`` / ``declination``) or an interpretation
    :class:`~welleng.interpretation.GeomagReference` from the operator's geomag
    rather than a generic model lookup.
    """

    wellbore_id: str
    b_total: Optional[float]        # field_strength (nT)
    dip: Optional[float]            # dip_angle (deg)
    declination: Optional[float]    # deg
    date: Optional[str]             # declination_date
    model: Optional[str]           # model_name (e.g. BGGM / IGRF)
    sequence_no: Optional[int]
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class AssemblyComponent:
    """A ``CD_ASSEMBLY_COMP`` row — one string component.

    Casing components additionally carry Landmark's **stored** ratings
    (``axial_rating`` klbf, ``pipe_pressure_burst`` psi, ``pressure_collapse``
    psi); these are ``None`` on unrated components (e.g. drill pipe). They are
    returned **verbatim as an oracle** -- welleng recomputes them, it does not
    treat them as truth. Note ``critical_percent_*`` are derating **inputs**
    (all 100.0 in Volve), not utilisation results. Full row in :attr:`raw`.

    Geometry / connection / material fields are surfaced verbatim in Landmark's
    stored units (OD/ID inches, lengths feet, ``approximate_weight`` lb/ft,
    moduli psi) -- the reader returns raw table data; the consumer converts and
    interprets. ⚠ ``approximate_weight`` is a stated nominal that does NOT always
    survive a geometry/measurement check (a tool-joint uplift convention can put
    it well above the plain-pipe weight); treat it as unverified, cross-check
    against the OD/ID geometry before trusting it.
    """

    assembly_id: str
    sect_type_code: str
    grade: str
    od_body: Optional[float]
    id_body: Optional[float]
    min_yield_stress: Optional[float]
    axial_rating: Optional[float]
    pipe_pressure_burst: Optional[float]
    pressure_collapse: Optional[float]
    sequence_no: Optional[int]
    # geometry / connection / mass (Landmark stored units; raw, see the ⚠ above)
    description: str = ""                        # catalog_key_desc
    length: Optional[float] = None              # component length, ft
    approximate_weight: Optional[float] = None  # nominal unit weight lb/ft (UNVERIFIED)
    od_connection: Optional[float] = None       # connection OD, in
    id_connection: Optional[float] = None       # connection ID, in
    length_tool_joint: Optional[float] = None   # tool-joint length, ft
    connection_name: str = ""                   # connection / thread name
    # material properties (for T&D / stress / sag consumers)
    material: str = ""                          # material name
    material_id: str = ""                       # material key
    density: Optional[float] = None             # material density (Landmark units)
    youngs_modulus: Optional[float] = None      # psi
    poissons_ratio: Optional[float] = None
    makeup_torque: Optional[float] = None       # connection make-up torque
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class LoadProfile:
    """An enabled design load profile for a case (``TU_LOAD_PROFILE``).

    ``parameters`` maps each parameter name to its value -- a float
    (``parameter_value_num``), a string (``parameter_value_txt``), or ``None``
    when the row carries neither, meaning the **application default** applies
    (NOT zero). Parameter semantics (design factors, load-case codes) are the
    design engine's, not the reader's.
    """

    case_id: str
    wellbore_id: str
    profile_name: str
    application: str
    parameters: Dict[str, object] = field(default_factory=dict)


@dataclass
class CustomLoadPoint:
    """One depth point of a custom load profile (``TU_CUSTOM_LOAD_PROFILE``)."""

    md: Optional[float]
    internal_pressure: Optional[float]
    external_pressure: Optional[float]

    @property
    def differential_pressure(self) -> Optional[float]:
        """internal - external (>0 burst, <0 collapse); ``None`` if either is."""
        if self.internal_pressure is None or self.external_pressure is None:
            return None
        return self.internal_pressure - self.external_pressure


@dataclass
class CustomLoadProfile:
    """A custom load profile for a (case, profile) -- the per-depth load."""

    case_id: str
    wellbore_id: str
    profile_name: str
    points: List[CustomLoadPoint] = field(default_factory=list)


@dataclass
class LoadHeader:
    """A named load within a case (``TU_LOAD_HEADER``)."""

    load_id: str
    case_id: str
    wellbore_id: str
    name: str
    load_type: Optional[int]
    load_category: Optional[int]
    sequence_no: Optional[int]


@dataclass
class DlsOverride:
    """A ``TU_DLS_OVERRIDE`` interval overriding the survey dogleg severity.

    ``md_top`` is NOT guaranteed shallower than ``md_base`` (inverted intervals
    occur) -- use :attr:`span` for an ordered (top, base) pair.
    """

    wellbore_id: str
    md_top: Optional[float]
    md_base: Optional[float]
    dogleg_severity: Optional[float]
    scenario_id: str = ""
    group_id: str = ""              # dls_override_group_id

    @property
    def span(self) -> tuple:
        """Ordered ``(top, base)`` MD, robust to inverted intervals."""
        a, b = self.md_top, self.md_base
        if a is None or b is None:
            return (a, b)
        return (min(a, b), max(a, b))


@dataclass
class FluidRheologyPoint:
    """A ``WP_FLUID_TEMP`` row — fluid rheology at one (temperature, pressure).

    ``fann`` is the joined raw Fann-viscometer data (rpm, deflection) from
    ``WP_FLUID_TEMP_FANN_DATA``. Rows without rheology numbers are skipped by
    :meth:`EDMReader.fluid_rheology` (an incomplete catalogue row is not a
    fluid definition).
    """

    fluid_id: str
    temp_id: str
    temperature: Optional[float]
    pressure: Optional[float]
    density: Optional[float]
    plastic_viscosity: Optional[float]
    yield_point: Optional[float]
    n_prime: Optional[float]
    k_prime: Optional[float]
    fann: List[tuple] = field(default_factory=list)   # (rpm, deflection)
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


@dataclass
class Assembly:
    """A ``CD_ASSEMBLY`` row — one string, with its components."""

    assembly_id: str
    wellbore_id: str
    string_type: str                # Casing | Tubing | Liner | BHA | ...
    name: str
    size: Optional[float]
    hole_size: Optional[float]
    md_base: Optional[float]
    tvd_base: Optional[float]
    components: List[AssemblyComponent] = field(default_factory=list)
    raw: Dict[str, str] = field(default_factory=dict, repr=False)


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


def _iter_rows(
    path: str, wanted: frozenset = _WANTED_TAGS
) -> Iterator[Tuple[str, Dict[str, str]]]:
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
        if elem.tag in wanted:
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
    with_geopressure : bool, default False
        Also materialise the geopressure / temperature prognoses, as-run
        geometry, formation tops, and grade/material catalogue (the large
        non-survey tables). Off by default keeps the light survey-only sweep.
        When ``True``, use :meth:`pore_pressure`, :meth:`frac_gradient`,
        :meth:`temperature`, :meth:`geometry`, :meth:`formations`,
        :meth:`grades`, :meth:`materials`.

    Attributes
    ----------
    schema : dict
        Human-readable name + description for each EDM table/field this reader
        surfaces (see :data:`EDM_SCHEMA`) -- so consumers, docs, and any viewer
        do not have to decode raw ``CD_*`` codes.
    """

    #: Human-readable table/field glossary (cryptic EDM code -> name + desc).
    schema = EDM_SCHEMA

    def __init__(
        self, path: str, source_units: str = "feet",
        with_geopressure: bool = False, with_assemblies: bool = False,
        with_load_cases: bool = False, with_tool_terms: bool = False,
    ):
        self.path = path
        self.source_units = source_units.lower()
        self.with_geopressure = with_geopressure
        self.with_assemblies = with_assemblies
        self.with_load_cases = with_load_cases
        self.with_tool_terms = with_tool_terms
        self._wanted = _WANTED_TAGS | (
            _GEOPRESSURE_TAGS if with_geopressure else frozenset()
        ) | (_ASSEMBLY_TAGS if with_assemblies else frozenset()) | (
            _LOAD_CASE_TAGS if with_load_cases else frozenset()) | (
            _TOOL_TERM_TAGS if with_tool_terms else frozenset())

        self.projects: Dict[str, Dict[str, str]] = {}
        self.sites: Dict[str, Dict[str, str]] = {}
        self.wells: Dict[str, Dict[str, str]] = {}
        self.wellbores: Dict[str, Wellbore] = {}
        self.datums: Dict[str, List[Dict[str, str]]] = {}  # by well_id
        self.tools: Dict[str, SurveyTool] = {}
        self.headers: Dict[str, SurveyHeader] = {}  # raw + definitive by id
        # Complete header list -- EDM reuses survey_header_id across distinct
        # surveys (a plan + actual + revisions can share an id), so the id-keyed
        # dict above is lossy; this keeps every parsed header for survey_headers().
        self._all_headers: List[SurveyHeader] = []
        self.programs: Dict[str, List[ProgramInterval]] = {}  # by def hdr id
        self._tool_terms: Dict[str, List[ToolTerm]] = {}  # by survey_tool_id
        self._magnetics: Dict[str, List[Magnetics]] = {}  # by wellbore_id

        # geopressure / geometry (populated only when with_geopressure)
        self._pp_groups: Dict[str, Dict[str, str]] = {}
        self._fg_groups: Dict[str, Dict[str, str]] = {}
        self._tg_groups: Dict[str, Dict[str, str]] = {}
        self._pp_rows: Dict[str, List[Dict[str, str]]] = {}   # by group_id
        self._fg_rows: Dict[str, List[Dict[str, str]]] = {}
        self._tg_rows: Dict[str, List[Dict[str, str]]] = {}
        self._hole_sects: Dict[str, List[HoleSection]] = {}   # by wellbore_id
        self._cases: Dict[str, List[Dict[str, str]]] = {}     # by hole_sect_group_id
        self._formations: Dict[str, List[Formation]] = {}     # by wellbore_id
        self.grades: Dict[str, Dict[str, str]] = {}           # by grade_id
        self.materials: Dict[str, Dict[str, str]] = {}        # by material_id

        # assemblies (populated only when with_assemblies)
        self._assemblies: Dict[str, Assembly] = {}            # by assembly_id
        self._assembly_comps: Dict[str, List[AssemblyComponent]] = {}  # by assembly_id

        # load cases (populated only when with_load_cases)
        self._load_profiles: Dict[tuple, LoadProfile] = {}    # by (case_id, profile)
        self._custom_loads: Dict[tuple, CustomLoadProfile] = {}  # by (case_id, profile)
        self._load_headers: List[LoadHeader] = []
        self._epp: Dict[str, Dict[str, object]] = {}          # load_id -> {name: value}
        self._epp_wb: Dict[str, str] = {}                     # load_id -> wellbore_id
        self._dls_overrides: Dict[str, List[DlsOverride]] = {}  # by wellbore_id

        # fluid rheology (populated only when with_geopressure)
        self._fluid_temp: Dict[tuple, FluidRheologyPoint] = {}  # by (fluid_id, temp_id)
        self._fann: Dict[tuple, List[tuple]] = {}             # by (fluid_id, temp_id)

        # station records grouped by header id (parsed floats)
        self._raw_stations: Dict[str, List[dict]] = {}
        self._def_stations: Dict[str, List[dict]] = {}

        # name -> id lookups (built after sweep)
        self.wellbore_name_to_id: Dict[str, str] = {}

        self._index()

    # -- construction --------------------------------------------------------
    @classmethod
    def open(
        cls, path: str, source_units: str = "feet",
        with_geopressure: bool = False, with_assemblies: bool = False,
        with_load_cases: bool = False, with_tool_terms: bool = False,
    ) -> "EDMReader":
        """Open and index an EDM file. Alias for the constructor."""
        return cls(path, source_units=source_units,
                   with_geopressure=with_geopressure,
                   with_assemblies=with_assemblies,
                   with_load_cases=with_load_cases,
                   with_tool_terms=with_tool_terms)

    def _index(self) -> None:
        for tag, a in _iter_rows(self.path, self._wanted):
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
            elif tag == "DP_MAGNETIC":
                self._magnetics.setdefault(a.get("wellbore_id", ""), []).append(
                    Magnetics(
                        wellbore_id=a.get("wellbore_id", ""),
                        b_total=_f(a, "field_strength"),
                        dip=_f(a, "dip_angle"),
                        declination=_f(a, "declination"),
                        date=a.get("declination_date"),
                        model=a.get("model_name"),
                        sequence_no=_i(a, "sequence_no"),
                        raw=a,
                    )
                )
            elif tag == "CD_SURVEY_TOOL":
                self.tools[a["survey_tool_id"]] = SurveyTool(
                    tool_id=a["survey_tool_id"],
                    name=a.get("tool_name", ""),
                    description=a.get("description", ""),
                    remarks=a.get("remarks", ""),
                    correlate=a.get("correlate", "N") == "Y",
                    is_range_limited=a.get("is_range_limited", "N") == "Y",
                    inclination_range_min=_f(a, "inclination_range_min"),
                    inclination_range_max=_f(a, "inclination_range_max"),
                    kind=classify_tool(
                        a.get("tool_name"), a.get("description"),
                        a.get("remarks"),
                    ),
                    raw=a,
                )
            elif tag == "CD_SURVEY_HEADER":
                h = SurveyHeader(
                    header_id=a["survey_header_id"],
                    wellbore_id=a.get("wellbore_id", ""),
                    kind="raw",
                    phase=a.get("phase"),
                    name=a.get("survey_name"),
                    tie_survey_header_id=a.get("tie_survey_header_id"),
                    survey_tool_id=a.get("survey_tool_id"),
                    md_min=_f(a, "md_min"),
                    md_max=_f(a, "md_max"),
                    raw=a,
                )
                self.headers[a["survey_header_id"]] = h  # id-keyed (lossy, joins)
                self._all_headers.append(h)               # complete (no dedup)
            elif tag == "DP_TOOL_TERM":
                self._tool_terms.setdefault(a["survey_tool_id"], []).append(
                    ToolTerm(
                        survey_tool_id=a["survey_tool_id"],
                        term_name=a.get("term_name", ""),
                        sequence_no=_i(a, "sequence_no"),
                        vector_type=a.get("vector_type"),
                        tie_type=a.get("tie_type"),
                        c_value=_f(a, "c_value"),
                        c_units=a.get("c_units"),
                        c_formula=a.get("c_formula"),
                        c_range=_i(a, "c_range"),
                        min_range=_f(a, "min_range"),
                        max_range=_f(a, "max_range"),
                        raw=a,
                    )
                )
            elif tag == "CD_DEFINITIVE_SURVEY_HEADER":
                h = SurveyHeader(
                    header_id=a["def_survey_header_id"],
                    wellbore_id=a.get("wellbore_id", ""),
                    kind="definitive",
                    phase=a.get("phase"),
                    name=a.get("name"),
                    tie_survey_header_id=a.get("tie_survey_header_id"),
                    survey_tool_id=a.get("survey_tool_id"),
                    raw=a,
                )
                self.headers[a["def_survey_header_id"]] = h
                self._all_headers.append(h)
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
            elif tag == "CD_PORE_PRESSURE_GROUP":
                self._pp_groups[a["pore_pressure_group_id"]] = a
            elif tag == "CD_PORE_PRESSURE":
                self._pp_rows.setdefault(
                    a["pore_pressure_group_id"], []).append(a)
            elif tag == "CD_FRAC_GRADIENT_GROUP":
                self._fg_groups[a["frac_gradient_group_id"]] = a
            elif tag == "CD_FRAC_GRADIENT":
                self._fg_rows.setdefault(
                    a["frac_gradient_group_id"], []).append(a)
            elif tag == "CD_TEMP_GRADIENT_GROUP":
                self._tg_groups[a["temp_gradient_group_id"]] = a
            elif tag == "CD_TEMP_GRADIENT":
                self._tg_rows.setdefault(
                    a["temp_gradient_group_id"], []).append(a)
            elif tag == "CD_HOLE_SECT":
                self._hole_sects.setdefault(
                    a.get("wellbore_id", ""), []).append(HoleSection(
                        wellbore_id=a.get("wellbore_id", ""),
                        sect_type_code=a.get("sect_type_code", ""),
                        od_casing=_f(a, "od_casing"),
                        id_casing=_f(a, "id_casing"),
                        id_drift=_f(a, "id_drift"),
                        hole_size=_f(a, "hole_size"),
                        length=_f(a, "length"),
                        md_shoe=_f(a, "md_shoe"),
                        group_id=a.get("hole_sect_group_id", ""),
                        description=a.get("catalog_key_desc", ""),
                        raw=a,
                    ))
            elif tag == "CD_CASE":
                hsg = a.get("hole_sect_group_id")
                if hsg:
                    self._cases.setdefault(hsg, []).append(a)
            elif tag == "CD_WELLBORE_FORMATION":
                self._formations.setdefault(
                    a.get("wellbore_id", ""), []).append(Formation(
                        name=a.get("formation_name", ""),
                        wellbore_id=a.get("wellbore_id", ""),
                        phase=a.get("phase"),
                        md=_f(a, "prognosed_md"),
                        tvd=_f(a, "prognosed_tvd"),
                        raw=a,
                    ))
            elif tag == "CD_GRADE":
                self.grades[a["grade_id"]] = a
            elif tag == "CD_MATERIAL":
                self.materials[a["material_id"]] = a
            elif tag == "CD_ASSEMBLY":
                self._assemblies[a["assembly_id"]] = Assembly(
                    assembly_id=a["assembly_id"],
                    wellbore_id=a.get("wellbore_id", ""),
                    string_type=a.get("string_type", ""),
                    name=a.get("assembly_name", ""),
                    size=_f(a, "assembly_size"),
                    hole_size=_f(a, "hole_size"),
                    md_base=_f(a, "md_assembly_base"),
                    tvd_base=_f(a, "tvd_assembly_base"),
                    raw=a,
                )
            elif tag == "CD_ASSEMBLY_COMP":
                self._assembly_comps.setdefault(
                    a.get("assembly_id", ""), []).append(AssemblyComponent(
                        assembly_id=a.get("assembly_id", ""),
                        sect_type_code=a.get("sect_type_code", ""),
                        grade=a.get("grade", ""),
                        od_body=_f(a, "od_body"),
                        id_body=_f(a, "id_body"),
                        min_yield_stress=_f(a, "min_yield_stress"),
                        axial_rating=_f(a, "axial_rating"),
                        pipe_pressure_burst=_f(a, "pipe_pressure_burst"),
                        pressure_collapse=_f(a, "pressure_collapse"),
                        sequence_no=_i(a, "sequence_no"),
                        description=a.get("catalog_key_desc", ""),
                        length=_f(a, "length"),
                        approximate_weight=_f(a, "approximate_weight"),
                        od_connection=_f(a, "od_connection"),
                        id_connection=_f(a, "id_connection"),
                        length_tool_joint=_f(a, "length_tool_joint"),
                        connection_name=a.get("connection_name", ""),
                        material=a.get("material", ""),
                        material_id=a.get("material_id", ""),
                        density=_f(a, "density"),
                        youngs_modulus=_f(a, "youngs_modulus"),
                        poissons_ratio=_f(a, "poissons_ratio"),
                        makeup_torque=_f(a, "makeup_torque"),
                        raw=a,
                    ))
            elif tag == "TU_LOAD_PROFILE":
                key = (a.get("case_id", ""), a.get("profile_name", ""))
                lp = self._load_profiles.get(key)
                if lp is None:
                    lp = LoadProfile(
                        case_id=a.get("case_id", ""),
                        wellbore_id=a.get("wellbore_id", ""),
                        profile_name=a.get("profile_name", ""),
                        application=a.get("application_name", ""),
                    )
                    self._load_profiles[key] = lp
                pname = a.get("parameter_name")
                if pname is not None:
                    if "parameter_value_num" in a:
                        lp.parameters[pname] = _f(a, "parameter_value_num")
                    elif "parameter_value_txt" in a:
                        lp.parameters[pname] = a["parameter_value_txt"]
                    else:
                        lp.parameters[pname] = None  # default applies (not 0)
            elif tag == "TU_CUSTOM_LOAD_PROFILE":
                key = (a.get("case_id", ""), a.get("profile_name", ""))
                cp = self._custom_loads.get(key)
                if cp is None:
                    cp = CustomLoadProfile(
                        case_id=a.get("case_id", ""),
                        wellbore_id=a.get("wellbore_id", ""),
                        profile_name=a.get("profile_name", ""),
                    )
                    self._custom_loads[key] = cp
                cp.points.append(CustomLoadPoint(
                    md=_f(a, "measured_depth"),
                    internal_pressure=_f(a, "internal_pressure"),
                    external_pressure=_f(a, "external_pressure"),
                ))
            elif tag == "TU_LOAD_HEADER":
                self._load_headers.append(LoadHeader(
                    load_id=a.get("load_id", ""),
                    case_id=a.get("case_id", ""),
                    wellbore_id=a.get("wellbore_id", ""),
                    name=a.get("name", ""),
                    load_type=_i(a, "load_type"),
                    load_category=_i(a, "load_category"),
                    sequence_no=_i(a, "sequence_no"),
                ))
            elif tag == "TU_EPP_PARAMETERS":
                lid = a.get("load_id", "")
                pname = a.get("name")
                if pname is not None:
                    val = (_f(a, "value_num") if "value_num" in a
                           else a.get("value_txt"))
                    self._epp.setdefault(lid, {})[pname] = val
                    self._epp_wb[lid] = a.get("wellbore_id", "")
            elif tag == "TU_DLS_OVERRIDE":
                self._dls_overrides.setdefault(
                    a.get("wellbore_id", ""), []).append(DlsOverride(
                        wellbore_id=a.get("wellbore_id", ""),
                        md_top=_f(a, "md_top"),
                        md_base=_f(a, "md_base"),
                        dogleg_severity=_f(a, "dogleg_severity"),
                        scenario_id=a.get("scenario_id", ""),
                        group_id=a.get("dls_override_group_id", ""),
                    ))
            elif tag == "WP_FLUID_TEMP":
                key = (a.get("fluid_id", ""), a.get("temp_id", ""))
                self._fluid_temp[key] = FluidRheologyPoint(
                    fluid_id=a.get("fluid_id", ""),
                    temp_id=a.get("temp_id", ""),
                    temperature=_f(a, "temperature"),
                    pressure=_f(a, "pressure"),
                    density=_f(a, "density"),
                    plastic_viscosity=_f(a, "plastic_viscosity"),
                    yield_point=_f(a, "yield_point"),
                    n_prime=_f(a, "n_prime_parameter"),
                    k_prime=_f(a, "k_prime_parameter"),
                    raw=a,
                )
            elif tag == "WP_FLUID_TEMP_FANN_DATA":
                key = (a.get("fluid_id", ""), a.get("temp_id", ""))
                self._fann.setdefault(key, []).append(
                    (_f(a, "rpm"), _f(a, "deflection")))
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
                # CD_SURVEY_PROGRAM rows may omit survey_tool_id; fall back to
                # the tool of the linked raw CD_SURVEY_HEADER so the definitive
                # tie-chain still resolves a per-leg error model (e.g. Volve
                # F-12, whose program interval carries no survey_tool_id).
                tool_id = iv.survey_tool_id
                if tool_id is None:
                    hdr = self.headers.get(iv.survey_header_id)
                    tool_id = hdr.survey_tool_id if hdr else None
                if tool_id is not None:
                    iv.tool = self.tools.get(tool_id)
        # attach components to their assembly, ordered bottom-up by sequence_no
        for aid, asm in self._assemblies.items():
            asm.components = sorted(
                self._assembly_comps.get(aid, []),
                key=lambda c: c.sequence_no or 0)

    # -- lookups -------------------------------------------------------------
    def _resolve_wellbore(self, wellbore) -> Wellbore:
        if isinstance(wellbore, Wellbore):
            return wellbore
        if wellbore in self.wellbores:
            return self.wellbores[wellbore]
        if wellbore in self.wellbore_name_to_id:
            return self.wellbores[self.wellbore_name_to_id[wellbore]]
        raise KeyError(f"unknown wellbore: {wellbore!r}")

    def _resolve_well_id(self, well) -> str:
        if well in self.wells:
            return well
        for wid, w in self.wells.items():
            if w.get("well_common_name") == well:
                return wid
        raise KeyError(f"unknown well: {well!r}")

    def well_position(self, well) -> WellPosition:
        """Surface position of a well (map offset + slot offset). Accepts a
        ``well_id`` or a ``well_common_name``. Always available (no opt-in)."""
        wid = self._resolve_well_id(well)
        a = self.wells[wid]
        return WellPosition(
            well_id=wid, name=a.get("well_common_name", wid),
            east=_f(a, "geo_offset_east"), north=_f(a, "geo_offset_north"),
            slot_ew=_f(a, "slot_ew"), slot_ns=_f(a, "slot_ns"),
        )

    def find_wellbores(self, substring: str) -> List[Wellbore]:
        """Return wellbores whose name contains ``substring`` (case-insensitive)."""
        s = substring.lower()
        return [
            wb for wb in self.wellbores.values() if s in wb.name.lower()
        ]

    # -- geopressure / geometry (opt-in) -------------------------------------
    def _require_geopressure(self) -> None:
        if not self.with_geopressure:
            raise RuntimeError(
                "geopressure/geometry tables were not indexed; open the reader "
                "with with_geopressure=True"
            )

    def _length_factor(self, units: str) -> float:
        """Multiplier from the reader's ``source_units`` to ``units`` (as in
        :meth:`WellboreSurvey.to_welleng`). Geopressure ``tvd`` is stored in
        source units (feet on Volve); convert it so it lines up with the
        metres a survey ``to_welleng`` returns."""
        src, u = self.source_units.lower(), units.lower()
        if src == u:
            return 1.0
        if src == "feet" and u in ("meters", "metres", "m"):
            return FEET_TO_METERS
        if src in ("meters", "metres") and u == "feet":
            return 1.0 / FEET_TO_METERS
        return 1.0

    def _profiles(self, kind, groups, rows, wellbore, phase, latest,
                  value_key, emw_key=None, units: Optional[str] = None
                  ) -> List[GeopressureProfile]:
        self._require_geopressure()
        wb = self._resolve_wellbore(wellbore).wellbore_id
        # Default to the reader's own source_units so one open_edm(...) call is
        # internally consistent -- geopressure TVD matches survey/formation TVD
        # (welleng#design-flag 2026-08-11). `units=` overrides per call.
        factor = self._length_factor(units or self.source_units)
        out = []
        for gid, g in groups.items():
            if g.get("wellbore_id") != wb:
                continue
            if phase is not None and g.get("phase") != phase:
                continue
            recs = sorted(rows.get(gid, []),
                          key=lambda r: _f(r, "tvd") or 0.0)
            if not recs:
                # No point rows: a group may instead store the LINEAR-gradient
                # form on its own row (temperature especially -- mudline_temp +
                # grad_tvd*(tvd - temp_tvd)). Synthesize a usable profile.
                grad = self._gradient_form_profile(kind, gid, g, wb, factor)
                if grad is not None:
                    out.append(grad)
                continue
            out.append(GeopressureProfile(
                kind=kind, phase=g.get("phase"), name=g.get("name", ""),
                wellbore_id=wb, group_id=gid, update_date=g.get("update_date"),
                tvd=np.array([_f(r, "tvd") for r in recs], dtype=float) * factor,
                value=np.array([_f(r, value_key) for r in recs], dtype=float),
                emw=(np.array([_f(r, emw_key) for r in recs], dtype=float)
                     if emw_key else None),
                permeable=(np.array([r.get("is_permeable_zone") == "Y"
                                     for r in recs]) if kind == "pore" else None),
            ))
        if latest and out:
            # Richest group (most points) = the full-depth prognosis; ties ->
            # PLAN. In Volve the ACTUAL phase is a single-point shallow anchor
            # (~seabed ref, copied across wells), so a date-based "latest" lands
            # on the anchor and drops the curve -- pick by point count instead.
            out.sort(key=lambda p: (len(p.tvd), p.phase == "PLAN"), reverse=True)
            return out[:1]
        return out

    def _gradient_form_profile(self, kind, gid, g, wb, factor: float = 1.0):
        """Synthesize a profile from the LINEAR-gradient attributes on a group
        row (no point children). Verified for temperature (``grad_tvd`` degF/ft
        from ``temp_tvd``/``mudline_temp``); returns ``None`` for other kinds
        (pore/frac gradient-attribute forms are not yet confirmed in Volve).

        ``factor`` converts source-unit lengths to the requested units: tvd /
        reference_tvd scale by ``factor``, and the per-length ``gradient`` by
        ``1/factor`` so ``value_at(tvd)`` is unchanged."""
        if kind != "temp":
            return None
        grad = _f(g, "grad_tvd")
        ref_tvd = _f(g, "temp_tvd")
        mudline = _f(g, "mudline_temp")
        surface = _f(g, "surface_ambient_temp")
        if grad is None or ref_tvd is None or mudline is None:
            return None
        ref_tvd *= factor
        grad = grad / factor
        # shallow anchors: surface (tvd 0) -> mudline (ref_tvd); gradient below
        if surface is not None:
            tvd = np.array([0.0, ref_tvd], dtype=float)
            val = np.array([surface, mudline], dtype=float)
        else:
            tvd = np.array([ref_tvd], dtype=float)
            val = np.array([mudline], dtype=float)
        return GeopressureProfile(
            kind=kind, phase=g.get("phase"), name=g.get("name", ""),
            wellbore_id=wb, group_id=gid, update_date=g.get("update_date"),
            tvd=tvd, value=val, is_gradient_form=True, gradient=grad,
            reference_tvd=ref_tvd, reference_value=mudline, surface_value=surface,
        )

    def pore_pressure(self, wellbore, phase: Optional[str] = None,
                      latest: bool = True, units: Optional[str] = None
                      ) -> List[GeopressureProfile]:
        """Pore-pressure prognoses (pressure in psi is canonical).

        ``phase`` filters PLAN | ACTUAL | PROTOTYPE (default ``None`` = all;
        note Volve's ACTUAL is a single-point shallow anchor, not a curve).
        ``latest`` keeps only the RICHEST group (the full-depth prognosis; ties
        -> PLAN) rather than a date-newest one. ``tvd`` is returned in ``units``,
        which defaults to the reader's ``source_units`` so it matches the survey
        and formation TVD from the same reader; pass ``units="meters"`` for SI.
        Pressure stays psi.
        """
        return self._profiles(
            "pore", self._pp_groups, self._pp_rows, wellbore, phase, latest,
            "pore_pressure", "pore_pressure_emw", units=units)

    def frac_gradient(self, wellbore, phase: Optional[str] = None,
                      latest: bool = True, units: Optional[str] = None
                      ) -> List[GeopressureProfile]:
        """Fracture-gradient prognoses (pressure in psi is canonical). ``tvd``
        in ``units`` (default: the reader's ``source_units``); see
        :meth:`pore_pressure` for the phase / richest-group / units behaviour."""
        return self._profiles(
            "frac", self._fg_groups, self._fg_rows, wellbore, phase, latest,
            "frac_gradient_pressure", "frac_gradient_emw", units=units)

    def temperature(self, wellbore, phase: Optional[str] = None,
                    latest: bool = True, units: Optional[str] = None
                    ) -> List[GeopressureProfile]:
        """Temperature (geothermal) prognoses. Volve stores degF; ``value``
        is the raw stored temperature (no unit conversion). ``tvd`` in
        ``units`` (default: the reader's ``source_units``)."""
        return self._profiles(
            "temp", self._tg_groups, self._tg_rows, wellbore, phase, latest,
            "temperature", units=units)

    def geometry(self, wellbore, group_id: Optional[str] = None
                 ) -> List[HoleSection]:
        """As-run hole/casing sections (OD/ID/drift/hole size/shoe MD), by shoe MD.

        ``group_id`` selects one design scenario (a ``hole_sect_group_id`` from
        :meth:`hole_section_groups`) -- returned as-is, no dedup. With no
        ``group_id`` the sections are pooled across scenarios and
        exact-duplicate rows collapsed (a convenience view; use a group for a
        single coherent string).
        """
        self._require_geopressure()
        wb = self._resolve_wellbore(wellbore).wellbore_id
        secs = sorted((h for h in self._hole_sects.get(wb, [])
                       if group_id is None or h.group_id == group_id),
                      key=lambda h: h.md_shoe or 0.0)
        if group_id is not None:
            return secs
        seen, uniq = set(), []
        for h in secs:
            key = (h.sect_type_code, h.od_casing, h.id_casing, h.id_drift,
                   h.hole_size, h.length, h.md_shoe)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(h)
        return uniq

    def hole_section_groups(self, wellbore) -> List[HoleSectionGroup]:
        """The geometry design scenarios for a wellbore.

        Each is a ``hole_sect_group_id`` with the case name(s)/phase(s) that
        reference it (via ``CD_CASE``) and its section count -- so a caller can
        pick a scenario for :meth:`geometry` by name/phase, not an opaque id.
        """
        self._require_geopressure()
        wb = self._resolve_wellbore(wellbore).wellbore_id
        counts: Dict[str, int] = {}
        for h in self._hole_sects.get(wb, []):
            counts[h.group_id] = counts.get(h.group_id, 0) + 1
        out = []
        for gid, n in counts.items():
            cases = self._cases.get(gid, [])
            out.append(HoleSectionGroup(
                group_id=gid, wellbore_id=wb,
                case_names=[c.get("case_name", "") for c in cases],
                phases=sorted({c.get("phase", "") for c in cases if c.get("phase")}),
                n_sections=n,
            ))
        return sorted(out, key=lambda g: -g.n_sections)

    def formations(self, wellbore) -> List[Formation]:
        """Prognosed formation tops for a wellbore, by MD."""
        self._require_geopressure()
        wb = self._resolve_wellbore(wellbore).wellbore_id
        return sorted(self._formations.get(wb, []),
                      key=lambda f: f.md or 0.0)

    def magnetics(self, wellbore) -> Optional[Magnetics]:
        """The wellbore's geomagnetic reference (operator's own model).

        Returns the operative ``DP_MAGNETIC`` row (highest ``sequence_no``) as a
        :class:`Magnetics` (``b_total`` / ``dip`` / ``declination`` / ``date`` /
        ``model``), or ``None`` if the EDM has no magnetics for the wellbore. Use
        it to build a per-well :class:`~welleng.survey.SurveyHeader` or an
        interpretation :class:`~welleng.interpretation.GeomagReference` from the
        operator's geomag rather than a generic model lookup. Always available
        (no opt-in flag). Access every row via ``magnetics_all``.
        """
        rows = self._magnetics.get(self._resolve_wellbore(wellbore).wellbore_id, [])
        if not rows:
            return None
        return max(rows, key=lambda m: m.sequence_no or 0)

    def magnetics_all(self, wellbore) -> List[Magnetics]:
        """All ``DP_MAGNETIC`` rows for a wellbore, by ``sequence_no``."""
        rows = self._magnetics.get(self._resolve_wellbore(wellbore).wellbore_id, [])
        return sorted(rows, key=lambda m: m.sequence_no or 0)

    def tool_terms(self, survey_tool_id: str) -> List[ToolTerm]:
        """Raw ``DP_TOOL_TERM`` IPM error terms for a survey tool, by
        ``sequence_no`` (opt-in: ``with_tool_terms=True``).

        The un-normalised per-tool term rows (``term_name`` / ``c_value`` /
        ``c_units`` / ``c_formula`` / ``vector_type`` / ``tie_type`` / range).
        For the assembled error *model* use
        :func:`welleng.errors.edm_ipm.parse_edm_ipm`; this is the raw dump a
        survey-program / IPM layer builds from. Empty list if the tool has none.
        """
        if not self.with_tool_terms:
            raise RuntimeError(
                "tool-term table (DP_TOOL_TERM) was not indexed; open the reader "
                "with with_tool_terms=True."
            )
        return sorted(self._tool_terms.get(survey_tool_id, []),
                      key=lambda t: t.sequence_no or 0)

    def survey_runs(self, wellbore=None) -> List[SurveyRun]:
        """Per-interval survey-tool assignments — which tool ran over which
        measured-depth range — from ``CD_SURVEY_PROGRAM`` joined to its raw
        ``CD_SURVEY_HEADER`` (wellbore + survey name).

        One :class:`SurveyRun` per program interval, ordered by wellbore then
        ``md_min``. Optionally filtered to one ``wellbore`` (id or common name).
        Always available (CD_SURVEY_PROGRAM is in the default sweep); the raw
        header supplies ``wellbore_id`` / ``survey_name``, the program interval
        supplies ``survey_tool_id`` and the ``md_min``/``md_max`` run range.
        """
        wb_id = (self._resolve_wellbore(wellbore).wellbore_id
                 if wellbore is not None else None)
        out: List[SurveyRun] = []
        for intervals in self.programs.values():
            for iv in intervals:
                hdr = self.headers.get(iv.survey_header_id)
                wbid = hdr.wellbore_id if hdr else ""
                if wb_id is not None and wbid != wb_id:
                    continue
                # Effective tool: the program interval's own tool, else the
                # linked header's tool (same header-fallback as the finalise
                # loop). Surfaces the resolved tool name/remarks/kind so the
                # tie-chain reads without a second lookup.
                eff_tool_id = iv.survey_tool_id
                if eff_tool_id is None and hdr is not None:
                    eff_tool_id = hdr.survey_tool_id
                tool = iv.tool or (
                    self.tools.get(eff_tool_id) if eff_tool_id else None)
                out.append(SurveyRun(
                    wellbore_id=wbid,
                    survey_name=hdr.name if hdr else None,
                    survey_tool_id=eff_tool_id,
                    md_min=iv.md_top,
                    md_max=iv.md_base,
                    survey_header_id=iv.survey_header_id,
                    def_survey_header_id=iv.def_survey_header_id,
                    sequence_no=iv.sequence_no,
                    tool_name=tool.name if tool else None,
                    tool_remarks=tool.remarks if tool else None,
                    tool_kind=tool.kind.name if tool else None,
                ))
        out.sort(key=lambda r: (
            r.wellbore_id, r.md_min if r.md_min is not None else 0.0))
        return out

    def assemblies(self, wellbore=None) -> List[Assembly]:
        """String assemblies (casing/tubing/liner/BHA), each with its components.

        With ``wellbore`` given, only that wellbore's strings; otherwise all.
        Casing components carry Landmark's stored axial/burst/collapse ratings
        (the oracle) -- see :class:`AssemblyComponent`. Requires
        ``with_assemblies=True``.
        """
        if not self.with_assemblies:
            raise RuntimeError(
                "assemblies were not indexed; open the reader with "
                "with_assemblies=True"
            )
        asms = list(self._assemblies.values())
        if wellbore is not None:
            wb = self._resolve_wellbore(wellbore).wellbore_id
            asms = [a for a in asms if a.wellbore_id == wb]
        return sorted(asms, key=lambda a: a.md_base or 0.0)

    def _require_load_cases(self) -> None:
        if not self.with_load_cases:
            raise RuntimeError(
                "load-case tables were not indexed; open the reader with "
                "with_load_cases=True"
            )

    def load_profiles(self, wellbore=None, application: Optional[str] = "StressCheck"
                      ) -> List[LoadProfile]:
        """Enabled design load profiles (design factors + load params), grouped
        per (case, profile). ``application`` filters by app (default StressCheck;
        ``None`` for all). Requires ``with_load_cases=True``."""
        self._require_load_cases()
        wb = self._resolve_wellbore(wellbore).wellbore_id if wellbore else None
        return [lp for lp in self._load_profiles.values()
                if (wb is None or lp.wellbore_id == wb)
                and (application is None or lp.application == application)]

    def custom_load_profiles(self, wellbore=None) -> List[CustomLoadProfile]:
        """Per-depth custom load profiles (the load a string sees), points by MD.
        Requires ``with_load_cases=True``."""
        self._require_load_cases()
        wb = self._resolve_wellbore(wellbore).wellbore_id if wellbore else None
        out = [cp for cp in self._custom_loads.values()
               if wb is None or cp.wellbore_id == wb]
        for cp in out:
            cp.points.sort(key=lambda p: p.md or 0.0)
        return out

    def load_headers(self, wellbore=None) -> List[LoadHeader]:
        """Named loads per case, sequence-ordered. Requires
        ``with_load_cases=True``."""
        self._require_load_cases()
        wb = self._resolve_wellbore(wellbore).wellbore_id if wellbore else None
        hs = [h for h in self._load_headers if wb is None or h.wellbore_id == wb]
        return sorted(hs, key=lambda h: (h.case_id, h.sequence_no or 0))

    def epp_parameters(self, wellbore=None) -> Dict[str, Dict[str, object]]:
        """External-pressure-profile parameters, keyed by ``load_id`` ->
        ``{name: value}``. These are PER-LOAD (each load carries its own set;
        values genuinely differ across loads of a case). Requires
        ``with_load_cases=True``."""
        self._require_load_cases()
        if wellbore is None:
            return self._epp
        wb = self._resolve_wellbore(wellbore).wellbore_id
        return {lid: v for lid, v in self._epp.items()
                if self._epp_wb.get(lid) == wb}

    def dls_overrides(self, wellbore) -> List[DlsOverride]:
        """Dogleg-severity override intervals for a wellbore (empty if none --
        which means the survey DLS governs). Requires ``with_load_cases=True``."""
        self._require_load_cases()
        wb = self._resolve_wellbore(wellbore).wellbore_id
        return sorted(self._dls_overrides.get(wb, []),
                      key=lambda d: d.span[0] or 0.0)

    # --- design-load cluster (raw-row readers) --------------------------------
    # CD_CASE / TU_CASE_ASSEMBLY_PARAMETER / TU_COMP_TEMP_DERATION_POINT hold the
    # casing/tubing design load cases, their input parameters, and per-component
    # yield deration. These return RAW rows -- the casing-design interpretation
    # (SF / collapse) is the consumer's. They read the EDM directly (one pass
    # over the requested table per call); cache if called hot.

    @staticmethod
    def _app_match(value: Optional[str], application: Optional[str]) -> bool:
        # case-insensitive substring so 'stresscheck' matches the versioned
        # 'StressCheck 5000.1', and 'wellcat' matches the upper-case 'WELLCAT'.
        if application is None:
            return True
        return value is not None and application.lower() in value.lower()

    def design_cases(
        self, wellbore=None, application: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """``CD_CASE`` rows -- the casing/tubing design load cases -- as raw
        dicts. Filter by ``wellbore`` (id or name) and/or ``application``.

        NOTE: ``CD_CASE`` carries no ``application_name``; the authoring app is
        in ``create_app_id`` (versioned strings, e.g. ``'StressCheck 5000.1'``,
        ``'WELLCAT'``, ``'COMPASS'``, ``'WELLPLAN ...'``). ``application`` matches
        it case-insensitively as a substring, so ``'StressCheck'`` /
        ``'WellCat'`` both work. Keys include case_id, case_name, phase,
        scenario_id, assembly_id, fluid_id, hole_sect_group_id, create_app_id.

        NOTE: ``application`` here means "AUTHORED by that tool" (create_app_id).
        To enumerate cases that carry a load CONFIG for a tool -- including
        COMPASS-authored cases that StressCheck wrote params onto -- use
        :meth:`cases_with_load_config`.
        """
        wb = self._resolve_wellbore(wellbore).wellbore_id if wellbore else None
        out: List[Dict[str, str]] = []
        for _tag, a in _iter_rows(self.path, frozenset({"CD_CASE"})):
            if wb is not None and a.get("wellbore_id") != wb:
                continue
            if not self._app_match(a.get("create_app_id"), application):
                continue
            out.append(dict(a))
        return out

    def case_parameters(
        self, case_id: str, application: Optional[str] = None
    ) -> Dict[str, Optional[float]]:
        """``TU_CASE_ASSEMBLY_PARAMETER`` for one ``case_id`` as
        ``{parameter_name: parameter_value_num}`` (e.g. ``'InitAppSurfPressure'``
        -- the load-case input values). ``application`` filters on the row's
        ``application_name`` (values ``'StressCheck'`` / ``'WELLCAT'``),
        case-insensitive.
        """
        out: Dict[str, Optional[float]] = {}
        for _tag, a in _iter_rows(
            self.path, frozenset({"TU_CASE_ASSEMBLY_PARAMETER"})
        ):
            if a.get("case_id") != case_id:
                continue
            if not self._app_match(a.get("application_name"), application):
                continue
            name = a.get("parameter_name")
            if name is not None:
                out[name] = _f(a, "parameter_value_num")
        return out

    def temp_deration(
        self, assembly_comp_id: str
    ) -> List[Tuple[float, float]]:
        """``TU_COMP_TEMP_DERATION_POINT`` for one component as sorted
        ``(deration_temperature, correction_factor)`` points -- the yield
        deration curve. Derated yield = base_yield * interp(correction_factor, T).
        """
        pts: List[Tuple[float, float]] = []
        for _tag, a in _iter_rows(
            self.path, frozenset({"TU_COMP_TEMP_DERATION_POINT"})
        ):
            if a.get("assembly_comp_id") != assembly_comp_id:
                continue
            t = _f(a, "deration_temperature")
            c = _f(a, "correction_factor")
            if t is not None and c is not None:
                pts.append((t, c))
        return sorted(pts)

    def case_temp_gradient(
        self, case_id: str, application: Optional[str] = None
    ) -> Dict[str, List[Tuple[float, float]]]:
        """``CD_CASE_TEMP_GRADIENT`` for one ``case_id`` as
        ``{profile_name: sorted [(md, casing_temp)]}`` -- the per-case
        temperature profile(s) (the WellCat "case temperature" that seeds the
        deration + APB baseline). ``md`` is in the file's source units (feet);
        ``casing_temp`` is raw (degF). ``application`` filters StressCheck vs
        WellCat.
        """
        out: Dict[str, List[Tuple[float, float]]] = {}
        for _tag, a in _iter_rows(
            self.path, frozenset({"CD_CASE_TEMP_GRADIENT"})
        ):
            if a.get("case_id") != case_id:
                continue
            if not self._app_match(a.get("application_name"), application):
                continue
            md = _f(a, "md")
            t = _f(a, "casing_temp")
            if md is None or t is None:
                continue
            out.setdefault(a.get("profile_name") or "", []).append((md, t))
        for pts in out.values():
            pts.sort()
        return out

    def cases_with_load_config(
        self, application: str = "StressCheck", wellbore=None
    ) -> List[Dict[str, str]]:
        """``CD_CASE`` rows for the cases that CARRY a load configuration for
        ``application`` -- i.e. that have ``TU_CASE_ASSEMBLY_PARAMETER`` rows
        with that ``application_name``. These are the **designable** cases.

        This differs from :meth:`design_cases` on purpose: ``design_cases`` filters
        by ``CD_CASE.create_app_id`` (the tool that AUTHORED the case), whereas
        this enumerates by where the load params actually LIVE. A case authored in
        COMPASS (the trajectory/link tool) can carry a StressCheck load config —
        it is returned here but NOT by ``design_cases(application='StressCheck')``.
        (Volve: 422 cases carry StressCheck params vs 169 StressCheck-authored;
        279 of the difference are COMPASS-authored.) Use this to enumerate the
        designable cases, then :meth:`case_parameters` per case_id.
        """
        want: set = set()
        for _tag, a in _iter_rows(
            self.path, frozenset({"TU_CASE_ASSEMBLY_PARAMETER"})
        ):
            if self._app_match(a.get("application_name"), application):
                cid = a.get("case_id")
                if cid is not None:
                    want.add(cid)
        wb = self._resolve_wellbore(wellbore).wellbore_id if wellbore else None
        out: List[Dict[str, str]] = []
        for _tag, a in _iter_rows(self.path, frozenset({"CD_CASE"})):
            if a.get("case_id") not in want:
                continue
            if wb is not None and a.get("wellbore_id") != wb:
                continue
            out.append(dict(a))
        return out

    def fluid_rheology(self, fluid_id: Optional[str] = None
                       ) -> List[FluidRheologyPoint]:
        """Fluid rheology points (with joined Fann data), by temperature.

        Rows carrying no rheology numbers are skipped (an incomplete catalogue
        row is not a fluid definition). ``fluid_id`` filters to one fluid.
        Requires ``with_geopressure=True``."""
        self._require_geopressure()
        out = []
        for key, pt in self._fluid_temp.items():
            if fluid_id is not None and pt.fluid_id != fluid_id:
                continue
            if all(v is None for v in (pt.plastic_viscosity, pt.yield_point,
                                       pt.n_prime, pt.k_prime)):
                continue  # no rheology -> not a definition
            pt.fann = self._fann.get(key, [])
            out.append(pt)
        return sorted(out, key=lambda p: (p.fluid_id, p.temperature or 0.0))

    def datum_set(self, well) -> List[Dict[str, str]]:
        """The full per-rig, dated datum realisations for a well (do NOT
        collapse to one RT -- resolve by a measurement's date/rig)."""
        well_id = well if well in self.wells else None
        if well_id is None:
            for wid, w in self.wells.items():
                if w.get("well_common_name") == well:
                    well_id = wid
                    break
        return self.datums.get(well_id or well, [])

    def survey_headers(
        self,
        wellbore=None,
        kind: Optional[str] = "definitive",
        phase: Optional[str] = None,
    ) -> List[SurveyHeader]:
        """Survey headers, filtered by wellbore / kind / phase.

        For a single ``wellbore`` (id or common name) the result is sorted by
        station count (descending), so the first entry is the sensible default
        selection. Pass ``wellbore=None`` for EVERY wellbore's headers (grouped
        by wellbore); ``kind=None`` for both raw ``CD_SURVEY_HEADER`` (incl.
        tool-less pilot/planned) AND definitive headers. This is the complete
        raw header set -- a superset of what :meth:`survey_runs` projects to
        (the tool-run view drops headers with no tool interval).
        """
        wb_id = (self._resolve_wellbore(wellbore).wellbore_id
                 if wellbore is not None else None)
        out = [
            h for h in self._all_headers     # complete set, not the lossy dict
            if (wb_id is None or h.wellbore_id == wb_id)
            and (kind is None or h.kind == kind)
            and (phase is None or h.phase == phase)
        ]
        # single wellbore: station-count desc (unchanged); all wellbores: group
        # by wellbore then station-count desc.
        out.sort(key=lambda h: (h.wellbore_id if wb_id is None else "",
                                -h.n_stations))
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


def open_edm(path: str, source_units: str = "feet",
             with_geopressure: bool = False,
             with_assemblies: bool = False,
             with_load_cases: bool = False,
             with_tool_terms: bool = False) -> EDMReader:
    """Convenience factory mirroring :meth:`EDMReader.open`."""
    return EDMReader.open(path, source_units=source_units,
                          with_geopressure=with_geopressure,
                          with_assemblies=with_assemblies,
                          with_load_cases=with_load_cases,
                          with_tool_terms=with_tool_terms)
