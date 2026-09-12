"""Typed views of the NLOG payloads, and hints for finding a document.

Why these exist, and what they are NOT
--------------------------------------
The NLOG REST API is **reverse-engineered**. These models are an *annotation
layer*: they name the fields, attach the caveats to the fields they apply to,
and hand back everything they did not anticipate. They are **not** a schema we
can enforce, because we do not have NLOG's schema.

⭐ **The caveats are the point, not the types.** Every NLOG defect this library
has hit was SEMANTIC, not structural, and typing would have caught none of
them: ``coordSystemCode`` is per-well so one well's x is 523292 m and its
neighbour's is 3.34 DEGREES; ``drpDatumCode`` says MSL while
``depthRefPointDescription`` says Rotary Table on the same well; the production
rows carry no units at all. A reader of a docstring misses those. A reader of
the field they are about does not.

Three rules these models follow
-------------------------------
1. ``extra="allow"`` everywhere. A reverse-engineered reader must never DROP a
   field it did not anticipate -- NLOG can add one tomorrow, and silently
   losing it is worse than not modelling it.
2. Optional unless confirmed always present. Declaring a field required is a
   claim about NLOG's schema that we cannot make.
3. **A field whose meaning is not in the payload says so in its own
   description.** Typing ``quantity_oil: float`` reads as "we know what this
   is"; read two ways those rows give a GOR of 77 or 85,000.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ASSET_TYPES",
    "DOCUMENT_KINDS",
    "BoreholeSummary",
    "DocumentHint",
    "DocumentRecord",
    "LogFileRecord",
    "NLOGModel",
    "SuggestHit",
    "classify_document",
]


class NLOGModel(BaseModel):
    """Base: keep unknown fields, never coerce silently."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @property
    def unmodelled(self) -> Dict[str, Any]:
        """Fields NLOG sent that this model does not name.

        Non-empty is not an error -- it is the honest state of a
        reverse-engineered reader, and worth looking at when something is
        missing from a result.
        """
        return dict(self.model_extra or {})


# --------------------------------------------------------------------------- #
# asset types and document kinds
# --------------------------------------------------------------------------- #
#: NLOG's coarse asset bucket. ⚠️ **The code does NOT determine the content.**
#: "Final Well Report" is served under BOTH ``AERA`` and ``ADPB`` on the same
#: well, so the code narrows the search and the TITLE identifies the document.
ASSET_TYPES: Dict[str, str] = {
    "AERA": "report (end-of-well, final well, geological)",
    "ADBM": "borehole measurement (log runs, MWD, drilling parameters)",
    "ADPB": "plot / composite (composite log, report plates)",
}

#: ``kind -> (what it is, what such a document USUALLY carries)``.
#:
#: ⚠️ The second half is a **hint about the genre**, not a claim about the file
#: in your hand. A report that should contain a casing tally sometimes does
#: not, and one that should not sometimes does. Use it to decide what to open
#: FIRST; confirm by reading.
DOCUMENT_KINDS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "end_of_well_report": (
        "Operator's end-of-well report, written at TD",
        ("daily operations summary", "bit records", "mud programme",
         "casing and cement record", "deviation survey", "formation tops",
         "LOT / FIT results", "problems and NPT"),
    ),
    "final_well_report": (
        "The consolidated final report, usually later and fuller than the EOWR",
        ("as-built casing and cement record", "completion or abandonment detail",
         "final deviation survey", "formation tops", "log inventory",
         "pressure test results", "well schematic"),
    ),
    "geological_final_well_report": (
        "The geological volume of the final report",
        ("lithological description", "formation tops and stratigraphy",
         "cuttings and core description", "shows and fluid indications",
         "petrophysical interpretation", "correlation panels"),
    ),
    "mwd_report": (
        "Contractor MWD/LWD end-of-well report",
        ("survey listing", "tool configuration and runs",
         "gamma and resistivity while drilling", "tool failures"),
    ),
    "composite_log": (
        "Composite log plot over the whole well",
        ("gamma", "resistivity", "density / neutron porosity",
         "lithology column", "formation tops", "casing shoes"),
    ),
    "log_run": (
        "A single wireline or memory logging run",
        ("the run's curves over its own depth interval",),
    ),
    "drilling_parameters": (
        "Drilling parameters log",
        ("WOB", "RPM", "torque", "ROP", "flow", "standpipe pressure",
         "mud weight in/out"),
    ),
    "core_report": (
        "Core or sidewall-core analysis",
        ("core description", "porosity and permeability",
         "grain density", "core photographs"),
    ),
    "unknown": (
        "Not recognised from the title",
        (),
    ),
}

#: Title patterns, most specific FIRST -- "Geological Final Well Report" must
#: not be taken as a plain final well report, and an "MWD End of Well Report"
#: is a contractor document rather than the operator's EOWR.
_TITLE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bgeolog\w*\s+final\s+well\s+report\b", "geological_final_well_report"),
    (r"\bmwd\b.*\bend\s+of\s+well\b", "mwd_report"),
    (r"\blwd\b.*\bend\s+of\s+well\b", "mwd_report"),
    (r"\bend\s+of\s+well\s+report\b", "end_of_well_report"),
    (r"\bfinal\s+well\s+report\b", "final_well_report"),
    (r"\bcomposite\s+log\b", "composite_log"),
    (r"\bdrilling\s+parameter", "drilling_parameters"),
    (r"\bcore\b", "core_report"),
    (r"\blog\s*run\b", "log_run"),
)


class DocumentHint(NLOGModel):
    """What a document probably is, and what it probably holds."""

    kind: str = Field(description="A key of DOCUMENT_KINDS, or 'unknown'.")
    description: str = Field(description="What that kind of document is.")
    likely_contains: Tuple[str, ...] = Field(
        default=(),
        description="What such a document USUALLY carries -- a genre hint for "
                    "deciding what to open first, NOT a claim about this file.",
    )
    matched_on: Optional[str] = Field(
        None,
        description="The title pattern that matched. None means the kind came "
                    "from the asset type alone, or not at all.",
    )
    confident: bool = Field(
        False,
        description="True only when the TITLE identified it. False means the "
                    "asset type was the only evidence, and the asset type does "
                    "not determine content -- 'Final Well Report' is served "
                    "under both AERA and ADPB.",
    )


def classify_document(title: Optional[str],
                      asset_type: Optional[str] = None) -> DocumentHint:
    """Best guess at what a document is, with how much to trust the guess.

    Resolution is by TITLE. The asset type is a bucket, not a classification:
    the same report appears under two codes on the same well, so it can narrow
    a search and cannot identify a document.

    ``kind="unknown"`` with ``confident=False`` is a real answer and is
    returned rather than a plausible one.
    """
    text = (title or "").lower()
    for pattern, kind in _TITLE_PATTERNS:
        if re.search(pattern, text):
            desc, contains = DOCUMENT_KINDS[kind]
            return DocumentHint(kind=kind, description=desc,
                                likely_contains=contains,
                                matched_on=pattern, confident=True)
    # Nothing in the title. The asset type can say what FAMILY it is at most.
    if asset_type in ASSET_TYPES:
        return DocumentHint(
            kind="unknown",
            description=f"{ASSET_TYPES[asset_type]} -- title did not identify it",
            confident=False,
        )
    desc, contains = DOCUMENT_KINDS["unknown"]
    return DocumentHint(kind="unknown", description=desc, confident=False)


# --------------------------------------------------------------------------- #
# payload models
# --------------------------------------------------------------------------- #
class SuggestHit(NLOGModel):
    """One name-search hit. ``object_id`` is the borehole id every call needs."""

    object_id: Optional[str] = Field(None, alias="objectId")
    title: Optional[str] = Field(
        None, description="NLOG's CANONICAL well name. Hyphenation is exact: "
                          "'P11-A-02A' resolves and 'P11-A02A' returns nothing.")
    xcoordinate: Optional[float] = Field(
        None, description="Map x. ⚠️ Projection is NOT in this payload.")
    ycoordinate: Optional[float] = Field(
        None, description="Map y. ⚠️ Projection is NOT in this payload.")

    @property
    def borehole_id(self) -> Optional[int]:
        return int(self.object_id) if self.object_id is not None else None


class DocumentRecord(NLOGModel):
    """One document in a borehole's file list."""

    bfile_dbk: Optional[int] = Field(None, alias="assetBfileDbk",
                                     description="Pass to fetch_document().")
    full_title: Optional[str] = Field(None, alias="fullTitle")
    bar_code_title: Optional[str] = Field(None, alias="barCodeTitle")
    asset_type: Optional[str] = Field(
        None, alias="assetTypeCode",
        description="AERA / ADBM / ADPB -- see ASSET_TYPES. ⚠️ A BUCKET, not "
                    "a classification: the same report appears under two.")
    file_type: Optional[str] = Field(None, alias="fileTypeCode",
                                     description="PDF / ZIP / TIF / LAS ...")
    file_size: Optional[int] = Field(None, alias="fileSize")
    has_file: Optional[bool] = Field(None, alias="hasBfile")
    lost: Optional[bool] = Field(
        None, description="NLOG's own marker that the file is missing. A "
                          "catalogued document is not a retrievable one.")

    @property
    def title(self) -> str:
        return self.full_title or self.bar_code_title or ""

    def hint(self) -> DocumentHint:
        """What this document probably is. See :func:`classify_document`."""
        return classify_document(self.title, self.asset_type)


class LogFileRecord(NLOGModel):
    """One log file in the log inventory -- LAS, LIS, DLIS, ASCII.

    ⚠️ **The inventory says which DEPTHS a file covers, never which CURVES.**
    A curve list needs the file itself; see
    :meth:`welleng.exchange.nlog.NLOGClient.find_log_curves`, which fetches the
    candidates and resolves their mnemonics rather than guessing from a name.
    """

    bfile_dbk: Optional[int] = Field(None, alias="documentBfileDbk",
                                     description="Pass to fetch_document(log=True).")
    file_name: Optional[str] = Field(None, alias="fileName")
    file_type: Optional[str] = Field(None, alias="fileTypeCode")
    top_depth: Optional[float] = Field(
        None, alias="topDepth",
        description="⚠️ Depth UNIT and REFERENCE are not in this payload. NLOG "
                    "is metric, but the file's own index may be feet or, on "
                    "some runs, seconds -- see LasFile.index_is_depth.")
    bottom_depth: Optional[float] = Field(None, alias="bottomDepth")
    description_code: Optional[str] = Field(
        None, alias="descriptionCode",
        description="CMP = composite, LOGRUN = a single run.")
    result_code: Optional[str] = Field(None, alias="resultCode")
    document_group: Optional[str] = Field(None, alias="documentGroup")

    def covers(self, depth: float) -> bool:
        """Whether this file's stated interval contains ``depth``.

        ``False`` when either bound is missing: an unstated interval is not a
        covering one.
        """
        if self.top_depth is None or self.bottom_depth is None:
            return False
        lo, hi = sorted((self.top_depth, self.bottom_depth))
        return lo <= depth <= hi


class BoreholeSummary(NLOGModel):
    """One row of the borehole catalogue."""

    borehole_id: Optional[int] = Field(None, alias="boreholeDbk")
    name: Optional[str] = Field(None, alias="boreholeName")
    status: Optional[str] = Field(None, alias="statusDescription")
    purpose: Optional[str] = Field(None, alias="purposeCd")
    result: Optional[str] = Field(None, alias="resultCode")
    on_offshore: Optional[str] = Field(None, alias="onOffshore")
    confidentiality_date: Optional[str] = Field(
        None, alias="confidentialityDate",
        description="A well past this date is public. ⚠️ Absence here is NOT "
                    "evidence that a well is unrestricted.")


def as_models(rows: List[dict], model) -> List[Any]:
    """Parse a list of payload rows, keeping every unmodelled field."""
    return [model.model_validate(r) for r in rows]
