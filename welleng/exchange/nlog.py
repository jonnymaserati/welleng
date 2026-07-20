"""Client for the NLOG (Dutch subsurface data portal) borehole REST API.

NLOG's map viewer at ``nlog.nl/nlog-mapviewer`` is a single-page app
backed by an undocumented but stable REST API. This module wraps it so
that Dutch well data — trajectories, document indexes, log inventories
— can be pulled programmatically instead of clicked through, and so
that unmeasured (assumed / back-filled) azimuth columns can be
detected automatically before an error model is attached to them.

Endpoint shape, determined by reading the app bundle
(``main-*.js``, ``URL_PREFIX = "/nlog-mapviewer"``):

    POST /nlog-mapviewer/rest/brh/<resource>
    body: the bare integer borehole id (NOT JSON-wrapped)
    resources: details, dirsurveys, documents, logdocuments,
               measurements, coreruns, photos

    GET  /brh-web/rest/brh/document/<bfileDbk>       (scanned reports)
    GET  /brh-web/rest/brh/logdocument/<bfileDbk>    (LIS/LAS/DLIS/TXT)

The borehole id is the number in a map-viewer URL, e.g.
``nlog.nl/nlog-mapviewer/brh/106523583`` -> ``106523583``.

Bulk alternative: NLOG publishes a monthly zip of all directional
surveys (``thematische_data_boringen.zip``); prefer it for
whole-database work and this API for per-well detail and documents.

No authentication. Be polite — the portal is a public service.

Licence note: NLOG data is published by TNO on behalf of the Dutch
state under its own terms; check them before redistributing.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

BASE = "https://www.nlog.nl"
API = f"{BASE}/nlog-mapviewer/rest/brh"
FILES = f"{BASE}/brh-web/rest/brh"

Resource = Literal[
    "details", "dirsurveys", "documents", "logdocuments",
    "measurements", "coreruns", "photos",
]

# Provenance verdicts for a directional survey's azimuth column.
AzimuthProvenance = Literal[
    "measured",          # azimuth varies — a real directional survey
    "all_zero_vertical", # all zero AND effectively vertical: harmless
    "all_zero_deviated", # all zero but the well has angle: FABRICATED
    "constant_assumed",  # one constant non-zero value: assumed direction
    "single_bearing",    # all deflection on ONE bearing: fabricated, any azimuth
    "no_survey",         # <=2 stations: a stub, not a survey
]


class NLOGError(RuntimeError):
    pass


@dataclass(frozen=True)
class DirSurvey:
    """One directional survey for a borehole, in SI units (m, degrees)."""

    borehole_name: str
    md: list[float]
    inc: list[float]
    azi: list[float]
    tvd: list[float]
    dx: list[float]
    dy: list[float]
    north_ref: str | None          # 'G' grid, 'T' true, 'M' magnetic
    coord_system: str | None
    proc_method: str | None        # 'MC' = minimum curvature
    convergence: float | None
    declination: float | None
    proc_date_ms: int | None
    remark: str | None

    @property
    def n_stations(self) -> int:
        return len(self.md)

    @property
    def max_inclination(self) -> float:
        return max(self.inc) if self.inc else 0.0

    def azimuth_provenance(self, inc_threshold: float = 0.5) -> AzimuthProvenance:
        """Classify whether the azimuth column is a measurement.

        The distinction matters because an ISCWSA tool model applied to
        a fabricated azimuth reports a confident position that was
        never surveyed.
        """
        if self.n_stations <= 2:
            return "no_survey"
        distinct = {round(a, 3) for a in self.azi}
        if distinct == {0.0}:
            return (
                "all_zero_deviated"
                if self.max_inclination > inc_threshold
                else "all_zero_vertical"
            )
        if len(distinct) == 1 and self.max_inclination > inc_threshold:
            return "constant_assumed"
        # Strongest detector: the whole deflection projected onto ONE
        # bearing. A real well never holds a perpendicular offset at
        # exactly zero across many stations. Catches back-filled
        # non-zero bearings that the azimuth tests above miss — e.g.
        # NLOG A12-01, whose 1971 TOTCO report set azimuth to the
        # TARGET DIRECTION and reported X deflection as 0.00 at all 21
        # stations while Y reached 88.85 m.
        if self._single_bearing() and self.max_inclination > inc_threshold:
            return "single_bearing"
        return "measured"

    def _single_bearing(self, min_stations: int = 4) -> bool:
        """True if one horizontal offset component is identically zero
        while the other moves — the signature of deflection projected
        onto an assumed bearing rather than surveyed."""
        dx = [v for v in self.dx if v is not None]
        dy = [v for v in self.dy if v is not None]
        if len(dx) < min_stations or len(dy) < min_stations:
            return False
        dx_flat = {round(v, 3) for v in dx} == {0.0}
        dy_flat = {round(v, 3) for v in dy} == {0.0}
        moved = max(max(abs(v) for v in dy), max(abs(v) for v in dx)) > 1.0
        return (dx_flat ^ dy_flat) and moved

    def lateral_displacement(self) -> float:
        """Magnitude of lateral displacement implied by the inclination
        record — the radius of the annulus when azimuth is unknown."""
        import math

        total = 0.0
        for i in range(1, len(self.md)):
            d_md = self.md[i] - self.md[i - 1]
            mean_inc = 0.5 * (self.inc[i] + self.inc[i - 1])
            total += d_md * math.sin(math.radians(mean_inc))
        return total

    def to_welleng(self, error_model: str | None = None, **header_kwargs):
        """Build a ``welleng.survey.Survey``. Raises if the azimuth is
        not a measurement and an error model was requested, because the
        result would be a confident position derived from a
        placeholder — pass ``error_model=None`` to build geometry only,
        or set ``force=True`` in ``header_kwargs`` to override."""
        import numpy as np
        import welleng as we

        force = header_kwargs.pop("force", False)
        prov = self.azimuth_provenance()
        if error_model and prov != "measured" and not force:
            raise NLOGError(
                f"{self.borehole_name}: azimuth provenance is {prov!r} — "
                "applying an error model would report a confident position "
                "derived from an unmeasured azimuth. Pass force=True to "
                "override, or error_model=None for geometry only."
            )
        header = we.survey.SurveyHeader(
            name=self.borehole_name,
            azi_reference="grid" if (self.north_ref or "G") == "G" else "true",
            **header_kwargs,
        )
        return we.survey.Survey(
            md=np.asarray(self.md), inc=np.asarray(self.inc),
            azi=np.asarray(self.azi), deg=True, header=header,
            error_model=error_model,
        )


class NLOGClient:
    """Minimal client for the NLOG borehole API."""

    def __init__(self, timeout: float = 30.0, user_agent: str = "welleng-nlog/1.0"):
        self.timeout = timeout
        self.user_agent = user_agent

    def _post(self, resource: Resource, borehole_id: int) -> Any:
        req = urllib.request.Request(
            f"{API}/{resource}",
            data=str(int(borehole_id)).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": self.user_agent},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001 - surface the endpoint context
            raise NLOGError(f"{resource} failed for {borehole_id}: {exc}") from exc

    # -- resources ----------------------------------------------------
    def details(self, borehole_id: int) -> dict:
        """Borehole metadata. NOTE the API quirk: unlike the other
        resources this one takes a JSON *array of string ids*, not a
        bare integer."""
        req = urllib.request.Request(
            f"{API}/details",
            data=json.dumps([str(int(borehole_id))]).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": self.user_agent},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
        except Exception as exc:  # noqa: BLE001
            raise NLOGError(f"details failed for {borehole_id}: {exc}") from exc
        return out[0] if isinstance(out, list) and out else out

    def suggest(self, query: str) -> list[dict]:
        """Name search -> [{objectId, title, xcoordinate, ycoordinate}].

        This is the bridge between the bulk CSV (which keys on
        WELLBORE / NITG_NR / UWI and carries no borehole id) and the
        API (which keys on the numeric id in the map-viewer URL).
        """
        url = (f"{BASE}/nlog-mapviewer/rest/search/suggest/brh/"
               f"{urllib.parse.quote(query)}")
        req = urllib.request.Request(
            url, headers={"Accept": "application/json",
                          "User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001
            raise NLOGError(f"suggest failed for {query!r}: {exc}") from exc

    def id_for_name(self, wellbore_name: str) -> int | None:
        """Resolve a bulk-dump WELLBORE name to a borehole id, exact
        match on title. Returns None if the portal does not know it."""
        target = wellbore_name.strip().upper()
        for hit in self.suggest(wellbore_name):
            if str(hit.get("title", "")).strip().upper() == target:
                return int(hit["objectId"])
        return None

    def documents(self, borehole_id: int) -> list[dict]:
        d = self._post("documents", borehole_id)
        return d if isinstance(d, list) else d.get("documents", [])

    def log_documents(self, borehole_id: int) -> list[dict]:
        d = self._post("logdocuments", borehole_id)
        return d if isinstance(d, list) else d.get("logDocuments", [])

    def dir_surveys(self, borehole_id: int) -> list[DirSurvey]:
        d = self._post("dirsurveys", borehole_id)
        name = d.get("boreholeName") or str(borehole_id)
        out = []
        for s in d.get("dirSurveys") or []:
            pts = s.get("dirSurveyPoints") or []
            out.append(DirSurvey(
                borehole_name=name,
                md=[p["ahDepth"] for p in pts],
                inc=[p["devAngle"] for p in pts],
                azi=[p["azimuth"] for p in pts],
                tvd=[p.get("tvDepth") for p in pts],
                dx=[p.get("dx") for p in pts],
                dy=[p.get("dy") for p in pts],
                north_ref=s.get("northRefCode"),
                coord_system=s.get("coordSystemCode"),
                proc_method=s.get("procMethodCode"),
                convergence=s.get("convergenceCorr"),
                declination=s.get("declinationCorr"),
                proc_date_ms=s.get("procDate"),
                remark=s.get("remark"),
            ))
        return out

    # -- file downloads ----------------------------------------------
    def fetch_document(self, bfile_dbk: int | str, *, log: bool = False) -> bytes:
        """Scanned report (``log=False``) or log file (``log=True``)."""
        url = f"{FILES}/{'logdocument' if log else 'document'}/{bfile_dbk}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    # -- convenience --------------------------------------------------
    def survey_documents(
        self, borehole_id: int, pattern: str = r"dip|survey|devi|direction"
    ) -> list[dict]:
        """Documents whose title suggests a directional or dipmeter
        record — used to check whether an unmeasured azimuth could be
        recovered from an archived report."""
        import re

        rx = re.compile(pattern, re.I)
        return [d for d in self.documents(borehole_id)
                if rx.search(json.dumps(d))]


def audit_borehole(borehole_id: int, client: NLOGClient | None = None) -> dict:
    """One-shot provenance report for a borehole.

    Returns the survey classification plus whether any archived
    document looks like it might contain the missing azimuth.
    """
    c = client or NLOGClient()
    surveys = c.dir_surveys(borehole_id)
    docs = c.survey_documents(borehole_id)
    return {
        "borehole_id": borehole_id,
        "name": surveys[0].borehole_name if surveys else None,
        "surveys": [
            {
                "n_stations": s.n_stations,
                "max_inclination": s.max_inclination,
                "provenance": s.azimuth_provenance(),
                "lateral_displacement_m": round(s.lateral_displacement(), 1),
                "remark": s.remark,
                "north_ref": s.north_ref,
            }
            for s in surveys
        ],
        "candidate_azimuth_documents": [
            str(d.get("fullTitle") or d.get("title") or d) for d in docs
        ],
    }


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import sys

    for arg in sys.argv[1:] or ["106523583"]:
        print(json.dumps(audit_borehole(int(arg)), indent=2))
