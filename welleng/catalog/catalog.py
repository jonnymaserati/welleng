"""API 5CT casing + tubing dimensional catalogue: loader + resolver.

The JSON data files (``data/casing.json``, ``data/tubing.json``) carry the
as-published imperial dimensional facts (cited to API Spec 5CT) in the same
``_meta`` + ``grades`` + rows layout as the companion drilling-mechanics
catalogues (``drillpipe.json`` et al.). This module parses them, indexes the
rows by ``(od_in, nominal_weight_ppf)`` and resolves a tubular's derived
dimensions (ID, wall, drift) plus - when a grade is given - its minimum yield.

SI conversion is applied per each file's ``_meta.to_SI`` factors: every
resolved spec carries both the as-published imperial values *and* their SI
counterparts (``*_m`` / ``*_pa``), so imperial-in-inches consumers (the
``welleng.schematic`` models) and SI-pure consumers are both served.

OSDU alignment (``TubularComponent.1.0.0``)::

    od_in              <-> MaximumOuterDiameter / TubularComponentNominalSize
    id_in              <-> InnerDiameter
    drift_in           <-> DriftDiameter
    nominal_weight_ppf <-> TubularComponentNominalWeight
    grade              <-> TubularComponentTubingGradeID
    yield_psi          <-> TubularComponentTubingGradeStrength
    (type)             <-> TubularComponentTypeID

Wall thickness is derived ((OD-ID)/2) and is not stored in OSDU.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DATA_DIR = Path(__file__).with_name("data")

# --- API TR 5C3 (7th ed., 2018) performance-property constants -------------
# Wall-tolerance factor for the historical API design equations: the -12.5%
# manufacturing tolerance on wall thickness gives k_wall = 0.875
# (API TR 5C3, 6.6.2.2 Eq. 10 / 8; = 87.5% minimum wall).
KWALL = 0.875
MIN_WALL_PCT = 87.5

# Empirical collapse-equation factors (Ac, Bc, Cc, Fc, Gc) per grade, read
# from API TR 5C3 Table 6 (plastic) + Table 7 (transition). Grades sharing a
# specified minimum yield share a factor row (e.g. L80/N80 -> "L-N-80").
# Cc is in psi. Only the grades carried by the catalogue are tabulated here;
# an absent grade yields a None collapse rating (flagged, never guessed).
_COLLAPSE_FACTORS: Dict[str, Tuple[float, float, float, float, float]] = {
    # grade: (Ac, Bc, Cc, Fc, Gc)
    "J55": (2.991, 0.0541, 1206.0, 1.989, 0.0360),
    "K55": (2.991, 0.0541, 1206.0, 1.989, 0.0360),
    "L80": (3.071, 0.0667, 1955.0, 1.998, 0.0434),
    "N80": (3.071, 0.0667, 1955.0, 1.998, 0.0434),
    "P110": (3.181, 0.0819, 2852.0, 2.066, 0.0532),
}


def plain_end_weight_ppf(od_in: float, wall_in: float) -> float:
    """Plain-end nominal mass (lb/ft) via the API formula ``10.69*(D-t)*t``.

    Historical API plain-end mass equation (API Spec 5CT / TR 5C3): geometry
    only, grade-independent. Rounded to 2 dp.
    """
    return round(10.69 * (od_in - wall_in) * wall_in, 2)


def pipe_body_yield_klb(od_in: float, id_in: float, min_yield_psi: float) -> float:
    """Pipe-body yield strength (klb), API TR 5C3 Eq. (11): ``fymn * Ap``.

    ``Ap = (pi/4)(D^2 - d^2)`` is the pipe cross-sectional area. Rounded to
    the nearest klb (API tabulation convention).
    """
    area = (math.pi / 4.0) * (od_in ** 2 - id_in ** 2)
    return round(min_yield_psi * area / 1000.0)


def internal_yield_pressure_psi(
    od_in: float, wall_in: float, min_yield_psi: float
) -> float:
    """Minimum internal yield pressure (psi), API TR 5C3 Eq. (10) - Barlow.

    ``piYAPI = 2*fymn*(k_wall*t)/D`` with ``k_wall = 0.875``. Rounded to the
    nearest 10 psi (API tabulation convention).
    """
    p = 2.0 * min_yield_psi * (KWALL * wall_in) / od_in
    return round(p / 10.0) * 10.0


def collapse_pressure_psi(
    od_in: float, wall_in: float, min_yield_psi: float, grade: str
) -> Optional[float]:
    """Collapse resistance (psi), API TR 5C3 8.4 four-regime design equations.

    Selects yield / plastic / transition / elastic regime by the D/t ratio
    against the boundaries (D/t)yp Eq.(36), (D/t)pt Eq.(38), (D/t)te Eq.(40),
    then evaluates Eq.(35)/(37)/(39)/(41). Returns ``None`` (flagged, not
    guessed) if the grade's empirical factors are not tabulated. Rounded to
    the nearest 10 psi.
    """
    factors = _COLLAPSE_FACTORS.get(grade)
    if factors is None:
        return None
    ac, bc, cc, fc, gc = factors
    fy = float(min_yield_psi)
    dt = od_in / wall_in
    # regime boundaries (Eqs 36, 38, 40)
    dt_yp = (
        math.sqrt((ac - 2.0) ** 2 + 8.0 * (bc + cc / fy)) + (ac - 2.0)
    ) / (2.0 * (bc + cc / fy))
    dt_pt = (fy * (ac - fc)) / (cc + fy * (bc - gc))
    dt_te = (2.0 + bc / ac) / (3.0 * (bc / ac))
    if dt <= dt_yp:  # yield-strength collapse, Eq. (35)
        p = 2.0 * fy * (dt - 1.0) / (dt ** 2)
    elif dt <= dt_pt:  # plastic collapse, Eq. (37)
        p = fy * (ac / dt - bc) - cc
    elif dt <= dt_te:  # transition collapse, Eq. (39)
        p = fy * (fc / dt - gc)
    else:  # elastic collapse, Eq. (41)
        p = 46.95e6 / (dt * (dt - 1.0) ** 2)
    return round(p / 10.0) * 10.0

# OSDU TubularComponent.1.0.0 alias map (welleng field -> OSDU property).
OSDU_ALIASES: Dict[str, str] = {
    "od_in": "MaximumOuterDiameter",
    "id_in": "InnerDiameter",
    "drift_in": "DriftDiameter",
    "nominal_weight_ppf": "TubularComponentNominalWeight",
    "grade": "TubularComponentTubingGradeID",
    "yield_psi": "TubularComponentTubingGradeStrength",
    "type": "TubularComponentTypeID",
}

# The row-array key inside each JSON document.
_ROW_KEY = {"casing": "casing", "tubing": "tubing"}


class CatalogError(KeyError):
    """Raised when a requested tubular is not in the catalogue."""


@dataclass(frozen=True)
class TubularSpec:
    """A resolved tubular: as-published imperial values + SI counterparts.

    Imperial fields (``*_in``, ``yield_psi``) are the API 5CT facts; the SI
    fields (``*_m``, ``yield_pa``) are converted via the file's ``to_SI``.
    """

    kind: str
    od_in: float
    nominal_weight_ppf: float
    wall_in: float
    id_in: float
    drift_in: float
    grade: Optional[str] = None
    yield_psi: Optional[float] = None
    # --- body performance (API TR 5C3): geometry-only + grade-dependent ---
    plain_end_weight_ppf: float = 0.0
    min_wall_pct: float = MIN_WALL_PCT
    max_yield_psi: Optional[float] = None
    min_uts_psi: Optional[float] = None
    pipe_body_yield_klb: Optional[float] = None
    internal_yield_pressure_psi: Optional[float] = None
    collapse_pressure_psi: Optional[float] = None
    # --- SI ---
    od_m: float = 0.0
    wall_m: float = 0.0
    id_m: float = 0.0
    drift_m: float = 0.0
    yield_pa: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


class Catalog:
    """A loaded, indexed dimensional catalogue for one ``kind``."""

    def __init__(self, kind: str, doc: dict):
        self.kind = kind
        self._meta = doc["_meta"]
        self._to_si = self._meta["to_SI"]
        self.grades: Dict[str, dict] = doc.get("grades", {})
        rows = doc[_ROW_KEY[kind]]
        # index by (od, weight) rounded to avoid float-key mismatch.
        self._by_key: Dict[Tuple[float, float], dict] = {}
        for row in rows:
            self._by_key[self._key(row["od_in"], row["nominal_weight_ppf"])] = row

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, kind: str) -> "Catalog":
        if kind not in _ROW_KEY:
            raise ValueError(
                f"unknown kind {kind!r}; expected one of {sorted(_ROW_KEY)}"
            )
        path = _DATA_DIR / f"{kind}.json"
        doc = json.loads(path.read_text())
        return cls(kind, doc)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _key(od: float, weight: float) -> Tuple[float, float]:
        return (round(float(od), 3), round(float(weight), 2))

    def _grade_entry(self, grade: str) -> dict:
        try:
            return self.grades[grade]
        except KeyError:
            raise CatalogError(
                f"unknown grade {grade!r} for {self.kind}; "
                f"available grades: {sorted(self.grades)}"
            )

    def _suggest(self, od: float, weight: float) -> str:
        same_od = sorted(
            w for (o, w) in self._by_key if o == round(float(od), 3)
        )
        if same_od:
            nearest = min(same_od, key=lambda w: abs(w - weight))
            return (
                f"no {self.kind} {od} in x {weight} lb/ft; available weights for "
                f"OD {od} in: {same_od} (nearest: {nearest} lb/ft)"
            )
        ods = sorted({o for (o, _) in self._by_key})
        return (
            f"no {self.kind} with OD {od} in; available ODs: {ods}"
        )

    # -- resolution ---------------------------------------------------------
    def resolve(
        self,
        od_in: float,
        nominal_weight_ppf: float,
        grade: Optional[str] = None,
    ) -> TubularSpec:
        row = self._by_key.get(self._key(od_in, nominal_weight_ppf))
        if row is None:
            raise CatalogError(self._suggest(od_in, nominal_weight_ppf))

        in_to_m = self._to_si["in_to_m"]
        psi_to_pa = self._to_si["psi_to_pa"]

        od, wall, id_ = row["od_in"], row["wall_in"], row["id_in"]

        # geometry-only body property (grade-independent).
        pe_weight = plain_end_weight_ppf(od, wall)

        # grade-dependent tensile + performance (None until a grade is given).
        yield_psi = max_yield = min_uts = None
        body_yield = internal_yield = collapse = None
        if grade is not None:
            entry = self._grade_entry(grade)
            yield_psi = float(entry["min_yield_psi"])
            max_yield = entry.get("max_yield_psi")
            min_uts = entry.get("min_uts_psi")
            body_yield = pipe_body_yield_klb(od, id_, yield_psi)
            internal_yield = internal_yield_pressure_psi(od, wall, yield_psi)
            collapse = collapse_pressure_psi(od, wall, yield_psi, grade)
        yield_pa = yield_psi * psi_to_pa if yield_psi is not None else None

        return TubularSpec(
            kind=self.kind,
            od_in=od,
            nominal_weight_ppf=row["nominal_weight_ppf"],
            wall_in=wall,
            id_in=id_,
            drift_in=row["drift_in"],
            grade=grade,
            yield_psi=yield_psi,
            plain_end_weight_ppf=pe_weight,
            min_wall_pct=MIN_WALL_PCT,
            max_yield_psi=max_yield,
            min_uts_psi=min_uts,
            pipe_body_yield_klb=body_yield,
            internal_yield_pressure_psi=internal_yield,
            collapse_pressure_psi=collapse,
            od_m=od * in_to_m,
            wall_m=wall * in_to_m,
            id_m=id_ * in_to_m,
            drift_m=row["drift_in"] * in_to_m,
            yield_pa=yield_pa,
        )

    def list_sizes(self) -> List[Tuple[float, float]]:
        """Sorted ``(od_in, nominal_weight_ppf)`` pairs available."""
        return sorted(self._by_key)


@lru_cache(maxsize=None)
def _catalog(kind: str) -> Catalog:
    return Catalog.load(kind)


def resolve(
    od_in: float,
    nominal_weight_ppf: float,
    grade: Optional[str] = None,
    kind: str = "casing",
) -> TubularSpec:
    """Resolve a tubular's dimensions from (OD, weight[, grade]).

    Returns a :class:`TubularSpec` with ``id_in``, ``wall_in``, ``drift_in``
    (imperial + SI) and, when ``grade`` is given, ``yield_psi`` / ``yield_pa``.
    Raises :class:`CatalogError` with a nearest-weight suggestion on no match.
    """
    return _catalog(kind).resolve(od_in, nominal_weight_ppf, grade)


def list_sizes(kind: str = "casing") -> List[Tuple[float, float]]:
    """Available ``(od_in, nominal_weight_ppf)`` pairs for ``kind``."""
    return _catalog(kind).list_sizes()


def grades(kind: str = "casing") -> Dict[str, dict]:
    """Grade -> ``{'min_yield_psi': ...}`` table for ``kind``."""
    return dict(_catalog(kind).grades)


# ===========================================================================
# Couplings / connections (API Spec 5CT Tables E.27-E.30)
# ===========================================================================

# Connection -> catalogue kind (casing round/buttress vs tubing upsets).
_CONNECTION_KIND: Dict[str, str] = {
    "STC": "casing", "LTC": "casing", "BTC": "casing",
    "NUE": "tubing", "EUE": "tubing",
}


@dataclass(frozen=True)
class CouplingSpec:
    """A resolved API coupling: regular OD (W), special-clearance OD (Wc),
    and minimum coupling length (NL), imperial + SI (API Spec 5CT).
    """

    od_in: float
    connection: str
    kind: str
    coupling_od_in: float                         # W (regular)
    coupling_length_in: float                     # NL (minimum)
    special_clearance_od_in: Optional[float] = None  # Wc (if API-tabulated)
    # --- SI ---
    coupling_od_m: float = 0.0
    coupling_length_m: float = 0.0
    special_clearance_od_m: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


class CouplingCatalog:
    """Loaded, indexed API 5CT coupling catalogue keyed by ``(od, connection)``."""

    def __init__(self, doc: dict):
        self._meta = doc["_meta"]
        self._to_si = self._meta["to_SI"]
        self._by_key: Dict[Tuple[float, str], dict] = {}
        for row in doc["couplings"]:
            self._by_key[self._key(row["od_in"], row["connection"])] = row

    @classmethod
    def load(cls) -> "CouplingCatalog":
        doc = json.loads((_DATA_DIR / "couplings.json").read_text())
        return cls(doc)

    @staticmethod
    def _key(od: float, connection: str) -> Tuple[float, str]:
        return (round(float(od), 3), str(connection).upper())

    def _suggest(self, od: float, connection: str) -> str:
        connection = str(connection).upper()
        conns = sorted({c for (_, c) in self._by_key})
        if connection not in conns:
            return (
                f"unknown connection {connection!r}; available connections: "
                f"{conns}"
            )
        ods = sorted({o for (o, c) in self._by_key if c == connection})
        return (
            f"no {connection} coupling for OD {od} in; available ODs for "
            f"{connection}: {ods}"
        )

    def resolve(self, od_in: float, connection: str) -> CouplingSpec:
        row = self._by_key.get(self._key(od_in, connection))
        if row is None:
            raise CatalogError(self._suggest(od_in, connection))
        in_to_m = self._to_si["in_to_m"]
        scc = row.get("special_clearance_od_in")
        return CouplingSpec(
            od_in=row["od_in"],
            connection=row["connection"],
            kind=row["kind"],
            coupling_od_in=row["coupling_od_in"],
            coupling_length_in=row["coupling_length_in"],
            special_clearance_od_in=scc,
            coupling_od_m=row["coupling_od_in"] * in_to_m,
            coupling_length_m=row["coupling_length_in"] * in_to_m,
            special_clearance_od_m=(scc * in_to_m) if scc is not None else None,
        )

    def connections(self) -> List[str]:
        return sorted({c for (_, c) in self._by_key})


@lru_cache(maxsize=None)
def _coupling_catalog() -> CouplingCatalog:
    return CouplingCatalog.load()


def resolve_coupling(
    od_in: float, connection: str, kind: str = "casing"
) -> CouplingSpec:
    """Resolve API 5CT coupling dimensions from ``(od_in, connection)``.

    ``connection`` is one of STC/LTC/BTC (casing) or NUE/EUE (tubing).
    Returns a :class:`CouplingSpec` with regular OD (W), special-clearance OD
    (Wc, when API tabulates one), and minimum coupling length (NL), imperial +
    SI. ``kind`` is validated against the connection. Raises
    :class:`CatalogError` listing the available connections on no match.
    """
    conn = str(connection).upper()
    expected = _CONNECTION_KIND.get(conn)
    if expected is None:
        raise CatalogError(
            f"unknown connection {conn!r}; available connections: "
            f"{sorted(_CONNECTION_KIND)}"
        )
    if kind is not None and kind != expected:
        raise CatalogError(
            f"connection {conn!r} is a {expected} connection, not {kind!r}; "
            f"casing connections: ['BTC', 'LTC', 'STC'], "
            f"tubing connections: ['EUE', 'NUE']"
        )
    return _coupling_catalog().resolve(od_in, conn)


def coupling_connections() -> List[str]:
    """All coupling/connection designations in the catalogue."""
    return _coupling_catalog().connections()


# ===========================================================================
# Connection performance - API TR 5C3 (7th ed., 2018) Sec 9 + Sec 10.2
# ===========================================================================
# Tensile joint strength (Sec 9) + coupling internal yield pressure
# (Sec 10.2). ALL equations transcribed from the PDF equation IMAGES (not
# pdftotext, which silently drops superscripts - Eq. 54 carries D^-0.59, NOT
# "0.74D - 0.59"). Verified against the duplicate statement in Annex J.
#
# Thread geometry these equations need but API 5CT does NOT tabulate (perfect-
# thread length L7, engaged length Let, coupling-root diameter d1) comes from
# API Spec 5B. Only the buttress L7 is baked in below (turn-key, cited to API
# 5B Table 5); round-thread Let / d1 are user-supplied (from API 5B) so no
# unvalidated per-size thread data is fabricated here.
#
# Make-up torque: API TR 5C3 gives make-up torque for ROUND thread only
# (Sec 15). BUTTRESS make-up torque is NOT in 5C3 (see API RP 5C1 or the
# manufacturer) and is deliberately not implemented here.

# API Spec 5B (16th ed., 2017) Table 5 - buttress perfect-thread length L7
# (in.), keyed by pipe OD (in.). L7 is the one per-size value not derivable
# from a closed form; E7, IB, Td, hB below ARE closed-form/constant per 5B.
_BTC_L7_IN: Dict[float, float] = {
    4.5: 1.6535, 5.0: 1.7785, 5.5: 1.8410, 6.625: 2.0285, 7.0: 2.2160,
    7.625: 2.4035, 8.625: 2.5285, 9.625: 2.5285, 10.75: 2.5285,
    11.75: 2.5285, 13.375: 2.5285, 16.0: 3.1245, 18.625: 3.1245,
    20.0: 3.1245,
}


def _buttress_d1_in(od_in: float) -> float:
    """Buttress coupling-thread root diameter d1 (in.), API TR 5C3 Eq. (61).

    ``d1 = E7 - (L7 + IB) Td + hB`` with the API Spec 5B buttress geometry:
    E7 = D4 - 0.062 (pitch dia; D4 = D + 0.016 for OD <= 13-3/8 in., else D),
    L7 the perfect-thread length (5B Table 5, :data:`_BTC_L7_IN`), taper
    Td = 0.0625 in./in. (OD <= 13-3/8) or 0.0833 (larger), triangle-stamp
    offset IB = 0.400 (4-1/2), 0.500 (5 .. 13-3/8) or 0.375 (larger), and
    buttress thread height hB = 0.062 in. (USC). Raises if L7 is untabulated.
    """
    key = round(float(od_in), 3)
    l7 = _BTC_L7_IN.get(key)
    if l7 is None:
        raise CatalogError(
            f"no API 5B buttress L7 for OD {od_in} in; "
            f"available: {sorted(_BTC_L7_IN)}"
        )
    if key <= 13.375:
        d4 = od_in + 0.016
        td = 0.0625
        ib = 0.400 if key == 4.5 else 0.500
    else:
        d4 = od_in
        td = 0.0833
        ib = 0.375
    e7 = d4 - 0.062
    return e7 - (l7 + ib) * td + 0.062


def buttress_pipe_thread_strength_klb(
    od_in: float, id_in: float, min_yield_psi: float, min_uts_psi: float
) -> float:
    """Buttress pipe-thread tensile strength (klb), API TR 5C3 Eq. (59).

    ``Pj = 0.95 Ap fumnp [1.008 - 0.0396 (1.083 - fymnp/fumnp) D]`` with
    ``Ap = (pi/4)(D^2 - d^2)``, pipe-body min tensile ``fumnp`` and min yield
    ``fymnp``. One of the two buttress limit terms (see Eq. 60). Rounded to
    the nearest klb (API tabulation convention).
    """
    ap = (math.pi / 4.0) * (od_in ** 2 - id_in ** 2)
    factor = 1.008 - 0.0396 * (1.083 - min_yield_psi / min_uts_psi) * od_in
    return round(0.95 * ap * min_uts_psi * factor / 1000.0)


def buttress_coupling_thread_strength_klb(
    od_in: float, coupling_od_in: float, coupling_uts_psi: float
) -> float:
    """Buttress coupling-thread tensile strength (klb), API TR 5C3 Eq. (60).

    ``Pj = 0.95 Ajc fumnc`` with ``Ajc = (pi/4)(W^2 - d1^2)`` (Eq. 57), W the
    coupling OD, d1 from :func:`_buttress_d1_in` (Eq. 61) and coupling min
    tensile ``fumnc``. The other buttress limit term (see Eq. 59). Rounded to
    the nearest klb.
    """
    d1 = _buttress_d1_in(od_in)
    ajc = (math.pi / 4.0) * (coupling_od_in ** 2 - d1 ** 2)
    return round(0.95 * ajc * coupling_uts_psi / 1000.0)


def buttress_joint_strength_klb(
    od_in: float,
    id_in: float,
    min_yield_psi: float,
    min_uts_psi: float,
    coupling_od_in: Optional[float] = None,
    coupling_uts_psi: Optional[float] = None,
) -> float:
    """Buttress (BTC) tensile joint strength (klb), API TR 5C3 Sec 9.2.3.

    The LESSER of the pipe-thread strength (Eq. 59) and the coupling-thread
    strength (Eq. 60). ``coupling_od_in`` defaults to the API 5CT coupling OD
    (W) from the catalogue; ``coupling_uts_psi`` defaults to the pipe-body
    ``min_uts_psi`` (standard API couplings share the pipe grade). Rounded to
    the nearest klb.
    """
    pipe = buttress_pipe_thread_strength_klb(
        od_in, id_in, min_yield_psi, min_uts_psi
    )
    if coupling_od_in is None:
        coupling_od_in = resolve_coupling(od_in, "BTC").coupling_od_in
    if coupling_uts_psi is None:
        coupling_uts_psi = min_uts_psi
    coupling = buttress_coupling_thread_strength_klb(
        od_in, coupling_od_in, coupling_uts_psi
    )
    return float(min(pipe, coupling))


def buttress_coupling_internal_yield_psi(
    od_in: float, coupling_od_in: float, coupling_yield_psi: float
) -> float:
    """Buttress coupling internal yield pressure (psi), API TR 5C3 Eq. (65).

    ``piYc = fymnc (W - d1)/W`` with W the coupling OD, d1 from Eq. (67)
    (= Eq. 61 buttress form; see :func:`_buttress_d1_in`) and coupling min
    yield ``fymnc``. Per Sec 10.1 this limits connection internal pressure
    only when lower than the pipe-body internal yield. Rounded to the nearest
    10 psi (API tabulation convention).
    """
    d1 = _buttress_d1_in(od_in)
    p = coupling_yield_psi * (coupling_od_in - d1) / coupling_od_in
    return round(p / 10.0) * 10.0


def round_thread_pipe_fracture_strength_klb(
    od_in: float, id_in: float, min_uts_psi: float
) -> float:
    """Round-thread (STC/LTC) pipe fracture strength (klb), API TR 5C3 Eq.(53).

    ``Pj = 0.95 Ajp fumnp`` with the last-perfect-thread pipe area
    ``Ajp = (pi/4)[(D - 0.1425)^2 - d^2]`` (Eq. 56). ONE of the three round-
    thread limit terms; the joint strength is the least of this, the pull-out
    strength (Eq. 54) and coupling fracture (Eq. 55). Rounded to nearest klb.
    """
    ajp = (math.pi / 4.0) * ((od_in - 0.1425) ** 2 - id_in ** 2)
    return round(0.95 * ajp * min_uts_psi / 1000.0)


def round_thread_pullout_strength_klb(
    od_in: float,
    id_in: float,
    engaged_thread_length_in: float,
    min_yield_psi: float,
    min_uts_psi: float,
) -> float:
    """Round-thread (STC/LTC) pull-out strength (klb), API TR 5C3 Eq. (54).

    ``Pj = 0.95 Ajp Let [(0.74 D^-0.59 fumnp)/(0.5 Let + 0.14 D)
    + fymnp/(Let + 0.14 D)]`` with ``Ajp`` from Eq. (56) and ``Let`` the
    engaged thread length (= L4 - M, from API Spec 5B; user-supplied since
    API 5CT does not tabulate it). Note ``D^-0.59`` is an EXPONENT (pdftotext
    drops it). Pull-out governs for most standard round-thread sizes/grades
    (Annex J.2.2.3). Rounded to the nearest klb.
    """
    ajp = (math.pi / 4.0) * ((od_in - 0.1425) ** 2 - id_in ** 2)
    lt = engaged_thread_length_in
    bracket = (
        (0.74 * od_in ** -0.59 * min_uts_psi) / (0.5 * lt + 0.14 * od_in)
        + min_yield_psi / (lt + 0.14 * od_in)
    )
    return round(0.95 * ajp * lt * bracket / 1000.0)


def round_thread_joint_strength_klb(
    od_in: float,
    id_in: float,
    min_yield_psi: float,
    min_uts_psi: float,
    engaged_thread_length_in: Optional[float] = None,
    coupling_od_in: Optional[float] = None,
    coupling_uts_psi: Optional[float] = None,
    coupling_root_dia_in: Optional[float] = None,
) -> float:
    """Round-thread (STC/LTC) tensile joint strength (klb), API TR 5C3 Sec 9.2.2.

    The LEAST of the pipe fracture strength (Eq. 53), the pull-out strength
    (Eq. 54, if ``engaged_thread_length_in`` [Let] is supplied) and the
    coupling fracture strength (Eq. 55, if ``coupling_root_dia_in`` [d1, from
    API 5B Eq. 58] is supplied). Let and d1 come from API Spec 5B (not 5CT).

    WARNING: pull-out normally GOVERNS (Annex J.2.2.3); with Let omitted this
    returns the fracture-only term, which OVERSTATES the joint strength. Supply
    Let for a valid rating. Rounded to the nearest klb.
    """
    terms = [round_thread_pipe_fracture_strength_klb(od_in, id_in, min_uts_psi)]
    if engaged_thread_length_in is not None:
        terms.append(
            round_thread_pullout_strength_klb(
                od_in,
                id_in,
                engaged_thread_length_in,
                min_yield_psi,
                min_uts_psi,
            )
        )
    if coupling_root_dia_in is not None and coupling_od_in is not None:
        uts = min_uts_psi if coupling_uts_psi is None else coupling_uts_psi
        ajc = (math.pi / 4.0) * (coupling_od_in ** 2 - coupling_root_dia_in ** 2)
        terms.append(round(0.95 * ajc * uts / 1000.0))
    return float(min(terms))


# ===========================================================================
# ConnectionSpec - full connection performance schema (VAM-datasheet field set)
# ===========================================================================

@dataclass(frozen=True)
class ConnectionSpec:
    """Full connection performance record (VAM/Tenaris datasheet field set).

    For **API** connections (STC/LTC/BTC/NUE/EUE) only the *dimensional* fields
    are populated (from API Spec 5CT Tables E.27-E.30): ``connection_od_in`` =
    regular coupling OD (W) and ``coupling_length_in`` = minimum length (NL).

    All premium-performance fields (efficiencies, strengths, pressure
    resistances, make-up torques, delta-turn, bending) are **proprietary,
    vendor/user-supplied** for premium threads (VAM, Tenaris TenarisHydril,
    etc.) and are deliberately left ``None`` here - they are NOT vendored and
    must NOT be fabricated. Populate them from the user's connection datasheet.
    """

    # --- dimensional (API 5CT for API connections; datasheet for premium) ---
    connection_od_in: Optional[float] = None
    connection_id_in: Optional[float] = None
    makeup_loss_in: Optional[float] = None
    coupling_length_in: Optional[float] = None
    # --- structural efficiencies (% of pipe body) ---
    tension_eff_pct: Optional[float] = None
    compression_eff_pct: Optional[float] = None
    internal_pressure_eff_pct: Optional[float] = None
    external_pressure_eff_pct: Optional[float] = None
    # --- absolute ratings ---
    tension_strength_klb: Optional[float] = None
    compression_strength_klb: Optional[float] = None
    internal_pressure_resistance_psi: Optional[float] = None
    external_pressure_resistance_psi: Optional[float] = None
    # --- service limits ---
    max_bending_deg_per_100ft: Optional[float] = None
    max_load_coupling_face_klb: Optional[float] = None
    # --- make-up ---
    makeup_torque_min_ftlb: Optional[float] = None
    makeup_torque_opt_ftlb: Optional[float] = None
    makeup_torque_max_ftlb: Optional[float] = None
    shouldering_torque_min_ftlb: Optional[float] = None
    shouldering_torque_max_ftlb: Optional[float] = None
    delta_turn_min: Optional[float] = None
    delta_turn_max: Optional[float] = None
    # --- identity ---
    connection_type: Optional[str] = None
    grade: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_api(
        cls,
        od_in: float,
        connection: str,
        kind: str = "casing",
        grade: Optional[str] = None,
    ) -> "ConnectionSpec":
        """Build an API-connection spec: dimensional fields from 5CT only.

        Fills ``connection_od_in`` (regular coupling OD W),
        ``coupling_length_in`` (min length NL), ``connection_type`` and
        ``grade``; every premium-performance field is left ``None`` (proprietary,
        user-supplied - never fabricated). To also compute API TR 5C3 joint
        strength (which needs the pipe wall) use :meth:`from_tubular`.
        """
        cpl = resolve_coupling(od_in, connection, kind=kind)
        return cls(
            connection_od_in=cpl.coupling_od_in,
            coupling_length_in=cpl.coupling_length_in,
            connection_type=cpl.connection,
            grade=grade,
        )

    @classmethod
    def from_tubular(
        cls,
        spec: "TubularSpec",
        connection: str,
    ) -> "ConnectionSpec":
        """Build an API-connection spec from a resolved :class:`TubularSpec`.

        Adds API TR 5C3 tensile joint strength to the 5CT dimensional fields:
        for **BTC** with a graded ``spec`` it fills ``tension_strength_klb``
        from the buttress joint strength (Eq. 59/60, turn-key). Round-thread
        (STC/LTC) tension is left ``None`` because its governing pull-out term
        needs the API 5B engaged length Let (call
        :func:`round_thread_joint_strength_klb` with Let). Requires
        ``spec.grade`` (and thus ``min_uts_psi``) for any strength value.
        """
        conn = str(connection).upper()
        cpl = resolve_coupling(spec.od_in, conn, kind=spec.kind)
        tension = None
        if (
            conn == "BTC"
            and spec.yield_psi is not None
            and spec.min_uts_psi is not None
        ):
            tension = buttress_joint_strength_klb(
                spec.od_in,
                spec.id_in,
                spec.yield_psi,
                spec.min_uts_psi,
                coupling_od_in=cpl.coupling_od_in,
            )
        return cls(
            connection_od_in=cpl.coupling_od_in,
            connection_id_in=spec.id_in,
            coupling_length_in=cpl.coupling_length_in,
            tension_strength_klb=tension,
            connection_type=cpl.connection,
            grade=spec.grade,
        )


# ===========================================================================
# Premium connections (proprietary metal-to-metal / gas-tight) - NAME registry
# ===========================================================================
# A recognition registry of premium connection designations (VAM, Hydril, ...)
# observed in real completion designs. It carries the NAME + (where unambiguous)
# the vendor + a provenance note ONLY. Dimensional and performance data are
# proprietary and vendor-supplied - deliberately null, never fabricated (the
# same policy as the API 5CT grade performance fields; see data/sources.md).


@dataclass(frozen=True)
class PremiumConnectionSpec:
    """A registered premium connection designation + provenance.

    Carries NO dimensions and NO ratings: those are proprietary, published on
    the vendor's public datasheet, and left ``None`` here (never fabricated).
    ``vendor`` is ``None`` when the brand owner was not determined with
    confidence (not guessed).
    """

    designation: str
    vendor: Optional[str] = None
    vendor_note: Optional[str] = None
    category: str = "premium"
    spec_source: Optional[str] = None
    # --- proprietary, vendor-supplied; null by policy ---
    id_in: Optional[float] = None
    drift_in: Optional[float] = None
    joint_efficiency: Optional[float] = None
    pressure_rating_psi: Optional[float] = None
    makeup_torque_ftlb: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


class PremiumConnectionCatalog:
    """Loaded registry of premium connection designations, keyed by name."""

    def __init__(self, doc: dict):
        self._meta = doc["_meta"]
        self._by_name: Dict[str, dict] = {}
        for row in doc["premium_connections"]:
            self._by_name[str(row["designation"]).upper()] = row

    @classmethod
    def load(cls) -> "PremiumConnectionCatalog":
        doc = json.loads((_DATA_DIR / "premium_connections.json").read_text())
        return cls(doc)

    def designations(self) -> List[str]:
        return sorted(row["designation"] for row in self._by_name.values())

    def resolve(self, designation: str) -> PremiumConnectionSpec:
        row = self._by_name.get(str(designation).upper())
        if row is None:
            raise CatalogError(
                f"unknown premium connection {designation!r}; available: "
                f"{self.designations()}"
            )
        return PremiumConnectionSpec(
            designation=row["designation"],
            vendor=row.get("vendor"),
            vendor_note=row.get("vendor_note"),
            category=row.get("category", "premium"),
            spec_source=row.get("spec_source"),
            id_in=row.get("id_in"),
            drift_in=row.get("drift_in"),
            joint_efficiency=row.get("joint_efficiency"),
            pressure_rating_psi=row.get("pressure_rating_psi"),
            makeup_torque_ftlb=row.get("makeup_torque_ftlb"),
        )


@lru_cache(maxsize=None)
def _premium_connection_catalog() -> PremiumConnectionCatalog:
    return PremiumConnectionCatalog.load()


def premium_connections() -> List[str]:
    """List the registered premium connection designations (names only)."""
    return _premium_connection_catalog().designations()


def resolve_premium_connection(designation: str) -> PremiumConnectionSpec:
    """Resolve a premium connection's registry entry (name + provenance).

    Dimensions/ratings are proprietary and vendor-supplied - the returned spec
    carries them as ``None`` (consult the vendor's public datasheet).
    """
    return _premium_connection_catalog().resolve(designation)
