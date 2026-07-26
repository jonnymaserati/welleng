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
        user-supplied - never fabricated).
        """
        cpl = resolve_coupling(od_in, connection, kind=kind)
        return cls(
            connection_od_in=cpl.coupling_od_in,
            coupling_length_in=cpl.coupling_length_in,
            connection_type=cpl.connection,
            grade=grade,
        )
