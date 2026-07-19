"""OSDU import/export for the well hierarchy — version-pinned and units-aware.

Maps `welleng.hierarchy` entities to/from OSDU Well-Known-Schema records so a
hierarchy can be **imported** from an OSDU data platform and **exported** back.

Two hard requirements:

1. **Explicit schema-version reference + quick adaptation.** OSDU schemas evolve
   (minor/patch per M-release). Every mapping pins the exact version in
   ``OSDU_SCHEMA_VERSIONS`` (one place to bump), the OSDU ``kind`` string carries
   it (``osdu:wks:master-data--Wellbore:1.1.0``), and ``from_osdu`` **recognises
   the incoming version** and warns/adapts on a mismatch rather than silently
   mis-mapping. To support a new schema version: add its version to the pin and,
   if fields moved, a version-specific branch in the entity mapper.
2. **Units-aware.** OSDU carries a unit-of-measure per numeric (e.g. a
   ``FrameOfReference``/``UnitOfMeasureID`` / ``AsIngestedCoordinates`` UoM).
   welleng works internally in **metres**; every length is converted on the
   boundary (``_to_m`` on import, ``_from_m`` on export). Volve/EDM sources are
   **feet** — do not assume metres.

The version-pin + kind parsing + units boundary + the round-trip contract are
implemented; per-entity field mapping is filled for the load-bearing entities
(Well, Wellbore, WellboreTrajectory) and extends by the same pattern. Schemas:
``https://community.opengroup.org/osdu/data/data-definitions``.
"""
from __future__ import annotations

import warnings
from typing import Any, Optional

from .hierarchy import (
    Datum, Field, Organisation, Site, Well, Wellbore, WellNetwork,
)

# --------------------------------------------------------------------------- #
# 1. explicit, pinned OSDU schema versions (bump here when a schema advances)
# --------------------------------------------------------------------------- #
#: The OSDU WKS versions this module maps against. Pin to your deployment's
#: schema registry. Bumping a value here (+ a version branch in the mapper if
#: fields moved) is the whole "adapt quickly" story.
OSDU_SCHEMA_VERSIONS: dict[str, str] = {
    "Organisation": "1.0.0",
    "Field": "1.0.0",
    "WellSiteStructure": "1.0.0",     # our Site
    "Well": "1.1.0",
    "Wellbore": "1.1.0",
    "WellboreTrajectory": "1.0.0",    # work-product-component
}

#: entity -> OSDU group-type (master-data vs work-product-component)
_OSDU_GROUP: dict[str, str] = {
    "Organisation": "master-data",
    "Field": "master-data",
    "WellSiteStructure": "master-data",
    "Well": "master-data",
    "Wellbore": "master-data",
    "WellboreTrajectory": "work-product-component",
}


def build_kind(entity: str, version: Optional[str] = None) -> str:
    """Build an OSDU ``kind`` string for a mapped entity.

    Assembles the fully-qualified OSDU Well-Known-Schema kind
    ``osdu:wks:<group>--<Entity>:<version>`` from the entity name, its
    group-type, and the pinned (or overridden) schema version.

    Parameters
    ----------
    entity : str
        The OSDU entity name — a key of :data:`OSDU_SCHEMA_VERSIONS`, e.g.
        ``"Wellbore"``, ``"Well"``, ``"WellSiteStructure"``.
    version : str or None, default None
        Schema version to embed. When ``None`` the pinned version from
        :data:`OSDU_SCHEMA_VERSIONS` is used.

    Returns
    -------
    str
        The OSDU kind string, e.g. ``"osdu:wks:master-data--Wellbore:1.1.0"``.

    Raises
    ------
    KeyError
        If ``entity`` is not a known mapped entity.

    Examples
    --------
    >>> from welleng.osdu import build_kind
    >>> build_kind('Wellbore')
    'osdu:wks:master-data--Wellbore:1.1.0'
    >>> build_kind('Wellbore', version='1.2.0')
    'osdu:wks:master-data--Wellbore:1.2.0'
    """
    group = _OSDU_GROUP[entity]
    ver = version or OSDU_SCHEMA_VERSIONS[entity]
    return f"osdu:wks:{group}--{entity}:{ver}"


def parse_kind(kind: str) -> tuple[str, str, str]:
    """Parse an OSDU ``kind`` string into its parts.

    Inverse of :func:`build_kind`: splits
    ``osdu:wks:<group>--<Entity>:<version>`` into its group-type, entity name,
    and schema version.

    Parameters
    ----------
    kind : str
        An OSDU kind string, e.g. ``"osdu:wks:master-data--Wellbore:1.1.0"``.

    Returns
    -------
    tuple of (str, str, str)
        ``(group, entity, version)`` — e.g.
        ``("master-data", "Wellbore", "1.1.0")``.

    Raises
    ------
    ValueError
        If ``kind`` does not match the expected OSDU kind structure.

    Examples
    --------
    >>> from welleng.osdu import parse_kind
    >>> parse_kind('osdu:wks:master-data--Wellbore:1.1.0')
    ('master-data', 'Wellbore', '1.1.0')
    """
    try:
        _, _, tail = kind.split(":", 2)          # -> "master-data--Wellbore:1.1.0"
        body, version = tail.rsplit(":", 1)
        group, entity = body.split("--", 1)
        return group, entity, version
    except ValueError as exc:                    # pragma: no cover - defensive
        raise ValueError(f"unrecognised OSDU kind: {kind!r}") from exc


def _check_version(entity: str, version: str) -> None:
    """Warn when an incoming record's schema version differs from the pin.

    Compares ``version`` against the pinned value in
    :data:`OSDU_SCHEMA_VERSIONS` and emits a :class:`UserWarning` on a mismatch
    — the record is still mapped, so a schema change is *recognised* and can be
    adapted rather than silently mis-mapped.

    Parameters
    ----------
    entity : str
        The OSDU entity name being mapped.
    version : str
        The schema version carried by the incoming record.

    Returns
    -------
    None

    Warns
    -----
    UserWarning
        When a pinned version exists for ``entity`` and ``version`` differs
        from it.
    """
    pinned = OSDU_SCHEMA_VERSIONS.get(entity)
    if pinned and version != pinned:
        warnings.warn(
            f"OSDU {entity} record is schema version {version}; this build is "
            f"pinned to {pinned}. Fields may have moved — verify the mapping / "
            f"add a version branch in welleng.osdu.",
            stacklevel=2,
        )


# --------------------------------------------------------------------------- #
# 2. units boundary (internal = metres)
# --------------------------------------------------------------------------- #
# Minimal UoM handling for the common length units; the general case defers to
# welleng.units (Pint). OSDU stores the UoM alongside each numeric.
_LENGTH_TO_M: dict[str, float] = {
    "m": 1.0, "metre": 1.0, "meter": 1.0,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "ftus": 1200.0 / 3937.0,                      # US survey foot
}


def _uom_factor(uom: Optional[str]) -> float:
    """Resolve an OSDU length unit-of-measure to a metres-per-unit factor.

    Handles the common length units directly and defers the general case to
    ``welleng.units`` (Pint).

    Parameters
    ----------
    uom : str or None
        The unit-of-measure token (e.g. ``"m"``, ``"ft"``, ``"ftUS"``), or a
        Pint-resolvable persistableReference. ``None`` is treated as metres.

    Returns
    -------
    float
        The multiplicative factor to convert a value in ``uom`` to metres.

    Warns
    -----
    UserWarning
        When ``uom`` cannot be resolved; it is then treated as metres
        (factor ``1.0``).
    """
    if uom is None:
        return 1.0                               # assume already metres (documented)
    key = str(uom).strip().lower().replace(" ", "")
    if key in _LENGTH_TO_M:
        return _LENGTH_TO_M[key]
    # general case: let Pint resolve it (e.g. a persistableReference)
    try:
        from .units import ureg  # type: ignore
        return ureg(key).to("meter").magnitude
    except Exception:
        warnings.warn(f"unknown length UoM {uom!r}; treating as metres.", stacklevel=2)
        return 1.0


def _to_m(value: Optional[float], uom: Optional[str]) -> Optional[float]:
    """Convert an OSDU length to internal metres.

    Parameters
    ----------
    value : float or None
        The length in ``uom`` units, or ``None``.
    uom : str or None
        The source unit-of-measure (see :func:`_uom_factor`).

    Returns
    -------
    float or None
        The length in metres, or ``None`` if ``value`` is ``None``.
    """
    return None if value is None else value * _uom_factor(uom)


def _from_m(value: Optional[float], uom: Optional[str]) -> Optional[float]:
    """Convert internal metres to an OSDU length in ``uom``.

    Parameters
    ----------
    value : float or None
        The length in metres, or ``None``.
    uom : str or None
        The target unit-of-measure (see :func:`_uom_factor`).

    Returns
    -------
    float or None
        The length expressed in ``uom`` units, or ``None`` if ``value`` is
        ``None``.
    """
    return None if value is None else value / _uom_factor(uom)


# --------------------------------------------------------------------------- #
# 3. import / export
# --------------------------------------------------------------------------- #
def from_osdu(record: dict[str, Any]) -> Any:
    """Map one OSDU record to the matching ``welleng.hierarchy`` entity.

    Dispatches on the record's ``kind`` to build the corresponding
    :mod:`welleng.hierarchy` entity, converting all lengths to internal metres
    and checking (warning on) the schema version.

    Parameters
    ----------
    record : dict
        An OSDU record shaped ``{"kind": ..., "id": ..., "data": {...}}``. The
        ``data`` payload carries the entity fields; if absent the record itself
        is used as the data.

    Returns
    -------
    Wellbore or Well or Organisation or Field or Site or dict
        The mapped hierarchy entity. ``WellboreTrajectory`` records return a
        plain dict of tie metadata (``wellbore_id``, ``top_md``, ``base_md``,
        ``azimuth_reference``) because the station bulk is a separately-loaded
        referenced dataset.

    Raises
    ------
    ValueError
        If the ``kind`` is malformed (via :func:`parse_kind`) or names an
        entity that has no ``from_osdu`` mapper.

    Warns
    -----
    UserWarning
        When the record's schema version differs from the pinned one.

    Notes
    -----
    Parent links (``WellID`` / ``KickOffWellbore``) are NOT resolved here — the
    caller wires them when assembling the :class:`~welleng.hierarchy.WellNetwork`
    (see :func:`network_from_osdu`). ``kickoff_md`` is derived, not native OSDU,
    so it is left ``None``.

    Examples
    --------
    >>> from welleng.osdu import from_osdu
    >>> rec = {'kind': 'osdu:wks:master-data--Wellbore:1.1.0', 'id': 'WB1',
    ...        'data': {'FacilityName': 'TopHole'}}
    >>> wb = from_osdu(rec)
    >>> type(wb).__name__, wb.id, wb.name
    ('Wellbore', 'WB1', 'TopHole')
    """
    kind = record.get("kind", "")
    _group, entity, version = parse_kind(kind)
    _check_version(entity, version)
    data = record.get("data", record)
    rid = record.get("id", data.get("id", ""))
    uom = data.get("LengthUnitOfMeasure")        # source UoM if the platform tags it

    if entity == "Wellbore":
        return Wellbore(
            id=rid, name=data.get("FacilityName", ""),
            # WellID / KickOffWellbore are the parent links (resolved by the caller
            # when assembling the WellNetwork); kickoff_md is DERIVED, not native.
            kickoff_md=None,
        )
    if entity == "Well":
        vm = (data.get("VerticalMeasurements") or [{}])[0]
        return Well(
            id=rid, name=data.get("FacilityName", ""),
            wellhead_depth=_to_m(
                vm.get("VerticalMeasurement"),
                vm.get("VerticalMeasurementUnitOfMeasureID") or uom),
            datum=Datum(name=vm.get("VerticalMeasurementPathID", "datum"),
                        elevation=_to_m(vm.get("VerticalMeasurement"), uom) or 0.0),
        )
    if entity == "WellboreTrajectory":
        # returns the tie metadata; station bulk is a referenced dataset, loaded
        # separately (Datasets[]). TopDepthMeasuredDepth -> section tie.
        return {
            "wellbore_id": data.get("WellboreID", ""),
            "top_md": _to_m(data.get("TopDepthMeasuredDepth"), uom),
            "base_md": _to_m(data.get("BaseDepthMeasuredDepth"), uom),
            "azimuth_reference": data.get("AzimuthReferenceType"),
        }
    if entity in ("Organisation", "Field", "WellSiteStructure"):
        cls = {"Organisation": Organisation, "Field": Field,
               "WellSiteStructure": Site}[entity]
        return cls(id=rid, name=(data.get("FacilityName")
                                 or data.get("OrganisationName")
                                 or data.get("FieldName", "")))
    raise ValueError(f"no from_osdu mapper for entity {entity!r}")


def to_osdu(entity: Any, *, version: Optional[str] = None,
            uom: str = "m") -> dict[str, Any]:
    """Map a ``welleng.hierarchy`` entity to an OSDU record.

    Inverse of :func:`from_osdu`: emits an OSDU record
    (``{"kind": ..., "id": ..., "data": {...}}``) at the pinned (or given)
    schema version, converting internal metres to the requested ``uom`` and
    encoding the parent edge (Well ``WellID`` or parent-wellbore
    ``KickOffWellbore``).

    Parameters
    ----------
    entity : Wellbore or Well or Organisation or Field or Site
        The hierarchy entity to export.
    version : str or None, keyword-only, default None
        Schema version to embed in the ``kind``; ``None`` uses the pin (see
        :func:`build_kind`).
    uom : str, keyword-only, default "m"
        The length unit-of-measure to emit numeric depths in.

    Returns
    -------
    dict
        The OSDU record.

    Raises
    ------
    ValueError
        If ``entity`` is not a type with a ``to_osdu`` mapper.

    Examples
    --------
    >>> from welleng.hierarchy import Well, Wellbore
    >>> from welleng.osdu import to_osdu
    >>> top = Wellbore(id='WB1', name='TopHole', parent=Well(id='W1', name='W1'))
    >>> lat = Wellbore(id='WB2', name='Lat1', parent=top, kickoff_md=1000.0)
    >>> rec = to_osdu(lat)
    >>> rec['kind']
    'osdu:wks:master-data--Wellbore:1.1.0'
    >>> rec['id'], rec['data']
    ('WB2', {'FacilityName': 'Lat1', 'KickOffWellbore': 'WB1'})
    """
    if isinstance(entity, Wellbore):
        parent = entity.parent
        data: dict[str, Any] = {"FacilityName": entity.name}
        if isinstance(parent, Well):
            data["WellID"] = parent.id
        elif isinstance(parent, Wellbore):
            data["KickOffWellbore"] = parent.id      # the parent-wellbore edge
        return {"kind": build_kind("Wellbore", version), "id": entity.id, "data": data}
    if isinstance(entity, Well):
        vm = []
        if entity.wellhead_depth is not None or entity.datum is not None:
            vm = [{
                "VerticalMeasurement": _from_m(
                    (entity.datum.elevation if entity.datum
                     else entity.wellhead_depth), uom),
                "VerticalMeasurementUnitOfMeasureID": uom,
            }]
        return {"kind": build_kind("Well", version), "id": entity.id,
                "data": {"FacilityName": entity.name, "VerticalMeasurements": vm}}
    if isinstance(entity, (Organisation, Field, Site)):
        ent = {"Organisation": "Organisation", "Field": "Field",
               "Site": "WellSiteStructure"}[type(entity).__name__]
        return {"kind": build_kind(ent, version), "id": entity.id,
                "data": {"FacilityName": entity.name}}
    raise ValueError(f"no to_osdu mapper for {type(entity).__name__}")


def network_from_osdu(records: list[dict[str, Any]]) -> WellNetwork:
    """Assemble a :class:`~welleng.hierarchy.WellNetwork` from OSDU records.

    Maps every ``Wellbore`` record via :func:`from_osdu`, then wires the
    ``KickOffWellbore`` / ``WellID`` parent edges into a network. Non-wellbore
    records are ignored.

    Parameters
    ----------
    records : list of dict
        OSDU records (as passed to :func:`from_osdu`). Only ``Wellbore``
        records contribute nodes.

    Returns
    -------
    WellNetwork
        The assembled network. A wellbore whose parent is a Well (a root, not
        yet added) is left with ``parent=None``.

    Notes
    -----
    Deriving each section's ``kickoff_md`` from the trajectory tie MDs is a
    follow-up, done once the referenced station datasets are loaded (OSDU has no
    native ``KickOffMD``).
    """
    net = WellNetwork()
    wellbores: dict[str, Wellbore] = {}
    parent_of: dict[str, str] = {}
    for rec in records:
        _g, entity, _v = parse_kind(rec.get("kind", ""))
        if entity != "Wellbore":
            continue
        wb = from_osdu(rec)
        wellbores[wb.id] = wb
        d = rec.get("data", rec)
        parent_of[wb.id] = d.get("KickOffWellbore") or d.get("WellID") or ""
    for wid, wb in wellbores.items():
        pid = parent_of.get(wid)
        wb.parent = wellbores.get(pid)   # None if parent is a Well (root)
        net.add(wb)
    return net
