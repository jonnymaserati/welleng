"""OSDU reference-data code lists — resolve, validate, annotate.

:mod:`welleng.osdu` maps the *shape* of an OSDU record (which entity, which
schema version, which units). This module supplies the other half: the
**controlled vocabularies** that a record's ``...TypeID`` / ``...ID`` fields
must carry. A record with the right shape and no reference-data ids is
structurally valid and semantically empty.

Design
------
**Annotation, never renaming.** welleng's own vocabulary does not change --
``azi_reference="true"`` stays ``"true"``. :func:`resolve` maps a value ONTO an
OSDU code and hands it back for the caller to record alongside; it never
rewrites the caller's string. Adoption that is not additive is a breaking
change for every consumer.

**Governance decides strictness.** OSDU publishes each list under one of three
governance models, and that is exactly the right validation policy, so it is
used directly rather than re-invented per list:

======== ==========================================================
governance behaviour of :func:`validate` on an unknown code
======== ==========================================================
FIXED    raise -- the set is closed, so an unknown value is an error
OPEN     warn and pass through -- operators may extend the list
LOCAL    pass through silently -- the shipped list is a starter set
======== ==========================================================

Data
----
One slimmed JSON per list in ``welleng/data/osdu/``, refreshed by
``scripts/fetch_osdu_reference_data.py``. Only the lists welleng actually uses
are carried; 451 are published. Each file records its source URL, retrieval
date, governance tier and ``AttributionAuthority`` -- several of these lists
are curated by PPDM rather than by OSDU, and that travels with the data.

Source: https://community.opengroup.org/osdu/data/data-definitions (Apache-2.0).

Example
-------
>>> from welleng import osdu_ref
>>> osdu_ref.resolve("VerticalMeasurementType", "RKB")
'RotaryTable'
>>> osdu_ref.resolve("AzimuthReferenceType", "grid")
'GridNorth'
>>> osdu_ref.osdu_id("CementPlugType", "Abandonment")
'reference-data--CementPlugType:Abandonment'
"""
from __future__ import annotations

import json
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

_DATA = Path(__file__).resolve().parent / "data" / "osdu"

#: Governance tiers, strictest first.
FIXED, OPEN, LOCAL = "FIXED", "OPEN", "LOCAL"


class OsduRefError(KeyError):
    """A reference-data list or code could not be resolved."""


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _load(list_name: str) -> dict:
    path = _DATA / f"{list_name}.json"
    if not path.is_file():
        raise OsduRefError(
            f"no vendored OSDU list {list_name!r}; available: "
            f"{', '.join(available())}. Add it to LISTS in "
            "scripts/fetch_osdu_reference_data.py and re-run that script."
        )
    return json.loads(path.read_text())


def available() -> List[str]:
    """Names of the vendored reference-data lists."""
    return sorted(p.stem for p in _DATA.glob("*.json"))


def codes(list_name: str) -> Dict[str, str]:
    """``{code: display name}`` for a list. Deprecated codes are not present."""
    return dict(_load(list_name)["codes"])


def governance(list_name: str) -> str:
    """``FIXED`` / ``OPEN`` / ``LOCAL`` -- who may extend the list."""
    return _load(list_name)["governance"]


def provenance(list_name: str) -> dict:
    """Source URL, retrieval date, licence and attribution authorities."""
    doc = _load(list_name)
    return {k: doc[k] for k in
            ("list", "kind", "governance", "source", "retrieved", "licence",
             "attribution")}


# --------------------------------------------------------------------------- #
# resolve / validate
# --------------------------------------------------------------------------- #
def _norm(value: str) -> str:
    """Fold case, spacing and separators so ``"Rotary Table"``, ``"rotary-table"``
    and ``"RotaryTable"`` compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


@lru_cache(maxsize=None)
def _index(list_name: str) -> Dict[str, str]:
    """Normalised lookup: code AND display name both point at the code."""
    idx: Dict[str, str] = {}
    for code, name in _load(list_name)["codes"].items():
        idx.setdefault(_norm(code), code)
        idx.setdefault(_norm(name), code)
    return idx


#: Field synonyms that no amount of string folding will bridge. Keyed by list,
#: then by the normalised local term. Deliberately small: an entry here is a
#: judgement that two vocabularies mean the same thing, and each one should be
#: defensible on its own.
_ALIASES: Dict[str, Dict[str, str]] = {
    "VerticalMeasurementType": {
        "rkb": "RotaryTable",            # rotary kelly bushing, used
        "kb": "KellyBushing",            # interchangeably in the field but
        "rt": "RotaryTable",             # OSDU separates the two
        "msl": "MeanSeaLevel",
        "df": "DrillFloor",
        "gl": "GroundLevel",
        "seabed": "Seafloor",
        # NB "mudline" needs no alias -- it is its own OSDU code (MLS), and
        # the list carries BOTH MLS and Seafloor for what is one surface. That
        # duplication is OSDU's, not ours; resolving each term to the code that
        # names it is the only defensible behaviour.
        # NOT mapped: "wellhead". OSDU has CasingHeadFlange, TubingHeadFlange
        # and TopBottomFlange, and a bare "wellhead" does not say which. An
        # unresolved value is recoverable; a wrong one is not.
    },
    "VerticalMeasurementPath": {
        "md": "MeasuredDepth",
        "tvd": "TrueVerticalDepth",
        "tvdss": "TrueVerticalDepth",    # subsea is a DATUM, not a path
    },
    "AzimuthReferenceType": {
        "true": "TrueNorth",
        "grid": "GridNorth",
        "magnetic": "MagneticNorth",
    },
}


def resolve(list_name: str, value: Optional[str]) -> Optional[str]:
    """Best OSDU code for a local ``value``, or ``None`` if there is no match.

    Matching is deliberately forgiving -- exact code, display name, then a
    small hand-kept alias table -- because the input is a field string, not an
    identifier. It is a LOOKUP, not a rewrite: the caller keeps its own value
    and records this alongside it.

    ``None`` means "no confident match", never "invalid".
    """
    if value is None:
        return None
    key = _norm(value)
    if not key:
        return None
    hit = _index(list_name).get(key)
    if hit is not None:
        return hit
    return _ALIASES.get(list_name, {}).get(key)


def validate(list_name: str, code: Optional[str], *,
             field: str = "value") -> Optional[str]:
    """Check ``code`` against the list, with strictness set by its governance.

    ``FIXED`` raises on an unknown code, ``OPEN`` warns and passes it through
    (operators may extend an OPEN list), ``LOCAL`` passes silently (the shipped
    values are a starter set). ``None`` is always allowed -- absence is not a
    validation question.
    """
    if code is None:
        return None
    if code in codes(list_name):
        return code
    gov = governance(list_name)
    if gov == FIXED:
        raise OsduRefError(
            f"{field}={code!r} is not in the OSDU {list_name} list, which is "
            "FIXED -- the set is closed, so this is a data error, not an "
            "extension."
        )
    if gov == OPEN:
        warnings.warn(
            f"{field}={code!r} is not a published OSDU {list_name} code. "
            "That list is OPEN, so an operator extension is legitimate -- but "
            "it will not be recognised by a consumer using the standard list.",
            stacklevel=2,
        )
    return code


def osdu_id(list_name: str, code: str, namespace: str = "") -> str:
    """The reference-data id a record's ``...ID`` field carries.

    ``namespace`` prefixes the id for a specific data partition; left empty the
    partition-relative form is returned, which is what a caller composing a
    record against its own namespace wants.
    """
    stem = f"reference-data--{list_name}:{code}"
    return f"{namespace}:{stem}" if namespace else stem
