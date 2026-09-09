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
from typing import Dict, List, NamedTuple, Optional

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


class Match(NamedTuple):
    """A resolution result, and HOW it was reached.

    An export layer has to be able to refuse a weak match, and a bare
    ``code | None`` cannot express "matched, but only on similarity". ``how``
    is one of:

    ``"exact"``
        the value IS the published code.
    ``"name"``
        it matched the code's display name (case and separators folded).
    ``"alias"``
        it matched a hand-kept field synonym -- a judgement that two
        vocabularies mean the same thing, defensible but ours, not OSDU's.
    ``"none"``
        no match; ``code`` is ``None``.

    Truthy exactly when a code was found, so ``if resolve_match(...)`` reads
    naturally.
    """

    code: Optional[str]
    how: str

    def __bool__(self) -> bool:
        return self.code is not None


def resolve_match(list_name: str, value: Optional[str]) -> Match:
    """:func:`resolve`, but reporting how the match was reached."""
    if value is None:
        return Match(None, "none")
    key = _norm(value)
    if not key:
        return Match(None, "none")
    lists = _load(list_name)["codes"]
    if value in lists:
        return Match(value, "exact")
    hit = _index(list_name).get(key)
    if hit is not None:
        return Match(hit, "exact" if _norm(hit) == key else "name")
    alias = _ALIASES.get(list_name, {}).get(key)
    return Match(alias, "alias") if alias else Match(None, "none")


def resolve(list_name: str, value: Optional[str]) -> Optional[str]:
    """Best OSDU code for a local ``value``, or ``None`` if there is no match.

    Matching is deliberately forgiving -- exact code, display name, then a
    small hand-kept alias table -- because the input is a field string, not an
    identifier. It is a LOOKUP, not a rewrite: the caller keeps its own value
    and records this alongside it.

    ``None`` means "no confident match", never "invalid". Use
    :func:`resolve_match` when the caller needs to act on HOW it matched.
    """
    return resolve_match(list_name, value).code


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


# --------------------------------------------------------------------------- #
# log curve mnemonics
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _curves() -> Dict[str, list]:
    """``{"Vendor:MNEMONIC": [property, unit quantity]}`` -- 42,919 entries.

    Stored gzipped (~200 KB) because a curve mnemonic is genuinely not
    guessable and the alternative is a hand-written family list that gets it
    wrong.
    """
    import gzip
    path = _DATA / "LogCurveType.json.gz"
    if not path.is_file():                                   # pragma: no cover
        raise OsduRefError(
            "the LogCurveType map is not installed; run "
            "scripts/fetch_osdu_reference_data.py"
        )
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)["curves"]


@lru_cache(maxsize=1)
def _by_mnemonic() -> Dict[str, List[tuple]]:
    idx: Dict[str, List[tuple]] = {}
    for key, (prop, unit) in _curves().items():
        vendor, _, mnemonic = key.partition(":")
        idx.setdefault(mnemonic.upper(), []).append((vendor, prop, unit))
    return idx


def curve_property(mnemonic: str, vendor: Optional[str] = None) -> Optional[str]:
    """The measured property behind a log-curve ``mnemonic``, or ``None``.

    Mnemonics are contractor- and vintage-specific: bulk density ships as
    ``RHOB`` (Schlumberger, Halliburton), ``RHOZ`` (Schlumberger) and ``BDCX``
    (Baker Hughes). This resolves them through the published vendor map rather
    than through a hand-written name list, which is what reported "no density
    curve" on 12 of 14 wells that had one.

    ``vendor`` narrows the lookup when the log says who ran it. Without it, a
    mnemonic that means **different things to different vendors** returns
    ``None`` rather than picking one -- an ambiguous answer is worse than none
    for a measurement that feeds a calculation.
    """
    hits = _by_mnemonic().get(str(mnemonic).strip().upper(), [])
    if vendor is not None:
        key = _norm(vendor)
        hits = [h for h in hits if _norm(h[0]) == key]
    props = {h[1] for h in hits if h[1]}
    if len(props) == 1:
        return props.pop()
    return None


def curve_quantity(mnemonic: str, vendor: Optional[str] = None) -> Optional[str]:
    """The UNIT QUANTITY behind a mnemonic (``"mass per volume"``, …), or None.

    More robust than :func:`curve_property` and usually the one to reach for.
    Vendors name the same measurement differently -- ``RHOB`` is *density* to
    one and *bulk density* to another -- so the property can disagree while the
    quantity does not, and the quantity is what makes two curves comparable.

    Still ``None`` when the quantity itself is ambiguous, which is the case
    that matters: ``DT`` is *time per length* to two vendors and plain *time*
    to a third, and those are not the same curve.
    """
    hits = _by_mnemonic().get(str(mnemonic).strip().upper(), [])
    if vendor is not None:
        key = _norm(vendor)
        hits = [h for h in hits if _norm(h[0]) == key]
    q = {h[2] for h in hits if h[2]}
    return q.pop() if len(q) == 1 else None


def curve_vendors(mnemonic: str) -> List[tuple]:
    """Every ``(vendor, property, unit quantity)`` published for a mnemonic.

    Use this when :func:`curve_property` returns ``None`` to see whether the
    mnemonic is unknown or merely ambiguous -- the two need different handling.
    """
    return sorted(_by_mnemonic().get(str(mnemonic).strip().upper(), []))
