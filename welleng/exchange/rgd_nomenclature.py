"""RGD lithostratigraphic nomenclature resolver.

Resolves the raw ``stratUnitId`` codes served by NLOG ``stratinterpretations``
(e.g. ``KNGLU``, ``KNNCM``, ``ZEZ1F``) to their name, rank and hierarchy. The
bundled reference table is digitised from the **pre-2020 RGD nomenclature** — Van
Adrichem Boogaert & Kouwe (1993), *Stratigraphic Nomenclature of the
Netherlands*, revision RGD/NOGEPA — which is the version NLOG's data uses; the
2020 DINO revamp renamed/retired codes, so it must NOT be used to resolve NLOG
codes. See ``docs/dev/NLOG_STRATIGRAPHY_NOMENCLATURE.md``.

Hierarchy is authoritative: RGD codes are prefix-nested (group -> formation ->
member) by construction, so a unit's parent is the longest proper prefix that is
itself a code, verified against the source's own section numbering and the
DINO-2021 survivors. **Names are OCR-raw and unverified** — the source is a
scanned document whose OCR spaces letters and confuses w/vv, r/n; use codes and
hierarchy, treat names as indicative only.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

Relation = str  # same|ancestor|descendant|sibling|unrelated|unknown


_DATA = "rgd_nomenclature_1993.json"


def _load() -> dict:
    return json.loads((files("welleng.exchange") / "data" / _DATA).read_text())


@lru_cache(maxsize=1)
def _units() -> dict:
    return _load()["units"]


def provenance() -> dict:
    """The reference table's provenance block (source, version, OCR caveat)."""
    return _load()["_provenance"]


def resolve(code: str) -> dict | None:
    """Resolve an RGD code to ``{code, name, rank, parent, ancestors}``.

    ``ancestors`` is the parent chain from the immediate parent up to the group
    root. Returns ``None`` if the code is not in the nomenclature (a caller must
    then treat it as unresolved, not guess). ``name`` may be ``None`` (informal or
    OCR-unrecoverable) and is in all cases OCR-raw — see the module docstring.
    """
    units = _units()
    u = units.get(code)
    if u is None:
        return None
    ancestors, p = [], u["parent"]
    while p:
        ancestors.append(p)
        p = units.get(p, {}).get("parent")
    return {"code": code, "name": u["name"], "rank": u["rank"],
            "parent": u["parent"], "ancestors": ancestors}


def related(a: str, b: str) -> Relation:
    """Classify the relationship between two RGD codes.

    ``'same'`` (identical), ``'ancestor'`` (``a`` is an ancestor of ``b`` — e.g.
    a formation and its member), ``'descendant'`` (``a`` is below ``b``),
    ``'sibling'`` (same immediate parent), ``'unrelated'``, or ``'unknown'`` (one
    or both codes are not in the nomenclature). A caller comparing depths across
    two codes should treat anything other than ``'unrelated'`` as the SAME surface
    family (compare via the common ancestor), and ``'unknown'`` as not comparable.
    """
    if a == b:
        return "same"
    ra, rb = resolve(a), resolve(b)
    if ra is None or rb is None:
        return "unknown"
    if a in rb["ancestors"]:
        return "ancestor"
    if b in ra["ancestors"]:
        return "descendant"
    if ra["parent"] and ra["parent"] == rb["parent"]:
        return "sibling"
    return "unrelated"


def common_ancestor(a: str, b: str) -> str | None:
    """The deepest code that is ``a``-or-an-ancestor and ``b``-or-an-ancestor, or
    ``None`` if they share none / either is unknown."""
    ra, rb = resolve(a), resolve(b)
    if ra is None or rb is None:
        return None
    chain_a = [a] + ra["ancestors"]
    chain_b = set([b] + rb["ancestors"])
    for c in chain_a:              # deepest first
        if c in chain_b:
            return c
    return None
