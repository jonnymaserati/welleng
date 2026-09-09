"""Fetch and slim the OSDU reference-data code lists welleng resolves against.

The OSDU value manifests carry ACL/legal envelopes and long descriptions we do
not need; this keeps the code, the display name, and the provenance, and writes
one small JSON per list into ``welleng/data/osdu/``.

Only the lists welleng actually uses are fetched -- there are 451 published and
vendoring them all would be dead weight. Add a row to ``LISTS`` when a new list
is wired up.

Run:  ``python scripts/fetch_osdu_reference_data.py``

Source: https://community.opengroup.org/osdu/data/data-definitions
Licence: Apache-2.0. Per-record ``AttributionAuthority`` is preserved -- several
of these lists are curated by PPDM, not by OSDU, and that has to travel with
the data.
"""
from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import unquote

RAW = ("https://community.opengroup.org/osdu/data/data-definitions/-/raw/"
       "master/ReferenceValues/Manifests/reference-data")

#: ``list name -> governance tier``. The tier is the OSDU folder and it decides
#: how strictly welleng validates a value (see ``welleng.osdu_ref``).
LISTS: dict[str, str] = {
    # depth / datum
    "VerticalMeasurementType": "OPEN",
    "VerticalMeasurementPath": "OPEN",
    # survey
    "AzimuthReferenceType": "OPEN",
    "GeoMagneticModel": "OPEN",
    # schematic
    "CementPlugType": "OPEN",
    "PerforationIntervalType": "OPEN",
    "AnnularFluidType": "OPEN",
    "LinerType": "LOCAL",
    # tubulars
    "TubularComponentGrade": "LOCAL",
    "TubularComponentType": "LOCAL",
    "TubularMaterialType": "OPEN",
    # pressure / integrity testing
    "FormationIntegrityTestType": "OPEN",
    "FormationIntegrityTestResult": "OPEN",
    "FormationIntegrityPressureDataSource": "OPEN",
    "WellPressureTestGaugeType": "OPEN",
    # fluids
    # NB not "FluidType" -- every one of its 27 records is DEPRECATED in
    # favour of WellFluidType. The deprecation filter below is what
    # surfaced that, by returning an empty list.
    "WellFluidType": "OPEN",
    "FluidRheologicalModelType": "OPEN",
    "FluidContactType": "FIXED",
    # logs
    "LogCurveMainFamily": "LOCAL",
}

#: Handled separately: 42,919 vendor mnemonics at 73 MB raw. Slimmed to
#: ``{Vendor:Mnemonic -> [property, unit quantity]}`` it is ~200 KB gzipped,
#: which is worth carrying because a curve mnemonic is not guessable --
#: bulk density alone ships as RHOB, RHOZ and BDCX depending on vendor.
CURVE_TYPES = ("LogCurveType", "LOCAL")

OUT = Path(__file__).resolve().parent.parent / "welleng" / "data" / "osdu"


def fetch(name: str, tier: str) -> dict:
    url = f"{RAW}/{tier}/{name}.1.json"
    with urllib.request.urlopen(url, timeout=60) as fh:      # noqa: S310
        doc = json.load(fh)

    codes: dict[str, str] = {}
    authorities: set[str] = set()
    for rec in doc.get("ReferenceData", []):
        data = rec.get("data", {})
        code = data.get("Code")
        if not code:
            continue
        # A deprecated record still has to RESOLVE (old data references it) but
        # must never be offered as a current code, so it is dropped here and
        # the caller keeps whatever string it had.
        if str(data.get("Description", "")).startswith("DEPRECATED"):
            continue
        codes[str(code)] = str(data.get("Name") or code)
        auth = data.get("AttributionAuthority")
        if auth:
            authorities.add(str(auth))

    if not codes:
        # A list that resolves to nothing is almost always a list that has been
        # superseded wholesale (every record DEPRECATED). Fail loudly: an empty
        # vocabulary shipped quietly would validate nothing and warn on
        # everything.
        raise RuntimeError(
            f"{name}: no live codes -- every record is deprecated, or the "
            "manifest shape changed. Check for a successor list."
        )
    return {
        "list": name,
        "kind": doc.get("kind", f"osdu:wks:reference-data--{name}:1.0.0"),
        "governance": tier,
        "source": f"{RAW}/{tier}/{name}.1.json",
        "retrieved": date.today().isoformat(),
        "licence": "Apache-2.0",
        "attribution": sorted(authorities),
        "codes": codes,
    }


def fetch_curve_types() -> dict:
    """The vendor-mnemonic map, slimmed hard and stored gzipped.

    Only the property name and the unit quantity are kept -- those are what let
    a caller ask for "bulk density" instead of guessing at RHOB / RHOZ / BDCX.
    """
    name, tier = CURVE_TYPES
    url = f"{RAW}/{tier}/{name}.1.json"
    with urllib.request.urlopen(url, timeout=600) as fh:     # noqa: S310
        doc = json.load(fh)

    out: dict[str, list] = {}
    for rec in doc.get("ReferenceData", []):
        data = rec.get("data", {})
        code = data.get("Code")
        if not code or str(data.get("Description", "")).startswith("DEPRECATED"):
            continue
        prop = (data.get("PropertyType") or {}).get("Name")
        uq = str(data.get("UnitQuantityID") or "")
        # ".../UnitQuantity:mass%20per%20volume:" -> "mass per volume"
        uq = unquote(uq.split(":")[-2]) if uq.count(":") >= 2 else ""
        out[str(code)] = [prop, uq or None]
    if not out:
        raise RuntimeError(f"{name}: no live codes")
    return {
        "list": name,
        "kind": doc.get("kind", f"osdu:wks:reference-data--{name}:1.0.0"),
        "governance": tier,
        "source": url,
        "retrieved": date.today().isoformat(),
        "licence": "Apache-2.0",
        "attribution": [],
        "curves": out,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, tier in sorted(LISTS.items()):
        try:
            doc = fetch(name, tier)
        except Exception as exc:                             # pragma: no cover
            print(f"FAILED {name}: {exc}", file=sys.stderr)
            return 1
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
        print(f"{name:<28} {tier:<6} {len(doc['codes']):>4} codes  "
              f"{', '.join(doc['attribution']) or '-'}")

    curves = fetch_curve_types()
    blob = json.dumps(curves, separators=(",", ":")).encode()
    path = OUT / "LogCurveType.json.gz"
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(blob)
    print(f"{'LogCurveType':<28} {CURVE_TYPES[1]:<6} "
          f"{len(curves['curves']):>4} mnemonics  "
          f"({path.stat().st_size / 1024:.0f} KB gzipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
