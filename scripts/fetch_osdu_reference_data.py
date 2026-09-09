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

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

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
}

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
