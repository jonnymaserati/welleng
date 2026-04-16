"""Convert OWSG Set A / Set B Excel workbooks to ISCWSA JSON tool models.

Each sheet in the OWSG xlsx becomes one ISCWSA-format JSON file
matching the schema vendored at ``welleng/errors/iscwsa_schema/``.
Mapping is mechanical -- the OWSG xlsx column layout was the de facto
source of the JSON schema's field set, so the conversion is
field-for-field.

Output goes to ``welleng/errors/iscwsa_json/owsg_<a|b>/<sheet>.json``.

Anything the converter encounters that doesn't fit the schema (units
not in the enum, unparseable propagation modes, missing required
fields, …) is flagged in the per-file ``conversion_warnings`` list
embedded in the JSON's ``metadata.tags`` for downstream cataloguing.
The same diagnostics are also printed to stdout so they're easy to
collect into a single GitHub-discussion-ready summary.

Usage:

    python -m welleng.errors.tools.owsg_to_json \\
        --xlsx /path/to/01-owsg-set-a-...xlsx \\
        --out  welleng/errors/iscwsa_json/owsg_a/

By default both Set A and Set B are searched at their canonical paths
under ``~/Dropbox/Documents/Reference/Wellbore Uncertainty/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# OWSG xlsx column conventions (verified across MWD, EMS, GYRO-NS, BLIND, ...)
# ---------------------------------------------------------------------------

COL_LABEL = 0       # 'Short Name:', 'Long Name:', ... (metadata key column)
COL_LABEL_VAL = 1   # value for the metadata key
COL_NO = 3          # term row number
COL_CODE = 4
COL_TERM_DESCRIPTION = 5
COL_WTFN = 6        # weight-function name (legacy welleng dispatcher uses this)
COL_WTFN_SOURCE = 7
COL_TYPE = 8        # 'Depth' / 'Sensor' / 'Align' / 'AziRef' / 'Readng'
COL_MAGNITUDE = 9
COL_UNITS = 10
COL_PROP = 11       # 'R' / 'S' / 'G' / 'W' / 'B'
COL_P1 = 12         # depth_factor
COL_P2 = 13         # inclination_factor
COL_P3 = 14         # azimuth_factor
COL_WTFN_COMMENT = 15
COL_DEPTH_FORMULA = 16
COL_INCLINATION_FORMULA = 17
COL_AZIMUTH_FORMULA = 18
COL_SING_NORTH = 19
COL_SING_EAST = 20
COL_SING_VERT = 21


# Mapping OWSG one-letter prop codes to the ISCWSA JSON enum.
PROP_MODE_MAP = {
    "R": "Random",
    "S": "Systematic",
    "G": "Global",
    "W": "Well",
    # "B" (Bias) is not in the ISCWSA enum -- flagged as a warning at conversion.
}

# Tool-type free-text → ISCWSA enum.
TOOL_TYPE_MAP = {
    "MWD": "Magnetic",
    "EMS": "Magnetic",
    "Magnetic": "Magnetic",
    "MAGNETIC": "Magnetic",
    "Film MMS": "Magnetic",
    "Film MSS": "Magnetic",
    "Gyro": "Gyroscopic",
    "GYRO": "Gyroscopic",
    "Gyroscopic": "Gyroscopic",
    "Inertial": "Inertial",
    "INERTIAL": "Inertial",
    "FINDS": "Inertial",
    "BHI RIGS": "Inertial",
    "BLIND": "Unknown",
    "Unknown": "Unknown",
    "INC-ONLY": "Unknown",
}

# Units in the ISCWSA enum (vendored schema, draft.json).
ALLOWED_UNITS = {"m", "1/m", "nT", "m/s2", "deg", "deg/nT", "deg/hr", "rad"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_nanish(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float):
        return pd.isna(x)
    return False


def _formula_str(x: Any) -> str:
    """Normalise a formula cell to a string the JSON schema accepts.

    The schema requires ``depth_formula`` etc. to be non-empty strings.
    OWSG cells that are the integer 0 mean "this term has no
    contribution to that axis"; the schema represents that as the
    string ``"0"``.
    """
    if _is_nanish(x):
        return "0"
    if isinstance(x, (int, float)):
        # Render integers without trailing ".0"
        if float(x).is_integer():
            return str(int(x))
        return str(float(x))
    return str(x).strip()


def _singularity_or_none(x: Any) -> str | None:
    """Singularity cells: ``None`` if the cell is empty or numeric 0."""
    if _is_nanish(x):
        return None
    if isinstance(x, (int, float)) and float(x) == 0.0:
        return None
    return str(x).strip()


def _parse_inc_range(raw: Any) -> int | None:
    """Strip ' deg' suffix and parse an integer; return None if missing."""
    if _is_nanish(raw):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(deg)?$", s)
    if not m:
        return None
    return int(round(float(m.group(1))))


def _to_iso_date(x: Any) -> str:
    if isinstance(x, (date, datetime)):
        return x.strftime("%Y-%m-%d")
    if _is_nanish(x):
        return "1970-01-01"
    return str(x)[:10]


def _scrub(x: Any) -> str:
    """Coerce a cell to a plain string for metadata fields, NaN → empty."""
    if _is_nanish(x):
        return ""
    return str(x)


def _det_uuid(seed: str) -> str:
    """Deterministic UUID from a seed string (so re-runs are stable)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"welleng/owsg/{seed}"))


# ---------------------------------------------------------------------------
# Per-sheet conversion
# ---------------------------------------------------------------------------


def _extract_metadata_kv(df: pd.DataFrame) -> dict[str, str]:
    """Walk col 0 (label) / col 1 (value) and collect the metadata pairs."""
    kv: dict[str, str] = {}
    for i in range(len(df)):
        label = df.iloc[i, COL_LABEL]
        if _is_nanish(label):
            continue
        key = str(label).rstrip(":").strip()
        # Skip the data columns' header row "OWSG Prefix" line which sits at
        # the same column-0 position (both COL_LABEL and COL_LABEL_VAL hold
        # header text, not a metadata value).
        if str(df.iloc[i, COL_LABEL_VAL]).strip().startswith("Short Name"):
            continue
        kv[key] = df.iloc[i, COL_LABEL_VAL]
    return kv


def _term_rows(df: pd.DataFrame) -> list[int]:
    rows = []
    for i in range(len(df)):
        v = df.iloc[i, COL_NO]
        if isinstance(v, (int, float)) and not pd.isna(v):
            rows.append(i)
    return rows


def convert_sheet(
    df: pd.DataFrame,
    sheet_name: str,
    set_label: str,
    source_xlsx: str,
) -> tuple[dict, list[str]]:
    """Convert one xlsx sheet into an ISCWSA-format JSON object.

    Returns ``(json_dict, warnings)``.
    """
    warnings: list[str] = []
    kv = _extract_metadata_kv(df)

    short_name = _scrub(kv.get("Short Name") or sheet_name)
    long_name = _scrub(kv.get("Long Name") or short_name)
    owsg_prefix = _scrub(kv.get("OWSG Prefix") or sheet_name)

    raw_tool_type = _scrub(kv.get("Tool Type") or short_name)
    tool_type = TOOL_TYPE_MAP.get(raw_tool_type, None)
    if tool_type is None:
        # Try to infer from sheet name keywords.
        if "GYRO" in sheet_name.upper():
            tool_type = "Gyroscopic"
        elif "BLIND" in sheet_name.upper():
            tool_type = "Unknown"
        elif "INC" in sheet_name.upper():
            tool_type = "Unknown"
        else:
            tool_type = "Magnetic"
        warnings.append(
            f"tool_type {raw_tool_type!r} not in OWSG→ISCWSA map; inferred {tool_type!r}"
        )

    inc_min = _parse_inc_range(kv.get("Inclination Range Min")) or 0
    inc_max = _parse_inc_range(kv.get("Inclination Range Max")) or 180

    metadata = {
        "schema_uuid": "00000000-0000-0000-0000-000000000000",
        "model_uuid": _det_uuid(f"{set_label}/{owsg_prefix}"),
        "model_id": owsg_prefix or short_name,
        "short_name": short_name or sheet_name,
        "long_name": long_name or short_name or sheet_name,
        "revision_number": int(kv.get("Revision No") or 0)
            if isinstance(kv.get("Revision No"), (int, float)) and not pd.isna(kv.get("Revision No"))
            else 0,
        "revision_date": _to_iso_date(kv.get("Revision Date")),
        "revision_comment": _scrub(kv.get("Revision Comment")) or "Initial conversion",
        "source": _scrub(kv.get("Source")) or f"OWSG {set_label} via welleng converter",
        "application": _scrub(kv.get("Application")) or short_name,
        "tool_type": tool_type,
        "author": "welleng OWSG-to-ISCWSA converter",
        "framework": "ISCWSA Rev5",
        "tags": [
            f"owsg-{set_label.lower()}",
            f"sheet-{sheet_name}",
            f"source-xlsx:{Path(source_xlsx).name}",
        ],
    }

    parameters = {"inc_min": int(inc_min), "inc_max": int(inc_max)}

    terms: list[dict] = []
    for i in _term_rows(df):
        code = _scrub(df.iloc[i, COL_CODE])
        if not code:
            continue
        magnitude = df.iloc[i, COL_MAGNITUDE]
        units_raw = _scrub(df.iloc[i, COL_UNITS])
        prop_raw = _scrub(df.iloc[i, COL_PROP])

        if units_raw not in ALLOWED_UNITS:
            warnings.append(
                f"term {code!r}: unit {units_raw!r} not in ISCWSA enum "
                f"{sorted(ALLOWED_UNITS)}"
            )
        if prop_raw not in PROP_MODE_MAP:
            warnings.append(
                f"term {code!r}: propagation mode {prop_raw!r} not in OWSG→ISCWSA map"
            )

        try:
            mag_value = float(magnitude) if not _is_nanish(magnitude) else 0.0
        except (TypeError, ValueError):
            mag_value = 0.0
            warnings.append(f"term {code!r}: non-numeric magnitude {magnitude!r}; defaulted to 0")

        # Capture the OWSG weight-function name as a non-schema field
        # so the conformance harness can dispatch to the matching legacy
        # welleng implementation without guessing. ISCWSA JSON schema
        # does not (yet) reserve this slot — we use a reserved-prefix
        # field so it round-trips cleanly without violating
        # additionalProperties:false in any consumer that tightens up.
        wtfn_raw = _scrub(df.iloc[i, COL_WTFN])

        term = {
            "uuid": _det_uuid(f"{set_label}/{owsg_prefix}/{code}"),
            "name": code,
            "x_owsg_weight_function": wtfn_raw,
            "value": mag_value,
            "units": units_raw if units_raw in ALLOWED_UNITS else units_raw,
            "depth_factor": int(df.iloc[i, COL_P1]) if not _is_nanish(df.iloc[i, COL_P1]) else 0,
            "inclination_factor": int(df.iloc[i, COL_P2]) if not _is_nanish(df.iloc[i, COL_P2]) else 0,
            "azimuth_factor": int(df.iloc[i, COL_P3]) if not _is_nanish(df.iloc[i, COL_P3]) else 0,
            "propagation_mode": PROP_MODE_MAP.get(prop_raw, prop_raw),
            "depth_formula": _formula_str(df.iloc[i, COL_DEPTH_FORMULA]),
            "inclination_formula": _formula_str(df.iloc[i, COL_INCLINATION_FORMULA]),
            "azimuth_formula": _formula_str(df.iloc[i, COL_AZIMUTH_FORMULA]),
            "north_singularity": _singularity_or_none(df.iloc[i, COL_SING_NORTH]),
            "east_singularity": _singularity_or_none(df.iloc[i, COL_SING_EAST]),
            "vertical_singularity": _singularity_or_none(df.iloc[i, COL_SING_VERT]),
            "hash_value": "00000000000000000000000000000000",
        }
        terms.append(term)

    if warnings:
        metadata["tags"].extend(f"warn:{w}" for w in warnings[:5])

    out = {
        "$schema": "../../iscwsa_schema/draft.json",
        "metadata": metadata,
        "parameters": parameters,
        "terms": terms,
        "hash_function": "SHA256",
        "hash_value": "00000000000000000000000000000000",
    }
    return out, warnings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


# Defaults resolve to the OWSG toolgroup workbooks vendored at
# data/iscwsa/toolgroups/ (relative to the welleng repo root). See that
# directory's README for source / licence / coverage.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOLGROUPS_DIR = _REPO_ROOT / "data" / "iscwsa" / "toolgroups"

DEFAULT_XLSX_A = str(
    _TOOLGROUPS_DIR / "toolgroup-owsg-a-rev-5-1-08-oct-2020-produced-23-sep-2022.xlsx"
)
DEFAULT_XLSX_B = str(
    _TOOLGROUPS_DIR / "toolgroup-owsg-b-rev-5-1-08-oct-2020-produced-23-sep-2022.xlsx"
)


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+\-.]", "_", s)


def convert_workbook(xlsx_path: str, out_dir: str, set_label: str) -> dict[str, list[str]]:
    """Loop every sheet, write one JSON per sheet, return per-sheet warnings."""
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    xl = pd.ExcelFile(xlsx_path)
    all_warnings: dict[str, list[str]] = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet_name=sheet, header=None)
        # Skip non-tool sheets if any (heuristic: must have at least 1 term row)
        if not _term_rows(df):
            continue
        json_obj, warns = convert_sheet(df, sheet, set_label, xlsx_path)
        out_path = out_dir_p / f"{_safe_filename(sheet)}.json"
        with open(out_path, "w") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)
        all_warnings[sheet] = warns
    return all_warnings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xlsx-a", default=DEFAULT_XLSX_A,
                   help="OWSG Set A xlsx (default: vendored Rev 5-1 in data/iscwsa/toolgroups/)")
    p.add_argument("--xlsx-b", default=DEFAULT_XLSX_B,
                   help="OWSG Set B xlsx (default: vendored Rev 5-1 in data/iscwsa/toolgroups/)")
    p.add_argument(
        "--out-root",
        default=str(Path(__file__).parents[2] / "errors" / "iscwsa_json"),
        help="Root output directory; subdirs owsg_a/ and owsg_b/ created beneath",
    )
    args = p.parse_args()

    out_root = Path(args.out_root)
    summary: dict[str, dict[str, list[str]]] = {}
    for label, path in (("A", args.xlsx_a), ("B", args.xlsx_b)):
        if not os.path.isfile(path):
            print(f"!! {path} not found, skipping Set {label}")
            continue
        out_dir = out_root / f"owsg_{label.lower()}"
        print(f"\n=== Converting Set {label} ({path}) → {out_dir} ===")
        per_sheet = convert_workbook(path, str(out_dir), f"Set{label}")
        summary[label] = per_sheet
        # One-line per sheet
        for sheet, warns in per_sheet.items():
            tag = "OK" if not warns else f"{len(warns)} warning(s)"
            print(f"  {sheet:50s}  {tag}")

    # Aggregate warnings catalogue
    print("\n=== Warnings catalogue ===")
    by_kind: dict[str, list[tuple[str, str, str]]] = {}
    for set_label, per_sheet in summary.items():
        for sheet, warns in per_sheet.items():
            for w in warns:
                # Bucket by the warning's 'kind' prefix (text up to ':')
                kind = w.split(":")[0] if ":" in w else w
                by_kind.setdefault(kind, []).append((set_label, sheet, w))
    for kind, entries in sorted(by_kind.items()):
        print(f"\n  {kind}  ({len(entries)} occurrence(s))")
        for set_label, sheet, w in entries[:5]:
            print(f"    Set {set_label} / {sheet}: {w}")
        if len(entries) > 5:
            print(f"    ... and {len(entries) - 5} more")


if __name__ == "__main__":
    main()
