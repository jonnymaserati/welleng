"""Validate the JSON+interpreter against Copsegrove diagnostic .txt files.

The Copsegrove ``svy`` rows give per-station per-error-source HLA sigmas:
``(sig_HighSide, sig_Lateral, sig_AlongHole)``. For RANDOM terms,
this equals ``abs(dpde) × magnitude`` directly — no propagation
between stations is involved, so the comparison is pure
formula-evaluation validation independent of welleng's propagation
machinery.

For SYSTEMATIC terms the Copsegrove ``svy`` value is the propagated
trapezoidal-rule accumulation across stations, which requires the
welleng propagation engine to reproduce. That's a separate piece of
work — this validator currently flags systematic terms as DEFERRED
rather than testing them.

Frame mapping (interpreter dpde columns → Copsegrove svy columns):
    dpde[0] (depth_formula)        → sig A/H (along-hole)
    dpde[1] (inclination_formula)  → sig H/S (high-side)
    dpde[2] (azimuth_formula)      → sig LAT (lateral)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import welleng as we

from .copsegrove import parse_copsegrove
from .interpreter import evaluate_formula


# Map JSON propagation_mode → comparison strategy
_RANDOM_MODES = {"Random"}
_DEFERRED_MODES = {"Systematic", "Global", "Well"}


@dataclass
class TermValidation:
    name: str
    propagation_mode: str
    n_stations: int
    max_abs_diff: float
    max_rel_diff: float
    verdict: str
    note: str = ""


def _bindings_for_station(
    md_arr: np.ndarray, inc_deg_arr: np.ndarray, azi_deg_arr: np.ndarray,
    G: float, dip_deg: float, b_total: float, latitude_deg: float,
) -> dict:
    """Build the per-station variable namespace for formula evaluation."""
    inc_rad = np.radians(inc_deg_arr)
    azi_rad = np.radians(azi_deg_arr)
    dip_rad = np.radians(dip_deg)
    return {
        "MD": md_arr,
        "Inc": inc_rad,
        "AzT": azi_rad,
        "AzM": azi_rad,    # Copsegrove provides "Az (True)" only
        "Az": azi_rad,
        "TVD": md_arr,     # rough; Copsegrove doesn't report TVD per term
        "Gfield": G,
        "Dip": dip_rad,
        "BField": b_total,
        "EarthRate": 0.262516,
        "Latitude": np.radians(latitude_deg),
        "RAD": np.pi / 180.0,
    }


def validate_random_terms(
    json_path: str | Path, copsegrove_path: str | Path,
    *, latitude_deg: float = 60.0, b_total: float = 50000.0,
    dip_deg: float = 72.0, G: float = 9.80665,
    abs_tol: float = 5e-4,    # Copsegrove prints to 4 dp
) -> list[TermValidation]:
    """Compare JSON+interpreter dpde * mag to Copsegrove svy sigmas, term by term.

    Iterates over every term in the JSON model:

    - Propagation Random:  evaluate ``abs(dpde) × magnitude`` at every
      station and compare to Copsegrove's ``svy`` rows for that term.
    - Propagation Systematic / Global / Well:  flagged as DEFERRED.

    The Copsegrove file's environmental constants (latitude / B-field /
    dip / G) are taken from the kwargs above (defaulting to the ISCWSA
    Standard Test Well 1 values: 60 N / 50000 nT / 72° / 9.80665 m/s²).
    """
    with open(json_path) as f:
        model = json.load(f)
    stations = parse_copsegrove(copsegrove_path)

    md = np.array([s.md for s in stations])
    inc = np.array([s.inc_deg for s in stations])
    azi = np.array([s.azi_deg for s in stations])
    bindings = _bindings_for_station(md, inc, azi, G, dip_deg, b_total, latitude_deg)

    results: list[TermValidation] = []
    for term in model["terms"]:
        name = term["name"]
        mag = float(term["value"])
        prop = term["propagation_mode"]
        n = len(stations)

        if prop in _DEFERRED_MODES:
            results.append(TermValidation(
                name=name, propagation_mode=prop, n_stations=n,
                max_abs_diff=float("nan"), max_rel_diff=float("nan"),
                verdict="DEFERRED",
                note="systematic propagation requires welleng engine wiring",
            ))
            continue

        if prop not in _RANDOM_MODES:
            results.append(TermValidation(
                name=name, propagation_mode=prop, n_stations=n,
                max_abs_diff=float("nan"), max_rel_diff=float("nan"),
                verdict="UNKNOWN_PROP",
                note=f"unknown propagation mode {prop!r}",
            ))
            continue

        # Evaluate the interpreter
        try:
            d = evaluate_formula(term["depth_formula"], bindings)
            i = evaluate_formula(term["inclination_formula"], bindings)
            a = evaluate_formula(term["azimuth_formula"], bindings)
            d = np.broadcast_to(np.asarray(d, dtype=float), (n,))
            i = np.broadcast_to(np.asarray(i, dtype=float), (n,))
            a = np.broadcast_to(np.asarray(a, dtype=float), (n,))
        except Exception as exc:
            results.append(TermValidation(
                name=name, propagation_mode=prop, n_stations=n,
                max_abs_diff=float("nan"), max_rel_diff=float("nan"),
                verdict="INTERP_FAIL", note=str(exc),
            ))
            continue

        # interpreter sigmas in HLA frame (axis order matches Copsegrove)
        sig_AH_interp = np.abs(d) * mag      # depth_formula → A/H
        sig_HS_interp = np.abs(i) * mag      # inclination_formula → H/S
        sig_LAT_interp = np.abs(a) * mag     # azimuth_formula → LAT

        # Pull the Copsegrove sigmas for this term across all stations
        # that report it. Skip stations missing the term in Copsegrove.
        diffs: list[float] = []
        rel_diffs: list[float] = []
        compared = 0
        missing = 0
        for k, s in enumerate(stations):
            if name not in s.sigmas_svy:
                missing += 1
                continue
            cs_HS, cs_LAT, cs_AH = s.sigmas_svy[name]
            for v_interp, v_cs in (
                (sig_HS_interp[k], cs_HS),
                (sig_LAT_interp[k], cs_LAT),
                (sig_AH_interp[k], cs_AH),
            ):
                d_abs = abs(float(v_interp) - float(v_cs))
                diffs.append(d_abs)
                if abs(v_cs) > 1e-30:
                    rel_diffs.append(d_abs / abs(v_cs))
            compared += 1

        if compared == 0:
            results.append(TermValidation(
                name=name, propagation_mode=prop, n_stations=n,
                max_abs_diff=float("nan"), max_rel_diff=float("nan"),
                verdict="NOT_IN_COPSEGROVE",
                note=f"term {name!r} not present in Copsegrove file",
            ))
            continue

        max_abs = max(diffs)
        max_rel = max(rel_diffs) if rel_diffs else float("nan")
        if max_abs <= abs_tol:
            verdict = "MATCH"
        elif max_abs <= 5 * abs_tol:
            verdict = "NEAR_MATCH"
        else:
            verdict = "DIFFER"
        results.append(TermValidation(
            name=name, propagation_mode=prop, n_stations=compared,
            max_abs_diff=max_abs, max_rel_diff=max_rel,
            verdict=verdict,
        ))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


COPSEGROVE_DIR = (
    "/home/jonathancorcutt/Dropbox/Documents/Reference/"
    "Wellbore Uncertainty/OWSGSetA_Rev2"
)
JSON_DIR = Path(__file__).parent / "iscwsa_json" / "owsg_a"

# Pairings: (JSON file, matching Copsegrove .txt for ISCWSA Test Well 1)
DEFAULT_PAIRINGS = [
    (JSON_DIR / "GYRO-NS.json",     f"{COPSEGROVE_DIR}/ISCWSA1_OWSG_GYRO-NS.txt"),
    (JSON_DIR / "GYRO-MWD.json",    f"{COPSEGROVE_DIR}/ISCWSA1_OWSG_GYRO-MWD.txt"),
]


def main() -> None:
    print(f"Validating JSON+interpreter Random-prop terms against\n"
          f"Copsegrove .txt diagnostics (4-dp precision):\n")
    for json_path, txt_path in DEFAULT_PAIRINGS:
        if not os.path.isfile(txt_path):
            print(f"  ! Copsegrove file missing: {txt_path}")
            continue
        results = validate_random_terms(str(json_path), txt_path)
        print(f"\n--- {Path(json_path).stem} (vs {Path(txt_path).name}) ---")
        print(f"  {'term':<14s}  {'prop':<12s}  {'verdict':<22s}  "
              f"{'max |Δ|':>10s}  {'note':<60s}")
        counts: dict[str, int] = {}
        for r in results:
            counts[r.verdict] = counts.get(r.verdict, 0) + 1
            max_abs = "n/a" if np.isnan(r.max_abs_diff) else f"{r.max_abs_diff:.2e}"
            print(f"  {r.name:<14s}  {r.propagation_mode:<12s}  "
                  f"{r.verdict:<22s}  {max_abs:>10s}  {r.note[:60]}")
        s = "  ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  > {s}")


if __name__ == "__main__":
    main()
