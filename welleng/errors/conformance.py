"""Parallel-paths conformance harness — legacy hand-coded vs JSON+interpreter.

For one OWSG-converted JSON tool model, runs every term that exists in
*both* the legacy welleng dispatcher (``tool_errors.py`` function table)
and the new JSON path (interpreter against the JSON's formula strings),
on a shared welleng Survey, and reports per-term per-station agreement.

A clean run on every shared term across every model is the
quantitative foundation of the "welleng is the conformance suite for
the ISCWSA JSON schema" claim. Discrepancies are findings — see the
project's CLAUDE.md (``## In progress: ISCWSA JSON conformance
suite``) for the strategic context and what to do with them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import welleng as we

from .interpreter import evaluate_formula
from . import tool_errors as TE


PROP_MODE_REVERSE = {
    "Random": "random",
    "Systematic": "systematic",
    "Global": "global",
    "Well": "well",
}


@dataclass
class TermResult:
    """One row of the agreement matrix."""
    name: str
    legacy_function: str | None
    legacy_available: bool
    interp_available: bool
    max_abs_diff: float
    max_rel_diff: float
    n_stations: int
    note: str = ""


def _bindings_from_survey(survey) -> dict[str, Any]:
    """Variable namespace for formula evaluation."""
    return {
        "MD": np.asarray(survey.md, dtype=float),
        "Inc": np.asarray(survey.inc_rad, dtype=float),
        "AzT": np.asarray(survey.azi_true_rad, dtype=float),
        "AzM": np.asarray(survey.azi_mag_rad, dtype=float),
        "Az": np.asarray(survey.azi_grid_rad, dtype=float),
        "TVD": np.asarray(survey.tvd, dtype=float),
        "Gfield": float(survey.header.G),
        "Dip": float(survey.header.dip),
        "BField": float(survey.header.b_total or 50000.0),
        "EarthRate": 0.262516,            # rad/hr (~ 15.041 deg/hr)
        "Latitude": np.radians(float(survey.header.latitude or 0.0)),
        "RAD": np.pi / 180.0,             # used in some singularity formulas
    }


class _ShimError:
    """Minimal stand-in to capture the dpde array each legacy weight
    function would have passed downstream to ``Error._generate_error``.
    """
    def __init__(self, survey):
        self.survey = survey
        self.survey_rad = np.column_stack(
            [survey.md, survey.inc_rad, survey.azi_grid_rad]
        )
        self._captured: dict[str, np.ndarray] = {}

    def _generate_error(self, code, e_DIA, propagation, NEV):
        self._captured[code] = e_DIA
        return e_DIA


def _legacy_eval_term(term_name: str, legacy_func_name: str, mag: float,
                      prop_mode: str, survey) -> np.ndarray | None:
    """Look the legacy weight function up by name and call it."""
    func = getattr(TE, legacy_func_name, None)
    if func is None:
        return None
    shim = _ShimError(survey)
    try:
        func(term_name, shim, mag=mag, propagation=prop_mode, NEV=True)
    except Exception:
        return None
    return shim._captured.get(term_name)


def _interp_eval_term(term: dict, mag: float, survey, bindings: dict) -> np.ndarray:
    """Evaluate the JSON term's three formula axes against the survey."""
    n = len(survey.md)
    d = evaluate_formula(term["depth_formula"], bindings)
    i = evaluate_formula(term["inclination_formula"], bindings)
    a = evaluate_formula(term["azimuth_formula"], bindings)
    d = np.broadcast_to(np.asarray(d, dtype=float), (n,))
    i = np.broadcast_to(np.asarray(i, dtype=float), (n,))
    a = np.broadcast_to(np.asarray(a, dtype=float), (n,))
    return np.column_stack([d, i, a]) * mag


def compare_model(json_path: str, survey, *,
                  rtol: float = 1e-9, atol: float = 1e-12) -> list[TermResult]:
    """Run a parallel-paths comparison on every term in the JSON model.

    Returns one ``TermResult`` per term. Terms that don't have a legacy
    weight function in welleng's dispatcher are still included with
    ``legacy_available=False`` -- the absence is itself a finding (it
    means welleng has no implementation yet, or the term was added in
    a tool model variant we haven't built coverage for).
    """
    with open(json_path) as f:
        model = json.load(f)
    bindings = _bindings_from_survey(survey)
    n = len(survey.md)

    results: list[TermResult] = []
    for term in model["terms"]:
        name = term["name"]
        mag = float(term["value"])
        prop_mode = PROP_MODE_REVERSE.get(term["propagation_mode"], "systematic")
        # Prefer the OWSG-stamped weight-function name (column 'Wt.Fn.'
        # from the source xlsx); fall back to deriving from the term
        # code (Code -> Wt.Fn. equivalence holds for most simple terms).
        wt_fn = term.get("x_owsg_weight_function") or name
        legacy_fn_name = wt_fn.replace("-", "_")

        try:
            interp_eDIA = _interp_eval_term(term, mag, survey, bindings)
            interp_ok = True
        except Exception as exc:
            results.append(TermResult(
                name=name, legacy_function=legacy_fn_name,
                legacy_available=False, interp_available=False,
                max_abs_diff=float("nan"), max_rel_diff=float("nan"),
                n_stations=n, note=f"interpreter error: {exc}",
            ))
            continue

        legacy_eDIA = _legacy_eval_term(name, legacy_fn_name, mag, prop_mode, survey)
        if legacy_eDIA is None:
            results.append(TermResult(
                name=name, legacy_function=legacy_fn_name,
                legacy_available=False, interp_available=interp_ok,
                max_abs_diff=float("nan"), max_rel_diff=float("nan"),
                n_stations=n, note="no legacy weight function in tool_errors.py",
            ))
            continue

        diff = np.abs(legacy_eDIA - interp_eDIA)
        denom = np.maximum(np.abs(legacy_eDIA), 1e-30)
        rel = diff / denom
        results.append(TermResult(
            name=name, legacy_function=legacy_fn_name,
            legacy_available=True, interp_available=interp_ok,
            max_abs_diff=float(diff.max()),
            max_rel_diff=float(rel.max()),
            n_stations=n,
        ))
    return results


# ---------------------------------------------------------------------------
# Standard test survey + CLI
# ---------------------------------------------------------------------------


def standard_test_survey():
    """ISCWSA Test Well 1 with the standard environmental setup."""
    here = Path(__file__).parent
    json_path = here.parent.parent / "test" / "test_data" / "error_mwdrev5_1_iscwsa_data.json"
    with open(json_path) as f:
        d = json.load(f)
    h = d["header"]
    sv = d["survey"]
    sh = we.survey.SurveyHeader(
        name="iscwsa-test-1",
        latitude=h["latitude"], b_total=h["b_total"],
        dip=np.degrees(h["dip"]), declination=np.degrees(h["declination"]),
        convergence=h.get("convergence", 0.0),
        G=h["G"], azi_reference=h["azi_reference"],
    )
    return we.survey.Survey(
        md=np.array(sv["md"]),
        inc=np.degrees(np.array(sv["inc"])),
        azi=np.degrees(np.array(sv["azi"])),
        header=sh,
    )


def _classify(r: TermResult) -> str:
    if not r.interp_available:
        return "INTERP_FAILED"
    if not r.legacy_available:
        return "NO_LEGACY"
    if r.max_abs_diff < 1e-10:
        return "MATCH"
    if r.max_abs_diff < 1e-6:
        return "NEAR_MATCH"
    return "DIFFER"


def _print_one_model(json_path: str, results: list[TermResult]) -> dict[str, int]:
    print(f"\n--- {Path(json_path).name} ---")
    print(f"  {'term':<14s}  {'wt fn':<14s}  {'verdict':<22s}  "
          f"{'max |Δ|':>10s}  {'note':<55s}")
    counts: dict[str, int] = {}
    for r in results:
        c = _classify(r)
        counts[c] = counts.get(c, 0) + 1
        max_abs = "n/a" if c in ("INTERP_FAILED", "NO_LEGACY") else f"{r.max_abs_diff:.2e}"
        print(f"  {r.name:<14s}  {r.legacy_function or '-':<14s}  "
              f"{c:<22s}  {max_abs:>10s}  {r.note[:55] if r.note else ''}")
    summary = "  ".join(f"{k}={v}" for k, v in counts.items())
    print(f"  > {summary}  (of {len(results)} terms)")
    return counts


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--json",
        help="single ISCWSA JSON tool model to compare against legacy welleng",
    )
    p.add_argument(
        "--summary", action="store_true",
        help="run across the canonical ISCWSA Rev 5.11 + 3 gyro tools and "
             "report the aggregate coverage matrix",
    )
    args = p.parse_args()

    survey = standard_test_survey()
    print(f"Survey: ISCWSA Test Well 1, {len(survey.md)} stations")

    if not args.summary and not args.json:
        args.summary = True

    if args.summary:
        # The four canonical tool models: one MWD (Rev 5.11), three gyro
        # variants (stationary, mixed, gyro-MWD). These are the gating
        # set for the conformance harness's "robust story".
        models = [
            "owsg_a/MWD+SRGM.json",      # ISCWSA MWD Rev 5.11 canonical
            "owsg_a/GYRO-NS.json",       # north-seeking gyro (stationary)
            "owsg_a/GYRO-NS-CT.json",    # mixed stationary + continuous
            "owsg_a/GYRO-MWD.json",      # gyro-MWD hybrid
        ]
        json_root = Path(__file__).parent / "iscwsa_json"
        agg: dict[str, int] = {}
        per_model: list[tuple[str, dict[str, int]]] = []
        # Track unique findings (term, error message) so the post can
        # quote each gap once even if it appears across multiple tools.
        findings: dict[str, list[str]] = {}
        for rel in models:
            path = str(json_root / rel)
            results = compare_model(path, survey)
            counts = _print_one_model(path, results)
            per_model.append((rel, counts))
            for k, v in counts.items():
                agg[k] = agg.get(k, 0) + v
            for r in results:
                if not r.interp_available and r.note:
                    findings.setdefault(r.note, []).append(f"{rel}::{r.name}")

        print("\n" + "=" * 80)
        print("AGGREGATE COVERAGE MATRIX")
        print("=" * 80)
        print(f"  {'tool':<32s}  MATCH  NEAR  DIFFER  INTERP_FAIL  NO_LEGACY  TOTAL")
        for rel, c in per_model:
            tot = sum(c.values())
            print(f"  {Path(rel).stem:<32s}  "
                  f"{c.get('MATCH', 0):>5d}  "
                  f"{c.get('NEAR_MATCH', 0):>4d}  "
                  f"{c.get('DIFFER', 0):>6d}  "
                  f"{c.get('INTERP_FAILED', 0):>11d}  "
                  f"{c.get('NO_LEGACY', 0):>9d}  "
                  f"{tot:>5d}")
        total = sum(agg.values())
        print(f"  {'TOTAL':<32s}  "
              f"{agg.get('MATCH', 0):>5d}  "
              f"{agg.get('NEAR_MATCH', 0):>4d}  "
              f"{agg.get('DIFFER', 0):>6d}  "
              f"{agg.get('INTERP_FAILED', 0):>11d}  "
              f"{agg.get('NO_LEGACY', 0):>9d}  "
              f"{total:>5d}")

        if findings:
            print("\n" + "=" * 80)
            print("INTERPRETER-FAILURE FINDINGS (schema gaps surfaced)")
            print("=" * 80)
            for note, where in sorted(findings.items()):
                print(f"\n  [{note}]")
                for w in where[:8]:
                    print(f"    {w}")
                if len(where) > 8:
                    print(f"    ... and {len(where) - 8} more")
        return

    results = compare_model(args.json, survey)
    _print_one_model(args.json, results)


if __name__ == "__main__":
    main()
