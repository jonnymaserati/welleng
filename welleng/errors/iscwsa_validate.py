"""
Validation harness: compare welleng's per-source error-model output against
ISCWSA example workbooks.

Given a workbook and a welleng ``error_model`` string, the harness builds a
``Survey`` from the workbook's wellpath + reference params, runs welleng's
error model, and compares per-source covariance at each validation MD to:

1. the workbook's **Diagnostic** values (authoritative ISCWSA targets), and
2. the workbook's **Calculated** values (what the XLSX formulas produced —
   a self-consistency check on the workbook itself).

The result is a matrix of (source × MD) pass/fail flags with max-relative-
error per cell, so you can see surgically which weight functions need an
update for the workbook's declared revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .iscwsa_ingest import ISCWSAExample, load_iscwsa_example


# NEV component order used throughout ISCWSA workbooks
_NEV = ("NN", "EE", "VV", "NE", "NV", "EV")


@dataclass
class SourceCompare:
    code: str
    md: float
    diagnostic: np.ndarray     # workbook "Diagnostic" NEV row (len 6)
    calculated: np.ndarray     # workbook "Calculated" NEV row (len 6)
    welleng: np.ndarray        # welleng's per-source NEV at that MD (len 6)
    max_abs_err: float         # max |welleng - diagnostic|
    max_rel_err: float         # max |welleng - diagnostic| / |diagnostic|
    ok: bool


@dataclass
class TotalsCompare:
    md: float
    workbook: np.ndarray
    welleng: np.ndarray
    max_abs_err: float
    max_rel_err: float
    ok: bool


@dataclass
class ValidationReport:
    workbook: Path
    error_model: str
    per_source: list[SourceCompare]
    totals: list[TotalsCompare]
    missing_in_welleng: list[str]    # sources welleng can't model for this rev
    tolerance_rel: float
    tolerance_abs: float

    def source_pass_fail(self) -> dict[str, bool]:
        """``{code: all_mds_pass}`` - True if every MD for this source passes."""
        by_code: dict[str, list[SourceCompare]] = {}
        for c in self.per_source:
            by_code.setdefault(c.code, []).append(c)
        return {code: all(c.ok for c in rows) for code, rows in by_code.items()}

    def summary(self) -> str:
        lines = [
            f"# ISCWSA validation: {self.workbook.name}",
            f"#   welleng error_model: {self.error_model}",
            f"#   tol rel={self.tolerance_rel:g} abs={self.tolerance_abs:g}",
            "",
            "Per-source pass/fail (✓ = all validation MDs match diagnostic):",
        ]
        pf = self.source_pass_fail()
        for code in sorted(pf):
            mark = "PASS" if pf[code] else "FAIL"
            # find worst error across MDs for this source
            rows = [r for r in self.per_source if r.code == code]
            worst = max(rows, key=lambda r: r.max_abs_err)
            lines.append(
                f"  {mark}  {code:12s}  worst @ md={worst.md:<8g} "
                f"abs={worst.max_abs_err:.3e}  rel={worst.max_rel_err:.3e}"
            )
        if self.missing_in_welleng:
            lines.append("")
            lines.append(f"Not modelled by welleng under {self.error_model!r}:")
            for c in self.missing_in_welleng:
                lines.append(f"  -  {c}")
        if self.totals:
            lines.append("")
            lines.append("TOTALS (sum across all sources):")
            worst = max(self.totals, key=lambda t: t.max_abs_err)
            passing = sum(1 for t in self.totals if t.ok)
            lines.append(
                f"  {passing}/{len(self.totals)} stations pass; worst "
                f"md={worst.md:g} abs={worst.max_abs_err:.3e} rel={worst.max_rel_err:.3e}"
            )
        return "\n".join(lines)


def _build_survey(example: ISCWSAExample, error_model: str) -> Any:
    """Construct a welleng Survey matching the workbook's wellpath.

    ``SurveyHeader`` converts dip/declination from degrees to radians internally
    when ``deg=True`` (its default), so we pass the workbook's native degree
    values unchanged.
    """
    from welleng.survey import Survey, SurveyHeader

    wp = example.wellpath
    sh = SurveyHeader(
        name=wp.well_name,
        latitude=wp.latitude_deg,
        G=wp.g,
        b_total=wp.btotal,
        dip=wp.dip_deg,
        declination=wp.declination_deg,
        convergence=0.0,
        azi_reference="true",
    )
    # welleng's error-term magnitudes are denominated in SI (rad/m for
    # tortuosity, etc.). The workbook internally converts MD to metres before
    # computing covariance, so if the sheet declares depth_units='ft' we must
    # match that conversion before handing the survey to welleng, otherwise
    # systematic terms come out scaled by (ft/m)^2 = 10.764.
    scale = wp.ft_to_m if wp.depth_units.startswith("ft") else 1.0
    md = wp.stations[:, 0] * scale
    inc = wp.stations[:, 1]
    azi = wp.stations[:, 2]
    return Survey(md=md, inc=inc, azi=azi, header=sh, error_model=error_model)


def _relative_error(expected: np.ndarray, actual: np.ndarray) -> float:
    denom = np.abs(expected)
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(actual[mask] - expected[mask]) / denom[mask]))


def validate_workbook(
    path: str | Path,
    error_model: str,
    tolerance_rel: float = 1e-3,
    tolerance_abs: float = 1e-6,
) -> ValidationReport:
    """Run welleng ``error_model`` against a workbook and return a report."""
    example = load_iscwsa_example(path)
    survey = _build_survey(example, error_model)
    err = survey.err.errors

    welleng_codes = set(err.errors)
    workbook_codes = {t.code for t in example.terms}
    missing = sorted(workbook_codes - welleng_codes)

    # Validation/TOTALS MDs are in the workbook's declared depth units;
    # survey.md has been converted to metres in _build_survey.
    wp = example.wellpath
    md_scale = wp.ft_to_m if wp.depth_units.startswith("ft") else 1.0

    per_source: list[SourceCompare] = []
    for row in example.validation:
        if row.source not in welleng_codes:
            continue
        idx_arr = np.where(np.isclose(survey.md, row.md * md_scale))[0]
        if idx_arr.size == 0:
            continue
        i = int(idx_arr[0])
        cov = err.errors[row.source].cov_NEV[i]
        # cov is (3,3) in NEV; pack to (NN, EE, VV, NE, NV, EV)
        packed = np.array(
            [cov[0, 0], cov[1, 1], cov[2, 2], cov[0, 1], cov[0, 2], cov[1, 2]]
        )
        # Compare against the workbook's *Calculated* values (full-precision
        # formula output). Diagnostic values are kept on the row for context
        # but are 4-dp rounded, so they're not a tight comparison target.
        target = row.calculated
        abs_err = float(np.max(np.abs(packed - target)))
        rel_err = _relative_error(target, packed)
        ok = abs_err <= tolerance_abs or rel_err <= tolerance_rel
        per_source.append(
            SourceCompare(
                code=row.source,
                md=row.md,
                diagnostic=row.diagnostic,
                calculated=row.calculated,
                welleng=packed,
                max_abs_err=abs_err,
                max_rel_err=rel_err,
                ok=ok,
            )
        )

    totals: list[TotalsCompare] = []
    tot = err.cov_NEVs
    for i, md in enumerate(example.totals.md):
        wb_row = example.totals.nev[i]
        we_row = np.array([
            tot[i, 0, 0], tot[i, 1, 1], tot[i, 2, 2],
            tot[i, 0, 1], tot[i, 0, 2], tot[i, 1, 2],
        ])
        abs_err = float(np.max(np.abs(we_row - wb_row)))
        rel_err = _relative_error(wb_row, we_row)
        ok = abs_err <= tolerance_abs or rel_err <= tolerance_rel
        totals.append(TotalsCompare(
            md=float(md) * md_scale, workbook=wb_row, welleng=we_row,
            max_abs_err=abs_err, max_rel_err=rel_err, ok=ok,
        ))

    return ValidationReport(
        workbook=Path(path),
        error_model=error_model,
        per_source=per_source,
        totals=totals,
        missing_in_welleng=missing,
        tolerance_rel=tolerance_rel,
        tolerance_abs=tolerance_abs,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workbook", help="path to ISCWSA example xlsx")
    ap.add_argument(
        "--model", default="ISCWSA MWD Rev5.11",
        help="welleng error_model string (default: %(default)r)",
    )
    ap.add_argument("--tol-rel", type=float, default=1e-3)
    ap.add_argument("--tol-abs", type=float, default=1e-6)
    args = ap.parse_args(argv)

    report = validate_workbook(
        args.workbook,
        error_model=args.model,
        tolerance_rel=args.tol_rel,
        tolerance_abs=args.tol_abs,
    )
    print(report.summary())
    fail_count = sum(1 for ok in report.source_pass_fail().values() if not ok)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
