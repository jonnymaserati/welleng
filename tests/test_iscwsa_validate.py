"""Smoke tests for the ISCWSA validation harness.

These do not assert pass/fail for any given error source — they lock in the
machinery of the harness (ingester + welleng model + comparison) so that
regressions in the harness itself are caught. Interpretation of which sources
pass or fail on which workbook is intentionally left as reporting output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

openpyxl = pytest.importorskip("openpyxl")

from welleng.errors.iscwsa_validate import validate_workbook  # noqa: E402


DATA = Path(__file__).parent.parent / "data" / "iscwsa" / "rev5_11"
EXAMPLES = [DATA / f"error-model-example-mwdrev5-1-iscwsa-{n}.xlsx" for n in (1, 2, 3)]
AVAILABLE = [p for p in EXAMPLES if p.exists()]

pytestmark = pytest.mark.skipif(
    not AVAILABLE, reason="ISCWSA rev 5.11 example workbooks not checked in"
)


@pytest.mark.parametrize("path", AVAILABLE, ids=lambda p: p.stem[-1])
def test_harness_runs_end_to_end(path):
    report = validate_workbook(path, error_model="ISCWSA MWD Rev5.11")
    # Harness produced per-source comparisons for every validation row whose
    # source is modelled by welleng.
    assert report.per_source, "expected at least one comparison row"
    # TOTALS comparison produced one entry per survey station.
    assert report.totals, "expected TOTALS comparisons"
    # Every welleng output is finite.
    for c in report.per_source:
        assert np.all(np.isfinite(c.welleng))
    # Summary string is well-formed and contains the workbook name.
    s = report.summary()
    assert path.name in s
    assert "PASS" in s or "FAIL" in s


def test_example1_known_passes():
    """Example #1 should match welleng's existing Rev5 model to float precision
    for the vast majority of error sources. Locks in the established baseline
    that the existing ``test_iscwsa_mwd_error`` also depends on.
    """
    p = DATA / "error-model-example-mwdrev5-1-iscwsa-1.xlsx"
    if not p.exists():
        pytest.skip("example 1 not present")
    report = validate_workbook(p, error_model="ISCWSA MWD Rev5.11",
                               tolerance_rel=1e-4, tolerance_abs=1e-10)
    pf = report.source_pass_fail()
    # At least 30 of 35 sources should match to float precision on Example #1.
    # The remaining handful (DEC-OH in particular) are the known-to-investigate
    # cases surfaced by this harness — do not tighten this bound without
    # reviewing those investigations.
    passing = sum(1 for ok in pf.values() if ok)
    assert passing >= 30, f"only {passing}/{len(pf)} sources pass on Example #1"
