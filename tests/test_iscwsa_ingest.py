"""Round-trip tests for the ISCWSA rev 5.1 example workbook ingester.

The reference workbooks live outside the repo (gitignored under
``data/iscwsa/``); tests skip cleanly when they are not present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

openpyxl = pytest.importorskip("openpyxl")

from welleng.errors.iscwsa_ingest import (  # noqa: E402
    ISCWSAExample,
    load_iscwsa_example,
    to_legacy_codes_dict,
)


DATA_DIR = Path(__file__).parent.parent / "data" / "iscwsa" / "rev5_11"
EXAMPLES = [DATA_DIR / f"error-model-example-mwdrev5-1-iscwsa-{n}.xlsx" for n in (1, 2, 3)]
AVAILABLE = [p for p in EXAMPLES if p.exists()]

pytestmark = pytest.mark.skipif(
    not AVAILABLE, reason="ISCWSA rev 5.1 example workbooks not checked in"
)


@pytest.fixture(scope="module", params=AVAILABLE, ids=lambda p: p.stem[-1])
def example(request) -> ISCWSAExample:
    return load_iscwsa_example(request.param)


def test_header_populated(example):
    h = example.header
    assert h.short_name  # populated
    assert float(h.revision_no) in (pytest.approx(5.0), pytest.approx(5.1))


def test_workbook_revision_from_readme(example):
    # Authoritative example-package rev is in Readme!A9; the per-tool
    # Revision No in the Model sheet is a separate (often stale) field.
    assert example.revision == "Revision 5.11"


def test_all_terms_parsed(example):
    assert len(example.terms) == 35
    codes = {t.code for t in example.terms}
    # rev 5 renames / additions
    assert {"XCLA", "XCLH", "SAGE", "XYM3E", "XYM4E"} <= codes
    # No blank codes or zero magnitudes
    for t in example.terms:
        assert t.code and t.wt_fn
        assert np.isfinite(t.magnitude)
        assert t.propagation in {"R", "S", "G", "W"}


def test_wellpath_and_totals_aligned(example):
    n_w = example.wellpath.stations.shape[0]
    n_t = example.totals.md.shape[0]
    assert n_w == n_t
    assert np.array_equal(example.wellpath.stations[:, 0], example.totals.md)


def test_reference_parameters(example):
    wp = example.wellpath
    assert np.isfinite(wp.latitude_deg)
    assert wp.g == pytest.approx(9.80665)
    assert wp.btotal > 30_000  # nT, any realistic magnetic field
    assert np.isfinite(wp.dip_deg)
    assert np.isfinite(wp.declination_deg)


def test_totals_final_station_nonzero(example):
    # Last MD row should have accumulated non-trivial covariance
    assert np.any(example.totals.nev[-1] > 0)
    assert np.any(example.totals.hla[-1] > 0)


def test_every_term_has_source_tab(example):
    # Every error term listed in the Model sheet should have a matching
    # per-source tab that we could parse.
    codes = {t.code for t in example.terms}
    missing = codes - set(example.sources)
    assert not missing, f"no per-source sheet for: {missing}"


def test_source_tab_xcla_blocks(example):
    xcla = example.sources["XCLA"]
    # dpde / e / estar / Sigma / Covariance block headers
    assert {"dpde", "e", "estar"} <= set(xcla.blocks)
    assert set(xcla.blocks["dpde"]) == {"D", "I", "A"}
    assert set(xcla.blocks["e"]) == {"N", "E", "V"}
    assert xcla.md.shape == example.totals.md.shape


def test_legacy_dict_shape(example):
    legacy = to_legacy_codes_dict(example)
    assert set(legacy) == {"header", "codes"}
    assert len(legacy["codes"]) == 35
    abz = legacy["codes"]["ABZ"]
    assert abz == {
        "function": "ABZ",
        "magnitude": pytest.approx(0.004),
        "unit": "m/s2",
        "propagation": "systematic",
    }
    # deg -> rad conversion preserved from the old extractor
    xym1 = legacy["codes"]["XYM1"]
    assert xym1["unit"] == "rad"
    assert xym1["magnitude"] == pytest.approx(np.radians(0.1), rel=1e-6)
