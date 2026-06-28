"""Guard: the shipped OWSG JSON models must be EXACTLY the generator's output.

``welleng/errors/iscwsa_json/owsg_{a,b}/*.json`` are generated artefacts, parsed
from the ISCWSA-issued OWSG spreadsheets by ``owsg_to_json.py``. They must never be
hand-edited: a past commit (6885334) hijacked ``GYRO-NS-CT.json``, overwriting its
ISCWSA sheet parameters (running speed 2743.2 m/hr, static-gyro gate 15 deg, NRF
0.4082) with SPE-90408 paper values (2160 / 17 deg / 0.5) just to satisfy a
validation test. That corrupts the authoritative model for every other consumer.

This test regenerates the models from the vendored source spreadsheets and asserts
they match what is committed. Any manual edit -- a "convenient" parameter override,
a hand-tweaked formula -- fails here. If a test needs different parameter values for
its own validation, it must override them LOCALLY, not mutate the shipped model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from welleng.errors.tools.owsg_to_json import (
    DEFAULT_XLSX_A,
    DEFAULT_XLSX_B,
    convert_workbook,
)

JSON_ROOT = Path(__file__).parent.parent / "welleng" / "errors" / "iscwsa_json"


@pytest.mark.parametrize(
    "label,xlsx",
    [("a", DEFAULT_XLSX_A), ("b", DEFAULT_XLSX_B)],
    ids=["owsg_a", "owsg_b"],
)
def test_shipped_json_matches_generator(label, xlsx, tmp_path):
    if not Path(xlsx).is_file():
        pytest.skip(f"source spreadsheet not vendored: {xlsx}")

    out_dir = tmp_path / f"owsg_{label}"
    convert_workbook(str(xlsx), str(out_dir), f"Set{label.upper()}")
    committed_dir = JSON_ROOT / f"owsg_{label}"

    mismatches: list[str] = []
    for generated in sorted(out_dir.glob("*.json")):
        committed = committed_dir / generated.name
        if not committed.exists():
            mismatches.append(f"{generated.name}: generated but not committed")
            continue
        if json.loads(generated.read_text()) != json.loads(committed.read_text()):
            mismatches.append(
                f"{generated.name}: differs from generator output (hand-edited?)"
            )

    assert not mismatches, (
        "Shipped OWSG JSON diverges from owsg_to_json output. These are GENERATED "
        "from the ISCWSA-issued spreadsheets -- do NOT hand-edit them. Regenerate "
        "with `python -m welleng.errors.tools.owsg_to_json`, or, if a test needs "
        "different parameters, override them locally in the test:\n  "
        + "\n  ".join(mismatches)
    )
