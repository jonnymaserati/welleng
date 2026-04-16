"""Spike comparison: legacy hand-coded weight functions vs JSON+interpreter.

Loads the existing welleng MWD path on a Survey, then runs the same
Survey through the new ISCWSA-JSON-driven formula interpreter, and
reports the per-station per-axis dpde diff for each shared term.

A passing run (machine-precision agreement on all three terms) proves
the interpreter architecture works and we can scale up to full tool
models.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import welleng as we
from welleng.errors.interpreter import evaluate_formula
from welleng.errors.tool_errors import DREF, DSF, ABZ


# Build a representative welleng Survey — using the ISCWSA standard
# Test Well 1 from the existing test data.
def _make_test_survey():
    here = os.path.dirname(__file__)
    json_path = os.path.join(
        here, "..", "..", "test", "test_data",
        "error_mwdrev5_1_iscwsa_data.json",
    )
    with open(json_path) as f:
        data = json.load(f)
    h = data["header"]
    sv = data["survey"]
    sh = we.survey.SurveyHeader(
        name="iscwsa-test-1",
        latitude=h["latitude"],
        b_total=h["b_total"],
        dip=np.degrees(h["dip"]),
        declination=np.degrees(h["declination"]),
        convergence=h.get("convergence", 0.0),
        G=h["G"],
        azi_reference=h["azi_reference"],
    )
    return we.survey.Survey(
        md=np.array(sv["md"]),
        inc=np.degrees(np.array(sv["inc"])),
        azi=np.degrees(np.array(sv["azi"])),
        header=sh,
    )


def _bindings_from_survey(survey: we.survey.Survey) -> dict:
    """Map welleng Survey attributes to formula-namespace variables."""
    return {
        "MD": np.asarray(survey.md, dtype=float),
        "Inc": np.asarray(survey.inc_rad, dtype=float),
        "AzT": np.asarray(survey.azi_true_rad, dtype=float),
        "AzM": np.asarray(survey.azi_mag_rad, dtype=float),
        "Az": np.asarray(survey.azi_grid_rad, dtype=float),
        "TVD": np.asarray(survey.tvd, dtype=float),
        "Gfield": float(survey.header.G),
        "Dip": float(survey.header.dip),
        # Earth-rate and Latitude as scalars (used by gyro terms)
        "EarthRate": 0.262516,           # rad/hr (≈ 15.041 deg/hr)
        "Latitude": np.radians(float(survey.header.latitude or 0.0)),
    }


class _ShimError:
    """Minimal stand-in for the welleng Error object's per-term context.

    The legacy weight functions are written to expect a full ``Error``
    object with ``.survey``, ``.survey_rad``, etc. For the spike we
    only need the per-axis ``dpde`` returned by each function before
    propagation, so we capture that via a lightweight monkey-patch on
    ``_generate_error``.
    """
    def __init__(self, survey):
        self.survey = survey
        self.survey_rad = np.column_stack(
            [survey.md, survey.inc_rad, survey.azi_grid_rad]
        )
        self._captured: dict[str, np.ndarray] = {}

    def _generate_error(self, code, e_DIA, propagation, NEV):
        # Reverse the legacy multiplication to recover dpde so we can
        # compare apples-to-apples with the interpreter (which produces
        # dpde, multiplied by magnitude downstream).
        self._captured[code] = e_DIA
        return e_DIA  # downstream caller doesn't use the return here.


def main():
    survey = _make_test_survey()
    n = len(survey.md)
    print(f"Survey: {n} stations, MD ∈ [{survey.md[0]:.0f}, {survey.md[-1]:.0f}] m")

    # ---------- Legacy path ----------
    legacy_err = _ShimError(survey)
    DREF("DRFR", legacy_err, mag=0.35, propagation="random")
    DSF("DSFS", legacy_err, mag=0.00056, propagation="systematic")
    ABZ("ABZ",  legacy_err, mag=0.004, propagation="systematic")

    legacy_eDIA = {
        "DRFR": legacy_err._captured["DRFR"],
        "DSFS": legacy_err._captured["DSFS"],
        "ABZ":  legacy_err._captured["ABZ"],
    }

    # ---------- Interpreter path ----------
    spike_path = os.path.join(
        os.path.dirname(__file__),
        "iscwsa_json", "spike_three_terms.json",
    )
    with open(spike_path) as f:
        model = json.load(f)
    bindings = _bindings_from_survey(survey)

    interp_eDIA = {}
    for term in model["terms"]:
        name = term["name"]
        mag = term["value"]
        d = evaluate_formula(term["depth_formula"], bindings)
        i = evaluate_formula(term["inclination_formula"], bindings)
        a = evaluate_formula(term["azimuth_formula"], bindings)
        # Broadcast scalars to per-station arrays
        d = np.broadcast_to(np.asarray(d, dtype=float), (n,))
        i = np.broadcast_to(np.asarray(i, dtype=float), (n,))
        a = np.broadcast_to(np.asarray(a, dtype=float), (n,))
        dpde = np.column_stack([d, i, a])
        interp_eDIA[name] = dpde * mag

    # ---------- Compare ----------
    print(f"\n{'term':<6s}  {'shape':>10s}  "
          f"{'max |Δ|':>14s}  {'max rel Δ':>14s}  {'verdict':>20s}")
    print("-" * 80)
    for name in ("DRFR", "DSFS", "ABZ"):
        L = legacy_eDIA[name]
        I = interp_eDIA[name]
        diff = np.abs(L - I)
        denom = np.maximum(np.abs(L), 1e-30)
        rel = diff / denom
        max_abs = float(diff.max())
        max_rel = float(rel.max())
        verdict = ("MATCH (machine precision)" if max_abs < 1e-12
                   else f"DIFFER (max Δ = {max_abs:.2e})")
        print(f"{name:<6s}  {str(L.shape):>10s}  "
              f"{max_abs:>14.2e}  {max_rel:>14.2e}  {verdict:>20s}")

    print()


if __name__ == "__main__":
    main()
