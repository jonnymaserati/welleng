"""Absolute validation of the gyro error model against SPE 90408 Appendix E.

SPE 90408-MS (Torkildsen et al. 2004) Appendix E publishes position
covariance matrices for six synthetic "Example Models" on the three ISCWSA
standard well bores, with a stated acceptance criterion: a correct
implementation agrees to within +/-1% of every tabulated value (or +/-2 units
where the value is < 200), "verified within these limits by independent
implementations".

This pins the two implemented Example Models on the ISCWSA standard wells:
  - **Model #1** XY accelerometer + XY stationary gyro (0-150 deg).
  - **Model #3** XYZ accel + XY static gyro (0-17) + XY continuous gyro
    (17-150) -- the hybrid, exercising inc gating + the stationary->continuous
    initialisation-seed carry (App. C Fig C1, boxes 9/12).

Unlike the conformance harness (which compares the JSON interpreter against
the legacy welleng weight functions and so cancels any common scale error),
this checks welleng's gyro output against absolute, externally-published
numbers -- exercising the vertical-singularity substitution, the depth-term
propagation, and the continuous init-seed.

Coverage / known gaps (xfail, see XFAIL below), per ISCWSA "Test Profile
Differences" (Copsegrove/Grindrod CDR-SM-03, 2020):
  - **True-vs-grid north**: Appendix E (gyro) treats the survey azimuth
    *numbers* as GRID and reports covariance in the GRID frame. Wells #2/#3
    have real UTM convergence (15N / 55S); Well #1 has ~0. We feed azimuth
    as true (convergence 0), so Wells #2/#3 pick up a frame offset. Verified:
    applying ~1 deg convergence (azi_true = grid + conv) + rotating the output
    to grid closes Well #2 Model #1 to 0.9%. We don't have the authoritative
    per-well convergence in-repo, so those cells are xfail(non-strict).
  - **Continuous re-initialisation**: Well #3 builds, drops back to vertical
    (inc=0 at 2460 m), then rebuilds -- requiring re-initialisation of the
    continuous gyro (App. C box 13 / min_D). Not implemented; the single
    init-seed carry over-predicts after the second build. xfail.

Model config + magnitudes: SPE 90408-MS Appendix D (D1-D7). Weight functions:
Tables 1/2 (accel), 3/4 (gyro), 6/7 (continuous), 9 (misalignment Alt.3),
10 (depth). Well geometry: Well #1 from the MWD test JSON; Wells #2/#3 from
the ISCWSA diagnostics .dat files. Fixtures:
``tests/test_data/spe90408_example_models/example_{1,3}.json``.

See ``docs/dev/VALIDATION.md`` for the repo-wide validation catalogue and the
full known-differences list.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import welleng as we
from welleng.survey import Survey, SurveyHeader
from welleng.errors.tool_errors import _json_to_em_adapter

DATA = Path(__file__).parent / "test_data"
DIAG = Path(__file__).parents[1] / "data" / "iscwsa" / "diagnostics"
FT = 0.3048          # foot -> metre
FT2 = FT * FT        # ft^2 -> m^2

# Paper's acceptance: within +/-1%, or +/-2 units where |value| < 200.
REL_TOL = 0.01
ABS_TOL_SMALL = 2.0
SMALL = 200

# --- SPE 90408-MS Appendix E reference covariances [NN, NE, NV, EE, EV, VV] ---
# Well #1 (Table E1) in m / m^2.
REF = {
    ("well1", "model_1"): {
        1200: [19, 0, 0, 18, 0, 2],
        2100: [1439, -373, -2, 144, -6, 9],
        5100: [134488, -35992, -18, 9807, -49, 137],
    },
    ("well1", "model_3"): {
        1200: [19, 0, 0, 18, 0, 2],
        2100: [923, -235, -2, 106, -6, 9],
        5100: [45445, -12136, -13, 3408, -38, 120],
        5400: [54685, -14608, -14, 4085, -39, 136],
        8000: [181521, -48567, -16, 13296, -38, 289],
    },
    # Well #2 (Table E2) in ft / ft^2 (checkpoints in ft).
    ("well2", "model_1"): {
        2000: [53, 0, 0, 49, 0, 10],
        5000: [278, -71, -55, 2293, -2, 54],
        7102: [1523, -858, -106, 4348, -48, 130],
        9398: [3209, -2483, -70, 11754, -66, 222],
        12500: [22661, -26724, -25, 42336, -23, 481],
    },
    ("well2", "model_3"): {
        2000: [53, 0, 0, 49, 0, 10],
        5000: [277, -75, -55, 2419, -2, 54],
        7102: [1725, -2187, -105, 4543, -48, 129],
        9398: [1540, -301, -68, 1133, -65, 218],
        12500: [3688, -4789, -31, 9211, -29, 441],
    },
    # Well #3 (Table E3) in m / m^2. (Model #1 blank past 3000 m -- pure
    # stationary diverges as inc -> 90 deg; those checkpoints omitted.)
    ("well3", "model_1"): {
        1110: [13, 0, -2, 134, 0, 2],
        2460: [51, 0, -18, 1981, 0, 20],
        3000: [118, 43, -22, 2027, 1, 30],
    },
    ("well3", "model_3"): {
        1110: [13, 0, -2, 142, 0, 2],
        2460: [50, 0, -17, 2179, 0, 20],
        3000: [92, 41, -21, 2226, 1, 30],
        3720: [1229, 318, -25, 2202, -1, 40],
        4030: [1447, -147, -27, 2129, -1, 43],
    },
}

FIXTURE = {"model_1": "example_1.json", "model_3": "example_3.json"}

# Combos not yet within band -- documented frame / re-initialisation gaps
# (see module docstring). xfail non-strict: passing cells are fine too.
XFAIL = {
    ("well2", "model_1"): "true-vs-grid north: App. E azimuths are grid; "
                          "Well #2 (UTM 15N) convergence ~1 deg not applied "
                          "(closes to 0.9% with it). CDR-SM-03.",
    ("well3", "model_1"): "Well #3 build-drop-rebuild needs continuous "
                          "re-initialisation (App. C box 13 / min_D), not "
                          "implemented; + grid/true (UTM 55S).",
    ("well3", "model_3"): "Well #3 re-initialisation (App. C box 13 / min_D) "
                          "not implemented; + grid/true (UTM 55S).",
}


def _build_well1() -> Survey:
    """ISCWSA Standard Test Well #1 from the raw MWD test JSON.

    inc/azi are stored in *degrees* (max 90 / 75); header angles in radians.
    (The conformance helper ``standard_test_survey()`` double-converts inc via
    ``np.degrees`` -> a 90-radian well; don't use it for an absolute check.)
    """
    d = json.loads((DATA / "error_mwdrev5_1_iscwsa_data.json").read_text())
    sv, h = d["survey"], d["header"]
    sh = SurveyHeader(
        name="iscwsa-1", latitude=h["latitude"], b_total=h["b_total"],
        dip=np.degrees(h["dip"]), declination=np.degrees(h["declination"]),
        convergence=np.degrees(h.get("convergence", 0.0)),
        G=h["G"], azi_reference=h["azi_reference"],
        earth_rate=h.get("earth_rate"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Survey(md=np.array(sv["md"]), inc=np.array(sv["inc"]),
                      azi=np.array(sv["azi"]), header=sh)


def _build_well_from_dat(num: int) -> Survey:
    """ISCWSA Standard Test Well #2/#3 geometry + reference params from the
    diagnostics .dat. The per-station ``SURVEY MD INC AZT TF TVD MODE`` lines
    give md/inc/azt (degrees); the MWD covariances in the file are ignored.
    azimuth is true-referenced (AZT)."""
    txt = (DIAG / f"ErrorModelDiagnostics_Rev5-1_ISCWSA#{num}.dat").read_text()
    m = re.search(
        r"Latitude:\s*([-\d.]+).*BTotal:\s*([-\d.]+).*Dip:\s*([-\d.]+)"
        r".*Declination:\s*([-\d.]+).*GTotal:\s*([-\d.]+)", txt)
    lat, b, dip, decl, g = map(float, m.groups())
    rows = []
    for ln in txt.splitlines():
        if ln.startswith("SURVEY"):
            f = ln.split()
            if len(f) >= 6:
                try:
                    rows.append((float(f[1]), float(f[2]), float(f[3])))
                except ValueError:
                    continue
    rows = np.array(rows)
    sh = SurveyHeader(
        name=f"iscwsa-{num}", latitude=lat, b_total=b, dip=dip,
        declination=decl, convergence=0.0, G=g, azi_reference="true")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Survey(md=rows[:, 0], inc=rows[:, 1], azi=rows[:, 2], header=sh)


# well -> (builder, depth_to_m, cov_to_ref_unit). Well #2 is reported in feet.
WELLS = {
    "well1": (_build_well1, 1.0, 1.0),
    "well2": (lambda: _build_well_from_dat(2), FT, 1.0 / FT2),
    "well3": (lambda: _build_well_from_dat(3), 1.0, 1.0),
}


def _example_model_cov(survey: Survey, fixture: str) -> np.ndarray:
    """Run an Example-Model JSON fixture through the interpreter on the survey
    and return the summed NEV covariance (n, 3, 3). The example models are
    fixtures, not registered tools, so borrow a wired ToolError and swap in
    the adapted ``em``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        em = we.error.ErrorModel(survey, error_model="GYRO-NS-CT")
    te = em.errors
    adapted = _json_to_em_adapter(
        json.loads((DATA / "spe90408_example_models" / fixture).read_text()))
    te.em = adapted
    te.errors = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for code, entry in adapted["codes"].items():
            te.errors[code] = te._call_interpreter(
                code, entry["_iscwsa_term"], entry["magnitude"],
                entry["propagation"])
    n = len(survey.md)
    cov = np.zeros((n, 3, 3))
    for v in te.errors.values():
        assert v is not None and getattr(v, "cov_NEV", None) is not None, (
            "an Example-Model term produced no covariance")
        cov += np.asarray(v.cov_NEV)
    assert np.all(np.isfinite(cov)), "non-finite covariance"
    return cov


def _interp_cov(md: np.ndarray, cov: np.ndarray, target: float) -> np.ndarray:
    return np.array([[np.interp(target, md, cov[:, a, b]) for b in range(3)]
                     for a in range(3)])


def _cases():
    out = []
    for (well, model), ref in REF.items():
        for depth in ref:
            p = pytest.param(well, model, depth, id=f"{well}-{model}-{depth}")
            if (well, model) in XFAIL:
                p = pytest.param(
                    well, model, depth, id=f"{well}-{model}-{depth}",
                    marks=pytest.mark.xfail(reason=XFAIL[(well, model)],
                                            strict=False))
            out.append(p)
    return out


@pytest.mark.parametrize("well,model,depth", _cases())
def test_appendix_e(well, model, depth):
    builder, depth_to_m, cov_to_ref = WELLS[well]
    survey = builder()
    cov = _example_model_cov(survey, FIXTURE[model])

    md = np.asarray(survey.md)
    c = _interp_cov(md, cov, depth * depth_to_m) * cov_to_ref
    got = {"NN": c[0, 0], "NE": c[0, 1], "NV": c[0, 2],
           "EE": c[1, 1], "EV": c[1, 2], "VV": c[2, 2]}
    ref = dict(zip(("NN", "NE", "NV", "EE", "EV", "VV"), REF[(well, model)][depth]))

    failures = []
    for name, gv in got.items():
        r = ref[name]
        ok = (abs(gv - r) <= ABS_TOL_SMALL if abs(r) < SMALL
              else abs((gv - r) / r) <= REL_TOL)
        if not ok:
            failures.append(f"{name}: got {gv:.2f}, ref {r} (Δ {gv - r:+.2f})")
    assert not failures, (
        f"SPE 90408 Appendix E {well} {model} @ {depth} outside ±1%/±2u:\n  "
        + "\n  ".join(failures))
