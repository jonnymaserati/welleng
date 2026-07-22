"""Validate welleng's per-term covariance against the ISCWSA Error Model
Diagnostics -- the authoritative per-source, per-station reference output the
ISCWSA publishes.

Why this exists: the only prior MWD per-term validation
(``test_iscwsa_mwd_error.py``) runs the ISCWSA Standard Test Well #1, which
stays azi <= 75 deg and inc <= 90 deg. (That test IS against ISCWSA's own
numbers -- its ``vd`` block matches the ISCWSA #1 diagnostics exactly -- so
well #1 is already covered; it just never leaves the benign regime.) Sign /
Abs-placement bugs in the weight functions are a 0.0 no-op in that regime and
slipped through (see ``test_xym_abs_sign.py``). ISCWSA wells #2 and #3 cross
azi 90/180/270 and inc 90 (well #3 reaches azi 283, inc 110), so they actually
exercise those branches -- against ISCWSA's own numbers, which removes the
false positives a home-rolled formula comparison produces.

The .dat gives, per station per source, the six NEV covariance components
[NN, EE, VV, NE, NV, EV] -- exactly ``welleng.error.get_errors(cov_NEV[i])``.
The files are the ISCWSA diagnostics verbatim (kept for provenance), alongside
the well #2/#3 geometry the gyro Appendix-E tests already use.
"""
import re
from pathlib import Path

import numpy as np
import pytest

import welleng as we
from welleng.error import get_errors

DIAG_DIR = Path(__file__).parent / "test_data"

# Known welleng-vs-ISCWSA differences pending a fix. TA0 ruled 2026-07-21:
# match ISCWSA. DRFR (random depth) carries no variance at the surface station
# in welleng (drk_dDepth zeros station 0), where ISCWSA gives mag^2. Fix is a
# foundational datum change (must not disturb the systematic depth terms that
# correctly match at station 0), tracked separately. Keyed (source, station).
KNOWN_DIFFS = {("DRFR", 0)}


def _parse_dat(path: Path):
    lines = path.read_text().splitlines()
    ref = {}
    for ln in lines:
        if ln.startswith("Latitude:"):
            for k, v in re.findall(r"(\w+):\s*(-?[\d.]+)", ln):
                ref[k] = float(v)
            break
    stations, i = [], 0
    while i < len(lines):
        p = lines[i].split()
        if p and p[0] == "SURVEY" and len(p) >= 6:
            try:
                md, inc, azt = float(p[1]), float(p[2]), float(p[3])
            except ValueError:
                i += 1
                continue
            src, j = {}, i + 2  # skip the "MD NN EE VV NE NV EV" header row
            while j < len(lines) and "nev" in lines[j]:
                q = lines[j].split()
                src[q[0]] = list(map(float, q[3:9]))  # NN EE VV NE NV EV
                j += 1
            stations.append({"md": md, "inc": inc, "azt": azt, "src": src})
            i = j
        else:
            i += 1
    return ref, stations


def _survey(ref, stations):
    md = np.array([s["md"] for s in stations])
    inc = np.array([s["inc"] for s in stations])
    azt = np.array([s["azt"] for s in stations])
    sh = we.survey.SurveyHeader(
        name="iscwsa-diag", b_total=ref["BTotal"], dip=ref["Dip"],
        declination=ref["Declination"], latitude=ref["Latitude"],
        G=ref["GTotal"], azi_reference="true",
    )
    return we.survey.Survey(
        md=md, inc=inc, azi=azt, header=sh, error_model="ISCWSA MWD Rev5.11",
    )


# The full ISCWSA MWD example-well suite (#1/#2/#3), verbatim diagnostics. The
# README claims "validated 35/35 sources against all three ISCWSA example
# workbooks" -- this test is what ENFORCES that claim in CI (per-term, per-station,
# against ISCWSA's own numbers). #1 is the benign standard well (also covered by
# test_iscwsa_mwd_error.py's vd block, which matches this .dat); #2/#3 add the wide
# regime (azi>90, inc>90) that #1 cannot exercise.
DIAG_FILES = [
    "iscwsa_diagnostics_1.dat",
    "iscwsa_diagnostics_2.dat",
    "iscwsa_diagnostics_3.dat",
]


@pytest.mark.parametrize("dat_name", DIAG_FILES, ids=lambda s: s.replace(".dat", ""))
def test_per_term_covariance_matches_iscwsa_diagnostics(dat_name):
    path = DIAG_DIR / dat_name
    if not path.exists():
        pytest.skip(f"diagnostic file not shipped: {dat_name}")
    ref, stations = _parse_dat(path)
    s = _survey(ref, stations)
    terms = s.err.errors.errors

    # .dat values are printed to 4 decimals -> ~5e-5 rounding floor; allow a
    # little more for the print precision plus any m^2 magnitude at depth.
    atol, rtol = 1e-3, 1e-3
    failures = []
    for name, term in terms.items():
        if not all(name in st["src"] for st in stations):
            continue  # source not in the ISCWSA output for this well
        for i, st in enumerate(stations):
            if (name, i) in KNOWN_DIFFS:
                continue
            got = np.array(get_errors(np.asarray(term.cov_NEV[i])))
            exp = np.array(st["src"][name])
            if not np.allclose(got, exp, atol=atol, rtol=rtol):
                d = np.abs(got - exp).max()
                failures.append(f"{name} st{i} (md={st['md']:.0f}) maxdiff={d:.2e}")

    assert not failures, (
        f"{dat_name}: {len(failures)} term/station cells diverge from the "
        f"ISCWSA diagnostic:\n  " + "\n  ".join(failures[:20])
    )


def test_known_diff_is_actually_present():
    """Guard: the DRFR station-0 gap we're excluding must still be real, so the
    KNOWN_DIFFS waiver can't silently mask a later regression / accidental fix."""
    path = DIAG_DIR / "iscwsa_diagnostics_3.dat"
    if not path.exists():
        pytest.skip("diagnostic #3 not shipped")
    ref, stations = _parse_dat(path)
    s = _survey(ref, stations)
    got = np.array(get_errors(np.asarray(s.err.errors.errors["DRFR"].cov_NEV[0])))
    exp = np.array(stations[0]["src"]["DRFR"])
    # if this ever starts matching, DRFR was fixed -> drop it from KNOWN_DIFFS.
    assert not np.allclose(got, exp, atol=1e-3), (
        "DRFR station-0 now matches ISCWSA -- remove ('DRFR', 0) from KNOWN_DIFFS"
    )
