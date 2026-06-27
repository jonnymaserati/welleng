"""Generate the anti-collision paper's diagnostics + validation numbers.

Reproducible from the PUBLIC ISCWSA standard set of clearance scenarios. Writes:
  - papers/data/anti-collision-diagnostics.csv  (per-offset: published / pedal /
    Mahalanobis min separation factors + the conservatism ratio)
Run from the welleng repo root with the [all] extra installed.
"""
import csv
import json
import os
import time

import numpy as np

from welleng.survey import Survey, make_survey_header
from welleng.clearance import IscwsaClearance, MahalanobisClearance

DATA = json.load(open("tests/test_data/clearance_iscwsa_well_data.json"))
OUT = "papers/data/anti-collision-diagnostics.csv"


def survey(well):
    sh = make_survey_header(DATA["wells"][well]["header"])
    radius = 0.4572 if well == "Reference well" else 0.3048
    w = DATA["wells"][well]
    return Survey(
        md=w["MD"], inc=w["IncDeg"], azi=w["AziDeg"], n=w["N"], e=w["E"],
        tvd=w["TVD"], radius=radius, header=sh, error_model="ISCWSA MWD Rev4",
    )


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ref = survey("Reference well")
    rows = []
    for well in [w for w in DATA["wells"] if w != "Reference well"]:
        off = survey(well)
        kop = 900.0 if well == "10 - well" else -np.inf
        published = float(np.min(DATA["wells"][well]["SF"]))
        ped = IscwsaClearance(ref, off, kop_depth=kop)
        ped_min = float(np.nanmin(ped.sf))
        mah_min = float(np.nanmin(MahalanobisClearance(ref, off, kop_depth=kop).sf))
        # validation error: welleng pedal vs published, at the original stations
        got = ped.sf[np.where(ped.ref.interpolated == False)]  # noqa: E712
        n = min(len(got), len(DATA["wells"][well]["SF"]))
        rel_err = float(np.nanmax(np.abs(
            (got[:n] - np.array(DATA["wells"][well]["SF"])[:n])
            / np.where(np.abs(np.array(DATA["wells"][well]["SF"])[:n]) < 0.1,
                       np.nan, np.array(DATA["wells"][well]["SF"])[:n]))) * 100)
        ratio = mah_min / ped_min if ped_min > 0 else float("nan")
        rows.append({
            "offset": well, "published_minSF": round(published, 3),
            "welleng_pedal_minSF": round(ped_min, 3),
            "pedal_vs_published_relerr_pct": round(rel_err, 3),
            "mahalanobis_minSF": round(mah_min, 3),
            "maha_over_pedal": round(ratio, 3) if ped_min > 0 else "",
            "verdict_pedal": "HIT" if ped_min < 1 else "clear",
            "verdict_maha": "HIT" if mah_min < 1 else "clear",
        })

    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # console summary for the paper tables
    print(f"{'offset':>10s} {'pub':>6s} {'pedal':>6s} {'err%':>5s} "
          f"{'maha':>6s} {'m/p':>5s} {'ped':>5s} {'mah':>5s}")
    for r in rows:
        print(f"{r['offset']:>10s} {r['published_minSF']:>6.3f} "
              f"{r['welleng_pedal_minSF']:>6.3f} "
              f"{r['pedal_vs_published_relerr_pct']:>5.2f} "
              f"{r['mahalanobis_minSF']:>6.3f} {str(r['maha_over_pedal']):>5s} "
              f"{r['verdict_pedal']:>5s} {r['verdict_maha']:>5s}")
    worst = max(r["pedal_vs_published_relerr_pct"] for r in rows)
    ratios = [r["maha_over_pedal"] for r in rows
              if isinstance(r["maha_over_pedal"], float) and r["maha_over_pedal"] > 0]
    print(f"\nworst pedal-vs-published rel err: {worst:.3f}%")
    print(f"maha/pedal ratio (clear wells): "
          f"min {min(ratios):.2f}, max {max(ratios):.2f}")
    # speed
    off = survey("06 - well")
    for cls, nm in [(IscwsaClearance, "pedal"), (MahalanobisClearance, "maha")]:
        t = time.perf_counter()
        for _ in range(5):
            cls(ref, off)
        print(f"speed {nm}: {(time.perf_counter() - t) / 5 * 1000:.0f} ms/well")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
