"""n_verts convergence: discretisation over-conservatism vs compute cost.

The mesh realises each ellipsoid cross-section as an n-vertex CIRCUMSCRIBED
polygon (scaled by 1/cos(pi/n) so it contains the ellipse). The extra standoff
this discretisation demands over the true ellipse is therefore closed-form:
  radial over-count = sec(pi/n) - 1      (what scales the separation factor)
  area  over-count  = sec^2(pi/n) - 1
Both vanish as n -> infinity; the analytic Mahalanobis method IS that limit
(zero discretisation error). Compute scales ~linearly with n. This script
tabulates the trade-off and times an actual combined-covariance mesh build.
"""
import csv
import os
import time

import numpy as np

import tests.test_clearance_iscwsa as t
from welleng.clearance import combined_cov_mesh

OUT = "papers/data/nverts-convergence.csv"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gs = t.generate_surveys(t.data)
    ref, off = gs["Reference well"], gs["06 - well"]

    rows = []
    for n in [6, 8, 12, 16, 24, 48]:
        bloat = 1.0 / np.cos(np.pi / n)
        radial = (bloat - 1.0) * 100.0
        area = (bloat ** 2 - 1.0) * 100.0
        # time an actual combined-covariance mesh build at this resolution
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            combined_cov_mesh(ref, off, n_verts=n)
        ms = (time.perf_counter() - t0) / reps * 1000.0
        rows.append({
            "n_verts": n,
            "radial_overcount_pct": round(radial, 2),
            "area_overcount_pct": round(area, 2),
            "mesh_build_ms": round(ms, 1),
        })

    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"{'n_verts':>7s} {'radial%':>8s} {'area%':>7s} {'build_ms':>9s}")
    for r in rows:
        print(f"{r['n_verts']:>7d} {r['radial_overcount_pct']:>8.2f} "
              f"{r['area_overcount_pct']:>7.2f} {r['mesh_build_ms']:>9.1f}")
    print("\nanalytic Mahalanobis (n -> inf): radial 0.00%, area 0.00%, ~35 ms (no mesh)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
