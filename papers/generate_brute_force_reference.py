"""Brute-force exactness check for MahalanobisClearance (the paper's "<1e-3"
claim). Compares the shipped broadphase+narrowphase separation factor against an
exhaustive all-pairs minimum-Mahalanobis search on both wells resampled to a fine
measured-depth step. Writes papers/data/brute-force-reference.csv.

This is the released, runnable code behind the statement that the method "agrees
with an exhaustive all-pairs reference sampled at 1 m to better than 1e-3".
"""
import csv
import os

import numpy as np

import tests.test_clearance_iscwsa as t
from welleng.clearance import MahalanobisClearance
from welleng.survey import _interpolate_pos_nev

OUT = "papers/data/brute-force-reference.csv"
STEP = 1.0          # measured-depth sampling of the brute-force reference, m
K, SM, SIGMA_PA = 3.5, 0.3, 0.5      # MahalanobisClearance defaults


def resample(survey, step):
    md = np.asarray(survey.md, float)
    cov = np.asarray(survey.cov_nev, float).reshape(-1, 3, 3)
    rad = np.asarray(survey.radius, float).reshape(-1)
    mdf = np.arange(md[0], md[-1] + step, step)
    # position by minimum curvature (matches MahalanobisClearance._at); cov/rad linear
    P = np.empty((len(mdf), 3))
    for r, q in enumerate(mdf):
        i = int(np.clip(np.searchsorted(md, q, side="right") - 1, 0, len(md) - 2))
        P[r] = _interpolate_pos_nev(survey, float(q - md[i]), i)
    C = np.empty((len(mdf), 3, 3))
    for a in range(3):
        for b in range(3):
            C[:, a, b] = np.interp(mdf, md, cov[:, a, b])
    return P, C, np.interp(mdf, md, rad)


def brute_min_sf(ref, off, step=STEP):
    Rp, Rc, Rr = resample(ref, step)
    Op, Oc, Ro = resample(off, step)
    best = np.inf
    for i in range(len(Rp)):
        d = Op - Rp[i]
        D = np.linalg.norm(d, axis=1)
        S = Rc[i] + Oc + SIGMA_PA ** 2 * np.eye(3)
        scale = np.divide(np.maximum(D - (Rr[i] + Ro + SM), 0.0), D,
                          out=np.zeros_like(D), where=D > 0)
        dp = d * scale[:, None]
        m = np.sqrt(np.einsum('oi,oij,oj->o', dp, np.linalg.inv(S), dp)) / K
        best = min(best, float(np.min(m)))
    return best


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gs = t.generate_surveys(t.data)
    ref = gs["Reference well"]
    rows = []
    # exclude Well 10 (sidetrack scanned below KOP — needs kop handling; its
    # crossing is verified separately in tests/test_clearance_iscwsa.py)
    for well in [w for w in gs if w not in ("Reference well", "10 - well")]:
        off = gs[well]
        shipped = float(np.nanmin(MahalanobisClearance(ref, off).sf))
        brute = brute_min_sf(ref, off)
        rows.append({
            "offset": well,
            "shipped_minSF": round(shipped, 4),
            "brute_force_minSF": round(brute, 4),
            "abs_diff": f"{abs(shipped - brute):.2e}",
        })

    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"{'offset':>10s} {'shipped':>9s} {'brute(1m)':>10s} {'|diff|':>9s}")
    worst = 0.0
    for r in rows:
        print(f"{r['offset']:>10s} {r['shipped_minSF']:>9.4f} "
              f"{r['brute_force_minSF']:>10.4f} {r['abs_diff']:>9s}")
        worst = max(worst, float(r["abs_diff"]))
    print(f"\nworst |difference| = {worst:.2e}  (claim: < 1e-3)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
