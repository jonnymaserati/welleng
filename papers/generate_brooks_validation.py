"""Validation against Brooks (SPE-116155): our separation-factor metric is the
Mahalanobis distance computed directly, sqrt(d' Sigma^-1 d'); Brooks computes the
same quantity by transforming to "Mahalanobis space" with T = V E^{-1/2} V^T
(spectral decomposition of Sigma) and taking the Euclidean norm ||T d'||. These
are algebraically identical; this script confirms it numerically at each well's
closest-approach point. Writes papers/data/brooks-validation.csv.
"""
import csv
import os

import numpy as np

import tests.test_clearance_iscwsa as t
from welleng.clearance import MahalanobisClearance

OUT = "papers/data/brooks-validation.csv"


def brooks_T(S):
    """Brooks's transform to Mahalanobis space (SPE-116155 Eq. 3)."""
    w, V = np.linalg.eigh(S)
    return V @ np.diag(w ** -0.5) @ V.T


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    gs = t.generate_surveys(t.data)
    ref = gs["Reference well"]
    Rp = np.column_stack([ref.n, ref.e, ref.tvd])
    Rc = np.asarray(ref.cov_nev).reshape(-1, 3, 3)

    rows = []
    for well in [w for w in gs if w != "Reference well"]:
        off = gs[well]
        Op = np.column_stack([off.n, off.e, off.tvd])
        Oc = np.asarray(off.cov_nev).reshape(-1, 3, 3)
        # drive the SHIPPED implementation in pure-metric mode (no floor, no
        # radii) so this exercises welleng's actual _sf_point, not an inline
        # re-derivation.
        mc = MahalanobisClearance(ref, off)
        mc.sigma_pa = 0.0
        mc.Sm = 0.0
        # find the closest-approach point via the shipped per-point metric
        best = (np.inf, None, None)
        for i in range(len(Rp)):
            for j in range(len(Op)):
                if np.linalg.eigvalsh(Rc[i] + Oc[j]).min() < 1e-9:
                    continue              # degenerate near-surface cov; skip
                m = mc._sf_point(Rp[i], Rc[i], 0.0, Op[j], Oc[j], 0.0) * mc.k
                if m < best[0]:
                    best = (m, i, j)
        m_ours, i, j = best               # welleng's shipped Mahalanobis distance
        S = Rc[i] + Oc[j]
        d = Op[j] - Rp[i]
        m_brooks = float(np.linalg.norm(brooks_T(S) @ d))  # Brooks's transform
        rows.append({
            "offset": well,
            "ours_sqrt_dT_Sinv_d": round(m_ours, 6),
            "brooks_norm_Td": round(m_brooks, 6),
            "abs_diff": f"{abs(m_ours - m_brooks):.2e}",
        })

    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"{'offset':>10s} {'ours':>10s} {'Brooks':>10s} {'|diff|':>9s}")
    worst = 0.0
    for r in rows:
        print(f"{r['offset']:>10s} {r['ours_sqrt_dT_Sinv_d']:>10.6f} "
              f"{r['brooks_norm_Td']:>10.6f} {r['abs_diff']:>9s}")
        worst = max(worst, float(r["abs_diff"]))
    print(f"\nworst |difference| = {worst:.2e}  (machine precision)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
