"""Figures for the anti-collision (reduced-conservatism) paper.

fig1 (schematic): why the pedal/separation rule is conservative — it measures the
   uncertainty ellipse by its support function (tangent distance) rather than the
   true boundary in the centre-to-centre direction.
fig2 (factual): minimum separation factor on the ISCWSA standard set — pedal
   rule vs the exact combined-ellipsoid Mahalanobis boundary.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT = "papers/figures"
os.makedirs(OUT, exist_ok=True)


# ---- fig1: support function vs true ray-boundary (schematic) ----
def fig1():
    a, b = 4.0, 1.4                  # eccentric combined-uncertainty ellipse (kσ)
    th = np.radians(38.0)            # oblique centre-to-centre direction
    u = np.array([np.cos(th), np.sin(th)])
    # true ellipse boundary distance along u (ray hits the ellipse)
    r_true = 1.0 / np.sqrt((u[0] / a) ** 2 + (u[1] / b) ** 2)
    # support function = tangent distance = how far the rule "reaches" along u
    h_sup = np.sqrt((a * u[0]) ** 2 + (b * u[1]) ** 2)
    P = u * 0.5 * (r_true + h_sup)   # offset centre: beyond the ellipse, inside reach

    perp = np.array([-u[1], u[0]])
    bp = u * r_true                  # true ellipse boundary along u
    tp = u * h_sup                   # support-function reach along u

    from adjustText import adjust_text
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.add_patch(Ellipse((0, 0), 2 * a, 2 * b, color="C0", alpha=0.16, ec="C0", lw=2))
    ax.text(-a + 0.3, -b + 0.25, "combined kσ uncertainty\n(Σ = Σ_ref + Σ_off)",
            color="C0", fontsize=8, style="italic")
    # the two WELL CENTRE points (use a ⊕ borehole-style marker for both)
    ax.plot(0, 0, marker="o", color="white", mec="C0", mew=1.8, ms=12, zorder=5)
    ax.plot(0, 0, marker="+", color="C0", mew=1.8, ms=10, zorder=6)
    ax.plot(P[0], P[1], marker="o", color="white", mec="C3", mew=1.8, ms=12, zorder=5)
    ax.plot(P[0], P[1], marker="+", color="C3", mew=1.8, ms=10, zorder=6)
    ax.annotate("", xy=P, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="0.4", lw=1.2))
    ax.plot([tp[0] - 2.6 * perp[0], tp[0] + 2.6 * perp[0]],
            [tp[1] - 2.6 * perp[1], tp[1] + 2.6 * perp[1]], color="C1", lw=1.5, ls="--")
    ax.plot([0, tp[0]], [0, tp[1]], color="C1", lw=0.8, ls=":")
    ax.plot(*bp, "o", color="C0", ms=8)
    ax.plot(*tp, "X", color="C1", ms=11)
    ax.annotate("", xy=tp, xytext=bp, arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
    # element labels, force-repelled apart (adjustText) with leader lines
    lbl = [
        (0.0, 0.0, "reference well centre", "C0"),
        (bp[0], bp[1], "true boundary\n(Mahalanobis = k)", "C0"),
        (tp[0], tp[1], "pedal reach\n(support fn √(uᵀΣu))", "C1"),
        (P[0], P[1], "offset well centre", "C3"),
    ]
    texts = [ax.text(x, y, s, color=c, fontsize=9, ha="center")
             for x, y, s, c in lbl]
    adjust_text(
        texts, x=[p[0] for p in lbl], y=[p[1] for p in lbl], ax=ax,
        expand=(1.6, 2.0), force_text=(0.6, 0.9),
        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7),
    )
    # the "over-reach" call-out: fixed in empty space, arrow to the gap
    gap_mid = 0.5 * (tp + bp)
    ax.annotate("over-reach: the rule 'collides',\nthe truth is clear",
                xy=gap_mid, xytext=(5.2, -1.6), fontsize=9, color="purple",
                ha="center", weight="bold",
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.4))
    ax.set_aspect("equal"); ax.set_xlim(-5.5, 9.5); ax.set_ylim(-4.0, 5.5)
    ax.set_xlabel("north [arb.]"); ax.set_ylabel("east [arb.]")
    ax.set_title("Why the separation rule is conservative: the support function\n"
                 "over-states the ellipsoid's reach toward an off-axis offset")
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(f"{OUT}/why-pedal-is-conservative.png", dpi=140, bbox_inches="tight")
    print("saved why-pedal-is-conservative.png")


# ---- fig2: pedal vs Mahalanobis min SF on the ISCWSA set (factual) ----
def fig2():
    rows = list(csv.DictReader(open("papers/data/anti-collision-diagnostics.csv")))
    names = [r["offset"].replace(" - well", "") for r in rows]
    ped = [float(r["welleng_pedal_minSF"]) for r in rows]
    mah = [float(r["mahalanobis_minSF"]) for r in rows]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - w / 2, ped, w, label="pedal / separation rule (ISCWSA)", color="C1")
    ax.bar(x + w / 2, mah, w, label="exact combined-ellipsoid (Mahalanobis)", color="C0")
    ax.axhline(1.0, color="k", lw=1, ls="--"); ax.text(len(names) - 0.6, 1.08, "SF = 1 (collision threshold)", fontsize=8)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_xlabel("ISCWSA standard-set offset well"); ax.set_ylabel("minimum separation factor")
    ax.set_title("Minimum separation factor: pedal rule vs exact Mahalanobis boundary\n"
                 "(same hit/clear verdict everywhere; Mahalanobis ≥ pedal — up to 1.48× less standoff)")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.25, axis="y")
    plt.tight_layout(); plt.savefig(f"{OUT}/pedal-vs-mahalanobis-iscwsa.png", dpi=140, bbox_inches="tight")
    print("saved pedal-vs-mahalanobis-iscwsa.png")


# ---- fig3: conservative surface construction (circumscribed polygon) ----
def fig3():
    from matplotlib.patches import Polygon
    a, b = 4.0, 2.2
    n = 8                                   # few verts so the difference is visible
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
    bloat = 1.0 / np.cos(np.pi / n)
    insc = np.column_stack([a * np.cos(ang), b * np.sin(ang)])
    circ = np.column_stack([a * bloat * np.cos(ang), b * bloat * np.sin(ang)])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(a * np.cos(t), b * np.sin(t), color="k", lw=2, label="true kσ uncertainty ellipse")
    ax.add_patch(Polygon(insc, closed=True, fill=False, ec="C3", lw=1.6, ls="--",
                         label="inscribed polygon — UNDER-counts (optimistic)"))
    ax.add_patch(Polygon(circ, closed=True, fill=False, ec="C0", lw=1.8,
                         label=f"circumscribed polygon (×1/cos(π/n)) — CONTAINS the\n"
                               f"ellipse, never under-counts (welleng default)"))
    ax.plot(insc[:, 0], insc[:, 1], "o", color="C3", ms=4)
    ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-3.6, 3.6)
    ax.set_xlabel("high side [arb.]"); ax.set_ylabel("low side [arb.]")
    ax.set_title("Conservative mesh-surface construction: the welleng default polygon\n"
                 "is CIRCUMSCRIBED, so the discretised surface never under-represents\n"
                 "the uncertainty for the given sigma (n = 8 shown for clarity)")
    ax.legend(loc="lower center", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(f"{OUT}/conservative-surface-construction.png", dpi=140, bbox_inches="tight")
    print("saved conservative-surface-construction.png")


fig1()
fig2()
fig3()
