"""Combining and forward-carrying overlapping surveys on public Volve data (F-4).

Reproduces the results in Corcutt (2026), "Combining and Forward-Carrying
Overlapping Wellbore Surveys". Well F-4 carries an MWD survey over the whole
run and a gyro overlapping the upper part, with MWD continuing alone below.

Two operations, both from ``welleng.combination`` (open, scalar reference):

  1. COMBINE the overlapping MWD + gyro with the best linear unbiased estimator
     (:func:`fuse_covariances`) -- the fused 1-sigma sits below either input.
  2. FORWARD-CARRY the gyro-observed systematic into the deep MWD-only section
     (:func:`carry_systematic_forward`) -- the constraint persists down-hole and
     reduces the deep covariance.

Data: derived from the public Equinor Volve EDM export, well F-4, CC BY 4.0 --
attribute Equinor and the Volve licence partners. The survey (ACTUAL
definitive), the two exact per-tool IPM error models, and the magnetic
reference are packaged in ``examples/data/volve_f4_survey_combination.json``
(20 kB) so the example runs without the ~211 MB source XML.

Run:  python examples/survey_combination_volve_f4.py
      python examples/survey_combination_volve_f4.py --plot   # save figures
"""
import argparse
import json
import os

import numpy as np

import welleng as we
from welleng.combination import carry_systematic_forward, fuse_covariances

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "volve_f4_survey_combination.json")
AX = ["N", "E", "V"]


def load_surveys():
    with open(DATA) as f:
        d = json.load(f)
    m = d["magnetic"]
    header = we.survey.SurveyHeader(
        name="F-4", azi_reference="grid",
        latitude=d["latitude"], longitude=d["longitude"],
        b_total=m["b_total"], dip=m["dip"], declination=m["declination"],
    )
    md = np.array(d["md_m"])
    inc = np.array(d["inc_deg"])
    azi = np.array(d["azi_deg"])
    common = dict(md=md, inc=inc, azi=azi, header=header, deg=True)
    mwd = we.survey.Survey(error_model=d["mwd_error_model"], **common)
    gyro = we.survey.Survey(error_model=d["gyro_error_model"], **common)
    ovl = np.where((md >= d["overlap_top_m"]) & (md <= d["gyro_base_m"]))[0]
    deep = np.where(md > d["gyro_base_m"])[0]
    return md, mwd, gyro, ovl, deep


def main(plot=False):
    md, mwd, gyro, ovl, deep = load_surveys()

    def sig(cov, i):
        return np.sqrt(cov[:, i, i])

    # --- 1. Combination over the overlap (independent surveys -> C = 0) ---
    fused = fuse_covariances(mwd.cov_nev[ovl], gyro.cov_nev[ovl])
    print("Combination over the overlap  "
          f"({md[ovl][0]:.0f}-{md[ovl][-1]:.0f} m MD)")
    print("  reduction at overlap base vs the better single survey:")
    for i in range(3):
        best = min(sig(mwd.cov_nev[ovl], i)[-1], sig(gyro.cov_nev[ovl], i)[-1])
        print(f"    sigma_{AX[i]}:  {best / sig(fused.cov_fused, i)[-1]:.2f} x")

    # --- 2. Forward-carry into the deep MWD-only section ---
    fc_g = carry_systematic_forward(mwd, gyro, ovl, deep, persist=("global",))
    fc_gs = carry_systematic_forward(mwd, gyro, ovl, deep,
                                     persist=("global", "systematic"))
    print(f"\nForward-carry into the deep MWD-only section "
          f"(> {md[deep][0]:.0f} m MD)")
    print("  reduction at TD  (sigma_nominal / sigma_carried):")
    for lbl, fc in [("declination only (any tool below)", fc_g),
                    ("+ sensor systematic (same tool below)", fc_gs)]:
        ratios = [float(np.sqrt(fc.cov_nominal[-1, i, i]
                                / fc.cov_carried[-1, i, i])) for i in range(3)]
        print(f"    {lbl}:")
        print(f"        N/E/V = {[round(x, 2) for x in ratios]}")

    if plot:
        _plot(md, mwd, gyro, ovl, deep, fused, fc_g, fc_gs, sig)


def _plot(md, mwd, gyro, ovl, deep, fused, fc_g, fc_gs, sig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = {"gyro": "#2a9d8f", "mwd": "#e76f51", "fused": "#264653"}

    fig, ax = plt.subplots(1, 3, figsize=(11, 6), sharey=True)
    for i, a in enumerate(ax):
        a.plot(sig(gyro.cov_nev[ovl], i), md[ovl], color=C["gyro"], label="gyro")
        a.plot(sig(mwd.cov_nev[ovl], i), md[ovl], color=C["mwd"], label="MWD")
        a.plot(sig(fused.cov_fused, i), md[ovl], color=C["fused"], lw=2.2,
               label="fused (BLUE)")
        a.set_title(f"$\\sigma_{AX[i]}$")
        a.set_xlabel("1$\\sigma$ (m)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("MD (m)")
    ax[0].invert_yaxis()
    ax[2].legend(loc="upper right", fontsize=8)
    fig.suptitle("Volve F-4 -- combining overlapping MWD + gyro (BLUE)")
    fig.tight_layout()
    fig.savefig("volve_f4_combination.png", dpi=120)

    fig, ax = plt.subplots(1, 3, figsize=(11, 6), sharey=True)
    for i, a in enumerate(ax):
        a.plot(np.sqrt(fc_gs.cov_nominal[:, i, i]), md[deep], color=C["mwd"],
               label="MWD-only (nominal)")
        a.plot(np.sqrt(fc_g.cov_carried[:, i, i]), md[deep], color=C["gyro"],
               ls="--", label="carried: declination")
        a.plot(np.sqrt(fc_gs.cov_carried[:, i, i]), md[deep], color=C["fused"],
               lw=2.2, label="carried: + sensor")
        a.set_title(f"$\\sigma_{AX[i]}$")
        a.set_xlabel("1$\\sigma$ (m)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("MD (m)")
    ax[0].invert_yaxis()
    ax[2].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Volve F-4 -- carrying the gyro-calibrated systematic forward")
    fig.tight_layout()
    fig.savefig("volve_f4_forward_carry.png", dpi=120)
    print("\nsaved volve_f4_combination.png, volve_f4_forward_carry.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true", help="save the two figures")
    main(**vars(p.parse_args()))
