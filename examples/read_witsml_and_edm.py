"""Worked example: read WITSML and EDM field data with welleng.

Uses the public Equinor **Volve** dataset (openly licensed). Download it and pass
the paths on the command line -- nothing is hard-coded:

    # WITSML realtime logs (a .zip of 1.4.1.1 log/tubular files) and/or the
    # EDM / COMPASS export (a single .xml):
    python examples/read_witsml_and_edm.py \
        --witsml /path/to/Volve_WITSML_realtime_2018-05-13.zip \
        --edm    /path/to/Volve.xml

Volve download: https://www.equinor.com/energy/volve-data-sharing

WITSML (`welleng.exchange.witsml`) and EDM (`welleng.exchange.edm_stream`) both
stream: a multi-GB WITSML zip is indexed from header prefixes alone, and the
211 MB EDM export is parsed in one bounded ``iterparse`` sweep.
"""
from __future__ import annotations

import argparse

from welleng.exchange.witsml import open_witsml
from welleng.exchange.edm_stream import open_edm


def demo_witsml(path: str) -> None:
    print(f"\n=== WITSML: {path} ===")
    r = open_witsml(path)
    n_logs = len(r.logs)  # indexes the zip (sets r.version) before we read it
    print(f"WITSML version {r.version} -- {n_logs} logs across "
          f"{len(r.wells)} well(s)")

    # Channel discovery: which logs carry a hookload channel?
    hkld = r.find("HKLD")
    print(f"\nlogs carrying HKLD: {len(hkld)}")
    if hkld:
        log = hkld[0]
        print(f"  reading '{log.name}' [{log.index_type}] "
              f"({len(log.mnemonics)} channels)")
        curves = log.curves(["HKLD"])
        idx, hk = curves[log.index_mnemonic], curves["HKLD"]
        print(f"  {len(hk)} samples; HKLD range "
              f"[{float(hk.min()):.1f}, {float(hk.max()):.1f}]")
        print(f"  index ({log.index_mnemonic}) spans {idx[0]} -> {idx[-1]}")

    # Temperature round-trip channels, if present:
    for mnem in ("MTIN", "ATMP_RT", "MTOA"):
        print(f"  {mnem:8s} carried by {len(r.find(mnem))} log(s)")

    # As-run BHA / string component tallies:
    tubs = r.tubulars()
    print(f"\ntubular strings: {len(tubs)}")
    if tubs and tubs[0].components:
        t = tubs[0]
        print(f"  '{t.name}' -- {len(t.components)} components "
              f"(bit at sequence {t.components[0].sequence})")


def demo_edm(path: str) -> None:
    print(f"\n=== EDM: {path} ===")
    edm = open_edm(path, with_geopressure=True)
    print(f"{len(edm.wells)} wells, {len(edm.wellbores)} wellbores")

    # Pick a wellbore that has geometry to demonstrate the accessors.
    wb = next((w.wellbore_id for w in edm.wellbores.values()
               if edm.geometry(w.wellbore_id)), None)
    if wb is None:
        print("  (no wellbore with geometry found)")
        return
    name = edm.wellbores[wb].name
    print(f"\nwellbore: {name}")

    for prof in edm.pore_pressure(wb, phase="PROTOTYPE", latest=True):
        print(f"  pore pressure [{prof.phase}]: {len(prof.tvd)} pts, "
              f"{prof.value.min():.0f}-{prof.value.max():.0f} psi "
              f"(pressure is canonical; emw is an RKB-datum view)")
    for prof in edm.temperature(wb, latest=True):
        print(f"  temperature: {len(prof.tvd)} pts, "
              f"{prof.value.min():.0f}-{prof.value.max():.0f} degF (raw)")

    geo = edm.geometry(wb)
    cased = [h for h in geo if h.sect_type_code == "CAS" and h.md_shoe]
    print(f"\n  as-run geometry: {len(geo)} sections "
          f"({len(cased)} cased with a shoe)")
    for h in cased[:5]:
        print(f"    {h.od_casing:g}\" casing  shoe MD {h.md_shoe:.0f}")

    seen, names = set(), []
    for f in edm.formations(wb):
        if f.name not in seen:
            seen.add(f.name)
            names.append(f.name)
    print(f"  formations: {names[:5]}")

    # The reader is self-documenting -- no need to decode raw CD_* codes:
    s = edm.schema["CD_HOLE_SECT"]
    print(f"\n  schema['CD_HOLE_SECT'] -> {s['name']}: {s['description']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--witsml", help="path to a WITSML 1.4.1.1 .zip or directory")
    ap.add_argument("--edm", help="path to an EDM / COMPASS .xml export")
    args = ap.parse_args()
    if not (args.witsml or args.edm):
        ap.error("pass --witsml and/or --edm (see the module docstring)")
    if args.witsml:
        demo_witsml(args.witsml)
    if args.edm:
        demo_edm(args.edm)


if __name__ == "__main__":
    main()
