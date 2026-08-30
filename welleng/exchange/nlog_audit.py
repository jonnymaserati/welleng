"""Survey-data-quality audit for the NLOG bulk deviation dump.

NLOG (the Netherlands oil & gas portal) publishes a monthly bulk CSV of every
Dutch well's deviation survey (``nlog_dirstelsel_*.csv`` inside
``thematische_data_boringen.zip``). This module audits that dump for the defects
that bite a downstream consumer, using only the geometry in the file (no external
truth needed):

1. **Surface coordinates** — the per-well surface position is given in three
   datums (RD / ED50-UTM31 / WGS84-UTM31); cross-transform them and flag any that
   disagree (a wrong header coordinate).
2. **Grid-azimuth alignment** — each well *declares* its azimuth reference in
   ``COORD_SYSTEM_CD`` (RD grid / ED50-UTM31 grid / true north / ...). Compare the
   stored azimuths against the direction of the reported ED50-UTM31 position steps
   (a purely geometric check, no magnetics) and classify the *actual* reference
   from the grid convergences. A well is flagged when the declared and actual
   references disagree, split by failure mode:

   - ``mis_grid`` — azimuth is in a *different* grid than declared (a fixed
     rotation by the convergence difference, up to ~3.4 deg across NL);
   - ``wrong_sign`` — the grid convergence was applied with the **wrong sign**
     (a ``2*gamma`` error — the classic loader bug);
   - ``reflected`` — the azimuth is mirrored (``360 - azi``);
   - ``unexplained`` — matches no known reference (likely corrupt).

3. **Azimuth provenance** — classify each well ``vertical`` / ``deviated``
   (azimuth recorded and varying) / ``single_bearing`` (near-constant azimuth) /
   ``azi_unrecorded`` (deviated but azimuth missing). The last is the "hollow
   annulus" case: the true horizontal position error is a *ring* at radius
   ``R = MD*sin(inc)``, which a per-station covariance cannot represent.
4. **Dogleg severity** — flag physically impossible spikes (> 30 deg/30 m).
5. **Structural** — non-monotonic / duplicate measured depths, and TVD drift
   (min-curvature TVD vs the reported ``TV_DEPTH_NAP``).

The checks are geometric and self-contained, so the report is reproducible by
anyone who downloads the same public dump. Requires ``pandas`` and ``pyproj``
(both welleng dependencies).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import pandas as pd
    from pyproj import Proj, Transformer
except ImportError as exc:                       # pragma: no cover
    raise ImportError(
        "nlog_audit needs pandas + pyproj (welleng runtime deps)"
    ) from exc

# datum EPSG codes used by the dump's three surface-coordinate columns
_EPSG_RD = "EPSG:28992"
_EPSG_ED50_UTM31 = "EPSG:23031"
_EPSG_WGS84_UTM31 = "EPSG:32631"
_EPSG_WGS84_GEO = "EPSG:4326"

#: declared COORD_SYSTEM_CD -> the grid whose north the azimuths use. Value is a
#: key into the per-well convergence dict; ``None`` means true/geographic north
#: (no grid convergence).
_DECLARED_GRID = {
    "RD": "rd",
    "ED50-UTM31": "utm",
    "TRUE NORTH": None,
    "ED50-GEOGR": None,
    "ETRS89-UTM31": "utm",          # ETRS89 ~ WGS84, UTM31 convergence ~ ED50
}


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class WellboreAudit:
    """Per-wellbore audit result (all angles in degrees, lengths in metres)."""

    wellbore: str                                # NLOG wellbore identifier
    n_stations: int                              # survey station count
    max_inc: Optional[float] = None              # maximum inclination
    provenance: str = "too_short"                # provenance class (see module doc)
    annulus_radius_m: Optional[float] = None     # max MD*sin(inc), annulus radius
    grid_declared: Optional[str] = None          # COORD_SYSTEM_CD as declared
    grid_actual: Optional[str] = None            # actual azimuth reference
    grid_status: Optional[str] = None            # grid defect class (see module doc)
    grid_rotation_deg: Optional[float] = None    # residual vs correct declared ref
    max_dls_deg30m: Optional[float] = None       # worst dogleg severity
    n_dls_spikes: int = 0                        # stations with DLS > 30 deg/30m
    nonmonotonic_md: int = 0                     # MD steps <= 0
    duplicate_md: int = 0                        # repeated MDs
    tvd_worst_step_m: Optional[float] = None     # worst |computed-reported| dTVD


@dataclass
class DumpAudit:
    """Whole-dump audit summary + the per-wellbore rows."""

    n_wellbores: int                             # wellbores audited
    surface_p95_m: dict = field(default_factory=dict)      # cross-datum residual p95
    n_surface_defects: int = 0                   # wells with a surface residual > 50 m
    projection: dict = field(default_factory=dict)         # grid_status -> count
    provenance: dict = field(default_factory=dict)         # provenance -> count
    n_dls_wells: int = 0                         # wells with any DLS spike
    wellbores: list = field(default_factory=list)          # list[WellboreAudit]

    def defects(self):
        """The wellbores worth a consumer's attention (grid or DLS or structural)."""
        return [
            w for w in self.wellbores
            if (w.grid_status not in (None, "ok", "unclassified"))
            or w.n_dls_spikes or w.nonmonotonic_md or w.duplicate_md
        ]


def load_dump(path):
    """Read an NLOG ``nlog_dirstelsel_*.csv`` into a DataFrame (numeric-coerced)."""
    df = pd.read_csv(path, sep=";", decimal=",", encoding="latin-1",
                     low_memory=False)
    for c in ("AH_DEPTH", "TV_DEPTH_NAP", "AZIMUTH", "DEV_ANGLE",
              "X_SURFACE_UTM31_ED50", "Y_SURFACE_UTM31_ED50",
              "DX_UTM31_ED50", "DY_UTM31_ED50",
              "X_SURFACE_RD", "Y_SURFACE_RD",
              "X_SURFACE_UTM31_WGS84", "Y_SURFACE_UTM31_WGS84"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _surface_residuals(heads):
    t_rd = Transformer.from_crs(_EPSG_RD, _EPSG_ED50_UTM31, always_xy=True)
    t_wg = Transformer.from_crs(_EPSG_WGS84_UTM31, _EPSG_ED50_UTM31, always_xy=True)
    xe1, ye1 = t_rd.transform(heads.X_SURFACE_RD.values, heads.Y_SURFACE_RD.values)
    xe2, ye2 = t_wg.transform(heads.X_SURFACE_UTM31_WGS84.values,
                              heads.Y_SURFACE_UTM31_WGS84.values)
    rd = np.hypot(xe1 - heads.X_SURFACE_UTM31_ED50.values,
                  ye1 - heads.Y_SURFACE_UTM31_ED50.values)
    wg = np.hypot(xe2 - heads.X_SURFACE_UTM31_ED50.values,
                  ye2 - heads.Y_SURFACE_UTM31_ED50.values)
    return rd, wg


def _convergences(heads):
    """Grid convergence (deg) at each well's surface, for RD and UTM31."""
    t = Transformer.from_crs(_EPSG_ED50_UTM31, _EPSG_WGS84_GEO, always_xy=True)
    lon, lat = t.transform(heads.X_SURFACE_UTM31_ED50.values,
                           heads.Y_SURFACE_UTM31_ED50.values)
    p_utm, p_rd = Proj(_EPSG_WGS84_UTM31), Proj(_EPSG_RD)
    g_utm = np.array([p_utm.get_factors(x, y).meridian_convergence
                      for x, y in zip(lon, lat)])
    g_rd = np.array([p_rd.get_factors(x, y).meridian_convergence
                     for x, y in zip(lon, lat)])
    return {"utm": g_utm, "rd": g_rd, None: np.zeros(len(lon))}


def _classify_grid(azi_offset, g_utm, g_rd, declared, tol=1.5):
    """Classify a well's actual azimuth reference from the observed offset.

    ``azi_offset`` is (ED50-UTM31 position-step direction) minus (survey-azimuth
    direction). The correct offset for azimuths in grid ``X`` is ``g_X - g_utm``.
    Returns ``(actual, status, rotation)``.
    """
    if not np.isfinite(azi_offset):
        return None, "unclassified", None
    # a declared system we have no convergence for (e.g. ED50-NEDTM) cannot be
    # verified -- leave it unclassified rather than force a spurious grid match.
    if declared not in _DECLARED_GRID:
        return None, "unclassified", None
    grids = {"rd": g_rd, "utm": g_utm, "true": 0.0}
    # correct reference for each candidate grid
    clean = {name: g - g_utm for name, g in grids.items()}
    dec = _DECLARED_GRID[declared]
    dec_name = {"rd": "rd", "utm": "utm", None: "true"}.get(dec, None)

    def _err(target):
        return abs(_wrap180(azi_offset - target))

    # 1. correct as declared?
    if dec_name and _err(clean[dec_name]) < tol:
        return dec_name, "ok", _err(clean[dec_name])
    # 2. wrong-sign convergence on the declared grid (2*gamma bug)?
    if dec_name in ("rd", "utm"):
        g_dec = grids[dec_name]
        if _err(-g_dec - g_utm) < tol:
            return dec_name, "wrong_sign", _err(-g_dec - g_utm)
    # 3. reflected azimuth (mirrored about north)?
    if dec_name and _err(-clean[dec_name]) < tol:
        return dec_name, "reflected", _err(-clean[dec_name])
    # 4. a DIFFERENT grid (mis-declared)?
    best = min(clean, key=lambda n: _err(clean[n]))
    if _err(clean[best]) < tol:
        return best, "mis_grid", _err(clean[best])
    return None, "unexplained", None


def _wellbore_metrics(name, g):
    """Per-wellbore geometry: provenance, DLS, structural, TVD, azimuth offset."""
    g = g.sort_values("AH_DEPTH")
    md = g.AH_DEPTH.values.astype(float)
    inc_deg = g.DEV_ANGLE.values.astype(float)
    azi_deg = g.AZIMUTH.values.astype(float)
    tvd = g.TV_DEPTH_NAP.values.astype(float)
    dx = g.DX_UTM31_ED50.values.astype(float)
    dy = g.DY_UTM31_ED50.values.astype(float)
    n = len(md)
    w = WellboreAudit(wellbore=str(name), n_stations=int(n))
    if n < 3:
        return w, np.nan

    inc = np.radians(inc_deg)
    azi = np.radians(azi_deg)
    dmd = np.diff(md)
    w.nonmonotonic_md = int(np.sum(dmd <= 0))
    w.duplicate_md = int(np.sum(np.isclose(dmd, 0.0)))
    ok = dmd > 0.1

    i1, i2, a1, a2 = inc[:-1], inc[1:], azi[:-1], azi[1:]
    cos_dl = np.clip(np.cos(i2 - i1) - np.sin(i1) * np.sin(i2)
                     * (1 - np.cos(a2 - a1)), -1, 1)
    dl = np.arccos(cos_dl)
    with np.errstate(invalid="ignore", divide="ignore"):
        rf = np.where(dl > 1e-9, 2 / dl * np.tan(dl / 2), 1.0)
    dn = 0.5 * dmd * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
    de = 0.5 * dmd * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
    dv = 0.5 * dmd * (np.cos(i1) + np.cos(i2)) * rf

    dls = np.degrees(dl[ok]) / dmd[ok] * 30.0
    w.max_dls_deg30m = float(np.max(dls)) if len(dls) else 0.0
    w.n_dls_spikes = int(np.sum(dls > 30.0))

    w.max_inc = float(np.nanmax(inc_deg))
    w.annulus_radius_m = float(np.nanmax(md * np.sin(np.radians(inc_deg))))
    dev = inc_deg > 3.0
    if w.max_inc < 3.0:
        w.provenance = "vertical"
    else:
        adev = azi_deg[dev]
        if adev.size == 0 or np.mean(~np.isfinite(adev)) > 0.8:
            w.provenance = "azi_unrecorded"
        else:
            a = np.radians(adev[np.isfinite(adev)])
            R = np.hypot(np.mean(np.sin(a)), np.mean(np.cos(a)))
            spread = np.degrees(np.sqrt(max(0.0, -2 * np.log(min(R, 1.0)))))
            w.provenance = "single_bearing" if spread < 5.0 else "deviated"

    # azimuth offset vs ED50-UTM31 position steps (for the grid check)
    step_e, step_n = np.diff(dx), np.diff(dy)
    horiz = np.hypot(de, dn)
    m = ok & (horiz > 2.0) & (np.hypot(step_e, step_n) > 1.0)
    azi_offset = np.nan
    if np.sum(m) >= 5:
        d = _wrap180(np.degrees(np.arctan2(step_e[m], step_n[m]))
                     - np.degrees(np.arctan2(de[m], dn[m])))
        s, c = np.median(np.sin(np.radians(d))), np.median(np.cos(np.radians(d)))
        azi_offset = float(np.degrees(np.arctan2(s, c)))

    tvd_ok = np.isfinite(tvd[1:]) & np.isfinite(tvd[:-1]) & ok
    if np.sum(tvd_ok) >= 5:
        resid = (np.diff(tvd) - dv)[tvd_ok]
        w.tvd_worst_step_m = float(np.max(np.abs(resid)))
    return w, azi_offset


def audit_dump(path, surface_defect_m=50.0):
    """Audit an NLOG bulk deviation dump. Returns a :class:`DumpAudit`.

    Parameters
    ----------
    path : str
        Path to an ``nlog_dirstelsel_*.csv`` (unzipped from the monthly
        ``thematische_data_boringen.zip``).
    surface_defect_m : float
        Cross-datum surface residual (m) above which a well is a header defect.
    """
    df = load_dump(path)
    heads = df.groupby("WELLBORE").first().reset_index()
    declared = df.groupby("WELLBORE").COORD_SYSTEM_CD.agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else None)

    rd_res, wg_res = _surface_residuals(heads)
    conv = _convergences(heads)
    conv_by_wb = {wb: (conv["utm"][i], conv["rd"][i])
                  for i, wb in enumerate(heads.WELLBORE.values)}
    surf_defect = int(np.sum((rd_res > surface_defect_m)
                             | (wg_res > surface_defect_m)))

    rows = []
    for name, g in df.groupby("WELLBORE"):
        w, off = _wellbore_metrics(name, g)
        w.grid_declared = declared.get(name)
        g_utm, g_rd = conv_by_wb.get(name, (np.nan, np.nan))
        actual, status, rot = _classify_grid(off, g_utm, g_rd, w.grid_declared)
        w.grid_actual, w.grid_status, w.grid_rotation_deg = actual, status, rot
        rows.append(w)

    prov = {}
    proj = {}
    for w in rows:
        prov[w.provenance] = prov.get(w.provenance, 0) + 1
        proj[w.grid_status] = proj.get(w.grid_status, 0) + 1
    return DumpAudit(
        n_wellbores=len(rows),
        surface_p95_m={"rd_vs_ed50": float(np.nanpercentile(rd_res, 95)),
                       "wgs84_vs_ed50": float(np.nanpercentile(wg_res, 95))},
        n_surface_defects=surf_defect,
        projection=proj,
        provenance=prov,
        n_dls_wells=int(sum(1 for w in rows if w.n_dls_spikes)),
        wellbores=rows,
    )


def _report(audit):
    """A short text summary of a :class:`DumpAudit`."""
    lines = [f"NLOG dump audit — {audit.n_wellbores} wellbores",
             f"  surface cross-datum p95: {audit.surface_p95_m}; "
             f"defects (>50 m): {audit.n_surface_defects}",
             "  grid-azimuth status:"]
    for k in ("ok", "mis_grid", "wrong_sign", "reflected", "unexplained",
              "unclassified"):
        if k in audit.projection:
            lines.append(f"    {k:13s} {audit.projection[k]}")
    lines.append("  azimuth provenance:")
    for k, v in sorted(audit.provenance.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {k:15s} {v}")
    lines.append(f"  wellbores with DLS spikes (>30 deg/30m): {audit.n_dls_wells}")
    return "\n".join(lines)


if __name__ == "__main__":       # pragma: no cover - CLI convenience
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m welleng.exchange.nlog_audit <dirstelsel.csv>")
        raise SystemExit(2)
    print(_report(audit_dump(sys.argv[1])))
