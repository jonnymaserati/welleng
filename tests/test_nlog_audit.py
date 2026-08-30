"""Tests for the NLOG bulk-dump survey audit."""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyproj")

from welleng.exchange import nlog_audit as A  # noqa: E402


# -- the grid-azimuth defect classifier (the core logic) ---------------------

@pytest.mark.parametrize("offset, declared, exp_actual, exp_status", [
    # correct: azimuth in the declared grid -> offset = g_grid - g_utm
    (1.0 - 3.0, "RD", "rd", "ok"),                 # g_rd=1.0, g_utm=3.0
    (0.0, "ED50-UTM31", "utm", "ok"),
    (-3.0, "TRUE NORTH", "true", "ok"),
    # wrong sign: RD convergence applied backwards -> offset = -g_rd - g_utm
    (-1.0 - 3.0, "RD", "rd", "wrong_sign"),
    # reflected: azimuth mirrored -> offset = -(g_rd - g_utm)
    (-(1.0 - 3.0), "RD", "rd", "reflected"),
    # mis-grid: declared RD but azimuth actually in UTM grid -> offset ~ 0
    (0.0, "RD", "utm", "mis_grid"),
    # unexplained: matches no reference
    (45.0, "RD", None, "unexplained"),
    # unclassified: a declared system with no convergence available
    (0.0, "ED50-NEDTM", None, "unclassified"),
    # unclassified: no offset (too little lateral)
    (np.nan, "RD", None, "unclassified"),
])
def test_classify_grid(offset, declared, exp_actual, exp_status):
    actual, status, _ = A._classify_grid(offset, g_utm=3.0, g_rd=1.0,
                                         declared=declared)
    assert status == exp_status
    assert actual == exp_actual


def test_wrong_sign_distinct_from_ok():
    # the wrong-sign and correct cases must NOT collide (2*gamma apart)
    ok = A._classify_grid(1.0 - 3.0, 3.0, 1.0, "RD")
    bad = A._classify_grid(-1.0 - 3.0, 3.0, 1.0, "RD")
    assert ok[1] == "ok" and bad[1] == "wrong_sign"


# -- end-to-end on a synthetic dump ------------------------------------------

def _synthetic_dump(tmp_path):
    """A tiny NLOG-format CSV with known-defect wells (consistent surface coords)."""
    from pyproj import Transformer
    # one NL surface location, expressed consistently in the three datums
    x_ed, y_ed = 600000.0, 5900000.0
    to_rd = Transformer.from_crs("EPSG:23031", "EPSG:28992", always_xy=True)
    to_wg = Transformer.from_crs("EPSG:23031", "EPSG:32631", always_xy=True)
    x_rd, y_rd = to_rd.transform(x_ed, y_ed)
    x_wg, y_wg = to_wg.transform(x_ed, y_ed)

    rows = []

    def add(wb, md, inc, azi, dev_off=(0.0, 0.0)):
        # tvd ~ md for near-vertical; simple straight offsets for the check
        rows.append(dict(
            WELLBORE=wb, COORD_SYSTEM_CD="RD", AH_DEPTH=md,
            TV_DEPTH_NAP=md, AZIMUTH=azi, DEV_ANGLE=inc,
            X_SURFACE_UTM31_ED50=x_ed, Y_SURFACE_UTM31_ED50=y_ed,
            DX_UTM31_ED50=dev_off[0], DY_UTM31_ED50=dev_off[1],
            X_SURFACE_RD=x_rd, Y_SURFACE_RD=y_rd,
            X_SURFACE_UTM31_WGS84=x_wg, Y_SURFACE_UTM31_WGS84=y_wg))

    # vertical well (max inc < 3)
    for md, inc in [(0, 0.0), (500, 0.5), (1000, 1.0)]:
        add("W_VERT", md, inc, 0.0)
    # inc-only: deviated but azimuth unrecorded (NaN)
    for md, inc in [(0, 0.0), (500, 10.0), (1000, 20.0), (1500, 25.0)]:
        add("W_INCONLY", md, inc, np.nan)
    # deviated with a real varying azimuth
    for md, inc, azi in [(0, 0.0, 0), (500, 15.0, 30), (1000, 30.0, 90),
                         (1500, 45.0, 135)]:
        add("W_DEV", md, inc, azi)
    # DLS spike: a 30 deg inclination jump over 1 m
    for md, inc, azi in [(0, 0.0, 0), (999, 2.0, 0), (1000, 32.0, 0),
                         (1001, 32.0, 0)]:
        add("W_SPIKE", md, inc, azi)
    # torsion: an azimuth FLIP (~180 deg) between two curved legs -> the
    # osculating plane inverts, invisible to DLS
    for md, inc, azi in [(0, 0.0, 0), (300, 20.0, 45), (600, 35.0, 45),
                         (900, 20.0, 225), (1200, 35.0, 225)]:
        add("W_FLIP", md, inc, azi)

    df = pd.DataFrame(rows)
    p = tmp_path / "dirstelsel.csv"
    df.to_csv(p, sep=";", decimal=",", index=False, encoding="latin-1")
    return str(p)


def test_audit_dump_end_to_end(tmp_path):
    audit = A.audit_dump(_synthetic_dump(tmp_path))
    assert audit.n_wellbores == 5
    # surface coords are internally consistent -> no defects
    assert audit.n_surface_defects == 0
    assert audit.surface_p95_m["rd_vs_ed50"] < 1.0
    # provenance classified correctly
    prov = {w.wellbore: w.provenance for w in audit.wellbores}
    assert prov["W_VERT"] == "vertical"
    assert prov["W_INCONLY"] == "azi_unrecorded"
    assert prov["W_DEV"] == "deviated"
    # the inc-only well carries a hollow-annulus radius (~MD*sin(inc))
    inconly = next(w for w in audit.wellbores if w.wellbore == "W_INCONLY")
    assert inconly.annulus_radius_m > 100
    # the DLS spike is flagged
    spike = next(w for w in audit.wellbores if w.wellbore == "W_SPIKE")
    assert spike.n_dls_spikes >= 1 and spike.max_dls_deg30m > 30
    assert audit.n_dls_wells >= 1
    # the azimuth flip is caught by TORSION, not DLS
    flip = next(w for w in audit.wellbores if w.wellbore == "W_FLIP")
    assert flip.n_plane_inversions >= 1 and flip.max_torsion_deg > 150
    assert audit.n_torsion_wells >= 1


def test_report_is_text(tmp_path):
    audit = A.audit_dump(_synthetic_dump(tmp_path))
    txt = A._report(audit)
    assert "NLOG dump audit" in txt and "grid-azimuth" in txt
