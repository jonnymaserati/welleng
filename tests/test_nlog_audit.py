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
    # inc-only: deviated but azimuth unrecorded (NaN); monotone -> TRUE annulus
    for md, inc in [(0, 0.0), (500, 10.0), (1000, 20.0), (1500, 25.0)]:
        add("W_INCONLY", md, inc, np.nan)
    # inc-only that PORPOISES: builds, drops back through vertical, re-builds ->
    # the two deviated legs carry independent unknown azimuths, so NOT a ring
    for md, inc in [(0, 0.0), (400, 15.0), (800, 1.0), (1200, 15.0), (1600, 20.0)]:
        add("W_PORPOISE", md, inc, np.nan)
    # deviated with a real varying azimuth
    for md, inc, azi in [(0, 0.0, 0), (500, 15.0, 30), (1000, 30.0, 90),
                         (1500, 45.0, 135)]:
        add("W_DEV", md, inc, azi)
    # DLS spike: a 30 deg inclination jump over 1 m
    for md, inc, azi in [(0, 0.0, 0), (999, 2.0, 0), (1000, 32.0, 0),
                         (1001, 32.0, 0)]:
        add("W_SPIKE", md, inc, azi)
    # torsion: an azimuth FLIP (~180 deg) between two curved DEVIATED legs ->
    # the osculating plane inverts, invisible to DLS
    for md, inc, azi in [(0, 0.0, 0), (300, 20.0, 45), (600, 35.0, 45),
                         (900, 20.0, 225), (1200, 35.0, 225)]:
        add("W_FLIP", md, inc, azi)
    # near-vertical wobble: big azimuth swings at low inclination -> the plane
    # flips geometrically but it is NOT a defect; the inc floor must NOT flag it
    for md, inc, azi in [(0, 0.0, 0), (300, 2.0, 20), (600, 3.0, 200),
                         (900, 2.0, 40), (1200, 3.0, 220)]:
        add("W_WOBBLE", md, inc, azi)

    df = pd.DataFrame(rows)
    p = tmp_path / "dirstelsel.csv"
    df.to_csv(p, sep=";", decimal=",", index=False, encoding="latin-1")
    return str(p)


def test_audit_dump_end_to_end(tmp_path):
    audit = A.audit_dump(_synthetic_dump(tmp_path))
    assert audit.n_wellbores == 7
    # surface coords are internally consistent -> no defects
    assert audit.n_surface_defects == 0
    assert audit.surface_p95_m["rd_vs_ed50"] < 1.0
    # provenance classified correctly
    prov = {w.wellbore: w.provenance for w in audit.wellbores}
    assert prov["W_VERT"] == "vertical"
    assert prov["W_INCONLY"] == "azi_unrecorded"
    assert prov["W_PORPOISE"] == "azi_unrecorded"
    assert prov["W_DEV"] == "deviated"
    # the inc-only well carries a hollow-annulus radius (integrated closure)
    inconly = next(w for w in audit.wellbores if w.wellbore == "W_INCONLY")
    assert inconly.annulus_radius_m > 100
    # monotone inc-only is a TRUE ring; the porpoising one is NOT (it decorrelates)
    assert inconly.annulus_valid is True and inconly.n_vertical_crossings == 0
    porp = next(w for w in audit.wellbores if w.wellbore == "W_PORPOISE")
    assert porp.annulus_valid is False and porp.n_vertical_crossings >= 1
    assert audit.n_true_annuli == 1
    # the DLS spike is flagged
    spike = next(w for w in audit.wellbores if w.wellbore == "W_SPIKE")
    assert spike.n_dls_spikes >= 1 and spike.max_dls_deg30m > 30
    assert audit.n_dls_wells >= 1
    # the azimuth flip is caught by TORSION, not DLS
    flip = next(w for w in audit.wellbores if w.wellbore == "W_FLIP")
    assert flip.n_plane_inversions >= 1 and flip.max_torsion_deg > 150
    assert audit.n_torsion_wells >= 1
    # near-vertical wobble is NOT flagged (inc floor) despite the plane flipping
    wobble = next(w for w in audit.wellbores if w.wellbore == "W_WOBBLE")
    assert wobble.n_plane_inversions == 0


def test_report_is_text(tmp_path):
    audit = A.audit_dump(_synthetic_dump(tmp_path))
    txt = A._report(audit)
    assert "NLOG dump audit" in txt and "grid-azimuth" in txt


# -- analytical grid-alignment rotation (Procrustes) -------------------------

def test_procrustes_recovers_known_rotation_exactly():
    # PROVE: the closed form recovers an applied rotation to machine precision,
    # regardless of the (varied) vector lengths.
    rng = np.random.default_rng(7)
    for deg in (0.0, 2.0, -3.4, 17.3, 179.0):
        th = np.radians(deg)
        a = rng.normal(size=(30, 2)) * rng.uniform(1, 50, (30, 1))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        b = (R @ a.T).T
        from welleng.utils import best_fit_rotation_2d
        got = np.degrees(best_fit_rotation_2d(a, b))
        assert abs(A._wrap180(got - deg)) < 1e-9


def test_procrustes_is_length_weighted():
    # a long, consistent leg dominates a short noisy one (vs an unweighted mean)
    a_n = np.array([100.0, 0.5])
    a_e = np.array([0.0, 0.5])
    # rotate the long leg by 2 deg, the short leg by 40 deg
    th1, th2 = np.radians(2.0), np.radians(40.0)
    b_n = np.array([100 * np.cos(th1), 0.5 * (np.cos(th2) - np.sin(th2))])
    b_e = np.array([100 * np.sin(th1), 0.5 * (np.sin(th2) + np.cos(th2))])
    from welleng.utils import best_fit_rotation_2d
    got = np.degrees(best_fit_rotation_2d(
        np.column_stack([a_n, a_e]), np.column_stack([b_n, b_e])))
    assert abs(got - 2.0) < 1.0          # pulled toward the long leg's 2 deg
