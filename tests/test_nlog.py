"""Tests for welleng.exchange.nlog — offline (no network) except where marked."""
import pytest
from welleng.exchange.nlog import DirSurvey, NLOGError


def _sv(md, inc, azi, name="TEST"):
    n = len(md)
    return DirSurvey(borehole_name=name, md=md, inc=inc, azi=azi,
                     tvd=[0.]*n, dx=[0.]*n, dy=[0.]*n, north_ref="G",
                     coord_system=None, proc_method="MC", convergence=0.,
                     declination=None, proc_date_ms=None, remark=None)


def test_provenance_measured():
    s = _sv([0, 500, 1000], [0, 2, 3], [0, 45, 90])
    assert s.azimuth_provenance() == "measured"


def test_provenance_all_zero_deviated_is_flagged():
    # the B13-01 case: real inclination, azimuth column all zeros
    s = _sv([0, 500, 1000, 1500], [0.5, 1.0, 2.0, 3.0], [0, 0, 0, 0])
    assert s.azimuth_provenance() == "all_zero_deviated"


def test_provenance_all_zero_vertical_is_harmless():
    s = _sv([0, 500, 1000, 1500], [0.0, 0.1, 0.2, 0.1], [0, 0, 0, 0])
    assert s.azimuth_provenance() == "all_zero_vertical"


def test_provenance_constant_assumed():
    s = _sv([0, 500, 1000, 1500], [0, 1, 2, 3], [90, 90, 90, 90])
    assert s.azimuth_provenance() == "constant_assumed"


def test_provenance_no_survey():
    # the F14-02 case: surface + TD only
    s = _sv([0, 1661], [0, 0], [0, 0])
    assert s.azimuth_provenance() == "no_survey"


def test_lateral_displacement_matches_geometry():
    # 1000 m held at 5 deg -> 1000*sin(5) = 87.2 m
    s = _sv([0, 1000], [5.0, 5.0], [0, 0])
    assert s.lateral_displacement() == pytest.approx(87.2, abs=0.2)


def test_to_welleng_refuses_error_model_on_fabricated_azimuth():
    s = _sv([0, 500, 1000, 1500], [0.5, 1, 2, 3], [0, 0, 0, 0])
    with pytest.raises(NLOGError, match="provenance"):
        s.to_welleng(error_model="ISCWSA MWD Rev5.11",
                     b_total=50000., dip=70., declination=0.)


def test_to_welleng_datums_tvd_to_first_station():
    # an NLOG survey starting at MD 30 with its own TVD datum: to_welleng must
    # produce ABSOLUTE TVD (30 m at the first station), not first-station-relative
    # (which would read 30 m shallow everywhere and mis-site a casing cut).
    import numpy as np
    s = DirSurvey(borehole_name="X", md=[30., 100., 200.], inc=[0., 0., 0.],
                  azi=[0., 0., 0.], tvd=[30., 100., 200.], dx=[0.] * 3, dy=[0.] * 3,
                  north_ref="G", coord_system=None, proc_method="MC",
                  convergence=0., declination=None, proc_date_ms=None, remark=None)
    sv = s.to_welleng()                       # geometry only (vertical, harmless)
    assert sv.tvd[0] == pytest.approx(30.0)
    import welleng as we
    node = we.survey.interpolate_md(sv, 87.5)
    assert float(np.asarray(node.tvd).ravel()[-1]) == pytest.approx(87.5)


def test_to_welleng_allows_override_and_geometry_only():
    s = _sv([0, 500, 1000, 1500], [0.5, 1, 2, 3], [0, 0, 0, 0])
    geom = s.to_welleng()                      # no error model: fine
    assert len(geom.md) == 4
    forced = s.to_welleng(error_model="ISCWSA MWD Rev5.11", force=True,
                          b_total=50000., dip=70., declination=0.)
    assert forced.cov_nev is not None


def _sv_xy(md, inc, azi, dx, dy):
    n = len(md)
    return DirSurvey(borehole_name="XY", md=md, inc=inc, azi=azi, tvd=[0.]*n,
                     dx=dx, dy=dy, north_ref="G", coord_system=None,
                     proc_method="MC", convergence=0., declination=None,
                     proc_date_ms=None, remark=None)


def test_single_bearing_detects_backfilled_nonzero_azimuth():
    # azimuth tests pass (varies), but all deflection is on one bearing
    s = _sv_xy([0, 500, 1000, 1500, 2000], [1, 2, 3, 3, 2],
               [45, 46, 45, 44, 45], [0., 0., 0., 0., 0.],
               [1., 10., 30., 60., 88.])
    assert s.azimuth_provenance() == "single_bearing"


def test_single_bearing_not_triggered_by_real_well():
    s = _sv_xy([0, 500, 1000, 1500, 2000], [1, 2, 3, 3, 2],
               [10, 40, 80, 120, 160], [0.5, 4., 12., 25., 40.],
               [1., 9., 25., 45., 60.])
    assert s.azimuth_provenance() == "measured"


def test_single_bearing_ignores_undeflected_wells():
    # both components ~zero: a vertical well, not a fabrication
    s = _sv_xy([0, 500, 1000, 1500], [0.1, 0.1, 0.1, 0.1],
               [10, 20, 30, 40], [0., 0., 0., 0.], [0., 0., 0., 0.])
    assert s.azimuth_provenance() == "measured"


# -- boreholes() + save_document(): offline, monkeypatched transport ------
import json as _json  # noqa: E402

from welleng.exchange import nlog as _nlog  # noqa: E402


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._data


def test_boreholes_posts_empty_filter_and_parses(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = req.data
        return _FakeResp(b'[{"boreholeName":"F06-07","boreholeDbk":123,'
                         b'"statusDescription":"Plugged and abandoned"}]')

    monkeypatch.setattr(_nlog.urllib.request, "urlopen", fake_urlopen)
    rows = _nlog.NLOGClient().boreholes()
    assert seen["url"].endswith("/brh/boreholes")
    assert seen["body"] == b"{}"                       # {} = whole catalogue
    assert rows[0]["boreholeDbk"] == 123
    assert "abandoned" in rows[0]["statusDescription"]


def test_boreholes_passes_filter_through(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["body"] = req.data
        return _FakeResp(b"[]")

    monkeypatch.setattr(_nlog.urllib.request, "urlopen", fake_urlopen)
    _nlog.NLOGClient().boreholes({"resultCode": "OAG"})
    assert _json.loads(seen["body"]) == {"resultCode": "OAG"}


def test_save_document_writes_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _nlog.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResp(b"%PDF-1.7 fake"),
    )
    p = tmp_path / "doc.pdf"
    n = _nlog.NLOGClient().save_document(999, p)
    assert n == len(b"%PDF-1.7 fake")
    assert p.read_bytes().startswith(b"%PDF")


# -- stratigraphy(): offline, response shape from a live NLOG bore ---------
_STRAT = {
    "boreholeName": "P11-A-02A",
    "depthRefPointDescription": "Rotary Table",
    "drpDatumCode": "MSL",
    "drpHeightInMeters": 46.5,
    "endAhDepthInMeters": 2691,
    "stratIntprts": [
        {
            "interpretationDate": 1161554400000,
            "stratSourceDescription": "EINDSLIP",
            "stratModelDescription": "RGD Lithostratigrafie",
            "preferredBln": "J",
            "stratIntvals": [
                {"topDepth": 0, "bottomDepth": 350, "stratUnitId": "NU",
                 "qualityDescription": None, "anomalyCode": None, "remark": None},
                {"topDepth": 350, "bottomDepth": 2691, "stratUnitId": "CKGR",
                 "qualityDescription": "TD", "anomalyCode": "UU", "remark": None},
            ],
        },
        {   # a second, NON-preferred interpretation
            "interpretationDate": 900000000000,
            "stratSourceDescription": "OLD",
            "stratModelDescription": "RGD Lithostratigrafie",
            "preferredBln": "N",
            "stratIntvals": [
                {"topDepth": 0, "bottomDepth": 2691, "stratUnitId": "NU",
                 "qualityDescription": None, "anomalyCode": None, "remark": None},
            ],
        },
    ],
}


def _patch_strat(monkeypatch):
    monkeypatch.setattr(_nlog.NLOGClient, "_post",
                        lambda self, resource, bid: _STRAT, raising=True)


def test_stratigraphy_returns_only_preferred_by_default(monkeypatch):
    _patch_strat(monkeypatch)
    cols = _nlog.NLOGClient().stratigraphy(163212895)
    assert len(cols) == 1
    assert cols[0].preferred is True
    assert cols[0].source == "EINDSLIP"


def test_stratigraphy_exposes_all_interpretations(monkeypatch):
    _patch_strat(monkeypatch)
    cols = _nlog.NLOGClient().stratigraphy(163212895, preferred_only=False)
    assert len(cols) == 2
    assert [c.preferred for c in cols] == [True, False]


def test_stratigraphy_carries_datum_and_md_intervals(monkeypatch):
    _patch_strat(monkeypatch)
    col = _nlog.NLOGClient().stratigraphy(163212895)[0]
    # datum travels with the record
    assert col.datum_description == "Rotary Table"
    assert col.datum_code == "MSL"
    assert col.datum_height_m == 46.5
    assert col.end_md == 2691
    # intervals parsed, depths are MD
    assert len(col.intervals) == 2
    iv0, iv1 = col.intervals
    assert (iv0.top_md, iv0.bottom_md, iv0.unit_id) == (0, 350, "NU")
    assert (iv1.unit_id, iv1.quality, iv1.anomaly) == ("CKGR", "TD", "UU")
# -- id_for_name(): alias-tolerant title resolution -----------------------
# Real P11 De Ruyter rows (public NLOG identifiers only): NLOG titles a bore
# "NAME (ALIAS)", which an exact match silently misses. Regression for the
# welleng-drilling finding (2026-09-06).
def _patch_suggest(monkeypatch, rows):
    monkeypatch.setattr(
        _nlog.NLOGClient, "suggest", lambda self, q: rows, raising=True
    )


def test_id_for_name_resolves_through_parenthetical_alias(monkeypatch):
    _patch_suggest(monkeypatch, [
        {"objectId": "228365142", "title": "P11-B-01 (P11-05)"},
    ])
    # exact match misses "P11-B-01 (P11-05)"; alias-tolerant resolves it
    assert _nlog.NLOGClient().id_for_name("P11-B-01") == 228365142


def test_id_for_name_prefers_exact_over_sibling_sidetrack(monkeypatch):
    # P11-A-02 and P11-A-02A are DIFFERENT bores; the exact query must not be
    # dragged onto the sidetrack by the alias-strip.
    _patch_suggest(monkeypatch, [
        {"objectId": "159376436", "title": "P11-A-02"},
        {"objectId": "163212895", "title": "P11-A-02A"},
    ])
    assert _nlog.NLOGClient().id_for_name("P11-A-02") == 159376436


def test_id_for_name_returns_none_when_alias_match_is_ambiguous(monkeypatch):
    # two bores sharing one base name via aliases must NOT silently resolve to
    # one of them (opposite of the single-hit case above).
    _patch_suggest(monkeypatch, [
        {"objectId": "1", "title": "P11-X-01 (P11-90)"},
        {"objectId": "2", "title": "P11-X-01 (P11-91)"},
    ])
    assert _nlog.NLOGClient().id_for_name("P11-X-01") is None


def test_id_for_name_none_when_unknown(monkeypatch):
    _patch_suggest(monkeypatch, [{"objectId": "9", "title": "F06-07"}])
    assert _nlog.NLOGClient().id_for_name("P11-B-01") is None
