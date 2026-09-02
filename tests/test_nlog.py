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
