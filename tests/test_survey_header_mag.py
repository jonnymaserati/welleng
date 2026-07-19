"""SurveyHeader magnetic-field lookup: historic dates, provenance, the gate.

The BGS default model (WMM) only covers a ~10-year window around the present;
a historic ``survey_date`` errors on it. The lookup must fall back to IGRF
*for the requested date* — never silently substitute today's field (a ~3.5 deg
declination bias at Volve for a 2008 well) nor overwrite the header's
``survey_date``.

``SurveyHeader.mag_source`` records where each geomagnetic value came from
(``'user'`` / ``'lookup'`` / ``'default'``) and magnetic error models refuse
``'default'`` — provided OR looked up, never a placeholder.

The BGS client is mocked so the tests run without network access.
"""

import warnings

import pytest

import welleng.survey as ws
from welleng.geomag import GeomagLookupError, _build_url


class _FakeLookup:
    """Mimics welleng.geomag.lookup_field.

    The WMM model raises for dates before 2025, as the live BGS service
    does; IGRF accepts historic dates. Distinct field values per model so
    tests can tell which epoch/model was used.
    """

    def __init__(self):
        self.calls = []
        self.altitudes = []
        self.fail_all = False

    def __call__(self, latitude, longitude, altitude=0.0, date=None,
                 model="wmm", **kwargs):
        self.calls.append(model)
        self.altitudes.append(altitude)
        if self.fail_all:
            raise GeomagLookupError("service unreachable")
        if model == "wmm" and date is not None and int(date[:4]) < 2025:
            raise GeomagLookupError(
                "400: Date for WMM 2025 must be between 2025.0 and 2035.0"
            )
        b = 50365.0 if model == "igrf" else 50943.0
        dec = -2.08 if model == "igrf" else 1.43
        return {
            "field-value": {
                "total-intensity": {"value": b},
                "inclination": {"value": 70.0},
                "declination": {"value": dec},
            }
        }


@pytest.fixture
def fake_lookup(monkeypatch):
    fake = _FakeLookup()
    monkeypatch.setattr(ws, "lookup_field", fake)
    return fake


def _header(**kwargs):
    return ws.SurveyHeader(
        name="volve-like",
        latitude=58.4416,
        longitude=1.8875,
        **kwargs,
    )


def test_historic_date_uses_igrf_for_requested_date(fake_lookup):
    sh = _header(survey_date="2008-12-31")
    assert fake_lookup.calls == ["wmm", "igrf"]
    assert sh.b_total == 50365.0          # the IGRF (requested-epoch) value
    assert sh.survey_date == "2008-12-31"  # never overwritten


def test_current_date_uses_default_model(fake_lookup):
    sh = _header(survey_date="2026-01-01")
    assert fake_lookup.calls == ["wmm"]
    assert sh.b_total == 50943.0


def test_total_failure_warns_and_keeps_defaults_and_date(fake_lookup):
    fake_lookup.fail_all = True
    with pytest.warns(UserWarning, match="2008-12-31"):
        sh = _header(survey_date="2008-12-31")
    assert sh.b_total == 50000.0           # mag_defaults, not a wrong epoch
    assert sh.survey_date == "2008-12-31"


def test_user_supplied_values_skip_lookup_entirely(fake_lookup):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sh = _header(
            survey_date="2008-12-31", b_total=50100.0, dip=71.0, declination=-1.5
        )
    assert sh.b_total == 50100.0
    assert fake_lookup.calls == []         # nothing missing -> no round-trip


def test_defaulted_location_skips_lookup(fake_lookup):
    # A lookup at the fallback location would be as fictitious as the
    # defaults — don't pay the round-trip for it.
    sh = ws.SurveyHeader(survey_date="2026-01-01")
    assert fake_lookup.calls == []
    assert sh.b_total == 50000.0


def test_altitude_passed_through_in_metres(fake_lookup):
    # welleng speaks metres end-to-end; the km conversion happens inside the
    # BGS client at the wire (see test_bgs_url_converts_altitude_to_km).
    _header(survey_date="2026-01-01", altitude=-90.0)
    assert fake_lookup.altitudes == [-90.0]


def test_bgs_url_converts_altitude_to_km():
    url = _build_url(58.4416, 1.8875, altitude=-90.0, date="2008-12-31",
                     model="igrf")
    assert url.startswith("https://geomag.bgs.ac.uk/web_service/GMModels/igrf/")
    assert "altitude=-0.09" in url
    assert "date=2008-12-31" in url
    assert "format=json" in url


# --- mag provenance + the ErrorModel gate (magnetic models refuse defaults) --


def _survey(header, error_model):
    import welleng as we

    return we.survey.Survey(
        md=[0.0, 500.0, 1000.0],
        inc=[0.0, 30.0, 60.0],
        azi=[45.0, 45.0, 45.0],
        header=header,
        error_model=error_model,
    ).err


def test_default_header_magnetic_model_raises():
    sh = ws.SurveyHeader()          # no location, no mag data, no lookup
    assert sh.mag_source == {
        "b_total": "default", "dip": "default", "declination": "default"
    }
    with pytest.raises(ValueError, match="magnetic model"):
        _survey(sh, "ISCWSA MWD Rev5.11")


def test_default_header_gyro_model_passes():
    sh = ws.SurveyHeader()          # gyro models don't consume B/dip/dec
    _survey(sh, "GYRO-MWD")


def test_explicit_mag_values_pass_gate():
    sh = ws.SurveyHeader(b_total=50365.0, dip=72.1, declination=-2.08)
    assert all(s == "user" for s in sh.mag_source.values())
    _survey(sh, "ISCWSA MWD Rev5.11")


def test_post_init_setattr_marks_user_and_passes_gate():
    # The common pattern: bare header, fields assigned afterwards from a file
    sh = ws.SurveyHeader()
    for k, v in {"b_total": 50365.0, "dip": 72.1, "declination": -2.08}.items():
        setattr(sh, k, v)
    assert all(s == "user" for s in sh.mag_source.values())
    _survey(sh, "ISCWSA MWD Rev5.11")


def test_lookup_at_user_location_passes_gate(fake_lookup):
    sh = _header(survey_date="2008-12-31")
    assert all(s == "lookup" for s in sh.mag_source.values())
    _survey(sh, "ISCWSA MWD Rev5.11")


def test_defaulted_location_fails_gate(fake_lookup):
    # No real location -> defaults -> refused by the magnetic gate
    sh = ws.SurveyHeader(survey_date="2026-01-01")
    assert all(s == "default" for s in sh.mag_source.values())
    with pytest.raises(ValueError, match="magnetic model"):
        _survey(sh, "ISCWSA MWD Rev5.11")


def test_shallow_copy_does_not_leak_provenance():
    import copy

    sh = ws.SurveyHeader(b_total=50365.0, dip=72.1, declination=-2.08)
    sh2 = copy.copy(sh)
    sh2.b_total = None
    assert sh.mag_source["b_total"] == "user"       # source unaffected
    assert sh2.mag_source["b_total"] == "default"
