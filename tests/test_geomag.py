"""Offline WMM validation against NOAA's official WMM2025 test values."""
import os

import numpy as np
import pytest

from welleng.geomag import (
    local_field, lookup_field, wmm_validity, GeomagLookupError,
)

_TV = os.path.join(os.path.dirname(__file__), "test_data", "wmm2025_testvalues.txt")


def _rows():
    out = []
    with open(_TV) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = [float(x) for x in line.split()]
            # year, alt_km, lat, lon, D, I, H, X, Y, Z, F
            out.append(v[:11])
    return out


def test_local_wmm_matches_official_test_values():
    """The bundled offline WMM reproduces NOAA's WMM2025 test values within the
    model's numerical-agreement spec (0.1 nT in components); this is the
    acceptance gate for the local synthesis."""
    worst = {k: 0.0 for k in "XYZHF"} | {"D": 0.0, "I": 0.0}
    for year, alt_km, lat, lon, D, Inc, H, X, Y, Z, F in _rows():
        r = local_field(lat, lon, alt_km * 1000.0, year)["field-value"]
        got = {
            "X": r["north-intensity"]["value"],
            "Y": r["east-intensity"]["value"],
            "Z": r["vertical-intensity"]["value"],
            "H": r["horizontal-intensity"]["value"],
            "F": r["total-intensity"]["value"],
            "D": r["declination"]["value"],
            "I": r["inclination"]["value"],
        }
        ref = dict(X=X, Y=Y, Z=Z, H=H, F=F, D=D, I=Inc)
        for k in worst:
            worst[k] = max(worst[k], abs(got[k] - ref[k]))
    # nT components: 0.1 nT is the WMM software spec; we are far inside it
    for k in "XYZHF":
        assert worst[k] < 0.1, f"{k} off {worst[k]:.4g} nT (>0.1 nT spec)"
    # D/I are derived and sensitive where H is small; still well inside a
    # hundredth of a degree (~100x below the model's physical uncertainty)
    assert worst["D"] < 0.02, f"declination off {worst['D']:.4g} deg"
    assert worst["I"] < 0.02, f"inclination off {worst['I']:.4g} deg"


def test_local_payload_shape_matches_bgs_contract():
    r = local_field(58.0, 2.0, 0.0, 2027.0)
    fv = r["field-value"]
    for k in ("total-intensity", "inclination", "declination"):
        assert "value" in fv[k] and "units" in fv[k]
    assert r["source"] == "local-wmm"
    assert r["model_name"] == "WMM2025"


def test_date_string_and_decimal_year_agree():
    a = local_field(60.0, 5.0, 0.0, 2026.5)["field-value"]["declination"]["value"]
    b = local_field(60.0, 5.0, 0.0, "2026-07-02")["field-value"]["declination"]["value"]
    assert abs(a - b) < 1e-3            # 2026-07-02 ~ 2026.500


def test_out_of_window_local_raises():
    vf, vt = wmm_validity()
    with pytest.raises(GeomagLookupError, match="validity window"):
        local_field(60.0, 5.0, 0.0, vt + 5.0)


def test_lookup_source_auto_uses_local_in_window(monkeypatch):
    # in-window auto must NOT touch the network -- break the BGS path to prove it
    import welleng.geomag as gm

    def _boom(*a, **k):
        raise AssertionError("BGS called for an in-window WMM request")

    monkeypatch.setattr(gm, "_bgs_field", _boom)
    r = gm.lookup_field(58.0, 2.0, 0.0, date=2027.0)     # auto, wmm, in window
    assert r["source"] == "local-wmm"


def test_lookup_source_local_forced_out_of_window_raises(monkeypatch):
    import welleng.geomag as gm
    vf, vt = wmm_validity()
    with pytest.raises(GeomagLookupError):
        gm.lookup_field(58.0, 2.0, 0.0, date=vt + 3.0, source="local")


def test_lookup_bad_source_raises():
    with pytest.raises(ValueError, match="source must be"):
        lookup_field(58.0, 2.0, source="nope")


def test_surveyheader_records_mag_model_offline():
    # a real location + missing field triggers a lookup; in-window it resolves
    # OFFLINE via the bundled WMM and stamps the provenance label
    import welleng as we
    h = we.survey.SurveyHeader(
        name="t", azi_reference="grid",
        latitude=58.0, longitude=2.0, altitude=0.0,
        survey_date="2027-01-01",
    )
    assert h.mag_source["declination"] == "lookup"
    assert h.mag_model == "WMM2025 (local-wmm)"


def test_surveyheader_user_values_leave_mag_model_none():
    import welleng as we
    h = we.survey.SurveyHeader(
        name="t", azi_reference="grid",
        b_total=50000.0, dip=72.0, declination=1.0,
    )
    assert h.mag_model is None
    assert all(v == "user" for v in h.mag_source.values())


def test_bundled_wmm_not_expired():
    # staleness gate: FAIL once the bundled WMM is past its window, so the
    # 5-yearly coefficient refresh (docs/dev/FUTURE_WORK.md) can't be silently
    # missed. Deliberately time-aware -- it is validating a real-world validity
    # window, not code behaviour. Out-of-window dates already fail SAFE at
    # runtime (local raises -> BGS fallback); this just forces the refresh.
    import warnings
    from datetime import timezone, datetime
    _, vt = wmm_validity()
    today = datetime.now(timezone.utc).date()
    now = today.year + (today.timetuple().tm_yday - 1) / 365.25
    assert now <= vt, (
        f"bundled WMM expired (valid to {vt:.0f}); refresh the coefficients "
        "(WMM2030) -- see docs/dev/FUTURE_WORK.md"
    )
    if vt - now < 0.5:
        warnings.warn(
            f"bundled WMM expires {vt:.0f} (<6 months); schedule the refresh",
            RuntimeWarning, stacklevel=2,
        )


# -- per-station field along a survey (the opt-in refinement) -----------------

def _erd_survey(lat=71.0, lon=25.0, reach_n=5000.0, reach_e=3000.0, tvd=3500.0):
    import welleng as we
    n = 200
    md = np.linspace(0, np.hypot(reach_n, reach_e) + tvd, n)
    inc = np.clip(np.linspace(0, 88, n), 0, 90)
    azi = np.full(n, np.degrees(np.arctan2(reach_e, reach_n)))
    sh = we.survey.SurveyHeader(
        name="erd", azi_reference="grid", latitude=lat, longitude=lon,
        b_total=54000.0, dip=79.0, declination=15.0, survey_date="2027-06-01")
    return we.survey.Survey(md=md, inc=inc, azi=azi, header=sh, deg=True)


def test_field_along_survey_station0_is_wellhead_reference():
    from welleng.geomag import field_along_survey, local_field
    s = _erd_survey()
    sf = field_along_survey(s)
    ref = local_field(71.0, 25.0, 0.0, "2027-06-01")["field-value"]
    assert abs(sf.declination[0] - ref["declination"]["value"]) < 1e-9
    assert abs(sf.d_declination[0]) < 1e-9          # delta is zero at station 0
    assert abs(sf.d_b_total[0]) < 1e-6
    assert sf.declination.shape == s.md.shape


def test_field_along_survey_lateral_gradient_is_real_and_bounded():
    # high-lat ERD: declination varies across the stepout (~0.05-0.1 deg span)
    from welleng.geomag import field_along_survey
    sf = field_along_survey(_erd_survey())
    span = sf.d_declination.max() - sf.d_declination.min()
    assert span > 0.01                              # a real, non-trivial shift
    assert span < 1.0                               # but not absurd


def test_field_along_survey_descend_adds_vertical():
    from welleng.geomag import field_along_survey
    s = _erd_survey()
    lateral = field_along_survey(s, descend=False).d_declination.max()
    descended = field_along_survey(s, descend=True).d_declination.max()
    # the TVD downward-continuation adds a little more on top of the lateral
    assert descended > lateral
    # but both are small (the vertical effect is minor)
    assert descended - lateral < 0.1


def test_field_along_survey_requires_wellhead_location():
    import welleng as we
    from welleng.geomag import field_along_survey, GeomagLookupError
    sh = we.survey.SurveyHeader(name="t", azi_reference="grid",
                                b_total=51000.0, dip=72.0, declination=1.0)
    s = we.survey.Survey(md=np.array([0., 1000.]), inc=np.array([0., 30.]),
                         azi=np.array([0., 45.]), header=sh, deg=True)
    with pytest.raises(GeomagLookupError, match="latitude/longitude"):
        field_along_survey(s)


def test_field_along_survey_typical_well_negligible():
    # a short, mid-latitude well: per-station refinement is below noise
    from welleng.geomag import field_along_survey
    s = _erd_survey(lat=58.44, lon=1.89, reach_n=600.0, reach_e=300.0, tvd=2500.0)
    sf = field_along_survey(s)
    assert abs(sf.d_declination).max() < 0.01       # < 0.01 deg -> not worth it here
