"""A gyro error model computed at the package fallback location."""
import warnings

import numpy as np
import pytest

import welleng as we
from welleng.errors import tool_errors


def _survey(model, latitude=None, longitude=None):
    h = we.survey.SurveyHeader(
        name="T", b_total=50000.0, dip=70.0, declination=0.0,
        latitude=latitude, longitude=longitude, azi_reference="true")
    return we.survey.Survey(md=[0, 500, 1000.0], inc=[0, 10, 20.0],
                            azi=[30, 30, 30.0], header=h, error_model=model)


def test_the_header_substitutes_a_location_and_records_that_it_did():
    """`latitude is None` is never observable downstream -- the header fills in
    ~51.5 N. `_location_defaulted` is the only thing that still knows."""
    h = we.survey.SurveyHeader(name="T", latitude=None, longitude=None)
    assert h.latitude == pytest.approx(51.4934)
    assert h._location_defaulted is True
    assert we.survey.SurveyHeader(
        name="T", latitude=60.0, longitude=2.0)._location_defaulted is False


def test_a_gyro_at_the_fallback_location_warns():
    """⚠️ The existing magnetic guard is INVERTED for this case: it refuses a
    magnetic model built on default field values and returns early for a gyro
    -- correct, a gyro needs no magnetic reference. But a gyro is the model
    that needs LATITUDE, so the one family that depends on location was the one
    family let through."""
    with pytest.warns(UserWarning, match="ANTI-CONSERVATIVE"):
        _survey("GYRO-NS")


def test_a_magnetic_model_does_not_warn_about_latitude(recwarn):
    """Its weight functions do not reference Latitude, and a warning where the
    absence cannot change an answer is noise a reader learns to ignore."""
    _survey("MWD+SRGM")
    assert not [w for w in recwarn.list if "Latitude" in str(w.message)]


def test_a_real_location_is_silent(recwarn):
    _survey("GYRO-NS", latitude=60.0, longitude=2.0)
    assert not [w for w in recwarn.list if "Latitude" in str(w.message)]


def test_strict_refuses(monkeypatch):
    """The refusal is the correct end state; the default is a migration
    courtesy, because turning it on raises in every downstream consumer that
    builds a gyro survey without a location."""
    monkeypatch.setattr(tool_errors, "STRICT_LOCATION", True)
    with pytest.raises(ValueError, match="ANTI-CONSERVATIVE"):
        _survey("GYRO-NS")
    _survey("GYRO-NS", latitude=60.0, longitude=2.0)      # still fine
    _survey("MWD+SRGM")                                   # not latitude-dependent


def test_the_size_of_the_error_is_what_the_message_claims():
    """A number in a warning is a claim like any other, and this test caught a
    wrong one: the message first said ~25% at 60 N, which is the RATIO (1.245)
    misread as a percentage. Understated-by is `1 - 1/ratio`."""
    fb = 1.0 / np.cos(np.radians(51.4934))
    understated = lambda lat: 1.0 - fb / (1.0 / np.cos(np.radians(lat)))  # noqa: E731
    assert understated(60.0) == pytest.approx(0.197, abs=0.01)
    assert understated(70.0) == pytest.approx(0.451, abs=0.01)
    msg = tool_errors._latitude_rad.__doc__
    assert "~20% at 60 deg N" in msg and "~45% at 70 deg N" in msg


def test_the_model_actually_divides_by_cos_latitude():
    """The premise. If the OWSG gyro terms stopped carrying 1/cos(Latitude),
    this whole guard would be theatre."""
    import glob
    import json
    import os
    root = os.path.join(os.path.dirname(we.__file__), "errors", "iscwsa_json")
    hits = 0
    for f in glob.glob(root + "/**/GYRO*.json", recursive=True):
        if "Cos(Latitude)" in json.dumps(json.load(open(f))):
            hits += 1
    assert hits, "no gyro model references Cos(Latitude)"


def test_a_survey_that_warned_still_computes():
    """Rule 2's counterweight: the fallback is still applied, it is just no
    longer silent. Refusing outright would break every synthetic example."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = _survey("GYRO-NS")
    assert s.cov_nev is not None and len(s.md) == 3
