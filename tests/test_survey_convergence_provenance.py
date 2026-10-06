"""Grid convergence: an unestablished value must not pass as a measured zero."""
import warnings

import numpy as np
import pytest

import welleng as we


def _survey(**kw):
    h = we.survey.SurveyHeader(
        name="T", azi_reference=kw.pop("azi_reference", "grid"),
        b_total=50000.0, dip=70.0, declination=0.0, **kw)
    return we.survey.Survey(md=[0, 500, 1000.0], inc=[0, 10, 20.0],
                            azi=[30, 30, 30.0], header=h)


def test_unset_convergence_is_recorded_as_assumed():
    """`0.0` is a real and common assumption -- which is exactly why it must
    not be the silent one. Absence has to be distinguishable from a measured
    zero, or nobody can tell a well on the central meridian from one where
    nobody looked."""
    assert we.survey.SurveyHeader(name="T").convergence_assumed is True
    assert we.survey.SurveyHeader(name="T", convergence=0.0
                                  ).convergence_assumed is False
    assert we.survey.SurveyHeader(name="T", convergence=1.0166
                                  ).convergence_assumed is False


def test_an_assumed_convergence_warns_on_a_grid_survey():
    """Grid convergence ROTATES EVERY AZIMUTH. One degree is ~52 m of lateral
    position at 3 km of departure, which is anti-collision territory."""
    with pytest.warns(UserWarning, match="grid convergence was never established"):
        _survey()


def test_it_does_not_warn_when_the_value_was_given(recwarn):
    _survey(convergence=0.0)
    assert not [w for w in recwarn.list
                if "convergence" in str(w.message)]


def test_it_does_not_warn_where_convergence_cannot_change_the_answer(recwarn):
    """A true-referenced survey never applies it, so a warning there would be
    noise -- and a warning a reader learns to ignore is worse than none."""
    for ref in ("true", "magnetic"):
        _survey(azi_reference=ref)
    assert not [w for w in recwarn.list
                if "convergence was never established" in str(w.message)]


def test_the_assumed_value_is_still_zero_so_nothing_silently_moves():
    """Rule 2's counterweight: refuse where you would have to GUESS, do not
    refuse because the answer is uncomfortable. Zero remains the applied value;
    what changed is that it is now visible."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _survey()
        b = _survey(convergence=0.0)
    assert np.allclose(a.azi_true_deg, b.azi_true_deg)


def test_a_serialised_header_survives_a_new_attribute():
    """The header serialises its whole __dict__ and it was splatted straight
    back into the constructor, which made the SERIALISED FORM the constructor
    signature -- so any attribute added later broke every round trip of every
    previously written file."""
    from welleng.hierarchy import _survey_header_from_dict
    d = dict(we.survey.SurveyHeader(name="T", convergence=1.0).__dict__)
    d["some_future_field"] = 42
    h = _survey_header_from_dict(d)
    assert h.name == "T"
    assert h.convergence_assumed is False
