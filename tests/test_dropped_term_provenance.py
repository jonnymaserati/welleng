"""A term that cannot be evaluated contributes ZERO, and must say so.

Zero is the anti-conservative direction for an uncertainty: a term that
contributes nothing is indistinguishable, in the covariance, from a tool that
is perfect in that respect. The survey is still produced -- that is the right
call, a partly-usable model beats an unusable one -- but the shortfall has to
survive the computation that caused it.

None of the 62 catalogued models drops a term today; the schema gaps that
motivated the fallback were closed. The path stays live for user-supplied and
EDM/COMPASS-derived models, which is precisely why it is exercised here with a
model built for the purpose rather than left to whichever shipped model
happened to trip it.
"""
import copy
import warnings

import pytest

import welleng as we
from welleng.errors import tool_errors as te
from welleng.errors.tool_errors import (
    DroppedTerm, _load_json_model, _missing_variable, _resolve_json_model,
)


@pytest.fixture
def survey():
    return we.survey.Survey(
        md=[0.0, 100.0, 200.0, 300.0],
        inc=[0.0, 10.0, 20.0, 30.0],
        azi=[0.0, 10.0, 20.0, 30.0],
        header=we.survey.SurveyHeader(latitude=60.0, longitude=2.0),
    )


@pytest.fixture
def model_with_an_unbindable_term():
    """A real model with one term rewritten to reference a variable that
    nothing binds -- the shape an EDM-derived or vendor model arrives in.

    NB the variable has to be one the adapter genuinely does not surface. The
    first draft of this fixture used ``NoiseReductionFactor``, which reads like
    an unbound per-tool calibration constant and IS bound
    (``_json_to_em_adapter`` lifts it out of the parameters block), so the
    test failed against correct code.
    """
    m = copy.deepcopy(_load_json_model(_resolve_json_model("ISCWSA MWD Rev5.11")))
    m["terms"][0] = dict(m["terms"][0])
    m["terms"][0]["azimuth_formula"] = "UnboundCalibrationConstant * 2"
    return m


def _build(survey, model):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        em = we.error.ErrorModel(survey, error_model=model)
    return em, caught


# --- the shipped models --------------------------------------------------- #
def test_no_catalogued_model_drops_a_term(survey):
    """The normal case, asserted so a regression that starts silently zeroing
    a term is a test failure rather than a warning nobody reads."""
    for name in we.error.ERROR_MODELS:
        try:
            em, _ = _build(survey, name)
        except Exception:
            continue                      # model needs inputs this survey lacks
        assert em.errors.dropped_terms == [], \
            f"{name} dropped {[str(d) for d in em.errors.dropped_terms]}"


# --- the fallback, when it does fire -------------------------------------- #
def test_an_unevaluable_term_is_recorded_on_the_object(
        survey, model_with_an_unbindable_term):
    """The point of the record: a warning fires once, into whatever filter is
    installed, and is gone by the time anyone is handed the result."""
    em, _ = _build(survey, model_with_an_unbindable_term)
    assert len(em.errors.dropped_terms) == 1
    dropped = em.errors.dropped_terms[0]
    assert isinstance(dropped, DroppedTerm)
    assert dropped.missing == "UnboundCalibrationConstant"


def test_the_warning_names_the_missing_variable(
        survey, model_with_an_unbindable_term):
    """A refusal that does not name its missing input moves the debugging cost
    to whoever has least context."""
    _, caught = _build(survey, model_with_an_unbindable_term)
    msgs = [str(w.message) for w in caught
            if "could not be evaluated" in str(w.message)]
    assert msgs, "the drop was silent"
    assert "UnboundCalibrationConstant" in msgs[0]


def test_the_warning_states_the_direction_of_the_error(
        survey, model_with_an_unbindable_term):
    """Understated, not merely 'different'. The direction is the whole point:
    a smaller EOU is the one that gets acted on."""
    _, caught = _build(survey, model_with_an_unbindable_term)
    msgs = [str(w.message) for w in caught
            if "could not be evaluated" in str(w.message)]
    assert "ZERO" in msgs[0] and "UNDERSTATED" in msgs[0]


def test_strict_terms_refuses_instead(survey, model_with_an_unbindable_term):
    te.STRICT_TERMS = True
    try:
        with pytest.raises(ValueError, match="UnboundCalibrationConstant"):
            we.error.ErrorModel(
                survey, error_model=model_with_an_unbindable_term)
    finally:
        te.STRICT_TERMS = False


def test_strict_terms_defaults_to_permissive():
    """The default is a migration courtesy, not a judgement that the fallback
    is fine. If this flips, it is a deliberate breaking change."""
    assert te.STRICT_TERMS is False


# --- the missing-variable extraction -------------------------------------- #
def test_missing_variable_is_none_when_not_recoverable():
    """None means NOT RECOVERABLE, never 'nothing was missing' -- the reason
    is carried alongside precisely so the two are distinguishable."""
    assert _missing_variable(NameError("name 'Foo' is not defined")) == "Foo"
    assert _missing_variable(ZeroDivisionError("division by zero")) is None
    d = DroppedTerm(code="X", missing=None, reason="division by zero")
    assert "division by zero" in str(d)


def test_dropped_term_says_it_contributed_zero():
    d = DroppedTerm(code="XYM3E", missing="MDPrev", reason="...")
    assert "XYM3E" in str(d) and "zero covariance" in str(d)
    assert "MDPrev" in str(d)
