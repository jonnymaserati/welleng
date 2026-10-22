"""The propagation mode a tool-code file declares is the one the model applies.

A weight function that overrode the declared mode left two answers to one
question: ``ISCWSA MWD Rev5.11.yaml`` declared XYM2 random and XYM4E systematic,
the functions forced them back to systematic and random, and anything reading
the file got the swapped modes. Reference: the 'Prop.' column of the Model tab
of the ISCWSA MWD Rev 5.11 example workbooks (XYM2 'S', XYM4E 'R').
"""
import numpy as np
import pytest

from welleng.error import ERROR_MODELS
from welleng.survey import Survey, SurveyHeader


def _declared(errors):
    return {k: v["propagation"] for k, v in errors.em["codes"].items()}


@pytest.mark.parametrize("model", sorted(ERROR_MODELS))
def test_applied_propagation_is_the_declared_one(model):
    survey = Survey(
        md=np.array([0.0, 500.0, 1000.0, 1500.0]),
        inc=np.array([0.0, 2.0, 3.0, 4.0]),
        azi=np.array([0.0, 45.0, 135.0, 200.0]),
        header=SurveyHeader(b_total=50_000.0, dip=70.0, declination=1.0),
        error_model=model,
    )
    applied = {k: e.propagation for k, e in survey.err.errors.errors.items()}
    declared = _declared(survey.err.errors)
    differ = {
        k: (declared[k], applied[k]) for k in applied
        if k in declared and declared[k] != applied[k]
    }
    assert not differ, differ


def test_rev511_xym_modes_match_the_iscwsa_model_tab():
    survey = Survey(
        md=np.array([0.0, 500.0, 1000.0]), inc=np.array([0.0, 2.0, 3.0]),
        azi=np.array([0.0, 45.0, 135.0]),
        header=SurveyHeader(b_total=50_000.0, dip=70.0, declination=1.0),
        error_model="ISCWSA MWD Rev5.11",
    )
    declared = _declared(survey.err.errors)
    assert declared["XYM1"] == "systematic"
    assert declared["XYM2"] == "systematic"
    assert declared["XYM3E"] == "random"
    assert declared["XYM4E"] == "random"
