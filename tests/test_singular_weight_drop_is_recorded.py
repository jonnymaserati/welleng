"""A vertical-hole weight that cannot be evaluated is recorded, never silently zero.

It contributes zero at the vertical stations, which understates the uncertainty, so it
is listed in ``ErrorModel.errors.dropped_terms`` with a warning, and refused under
``STRICT_TERMS``, as the main evaluation path does.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import welleng as we
from welleng.errors import tool_errors as TE

MODEL = Path(__file__).parent.parent / "welleng/errors/iscwsa_json/owsg_a/MWD+SRGM.json"


def _model_with_broken_singular_weight():
    model = json.loads(MODEL.read_text())
    for term in model["terms"]:
        if term["name"] == "XYM3E":
            term["north_singularity"] = "NotAVariable * 2"
    return model


def _survey(model):
    md = np.arange(0.0, 301.0, 30.0)
    inc = np.concatenate(([0.0, 0.0], np.linspace(1.0, 10.0, len(md) - 2)))
    header = we.survey.SurveyHeader(
        name="t", latitude=53.0, b_total=49500.0, dip=67.0, declination=1.5,
        azi_reference="true",
    )
    return we.survey.Survey(
        md=md, inc=inc, azi=np.full_like(md, 45.0), header=header,
        error_model=model,
    )


def test_a_failed_singular_weight_is_listed_and_warned():
    with pytest.warns(RuntimeWarning, match="vertical-hole weight"):
        s = _survey(_model_with_broken_singular_weight())
        s.cov_nev
    dropped = s.err.errors.dropped_terms
    assert [d.code for d in dropped] == ["XYM3E (north_singularity)"]
    assert dropped[0].missing == "NotAVariable"


def test_strict_terms_refuses_it(monkeypatch):
    monkeypatch.setattr(TE, "STRICT_TERMS", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="vertical-hole weight"):
            _survey(_model_with_broken_singular_weight()).cov_nev


def test_the_shipped_model_drops_nothing():
    s = _survey(json.loads(MODEL.read_text()))
    s.cov_nev
    assert s.err.errors.dropped_terms == []
