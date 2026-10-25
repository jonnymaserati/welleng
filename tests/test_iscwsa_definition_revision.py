"""The ISCWSA Error Model Definition revision applied (SurveyHeader.iscwsa_definition).

Between Rev 5.03 and Rev 5.13 the one change to a formula is the damping term M
of XYM3E/XYM4E (Definition s5.1.2 eq. (39)). Rev 5.03: M = max(1, sqrt(L_min / dD))
at every interval. Rev 5.13 (from Rev 5.12): that value only where
0.1 m < dD < L_min, else M = 1.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pytest

import welleng as we
from welleng.errors import tool_errors as TE

JSON_ROOT = Path(__file__).parent.parent / "welleng" / "errors" / "iscwsa_json"


def _header(revision="5.13"):
    return we.survey.SurveyHeader(
        name="t", latitude=53.0, b_total=49500.0, dip=67.0, declination=1.5,
        azi_reference="true", iscwsa_definition=revision,
    )


def _survey(md, revision="5.13", model="ISCWSA MWD Rev5.11"):
    md = np.asarray(md, dtype=float)
    inc = np.linspace(5.0, 40.0, len(md))
    azi = np.linspace(20.0, 80.0, len(md))
    return we.survey.Survey(
        md=md, inc=inc, azi=azi, header=_header(revision), error_model=model
    )


# Intervals 30, 0.05, 30, 5, 30, 12 m.
MD = np.cumsum([0.0, 30.0, 0.05, 30.0, 5.0, 30.0, 12.0])


def _damping(revision, header=None):
    return TE._misalignment_damping(_survey(MD, revision).err, header)


def test_damping_follows_the_revision():
    d = np.diff(MD)
    expected_503 = np.concatenate(([1.0], np.maximum(1.0, np.sqrt(10.0 / d))))
    expected_513 = np.where(
        np.concatenate(([False], (d > 0.1) & (d < 10.0))), expected_503, 1.0
    )
    np.testing.assert_array_equal(_damping("5.03"), expected_503)
    np.testing.assert_array_equal(_damping("5.13"), expected_513)
    # The two revisions differ only at the 0.05 m interval.
    assert np.flatnonzero(_damping("5.03") != _damping("5.13")).tolist() == [2]
    assert _damping("5.13")[2] == 1.0
    assert _damping("5.03")[2] == pytest.approx(np.sqrt(200.0), rel=1e-15)


def test_minimum_course_length_comes_from_the_tool_model():
    m = _damping("5.13", {"Misalignment Min Course Length": "40 m"})
    assert m[4] == pytest.approx(np.sqrt(40.0 / 5.0), rel=1e-15)
    assert m[6] == pytest.approx(np.sqrt(40.0 / 12.0), rel=1e-15)
    feet = _damping("5.13", {"Misalignment Min Course Length": "40 ft"})
    assert feet[4] == pytest.approx(np.sqrt(40.0 * 0.3048 / 5.0), rel=1e-15)


@pytest.mark.parametrize("model", ["ISCWSA MWD Rev5.11", "MWD+SRGM"])
def test_both_paths_apply_the_revision(model):
    # Hand-coded (YAML) and formula (JSON) models: the covariance moves between
    # revisions only through the 0.05 m interval's XYM3E/XYM4E terms.
    new = _survey(MD, "5.13", model).err.errors
    old = _survey(MD, "5.03", model).err.errors
    changed = {
        k for k in new.errors
        if not np.array_equal(new.errors[k].cov_NEV, old.errors[k].cov_NEV)
    }
    assert changed and changed <= {"XYM3E", "XYM4E", "XYM3L", "XYM4L"}


def test_a_repeated_depth_is_refused_only_where_rev_503_has_no_value():
    md = np.cumsum([0.0, 30.0, 0.0, 30.0])
    with pytest.raises(ValueError, match="iscwsa_definition"):
        TE._misalignment_damping(_survey(md, "5.03").err)
    assert TE._misalignment_damping(_survey(md, "5.13").err)[2] == 1.0


def test_an_unknown_revision_is_refused_naming_the_field():
    with pytest.raises(ValueError, match="iscwsa_definition"):
        _header("5.11")


def test_the_result_records_the_revision():
    current = _survey(MD, "5.13").err
    assert current.iscwsa_definition == "5.13"
    assert current.iscwsa_definition_superseded is False
    old = _survey(MD, "5.03").err
    assert old.iscwsa_definition == "5.03"
    assert old.iscwsa_definition_superseded is True


def test_every_sheet_damping_term_is_the_expression_replaced():
    # The revision reaches a formula model only where the sheet writes M as the
    # expression the interpreter replaces. A sheet writing it otherwise fails here.
    for path in glob.glob(str(JSON_ROOT / "owsg_*" / "*.json")):
        for term in json.load(open(path))["terms"]:
            fields = [term.get(f) for f in TE._DAMPED_FIELDS]
            text = " ".join(f for f in fields if isinstance(f, str))
            if "MDPrev" in text and term["name"].startswith("XYM"):
                assert TE._SHEET_DAMPING in text, f"{Path(path).name} {term['name']}"
                stripped = text.replace(TE._SHEET_DAMPING, "")
                assert "MDPrev" not in stripped, f"{Path(path).name} {term['name']}"
