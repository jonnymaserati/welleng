"""Unit-string handling for JSON-driven error terms.

The OWSG xlsx (and its JSON conversion) writes B-field-coupled angle
magnitudes as ``deg.nT`` (value in deg*nT; the DBH-family weight divides
by BField so the output is degrees). A silent miss in the unit table
treated these as dimensionless -- every DBH term evaluated 57.3x too big
(EMS+HRGM total horizontal ~25x) on the JSON/dict model path. The legacy
YAML tools ship pre-converted magnitudes and were never affected.
"""
import json
import os
import warnings

import numpy as np
import pytest

import welleng as we
from welleng.errors.tool_errors import _MAG_UNIT_TO_BASE, _unit_scale

JSON_DIR = os.path.join(
    os.path.dirname(we.errors.tool_errors.__file__), "iscwsa_json"
)


def _survey(model):
    md = np.arange(0, 4001, 30.0)
    inc = np.clip((md - 500) / 40.0, 0, 60)
    azi = np.full_like(md, 45.0)
    sh = we.survey.SurveyHeader(
        name="t", latitude=53.0, longitude=4.0, b_total=49500.0, dip=67.0,
        declination=1.5, azi_reference="true",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return we.survey.Survey(
            md=md, inc=inc, azi=azi, header=sh, error_model=model
        )


def test_deg_nt_unit_registered():
    assert _MAG_UNIT_TO_BASE["deg.nT"] == pytest.approx(np.pi / 180.0)


def test_unknown_unit_warns_not_silent():
    with pytest.warns(UserWarning, match="unknown error-term unit"):
        assert _unit_scale("furlong/fortnight") == 1.0


def test_dbh_term_scaled_to_radians():
    """DBH-family azimuth error = mag[deg*nT] * (pi/180) / (B*cos(dip)).

    Evaluated through the dict-model (JSON interpreter) path, which is the
    path exposed to ``deg.nT`` (legacy YAML magnitudes are pre-converted).
    """
    model = json.load(open(os.path.join(JSON_DIR, "owsg_b", "EMS+HRGM.json")))
    dbh_u = next(t for t in model["terms"] if t["name"] == "DBH-U")
    assert dbh_u["units"] == "deg.nT"
    s = _survey(model)
    term = s.err.errors.errors["DBH-U"]
    # sigma_e azimuth component at a high-inc station
    e_azi = np.asarray(term.e_DIA)[-1, 2] if hasattr(term, "e_DIA") else None
    expected = (
        dbh_u["value"] * np.pi / 180.0
        / (49500.0 * np.cos(np.radians(67.0)))
    )
    if e_azi is not None:
        assert e_azi == pytest.approx(expected, rel=1e-6)
    # end-to-end guard either way: total horizontal must be survey-grade,
    # not the ~550 m the dimensionless miss produced
    smax = np.sqrt(np.linalg.eigvalsh(np.asarray(s.cov_nev)[-1][:2, :2]).max())
    assert smax < 50.0
