"""ISCWSA Rotating 6-Axis Magnetic Survey Error Model V1 (2026-06-02).

Source: ISCWSA_Rotating_6axis_Magnetic_Survey_Error_Model_V1_2026-06.xlsx
(single sheet ``Rot_6Axis_MWD+SRGM``, prefix D007Ma), vendored under
``data/iscwsa/toolgroups/`` and converted to
``welleng/errors/iscwsa_json/rot6axis/`` by the standard OWSG converter.

No worked-example covariances are published for this model (the sheet's
notes mark the magnitudes as software-testing values), so these tests pin
structure and behaviour: name resolution, full term evaluation, vertical-
hole singularity handling, and a sanity band against the MWD Rev5.11 base
the model is derived from.
"""
import warnings

import numpy as np
import pytest

import welleng as we

MODEL = "ISCWSA Rot 6Axis MWD+SRGM"


def _survey(inc_max=60.0, error_model=MODEL):
    md = np.arange(0, 4001, 30.0)
    inc = np.clip((md - 500) / 40.0, 0, inc_max)
    azi = np.full_like(md, 45.0)
    sh = we.survey.SurveyHeader(
        name="t", latitude=53.0, longitude=4.0, b_total=49500.0, dip=67.0,
        declination=1.5, azi_reference="true",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return we.survey.Survey(
            md=md, inc=inc, azi=azi, header=sh, error_model=error_model
        )


def _smax(s):
    return np.sqrt(np.linalg.eigvalsh(np.asarray(s.cov_nev)[-1][:2, :2]).max())


def test_registered_and_resolves():
    from welleng.error import ERROR_MODELS
    assert MODEL in ERROR_MODELS
    s = _survey()
    assert len(s.err.errors.errors) == 38   # every sheet term evaluated


def test_sane_vs_rev511_base():
    """Derived from MWD Rev5+SRGM with rotation terms swapped in -- total
    should land near the base model, not orders off (guards unit/formula
    regressions)."""
    r = _smax(_survey()) / _smax(_survey(error_model="ISCWSA MWD Rev5.11"))
    assert 0.7 < r < 1.3


def test_vertical_hole_singularity_finite():
    """ROT-AN1/CA1/DSC2 carry 1/Sin(Inc)-type azimuth weights with N/E
    singularity replacements -- a fully vertical well must stay finite."""
    s = _survey(inc_max=0.0)
    cov = np.asarray(s.cov_nev)
    assert np.all(np.isfinite(cov))
    assert _smax(s) < 10.0


def test_magnetic_gate_applies():
    """tool_type Magnetic -> the mag-reference gate must refuse package
    default field values."""
    md = np.arange(0, 301, 30.0)
    sh = we.survey.SurveyHeader(name="t")   # no mag data, no location
    with pytest.raises(ValueError, match="magnetic model"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        we.survey.Survey(
            md=md, inc=np.zeros_like(md), azi=np.zeros_like(md),
            header=sh, error_model=MODEL,
        )
