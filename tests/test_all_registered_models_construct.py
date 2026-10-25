"""Every registered error model must construct and produce finite covariance.

Class-of-defect guard: a missing per-model parameter (e.g. the OWSG
INC-ONLY sheets omit the XCL Tortuosity cell while carrying XCLA/XCLH
terms) is invisible until someone picks that tool. Construct a Survey
with every Short Name in the registry on a gentle low-inclination well
(inside every model's inclination-range gate) with a full magnetic
reference (satisfies the mag gate).
"""
import warnings

import numpy as np
import pytest

import welleng as we
from welleng.error import ERROR_MODELS

# The published CB_Film gyro sheets give a continuous z-gyro drift term (GZ-GD) but no
# running speed for it, so the term cannot be evaluated. These models are refused,
# naming the missing parameter, rather than built with that term at zero uncertainty.
NOT_EVALUABLE = {
    "CB_Film_GMS", "CB_Film_GMS_Fl", "CB_Film_GSS", "CB_Film_GSS_Fl",
}


def test_models_missing_a_parameter_are_refused_naming_it():
    for model in sorted(NOT_EVALUABLE):
        with pytest.raises(ValueError, match="GZRunningSpeed"):
            _construct(model)


@pytest.mark.parametrize("model", sorted(set(ERROR_MODELS) - NOT_EVALUABLE))
def test_model_constructs_and_is_finite(model):
    cov = _construct(model)
    assert cov.shape == (34, 3, 3)
    assert np.all(np.isfinite(cov))
    # positive-semidefinite-ish sanity at TD
    assert np.all(np.linalg.eigvalsh(cov[-1]) >= -1e-8)


def _construct(model):
    md = np.arange(0, 1001, 30.0)
    inc = np.clip((md - 100) / 300.0, 0, 3.0)   # peaks at 3 deg
    azi = np.full_like(md, 45.0)
    sh = we.survey.SurveyHeader(
        name="t", latitude=53.0, longitude=4.0, b_total=49500.0, dip=67.0,
        declination=1.5, azi_reference="true",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = we.survey.Survey(
            md=md, inc=inc, azi=azi, header=sh, error_model=model
        )
        return np.asarray(s.cov_nev)
