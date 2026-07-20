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


@pytest.mark.parametrize("model", sorted(ERROR_MODELS))
def test_model_constructs_and_is_finite(model):
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
        cov = np.asarray(s.cov_nev)
    assert cov.shape == (len(md), 3, 3)
    assert np.all(np.isfinite(cov))
    # positive-semidefinite-ish sanity at TD
    assert np.all(np.linalg.eigvalsh(cov[-1]) >= -1e-8)
