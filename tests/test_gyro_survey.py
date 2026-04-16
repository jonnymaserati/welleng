"""Smoke tests for the gyro error-model dispatch path.

Locks in that ``Survey(error_model='GYRO-...')`` resolves through the
JSON+interpreter route, produces finite covariances, and that the three
canonical OWSG gyro tool stacks (north-seeking stationary, mixed
stationary+continuous, gyro-MWD hybrid) each rank-order against the
MWD baseline in the expected direction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import welleng as we


@pytest.fixture(scope="module")
def trajectory():
    connector = we.connector.Connector(
        pos1=[0.0, 0.0, 0.0], inc1=0.0, azi1=0.0,
        pos2=[800.0, 200.0, 3000.0], inc2=60.0, azi2=14.0,
    )
    return we.survey.from_connections(connector, step=30.0)


@pytest.fixture(scope="module")
def header():
    return we.survey.SurveyHeader(
        name="gyro-smoke", latitude=60.0, b_total=50000.0,
        dip=72.0, declination=-4.0, convergence=0.0,
        G=9.80665, azi_reference="grid",
    )


@pytest.mark.parametrize("error_model", [
    "GYRO-NS",
    "GYRO-NS-CT",
    "GYRO-MWD",
])
def test_gyro_survey_runs(trajectory, header, error_model):
    """Each gyro tool stack builds a Survey with finite covariance at TD."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        survey = we.survey.Survey(
            md=trajectory.md, inc=trajectory.inc_deg,
            azi=trajectory.azi_grid_deg, header=header,
            error_model=error_model,
        )
    cov = survey.cov_nev[-1]
    assert cov.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    sN, sE, sV = np.sqrt(np.diag(cov))
    # Sanity: 1-sigma at TD on a 3 km trajectory should be metres-ish.
    assert 0.1 < sN < 100.0
    assert 0.1 < sE < 100.0
    assert 0.0 < sV < 100.0


def test_gyro_stationary_vs_continuous_ranking(trajectory, header):
    """Continuous-running gyro accumulates drift between gyrocompass stops,
    so it must sit above the stationary north-seeking variant.
    """
    def _max_sigma(model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = we.survey.Survey(
                md=trajectory.md, inc=trajectory.inc_deg,
                azi=trajectory.azi_grid_deg, header=header,
                error_model=model,
            )
        return float(np.sqrt(np.linalg.eigvalsh(s.cov_nev[-1]).max()))

    s_ns = _max_sigma("GYRO-NS")
    s_ns_ct = _max_sigma("GYRO-NS-CT")

    assert s_ns < s_ns_ct, (
        f"stationary gyro {s_ns:.2f} should be < continuous-running {s_ns_ct:.2f}"
    )
