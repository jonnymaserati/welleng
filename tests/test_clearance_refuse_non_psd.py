"""A covariance with a negative variance is refused at clearance input.

Before the check, a reference covariance diag(1, 1, -0.5) m^2 with the offset
10 m along the negative axis gave a Mahalanobis separation factor of +inf
(clear) with sigma_pa = 0, 0.0 with sigma_pa = 0.5, and NaN on the pedal path.
The same matrix with +0.5 gives finite factors. The two must report
differently: the PSD one is computed, the other refused naming the field and
the station.
"""
import numpy as np
import pytest

from welleng.clearance import (
    IscwsaClearance, MahalanobisClearance, _check_psd, combined_cov_mesh,
)
from welleng.survey import Survey, SurveyHeader

MAG_REF = dict(b_total=50_000., dip=72., declination=-2.)


def _survey(east, last):
    """A vertical well 0-300 m with bare covariances; ``last`` at the end."""
    md = np.array([0.0, 100.0, 200.0, 300.0])
    cov = np.stack([np.eye(3) * 0.1 * (i + 1) for i in range(3)] + [last])
    return Survey(md=md, inc=np.zeros(4), azi=np.zeros(4),
                  header=SurveyHeader(**MAG_REF), start_nev=[0.0, east, 0.0],
                  cov_nev=cov)


BAD = np.diag([1.0, 1.0, -0.5])
GOOD = np.diag([1.0, 1.0, 0.5])


@pytest.mark.parametrize("cls", [IscwsaClearance, MahalanobisClearance])
@pytest.mark.parametrize("sigma_pa", [0.0, 0.5])
def test_negative_variance_in_reference_is_refused(cls, sigma_pa):
    with pytest.raises(ValueError, match=r"reference\.cov_nev .* station 3"):
        cls(_survey(0.0, BAD), _survey(10.0, GOOD), sigma_pa=sigma_pa)


@pytest.mark.parametrize("cls", [IscwsaClearance, MahalanobisClearance])
def test_negative_variance_in_offset_is_refused(cls):
    with pytest.raises(ValueError, match=r"offset\.cov_nev .* station 3"):
        cls(_survey(0.0, GOOD), _survey(10.0, BAD))


@pytest.mark.parametrize("cls", [IscwsaClearance, MahalanobisClearance])
@pytest.mark.parametrize("sigma_pa", [0.0, 0.5])
def test_positive_semidefinite_input_is_computed(cls, sigma_pa):
    c = cls(_survey(0.0, GOOD), _survey(10.0, GOOD), sigma_pa=sigma_pa)
    assert np.all(np.isfinite(np.ravel(c.sf)))


def test_combined_cov_mesh_refuses_negative_variance():
    with pytest.raises(ValueError, match=r"other\.cov_nev .* station 3"):
        combined_cov_mesh(_survey(0.0, GOOD), _survey(10.0, BAD))


def test_rounding_of_a_psd_sum_passes():
    """An outer-product sum whose smallest eigenvalue is negative only by
    rounding is accepted; a negative variance a millionth of the largest is
    not."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(2, 3)) * 10.0
    c = v.T @ v                     # rank 2: one eigenvalue is zero, +/- rounding
    assert np.linalg.eigvalsh(c)[0] < 1e-12 * np.linalg.eigvalsh(c)[-1]
    _check_psd(c, "c")
    lam = np.linalg.eigvalsh(c)[-1]
    with pytest.raises(ValueError, match="smallest eigenvalue"):
        _check_psd(c - 1e-6 * lam * np.eye(3), "c")


def test_non_finite_is_refused():
    c = np.eye(3)
    c[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _check_psd(c, "c")
