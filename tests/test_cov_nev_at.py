"""Arc-faithful interior covariance ``ErrorModel.cov_nev_at``.

Guards the interior-covariance evaluation used by clearance / pathfinder EOU:
- endpoint recovery (f->0 == cov_NEV[i], f->1 == cov_NEV[i+1]) to machine precision
  -- the regression test for the f->0 option-c bug (partial out-leg
  coupling was dropped);
- continuity across a station;
- the term classification (standard / course_length XCLA-XCLH / linear residual).
"""
import json

import numpy as np
import pytest

from welleng.survey import Survey, SurveyHeader

DATA = "tests/test_data/error_mwdrev5_1_iscwsa_data.json"
MODEL = "ISCWSA MWD Rev5.11"


@pytest.fixture(scope="module")
def err():
    wd = json.load(open(DATA))
    sh = SurveyHeader()
    for k, v in wd["header"].items():
        setattr(sh, k, v)
    s = Survey(
        md=wd["survey"]["md"], inc=wd["survey"]["inc"],
        azi=wd["survey"]["azi"], header=sh, error_model=MODEL,
    )
    return s.err


def test_endpoint_recovery(err):
    """Interior cov reproduces the stored station covs at both ends to machine
    precision -- both ends, not just f->1 (the f->0 coupling regression)."""
    md = err.survey_rad[:, 0]
    worst_1 = worst_0 = 0.0
    for i in range(len(md) - 1):
        eps = 1e-9 * max(1.0, md[i + 1])
        c1 = err.cov_nev_at(md[i + 1] - eps)
        ref1 = err.errors.cov_NEVs[i + 1]
        worst_1 = max(
            worst_1,
            np.max(np.abs(c1 - ref1)) / max(1e-9, np.max(np.abs(ref1))))
        c0 = err.cov_nev_at(md[i] + 1e-9 * max(1.0, md[i]))
        ref0 = err.errors.cov_NEVs[i]
        worst_0 = max(
            worst_0,
            np.max(np.abs(c0 - ref0)) / max(1e-9, np.max(np.abs(ref0))))
    assert worst_1 < 1e-7, f"f->1 recovery {worst_1:.2e}"
    assert worst_0 < 1e-7, f"f->0 recovery {worst_0:.2e} (partial-coupling regression)"


def test_continuity_across_station(err):
    """cov is continuous across a station: leg[i-1,i] at f->1 == leg[i,i+1] at f->0."""
    md = err.survey_rad[:, 0]
    i = 61  # a mid-build station (worst pre-fix case)
    a = err.cov_nev_at(md[i] - 1e-7)
    b = err.cov_nev_at(md[i] + 1e-7)
    scale = max(1e-9, np.max(np.abs(err.errors.cov_NEVs[i])))
    assert np.max(np.abs(a - b)) / scale < 1e-6


def test_interior_is_positive_semidefinite(err):
    """A covariance at an arbitrary interior point must stay PSD."""
    md = err.survey_rad[:, 0]
    for q in np.linspace(md[1], md[-2], 50):
        cov = err.cov_nev_at(q)
        assert np.allclose(cov, cov.T)
        assert np.min(np.linalg.eigvalsh(cov)) > -1e-9


def _build(rep):
    wd = json.load(open(DATA))
    sh = SurveyHeader()
    for k, v in wd["header"].items():
        setattr(sh, k, v)
    sh.xcl_representation = rep
    return Survey(
        md=wd["survey"]["md"], inc=wd["survey"]["inc"],
        azi=wd["survey"]["azi"], header=sh, error_model=MODEL,
    )


def test_xcl_dia_representation_matches_nev_direct():
    """The DIA (angle-error) recast of XCLA/XCLH reproduces the NEV-direct station
    covariance to machine precision (conformance), and exposes a samplable e_DIA
    perturbation (the MC-surface benefit) the NEV-direct form does not."""
    nev = _build("nev_direct").err
    dia = _build("dia").err
    assert np.max(np.abs(nev.errors.cov_NEVs - dia.errors.cov_NEVs)) < 1e-9
    for name, comp in (("XCLA", 2), ("XCLH", 1)):
        assert np.max(np.abs(
            nev.errors.errors[name].cov_NEV - dia.errors.errors[name].cov_NEV
        )) < 1e-9
        # DIA form carries a real inc/azi (DIA) perturbation -- the MC-samplable
        # quantity that folds XCL into the standard measurement-error surface.
        assert np.count_nonzero(dia.errors.errors[name].e_DIA[:, comp]) > 0


def test_term_classification(err):
    """XCLA/XCLH are course-length; the residual non-standard terms fall to linear."""
    classes, xcl_mag, _, _ = err._interior_prep()
    assert classes["XCLA"] == "course_length"
    assert classes["XCLH"] == "course_length"
    # XCLA/XCLH magnitude reconstructed from stored e_NEV (model-general,
    # = ISCWSA 0.167)
    assert abs(xcl_mag["XCLA"] - 0.167) < 1e-6
    assert abs(xcl_mag["XCLH"] - 0.167) < 1e-6
