"""ABIZ and AMIL weight functions against ISCWSA Error Model Definition Rev 5.13 s11.2.

Term 32, AMIL: azimuth weight sin I sin Am / (B cos dip).
Term 35, ABIZ: inclination weight -sin I / G; azimuth weight
sin I cos I sin Am (tan dip cos I + sin I cos Am) / (G (1 - sin^2 I sin^2 Am)).
Asserted station by station, 1e-12 relative, on a well covering 5-120 deg inclination
and every azimuth quadrant: ABIZ through the axial-interference-corrected MWD tool
model, AMIL through the uncorrected one (each model carries one of the two).
"""
import warnings

import numpy as np

import welleng as we

def _survey(model):
    n = 300
    md = np.linspace(0.0, 6000.0, n)
    inc = np.linspace(5.0, 120.0, n)
    azi = np.mod(np.linspace(10.0, 10.0 + 3 * 360.0, n), 360.0)
    header = we.survey.SurveyHeader(
        name="t", latitude=60.0, b_total=50300.0, dip=72.3, declination=2.1,
        convergence=0.0, G=9.80665, azi_reference="true",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return we.survey.Survey(
            md=md, inc=inc, azi=azi, header=header, error_model=model
        )


def _weights(survey, code):
    term = survey.err.errors.errors[code]
    mag = survey.err.errors.em["codes"][code]["magnitude"]
    return np.asarray(term.e_DIA) / mag


def _close(got, ref):
    return np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_abiz_matches_definition_term_35():
    survey = _survey("MWD+SRGM+AX")
    i, am = survey.inc_rad, survey.azi_mag_rad
    th, g = survey.header.dip, survey.header.G
    w = _weights(survey, "ABIZ")
    den = 1 - np.sin(i) ** 2 * np.sin(am) ** 2
    az = (np.sin(i) * np.cos(i) * np.sin(am)
          * (np.tan(th) * np.cos(i) + np.sin(i) * np.cos(am)) / (g * den))
    assert _close(w[:, 1], -np.sin(i) / g)
    assert _close(w[:, 2], az)


def test_amil_matches_definition_term_32():
    survey = _survey("MWD+SRGM")
    i, am = survey.inc_rad, survey.azi_mag_rad
    w = _weights(survey, "AMIL")
    ref = np.sin(i) * np.sin(am) / (survey.header.b_total * np.cos(survey.header.dip))
    assert _close(w[:, 2], ref)
