# ruff: noqa: E501 -- one Definition table row per line, so each row reads against the page.
"""Every error weighting function against ISCWSA Error Model Definition Rev 5.13, section 11.

The table below is transcribed from the Definition's weighting-function tables (Rev 5.13,
pp. 62-69: s11.1 common terms, s11.2 MWD, s11.3 gyro, s11.4 utility). Two implementations are
each asserted against it, station by station, on the ISCWSA test well 1 and on a synthetic
well that covers inclinations from 0.5 to 178 degrees and every azimuth quadrant:

- the hand-coded weight function (what the YAML tool models call), and
- the OWSG tool-model formula (what the formula interpreter evaluates), for every term of
  every shipped OWSG model.

Notation (Definition s4.3): I inclination, Am magnetic azimuth, At true azimuth, th dip, G
gravity, B total field, phi latitude, Om earth rate (7.292115e-5 rad/s), f noise reduction
factor. w12 = |sin I| and w34 = |cos I| (the OWSG misalignment weights), M = max(1,
sqrt(10 / dD)). The AXYZ-SF inclination weight is printed "1 3 sinI cosI"; it is read as 1.3.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import welleng as we
from welleng.error import ErrorModel
from welleng.errors import conformance as C
from welleng.errors import tool_errors as TE

s, c, t = np.sin, np.cos, np.tan


def den(v):  # 1 - sin^2 I sin^2 Am
    return 1 - s(v.I)**2 * s(v.Am)**2

def Z(v):
    return 0.0 * v.I


DEFINITION = {
 # 11.1
 "XYM1": (1,  lambda v: Z(v), lambda v: v.w12, lambda v: Z(v)),
 "XYM2": (2,  lambda v: Z(v), lambda v: Z(v), lambda v: -v.w12 / s(v.I)),
 "XYM3": (3,  lambda v: Z(v), lambda v: v.w34*c(v.At), lambda v: -v.w34*s(v.At)/s(v.I)),
 "XYM4": (4,  lambda v: Z(v), lambda v: v.w34*s(v.At), lambda v: v.w34*c(v.At)/s(v.I)),
 "SAG":  (5,  lambda v: Z(v), lambda v: s(v.I), lambda v: Z(v)),
 "DREF": (6,  lambda v: 1+Z(v), lambda v: Z(v), lambda v: Z(v)),
 "DSF":  (8,  lambda v: v.D, lambda v: Z(v), lambda v: Z(v)),
 "DST":  (9,  lambda v: v.D*v.TVD, lambda v: Z(v), lambda v: Z(v)),
 "XYM3E":(11, lambda v: Z(v), lambda v: v.M*v.w34*c(v.At), lambda v: -v.M*v.w34*s(v.At)/s(v.I)),
 "XYM4E":(12, lambda v: Z(v), lambda v: v.M*v.w34*s(v.At), lambda v: v.M*v.w34*c(v.At)/s(v.I)),
 "SAGE": (13, lambda v: Z(v), lambda v: s(v.I)**0.25, lambda v: Z(v)),
 # 11.2 MWD
 "ABXY_TI1": (18, lambda v: Z(v), lambda v: -c(v.I)/v.G, lambda v: t(v.th)*c(v.I)*s(v.Am)/v.G),
 "ABXY_TI2": (19, lambda v: Z(v), lambda v: Z(v), lambda v: (1/t(v.I) - t(v.th)*c(v.Am))/v.G),
 "ABZ":      (20, lambda v: Z(v), lambda v: -s(v.I)/v.G, lambda v: t(v.th)*s(v.I)*s(v.Am)/v.G),
 "ASXY_TI1": (21, lambda v: Z(v), lambda v: s(v.I)*c(v.I)/np.sqrt(2), lambda v: -t(v.th)*s(v.I)*c(v.I)*s(v.Am)/np.sqrt(2)),
 "ASXY_TI2": (22, lambda v: Z(v), lambda v: s(v.I)*c(v.I)/2, lambda v: -t(v.th)*s(v.I)*c(v.I)*s(v.Am)/2),
 "ASXY_TI3": (23, lambda v: Z(v), lambda v: Z(v), lambda v: (t(v.th)*s(v.I)*c(v.Am) - c(v.I))/2),
 "ASZ":      (24, lambda v: Z(v), lambda v: -s(v.I)*c(v.I), lambda v: t(v.th)*s(v.I)*c(v.I)*s(v.Am)),
 "MBXY_TI1": (25, lambda v: Z(v), lambda v: Z(v), lambda v: -c(v.I)*s(v.Am)/(v.B*c(v.th))),
 "MBXY_TI2": (26, lambda v: Z(v), lambda v: Z(v), lambda v: c(v.Am)/(v.B*c(v.th))),
 "MBZ":      (27, lambda v: Z(v), lambda v: Z(v), lambda v: -s(v.I)*s(v.Am)/(v.B*c(v.th))),
 "MSXY_TI1": (28, lambda v: Z(v), lambda v: Z(v), lambda v: s(v.I)*s(v.Am)*(t(v.th)*c(v.I)+s(v.I)*c(v.Am))/np.sqrt(2)),
 "MSXY_TI2": (29, lambda v: Z(v), lambda v: Z(v), lambda v: s(v.Am)*(t(v.th)*s(v.I)*c(v.I) - c(v.I)**2*c(v.Am) - c(v.Am))/2),
 "MSXY_TI3": (30, lambda v: Z(v), lambda v: Z(v), lambda v: (c(v.I)*c(v.Am)**2 - c(v.I)*s(v.Am)**2 - t(v.th)*s(v.I)*c(v.Am))/2),
 "MSZ":      (31, lambda v: Z(v), lambda v: Z(v), lambda v: -(s(v.I)*c(v.Am) + t(v.th)*c(v.I))*s(v.I)*s(v.Am)),
 "AMIL":     (32, lambda v: Z(v), lambda v: Z(v), lambda v: s(v.I)*s(v.Am)/(v.B*c(v.th))),
 "ABIXY_TI1":(33, lambda v: Z(v), lambda v: -c(v.I)/v.G, lambda v: c(v.I)**2*s(v.Am)*(t(v.th)*c(v.I)+s(v.I)*c(v.Am))/(v.G*den(v))),
 "ABIXY_TI2":(34, lambda v: Z(v), lambda v: Z(v), lambda v: -(t(v.th)*c(v.Am) - 1/t(v.I))/(v.G*den(v))),
 "ABIZ":     (35, lambda v: Z(v), lambda v: -s(v.I)/v.G, lambda v: s(v.I)*c(v.I)*s(v.Am)*(t(v.th)*c(v.I)+s(v.I)*c(v.Am))/(v.G*den(v))),
 "ASIXY_TI1":(36, lambda v: Z(v), lambda v: s(v.I)*c(v.I)/np.sqrt(2), lambda v: -s(v.I)*c(v.I)**2*s(v.Am)*(t(v.th)*c(v.I)+s(v.I)*c(v.Am))/(np.sqrt(2)*den(v))),
 "ASIXY_TI2":(37, lambda v: Z(v), lambda v: s(v.I)*c(v.I)/2, lambda v: -s(v.I)*c(v.I)**2*s(v.Am)*(t(v.th)*c(v.I)+s(v.I)*c(v.Am))/(2*den(v))),
 "ASIXY_TI3":(38, lambda v: Z(v), lambda v: Z(v), lambda v: (t(v.th)*s(v.I)*c(v.Am) - c(v.I))/(2*den(v))),
 "ASIZ":     (39, lambda v: Z(v), lambda v: -s(v.I)*c(v.I), lambda v: s(v.I)*c(v.I)**2*s(v.Am)*(t(v.th)*c(v.I)+s(v.I)*c(v.Am))/den(v)),
 "MBIXY_TI1":(40, lambda v: Z(v), lambda v: Z(v), lambda v: -c(v.I)*s(v.Am)/(v.B*c(v.th)*den(v))),
 "MBIXY_TI2":(41, lambda v: Z(v), lambda v: Z(v), lambda v: c(v.Am)/(v.B*c(v.th)*den(v))),
 "MSIXY_TI1":(42, lambda v: Z(v), lambda v: Z(v), lambda v: s(v.I)*s(v.Am)*(t(v.th)*c(v.I)+s(v.I)*c(v.Am))/(np.sqrt(2)*den(v))),
 "MSIXY_TI2":(43, lambda v: Z(v), lambda v: Z(v), lambda v: s(v.Am)*(t(v.th)*s(v.I)*c(v.I) - c(v.I)**2*c(v.Am) - c(v.Am))/(2*den(v))),
 "MSIXY_TI3":(44, lambda v: Z(v), lambda v: Z(v), lambda v: (c(v.I)*c(v.Am)**2 - c(v.I)*s(v.Am)**2 - t(v.th)*s(v.I)*c(v.Am))/(2*den(v))),
 "AZ":       (45, lambda v: Z(v), lambda v: Z(v), lambda v: 1+Z(v)),
 "DBH":      (46, lambda v: Z(v), lambda v: Z(v), lambda v: 1/(v.B*c(v.th))+Z(v)),
 "MFI":      (47, lambda v: Z(v), lambda v: Z(v), lambda v: -s(v.I)*s(v.Am)*(t(v.th)*c(v.I)+s(v.I)*c(v.Am))/(v.B*den(v))),
 "MDI":      (48, lambda v: Z(v), lambda v: Z(v), lambda v: -s(v.I)*s(v.Am)*(c(v.I) - t(v.th)*s(v.I)*c(v.Am))/den(v)),
 # 11.3 gyro (stationary / either mode); AXYZ-SF printed "1 3 sinI cosI"
 "AXYZ_XYB": (49, lambda v: Z(v), lambda v: c(v.I)/v.G, lambda v: Z(v)),
 "AXYZ_ZB":  (50, lambda v: Z(v), lambda v: s(v.I)/v.G, lambda v: Z(v)),
 "AXYZ_SF":  (51, lambda v: Z(v), lambda v: 1.3*s(v.I)*c(v.I), lambda v: Z(v)),
 "AXYZ_MIS": (52, lambda v: Z(v), lambda v: 1+Z(v), lambda v: Z(v)),
 "GXY_B1":   (70, lambda v: Z(v), lambda v: Z(v), lambda v: s(v.At)/(v.Om*c(v.phi)*c(v.I))),
 "GXY_B2":   (71, lambda v: Z(v), lambda v: Z(v), lambda v: c(v.At)/(v.Om*c(v.phi))),
 "GXY_RN":   (72, lambda v: Z(v), lambda v: Z(v), lambda v: v.f*np.sqrt(1 - c(v.At)**2*s(v.I)**2)/(v.Om*c(v.phi)*c(v.I))),
 "GXY_G1":   (73, lambda v: Z(v), lambda v: Z(v), lambda v: c(v.At)*s(v.I)/(v.Om*c(v.phi))),
 "GXY_G2":   (74, lambda v: Z(v), lambda v: Z(v), lambda v: c(v.At)*c(v.I)/(v.Om*c(v.phi))),
 "GXY_G3":   (75, lambda v: Z(v), lambda v: Z(v), lambda v: s(v.At)/(v.Om*c(v.phi))),
 "GXY_G4":   (76, lambda v: Z(v), lambda v: Z(v), lambda v: s(v.At)*t(v.I)/(v.Om*c(v.phi))),
 "EXT_REF":  (79, lambda v: Z(v), lambda v: Z(v), lambda v: 1+Z(v)),
 # 11.4 utility
 "CNA":      (88, lambda v: Z(v), lambda v: Z(v), lambda v: 1/s(v.I)),
 "CNI":      (89, lambda v: Z(v), lambda v: 1+Z(v), lambda v: Z(v)),
}
# 11.5 vertical substitutions (north, east, vertical) per unit sigma and unit length
VERTICAL = {
 "XYM3": lambda v: (1, 0, 0), "XYM4": lambda v: (0, 1, 0),
 "XYM3E": lambda v: (v.M, 0, 0), "XYM4E": lambda v: (0, v.M, 0),
 "ABXY_TI2": lambda v: (-s(v.Am)/v.G, c(v.Am)/v.G, 0),
 "ABIXY_TI2": lambda v: (-s(v.Am)/v.G, c(v.Am)/v.G, 0),
 "CNA": lambda v: (-s(v.Az), c(v.Az), 0),
}


JSON_ROOT = Path(__file__).parent.parent / "welleng" / "errors" / "iscwsa_json"
EARTH_RATE = 7.292115e-5 * 3600.0  # rad/hr, Definition s4.3
TOL = 1e-12


def _synthetic_survey():
    rng = np.random.default_rng(0)
    n = 400
    inc = np.clip(np.cumsum(rng.normal(0.9, 2.0, n)), 0.5, 178.0)
    azi = np.mod(np.cumsum(rng.normal(0.0, 6.0, n)) + 37.0, 360.0)
    md = np.concatenate(([0.0], np.cumsum(rng.uniform(3.0, 60.0, n - 1))))
    header = we.survey.SurveyHeader(
        name="synthetic", latitude=60.4, b_total=50300.0, dip=72.3,
        declination=2.1, convergence=-1.3, G=9.80665, azi_reference="true",
    )
    return we.survey.Survey(md=md, inc=inc, azi=azi, header=header)


SURVEYS = {"iscwsa_well_1": C.standard_test_survey(), "synthetic": _synthetic_survey()}


def _namespace(survey, noise_reduction_factor=1.0):
    md = np.asarray(survey.md, float)
    inc = np.asarray(survey.inc_rad, float)
    return SimpleNamespace(
        I=inc, Am=np.asarray(survey.azi_mag_rad, float),
        At=np.asarray(survey.azi_true_rad, float), Az=np.asarray(survey.azi_grid_rad, float),
        D=md, TVD=np.asarray(survey.tvd, float), th=float(survey.header.dip),
        G=float(survey.header.G), B=float(survey.header.b_total),
        phi=np.radians(float(survey.header.latitude)), Om=EARTH_RATE,
        f=noise_reduction_factor, w12=np.abs(np.sin(inc)), w34=np.abs(np.cos(inc)),
        M=np.maximum(1.0, np.sqrt(10.0 / np.diff(md, prepend=np.nan))),
    )


def _definition_eDIA(key, v):
    _, d, i, a = DEFINITION[key]
    n = len(v.I)
    return np.column_stack([np.broadcast_to(f(v), (n,)) for f in (d, i, a)])


def _regular(survey):
    # Away from the vertical singularity, and not station 0 (no previous station).
    regular = np.asarray(survey.inc_rad) >= survey.header.vertical_inc_limit
    regular[0] = False
    return regular


def _relative(got, ref, mask):
    got, ref = np.asarray(got, float)[mask], np.asarray(ref, float)[mask]
    return float(np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-300))


CODED = sorted(k for k in DEFINITION if getattr(TE, k, None) is not None)


@pytest.mark.parametrize("survey_name", sorted(SURVEYS))
@pytest.mark.parametrize("key", CODED)
def test_hand_coded_weight_matches_definition(key, survey_name):
    survey = SURVEYS[survey_name]
    shim = C._ShimError(survey, ErrorModel(survey, error_model="ISCWSA MWD Rev5.11"))
    getattr(TE, key)(key, shim, mag=1.0, propagation="systematic", NEV=True)
    got = shim._captured[key]
    ref = _definition_eDIA(key, _namespace(survey))
    assert _relative(got, ref, _regular(survey)) <= TOL, (
        f"{key} (Definition Rev 5.13 term {DEFINITION[key][0]}) differs on {survey_name}"
    )


def _owsg_terms():
    for path in sorted(glob.glob(str(JSON_ROOT / "owsg_*" / "*.json"))):
        model = json.load(open(path))
        for term in model["terms"]:
            key = (term.get("x_owsg_weight_function") or term["name"]).replace("-", "_")
            if key in DEFINITION:
                yield pytest.param(path, term["name"], key, id=f"{Path(path).name}:{term['name']}")


@pytest.mark.parametrize("path,name,key", list(_owsg_terms()))
def test_owsg_formula_matches_definition(path, name, key):
    model = json.load(open(path))
    params = model.get("parameters") or {}
    term = next(t for t in model["terms"] if t["name"] == name)
    for survey in SURVEYS.values():
        bindings = C._bindings_from_survey(survey)
        bindings.update({k: float(v) for k, v in params.items() if isinstance(v, (int, float))})
        bindings["NoiseReductionFactor"] = 1.0
        got = C._interp_eval_term(term, 1.0, survey, bindings)
        ref = _definition_eDIA(key, _namespace(survey, 1.0))
        assert _relative(got, ref, _regular(survey)) <= TOL, (
            f"{Path(path).name} {name} differs from Definition Rev 5.13 term {DEFINITION[key][0]}"
        )


def test_the_check_catches_a_wrong_weight():
    # Known positive: the ABIZ azimuth weight multiplied by G instead of divided by it.
    survey = SURVEYS["synthetic"]
    v = _namespace(survey)
    ref = _definition_eDIA("ABIZ", v)
    wrong = ref.copy()
    wrong[:, 2] *= v.G ** 2
    assert _relative(wrong, ref, _regular(survey)) > TOL
