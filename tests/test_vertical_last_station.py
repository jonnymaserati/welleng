"""A survey ending vertical carries its last leg like any other.

The vertical-hole (singular) weights of the lateral terms gave every station its own
term over the leg into it except the last, which was left zero, so a survey ending
vertical took half the error of its final leg. Reference: the ISCWSA standard
clearance well set (r4, May 2017), offset 09, a vertical well whose last leg is 10 m
(2190 -> 2200 m): the set's sigmaH rises 3.9253 -> 3.9432 m, the same 0.0018 m/m
as its 30 m legs.
"""
import numpy as np

from welleng.survey import Survey, SurveyHeader


def _vertical(md, model):
    n = len(md)
    return Survey(md=np.asarray(md, float), inc=np.zeros(n), azi=np.zeros(n),
                  header=SurveyHeader(b_total=50_000.0, dip=70.0, declination=0.0),
                  error_model=model)


def test_a_vertical_last_leg_grows_at_the_rate_of_the_legs_before():
    # Rev 4's vertical lateral is systematic (XYM3, XYM4), so it grows linearly in
    # MD and a 10 m last leg adds a third of a 30 m one. (Rev 5's is mostly random
    # with a short-leg amplifier, which this rate check does not describe.)
    s = _vertical(np.r_[np.arange(0.0, 2191.0, 30.0), 2200.0], "ISCWSA MWD Rev4")
    h = np.sqrt(np.asarray(s.cov_hla)[:, 0, 0])
    per_m_before = (h[-2] - h[-3]) / 30.0
    per_m_last = (h[-1] - h[-2]) / 10.0
    assert abs(per_m_last / per_m_before - 1.0) < 0.05, (per_m_before, per_m_last)


def test_the_clearance_set_offset_09_ends_on_its_listed_sigma():
    import json
    with open("tests/test_data/clearance_iscwsa_well_data.json") as f:
        w = json.load(f)["wells"]["09 - well"]
    s = _vertical(w["MD"], "ISCWSA MWD Rev4")
    h = np.sqrt(np.asarray(s.cov_hla)[:, 0, 0])
    # the set's tie-station convention differs from ISCWSA's Rev 4 workbook by
    # under 1 mm (0.95 mm at most over the set); the half-leg miss was 9 mm
    assert abs(h[-1] - w["sigmaH"][-1]) < 1.0e-3
    assert abs((h[-1] - h[-2]) - (w["sigmaH"][-1] - w["sigmaH"][-2])) < 1.0e-4
