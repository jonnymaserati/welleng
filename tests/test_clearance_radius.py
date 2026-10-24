"""
The clearance takes each well's radius from its Survey. A radius passed to the
clearance as well (Rr, Ro) is checked against the survey's and refused when it
disagrees: before, it was accepted and silently ignored, because Survey always
populates radius.

Wells: the ISCWSA standard clearance set (reference well radius 0.4572 m,
offset 0.3048 m, as the set states).
"""
import json

import numpy as np
import pytest

from welleng.clearance import IscwsaClearance, MahalanobisClearance
from welleng.survey import Survey, make_survey_header

with open("tests/test_data/clearance_iscwsa_well_data.json") as f:
    DATA = json.load(f)

R_REF, R_OFF = 0.4572, 0.3048
CLASSES = [IscwsaClearance, MahalanobisClearance]


def _survey(well, radius):
    w = DATA["wells"][well]
    return Survey(
        md=w["MD"], inc=w["IncDeg"], azi=w["AziDeg"],
        n=w["N"], e=w["E"], tvd=w["TVD"],
        radius=radius,
        header=make_survey_header(w["header"]),
        error_model="ISCWSA MWD Rev4",
        start_xyz=[w["E"][0], w["N"][0], w["TVD"][0]],
        start_nev=[w["N"][0], w["E"][0], w["TVD"][0]],
        deg=True, unit="meters",
    )


@pytest.fixture(scope="module")
def wells():
    return _survey("Reference well", R_REF), _survey("07 - well", R_OFF)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("kw", [{"Rr": 5.0}, {"Ro": 5.0}, {"Rr": R_OFF}, {"Ro": R_REF}])
def test_disagreeing_radius_is_refused(wells, cls, kw):
    ref, off = wells
    name = next(iter(kw))
    with pytest.raises(ValueError, match=name):
        cls(ref, off, **kw)


@pytest.mark.parametrize("cls", CLASSES)
def test_wrong_length_radius_is_refused(wells, cls):
    ref, off = wells
    with pytest.raises(ValueError, match="Rr"):
        cls(ref, off, Rr=np.full(3, R_REF))


@pytest.mark.parametrize("cls", CLASSES)
def test_agreeing_radius_is_bitwise_the_same(wells, cls):
    ref, off = wells
    a = cls(ref, off)
    b = cls(ref, off, Rr=R_REF, Ro=np.full(len(off.md), R_OFF))
    assert np.array_equal(a.sf, b.sf, equal_nan=True)
    assert np.array_equal(a.Rr, ref.radius) and np.array_equal(a.Ro, off.radius)


@pytest.mark.parametrize("cls", CLASSES)
def test_survey_radius_is_the_one_used(wells, cls):
    # Opposite conditions report differently: a larger hole on the surveys
    # must lower the separation factor (it enters as -(Rr + Ro + Sm)).
    ref, off = wells
    small = np.nanmin(cls(ref, off).sf)
    big = np.nanmin(cls(_survey("Reference well", 5.0), _survey("07 - well", 5.0)).sf)
    assert big < small
