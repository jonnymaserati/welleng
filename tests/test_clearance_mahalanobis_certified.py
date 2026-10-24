"""
MahalanobisClearance.min_sf against a certified minimum on the ISCWSA standard
clearance set (Rev 5.11 error model; k = 3.5, sigma_pa = 0.5 m, Sm = 0.3 m;
radii 0.4572 m reference, 0.3048 m offset).

Reference: an independent branch-and-bound minimisation of the same separation
factor over the arcs and survey stations of both wells, which returns an
attained value and a certified lower bound on the minimum, stopping when no
region can improve on the attained value by more than 1e-6. Tolerance: the
minimum reported here lies between the lower bound and the attained value plus
1e-6. Scenario 10 is taken from surface (no kick-off depth); its minimum is 0
on the shared trunk.

Scenario 04's minimum sits exactly on a reference station (1860 m), on a kink
of the separation-factor surface; a search spanning the kink stalls 1.6e-5
above it.
"""
import json
import warnings

import numpy as np
import pytest

from welleng.clearance import MahalanobisClearance
from welleng.survey import Survey, SurveyHeader

with open("tests/test_data/clearance_iscwsa_well_data.json") as f:
    DATA = json.load(f)

HEADER = dict(latitude=60.0, b_total=50000.0, dip=70.0, declination=0.0,
              convergence=0.0, G=9.80665, azi_reference="grid",
              xcl_representation="dia")

# scenario: (certified lower bound, attained value)
CERTIFIED = {
    "01": (1.4312869029957904, 1.431287453531794),
    "02": (3.487214759048442, 3.4872156243218266),
    "03": (0.6073280083191959, 0.607329005168445),
    "04": (0.47845095954034284, 0.47845174683491093),
    "05": (1.559348598001059, 1.5593495360575207),
    "06": (1.0726608085880576, 1.072661783833021),
    "07": (2.053506545708857, 2.0535074030800935),
    "08": (1.664557558988737, 1.664558248296436),
    "09": (0.0, 0.0),
    "10": (0.0, 0.0),
    "11": (0.07865684857454565, 0.07865754714855562),
}
TOL = 1e-6


def _survey(name):
    w = DATA["wells"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Survey(
            md=w["MD"], inc=w["IncDeg"], azi=w["AziDeg"],
            n=w["N"], e=w["E"], tvd=w["TVD"],
            radius=0.4572 if name == "Reference well" else 0.3048,
            header=SurveyHeader(name=name, **HEADER),
            error_model="ISCWSA MWD Rev5.11",
        )


@pytest.fixture(scope="module")
def reference():
    return _survey("Reference well")


@pytest.mark.parametrize("scenario", sorted(CERTIFIED))
def test_min_sf_within_certified_bounds(reference, scenario):
    lower, attained = CERTIFIED[scenario]
    got = MahalanobisClearance(
        reference, _survey(f"{scenario} - well"), k=3.5, sigma_pa=0.5, Sm=0.3
    ).min_sf
    assert np.isfinite(got)
    assert lower - 1e-12 <= got <= attained + TOL, (scenario, got, lower, attained)
