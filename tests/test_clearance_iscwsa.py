from welleng.survey import Survey, make_survey_header
from welleng.clearance import IscwsaClearance
import numpy as np
import json

"""
Test that the ISCWSA clearance model is working within a defined tolerance
(the default has been set to 0.5%), testing against the ISCWSA standard
set of wellpaths for evaluating clearance scenarios using the MWD Rev4
error model.
"""

# Set test tolerance as percentage
TOLERANCE = 0.3

# Read well and validation data
filename = (
    "tests/test_data/clearance_iscwsa_well_data.json"
)
data = json.load(open(filename))


def generate_surveys(data):
    # Generate surveys for imported wells
    surveys = {}

    for well in data['wells']:
        sh = make_survey_header(data["wells"][well]["header"])

        if well == "Reference well":
            radius = 0.4572
        else:
            radius = 0.3048

        s = Survey(
            md=data["wells"][well]["MD"],
            inc=data["wells"][well]["IncDeg"],
            azi=data["wells"][well]["AziDeg"],
            n=data["wells"][well]["N"],
            e=data["wells"][well]["E"],
            tvd=data["wells"][well]["TVD"],
            radius=radius,
            header=sh,
            error_model="ISCWSA MWD Rev4",
            start_xyz=[
                data["wells"][well]["E"][0],
                data["wells"][well]["N"][0],
                data["wells"][well]["TVD"][0]
                ],
            start_nev=[
                data["wells"][well]["N"][0],
                data["wells"][well]["E"][0],
                data["wells"][well]["TVD"][0]
                ],
            deg=True,
            unit="meters"
        )
        surveys[well] = s

    return surveys


def test_clearance_iscwsa(data=data, tolerance=TOLERANCE):
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]

    # Perform clearance checks for each survey
    for well in surveys:
        if well == "Reference well":
            continue
        else:
            offset = surveys[well]
            # skip well 10
            if well == "10 - well":
                continue
            else:
                result = IscwsaClearance(reference, offset, minimize_sf=True)

            normalized = np.absolute(
                result.sf[np.where(result.ref.interpolated == False)]
                - np.array(data["wells"][well]["SF"])
            ) / np.array(data["wells"][well]["SF"]) * 100

            assert np.all(normalized < tolerance)

            pass

# make above test runnanble separately
if __name__ == '__main__':
    test_clearance_iscwsa(data=data, tolerance=TOLERANCE)
