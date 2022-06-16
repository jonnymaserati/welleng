import json
import os
import unittest

import numpy as np

from welleng.clearance import ISCWSA, Clearance
from welleng.survey import Survey, make_survey_header

"""
Test that the ISCWSA clearance model is working within a defined tolerance
(the default has been set to 0.5%), testing against the ISCWSA standard
set of wellpaths for evaluating clearance scenarios using the MWD Rev4
error model.

https://www.iscwsa.net/files/156/

"""

# Set test tolerance as percentage
TOLERANCE = 0.5


class TestClearanceISCWSA(unittest.TestCase):

    def test_clearance_iscwsa(self):
        """
        This test checks if the welleng models are performing close to the standard
        ISCWSA model within a certain tolerance.
        """
        # Read well and validation data
        filename = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                'test_data', 'clearance_iscwsa_well_data.json'
            )
        )
        data = json.load(open(filename))

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
                    c = Clearance(reference, offset)

                result = ISCWSA(c)

                normalized = np.absolute(
                    np.array(result.SF) - np.array(data["wells"][well]["SF"])
                ) / np.array(data["wells"][well]["SF"]) * 100

                assert np.all(normalized < TOLERANCE)


def generate_surveys(data: dict) -> dict:

    """
    Extract surveys for all well in data json.
    """

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
