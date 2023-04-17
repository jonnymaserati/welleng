import json

import numpy as np

from welleng.clearance import IscwsaClearance
from welleng.survey import Survey, SurveyHeader, make_survey_header

"""
Test that the ISCWSA clearance model is working within a defined tolerance,
testing against the ISCWSA standard set of wellpaths for evaluating clearance
scenarios using the MWD Rev4 error model.
"""

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


SURVEYS = generate_surveys(data)


# TODO: get some decent diagnostic data here to test.
def test_min_curve():
    """
    Test that the minimum curvature calcs are not broken by comparing a
    calculated trajectory with diagnostic survey data.
    """
    for diagnostic, survey in zip(
        [well for well in data.get('wells').values()], SURVEYS.values()
    ):
        assert (
            np.allclose(
                np.array(diagnostic.get('TVD')), np.around(survey.tvd, 2)
            )
        ), "TVD calculation"

        assert (
            np.allclose(
                np.array(diagnostic.get('N')), np.around(survey.n, 2)
            )
        ), "North calculation"

        assert (
            np.allclose(
                np.array(diagnostic.get('E')), np.around(survey.e, 2)
            )
        ), "East calculation"

        pass

    pass


def test_modified_tortuosity_index(figure=False):
    s = SURVEYS.get('11 - well')

    md, inc, azi = np.array([
        [0, 0, 0],
        [100, 0, 0],
        [1000, 30, 45],
        [2000, 30, 45],
        [3000, 0, 45]
    ]).T

    s = Survey(md, inc, azi).interpolate_survey(step=30)

    mti = s.modified_tortuosity_index(
        data=True,
        mode='actual',
        # dls_tol=None, step=30, dls_noise=None
    )

    if figure:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=mti.get('survey').md,
                y=mti.get('mti')
            )
        )

        fig.show()

    pass


# make above test runnanble separately
if __name__ == '__main__':
    test_min_curve()
    test_modified_tortuosity_index()
