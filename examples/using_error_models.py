'''
examples/using_error_models.py
------------------------------
An example to demonstrate how to generate and compare different error models
for a given wellbore survey, plotting the results for visual comparison and
also printing the covariance data at the well TD.

author: Jonny Corcutt
email: jonnycorcutt@gmail.com
date: 29-09-2021
'''
import numpy as np

import welleng as we
from welleng.error import make_diagnostic_data


def get_errors(error):
    """
    A helper function to extract the unique values from an error covariance
    matrix - since they're symetrical along the diagonal.
    """
    nn, ne, nv = error[0]
    _, ee, ev = error[1]
    _, __, vv = error[2]

    return [nn, ee, vv, ne, nv, ev]


def main():
    # get a list of available error models
    error_models = we.error.get_error_models()

    print("Error Models:")
    [print(f"{e}") for e in error_models]

    # create the ISCWSA diagnostic well path #1
    sh = we.survey.SurveyHeader(
        latitude=60.,
        longitude=2.,
        G=9.80665,
        b_total=50_000.,
        dip=72.,
        declination=-4.,
        azi_reference='true',
        earth_rate=0.26251614
    )

    s = we.survey.Survey(
        md=np.array([0., 1200., 2100., 5100., 5400., 8000.]),
        inc=np.array([0., 0., 60., 60., 90., 90.]),
        azi=np.array([0., 0., 75., 75., 75., 75.]),
        radius=0.32,
        header=sh,
        deg=True,
        unit="meters"
    ).interpolate_survey(step=30.)

    meshes = []

    # we'll compare the results of a 'regular' versus a more accurate survey
    for e in ['MWD+SRGM', 'MWD+IFR2+SAG+MS']:
        # here's the method to (re)allocate the error model associated to
        # the survey instance and the internal function to recalculate the
        # survey errors
        s.error_model = e
        s._get_errors()

        # extract the diagnostic data fom the error model (to compare with
        # the ISCWSA diagnostic data for example)
        diagnostic = make_diagnostic_data(s)

        print(f"\nError Model: {e}")
        print("Error at TD (1-sigma squared):")
        print([
            f"{l}: {v}"
            for (l, v) in list(zip(
                ['NN', 'EE', 'VV', 'NE', 'NV', 'EV'],
                diagnostic[8000.0]['TOTAL']
            ))
        ])

        meshes.append(we.mesh.WellMesh(s).mesh)

    # then plot them
    we.visual.plot(meshes, colors=['red', 'blue'])


if __name__ == "__main__":
    main()
    print("Done")
