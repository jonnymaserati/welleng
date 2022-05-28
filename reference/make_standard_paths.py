"""
Code for generating the standard ISCWSA paths referenced in SPE-67616-PA in
order to utilise the diagnostic data provided in the paper to test software
implementation.

author: Jonathan Corcutt
"""

import os
import pickle

import welleng as we

PATH = os.path.dirname(os.path.abspath(__file__))


def main():

    # make ISCWSA No. 1
    md = [0., 1200.0, 2100., 5100., 5400., 8000.]
    inc = [0., 0., 60., 60., 90., 90.]
    azi = [0., 0., 75., 75., 75., 75.]

    sh = we.survey.SurveyHeader(
        name="iscwsa_1",
        latitude=60.,
        longitude=2.,
        G=9.80665,
        b_total=50_000.,
        dip=72.,
        declination=4.0,
    )

    survey = we.survey.Survey(
        md=md,
        inc=inc,
        azi=azi,
        header=sh
    )

    survey_interpolated = we.connector.interpolate_survey(survey, step=30)

    survey_interpolated.get_error(error_model='GYRO-MWD')

    filename = os.path.join(PATH, f"{sh.name}.pkl")

    with open(filename, 'wb') as f:
        pickle.dump(survey_interpolated, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

    print("Done")
