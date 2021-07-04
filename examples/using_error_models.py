import welleng as we
import numpy as np


def get_errors(error):
    nn, ne, nv = error[0]
    _, ee, ev = error[1]
    _, __, vv = error[2]

    return [nn, ee, vv, ne, nv, ev]


def make_diagnostic_data(survey):
    diagnostic = {}
    dia = np.stack((survey.md, survey.inc_deg, survey.azi_grid_deg), axis=1)
    for i, d in enumerate(survey.md):
        diagnostic[d] = {}
        total = []
        for k, v in survey.err.errors.errors.items():
            diagnostic[d][k] = get_errors(v.cov_NEV.T[i])
            total.extend(diagnostic[d][k])
        diagnostic[d]['TOTAL'] = np.sum((np.array(
            total
        ).reshape(-1, len(diagnostic[d][k]))), axis=0)
    return diagnostic


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

    s = we.connector.interpolate_survey(we.survey.Survey(
        md=np.array([0., 1200., 2100., 5100., 5400., 8000.]),
        inc=np.array([0., 0., 60., 60., 90., 90.]),
        azi=np.array([0., 0., 75., 75., 75., 75.]),
        radius=0.32,
        header=sh,
        deg=True,
        unit="meters"
    ), step=30.)

    meshes = []

    # we'll compare the results of a 'regular' versus a more accurate survey
    for e in ['MWD+SRGM', 'MWD+IFR2+SAG+MS_Fl']:
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
