import numpy as np
from welleng.version import __version__


def export_csv(survey, filename, rtol=0.1, atol=0.1, dls_cont=False):
    """
    Function to export a minimalist (only the control points - i.e. the
    begining and end points of hold and/or turn sections) survey to input into third
    party trajectory planning software.

    Parameters
    ----------
    survey: welleng.survey.Survey object
    filename: str
        The path and filename for saving the text file.
    rtol: float
        The relative tolerance used when comparing normal vectors
    atol: float
        The absolute tolerance used when comparing normal vectors
    dls_cont: bool
        Whether to explicitly check for dls continuity. May results in a
        larger number of control points but a trajectory that is a closer
        fit to the survey.
    """
    sections = survey._get_sections(rtol=rtol, atol=atol, dls_cont=dls_cont)

    data = [[
        tp.md,
        tp.inc,
        tp.azi,
        tp.location[1],
        tp.location[0],
        tp.location[2],
        tp.dls,
        tp.toolface,
        tp.build_rate,
        tp.turn_rate,
    ] for tp in sections]

    data = np.vstack(data[1:])

    headers = ','.join([
        'MD',
        'INC (deg)',
        'AZI (deg)',
        'NORTHING (m)',
        'EASTING (m)',
        'TVDSS (m)',
        'DLS',
        'TOOLFACE',
        'BUILD RATE',
        'TURN RATE'
    ])

    if filename is None:
        try:
            import pandas as pd

            df = pd.DataFrame(
                data,
                columns=headers.split(',')
            )
            return df
        except ImportError:
            print("Missing pandas dependency")

    comments = [
        f"welleng, version: {__version__}\n"
        f"author, Jonny Corcutt\n"
    ]
    comments.extend([
        f"{k}, {v}\n" for k, v in vars(survey.header).items()
    ])
    comments += f"\n"
    comments = ''.join(comments)

    np.savetxt(
        filename,
        data,
        delimiter=',',
        fmt='%.3f',
        header=headers,
        comments=comments
    )
