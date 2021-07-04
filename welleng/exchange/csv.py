import numpy as np
from scipy.optimize import minimize
from ..version import __version__
from ..survey import export_csv as x_csv


def export_csv(
    survey, filename, tolerance=0.1, dls_cont=False, decimals=3, **kwargs
):
    """
    Wrapper for survey.export_csv

    Function to export a minimalist (only the control points - i.e. the
    begining and end points of hold and/or turn sections) survey to input into third
    party trajectory planning software.

    Parameters
    ----------
    survey: welleng.survey.Survey object
    filename: str
        The path and filename for saving the text file.
    tolerance: float (default: 0.1)
        How close the the final N, E, TVD position of the minimalist survey
        should be to the original survey point (e.g. within 1 meter)
    dls_cont: bool
        Whether to explicitly check for dls continuity. May result in a
        larger number of control points but a trajectory that is a closer
        fit to the survey.
    decimals: int (default: 3)
        Number of decimal places provided in the output file listing
    """

    return x_csv(
        survey, filename, tolerance=tolerance, dls_cont=dls_cont,
        decimals=decimals, **kwargs
    )
