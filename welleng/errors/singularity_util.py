from math import pi
from typing import Tuple

import numpy as np
from numpy import cos, sin, sqrt

from welleng.units import TORTUOSITY_RAD_PER_M


def calculate_error_singularity(
        sing: bool,
        error: 'Error',
        e_DIA: np.ndarray,
        magnitude: float,
        code: str,
        propagation: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function calcualtes the e_DIA, e_NEV, and e_NEV_star for all terms that include a singularity. This is a
    workaround to adjust for the lateral vector types in the code. These functions replicate the approaches used by
    welleng to handle these specific error terms.
    """

    if "abxy" in code.lower() or "abixy" in code.lower():
        e_DIA, e_NEV, e_NEV_star = singularity_abxy(sing, error, e_DIA, magnitude)

    elif code.lower() == "cna":
        e_DIA, e_NEV, e_NEV_star = singularity_cna(sing, error, e_DIA, magnitude)
    elif code.lower() == "xym3":
        e_DIA, e_NEV, e_NEV_star = singularity_xym3(sing, error, e_DIA, magnitude)

    elif code.lower() == "xym3e":
        e_DIA, e_NEV, e_NEV_star = singularity_xym3e(sing, error, e_DIA, magnitude)

    elif code.lower() == "xym3l":
        e_DIA, e_NEV, e_NEV_star = singularity_xym3l(sing, error, e_DIA, magnitude)

    elif code.lower() == "xym4":
        e_DIA, e_NEV, e_NEV_star = singularity_xym4(sing, error, e_DIA, magnitude)

    elif code.lower() == "xym4e":
        e_DIA, e_NEV, e_NEV_star = singularity_xym4e(sing, error, e_DIA, magnitude, propagation)

    elif code.lower() == "xym4l":
        e_DIA, e_NEV, e_NEV_star = singularity_xym4l(sing, error, e_DIA, magnitude)

    elif code.lower() == "xcla":
        e_DIA, e_NEV, e_NEV_star = singularity_xcla(error, magnitude)

    else:
        raise AttributeError(f"error term {code.lower()} not in the list of error codes included for singularity")

    return e_DIA, e_NEV, e_NEV_star


def singularity_abxy(sing, error, e_DIA, mag):
    e_NEV = error._e_NEV(e_DIA)
    n = np.array(
        0.5 * error.drdp_sing['double_delta_md']
        * -sin(error.drdp_sing['azi2']) * mag
    ) / error.survey.header.G
    e = np.array(
        0.5 * error.drdp_sing['double_delta_md']
        * cos(error.drdp_sing['azi2']) * mag
    ) / error.survey.header.G
    v = np.zeros_like(n)
    e_NEV_sing = np.vstack(
        (
            np.zeros((1, 3)),
            np.stack((n, e, v), axis=-1),
            np.zeros((1, 3))
        )
    )

    e_NEV_sing[1, 1] = (
        (
            error.survey.md[2]
            + error.survey.md[1]
            - 2 * error.survey.md[0]
        ) / 2
        * mag * cos(error.survey.azi_true_rad[1])
        / error.survey.header.G
    )
    e_NEV[sing] = e_NEV_sing[sing]

    e_NEV_star = error._e_NEV_star(e_DIA)
    n = np.array(
        0.5 * error.drdp_sing['delta_md']
        * -sin(error.drdp_sing['azi2']) * mag
    ) / error.survey.header.G
    e = np.array(
        0.5 * error.drdp_sing['delta_md']
        * cos(error.drdp_sing['azi2']) * mag
    ) / error.survey.header.G
    v = np.zeros_like(n)
    e_NEV_star_sing = np.vstack(
        (
            np.zeros((1, 3)),
            np.stack((n, e, v), axis=-1),
            np.zeros((1, 3))
        )
    )

    e_NEV_star_sing[1, 1] = (
        (
            error.survey.md[1] - error.survey.md[0]
        ) * mag
        * (
            cos(error.survey.azi_true_rad[1])
            / error.survey.header.G
        )
    )
    e_NEV_star[sing] = e_NEV_star_sing[sing]

    return e_DIA, e_NEV, e_NEV_star


def singularity_cna(sing, error, e_DIA, mag):
    e_NEV = error._e_NEV(e_DIA)
    n = (
        np.array(0.5 * error.drdp_sing['double_delta_md'])
        * -sin(getattr(error.survey, f"azi_{error.survey.header.azi_reference}_rad")[1: -1])
        * mag
    )
    e = (
        np.array(0.5 * error.drdp_sing['double_delta_md'])
        * cos(getattr(error.survey, f"azi_{error.survey.header.azi_reference}_rad")[1: -1])
        * mag
    )
    v = np.zeros_like(n)
    e_NEV_sing = np.vstack(
        (
            np.zeros((1, 3)),
            np.stack((n, e, v), axis=-1),
            np.zeros((1, 3))
        )
    )
    e_NEV[sing] = e_NEV_sing[sing]

    e_NEV_star = error._e_NEV_star(e_DIA)
    n = (
        np.array(0.5 * error.drdp_sing['delta_md'])
        * -sin(getattr(error.survey, f"azi_{error.survey.header.azi_reference}_rad")[1: -1])
        * mag
    )
    e = (
        np.array(0.5 * error.drdp_sing['delta_md'])
        * cos(getattr(error.survey, f"azi_{error.survey.header.azi_reference}_rad")[1: -1])
        * mag
    )
    e_NEV_star_sing = np.vstack(
        (
            np.zeros((1, 3)),
            np.stack((n, e, v), axis=-1),
            np.zeros((1, 3))
        )
    )
    e_NEV_star[sing] = e_NEV_star_sing[sing]

    return e_DIA, e_NEV, e_NEV_star


def singularity_xym3(sing, error, e_DIA, mag):
    e_NEV = error._e_NEV(e_DIA)
    n = np.array(0.5 * error.drdp_sing['double_delta_md'] * mag)
    e = np.zeros(len(error.drdp_sing['double_delta_md']))
    v = np.zeros_like(n)
    e_NEV_sing = np.vstack(
        (
            np.zeros((1, 3)),
            np.stack((n, e, v), axis=-1),
            np.zeros((1, 3))
        )
    )
    e_NEV[sing] = e_NEV_sing[sing]

    e_NEV_star = error._e_NEV_star(e_DIA)
    n = np.array(0.5 * error.drdp_sing['delta_md'] * mag)
    e = np.zeros(len(error.drdp_sing['delta_md']))
    v = np.zeros_like(n)
    e_NEV_star_sing = np.vstack(
        (
            np.zeros((1, 3)),
            np.stack((n, e, v), axis=-1),
            np.zeros((1, 3))
        )
    )
    e_NEV_star[sing] = e_NEV_star_sing[sing]
    return e_DIA, e_NEV, e_NEV_star


def singularity_xym3e(sing, error, e_DIA, mag):
    e_NEV = error._e_NEV(e_DIA)
    e_NEV_sing = np.zeros_like(e_NEV)
    e_NEV_sing[:, 0] = e_NEV[:, 0]
    e_NEV[sing] = e_NEV_sing[sing]

    e_NEV_star = error._e_NEV_star(e_DIA)
    e_NEV_star_sing = np.zeros_like(e_NEV_star)
    e_NEV_star_sing[:, 0] = e_NEV_star[:, 0]
    e_NEV_star[sing] = e_NEV_star_sing[sing]
    return e_DIA, e_NEV, e_NEV_star


def singularity_xym3l(sing, error, e_DIA, mag):
    coeff = np.ones(len(error.survey.md) - 1)
    coeff = np.amax(np.stack((
        coeff,
        sqrt(
            10 / (error.survey.md[1:] - error.survey.md[:-1])
        )
    ), axis=-1), axis=-1)

    e_NEV = error._e_NEV(e_DIA)
    e_NEV_sing = np.zeros_like(e_NEV)
    e_NEV_sing[1:-1, 0] = (
        coeff[:-1]
        * (
            error.survey.md[2:]
            - error.survey.md[:-2]
        ) / 2
        * mag
    )
    e_NEV_sing[1, 0] = (
        coeff[1]
        * (
            error.survey.md[2] + error.survey.md[1]
            - 2 * error.survey.md[0]
        ) / 2
        * mag
    )
    e_NEV_sing[-1, 0] = (
        coeff[-1]
        * (
            error.survey.md[-1]
            - error.survey.md[-2]
        ) / 2
        * mag
    )

    e_NEV[sing] = e_NEV_sing[sing]

    e_NEV_star = error._e_NEV_star(e_DIA)
    e_NEV_star_sing = np.zeros_like(e_NEV)
    e_NEV_star_sing[1:, 0] = (
        (
            error.survey.md[1:]
            - error.survey.md[:-1]
        ) / 2
        * mag
    )

    e_NEV_star[sing] = e_NEV_star_sing[sing]
    return e_DIA, e_NEV, e_NEV_star


def singularity_xym4(sing, error, e_DIA, mag):
    e_NEV = error._e_NEV(e_DIA)
    n = np.zeros(len(error.drdp_sing['double_delta_md']))
    e = np.array(0.5 * error.drdp_sing['double_delta_md'] * mag)
    v = np.zeros_like(n)
    e_NEV_sing = np.vstack(
        (
            np.zeros((1, 3)),
            np.stack((n, e, v), axis=-1),
            np.zeros((1, 3))
        )
    )
    e_NEV[sing] = e_NEV_sing[sing]

    e_NEV_star = error._e_NEV_star(e_DIA)
    n = np.zeros(len(error.drdp_sing['delta_md']))
    e = np.array(0.5 * error.drdp_sing['delta_md'] * mag)
    v = np.zeros_like(n)
    e_NEV_star_sing = np.vstack(
        (
            np.zeros((1, 3)),
            np.stack((n, e, v), axis=-1),
            np.zeros((1, 3))
        )
    )
    e_NEV_star[sing] = e_NEV_star_sing[sing]

    return e_DIA, e_NEV, e_NEV_star


def singularity_xym4e(sing, error, e_DIA, mag, propagation: str = 'random'):
    coeff = np.ones(len(error.survey.md))
    coeff[1:-1] = np.amax(np.stack((
        coeff[1:-1],
        sqrt(
            10 / error.drdp_sing['delta_md']
        )
    ), axis=-1), axis=-1)
    coeff[-1] = np.amax(np.stack((
        coeff[-1],
        sqrt(
            10 / (error.survey.md[-1] - error.survey.md[-2])
        )
    ), axis=-1), axis=-1)

    xym3e = XYM3E("xym3e", error, mag, propagation)
    e_NEV = error._e_NEV(e_DIA)
    e_NEV_sing = np.zeros_like(e_NEV)
    e_NEV_sing[:, 1] = xym3e.e_NEV[:, 0]
    e_NEV[sing] = e_NEV_sing[sing]

    e_NEV_star = error._e_NEV_star(e_DIA)
    e_NEV_star_sing = np.zeros_like(e_NEV_star)
    e_NEV_star_sing[:, 1] = xym3e.e_NEV_star[:, 0]
    e_NEV_star[sing] = e_NEV_star_sing[sing]

    return e_DIA, e_NEV, e_NEV_star


def singularity_xym4l(sing, error, e_DIA, mag):
    coeff = np.ones(len(error.survey.md))
    coeff[1:] = np.amax(np.stack((
        coeff[1:],
        sqrt(
            10 / (error.survey.md[1:] - error.survey.md[:-1])
        )
    ), axis=-1), axis=-1)

    e_NEV = error._e_NEV(e_DIA)
    e_NEV_sing = np.zeros_like(e_NEV)
    e_NEV_sing[1:-1, 1] = (
        coeff[1:-1]
        * (
            error.survey.md[2:]
            - error.survey.md[:-2]
        ) / 2
        * mag
    )
    e_NEV_sing[1, 1] = (
        coeff[1]
        * (
            error.survey.md[2] + error.survey.md[1]
            - 2 * error.survey.md[0]
        ) / 2
        * mag
    )
    e_NEV_sing[-1, 1] = (
        coeff[-1]
        * (
            error.survey.md[-1]
            - error.survey.md[-2]
        ) / 2
        * mag
    )

    e_NEV[sing] = e_NEV_sing[sing]

    e_NEV_star = error._e_NEV_star(e_DIA)
    e_NEV_star_sing = np.zeros_like(e_NEV)
    e_NEV_star_sing[1:, 1] = (
        (
            error.survey.md[1:]
            - error.survey.md[:-1]
        ) / 2
        * mag
    )
    e_NEV_star_sing[1, 1] = (
        (
            error.survey.md[1]
            - error.survey.md[0]
        )
        * mag
    )

    e_NEV_star[sing] = e_NEV_star_sing[sing]
    return e_DIA, e_NEV, e_NEV_star


def XYM3E(code, error, mag=0.00524, propagation='random', NEV=True, **kwargs):
    coeff = np.ones(len(error.survey.md))
    coeff[1:-1] = np.amax(np.stack((
        coeff[1:-1],
        sqrt(
            10 / error.drdp_sing['delta_md']
        )
    ), axis=-1), axis=-1)
    coeff[-1] = np.amax(np.stack((
        coeff[-1],
        sqrt(
            10 / (error.survey.md[-1] - error.survey.md[-2])
        )
    ), axis=-1), axis=-1)

    dpde = np.zeros((len(error.survey.md), 3))
    dpde[1:, 1] = np.absolute(
        cos(error.survey.inc_rad[1:])
        * cos(error.survey.azi_true_rad[1:])
        * coeff[1:]
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[1:, 2] = (
            (
                -np.absolute(cos(error.survey.inc_rad[1:]))
                * sin(error.survey.azi_true_rad[1:])
                / sin(error.survey.inc_rad[1:])
            )
            * coeff[1:]
        )
    dpde[1:, 2] = np.where(
        error.survey.inc_rad[1:] < error.survey.header.vertical_inc_limit,
        coeff[1:],
        dpde[1:, 2]
    )

    e_DIA = dpde * mag

    sing = np.where(
        error.survey.inc_rad < error.survey.header.vertical_inc_limit
    )
    if len(sing[0]) < 1:
        return error._generate_error(code, e_DIA, propagation, NEV)
    else:
        e_DIA, e_NEV, e_NEV_star = singularity_xym3e(sing, error, e_DIA, mag)
        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )


def singularity_xcla(error, mag, tortuosity=TORTUOSITY_RAD_PER_M):
    dpde = np.zeros((len(error.survey_rad), 3))

    def manage_sing(error):
        temp = np.absolute(
            sin(error.survey.inc_rad[1:])
            * (((
                error.survey.azi_true_rad[1:]
                - error.survey.azi_true_rad[:-1]
                + pi
            ) % (2 * pi)) - pi)
        )
        temp[np.where(
            error.survey.inc_rad[:-1] < error.survey.header.vertical_inc_limit
        )] = 0
        return temp

    dpde[1:, 0] = (
        (error.survey.md[1:] - error.survey.md[0:-1])
        * np.amax(np.stack((
            manage_sing(error),
            (
                tortuosity
                * (error.survey.md[1:] - error.survey.md[0:-1])
            )
        ), axis=-1), axis=-1)
        * -sin(error.survey.azi_true_rad[1:])
    )

    dpde[1:, 1] = (
        (error.survey.md[1:] - error.survey.md[0:-1])
        * np.amax(np.stack((
            manage_sing(error),
            (
                tortuosity
                * (error.survey.md[1:] - error.survey.md[0:-1])
            )
        ), axis=-1), axis=-1)
        * cos(error.survey.azi_true_rad[1:])
    )

    e_DIA = dpde * mag

    return e_DIA, e_DIA, e_DIA


def calc_xclh(code, error, mag=0.0167, propagation='random', NEV=True, **kwargs):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[1:, 0] = (
        (error.survey.md[1:] - error.survey.md[0:-1])
        * np.amax(np.stack((
            np.absolute(
                (error.survey.inc_rad[1:] - error.survey.inc_rad[:-1])
            ),
            (
                TORTUOSITY_RAD_PER_M
                * (error.survey.md[1:] - error.survey.md[0:-1])
            )
        ), axis=-1), axis=-1)
        * cos(error.survey.inc_rad[1:])
        * cos(error.survey.azi_true_rad[1:])
    )

    dpde[1:, 1] = (
        (error.survey.md[1:] - error.survey.md[0:-1])
        * np.amax(np.stack((
            np.absolute(
                (error.survey.inc_rad[1:] - error.survey.inc_rad[:-1])
            ),
            (
                TORTUOSITY_RAD_PER_M
                * (error.survey.md[1:] - error.survey.md[0:-1])
            )
        ), axis=-1), axis=-1)
        * cos(error.survey.inc_rad[1:])
        * sin(error.survey.azi_true_rad[1:])
    )

    dpde[1:, 2] = (
        (error.survey.md[1:] - error.survey.md[0:-1])
        * np.amax(np.stack((
            np.absolute(
                (error.survey.inc_rad[1:] - error.survey.inc_rad[:-1])
            ),
            (
                TORTUOSITY_RAD_PER_M
                * (error.survey.md[1:] - error.survey.md[0:-1])
            )
        ), axis=-1), axis=-1)
        * -sin(error.survey.inc_rad[1:])
    )

    e_DIA = dpde * mag

    return error._generate_error(
        code, e_DIA, propagation, NEV, e_NEV=e_DIA, e_NEV_star=e_DIA
    )
