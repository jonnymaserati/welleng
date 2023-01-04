import os
from enum import Enum

import numpy as np
import yaml

from .error_formula_extractor.models import SurveyToolErrorModel
from .errors.tool_errors import ToolError

# TODO: there's likely an issue with TVD versus TVDSS that
# needs to be resolved. This model assumes TVD relative to
# rig floor, but often a TVDSS is provided instead (with a
# negative value for rig floor elevation).

ACCURACY = 1e-4
PATH = os.path.dirname(__file__)
TOOL_INDEX_FILENAME = os.path.join(
    '', *[PATH, 'errors', 'tool_index.yaml']
)


class ISCWSAErrorModel(Enum):
    Rev4 = "ISCWSA MWD Rev4"
    Rev5 = "ISCWSA MWD Rev5"


def get_tool_index():
    with open(TOOL_INDEX_FILENAME, 'r') as f:
        tool_index = yaml.safe_load(f)
    return tool_index


def get_error_models(tool_index=None):
    if tool_index is None:
        tool_index = get_tool_index()
    error_models = [
        v['Short Name']
        for _, v in tool_index.items()
    ]
    return error_models


TOOL_INDEX = get_tool_index()
ERROR_MODELS = get_error_models(TOOL_INDEX)


class Error:
    """
    Standard components of a well bore survey error.
    """

    def __init__(
            self,
            code: str,
            propagation: str,
            e_DIA: np.ndarray,
            cov_DIA: np.ndarray,
            e_NEV: np.ndarray,
            e_NEV_star: np.ndarray,
            sigma_e_NEV: np.ndarray,
            cov_NEV: np.ndarray
    ):
        self.code = code
        self.propagation = propagation
        self.e_DIA = e_DIA
        self.cov_DIA = cov_DIA
        self.e_NEV = e_NEV
        self.e_NEV_star = e_NEV_star
        self.sigma_e_NEV = sigma_e_NEV
        self.cov_NEV = cov_NEV


class ErrorModel:
    """
    A class to initiate the field parameters and error magnitudes
    for subsequent error calculations.
    """

    def __init__(
        self,
        survey: 'Survey',
        error_model: SurveyToolErrorModel,
        error_from_edm: bool = False
    ):
        self.error_model = error_model
        self.survey = survey

        self.survey_rad = np.stack((
            self.survey.md,
            self.survey.inc_rad,
            self.survey.azi_true_rad
        ), axis=-1)

        self.survey_drdp = self.survey_rad
        self.drdp = self._drdp(self.survey_drdp)
        self.drdp_sing = self._drdp_sing(self.survey_drdp)

        model = error_model
        if not error_from_edm:
            for k, v in TOOL_INDEX.items():
                if v['Short Name'] == self.error_model:
                    model = k
                    break

        self.errors = ToolError(
            error=self,
            model=model,
            is_error_from_edm=error_from_edm
        )

    def _e_NEV(self, e_DIA: np.ndarray) -> np.ndarray:
        """
        This function calculates error in NEV at all stations based on error in
         DIA.
        """

        D, I, A = e_DIA.T
        arr = np.array([
            (self.drdp[:, 0] + self.drdp[:, 9]) * D
            + (self.drdp[:, 3] + self.drdp[:, 12]) * I
            + (self.drdp[:, 6] + self.drdp[:, 15]) * A,

            (self.drdp[:, 1] + self.drdp[:, 10]) * D
            + (self.drdp[:, 4] + self.drdp[:, 13]) * I
            + (self.drdp[:, 7] + self.drdp[:, 16]) * A,

            (self.drdp[:, 2] + self.drdp[:, 11]) * D
            + (self.drdp[:, 5] + self.drdp[:, 14]) * I
            + (self.drdp[:, 8] + self.drdp[:, 17]) * A,
        ]).T

        arr[0] = 0

        return arr

    def _e_NEV_star(self, e_DIA: np.ndarray) -> np.ndarray:
        """
        Calculate the error at a station based on error in DIA
        """
        D, I, A = e_DIA.T
        arr = np.array([
            self.drdp[:, 0] * D
            + self.drdp[:, 3] * I
            + self.drdp[:, 6] * A,

            self.drdp[:, 1] * D
            + self.drdp[:, 4] * I
            + self.drdp[:, 7] * A,

            self.drdp[:, 2] * D
            + self.drdp[:, 5] * I
            + self.drdp[:, 8] * A
        ]).T

        arr[0] = 0

        return arr

    def _cov(self, arr: np.ndarray) -> np.ndarray:
        """
        Returns a covariance matrix from an (n,3) array.
        The structure of the matrix returned is (3, 3, n).

        """

        x, y, z = np.array(arr).T
        result = np.array([
            [x * x, x * y, x * z],
            [y * x, y * y, y * z],
            [z * x, z * y, z * z]
        ])

        return result

    def _sigma_e_NEV_systematic(self, e_NEV, e_NEV_star):
        return e_NEV_star + np.vstack(
            (
                e_NEV[0],
                np.cumsum(e_NEV, axis=0)[:-1]
            )
        )

    def _generate_error(
        self,
        code: str,
        e_DIA: np.ndarray,
        propagation: str = 'systematic',
        NEV: bool = True,
        e_NEV: bool = None,
        e_NEV_star: bool = None
    ) :
        """
        Calculate the error for a tool at the current station (e_NEV) and
        the error at the final survey station
        (e_NEV_star) in Northing Easting Vertical (NEV) using the error code,
        error in DIA [Depth, inclination, Azimuth],
        and the dr/dp calculated earlier.

        """

        if not NEV:
            return e_DIA
        else:
            cov_DIA = self._cov(e_DIA)

            if e_NEV is None:
                e_NEV = self._e_NEV(e_DIA)
                if "xcl" in code.lower():
                    # for xcl terms, e_NEV_star == e_NEV
                    e_NEV_star = e_NEV
                else:
                    e_NEV_star = self._e_NEV_star(e_DIA)

            if propagation == 'systematic':
                sigma_e_NEV = self._sigma_e_NEV_systematic(e_NEV, e_NEV_star)
                cov_NEV = self._cov(sigma_e_NEV)

            elif propagation == 'random':
                sigma_e_NEV = np.cumsum(self._cov(e_NEV), axis=-1)
                cov_NEV = np.add(
                    self._cov(e_NEV_star),
                    np.concatenate(
                        (
                            np.array(np.zeros((3, 3, 1))),
                            np.array(sigma_e_NEV[:, :, :-1])
                        ), axis=-1)
                )

            elif propagation == 'global' or propagation == "well_to_well":
                sigma_e_NEV = np.cumsum(e_NEV, axis=0)
                cov_NEV = self._cov(
                    np.add(
                        e_NEV_star,
                        np.concatenate(
                            (np.array(np.zeros((1, 3))),
                             np.array(sigma_e_NEV[:-1, :])
                             ),
                            axis=0)
                    )
                )

            else:
                return

            return Error(
                code,
                propagation,
                e_DIA,
                cov_DIA,
                e_NEV,
                e_NEV_star,
                sigma_e_NEV,
                cov_NEV
            )

    def drk_dDepth(self, survey: 'Survey'):
        '''
        survey1 is previous survey station (with inc and azi in radians)
        survey2 is current survey station (with inc and azi in radians)
        '''
        # TODO: This is essentially minimum curvature... use function from
        # utils instead (it's already in self.mc)
        md1, inc1, azi1 = np.array(survey[:-1]).T
        md2, inc2, azi2 = np.array(survey[1:]).T

        N = np.array(
            0.5 * (
                np.sin(inc1) * np.cos(azi1)
                + np.sin(inc2) * np.cos(azi2)
            )
        )

        E = np.array(
            0.5 * (
                np.sin(inc1) * np.sin(azi1)
                + np.sin(inc2) * np.sin(azi2)
            )
        )

        V = np.array(
            0.5 * (
                np.cos(inc1) + np.cos(inc2)
            )
        )

        return np.vstack(
            (
                np.array(np.zeros((1, 3))),
                np.stack((N, E, V), axis=-1)
            )
        )

    def drk_dInc(self, survey: 'Survey'):
        '''
        survey1 is previous survey station (with inc and azi in radians)
        survey2 is current survey station (with inc and azi in radians)
        '''
        md1, inc1, azi1 = np.array(survey[:-1]).T
        md2, inc2, azi2 = np.array(survey[1:]).T
        delta_md = md2 - md1

        N = np.array(0.5 * ((delta_md) * np.cos(inc2) * np.cos(azi2)))
        E = np.array(0.5 * ((delta_md) * np.cos(inc2) * np.sin(azi2)))
        V = np.array(0.5 * (-delta_md * np.sin(inc2)))

        if (isinstance(self.error_model, str) and self.error_model.lower().split()[-1] != 'rev4') \
                or not isinstance(self.error_model, str):
            N[0] *= 2

        return np.vstack(
            (
                np.array(np.zeros((1, 3))),
                np.stack((N, E, V), axis=-1)
            )
        )

    def drk_dAz(self, survey: 'Survey'):
        '''
        survey1 is previous survey station (with inc and azi in radians)
        survey2 is current survey station (with inc and azi in radians)
        '''
        md1, inc1, azi1 = np.array(survey[:-1]).T
        md2, inc2, azi2 = np.array(survey[1:]).T
        delta_md = md2 - md1

        N = np.array(-0.5 * ((delta_md) * np.sin(inc2) * np.sin(azi2)))
        E = np.array(0.5 * ((delta_md) * np.sin(inc2) * np.cos(azi2)))
        V = np.zeros_like(N)

        return np.vstack(
            (
                np.array(np.zeros((1, 3))),
                np.stack((N, E, V), axis=-1)
            )
        )

    def drkplus1_dDepth(self, survey: 'Survey'):
        '''
        survey2 is current survey station (with inc and azi in radians)
        survey3 is next survey station (with inc and azi in radians)
        '''
        return np.vstack(
            (
                self.drk_dDepth(survey)[1:] * -1,
                np.array(np.zeros((1, 3)))
            )
        )

    def drkplus1_dInc(self, survey: 'Survey'):
        '''
        survey2 is current survey station (with inc and azi in radians)
        survey3 is next survey station (with inc and azi in radians)
        '''

        md2, inc2, azi2 = np.array(survey[:-1]).T
        md3, inc3, azi3 = np.array(survey[1:]).T
        delta_md = md3 - md2

        N = np.array(0.5 * ((delta_md) * np.cos(inc2) * np.cos(azi2)))
        E = np.array(0.5 * ((delta_md) * np.cos(inc2) * np.sin(azi2)))
        V = np.array(0.5 * (-(delta_md) * np.sin(inc2)))

        return np.vstack(
            (
                np.stack((N, E, V), axis=-1),
                np.array(np.zeros((1, 3)))
            )
        )

    def drkplus1_dAz(self, survey: 'Survey'):
        '''
        survey2 is current survey station (with inc and azi in radians)
        survey3 is next survey station (with inc and azi in radians)
        '''
        md2, inc2, azi2 = np.array(survey[:-1]).T
        md3, inc3, azi3 = np.array(survey[1:]).T
        delta_md = md3 - md2

        N = np.array(-0.5 * ((delta_md) * np.sin(inc2) * np.sin(azi2)))
        E = np.array(0.5 * ((delta_md) * np.sin(inc2) * np.cos(azi2)))
        V = np.zeros_like(N)

        return np.vstack(
            (
                np.stack((N, E, V), axis=-1),
                np.array(np.zeros((1, 3)))
            )
        )

    def _drdp(self, survey: 'Survey'):
        """
        This function calculates dr/dp used to calculate errors.
        Refer to Appendix A in the ISCWSA Introduction to Wellbore positioning book
        for equations (Pages 218-220).

        :param survey:
        :return:
        """

        return np.hstack((
            self.drk_dDepth(survey),
            self.drk_dInc(survey),
            self.drk_dAz(survey),
            self.drkplus1_dDepth(survey),
            self.drkplus1_dInc(survey),
            self.drkplus1_dAz(survey)
        ))

    def _drdp_sing(self, survey: 'Survey'):
        '''
        survey1 is previous survey station (with inc and azi in radians)
        survey2 is current survey station
        survey3 is next survey station (with inc and azi in radians)
        '''
        md1, inc1, azi1 = np.array(survey[:-2]).T
        md2, inc2, azi2 = np.array(survey[1:-1]).T
        md3, inc3, azi3 = np.array(survey[2:]).T
        double_delta_md = md3 - md1
        delta_md = md2 - md1

        return dict(
            double_delta_md=double_delta_md,
            delta_md=delta_md,
            azi2=azi2
        )


def get_errors(error):
    nn, ne, nv = error[0]
    _, ee, ev = error[1]
    _, __, vv = error[2]

    return [nn, ee, vv, ne, nv, ev]


def make_diagnostic_data(survey):
    diagnostic = {}
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
