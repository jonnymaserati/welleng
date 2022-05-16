import os
import numpy as np
import yaml
from .errors.tool_errors import ErrorModel, ToolError
from welleng.survey import Survey

# TODO: there's likely an issue with TVD versus TVDSS that
# needs to be resolved. This model assumes TVD relative to
# rig floor, but often a TVDSS is provided instead (with a
# negative value for rig floor elevation).

ACCURACY = 1e-4
PATH = os.path.dirname(__file__)
TOOL_INDEX_FILENAME = os.path.join(
    '', *[PATH, 'errors', 'tool_index.yaml']
)


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


class ErrorModel():
    """
    A class to initiate the field parameters and error magnitudes
    for subsequent error calculations.
    """

    class Error:
        '''
        Standard components of a well bore survey error.
        '''
        def __init__(
            self,
            code,
            propagation,
            e_DIA,
            cov_DIA,
            e_NEV,
            e_NEV_star,
            sigma_e_NEV,
            cov_NEV
        ):

            self.code = code
            self.propagation = propagation
            self.e_DIA = e_DIA
            self.cov_DIA = cov_DIA
            self.e_NEV = e_NEV
            self.e_NEV_star = e_NEV_star
            self.sigma_e_NEV = sigma_e_NEV
            self.cov_NEV = cov_NEV

    def __init__(
        self,
        survey: Survey,
        error_model: str = "ISCWSA MWD Rev5",
    ):
        assert error_model in ERROR_MODELS, "Unrecognized error model"
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

        for k, v in TOOL_INDEX.items():
            if v['Short Name'] == self.error_model:
                model = k
                break

        self.errors = ToolError(
            error=self,
            model=model
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
            [x*x, x*y, x*z],
            [y*x, y*y, y*z],
            [z*x, z*y, z*z]
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
    ) -> ErrorModel:
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
            else:
                return

            return ErrorModel.Error(
                code,
                propagation,
                e_DIA,
                cov_DIA,
                e_NEV,
                e_NEV_star,
                sigma_e_NEV,
                cov_NEV
            )

    def drk_dDepth(
            self,
            inc1: np.ndarray,
            azi1: np.ndarray,
            inc2: np.ndarray,
            azi2: np.ndarray,
    ) -> np.ndarray:
        """
        This function calculates drk/dDepth
        Refer to the notion document for information about the calculation.
        survey1 is previous survey station
        survey2 is current survey station
        For both stations, md in meters, inc and azi in radians
        """

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

    def drk_dInc(
            self,
            inc: np.ndarray,
            azi: np.ndarray,
            delta_md: float
    ) -> np.ndarray:
        """
        This function calculates drk/dInc
        Refer to the notion document for information about the calculation.
        survey1 is previous survey station
        survey2 is current survey station
        For both stations, md in meters, inc and azi in radians
        """

        N = np.array(0.5 * (delta_md * np.cos(inc) * np.cos(azi)))
        E = np.array(0.5 * (delta_md * np.cos(inc) * np.sin(azi)))
        V = np.array(0.5 * (-delta_md * np.sin(inc)))

        if self.error_model.lower().split()[-1] != 'rev4':
            N[0] *= 2

        return np.vstack(
            (
                np.array(np.zeros((1, 3))),
                np.stack((N, E, V), axis=-1)
            )
        )

    def drk_dAz(
            self,
            inc: np.ndarray,
            azi: np.ndarray,
            delta_md: float
    ) -> np.ndarray:
        """
        This function calculates drk/dAzi
        Refer to the notion document for information about the calculation.
        survey1 is previous survey station
        survey2 is current survey station
        For both stations, md in meters, inc and azi in radians
        """

        N = np.array(-0.5 * (delta_md * np.sin(inc) * np.sin(azi)))
        E = np.array(0.5 * (delta_md * np.sin(inc) * np.cos(azi)))
        V = np.zeros_like(N)

        return np.vstack(
            (
                np.array(np.zeros((1, 3))),
                np.stack((N, E, V), axis=-1)
            )
        )

    @staticmethod
    def get_survey_data(survey: Survey):
        md1, inc1, azi1 = np.array(survey[:-1]).T
        md2, inc2, azi2 = np.array(survey[1:]).T
        delta_md = md2 - md1
        return (md1, inc1, azi1), (md2, inc2, azi2), delta_md

    def drkplus1_dDepth(self, survey) -> np.ndarray:
        """
        This function calculates drk+1/dDepth
        Refer to the notion document for information about the calculation.
        survey1 is previous survey station
        survey2 is current survey station
        For both stations, md in meters, inc and azi in radians
        """
        return np.vstack(
            (
                self.drk_dDepth(survey)[1:] * -1,
                np.array(np.zeros((1, 3)))
            )
        )

    def drkplus1_dInc(
            self,
            inc: np.ndarray,
            azi: np.ndarray,
            delta_md: float
    ) -> np.ndarray:
        """
        This function calculates drk+1/dInc
        Refer to the notion document for information about the calculation.
        survey1 is previous survey station
        survey2 is current survey station
        For both stations, md in meters, inc and azi in radians
        """

        self.drk_dInc(inc, azi, delta_md)

    def drkplus1_dAz(
            self,
            inc: np.ndarray,
            azi: np.ndarray,
            delta_md: float
    ) -> np.ndarray:
        """
        This function calculates drk+1/dAzi

        Refer to the notion document for information about the calculation.
        survey1 is previous survey station
        survey2 is current survey station
        For both stations: md in meters, inc and azi in radians
        """

        self.drk_dAz(inc, azi, delta_md)

    def _drdp(self, survey: Survey) -> np.ndarray:
        """
        This function calculates drdp matrix from the survey data
        """

        (_, inc1, azi1), (_, inc2, azi2), delta_md = \
            self.get_survey_data(survey)

        return np.hstack((
            self.drk_dDepth(inc1, azi1, inc2, azi2),
            self.drk_dInc(inc2, azi2, delta_md),
            self.drk_dAz(inc2, azi2, delta_md),
            self.drkplus1_dDepth(survey),
            self.drkplus1_dInc(inc1, azi1, delta_md),
            self.drkplus1_dAz(inc1, azi1, delta_md)
        ))

    def _drdp_sing(self, survey: Survey) -> dict:
        """
        This function gets the azimuth of the current survey station (azi2) and
        MD difference between the current and the previous station and the md
        difference between the previous and the next stations.
        These variables are used extensively in ISCWSA tool error models.
        """
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
