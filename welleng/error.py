import os
import numpy as np
import yaml
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
    Standard well bore survey error.
    """
    def __init__(
        self,
        code,
        propagation,
        e_dia,
        cov_dia,
        e_nev,
        e_nev_star,
        sigma_e_nev,
        cov_nev
    ):
        """
        Container for the calculated data for a given survey error.

        Parameters
        ----------
        code : str
            The error code.
        propagation : str {'systematic', 'random', 'global'}
            The method for propagating the tool error.
        e_dia : ndarray
            An (n, 3) array of errors in the Depth, Inclination and Azimuth
            (DIA) domain.
        cov_dia : ndarray
            An (3, 3, n) array of the covariance matrix of errors in the DIA
            domain.
        e_nev : ndarray
            An (n, 3) array of errors in the North, East and Vertical (NEV)
            domain.
        e_nev_star : ndarray
            An (n, 3) array of processed errors in the NEV domain.
        sigma_e_nev : ndarray
            An (n, 3) or (n, 6) array (systematic or global/random
            respectively) of sigma values in the NEV domain.
        cov_nev : ndarray
            An (3, 3, n) array of the covariance matrix of errors in the NEV
            domain.
        """

        self.code = code
        self.propagation = propagation
        self.e_dia = e_dia
        self.cov_dia = cov_dia
        self.e_nev = e_nev
        self.e_nev_star = e_nev_star
        self.sigma_e_nev = sigma_e_nev
        self.cov_nev = cov_nev


class ErrorModel:
    """
    A class to initiate the field parameters and error magnitudes
    for subsequent error calculations.
    """
    def __init__(
        self,
        survey,
        error_model=None,
    ):
        """
        Generate errors for a pre-defined ISCWSA error model.

        Parameters
        ----------
        survey : Survey
            A :class:`Survey` instance.
        error_model : None | str, optional
            Name of the standard error model. If ``None`` is provided will
            default to ``ISCWSA MWD Rev5``.

        See Also
        --------
        welleng.survey.Survey
        """
        error_model = "ISCWSA MWD Rev5" if error_model is None else error_model

        # error_models = ERROR_MODELS
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

        # if self.error_model.split("_")[0] == "iscwsa":
        #     self.errors = iscwsaMwd(
        #         error=self,
        #         model=self.error_model
        #     )

        for k, v in TOOL_INDEX.items():
            if v['Short Name'] == self.error_model:
                model = k
                break

        self.errors = ToolError(
            error=self,
            model=model
        )

    def _e_nev(self, e_dia):
        D, I, A = e_dia.T
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

    def _e_nev_star(self, e_dia):
        D, I, A = e_dia.T
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

    def _cov(self, arr):
        '''
        Returns a covariance matrix from an (n,3) array.
        '''
        # Mitigate overflow
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     coeff = np.nan_to_num(
        #         arr / np.abs(arr) * ACCURACY,
        #         nan=ACCURACY
        #     )
        # arr = np.where(np.abs(arr) > ACCURACY, arr, coeff)

        x, y, z = np.array(arr).T
        result = np.array([
            [x*x, x*y, x*z],
            [y*x, y*y, y*z],
            [z*x, z*y, z*z]
        ])

        return result

    def _sigma_e_nev_systematic(self, e_nev, e_nev_star):
        return e_nev_star + np.vstack(
            (
                e_nev[0],
                np.cumsum(e_nev, axis=0)[:-1]
            )
        )

    def _generate_error(
        self,
        code,
        e_dia,
        propagation='systematic',
        nev=True,
        e_nev=None,
        e_nev_star=None
    ):
        if not nev:
            return e_dia
        else:
            cov_dia = self._cov(e_dia)
            if e_nev is None:
                e_nev = self._e_nev(e_dia)
                e_nev_star = self._e_nev_star(e_dia)
            if propagation == 'systematic':
                sigma_e_nev = self._sigma_e_nev_systematic(e_nev, e_nev_star)
                cov_nev = self._cov(sigma_e_nev)
            elif propagation == 'random':
                sigma_e_nev = np.cumsum(self._cov(e_nev), axis=-1)
                cov_nev = np.add(
                    self._cov(e_nev_star),
                    np.concatenate(
                        (
                            np.array(np.zeros((3, 3, 1))),
                            np.array(sigma_e_nev[:, :, :-1])
                        ), axis=-1)
                    )
            else:
                return

            return Error(
                code,
                propagation,
                e_dia,
                cov_dia,
                e_nev,
                e_nev_star,
                sigma_e_nev,
                cov_nev
            )

    def drk_dDepth(self, survey):
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

    def drk_dInc(self, survey):
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

        if self.error_model.lower().split()[-1] != 'rev4':
            N[0] *= 2

        return np.vstack(
            (
                np.array(np.zeros((1, 3))),
                np.stack((N, E, V), axis=-1)
            )
        )

    def drk_dAz(self, survey):
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

    def drkplus1_dDepth(self, survey):
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

    def drkplus1_dInc(self, survey):
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

    def drkplus1_dAz(self, survey):
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

    def _drdp(self, survey):

        return np.hstack((
            self.drk_dDepth(survey),
            self.drk_dInc(survey),
            self.drk_dAz(survey),
            self.drkplus1_dDepth(survey),
            self.drkplus1_dInc(survey),
            self.drkplus1_dAz(survey)
        ))

    def _drdp_sing(self, survey):
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
    dia = np.stack((survey.md, survey.inc_deg, survey.azi_grid_deg), axis=1)
    for i, d in enumerate(survey.md):
        diagnostic[d] = {}
        total = []
        for k, v in survey.err.errors.errors.items():
            diagnostic[d][k] = get_errors(v.cov_nev.T[i])
            total.extend(diagnostic[d][k])
        diagnostic[d]['TOTAL'] = np.sum((np.array(
            total
        ).reshape(-1, len(diagnostic[d][k]))), axis=0)
    return diagnostic
