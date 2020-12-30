import numpy as np
from numpy import sin, cos, tan, pi, sqrt
import yaml

# import welleng.error
from ..utils import NEV_to_HLA

FILENAME = 'welleng/errors/error_codes.yaml'


class iscwsaMwd:
    def __init__(
        self,
        error,
        model
    ):
        """
        Class using the ISCWSA MWD (Rev4) model to determine well bore
        uncertatinty.

        Parameters
        ----------
            error: an intitiated welleng.error.ErrorModel object

        Returns
        -------
            errors: welleng.error.ErrorModel object
                A populated ErrorModel object for the selected error model.
        """
        error.__init__
        self.e = error
        self.errors = {}

        with open(FILENAME, 'r') as file:
            iscwsa_error_models = yaml.full_load(file)
        self.em = iscwsa_error_models[model]

        self._initiate_func_dict()

        for err in self.em['codes']:
            func = self.em['codes'][err]['function']
            mag = self.em['codes'][err]['magnitude']
            propagation = self.em['codes'][err]['propagation']
            self.errors[err] = (
                self.call_func(
                    code=err,
                    func=func,
                    error=self.e,
                    mag=mag,
                    propagation=propagation,
                )
            )

        self.cov_NEVs = np.zeros((3, 3, len(self.e.survey_rad)))
        for _, value in self.errors.items():
            self.cov_NEVs += value.cov_NEV

        self.cov_HLAs = NEV_to_HLA(self.e.survey_rad, self.cov_NEVs)

    def call_func(self, code, func, error, mag, propagation):
        """
        Function for calling functions by mapping function labels to their
        functions.
        """
        assert func in self.func_dict, f"no function for function {func}"

        return self.func_dict[func](code, error, mag, propagation)

    def _initiate_func_dict(self):
        """
        This dictionary will need to be updated if/when additional error
        functions are added to the model.
        """
        self.func_dict = {
            'ABXY_TI1': ABXY_TI1,
            'ABXY_TI2': ABXY_TI2,
            'ABZ': ABZ,
            'AMIL': AMIL,
            'ASXY_TI1': ASXY_TI1,
            'ASXY_TI2': ASXY_TI2,
            'ASXY_TI3': ASXY_TI3,
            'ASZ': ASZ,
            'DBH': DBH,
            'AZ': AZ,
            'DREF': DREF,
            'DSF': DSF,
            'DST': DST,
            'MBXY_TI1': MBXY_TI1,
            'MBXY_TI2': MBXY_TI2,
            'MBZ': MBZ,
            'MSXY_TI1': MSXY_TI1,
            'MSXY_TI2': MSXY_TI2,
            'MSXY_TI3': MSXY_TI3,
            'MSZ': MSZ,
            'SAG': SAG,
            'XYM1': XYM1,
            'XYM2': XYM2,
            'XYM3': XYM3,
            'XYM4': XYM4,
        }


# error functions #
def DREF(code, error, mag=0.35, propagation='random', NEV=True):
    dpde = np.full((len(error.survey_rad), 3), [1., 0., 0.])
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def DSF(code, error, mag=0.00056, propagation='systematic', NEV=True):
    dpde = np.full((len(error.survey_rad), 3), [1., 0., 0.])
    dpde = dpde * np.array(error.survey_rad)
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def DST(code, error, mag=0.00000025, propagation='systematic', NEV=True):
    dpde = np.full((len(error.survey_rad), 3), [1., 0., 0.])
    dpde[:, 0] = error.survey.tvd
    dpde = dpde * np.array(error.survey_rad)
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ABXY_TI1(code, error, mag=0.0040, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = -cos(error.survey_rad[:, 1]) / error.survey.header.G
    dpde[:, 2] = (
        cos(error.survey_rad[:, 1])
        * tan(error.survey.header.dip)
        * sin(error.survey.azi_mag_rad)
    ) / error.survey.header.G
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ABXY_TI2(code, error, mag=0.004, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.nan_to_num(
            (
                (
                    tan(-(error.survey_rad[:, 1]) + (pi/2))
                    - tan(error.survey.header.dip)
                    * cos(error.survey.azi_mag_rad)
                ) / error.survey.header.G
            ),
            posinf=0.0,
            neginf=0.0
        )
    e_DIA = dpde * mag

    sing = np.where(
        error.survey_rad[:, 1] < error.survey.header.vertical_inc_limit
    )
    if len(sing[0]) < 1:
        return error._generate_error(code, e_DIA, propagation, NEV)
    else:
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
        e_NEV_star[sing] = e_NEV_star_sing[sing]

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )


def ABZ(code, error, mag=0.004, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = -sin(np.array(error.survey_rad)[:, 1]) / error.survey.header.G
    dpde[:, 2] = (
        sin(np.array(error.survey_rad)[:, 1])
        * tan(error.survey.header.dip) * sin(error.survey.azi_mag_rad)
    ) / error.survey.header.G
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASXY_TI1(code, error, mag=0.0005, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = sin(
        np.array(error.survey_rad)[:, 1]
    ) * cos(np.array(error.survey_rad)[:, 1]) / sqrt(2)
    dpde[:, 2] = (
        sin(np.array(error.survey_rad)[:, 1])
        * -tan(error.survey.header.dip) * cos(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
    ) / sqrt(2)
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASXY_TI2(code, error, mag=0.0005, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = sin(
        np.array(error.survey_rad)[:, 1]
    ) * cos(np.array(error.survey_rad)[:, 1]) / 2
    dpde[:, 2] = (
        sin(np.array(error.survey_rad)[:, 1])
        * -tan(error.survey.header.dip) * cos(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
    ) / 2
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASXY_TI3(code, error, mag=0.0005, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        sin(np.array(error.survey_rad)[:, 1])
        * tan(error.survey.header.dip) * cos(error.survey.azi_mag_rad)
        - cos(np.array(error.survey_rad)[:, 1])) / 2
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASZ(code, error, mag=0.0005, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = (
        -sin(np.array(error.survey_rad)[:, 1])
        * cos(np.array(error.survey_rad)[:, 1])
    )
    dpde[:, 2] = (
        sin(np.array(error.survey_rad)[:, 1])
        * tan(error.survey.header.dip)
        * cos(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MBXY_TI1(code, error, mag=70.0, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        -cos(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
    ) / (error.survey.header.b_total * cos(error.survey.header.dip))
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MBXY_TI2(code, error, mag=70.0, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        cos(error.survey.azi_mag_rad)
        / (
            error.survey.header.b_total
            * cos(error.survey.header.dip)
        )
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MBZ(code, error, mag=70.0, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        -sin(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
    ) / (error.survey.header.b_total * cos(error.survey.header.dip))
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MSXY_TI1(code, error, mag=0.0016, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        sin(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
        * (
            tan(error.survey.header.dip)
            * cos(np.array(error.survey_rad)[:, 1])
            + sin(np.array(error.survey_rad)[:, 1])
            * cos(error.survey.azi_mag_rad)
        ) / sqrt(2)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MSXY_TI2(code, error, mag=0.0016, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        sin(error.survey.azi_mag_rad) * (
            tan(error.survey.header.dip)
            * sin(np.array(error.survey_rad)[:, 1])
            * cos(np.array(error.survey_rad)[:, 1])
            - cos(np.array(error.survey_rad)[:, 1])
            * cos(np.array(error.survey_rad)[:, 1])
            * cos(error.survey.azi_mag_rad) - cos(error.survey.azi_mag_rad)
        ) / 2
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MSXY_TI3(code, error, mag=0.0016, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        cos(np.array(error.survey_rad)[:, 1])
        * cos(error.survey.azi_mag_rad) * cos(error.survey.azi_mag_rad)
        - cos(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad) * sin(error.survey.azi_mag_rad)
        - tan(error.survey.header.dip) * sin(np.array(error.survey_rad)[:, 1])
        * cos(error.survey.azi_mag_rad)
    ) / 2
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MSZ(code, error, mag=0.0016, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = -(
        sin(np.array(error.survey_rad)[:, 1])
        * cos(error.survey.azi_mag_rad)
        + tan(error.survey.header.dip) * cos(np.array(error.survey_rad)[:, 1])
    ) * sin(np.array(error.survey_rad)[:, 1]) * sin(error.survey.azi_mag_rad)
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def AZ(code, error, mag=0.00628, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = 1
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def DBH(code, error, mag=np.radians(5000), propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = 1 / (
        error.survey.header.b_total * cos(error.survey.header.dip)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def DBHR(code, error, mag=np.radians(3000), propagation='random', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = 1 / (
        error.survey.header.b_total * cos(error.survey.header.dip)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def AMIL(code, error, mag=220.0, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        -sin(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
        / (error.survey.header.b_total * cos(error.survey.header.dip))
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def SAG(code, error, mag=0.00349, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = sin(np.array(error.survey_rad)[:, 1])
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def XYM1(code, error, mag=0.00175, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = np.absolute(sin(np.array(error.survey_rad)[:, 1]))
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def XYM2(code, error, mag=0.00175, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = -1
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def XYM3(code, error, mag=0.00175, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = (
        np.absolute(cos(np.array(error.survey_rad)[:, 1]))
        * cos(error.survey.azi_true_rad)
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.nan_to_num(
            -(
                np.absolute(cos(np.array(error.survey_rad)[:, 1]))
                * sin(error.survey.azi_true_rad)
            ) / sin(np.array(error.survey_rad)[:, 1]),
            posinf=0.0,
            neginf=0.0
        )
    e_DIA = dpde * mag

    sing = np.where(
        error.survey_rad[:, 1] < error.survey.header.vertical_inc_limit
    )
    if len(sing[0]) < 1:
        return error._generate_error(code, e_DIA, propagation, NEV)
    else:
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

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )


def XYM4(code, error, mag=0.00175, propagation='systematic', NEV=True):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = np.absolute(
        cos(np.array(error.survey_rad)[:, 1])
    ) * sin(error.survey.azi_true_rad)
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.nan_to_num(
            (
                np.absolute(np.cos(np.array(error.survey_rad)[:, 1]))
                * cos(error.survey.azi_true_rad)
            )
            / sin(np.array(error.survey_rad)[:, 1]),
            posinf=0.0,
            neginf=0.0
            )
    e_DIA = dpde * mag

    sing = np.where(
        error.survey_rad[:, 1] < error.survey.header.vertical_inc_limit
    )
    if len(sing[0]) < 1:
        return error._generate_error(code, e_DIA, propagation, NEV)
    else:
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

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )
