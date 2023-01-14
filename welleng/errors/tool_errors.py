from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable, List

import numpy as np
import yaml
from numpy import cos, pi, sin, sqrt, tan

from ..error_formula_extractor.enums import Propagation, VectorType
from ..error_formula_extractor.models import ErrorTerm, SurveyToolErrorModel
from ..units import METER_TO_FOOT
from ..utils import NEV_to_HLA
from .singularity_util import calc_xclh, calculate_error_singularity

# since this is running on different OS flavors
PATH = os.path.dirname(__file__)
TOOL_INDEX = os.path.join(
    '', *[PATH, 'tool_index.yaml']
)

ACCURACY = 1e-6


class ToolError:
    def __init__(
            self,
            error: 'Error',
            model: str,
            is_error_from_edm: bool = False
    ):
        """
        Class using the ISCWSA listed tool errors to determine well bore
        uncertainty.

        Parameters
        ----------
        error: an intitiated welleng.error.ErrorModel object
        model: string

        Returns
        -------
            errors: welleng.error.ErrorModel object
                A populated ErrorModel object for the selected error model.
        """
        error.__init__

        self.e = error
        self.errors = {}
        self.map = None
        self.model = None
        self.converted_covs = {}

        if not is_error_from_edm:
            self._calculate_error_from_welleng_models(model)
            return

        self._calculate_error_from_edm_models(model)

    def _calculate_error_from_edm_models(self, model: SurveyToolErrorModel):
        self.extract_tortuosity(error_from_edm=True)

        self.model = model
        for term in model.error_terms:
            if term.term_name in self.errors.keys():
                term.term_name += "_2"
            if term.term_name not in self.errors.keys():
                self.errors[term.term_name] = (
                    self.call_func_from_edm(
                        term=term,
                        func=term.error_function,
                        error=self.e,
                        mag=term.magnitude,
                        propagation=term.tie_type,
                        vector_type=term.vector_type
                    )
                )

        self.errors = {
            key: value
            for key, value in self.errors.items()
            if type(value) not in [list, float, int, np.ndarray]
        }

        self.cov_NEVs = np.zeros((3, 3, len(self.e.survey_rad)))
        for _, value in self.errors.items():
            self.cov_NEVs += value.cov_NEV

        self.cov_HLAs = NEV_to_HLA(self.e.survey_rad, self.cov_NEVs)

        return self.errors

    def call_func_from_edm(
            self,
            term: ErrorTerm,
            func: List[Callable],
            error: 'Error',
            mag: List[float],
            propagation: List[Propagation],
            vector_type: List[VectorType]
    ) -> np.ndarray:
        """
        This function calculates the covariances from the error model imported from the EDM file.
        """

        # initiate the variable map if it is not available.
        if not self.map:
            self._initiate_map(error)

        dpde = np.zeros((len(error.survey_rad) - 1, 3))

        # check if the given term is an intermediate step
        intermediate_step = propagation[0] == Propagation.NA
        sing_calc = False

        if term.term_name.lower() == "xclh":
            # TODO: Check with Steve why this calculation was different in the excel file
            return calc_xclh(term.term_name, error, mag[0], propagation[0].value)

        for vector_no in range(len(vector_type)):
            # for each vector in vector_type, first, extract all arguments from the equation.

            # if vector_type[vector_no] == VectorType.LATERAL:
            #     sing_calc = True

            args = []
            for arg in term.arguments[vector_no]:
                # for each argument first check if the argument is in the map or is one of the previously calculated
                # error terms
                self.backfill_terms(arg, term)

                if arg.lower() not in self.map.keys():
                    args.append(self.errors[arg.lower()])
                    continue

                args.append(self.map[arg.lower()])

            if intermediate_step:
                return np.vectorize(func[0])(*args) * term.magnitude[0]

            try:
                dpde[:, vector_type[vector_no].column_no] = np.vectorize(func[vector_no])(*args)

            except ZeroDivisionError:
                # if there is a zero division error, calculate the covariance row by row. If there is a zero
                # division error for a row, put zero as the value, same if the calculation returns np.inf.
                for index in range(len(np.array(error.survey_rad)) - 1):
                    vars = [
                        val if not type(val) == np.ndarray else val[index]
                        for val in args
                    ]

                    try:
                        val_to_put = func[vector_no](*vars)

                    except ZeroDivisionError:
                        val_to_put = 0

                    if val_to_put == np.inf:
                        val_to_put = 0

                    dpde[:, vector_type[vector_no].column_no][index] = val_to_put

        dpde = np.vstack((np.array([0., 0., 0.]), dpde))

        e_DIA = dpde.round(decimals=10) * mag[0]
        if not sing_calc:
            return error._generate_error(
                term.term_name,
                e_DIA,
                propagation[0].value,
                NEV=True
            )
        else:
            return self.caclcualte_error_with_sing(
                error,
                term.term_name,
                e_DIA,
                propagation[0].value,
                NEV=True,
                magnitude=mag[0]
            )

    def _initiate_map(self, error):
        self.map = {
            "tmd": np.array(error.survey_rad)[:, 0][1:],
            "tvd": np.array(error.survey.tvd)[1:],
            "inc": np.array(error.survey_rad)[:, 1][1:],
            "azi": np.array(error.survey_rad)[:, 2][1:],
            "gtot": error.survey.header.G,
            "dip": error.survey.header.dip,
            "erot": error.survey.header.earth_rate,
            "mtot": error.survey.header.b_total,
            "lat": np.radians(error.survey.header.latitude),
            "smd": np.diff(np.array(error.survey_rad)[:, 0]),
            "dmd": np.array(error.survey_rad)[:, 0][1:],
            "din": np.diff(np.array(error.survey_rad)[:, 1]),
            "daz": np.diff(np.array(error.survey_rad)[:, 2]),
            "azt": error.survey.azi_true_rad[1:],
            "dazt": np.diff(error.survey.azi_true_rad),  # this term is not available in Compass
            "azm": error.survey.azi_mag_rad[1:],
            "xcltortuosity": (np.radians(1) / 100) * METER_TO_FOOT
        }

    def backfill_terms(self, arg: str, term: ErrorTerm):
        if arg.lower() not in self.map.keys() and arg.lower() not in self.errors.keys():
            term_backfilled = False
            # if the argument is not in the map or in the previously calculated error terms, it needs to be
            # back-filled before it can be used for the current term.
            for back_filled_term in self.model.error_terms:
                if back_filled_term.term_name == arg.lower():
                    self.errors[back_filled_term.term_name] = (
                        self.call_func_from_edm(
                            term=back_filled_term,
                            func=back_filled_term.error_function,
                            error=self.e,
                            mag=back_filled_term.magnitude,
                            propagation=back_filled_term.tie_type,
                            vector_type=back_filled_term.vector_type
                        )
                    )
                    term_backfilled = True
                    break

            if not term_backfilled:
                raise KeyError(
                    f"{arg} not in {list(self.errors.keys())} or in the map: {list(self.map.keys())} "
                    f" for error term {term.term_name}")

    @staticmethod
    def caclcualte_error_with_sing(
            error: 'Error',
            code: str,
            e_DIA: np.ndarray,
            propagation: str,
            NEV: bool,
            magnitude: float
    ) -> 'Error':

        sing = np.where(
            error.survey.inc_rad < error.survey.header.vertical_inc_limit
        )

        if len(sing[0]) < 1:
            return error._generate_error(code, e_DIA, propagation, NEV)
        else:
            e_DIA, e_NEV, e_NEV_star = calculate_error_singularity(sing, error, e_DIA, magnitude, code, propagation)

            return error._generate_error(
                code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
            )

    def _calculate_error_from_welleng_models(self, model: str):

        filename = os.path.join(
            '', *[PATH, 'tool_codes', f"{model}.yaml"]
        )

        with open(filename, 'r') as file:
            self.em = yaml.safe_load(file)

        self.extract_tortuosity()

        # for gyro tools the continuous survey errors need to be done last
        self.em['codes'] = OrderedDict(self.em['codes'])
        gyro_continuous = ['GXY-GD', 'GXY-GRW']
        gyro_stationary = ['GXY-B1S', 'GXY-B2S', 'GXY-G4', 'GXY-RN']
        for tool in gyro_continuous:
            if tool in self.em['codes']:
                self.gyro_continuous = []
                self.em['codes'].move_to_end(tool)
                self.gyro_continuous.append(tool)
        self.gyro_stationary = [
            tool for tool in gyro_stationary
            if tool in self.em['codes']
        ]

        self.extract_tortuosity()
        # if model == "iscwsa_mwd_rev5":
        # if model == "ISCWSA MWD Rev5":
        # assert self.tortuosity is not None, (
        #     "No default tortuosity defined in model header"
        # )

        if "Inclination Range Max" in self.em['header'].keys():
            value = np.radians(float(
                self.em['header']['Inclination Range Max'].split(" ")[0]
            ))
            assert np.amax(self.e.survey.inc_rad) < value, (
                "Model not suitable for this well path inclination"
            )

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
                    tortuosity=self.tortuosity,
                    header=self.em['header'],
                    errors=self
                )
            )

        self.cov_NEVs = np.zeros((3, 3, len(self.e.survey_rad)))
        for _, value in self.errors.items():
            self.cov_NEVs += value.cov_NEV

        self.cov_HLAs = NEV_to_HLA(self.e.survey_rad, self.cov_NEVs)

    def _get_the_func_out(self, err):
        if err in self.exceptional_funcs:
            func = self.exceptional_funcs[err]
        else:
            func = self.em['codes'][err]['function']

        return func

    def call_func(
            self,
            code: str,
            func: str,
            error: "Error",
            mag: float,
            propagation: str,
            **kwargs
    ):
        """
        Function for calling functions by mapping function labels to their
        functions.
        """
        assert func in self.func_dict, f"no function for function {func}"

        return self.func_dict[func](code, error, mag, propagation, **kwargs)

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
            'SAGE': SAGE,
            'XCL': XCL,  # requires an exception
            'XYM3L': XYM3L,  # looks like there's a mistake in the ISCWSA model
            'XYM4L': XYM4L,
            'XCLA': XCLA,
            'XCLH': XCLH,
            'XYM3E': XYM3E,  # Needs QAQC
            'XYM4E': XYM4E,  # Need QAQC
            'ASIXY_TI1': ASIXY_TI1,  # Needs QAQC
            'ASIXY_TI2': ASIXY_TI2,  # Needs QAQC
            'ASIXY_TI3': ASIXY_TI3,  # Needs QAQC
            'ABIXY_TI1': ABIXY_TI1,  # Needs QAQC
            'ABIXY_TI2': ABIXY_TI2,  # Needs QAQC
            'ABIZ': ABIZ,  # Needs QAQC
            'ASIZ': ASIZ,  # Needs QAQC
            'MBIXY_TI1': MBIXY_TI1,  # Needs QAQC
            'MBIXY_TI2': MBIXY_TI2,  # Needs QAQC
            'MDI': MDI,  # Needs QAQC
            'AXYZ_MIS': AXYZ_MIS,  # Needs QAQC
            'AXYZ_SF': AXYZ_SF,  # Needs QAQC
            'AXYZ_ZB': AXYZ_ZB,  # Needs QAQC
            'GXY_B1': GXY_B1,  # Needs QAQC
            'GXY_B2': GXY_B2,  # Needs QAQC
            'GXY_G1': GXY_G1,  # Needs QAQC
            'GXY_G4': GXY_G4,  # Needs QAQC
            'GXY_RN': GXY_RN,  # Needs QAQC
            'GXY_GD': GXY_GD,  # Needs QAQC
            'GXY_GRW': GXY_GRW,  # Needs QAQC
            'MFI': MFI,  # Needs QAQC
            'MSIXY_TI1': MSIXY_TI1,  # Needs QAQC
            'MSIXY_TI2': MSIXY_TI1,  # Needs QAQC
            'MSIXY_TI3': MSIXY_TI1,  # Needs QAQC
            'AMID': AMID,  # Needs QAQC
            'CNA': CNA,  # Needs QAQC
            'CNI': CNI,  # Needs QAQC
        }

    def extract_tortuosity(self, error_from_edm: bool = False):
        # Tortuosity here is used for calculating dp/de based on tool error codes.
        # Some tools use tortuosity for this calculation
        # Ideally, we want tortuosity to be 0 but since this is a
        # recommendation by ISCWSA kept as is.
        if error_from_edm:
            # Not sure how to get tortuosity based on survey tools from EDM.
            self.tortuosity = None
            return

        if 'Default Tortusity (rad/m)' in self.em['header']:
            self.tortuosity = self.em['header']['Default Tortusity (rad/m)']
        elif 'XCL Tortuosity' in self.em['header']:
            # assuming that this is always 1 deg / 100 ft but this might not
            # be the case
            # TODO use pint to handle this string inputs
            self.tortuosity = (np.radians(1.) / 100) * 3.281
        else:
            self.tortuosity = None


def _funky_denominator(error):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.nan_to_num(
            (
                1 - sin(error.survey.inc_rad) ** 2
                * sin(error.survey.azi_mag_rad) ** 2
            ),
        )
    return result


def _get_ref_init_error(
        dpde: np.ndarray,
        error: "Error",
        **kwargs
) -> np.ndarray:
    """
    Function that identifies where the continuous gyro begins, initiates and
    then carries the static errors during the continuous modes.
    """
    temp = [0.0]
    for coeff, inc in zip(dpde[1:, 2], error.survey.inc_rad[1:]):
        if inc > kwargs['header']['XY Static Gyro']['End Inc']:
            temp.append(temp[-1])
        else:
            temp.append(coeff)
    dpde[:, 2] = temp

    return dpde


def get_initial_error_gyro(error: "Error", **kwargs) -> List:
    """
    Calculate initial error for a tool based on static gyro
    """

    init_error = []
    for i, (u, l) in enumerate(zip(
        error.survey.inc_rad[1:], error.survey.inc_rad[:-1]
    )):
        init_error.append(0.0)
        if all((
            u > kwargs['header']['XY Static Gyro']['End Inc'],
            l <= kwargs['header']['XY Static Gyro']['End Inc']
        )):
            for tool in kwargs['errors'].gyro_stationary:
                temp = kwargs['errors'].errors[tool].e_DIA[i - 1][2]
                if tool in ['GXY_RN']:
                    temp *= kwargs['header']['Noise Reduction Factor']
                init_error[-1] += temp

    return init_error


# error functions #
def DREF(code, error, mag=0.35, propagation='random', NEV=True, **kwargs):
    dpde = np.full((len(error.survey_rad), 3), [1., 0., 0.])
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def DSF(code, error, mag=0.00056, propagation='systematic', NEV=True, **kwargs):
    dpde = np.full((len(error.survey_rad), 3), [1., 0., 0.])
    dpde = dpde * np.array(error.survey_rad)
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def DST(
    code, error, mag=0.00000025, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.full((len(error.survey_rad), 3), [1., 0., 0.])
    dpde[:, 0] = error.survey.tvd
    dpde = dpde * np.array(error.survey_rad)
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ABIZ(
    code, error, mag=0.0040, propagation='systematic', NEV=True, **kwargs
):
    denom = _funky_denominator(error) / error.survey.header.G
    denom = np.where(denom > ACCURACY, denom, ACCURACY)

    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = -sin(error.survey.inc_rad) / error.survey.header.G
    dpde[:, 2] = (
        sin(error.survey.inc_rad)
        * cos(error.survey.inc_rad)
        * sin(error.survey.azi_mag_rad)
        * (
            tan(error.survey.header.dip)
            * cos(error.survey.inc_rad)
            + sin(error.survey.inc_rad) * cos(error.survey.azi_mag_rad)
        )
    ) / denom

    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ABIXY_TI1(
    code, error, mag=0.0040, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = -cos(error.survey.inc_rad) / error.survey.header.G
    dpde[:, 2] = (
        cos(error.survey.inc_rad) ** 2
        * sin(error.survey.azi_mag_rad)
        * (
            tan(error.survey.header.dip)
            * cos(error.survey.inc_rad)
            + sin(error.survey.inc_rad) * cos(error.survey.azi_mag_rad)
        )
    ) / (
        error.survey.header.G * (
            _funky_denominator(error)
        )
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ABXY_TI1(
    code, error, mag=0.0040, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = -cos(error.survey.inc_rad) / error.survey.header.G
    dpde[:, 2] = (
        cos(error.survey.inc_rad)
        * tan(error.survey.header.dip)
        * sin(error.survey.azi_mag_rad)
    ) / error.survey.header.G
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ABIXY_TI2(
    code, error, mag=0.004, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.nan_to_num(
            (
                -(
                    tan(error.survey.header.dip)
                    * cos(error.survey.azi_mag_rad)
                    - tan(
                        pi / 2 - error.survey.inc_rad
                    )
                ) / (
                    error.survey.header.G
                    * (
                        _funky_denominator(error)
                    )
                )
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
            (error.survey.md[1] - error.survey.md[0])
            * mag
            * (
                cos(error.survey.azi_true_rad[1])
                / error.survey.header.G
            )
        )
        e_NEV_star[sing] = e_NEV_star_sing[sing]

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )


def ABXY_TI2(
    code, error, mag=0.004, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.nan_to_num(
            (
                (
                    tan(-(error.survey_rad[:, 1]) + (pi / 2))
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
        if error.error_model.lower().split(' ')[-1] != 'rev4':
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
        if error.error_model.lower().split(' ')[-1] != 'rev4':
            e_NEV_star_sing[1, 1] = (
                (error.survey.md[1] - error.survey.md[0])
                * mag
                * (
                    cos(error.survey.azi_true_rad[1])
                    / error.survey.header.G
                )
            )
        e_NEV_star[sing] = e_NEV_star_sing[sing]

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )


def AMID(
    code,
    error,
    mag=0.04363323129985824,
    propagation='systematic',
    NEV=True
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        sin(error.survey.inc_rad)
        * sin(error.survey.azi_mag_rad)
    )
    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def ABZ(code, error, mag=0.004, propagation='systematic', NEV=True, **kwargs):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = -sin(np.array(error.survey_rad)[:, 1]) / error.survey.header.G
    dpde[:, 2] = (
        sin(np.array(error.survey_rad)[:, 1])
        * tan(error.survey.header.dip) * sin(error.survey.azi_mag_rad)
    ) / error.survey.header.G
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASXY_TI1(
    code, error, mag=0.0005, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = (
        sin(error.survey.inc_rad)
        * cos(error.survey.inc_rad)
    ) / sqrt(2)
    dpde[:, 2] = (
        sin(error.survey.inc_rad)
        * -tan(error.survey.header.dip)
        * cos(error.survey.inc_rad)
        * sin(error.survey.azi_mag_rad)
    ) / sqrt(2)
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASIXY_TI1(
    code, error, mag=0.0005, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = (
        sin(error.survey.inc_rad)
        * cos(error.survey.inc_rad)
        / sqrt(2)
    )
    dpde[:, 2] = -(
        sin(error.survey.inc_rad)
        * cos(error.survey.inc_rad) ** 2
        * sin(error.survey.azi_mag_rad)
        * (
            tan(error.survey.header.dip)
            * cos(error.survey.inc_rad)
            + sin(error.survey.inc_rad)
            * cos(error.survey.azi_mag_rad)
        )
    ) / (
        sqrt(2) * _funky_denominator(error)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASXY_TI2(
    code, error, mag=0.0005, propagation='systematic', NEV=True, **kwargs
):
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


def ASIXY_TI2(
    code, error, mag=0.0005, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = (
        sin(error.survey.inc_rad)
        * cos(error.survey.inc_rad)
        / 2
    )
    dpde[:, 2] = -(
        sin(error.survey.inc_rad)
        * cos(error.survey.inc_rad) ** 2
        * sin(error.survey.azi_mag_rad)
        * (
            tan(error.survey.header.dip)
            * cos(error.survey.inc_rad)
            + sin(error.survey.inc_rad) * cos(error.survey.azi_mag_rad)
        )
    ) / (
        2 * _funky_denominator(error)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASXY_TI3(
    code, error, mag=0.0005, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        sin(np.array(error.survey_rad)[:, 1])
        * tan(error.survey.header.dip) * cos(error.survey.azi_mag_rad)
        - cos(np.array(error.survey_rad)[:, 1])) / 2
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASIXY_TI3(
    code, error, mag=0.0005, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        tan(error.survey.header.dip)
        * sin(error.survey.inc_rad)
        * cos(error.survey.azi_mag_rad)
        - cos(error.survey.inc_rad)
    ) / (
        2 * _funky_denominator(error)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def ASZ(code, error, mag=0.0005, propagation='systematic', NEV=True, **kwargs):
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


def ASIZ(
    code, error, mag=0.0005, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = (
        -sin(error.survey.inc_rad)
        * cos(error.survey.inc_rad)
    )
    dpde[:, 2] = (
        sin(error.survey.inc_rad)
        * cos(error.survey.inc_rad) ** 2
        * sin(error.survey.azi_mag_rad)
        * (
            tan(error.survey.header.dip)
            * cos(error.survey.inc_rad)
            + sin(error.survey.inc_rad)
            * cos(error.survey.azi_mag_rad)
        )
    ) / (
        _funky_denominator(error)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def AXYZ_MIS(
    code, error, mag=0.0001658062789394613, propagation='systematic', NEV=True,
    **kwargs
):
    """
    SPE 90408 Table 1
    """
    dpde = np.full((len(error.survey_rad), 3), [0., 1., 0.])
    dpde = dpde * np.array(error.survey_rad)
    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def AXYZ_SF(
    code, error, mag=0.000111, propagation='systematic', NEV=True,
    **kwargs
):
    """
    SPE 90408 Table 1
    """
    dpde = np.full((len(error.survey_rad), 3), [0., 1., 0.])
    dpde[:, 1] = (
        1.3 * sin(error.survey.inc_rad) * cos(error.survey.inc_rad)
    )
    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def AXYZ_ZB(
    code, error, mag=0.0017, propagation='systematic', NEV=True,
    **kwargs
):
    """
    SPE 90408 Table 1
    """
    dpde = np.full((len(error.survey_rad), 3), [0., 1., 0.])
    dpde[:, 1] = (
        sin(error.survey.inc_rad) / error.survey.header.G
    )
    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def CNA(code, error, mag=0.35, propagation='systematic', NEV=True, **kwargs):
    dpde = np.full((len(error.survey_rad), 3), [0., 0., 0.])
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.nan_to_num(
            1 / sin(error.survey.inc_rad),
            posinf=1,
            neginf=-1
        )

    e_DIA = dpde * mag

    sing = np.where(
        error.survey.inc_rad < error.survey.header.vertical_inc_limit
    )
    if len(sing[0]) < 1:
        return error._generate_error(code, e_DIA, propagation, NEV)
    else:
        e_NEV = error._e_NEV(e_DIA)
        n = (
            np.array(0.5 * error.drdp_sing['double_delta_md'])
            * -sin(getattr(
                error.survey, f"azi_{error.survey.header.azi_reference}_rad"
            )[1: -1])
            * mag
        )
        e = (
            np.array(0.5 * error.drdp_sing['double_delta_md'])
            * cos(getattr(
                error.survey, f"azi_{error.survey.header.azi_reference}_rad"
            )[1: -1])
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
            * -sin(getattr(
                error.survey, f"azi_{error.survey.header.azi_reference}_rad"
            )[1: -1])
            * mag
        )
        e = (
            np.array(0.5 * error.drdp_sing['delta_md'])
            * cos(getattr(
                error.survey, f"azi_{error.survey.header.azi_reference}_rad"
            )[1: -1])
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

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )


def CNI(
    code, error, mag=0.35, propagation='systematic', NEV=True,
    **kwargs
):
    dpde = np.full((len(error.survey_rad), 3), [0., 1., 0.])

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def GXY_B1(
    code, error, mag=0.002617993877991494, propagation='random',
    NEV=True, **kwargs
):
    """
    SPE 90408 Table 4
    """

    dpde = np.full((len(error.survey_rad), 3), [0., 0., 1.])
    dpde[:, 2] = np.where(
        error.survey.inc_rad <= kwargs['header']['XY Static Gyro']['End Inc'],
        sin(error.survey.azi_true_rad)
        / (
            error.survey.header.earth_rate
            * cos(np.radians(error.survey.header.latitude))
            * cos(error.survey.inc_rad)
        ),
        np.zeros_like(error.survey.md)
    )
    dpde = _get_ref_init_error(dpde, error, **kwargs)

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def GXY_B2(
    code, error, mag=0.002617993877991494, propagation='random',
    NEV=True, **kwargs
):
    """
    SPE 90408 Table 4
    """
    dpde = np.full((len(error.survey_rad), 3), [0., 0., 1.])
    dpde[:, 2] = np.where(
        error.survey.inc_rad <= kwargs['header']['XY Static Gyro']['End Inc'],
        cos(error.survey.azi_true_rad)
        / (
            error.survey.header.earth_rate
            * cos(np.radians(error.survey.header.latitude))
        ),
        np.zeros_like(error.survey.md)
    )
    dpde = _get_ref_init_error(dpde, error, **kwargs)

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def GXY_G1(
    code, error, mag=0.006981317007977318, propagation='systematic',
    NEV=True, **kwargs
):
    """
    SPE 90408 Table 4
    """
    dpde = np.full((len(error.survey_rad), 3), [0., 0., 1.])
    dpde[:, 2] = np.where(
        error.survey.inc_rad <= kwargs['header']['XY Static Gyro']['End Inc'],
        cos(error.survey.azi_true_rad) * sin(error.survey.inc_rad)
        / (
            error.survey.header.earth_rate
            * cos(np.radians(error.survey.header.latitude))
        ),
        np.zeros_like(error.survey.md)
    )
    dpde = _get_ref_init_error(dpde, error, **kwargs)

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def GXY_G4(
    code, error, mag=0.010471975511965976, propagation='systematic',
    NEV=True, **kwargs
):
    """
    SPE 90408 Table 4
    """
    dpde = np.full((len(error.survey_rad), 3), [0., 0., 1.])
    dpde[:, 2] = np.where(
        error.survey.inc_rad <= kwargs['header']['XY Static Gyro']['End Inc'],
        sin(error.survey.azi_true_rad) * tan(error.survey.inc_rad)
        / (
            error.survey.header.earth_rate
            * cos(np.radians(error.survey.header.latitude))
        ),
        np.zeros_like(error.survey.md)
    )
    dpde = _get_ref_init_error(dpde, error, **kwargs)

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def GXY_RN(
    code, error, mag=0.006981317007977318, propagation='random',
    NEV=True, **kwargs
):
    """
    SPE 90408 Table 4
    """
    dpde = np.full((len(error.survey_rad), 3), [0., 0., 1.])
    dpde[:, 2] = np.where(
        error.survey.inc_rad <= kwargs['header']['XY Static Gyro']['End Inc'],
        1.0
        * (
            np.sqrt(
                1 - cos(error.survey.azi_true_rad) ** 2
                * sin(error.survey.inc_rad) ** 2
            )
            / (
                error.survey.header.earth_rate
                * cos(np.radians(error.survey.header.latitude))
                * cos(error.survey.inc_rad)
            )
        ),
        np.zeros_like(error.survey.md)
    )
    dpde = _get_ref_init_error(dpde, error, **kwargs)
    dpde_systematic = np.zeros_like(dpde)
    index_systematic = np.where(
        error.survey.inc_rad > kwargs['header']['XY Static Gyro']['End Inc']
    )
    np.put(
        dpde_systematic[:, 2],
        index_systematic,
        (
            dpde[index_systematic][:, 2]
            * kwargs['header']['Noise Reduction Factor']
        )
    )
    e_DIA_systematic = dpde_systematic * mag

    result_systematic = error._generate_error(
        code, e_DIA_systematic, 'systematic', NEV
    )

    np.put(
        dpde[:, 2],
        index_systematic,
        np.zeros(len(index_systematic))
    )

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    result.cov_NEV += result_systematic.cov_NEV

    return result


def GXY_GD(
    code, error, mag=0.008726646259971648, propagation='systematic',
    NEV=True, **kwargs
):
    """
    SPE 90408 Table 7
    """

    gyro_header = kwargs['header']
    dpde = np.full((len(error.survey_rad), 3), [0., 0., 1.])
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.where(
            error.survey.inc_rad > gyro_header['XY Static Gyro']['End Inc'],
            np.append(
                np.array([0]),
                (
                    (error.survey.md[1:] - error.survey.md[:-1])
                    / (float(gyro_header['XY Continuous Gyro']['Running Speed'].split()[0])
                       * sin((error.survey.inc_rad[1:] + error.survey.inc_rad[:-1]) / 2)
                       )
                )
            ),
            np.zeros_like(error.survey.md)
        )

    init_error = get_initial_error_gyro(error, kwargs)
    temp = [0.0]
    for i, (u, e) in enumerate(zip(dpde[1:, 2], init_error)):
        temp.append(0.0)
        if u != 0.0:
            temp[-1] += temp[-2] + u * mag
    dpde[:, 2] = temp

    e_DIA = dpde

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def GXY_GRW(
    code, error, mag=0.004363323129985824, propagation='systematic',
    NEV=True, **kwargs
):
    """
    SPE 90408 Table 7
    """

    dpde = np.full((len(error.survey_rad), 3), [0., 0., 1.])

    gryo_header = kwargs['header']
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.where(
            error.survey.inc_rad > gryo_header['XY Static Gyro']['End Inc'],
            np.append(
                np.array([0]),
                (error.survey.md[1:] - error.survey.md[:-1])
                / (float(gryo_header['XY Continuous Gyro']['Running Speed'].split()[0])
                    * sin((error.survey.inc_rad[1:] + error.survey.inc_rad[:-1]) / 2) ** 2
                   )
            ),
            np.zeros_like(error.survey.md)
        )
    init_error = get_initial_error_gyro(error, kwargs)
    temp = [0.0]
    for i, (u, e) in enumerate(zip(dpde[1:, 2], init_error)):
        temp.append(0.0)
        if u != 0.0:
            temp[-1] += np.sqrt(temp[-2] ** 2 + u * mag)
    dpde[:, 2] = temp

    e_DIA = dpde

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def MBXY_TI1(
    code, error, mag=70.0, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        -cos(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
    ) / (error.survey.header.b_total * cos(error.survey.header.dip))
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MBIXY_TI1(
    code, error, mag=70.0, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        -cos(error.survey.inc_rad)
        * sin(error.survey.azi_mag_rad)
    ) / (
        error.survey.header.b_total
        * cos(error.survey.header.dip)
        * (
            _funky_denominator(error)
        )
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MBXY_TI2(
    code, error, mag=70.0, propagation='systematic', NEV=True, **kwargs
):
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


def MBIXY_TI2(
    code, error, mag=70.0, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        cos(error.survey.azi_mag_rad)
        / (
            error.survey.header.b_total
            * cos(error.survey.header.dip)
            * (
                _funky_denominator(error)
            )
        )
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MBZ(code, error, mag=70.0, propagation='systematic', NEV=True, **kwargs):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        -sin(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
    ) / (error.survey.header.b_total * cos(error.survey.header.dip))
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MFI(
    code, error, mag=70, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        -sin(error.survey.inc_rad)
        * sin(error.survey.azi_mag_rad)
        * (
            tan(error.survey.header.dip)
            * cos(error.survey.inc_rad)
            + sin(error.survey.inc_rad)
            * cos(error.survey.azi_mag_rad)
        ) / (
            _funky_denominator(error)
        )
        / error.survey.header.b_total
    )

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def MSXY_TI1(
    code, error, mag=0.0016, propagation='systematic', NEV=True, **kwargs
):
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


def MSXY_TI2(
    code, error, mag=0.0016, propagation='systematic', NEV=True, **kwargs
):
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


def MSXY_TI3(
    code, error, mag=0.0016, propagation='systematic', NEV=True, **kwargs
):
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


def MSIXY_TI1(
    code, error, mag=0.0016, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        sin(error.survey.inc_rad)
        * sin(error.survey.azi_mag_rad)
        * (
            tan(error.survey.header.dip)
            * cos(error.survey.inc_rad)
            + sin(error.survey.inc_rad)
            * cos(error.survey.azi_mag_rad)
        ) / (
            sqrt(2)
            * (
                _funky_denominator(error)
            )
        )
    )

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def MSIXY_TI2(
    code, error, mag=0.0016, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        sin(error.survey.azi_mag_rad)
        * (
            tan(error.survey.header.dip)
            * sin(error.survey.inc_rad)
            * cos(error.survey.inc_rad)
            - cos(error.survey.inc_rad) ** 2
            * cos(error.survey.azi_mag_rad)
            - cos(error.survey.azi_mag_rad)
        ) / (
            2 * (
                _funky_denominator(error)
            )
        )
    )

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def MSIXY_TI3(
    code, error, mag=0.0016, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        (
            cos(error.survey.inc_rad)
            * cos(error.survey.azi_mag_rad) ** 2
            - cos(error.survey.inc_rad)
            * sin(error.survey.azi_mag_rad) ** 2
            - tan(error.survey.header.dip)
            * sin(error.survey.inc_rad)
            * cos(error.survey.azi_mag_rad)
        ) / (
            2 * (
                _funky_denominator(error)
            )
        )
    )

    e_DIA = dpde * mag

    result = error._generate_error(code, e_DIA, propagation, NEV)

    return result


def MSZ(
    code, error, mag=0.0016, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = -(
        sin(np.array(error.survey_rad)[:, 1])
        * cos(error.survey.azi_mag_rad)
        + tan(error.survey.header.dip) * cos(np.array(error.survey_rad)[:, 1])
    ) * sin(np.array(error.survey_rad)[:, 1]) * sin(error.survey.azi_mag_rad)
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def AZ(code, error, mag=0.00628, propagation='systematic', NEV=True, **kwargs):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = 1
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def DBH(
    code, error, mag=np.radians(0.09), propagation='systematic', NEV=True,
    **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = 1 / (
        error.survey.header.b_total * cos(error.survey.header.dip)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def MDI(
    code, error, mag=np.radians(5000), propagation='systematic', NEV=True,
    **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        -sin(error.survey.inc_rad)
        * sin(error.survey.azi_mag_rad)
        * (
            cos(error.survey.inc_rad)
            - tan(error.survey.header.dip)
            * sin(error.survey.inc_rad)
            * cos(error.survey.azi_mag_rad)
        )
    ) / (
        _funky_denominator(error)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def DBHR(
    code, error, mag=np.radians(3000), propagation='random', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = 1 / (
        error.survey.header.b_total * cos(error.survey.header.dip)
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def AMIL(code, error, mag=220.0, propagation='systematic', NEV=True, **kwargs):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = (
        -sin(np.array(error.survey_rad)[:, 1])
        * sin(error.survey.azi_mag_rad)
        / (error.survey.header.b_total * cos(error.survey.header.dip))
    )
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def SAG(
    code, error, mag=0.00349, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = sin(np.array(error.survey_rad)[:, 1])
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def SAGE(
    code, error, mag=0.00175, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = sin(np.array(error.survey.inc_rad)) ** 0.25
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def XYM1(
    code, error, mag=0.00175, propagation='systematic', NEV=True, **kwargs
):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 1] = np.absolute(sin(np.array(error.survey.inc_rad)))
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def XYM2(
    code, error, mag=0.00175, propagation='systematic', NEV=True, **kwargs
):
    propagation = 'systematic'  # incorrect in the rev5 model tab
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[:, 2] = -1
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def XYM3(
    code, error, mag=0.00175, propagation='systematic', NEV=True, **kwargs
):
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
        e_NEV = error._e_NEV(e_DIA)
        e_NEV_sing = np.zeros_like(e_NEV)
        e_NEV_sing[:, 0] = e_NEV[:, 0]
        e_NEV[sing] = e_NEV_sing[sing]

        e_NEV_star = error._e_NEV_star(e_DIA)
        e_NEV_star_sing = np.zeros_like(e_NEV_star)
        e_NEV_star_sing[:, 0] = e_NEV_star[:, 0]
        e_NEV_star[sing] = e_NEV_star_sing[sing]

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )

    return error._generate_error(code, e_DIA, propagation, NEV)


def XYM4(
    code, error, mag=0.00175, propagation='systematic', NEV=True, **kwargs
):
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


def XYM4E(code, error, mag=0.00524, propagation='random', NEV=True, **kwargs):
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
    dpde[1:, 1] = (
        cos(error.survey.inc_rad[1:])
        * sin(error.survey.azi_true_rad[1:])
        * coeff[1:]
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[1:, 2] = np.nan_to_num(
            (
                (
                    cos(error.survey.inc_rad[1:])
                    * cos(error.survey.azi_true_rad[1:])
                    / sin(error.survey.inc_rad[1:])
                )
                * coeff[1:]
            ),
            posinf=0,
            neginf=0
        )

    e_DIA = dpde * mag

    sing = np.where(
        error.survey.inc_rad < error.survey.header.vertical_inc_limit
    )
    if len(sing[0]) < 1:
        return error._generate_error(code, e_DIA, propagation, NEV)
    else:
        # this is a bit of a cop out way of handling these exceptions, but it's
        # simple and it works...
        xym3e = XYM3E(
            code, error, mag=mag, propagation=propagation, NEV=NEV
        )
        e_NEV = error._e_NEV(e_DIA)
        e_NEV_sing = np.zeros_like(e_NEV)
        e_NEV_sing[:, 1] = xym3e.e_NEV[:, 0]
        e_NEV[sing] = e_NEV_sing[sing]

        e_NEV_star = error._e_NEV_star(e_DIA)
        e_NEV_star_sing = np.zeros_like(e_NEV_star)
        e_NEV_star_sing[:, 1] = xym3e.e_NEV_star[:, 0]
        e_NEV_star[sing] = e_NEV_star_sing[sing]

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )


def XCL(code, error, mag=0.0167, propagation='random', NEV=True, **kwargs):
    """
    Dummy function to manage the ISCWSA workbook not correctly defining the
    weighting functions.
    """
    tortuosity = kwargs['tortuosity']
    if code == "XCLA":
        return XCLA(
            code, error, mag=mag, propagation=propagation, NEV=NEV,
            tortuosity=tortuosity
        )
    else:
        return XCLH(
            code, error, mag=mag, propagation=propagation, NEV=NEV,
            tortuosity=tortuosity
        )


def XCLA(code, error, mag=0.167, propagation='random', NEV=True, **kwargs):
    dpde = np.zeros((len(error.survey_rad), 3))

    def manage_sing(error, kwargs):
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
            manage_sing(error, kwargs),
            (
                kwargs['tortuosity']
                * (error.survey.md[1:] - error.survey.md[0:-1])
            )
        ), axis=-1), axis=-1)
        * -sin(error.survey.azi_true_rad[1:])
    )

    dpde[1:, 1] = (
        (error.survey.md[1:] - error.survey.md[0:-1])
        * np.amax(np.stack((
            manage_sing(error, kwargs),
            (
                kwargs['tortuosity']
                * (error.survey.md[1:] - error.survey.md[0:-1])
            )
        ), axis=-1), axis=-1)
        * cos(error.survey.azi_true_rad[1:])
    )

    e_DIA = dpde * mag

    return error._generate_error(
        code, e_DIA, propagation, NEV, e_NEV=e_DIA, e_NEV_star=e_DIA
    )


def XCLH(code, error, mag=0.0167, propagation='random', NEV=True, **kwargs):
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[1:, 0] = (
        (error.survey.md[1:] - error.survey.md[0:-1])
        * np.amax(np.stack((
            np.absolute(
                (error.survey.inc_rad[1:] - error.survey.inc_rad[:-1])
            ),
            (
                kwargs['tortuosity']
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
                kwargs['tortuosity']
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
                kwargs['tortuosity']
                * (error.survey.md[1:] - error.survey.md[0:-1])
            )
        ), axis=-1), axis=-1)
        * -sin(error.survey.inc_rad[1:])
    )

    e_DIA = dpde * mag

    return error._generate_error(
        code, e_DIA, propagation, NEV, e_NEV=e_DIA, e_NEV_star=e_DIA
    )


def XYM3L(code, error, mag=0.0167, propagation='random', NEV=True, **kwargs):
    coeff = np.ones(len(error.survey.md) - 1)
    coeff = np.amax(np.stack((
        coeff,
        sqrt(
            10 / (error.survey.md[1:] - error.survey.md[:-1])
        )
    ), axis=-1), axis=-1)

    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[1:, 1] = np.absolute(
        cos(error.survey.inc_rad[1:])
        * cos(error.survey.azi_true_rad[1:])
        * coeff
    )
    dpde[0, 1] = dpde[1, 1]

    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[1:, 2] = np.nan_to_num(
            (
                -np.absolute(
                    cos(error.survey.inc_rad[1:])
                )
                * (
                    sin(error.survey.azi_true_rad[1:])
                    / sin(error.survey.inc_rad[1:])
                )
                * coeff
            ),
            posinf=0,
            neginf=0
        )

    dpde[0, 2] = dpde[1, 2]

    e_DIA = dpde * mag

    sing = np.where(
        error.survey_rad[:, 1] < error.survey.header.vertical_inc_limit
    )
    if len(sing[0]) < 1:
        return error._generate_error(code, e_DIA, propagation, NEV)
    else:
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

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )


def XYM4L(code, error, mag=0.0167, propagation='random', NEV=True, **kwargs):
    propagation = 'random'
    coeff = np.ones(len(error.survey.md))
    coeff[1:] = np.amax(np.stack((
        coeff[1:],
        sqrt(
            10 / (error.survey.md[1:] - error.survey.md[:-1])
        )
    ), axis=-1), axis=-1)

    dpde = np.zeros((len(error.survey_rad), 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.nan_to_num(
            np.absolute(
                cos(error.survey.inc_rad)
                * cos(error.survey.azi_true_rad)
                / sin(error.survey.inc_rad)
                * coeff
            ),
            posinf=0,
            neginf=0,
        )

    dpde[:, 1] = (
        np.absolute(
            cos(error.survey.inc_rad)
        )
        * (
            sin(error.survey.azi_true_rad)
        )
        * coeff
    )

    e_DIA = dpde * mag

    sing = np.where(
        error.survey_rad[:, 1] < error.survey.header.vertical_inc_limit
    )
    if len(sing[0]) < 1:
        return error._generate_error(code, e_DIA, propagation, NEV)
    else:
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

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )
