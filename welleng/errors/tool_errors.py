import numpy as np
from numpy import sin, cos, tan, pi, sqrt
from numpy.char import index
import json
import yaml
import os
from collections import OrderedDict
# import imp

# import welleng.error
from ..utils import NEV_to_HLA
from .interpreter import evaluate_formula


# Map ISCWSA JSON propagation_mode enum -> the lowercase string the
# legacy welleng dispatch expects.
_JSON_PROP_TO_LEGACY = {
    "Random": "random",
    "Systematic": "systematic",
    "Global": "global",
    "Well": "well",
}


# Unit-conversion table: declared unit -> multiplier to convert magnitude
# to the SI / base-unit form welleng's downstream propagation expects.
# Legacy YAML tools store magnitudes already in this base; the new
# JSON converter copies OWSG-xlsx values verbatim, so the dispatcher
# converts them at evaluation time.
_MAG_UNIT_TO_BASE = {
    "m":       1.0,            # depth / length already in metres
    "1/m":     1.0,
    "nT":      1.0,            # magnetic field already in nT
    "m/s2":    1.0,            # accel already in m/s^2
    "rad":     1.0,            # angles already in radians
    "deg":     np.pi / 180.0,  # angle: degrees -> radians
    "deg/hr":  np.pi / 180.0,  # gyro rate: deg/hr -> rad/hr
    "deg/nT":  np.pi / 180.0,  # B-field-coupled angle gradient
    "-":       1.0,            # dimensionless (scale factors)
    "":        1.0,            # missing unit treated as dimensionless
}


def _resolve_json_model(model_name: str) -> str | None:
    """Find the ISCWSA-format JSON tool model for the given name.

    `model_name` may be either an OWSG prefix (e.g. 'A020Gb') or a
    Short Name (e.g. 'GYRO-NS'). Searches every JSON under
    welleng/errors/iscwsa_json/ and matches against the file basename,
    its metadata.model_id, and its metadata.short_name.

    Returns the absolute path of the first match, or None if no JSON
    file matches.
    """
    json_root = os.path.join(PATH, 'iscwsa_json')
    if not os.path.isdir(json_root):
        return None
    # Cheap path: filename matches model name directly.
    for sub in os.listdir(json_root):
        sub_p = os.path.join(json_root, sub)
        if os.path.isdir(sub_p):
            cand = os.path.join(sub_p, f"{model_name}.json")
            if os.path.isfile(cand):
                return cand
    # Fallback: walk every JSON, peek metadata for matching id / short_name.
    for sub in os.listdir(json_root):
        sub_p = os.path.join(json_root, sub)
        if not os.path.isdir(sub_p):
            continue
        for fn in os.listdir(sub_p):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(sub_p, fn)
            try:
                with open(path) as f:
                    md = json.load(f).get("metadata", {})
            except Exception:
                continue
            if (md.get("model_id") == model_name
                    or md.get("short_name") == model_name):
                return path
    return None


def _json_to_em_adapter(model: dict) -> dict:
    """Adapt an ISCWSA-format JSON tool model to the YAML-like dict
    structure the legacy ToolError code expects (`header` + `codes`).

    The resulting dict's ``codes`` entries carry the formula strings
    (``depth_formula``, ``inclination_formula``, ``azimuth_formula``)
    in addition to the legacy ``magnitude`` / ``propagation`` /
    ``unit`` fields, so the interpreter-driven term-evaluation path
    has everything it needs without re-reading the JSON.
    """
    md = model.get("metadata", {})
    pp = model.get("parameters", {})
    inc_max_deg = pp.get("inc_max", 180)
    inc_min_deg = pp.get("inc_min", 0)
    header = {
        "Short Name": md.get("short_name", ""),
        "Long Name": md.get("long_name", ""),
        "Application": md.get("application", ""),
        "Source": md.get("source", ""),
        "Tool Type": md.get("tool_type", ""),
        "Inclination Range Min": f"{inc_min_deg} deg",
        "Inclination Range Max": f"{inc_max_deg} deg",
        "Revision No": md.get("revision_number", ""),
        "Revision Date": md.get("revision_date", ""),
        # Required for the legacy code path to be happy when JSON tool
        # carries no tortuosity (we synthesise a default).
        "Default Tortusity (rad/m)": 0.000572615,
    }
    codes = OrderedDict()
    for term in model.get("terms", []):
        name = term["name"]
        codes[name] = {
            "function": "_INTERPRETER_",
            "magnitude": float(term["value"]),
            "propagation": _JSON_PROP_TO_LEGACY.get(
                term["propagation_mode"], "systematic"
            ),
            "unit": term.get("units", ""),
            # Carry through the formula strings for the interpreter.
            "_iscwsa_term": term,
        }
    return {"header": header, "codes": codes}

# since this is running on different OS flavors
PATH = os.path.dirname(__file__)
TOOL_INDEX = os.path.join(
    '', *[PATH, 'tool_index.yaml']
)

ACCURACY = 1e-6


class ToolError:
    def __init__(
        self,
        error,
        model
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

        # Resolve the tool model file. Try the legacy YAML location
        # first (welleng/errors/tool_codes/<model>.yaml); if it doesn't
        # exist, fall back to an ISCWSA-format JSON tool model under
        # welleng/errors/iscwsa_json/. Gyro tools ship via the JSON
        # path -- the legacy YAML weight functions for AXYZ_*/GXY_*
        # were deleted in favour of the formula-string interpreter
        # (welleng/errors/interpreter.py).
        #
        # The dispatch in welleng.error.ErrorModel maps the user's
        # `error_model='X'` (a Short Name) to an OWSG prefix `model`.
        # Some JSON files are named by Short Name (the converter's
        # default), so we walk the iscwsa_json/ tree looking at both
        # the OWSG prefix and the metadata.short_name of each JSON.
        yaml_filename = os.path.join(PATH, 'tool_codes', f"{model}.yaml")
        json_filename = _resolve_json_model(model)

        self._json_path = None
        if os.path.isfile(yaml_filename):
            with open(yaml_filename, 'r') as file:
                self.em = yaml.safe_load(file)
        elif json_filename is not None:
            with open(json_filename) as file:
                self._json_model = json.load(file)
            self._json_path = json_filename
            # Adapter: shape the JSON into a YAML-like dict so the
            # downstream code that expects ``self.em['header']`` and
            # ``self.em['codes']`` keeps working without a refactor.
            self.em = _json_to_em_adapter(self._json_model)
        else:
            raise FileNotFoundError(
                f"No tool model file found for {model!r}. Searched:\n"
                f"  YAML: {yaml_filename}\n"
                f"  JSON: {os.path.join(PATH, 'iscwsa_json', '**')}"
            )

        # self.em = iscwsa_error_models[model]
        #     iscwsa_error_models = yaml.safe_load(file)
        # self.em = iscwsa_error_models[model]
        if 'Default Tortusity (rad/m)' in self.em['header']:
            self.tortuosity = self.em['header']['Default Tortusity (rad/m)']
        elif 'XCL Tortuosity' in self.em['header']:
            # assuming that this is always 1 deg / 100 ft but this might not
            # be the case
            # TODO use pint to handle this string inputs
            self.tortuosity = (np.radians(1.) / 100) * 3.281
        else:
            self.tortuosity = None

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
            entry = self.em['codes'][err]
            func = entry['function']
            mag = entry['magnitude']
            propagation = entry['propagation']
            if func == "_INTERPRETER_":
                # JSON-tool path: evaluate the term's formula strings
                # against per-station bindings, build dpde, hand off to
                # the existing _generate_error machinery.
                self.errors[err] = self._call_interpreter(
                    code=err,
                    term=entry["_iscwsa_term"],
                    mag=mag,
                    propagation=propagation,
                )
            else:
                self.errors[err] = self.call_func(
                    code=err,
                    func=func,
                    error=self.e,
                    mag=mag,
                    propagation=propagation,
                    tortuosity=self.tortuosity,
                    header=self.em['header'],
                    errors=self,
                )

        shape = (len(self.e.survey_rad), 3, 3)
        self.cov_NEVs = np.zeros(shape)
        self.cov_NEVs_random = np.zeros(shape)
        self.cov_NEVs_systematic = np.zeros(shape)
        self.cov_NEVs_global = np.zeros(shape)
        self.cov_NEVs_within_pad = np.zeros(shape)
        for _, value in self.errors.items():
            self.cov_NEVs += value.cov_NEV
            if value.propagation == 'random':
                self.cov_NEVs_random += value.cov_NEV
            elif value.propagation == 'systematic':
                self.cov_NEVs_systematic += value.cov_NEV
            elif value.propagation == 'global':
                self.cov_NEVs_global += value.cov_NEV
            elif value.propagation == 'within_pad':
                self.cov_NEVs_within_pad += value.cov_NEV

        self.cov_HLAs = NEV_to_HLA(self.e.survey_rad, self.cov_NEVs)

    def _get_the_func_out(self, err):
        if err in self.exceptional_funcs:
            func = self.exceptional_funcs[err]
        else:
            func = self.em['codes'][err]['function']

        return func

    def call_func(self, code, func, error, mag, propagation, **kwargs):
        """
        Function for calling functions by mapping function labels to their
        functions.
        """
        assert func in self.func_dict, f"no function for function {func}"

        return self.func_dict[func](code, error, mag, propagation, **kwargs)

    def _call_interpreter(self, code, term, mag, propagation):
        """JSON-tool term evaluation via formula-string interpreter.

        Mirrors the `call_func` contract for legacy hand-coded weight
        functions: returns an `Error` object with cov_NEV computed via
        the existing `error._generate_error` machinery. The only
        difference is the source of the per-station per-axis dpde --
        the interpreter evaluates the term's depth/inclination/azimuth
        formula strings against per-station survey bindings.

        Terms that fail to evaluate (formula references variables not
        bound -- e.g. cross-station MDPrev/AzPrev/IncPrev, or per-tool
        calibration constants like NoiseReductionFactor) raise. These
        are the schema-gap findings catalogued for the ISCWSA Discussion
        post; see welleng/errors/conformance.py output.
        """
        survey = self.e.survey
        n = len(survey.md)
        bindings = {
            "MD": np.asarray(survey.md, dtype=float),
            "Inc": np.asarray(survey.inc_rad, dtype=float),
            "AzT": np.asarray(survey.azi_true_rad, dtype=float),
            "AzM": np.asarray(survey.azi_mag_rad, dtype=float),
            "Az": np.asarray(survey.azi_grid_rad, dtype=float),
            "TVD": np.asarray(survey.tvd, dtype=float),
            "Gfield": float(survey.header.G),
            "Dip": float(survey.header.dip),
            "BField": float(survey.header.b_total or 50000.0),
            # EarthRate is needed by gyro-azimuth terms; standard value
            # in rad/hr (15.041 deg/hr).
            "EarthRate": 0.262516,
            "Latitude": np.radians(float(survey.header.latitude or 0.0)),
            "RAD": np.pi / 180.0,
        }
        try:
            d = evaluate_formula(term["depth_formula"], bindings)
            i = evaluate_formula(term["inclination_formula"], bindings)
            a = evaluate_formula(term["azimuth_formula"], bindings)
        except Exception as exc:
            # Term references variables the interpreter doesn't know how
            # to bind -- typically cross-station refs (MDPrev/AzPrev/
            # IncPrev) or per-tool calibration constants
            # (NoiseReductionFactor, XY_Gyro_Drift, ...). These are
            # documented schema gaps in the ISCWSA JSON spec; see
            # welleng/errors/conformance.py output for the catalogue.
            #
            # User-facing behaviour: emit a warning identifying the
            # missing variable, then contribute zero from this term so
            # the model as a whole still produces a usable Survey.
            import warnings
            warnings.warn(
                f"JSON tool model term {code!r} could not be evaluated "
                f"({exc}). Term contributes zero covariance to this Survey. "
                f"This is a known ISCWSA-JSON schema gap; see "
                f"welleng/errors/conformance.py.",
                RuntimeWarning,
            )
            d = np.zeros(n)
            i = np.zeros(n)
            a = np.zeros(n)
        d = np.broadcast_to(np.asarray(d, dtype=float), (n,))
        i = np.broadcast_to(np.asarray(i, dtype=float), (n,))
        a = np.broadcast_to(np.asarray(a, dtype=float), (n,))
        dpde = np.column_stack([d, i, a])
        # Convert the declared magnitude to welleng's SI base. OWSG xlsx
        # publishes magnitudes in the term's natural unit (deg, deg/hr,
        # m, ...); the propagation engine expects rad / m / m/s^2 /
        # rad/hr internally. The legacy YAML tools shipped pre-converted
        # values, so this conversion only fires for JSON-driven tools.
        unit = term.get("units") or term.get("unit") or "-"
        scale = _MAG_UNIT_TO_BASE.get(unit, 1.0)
        e_DIA = dpde * (mag * scale)
        return self.e._generate_error(code, e_DIA, propagation, NEV=True)

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
            'XYM3L': XYM3L,
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
            # Gyro-tool weight functions (AXYZ_*, GXY_*) previously lived
            # here as a half-finished first attempt flagged '# Needs QAQC'.
            # Deleted 2026-04-16 in favour of a JSON-schema + interpreter
            # path (welleng/errors/interpreter.py) that consumes the
            # ISCWSA error-models JSON schema directly. Gyro tools will be
            # added as JSON files driven by that interpreter, not as
            # hand-coded Python weight functions.
            'MFI': MFI,  # Needs QAQC
            'MSIXY_TI1': MSIXY_TI1,  # Needs QAQC
            'MSIXY_TI2': MSIXY_TI2,  # Needs QAQC
            'MSIXY_TI3': MSIXY_TI3,  # Needs QAQC
            'DBHR': DBHR,  # Needs QAQC
            'AMID': AMID,  # Needs QAQC
            'CNA': CNA,  # Needs QAQC
            'CNI': CNI,  # Needs QAQC
        }


def _funky_denominator(error):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.nan_to_num((
            1 - sin(error.survey.inc_rad) ** 2
            * sin(error.survey.azi_mag_rad) ** 2
            ),
            # nan=1e-6,
            # posinf=1.0,
            # neginf=-1.0
        )
    # ACCURACY = 1e-6
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     coeff = np.nan_to_num(
    #         result / np.abs(result) * ACCURACY,
    #         nan=ACCURACY
    #     )
    # result = np.where(np.abs(result) > ACCURACY, result, coeff)
    return result


# error functions #
def DREF(code, error, mag=0.35, propagation='random', NEV=True, **kwargs):
    dpde = np.full((len(error.survey_rad), 3), [1., 0., 0.])
    e_DIA = dpde * mag

    return error._generate_error(code, e_DIA, propagation, NEV)


def DSF(
    code, error, mag=0.00056, propagation='systematic', NEV=True, **kwargs
):
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
                        pi/2 - error.survey.inc_rad
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


def AMID(code, error, mag=0.04363323129985824, propagation='systematic',
    NEV=True, **kwargs
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
        *  cos(error.survey.azi_mag_rad)
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


# Gyro accelerometer-related weights (AXYZ_MIS, AXYZ_SF, AXYZ_ZB) and the
# continuous-mode initialisation helper (_get_ref_init_error) previously
# lived here as a half-finished SPE 90408 (Torkildsen et al. 2008)
# implementation flagged '# Needs QAQC'. Removed 2026-04-16 in favour of
# a JSON-schema + interpreter path (welleng/errors/interpreter.py) that
# consumes the ISCWSA error-models JSON schema directly.


def CNA(
    code, error, mag=0.35, propagation='systematic', NEV=True,
    **kwargs
):
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

    # result = error._generate_error(code, e_DIA, propagation, NEV)

    # return result


def CNI(
    code, error, mag=0.35, propagation='systematic', NEV=True,
    **kwargs
):
    dpde = np.full((len(error.survey_rad), 3), [0., 1., 0.])

    e_DIA = dpde * mag

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
        # Rev 5.11 XCLA workbook formula: ABS(SIN(inc) * SIN(ABS(wrapped_azi_delta))).
        azi_delta_wrapped = ((
            error.survey.azi_true_rad[1:]
            - error.survey.azi_true_rad[:-1]
            + pi
        ) % (2 * pi)) - pi
        temp = np.absolute(
            sin(error.survey.inc_rad[1:])
            * sin(np.absolute(azi_delta_wrapped))
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

    # Rev 5.11 XYM3E/XYM3L weights per workbook: ABS(COS(inc)) only, cos(azi)
    # keeps its sign so azimuths beyond 90° flip contribution correctly.
    dpde = np.zeros((len(error.survey_rad), 3))
    dpde[1:, 1] = (
        np.absolute(cos(error.survey.inc_rad[1:]))
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
        # Rev 5.11 "funny stuff": at SING station 1 the workbook uses the full
        # first interval, not the halved value (see ABXY precedent at line ~409).
        e_NEV_star_sing[1, 0] = (
            (error.survey.md[1] - error.survey.md[0])
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

    # Rev 5.11 XYM4E/XYM4L azi-axis weight per workbook:
    # ABS(COS(inc)) * COS(azi) / SIN(inc) * coeff.
    dpde = np.zeros((len(error.survey_rad), 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        dpde[:, 2] = np.nan_to_num(
            np.absolute(cos(error.survey.inc_rad))
            * cos(error.survey.azi_true_rad)
            / sin(error.survey.inc_rad)
            * coeff,
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
