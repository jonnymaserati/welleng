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
    "deg/sqr(hr)": np.pi / 180.0,  # gyro random-walk coeff: deg/sqrt(hr) -> rad/sqrt(hr)
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
    # NB: listdir order is filesystem-dependent, so SORT every walk -- several
    # tools share a `model_id` (e.g. GYRO-NS-CT.json and GYRO-NS-CT_Fl.json are
    # both 'A021Gc'), and an unsorted walk resolved to an arbitrary one (passed
    # locally, KeyError'd in CI). Sorting makes resolution deterministic and
    # picks the base name over the `_Fl` ("floating") variant ('.' < '_').
    # Cheap path: filename matches model name directly.
    for sub in sorted(os.listdir(json_root)):
        sub_p = os.path.join(json_root, sub)
        if os.path.isdir(sub_p):
            cand = os.path.join(sub_p, f"{model_name}.json")
            if os.path.isfile(cand):
                return cand
    # Fallback: walk every JSON, peek metadata for matching id / short_name.
    for sub in sorted(os.listdir(json_root)):
        sub_p = os.path.join(json_root, sub)
        if not os.path.isdir(sub_p):
            continue
        for fn in sorted(os.listdir(sub_p)):
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
    # Continuous / stationary gyro tool parameters. The ISCWSA Rev5 gyro
    # weight functions (GXY-GD/GRW running integral, GXY-RN noise, and the
    # static-gyro inclination gate) need per-tool calibration constants that
    # the OWSG->ISCWSA converter does not yet have a schema home for. Until
    # the schema carries them natively, surface whatever the `parameters`
    # block provides so the interpreter can bind them (see _call_interpreter).
    # NB: running speed in m/hr, inclination gates in radians.
    for src_key, hdr_key in (
        ("gxy_running_speed", "GXYRunningSpeed"),
        ("xy_static_gyro_end_inc", "XY Static Gyro End Inc"),
        ("noise_reduction_factor", "Noise Reduction Factor"),
    ):
        if src_key in pp:
            header[hdr_key] = pp[src_key]
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
        md = np.asarray(survey.md, dtype=float)
        inc = np.asarray(survey.inc_rad, dtype=float)
        azt = np.asarray(survey.azi_true_rad, dtype=float)
        azm = np.asarray(survey.azi_mag_rad, dtype=float)
        azg = np.asarray(survey.azi_grid_rad, dtype=float)

        # Previous-station ("...Prev") bindings: cross-station weight
        # functions (e.g. the continuous-gyro running integral) reference
        # the previous survey station. Index 0 has no predecessor, so it
        # is padded with its own value (the gate below zeroes station 0's
        # contribution anyway).
        def _prev(arr):
            return np.concatenate(([arr[0]], arr[:-1]))

        # Per-tool calibration constants surfaced by the JSON adapter
        # (see _json_to_em_adapter). EarthRate falls back to the standard
        # sidereal value (15.041 deg/hr) used by the gyro-azimuth terms.
        hdr = self.em.get("header", {}) if hasattr(self, "em") else {}
        earth_rate = float(getattr(survey.header, "earth_rate", None) or 0.262516)
        nrf = float(
            hdr.get("Noise Reduction Factor")
            if hdr.get("Noise Reduction Factor") is not None
            else getattr(survey.header, "noise_reduction_factor", 1.0)
        )
        bindings = {
            "MD": md, "Inc": inc, "AzT": azt, "AzM": azm, "Az": azg,
            "MDPrev": _prev(md), "IncPrev": _prev(inc),
            "AzTPrev": _prev(azt), "AzMPrev": _prev(azm), "AzPrev": _prev(azg),
            "TVD": np.asarray(survey.tvd, dtype=float),
            "Gfield": float(survey.header.G),
            "Dip": float(survey.header.dip),
            "BField": float(survey.header.b_total or 50000.0),
            "EarthRate": earth_rate,
            "Latitude": np.radians(float(survey.header.latitude or 0.0)),
            "NoiseReductionFactor": nrf,
            "RAD": np.pi / 180.0,
        }
        if hdr.get("GXYRunningSpeed") is not None:
            bindings["GXYRunningSpeed"] = float(hdr["GXYRunningSpeed"])

        # Recurrence terms (continuous gyro): the azimuth formula references
        # its own accumulated state from the previous station
        # (XY_Gyro_Drift = running drift integral; XY_Gyro_Random_Walk =
        # running random-walk). These cannot be evaluated by a single
        # vectorised call -- evaluate them station-by-station with the
        # static-gyro inclination gate, then return.
        recur = self._recurrence_var(term["azimuth_formula"])
        if recur is not None:
            return self._call_interpreter_recurrence(
                code, term, mag, propagation, bindings, recur, hdr
            )

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
        dpde = np.column_stack([d, i, a]).astype(float)
        # Cross-station terms (those referencing a "...Prev" variable, e.g.
        # the XYM3E/XYM4E station-spacing amplifier Max(1, sqrt(10/(MD-MDPrev)))
        # or the XCL tortuosity terms) have no predecessor at station 0. The
        # `_prev` padding fabricates MDPrev[0]=MD[0], so MD-MDPrev=0 there and
        # the formula evaluates to inf/nan. Zero the first station's
        # contribution, matching the deleted hand-coded weight functions which
        # forced station 0 to zero (np.append([0], ...)).
        if any(
            isinstance(term.get(f), str) and "Prev" in term[f]
            for f in ("depth_formula", "inclination_formula", "azimuth_formula")
        ):
            dpde[0] = 0.0
        # Per-term inclination gating + stationary->continuous carry.
        # A term may be active only within [inc_min_deg, inc_max_deg]
        # (e.g. a stationary-gyro term that runs below the static-gyro end
        # inclination). ``carry_above_max`` freezes the weight at the gate
        # (inc = inc_max) for stations above it -- the stationary azimuth
        # error at the mode switch is carried as a constant initialisation
        # error through the continuous zone (SPE 90408 "Initialization of
        # Continuous Surveys"). The gate weight is interpolated at exactly
        # inc_max (the paper notes covariance is underestimated otherwise).
        if term.get("inc_min_deg") is not None or term.get("inc_max_deg") is not None:
            dpde = self._apply_inc_gating(dpde, inc, term)
        # Convert the declared magnitude to welleng's SI base. OWSG xlsx
        # publishes magnitudes in the term's natural unit (deg, deg/hr,
        # m, ...); the propagation engine expects rad / m / m/s^2 /
        # rad/hr internally. The legacy YAML tools shipped pre-converted
        # values, so this conversion only fires for JSON-driven tools.
        unit = term.get("units") or term.get("unit") or "-"
        scale = _MAG_UNIT_TO_BASE.get(unit, 1.0)
        e_DIA = dpde * (mag * scale)
        # Vertical-hole singularity: OWSG/ISCWSA azimuth weight functions carry
        # a 1/sin(inc) factor that cancels the sin(inc) of the position
        # Jacobian in deviated hole but evaluates to nan at an exactly-vertical
        # station. When the JSON term supplies position-space singular weights
        # (north/east/vertical_singularity), substitute the ISCWSA Rev5.13
        # Sec 11.5 singular vectors at those stations.
        if any(
            isinstance(term.get(k), str)
            for k in ("north_singularity", "east_singularity",
                      "vertical_singularity")
        ):
            e_DIA = np.nan_to_num(e_DIA, nan=0.0, posinf=0.0, neginf=0.0)
            e_NEV, e_NEV_star = self._call_interpreter_singularity(
                e_DIA, term, bindings, mag * scale
            )
            return self.e._generate_error(
                code, e_DIA, propagation, NEV=True,
                e_NEV=e_NEV, e_NEV_star=e_NEV_star,
            )
        return self.e._generate_error(code, e_DIA, propagation, NEV=True)

    # Recurrence (self-referential) state variables a continuous-gyro
    # azimuth formula may carry, mapped to accumulation mode. The mode
    # selects how the coefficient `state` (h_i) accumulates inside the
    # formula; the magnitude enters linearly afterwards in both cases
    # (e_DIA = state * mag * scale -- see SPE 90408-PA Eqs 5c / 6c):
    #   - "drift"       : linear running integral, h_i = h_{i-1} + dD/c/sin
    #   - "random_walk" : quadrature running walk, h_i = sqrt(h_{i-1}^2 + dD/c/sin^2)
    # XYZ variants (SPE 90408 Table 6) accumulate with NO sin(I) factor and
    # Z variants (Table 8) with a 1/cos(I) factor -- the formula difference
    # lives entirely in the JSON azimuth_formula; only the var name +
    # accumulation mode are registered here.
    _RECURRENCE_VARS = {
        "XY_Gyro_Drift": "drift",
        "XY_Gyro_Random_Walk": "random_walk",
        "XYZ_Gyro_Drift": "drift",
        "XYZ_Gyro_Random_Walk": "random_walk",
        "Z_Gyro_Drift": "drift",
        "Z_Gyro_Random_Walk": "random_walk",
    }

    def _recurrence_var(self, formula):
        """Return (var_name, mode) if `formula` references a recurrence
        state variable, else None."""
        if not isinstance(formula, str):
            return None
        for var, mode in self._RECURRENCE_VARS.items():
            if var in formula:
                return var, mode
        return None

    def _call_interpreter_recurrence(
        self, code, term, mag, propagation, bindings, recur, hdr
    ):
        """Evaluate a continuous-gyro recurrence term station-by-station.

        Implements the continuous-survey gyro azimuth weight functions of
        SPE 90408-PA (Torkildsen et al. 2008, SPE Drilling & Completion,
        DOI 10.2118/90408-PA):

        - Coefficient recurrence per Table 7 (xy gyro) -- drift
          ``h_i = h_{i-1} + dD_i / (c*sin((I_{i-1}+I_i)/2))`` and random walk
          ``h_i = sqrt(h_{i-1}^2 + dD_i / (c*sin^2((I_{i-1}+I_i)/2)))``,
          where ``c`` is the running speed and ``dD = MD - MDPrev``. The
          analogous z-gyro / xyz coefficients are Tables 6 and 8.
        - Azimuth error is the magnitude times the coefficient, ``dA_i =
          v * h_i`` (Eqs 5a-5c drift, 6a-6c random walk) -- LINEAR in the
          magnitude for both modes; the sqrt of random walk is contained in
          ``h_i``, not applied to ``v``.

        Below the ``XY Static Gyro End Inc`` inclination gate the continuous
        survey is not running, so the contribution is zero and the
        accumulator resets. Using the stationary tool's end-inc as the
        continuous start-inc follows Table 11 note 3. Both modes propagate
        systematically within an initialisation section (Table 6/7 mode = S).
        """
        import warnings

        var, mode = recur
        formula = term["azimuth_formula"]
        n = len(bindings["MD"])
        inc = bindings["Inc"]
        inc_deg = np.degrees(inc)

        # A recurrence term is either:
        #  (a) zone-gated: ``inc_min_deg``/``inc_max_deg`` define the active
        #      inclination window (e.g. Model #4's z-continuous 0-17deg and
        #      xy-continuous 17-150deg). ``carry_above_max`` freezes the
        #      accumulated value above the window (SPE 90408 App. C Box 12,
        #      "non-actual mode preservation" -- a continuous source that has
        #      handed off to another mode preserves its last value as a
        #      systematic carry); below the window the term has not yet
        #      started (zero). The z- and xy-continuous are independent error
        #      SOURCES summed in the covariance -- no cross-seeding needed.
        #  (b) single-gate (default): accumulate above the
        #      ``XY Static Gyro End Inc`` gate, reset below. (Models #3/#5/#6
        #      and the from-station-0 case via a negative gate.)
        lo_deg = term.get("inc_min_deg")
        hi_deg = term.get("inc_max_deg")
        zoned = lo_deg is not None or hi_deg is not None
        carry = bool(term.get("carry_above_max"))
        lo = -np.inf if lo_deg is None else float(lo_deg)
        hi = np.inf if hi_deg is None else float(hi_deg)

        gate = hdr.get("XY Static Gyro End Inc")
        if "GXYRunningSpeed" not in bindings or (gate is None and not zoned):
            warnings.warn(
                f"continuous-gyro term {code!r} needs 'GXYRunningSpeed' (and, "
                f"unless inc-zoned, 'XY Static Gyro End Inc') tool parameters "
                f"(JSON parameters block); missing -> zero covariance.",
                RuntimeWarning,
            )
            zeros = np.zeros((n, 3))
            return self.e._generate_error(code, zeros, propagation, NEV=True)
        gate = float(gate) if gate is not None else -np.inf

        # Per-station scalar bindings reused each step (running speed,
        # earth rate, NRF, ... already live in `bindings`).
        base = {k: v for k, v in bindings.items() if np.isscalar(v)}

        state = 0.0
        states = np.zeros(n)
        for k in range(1, n):
            if zoned:
                active = lo <= inc_deg[k] <= hi
                frozen = carry and inc_deg[k] > hi
            else:
                active = inc[k] > gate
                frozen = False
            if active:
                step_ns = dict(base)
                step_ns.update({
                    "MD": float(bindings["MD"][k]),
                    "MDPrev": float(bindings["MDPrev"][k]),
                    "Inc": float(inc[k]),
                    "IncPrev": float(bindings["IncPrev"][k]),
                    "AzT": float(bindings["AzT"][k]),
                    "AzTPrev": float(bindings["AzTPrev"][k]),
                    var: state,
                })
                state = float(evaluate_formula(formula, step_ns))
            elif frozen:
                pass  # above the window: preserve the handed-off value
            else:
                state = 0.0  # outside the active survey: not running / reset
            states[k] = state

        unit = term.get("units") or term.get("unit") or "-"
        scale = _MAG_UNIT_TO_BASE.get(unit, 1.0)
        # The magnitude multiplies the accumulated coefficient `state` (h_i)
        # LINEARLY for both drift and random walk. Per SPE 90408-PA
        # (Torkildsen et al. 2008) Eqs 5a-5c / 6a-6c, the azimuth error is
        # dA_i = v * h_i: the quadrature (sqrt) accumulation of random walk
        # lives entirely inside h_i (the `Sqr(state^2 + ...)` recurrence,
        # Table 7), so v_rw enters linearly exactly as the drift rate v_d
        # does. Units: drift v_d [deg/hr] * h_i [hr] and walk v_rw
        # [deg/sqrt(hr)] * h_i [sqrt(hr)] both give deg, then `scale` -> rad.
        outer = mag * scale

        e_DIA = np.zeros((n, 3))
        e_DIA[:, 2] = states * outer
        return self.e._generate_error(code, e_DIA, propagation, NEV=True)

    def _dpde_at_inc(self, dpde, inc_deg, gate):
        """Interpolate the per-axis weight (dpde row) at inc == gate, at the
        first station where inclination crosses the gate. SPE 90408 App. C
        Box 9 (act_init_inc): the initialisation weight is taken at the mode-
        switch inclination, interpolated since a station rarely lands on it."""
        above = np.where(inc_deg > gate)[0]
        if above.size == 0:
            return np.zeros(dpde.shape[1])
        j = int(above[0])
        if j == 0:
            return dpde[0].copy()
        lo_i, hi_i = inc_deg[j - 1], inc_deg[j]
        if hi_i == lo_i:
            return dpde[j].copy()
        f = (gate - lo_i) / (hi_i - lo_i)
        return dpde[j - 1] + f * (dpde[j] - dpde[j - 1])

    def _carry_per_section(self, dpde, inc_deg, gate, below):
        """Per-continuous-section stationary-init carry (SPE 90408 App. C
        Boxes 2/3/9).

        A gyro tool that surveys stationary below ``gate`` (= the stationary
        ``init_inc``/``end_inc``) and continuous above it RE-INITIALISES the
        continuous survey every time the well bore drops back below the gate
        and rebuilds (Box 3 sets ``initialised = FALSE`` whenever inc < gate;
        Box 9 then recomputes ``d_A_init(j) = dA(j, act_init_inc, A(i))`` at
        the *new* initialisation point on the next entry into continuous
        mode). The carried weight is therefore the gate-crossing value of the
        MOST RECENT up-crossing -- using the azimuth at that crossing -- not a
        single global value frozen at the first crossing.

        Returns ``out`` with, for each station above ``gate``, the weight
        interpolated at ``inc == gate`` on its own section's up-crossing.
        Below the gate the station keeps its live stationary weight
        (``below='live'``, ``carry_above_max``) or zero (``below='zero'``,
        ``carry_only`` init seed).
        """
        out = dpde.copy() if below == "live" else np.zeros_like(dpde)
        carried = None
        for k in range(len(inc_deg)):
            if inc_deg[k] > gate:
                if carried is None:               # first station of a section
                    if k == 0:
                        carried = dpde[0].copy()
                    else:
                        lo_i, hi_i = inc_deg[k - 1], inc_deg[k]
                        f = 0.0 if hi_i == lo_i else (gate - lo_i) / (hi_i - lo_i)
                        carried = dpde[k - 1] + f * (dpde[k] - dpde[k - 1])
                out[k] = carried
            else:
                carried = None                    # below gate: re-init pending
        return out

    def _apply_inc_gating(self, dpde, inc, term):
        """Inclination-window gating + stationary->continuous carry.

        - ``inc_min_deg``/``inc_max_deg``: term active only in that window;
          outside contributes zero.
        - ``carry_above_max``: above inc_max, freeze the weight at the gate
          value (SPE 90408 App. C Box 12: stationary terms d_A(j,i)=d_A(j,i-1)
          above the mode switch -> constant initialisation error carried
          through the continuous zone).
        - ``carry_only``: zero below inc_min, frozen at the gate value above
          it (used for the systematic initialisation seed of a stationary
          *random* term, which is carried systematic -- App. C Box 9).

        When the tool has a positive stationary ``init_inc`` (``XY Static Gyro
        End Inc`` >= 0) the carry RE-INITIALISES per continuous section
        (``_carry_per_section``): a drop back below the gate followed by a
        rebuild re-gyrocompasses, so the carried weight above the gate uses
        the azimuth of the most-recent crossing, not the first. For a tool
        that initialises at the first survey station (negative ``init_inc``,
        e.g. an XYZ system, Table 11 note 2) there is no stationary mode to
        drop into, so the original single global-first carry is kept.
        """
        inc_deg = np.degrees(inc)
        lo = term.get("inc_min_deg")
        hi = term.get("inc_max_deg")
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        hdr = self.em.get("header", {}) if hasattr(self, "em") else {}
        init_inc = hdr.get("XY Static Gyro End Inc")
        per_section = init_inc is not None and float(init_inc) >= 0.0
        out = dpde.copy()
        if term.get("carry_only"):
            gate = lo
            if per_section:
                return self._carry_per_section(dpde, inc_deg, gate, below="zero")
            above = inc_deg > gate
            gate_dpde = self._dpde_at_inc(dpde, inc_deg, gate)
            out[:] = 0.0
            out[above] = gate_dpde
            return out
        if term.get("carry_above_max") and np.isfinite(hi):
            if per_section:
                out = self._carry_per_section(dpde, inc_deg, hi, below="live")
                out[inc_deg < lo] = 0.0
                return out
            above = inc_deg > hi
            gate_dpde = self._dpde_at_inc(dpde, inc_deg, hi)
            out[above] = gate_dpde
            out[inc_deg < lo] = 0.0
            return out
        active = (inc_deg >= lo) & (inc_deg <= hi)
        out[~active] = 0.0
        return out

    def _call_interpreter_singularity(self, e_DIA, term, bindings, sigma):
        """Build e_NEV / e_NEV_star with vertical-hole singularity substitution.

        At a station with ``inc < vertical_inc_limit`` the term's azimuth
        weight function is undefined (1/sin(inc) -> nan), so the position-error
        contribution is taken directly from the JSON ``*_singularity`` weights
        (``VertWftFn = [north, east, vertical]``) using the ISCWSA Rev5.13
        Sec 11.5 singular vectors (mirrors the hand-coded ABXY-TI2/XYM3E SING
        branch):

            interior k : e_k  = sigma * (D_{k+1}-D_{k-1})/2 * VertWftFn  (eq 20)
                         e*_k = sigma * (D_k    -D_{k-1})/2 * VertWftFn  (eq 21)
            station 1  : e_1  = sigma * (D_2+D_1-2 D_0)/2   * VertWftFn  (eq 33)
                         e*_1 = sigma * (D_1-D_0)           * VertWftFn  (eq 34)

        Station 0 is the tie-on (always zero). The eq 34 full-interval star
        (vs the halved interior eq 21) is the Rev5 behaviour; Rev4 keeps it
        halved, matching the ``drk_dInc`` first-station gate.
        """
        survey = self.e.survey
        n = len(survey.md)
        e_NEV = self.e._e_NEV(e_DIA)
        e_NEV_star = self.e._e_NEV_star(e_DIA)

        sing = np.where(
            np.asarray(survey.inc_rad) < survey.header.vertical_inc_limit
        )[0]
        if sing.size == 0:
            return e_NEV, e_NEV_star

        def _vw(key):
            f = term.get(key)
            if not isinstance(f, str):
                return np.zeros(n)
            try:
                val = evaluate_formula(f, bindings)
            except Exception:
                # Singular weight references a variable the interpreter can't
                # bind (e.g. XCLTortuosity) -- a documented schema gap; the
                # component contributes zero, same as the main eval path.
                return np.zeros(n)
            return np.broadcast_to(
                np.asarray(val, dtype=float), (n,)
            ).astype(float)

        # Position-space weight, already carrying the source magnitude `sigma`.
        vw = np.column_stack([
            _vw("north_singularity"),
            _vw("east_singularity"),
            _vw("vertical_singularity"),
        ]) * sigma

        md = np.asarray(survey.md, dtype=float)
        double_delta = np.zeros(n)   # D_{k+1} - D_{k-1}
        delta = np.zeros(n)          # D_k     - D_{k-1}
        double_delta[1:-1] = md[2:] - md[:-2]
        delta[1:-1] = md[1:-1] - md[:-2]

        e_sing = 0.5 * double_delta[:, None] * vw      # eq 20
        estar_sing = 0.5 * delta[:, None] * vw         # eq 21

        # First surveyed station (index 1; index 0 is the tie-on -> zero).
        if n >= 3 and 1 in sing:
            e_sing[1] = ((md[2] + md[1] - 2 * md[0]) / 2) * vw[1]   # eq 33
            if self.e.error_model.lower().split()[-1] == 'rev4':
                estar_sing[1] = ((md[1] - md[0]) / 2) * vw[1]
            else:
                estar_sing[1] = (md[1] - md[0]) * vw[1]            # eq 34

        # A singular weight may be inf where MD==MDPrev (station 0 padding);
        # 0*inf -> nan there. Sanitise; station 0 is the tie-on (zero anyway).
        e_sing = np.nan_to_num(e_sing, nan=0.0, posinf=0.0, neginf=0.0)
        estar_sing = np.nan_to_num(estar_sing, nan=0.0, posinf=0.0, neginf=0.0)

        e_NEV[sing] = e_sing[sing]
        e_NEV_star[sing] = estar_sing[sing]
        return e_NEV, e_NEV_star

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
        # Rev 5.13 eq (21): e* carries VertWftFn, which includes the damping M
        # (coeff). coeff (len N-1) aligns elementwise with md[1:]-md[:-1].
        e_NEV_star_sing[1:, 0] = (
            coeff
            * (
                error.survey.md[1:]
                - error.survey.md[:-1]
            ) / 2
            * mag
        )
        # Rev 5.11 "funny stuff": at SING station 1 the workbook uses the full
        # first interval, not the halved value (see ABXY precedent at line ~409).
        # Rev 5.13 eq (34): M (coeff[0]) applies here too.
        e_NEV_star_sing[1, 0] = (
            coeff[0]
            * (error.survey.md[1] - error.survey.md[0])
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
        # Rev 5.13 eq (21): e* carries VertWftFn including damping M (coeff).
        # XYM4L coeff is len N (coeff[0]=1); coeff[1:] aligns with md[1:]-md[:-1].
        e_NEV_star_sing[1:, 1] = (
            coeff[1:]
            * (
                error.survey.md[1:]
                - error.survey.md[:-1]
            ) / 2
            * mag
        )
        # Rev 5.13 eq (34): full first interval, with M (coeff[1]).
        e_NEV_star_sing[1, 1] = (
            coeff[1]
            * (
                error.survey.md[1]
                - error.survey.md[0]
            )
            * mag
        )

        e_NEV_star[sing] = e_NEV_star_sing[sing]

        return error._generate_error(
            code, e_DIA, propagation, NEV, e_NEV, e_NEV_star
        )
