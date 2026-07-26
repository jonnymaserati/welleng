"""ISCWSA error models for computing wellbore positional uncertainty.

The ``"ISCWSA MWD Rev5"`` string remains a selectable error model, but is now
a deprecated alias for the Rev 5.11 compliant implementation (`"ISCWSA MWD
Rev5.11"`). As of welleng 0.10.0 the Rev5 YAML and weight functions were
corrected against the ISCWSA Rev 5.11 example workbooks, so users who
previously selected ``"ISCWSA MWD Rev5"`` will get slightly different (and
correct) covariance output. ``"ISCWSA MWD Rev4"`` is unchanged.
"""

import os
import re
import warnings

import numpy as np
import yaml
from .errors.tool_errors import ToolError
from .utils import cov_from_vec


# Mapping of deprecated error-model strings → the canonical replacement.
# Passing the deprecated name still works but emits a ``DeprecationWarning``.
_DEPRECATED_ERROR_MODEL_ALIASES = {
    "ISCWSA MWD Rev5": "ISCWSA MWD Rev5.11",
}

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
    """Load the tool error model index from the bundled YAML file.

    Returns
    -------
    dict
        Mapping of tool model names to their configuration parameters.
    """
    with open(TOOL_INDEX_FILENAME, 'r') as f:
        tool_index = yaml.safe_load(f)
    return tool_index


def get_error_models(tool_index=None):
    """Return a list of available error model short names.

    Parameters
    ----------
    tool_index : dict, optional
        Pre-loaded tool index dict. If None, loads from disk.

    Returns
    -------
    list of str
        Short names of all registered error models.
    """
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

    Attributes
    ----------
    error_model : str
        Name of the error model used (e.g. ``'ISCWSA MWD Rev5.11'``, the
        current default; ``'ISCWSA MWD Rev4'`` for legacy Rev 4 behaviour).
    survey : welleng.survey.Survey
        The input Survey object.
    errors : welleng.errors.tool_errors.ToolError
        ToolError object containing per-source error magnitudes and
        covariance data.
    survey_rad : numpy.ndarray
        Array of (md, inc_rad, azi_true_rad) per station, shape (n, 3).
    drdp : numpy.ndarray
        Jacobian of position with respect to survey parameters (depth,
        inclination, azimuth) in NEV coordinates.
    cov_NEVs : numpy.ndarray
        Summed covariance matrices in NEV coordinates per station, shape
        (n, 3, 3). Accessible via ``errors.cov_NEVs``.
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
            """Initialize an Error with computed error vectors and covariances.

            Parameters
            ----------
            code : str
                The error source code identifier.
            propagation : str
                Propagation type ('systematic', 'random', 'global',
                or 'within_pad').
            e_DIA : numpy.ndarray
                Error vectors in Depth-Inclination-Azimuth coordinates.
            cov_DIA : numpy.ndarray
                Covariance matrices in DIA coordinates.
            e_NEV : numpy.ndarray
                Error vectors in North-East-Vertical coordinates.
            e_NEV_star : numpy.ndarray
                Single-station NEV error vectors.
            sigma_e_NEV : numpy.ndarray
                Cumulative NEV error vectors.
            cov_NEV : numpy.ndarray
                Covariance matrices in NEV coordinates.
            """
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
        survey,
        error_model="ISCWSA MWD Rev5.11",
    ):
        """Initialize the error model for a given survey.

        Parameters
        ----------
        survey : welleng.survey.Survey
            The survey to compute errors for.
        error_model : str or dict, optional
            Name of the error model to apply. Defaults to the Rev 5.11
            compliant ``"ISCWSA MWD Rev5.11"``. The legacy name
            ``"ISCWSA MWD Rev5"`` is accepted as a deprecated alias.
            Alternatively a prebuilt ISCWSA-JSON-shaped model dict — e.g.
            a COMPASS IPM imported from an EDM export by
            ``welleng.errors.edm_ipm`` — evaluated by the formula
            interpreter without any file resolution.
        """

        if isinstance(error_model, dict):
            self.error_model = error_model.get(
                'metadata', {}
            ).get('short_name', 'custom-dict-model')
            self.survey = survey
            self.survey_rad = np.stack((
                self.survey.md,
                self.survey.inc_rad,
                self.survey.azi_true_rad
            ), axis=-1)
            self.survey_drdp = self.survey_rad
            self.drdp = self._drdp(self.survey_drdp)
            self.drdp_sing = self._drdp_sing(self.survey_drdp)
            self._validate_mag_reference(error_model)
            self.errors = ToolError(error=self, model=error_model)
            return

        if error_model in _DEPRECATED_ERROR_MODEL_ALIASES:
            replacement = _DEPRECATED_ERROR_MODEL_ALIASES[error_model]
            warnings.warn(
                f"error_model={error_model!r} is deprecated; use "
                f"{replacement!r} instead. From welleng 0.10.0 onward the "
                "Rev5 model is Rev 5.11 compliant, so this alias now "
                "produces Rev 5.11 results (different to pre-0.10.0 output).",
                DeprecationWarning,
                stacklevel=2,
            )
            error_model = replacement

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

        self._validate_mag_reference(model)

        try:
            self.errors = ToolError(
                error=self,
                model=model
            )
        except FileNotFoundError:
            # JSON-only floating-rig variants share their base model's OWSG
            # prefix (e.g. GYRO-MWD_Fl.json carries model_id 'A019Gb'), so
            # the index key 'A019Gb_Fl' resolves no file. The JSON walk
            # matches metadata.short_name, so retry with the Short Name.
            self.errors = ToolError(
                error=self,
                model=self.error_model
            )

    def _validate_mag_reference(self, model) -> None:
        """Magnetic models need a real geomagnetic reference — refuse package
        defaults.

        Magnetic-compass models' weighting functions consume the header's
        ``b_total``/``dip``/``declination``; gyro models (OWSG ``A<nn>G*``,
        the GYRO short names, the SPE-90408 example model, and dict models
        whose ``metadata.tool_type`` says gyro / inclination-only) do not.
        Those values must be either user-provided or looked up for a
        user-provided location — the header's fallback values (or a lookup at
        the fallback location) are placeholders, not a geomagnetic reference,
        and would silently bias every magnetic term. Unclassified models are
        treated as magnetic (the strict direction). See
        ``SurveyHeader.mag_source``.
        """
        if isinstance(model, dict):
            tool_type = str(
                model.get('metadata', {}).get('tool_type', '')
            ).lower()
            is_gyro = tool_type in ('gyro', 'inclination_only')
        else:
            is_gyro = (
                re.match(r"^A\d+G", model) is not None
                or 'GYRO' in self.error_model.upper()
                or '90408' in model
            )
        if is_gyro:
            return
        sources = getattr(self.survey.header, 'mag_source', None)
        if sources is None:     # header predates provenance tracking
            return
        bad = sorted(f for f, s in sources.items() if s == 'default')
        if bad:
            raise ValueError(
                f"error_model {self.error_model!r} is a magnetic model but "
                f"the survey header's {', '.join(bad)} came from package "
                "defaults. Provide b_total, dip and declination on the "
                "SurveyHeader, or provide the well's latitude/longitude "
                "(+ survey_date) so they can be looked up from the BGS "
                "geomagnetic web service."
            )

    def _e_NEV(self, e_DIA):
        D, I, A = e_DIA.T
        arr = np.column_stack([
            (self.drdp[:, 0] + self.drdp[:, 9]) * D
            + (self.drdp[:, 3] + self.drdp[:, 12]) * I
            + (self.drdp[:, 6] + self.drdp[:, 15]) * A,

            (self.drdp[:, 1] + self.drdp[:, 10]) * D
            + (self.drdp[:, 4] + self.drdp[:, 13]) * I
            + (self.drdp[:, 7] + self.drdp[:, 16]) * A,

            (self.drdp[:, 2] + self.drdp[:, 11]) * D
            + (self.drdp[:, 5] + self.drdp[:, 14]) * I
            + (self.drdp[:, 8] + self.drdp[:, 17]) * A,
        ])

        arr[0] = 0

        return arr

    def _e_NEV_star(self, e_DIA):
        D, I, A = e_DIA.T
        arr = np.column_stack([
            self.drdp[:, 0] * D
            + self.drdp[:, 3] * I
            + self.drdp[:, 6] * A,

            self.drdp[:, 1] * D
            + self.drdp[:, 4] * I
            + self.drdp[:, 7] * A,

            self.drdp[:, 2] * D
            + self.drdp[:, 5] * I
            + self.drdp[:, 8] * A
        ])

        # NB: station 0 is NOT blanket-zeroed here (unlike ``_e_NEV``). ``drdp``
        # seeds the station-0 depth column with the wellbore tangent (see
        # ``_drdp``), so a random depth source's own measurement error carries
        # its full variance at the surface (ISCWSA DRFR cov_VV(0) = mag^2).
        # inc/azi columns are zero at station 0 -> every non-depth term (and any
        # systematic depth term, whose weight vanishes at md=0) stays 0 there,
        # exactly matching the ISCWSA diagnostics.

        return arr

    def _sigma_e_NEV_systematic(self, e_NEV, e_NEV_star):
        return e_NEV_star + np.vstack(
            (
                e_NEV[0],
                np.cumsum(e_NEV, axis=0)[:-1]
            )
        )

    @staticmethod
    def _partial_star_drk(inc_i, azi_i, inc_q, azi_q, delta_md):
        """Own-station (``*``) position-weighting ``drk`` for the PARTIAL
        minimum-curvature leg from survey station ``i`` to an interior point
        ``q`` (``q`` playing the role of the leg's second station). Same form as
        ``drk_dDepth``/``drk_dInc``/``drk_dAz`` rows>=1, evaluated with the
        interior angles and the partial ``delta_md = md_q - md_i``. Returns the
        three (3,) NEV weight vectors ``(d_depth, d_inc, d_az)``.
        """
        vec_i = np.array([
            np.sin(inc_i) * np.cos(azi_i),
            np.sin(inc_i) * np.sin(azi_i),
            np.cos(inc_i),
        ])
        vec_q = np.array([
            np.sin(inc_q) * np.cos(azi_q),
            np.sin(inc_q) * np.sin(azi_q),
            np.cos(inc_q),
        ])
        d_depth = 0.5 * (vec_i + vec_q)
        d_inc = 0.5 * delta_md * np.array([
            np.cos(inc_q) * np.cos(azi_q),
            np.cos(inc_q) * np.sin(azi_q),
            -np.sin(inc_q),
        ])
        d_az = 0.5 * delta_md * np.array([
            -np.sin(inc_q) * np.sin(azi_q),
            np.sin(inc_q) * np.cos(azi_q),
            0.0,
        ])
        return d_depth, d_inc, d_az

    @staticmethod
    def _partial_plus1_drk(inc_i, azi_i, d_depth, delta_md):
        """Coupling (``drkplus1``) weights of the NEAR station ``i`` for the
        partial leg ``i -> q`` -- station i's measurement error acting through
        the partial out-leg (``drkplus1_dDepth/dInc/dAz`` rows, near-station
        angles + partial ``delta_md``). ``d_depth`` is ``_partial_star_drk``'s
        depth weight (``drkplus1_dDepth == -drk_dDepth``)."""
        p_depth = -d_depth
        p_inc = 0.5 * delta_md * np.array([
            np.cos(inc_i) * np.cos(azi_i),
            np.cos(inc_i) * np.sin(azi_i),
            -np.sin(inc_i),
        ])
        p_az = 0.5 * delta_md * np.array([
            -np.sin(inc_i) * np.sin(azi_i),
            np.sin(inc_i) * np.cos(azi_i),
            0.0,
        ])
        return p_depth, p_inc, p_az

    # course-length recurrence terms handled by the partial-course-length
    # interior convention (Codling SPE-187249; ISCWSA-validated weight funcs in
    # ``welleng.errors.tool_errors``). One-oracle with welleng-assay's symbolic
    # Propagator (2026-07-24 ruling).
    _COURSE_LENGTH_TERMS = ("XCLA", "XCLH")

    def _interior_prep(self):
        """Classify each error source for interior evaluation + cache the data
        the classes need. Cached once per model. Returns
        ``(classes, xcl_mag, tortuosity, vertical_inc_limit)`` where ``classes``
        maps each source name to one of:

        - ``"standard"`` -- the own-station ``drk * e_DIA`` reconstruction
          reproduces the stored ``e_NEV_star`` at EVERY station: exact analytical
          option-c interior (the ~29 ISCWSA weighting-function terms).
        - ``"course_length"`` -- XCLA/XCLH, the cross-station course-length
          recurrence terms (Max(Δangle, tortuosity·course-length)). Interior uses
          the partial-course-length convention (:meth:`_xcl_partial_enev`): NOT
          MC-validated (course length has no independent MC ground truth at a
          fractional point); a STATED convention, station-exact at f=0,1,
          continuous between, one-oracle with assay. ``xcl_mag`` holds each such
          term's magnitude, reconstructed from its stored ``e_NEV`` (model-general
          -- reads the sheet value, not the function default).
        - ``"linear"`` -- any other ring-fenced term (e.g. ABXY-TI*S, XYM*E) the
          ``drk * e_DIA`` form cannot reproduce and that is not a course-length
          term: falls back to linear covariance interpolation. Self-protecting
          (surface-tie-on first-leg terms land here too), safe.
        """
        cached = getattr(self, "_interior_prep_cache", None)
        if cached is not None:
            return cached
        md, inc, azi = self.survey_rad.T
        azt = self.survey.azi_true_rad
        tort = float(getattr(self.errors, "tortuosity", 0.0) or 0.0)
        vlim = float(getattr(self.survey.header, "vertical_inc_limit", 0.0))
        dmd = md[1:] - md[:-1]
        classes, xcl_mag = {}, {}
        for name, src in self.errors.errors.items():
            if name in self._COURSE_LENGTH_TERMS:
                classes[name] = "course_length"
                xcl_mag[name] = self._reconstruct_xcl_mag(
                    name, src, md, inc, azt, dmd, tort, vlim
                )
                continue
            ok = True
            for k in range(1, len(md)):
                ddk, dik, dak = self._partial_star_drk(
                    inc[k - 1], azi[k - 1], inc[k], azi[k], md[k] - md[k - 1]
                )
                D, I, A = src.e_DIA[k]
                recon = ddk * D + dik * I + dak * A
                ref = src.e_NEV_star[k]
                scale = max(1e-9, float(np.max(np.abs(ref))))
                if np.max(np.abs(recon - ref)) > 1e-6 * scale:
                    ok = False
                    break
            classes[name] = "standard" if ok else "linear"
        cached = (classes, xcl_mag, tort, vlim)
        self._interior_prep_cache = cached
        return cached

    def _interior_angles(self, i, f):
        """Interior (inc, azi) at arc-fraction ``f`` on leg ``[i, i+1]``, radians.

        The same minimum-curvature slerp the position interpolation uses,
        ``sin((1-f)a)/sin(a) t_i + sin(f a)/sin(a) t_{i+1}``.
        """
        smd, sinc, sazi = self.survey_rad.T
        dogleg = float(self.survey.dogleg[i + 1])
        if dogleg < 1e-9:
            return sinc[i], sazi[i]
        vec_i = np.array([
            np.sin(sinc[i]) * np.cos(sazi[i]),
            np.sin(sinc[i]) * np.sin(sazi[i]),
            np.cos(sinc[i]),
        ])
        vec_j = np.array([
            np.sin(sinc[i + 1]) * np.cos(sazi[i + 1]),
            np.sin(sinc[i + 1]) * np.sin(sazi[i + 1]),
            np.cos(sinc[i + 1]),
        ])
        theta = dogleg * f
        u = (vec_j - np.cos(dogleg) * vec_i) / np.sin(dogleg)
        vec_q = np.cos(theta) * vec_i + np.sin(theta) * u
        vec_q = vec_q / np.linalg.norm(vec_q)
        return (
            float(np.arccos(np.clip(vec_q[2], -1.0, 1.0))),
            float(np.arctan2(vec_q[1], vec_q[0])) % (2 * np.pi),
        )

    def _interior_stacks(self):
        """Per-class source arrays, stacked once per model, for :meth:`cov_nev_at`.

        A single interior evaluation touches every error source -- 35 on the
        default MWD model -- and looping them in Python costs more than the
        arithmetic does: at 3-vector sizes numpy's per-call overhead dominates,
        and the loop issued ~37 separate ``np.outer`` products per query. These
        stacks let one query be evaluated with a handful of array ops over the
        SOURCE axis instead.

        This vectorises over SOURCES within a single scalar query. It does NOT
        make the public entry point batched -- ``cov_nev_at`` takes one measured
        depth and returns one (3, 3) -- and there is deliberately no per-leg or
        per-query cache here: everything below is model-invariant, built once and
        indexed. The batched/vectorised form of the interior covariance is
        welleng-api's and welleng-assay's, not open core's.

        Returns a dict with, for the ``"standard"`` sources, ``e_DIA`` (S, n, 3),
        ``e_NEV_star`` (S, n, 3), ``sigma_e_NEV`` (S, n, 3) and the boolean
        ``random`` mask (S,), plus ``cov_random`` (n, 3, 3), the per-station sum
        of the random sources' ``cov_NEV - outer(e_NEV_star, e_NEV_star)``, which
        is the query-independent part of their contribution.
        """
        cached = getattr(self, "_interior_stacks_cache", None)
        if cached is not None:
            return cached
        classes, _, _, _ = self._interior_prep()
        std = [s for n, s in self.errors.errors.items()
               if classes[n] == "standard"]
        lin = [s for n, s in self.errors.errors.items()
               if classes[n] == "linear"]
        n = len(self.survey_rad)
        z3 = np.zeros((0, n, 3))
        e_dia = np.array([s.e_DIA for s in std]) if std else z3
        e_star = np.array([s.e_NEV_star for s in std]) if std else z3
        rand = np.array([s.propagation == 'random' for s in std], dtype=bool)
        # sigma_e_NEV is the correlated running sum, so it is only read on the
        # NON-random branch. A random source may carry a per-section (n, 3, 3)
        # form instead of (n, 3) (the carried-init terms -- DBHR / DECR / DRFR),
        # which will not stack; those rows are zero-filled and never indexed.
        sig = np.zeros((len(std), n, 3))
        for k, s in enumerate(std):
            if not rand[k]:
                sig[k] = s.sigma_e_NEV
        cov_random = np.zeros((n, 3, 3))
        for s, is_rand in zip(std, rand):
            if is_rand:
                cov_random += s.cov_NEV - np.einsum(
                    'ki,kj->kij', s.e_NEV_star, s.e_NEV_star
                )
        cov_linear = (np.sum([s.cov_NEV for s in lin], axis=0)
                      if lin else np.zeros((n, 3, 3)))
        cached = dict(
            e_DIA=e_dia, e_NEV_star=e_star, sigma_e_NEV=sig, random=rand,
            cov_random=cov_random, cov_linear=cov_linear,
        )
        self._interior_stacks_cache = cached
        return cached

    @staticmethod
    def _reconstruct_xcl_mag(name, src, md, inc, azt, dmd, tort, vlim):
        """Recover an XCLA/XCLH term's magnitude from its stored ``e_NEV`` (the
        direction unit-norms to 1, so ``mag = ||e_NEV[k]|| / (Δmd_k * w_k)`` is
        constant across stations). Model-general -- reads the model's actual
        magnitude, not the ``tool_errors`` function default."""
        if name == "XCLA":
            azw = ((azt[1:] - azt[:-1] + np.pi) % (2 * np.pi)) - np.pi
            geom = np.abs(np.sin(inc[1:]) * np.sin(np.abs(azw)))
            geom[inc[:-1] < vlim] = 0.0
        else:  # XCLH
            geom = np.abs(inc[1:] - inc[:-1])
        w = np.maximum(geom, tort * dmd)
        denom = dmd * w
        good = denom > 0
        if not np.any(good):
            return 0.0
        return float(np.nanmedian(
            np.linalg.norm(src.e_NEV[1:], axis=1)[good] / denom[good]
        ))

    def _xcl_partial_enev(self, name, mag, tort, vlim, inc_i, azt_i,
                          inc_q, azt_q, Lq):
        """XCLA/XCLH e_NEV over the partial course length ``Lq`` = md_q - md_i,
        with min-curve interior angles ``inc_q, azt_q`` -- the partial-interval
        evaluation of the ISCWSA-validated weight (``tool_errors.XCLA``/``XCLH``).
        NEV-direct (``e_NEV_star == e_NEV``)."""
        if name == "XCLA":
            aw = ((azt_q - azt_i + np.pi) % (2 * np.pi)) - np.pi
            s_q = 0.0 if inc_i < vlim else abs(
                np.sin(inc_q) * np.sin(abs(aw))
            )
            w = max(s_q, tort * Lq)
            return mag * Lq * w * np.array([-np.sin(azt_q), np.cos(azt_q), 0.0])
        w = max(abs(inc_q - inc_i), tort * Lq)
        return mag * Lq * w * np.array([
            np.cos(inc_q) * np.cos(azt_q),
            np.cos(inc_q) * np.sin(azt_q),
            -np.sin(inc_q),
        ])

    def cov_nev_at(self, md):
        """Arc-faithful ISCWSA covariance at an interior measured depth ``md``.

        Evaluates the (3, 3) NEV covariance directly ON the minimum-curvature
        arc at ``md``, by propagating each error source's stored station values
        through the partial-leg interpolation Jacobian -- NOT by interpolating
        the assembled covariance matrix linearly between stations (which
        under-reports the separation factor by up to ~25% near doglegs) and NOT
        by inserting ``md`` as a real survey station (which would add a spurious
        extra measurement and perturb the propagation). The interpolated point
        is not a new measurement: it inherits the bounding stations' sources.

        For an interior point ``q`` at arc-fraction ``f`` on leg ``[i, i+1]``,
        station ``i`` and station ``i+1`` both drive the partial leg (via the
        min-curve slerp), so the interior propagates BOTH: the own weight
        ``drk(i->q)`` (far station) AND station i's out-leg coupling
        ``drkplus1(i->q)`` (near station). This is exact at BOTH ends -- the
        own-only form (drk alone) is exact at f->1 but drops the coupling and
        biases f->0 by ~1 leg. With ``qi = drk(i->q).e_DIA[i]``,
        ``qj = drk(i->q).e_DIA[i+1]``, ``coup = drkplus1(i->q).e_DIA[i]``:

        - systematic/global/well/within_pad (correlated vector sum):
          ``sigma(q) = (1-f) qi + f qj + sigma_e_NEV[i] + coup``,
          contributing ``outer(sigma(q))``.
        - random (two INDEPENDENT measurements -> two outer products):
          ``cov_NEV[i] - outer(e_NEV_star[i]) + outer(g_i) + outer(g_j)`` with
          ``g_i = e_NEV_star[i] + coup + (1-f) qi`` and ``g_j = f qj`` -- the
          partial q-own term splits (1-f)/f across the two stations (slerp-
          Jacobian ~ f; exact at both ends, ~slerp tolerance interior -- assay's
          symbolic Propagator is the exact oracle).

        XCLA/XCLH (the course-length recurrence terms, typically dominant on
        deviated wells) use the partial-course-length convention
        (``cov_NEV[i] + outer(e_NEV(i->q))``, :meth:`_xcl_partial_enev`) -- a
        STATED convention (not MC-validated: course length has no independent MC
        ground truth at a fractional point), station-exact at f=0,1, one-oracle
        with welleng-assay. Any remaining ring-fenced term (:meth:`_interior_prep`
        class ``"linear"``) uses linear covariance interpolation. Reproduces the
        stored ``cov_NEV[i+1]`` at ``f -> 1`` to machine precision. See
        ``docs/dev/CLEARANCE_ANALYTICAL_COV.md``.

        INTERIOR ACCURACY IS GEOMETRY-DEPENDENT — do not read "exact at both
        ends" as "accurate throughout". Both this boundary-anchored form and
        welleng-assay's continuous transport are FIRST-ORDER interior
        approximations of a nonlinear propagation; the interpolated-position
        Monte Carlo is the oracle, and the two forms diverge from it most where
        the ``1/sin(inc)`` azimuth weights are ill-conditioned. Measured against
        MC on a build well at ``f = 0.5``:

            inc 18 deg (low-inc build)   core 19.2%
            inc 36 deg                   core  2.6%
            inc 60 deg                   core 0.33%

        So this form OVER-states relative to MC, and by a wide margin at low
        inclination. Station values (``f = 0`` and ``f = 1``) are unaffected —
        those are exact. Aligning the boundary form to a continuous transport is
        tracked for 0.27; until then, treat sub-30 deg inclination interiors as
        indicative and take a station value, or welleng-assay's continuous form,
        where the number is load-bearing.

        A comparison column for welleng-assay's form was published here in
        0.26.0rc9 and is WITHDRAWN pending re-measurement: it was taken before
        their 2026-07-26 ``dref`` fix removed a constant VV double-count, so it
        overstated their error and understated the gap. Core's figures above are
        core-vs-MC and cannot be affected by a fix on assay's side, so they
        stand; the DIRECTION of the guidance (prefer a station value or the
        continuous form at low inclination) is unchanged and if anything
        stronger. Do not quote a ratio between the two forms until the corrected
        column lands.

        Parameters
        ----------
        md : float
            Measured depth of the interior point (survey depth units).

        Returns
        -------
        numpy.ndarray
            The (3, 3) NEV covariance at ``md``.
        """
        smd, sinc, sazi = self.survey_rad.T
        n = len(smd)
        i = int(np.searchsorted(smd, md) - 1)
        i = max(0, min(i, n - 2))
        seg = smd[i + 1] - smd[i]
        f = 0.0 if seg == 0.0 else float((md - smd[i]) / seg)

        inc_q, azi_q = self._interior_angles(i, f)

        Lq = md - smd[i]
        # partial-leg weights [i -> q]: drk (own, far station = q) and drkplus1
        # (coupling, near station = i). Both are needed so the interior is exact
        # at BOTH ends (option-c's own-only term is exact at f->1 but drops
        # station i's partial out-leg coupling, biasing f->0 by ~1 leg).
        dd, di, da = self._partial_star_drk(
            sinc[i], sazi[i], inc_q, azi_q, Lq
        )
        pd, pi_, pa = self._partial_plus1_drk(sinc[i], sazi[i], dd, Lq)
        classes, xcl_mag, tort, vlim = self._interior_prep()
        # interior TRUE azimuth (XCLA/XCLH weights use azi_true); grid->true
        # convergence is carried at station i and applied to the interior grid azi.
        azt = self.survey.azi_true_rad
        azt_q = azi_q + (azt[i] - sazi[i])
        # STANDARD sources, all at once over the source axis (see
        # _interior_stacks): the physics below is identical to the per-source
        # form, only the loop is gone. J maps a source's (D, I, A) to NEV
        # through the partial-leg weights, so `E @ J` is `dd*D + di*I + da*A`
        # for every source in one product.
        st = self._interior_stacks()
        J = np.stack((dd, di, da))                   # drk(i->q),      (3, 3)
        Jp = np.stack((pd, pi_, pa))                 # drkplus1(i->q), (3, 3)
        E_i, E_j = st["e_DIA"][:, i], st["e_DIA"][:, i + 1]
        qi = E_i @ J                                 # drk(i->q) . e_DIA[i]
        qj = E_j @ J                                 # drk(i->q) . e_DIA[i+1]
        coup = E_i @ Jp                              # station-i partial coupling
        rand = st["random"]
        # random: two INDEPENDENT measurements -> two outer products. The partial
        # q-own term splits (1-f)/f between stations i and i+1 (slerp-Jacobian
        # ~ f; exact at both ends, ~slerp tolerance in the interior -- assay's
        # symbolic is the exact oracle). Both endpoints recover
        # cov_NEV[i]/[i+1] exactly. The query-independent
        # `cov_NEV - outer(e_NEV_star, e_NEV_star)` part is pre-summed.
        g_i = st["e_NEV_star"][rand, i] + coup[rand] + (1.0 - f) * qi[rand]
        g_j = f * qj[rand]
        # systematic / global / well / within_pad: correlated vector sum.
        sg = ((1.0 - f) * qi[~rand] + f * qj[~rand]
              + st["sigma_e_NEV"][~rand, i] + coup[~rand])
        cov = (
            st["cov_random"][i]
            + np.einsum('si,sj->ij', g_i, g_i)
            + np.einsum('si,sj->ij', g_j, g_j)
            + np.einsum('si,sj->ij', sg, sg)
        )
        for name, src in self.errors.errors.items():
            cls = classes[name]
            if cls == "course_length":
                enq = self._xcl_partial_enev(
                    name, xcl_mag[name], tort, vlim,
                    sinc[i], azt[i], inc_q, azt_q, Lq
                )
                cov += src.cov_NEV[i] + np.outer(enq, enq)
            elif cls == "linear":
                cov += src.cov_NEV[i] + f * (src.cov_NEV[i + 1] - src.cov_NEV[i])
        # A covariance is symmetric; enforce it against floating-point drift from
        # the outer-product sums so downstream (e.g. the Mahalanobis solve) sees
        # a clean symmetric matrix.
        return 0.5 * (cov + cov.T)

    def _cov_NEV_carry_per_section(self, e_NEV, e_NEV_star, sections):
        """Per-continuous-section RSS of the systematic running-sum outer
        products (ISCWSA v5.13 Sec 7.3 pt14 / eqs 44-46).

        A RANDOM carried initialisation seed (a ``carry_only`` term, e.g. the
        gyro-compass init GRN-INIT) is propagated *systematic within* each
        continuous survey section but RE-RANDOMISES at every re-initialisation
        (a drop below the stationary init gate followed by a rebuild). Its
        covariance is therefore the root-sum-square (independent sum) of the
        per-section systematic running-sum outer products

            Sum_sec (Sum_{i in sec} e)(Sum_{i in sec} e)^T

        rather than one fully-correlated cumsum across the whole well
        (Sum_all e)(Sum_all e)^T. ``sections`` are the maximal continuous runs
        (``ToolError._continuous_sections``); a single section reduces this to
        the standard single outer product, so non-re-initialised wells are
        bit-identical. ``sigma_e_NEV`` (consumed by ``clearance.py``) is left
        as the global running sum -- only ``cov_NEV`` becomes the RSS.
        """
        cov_NEV = np.zeros((e_NEV.shape[0], 3, 3))
        for start, stop in sections:
            mask = np.zeros(e_NEV.shape[0], dtype=bool)
            mask[start:stop] = True
            e_sec = np.where(mask[:, None], e_NEV, 0.0)
            e_sec_star = np.where(mask[:, None], e_NEV_star, 0.0)
            sigma_sec = self._sigma_e_NEV_systematic(e_sec, e_sec_star)
            cov_NEV += cov_from_vec(sigma_sec)
        return cov_NEV

    def _generate_error(
        self,
        code,
        e_DIA,
        propagation='systematic',
        NEV=True,
        e_NEV=None,
        e_NEV_star=None,
        sections=None
    ):
        if not NEV:
            return e_DIA
        else:
            cov_DIA = cov_from_vec(e_DIA)
            if e_NEV is None:
                e_NEV = self._e_NEV(e_DIA)
                e_NEV_star = self._e_NEV_star(e_DIA)
            if propagation in ('systematic', 'global', 'within_pad', 'well'):
                # ``sigma_e_NEV`` stays the global systematic running sum (it
                # is consumed by ``clearance.py``). When ``sections`` is given
                # -- only for a ``carry_only`` random init seed under the
                # per-section re-init regime -- ``cov_NEV`` becomes the eqs
                # 44-46 per-section RSS instead of the single fully-correlated
                # outer product. A single section reduces to the same result,
                # so non-re-initialised wells are bit-identical.
                sigma_e_NEV = self._sigma_e_NEV_systematic(e_NEV, e_NEV_star)
                if sections is None:
                    cov_NEV = cov_from_vec(sigma_e_NEV)
                else:
                    cov_NEV = self._cov_NEV_carry_per_section(
                        e_NEV, e_NEV_star, sections
                    )
            elif propagation == 'random':
                sigma_e_NEV = np.cumsum(cov_from_vec(e_NEV), axis=0)
                cov_NEV = np.add(
                    cov_from_vec(e_NEV_star),
                    np.concatenate(
                        (
                            np.zeros((1, 3, 3)),
                            sigma_e_NEV[:-1]
                        ), axis=0)
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

    def drk_dDepth(self, survey):
        """Derivative of position with respect to measured depth at each station.

        Equal to 0.5 * (unit_vec[i] + unit_vec[i+1]) in NEV coordinates --
        the direction-cosine part of minimum curvature without the RF or
        delta_md. When the survey starts at the zero datum (md[0] == 0) station
        0 has no segment above it and takes the full station-0 wellbore tangent
        instead of the half-segment average: a depth error at the first station
        shifts the along-hole position by the full tangent (ISCWSA random depth
        carries its full variance at the surface, DRFR cov_VV(0) = mag^2). A
        survey that starts below the datum (md[0] != 0) is tied on, so its
        station 0 stays zero (see ``_drdp``). This row-0 value is consumed only
        by ``_e_NEV_star`` (``drkplus1_dDepth`` slices ``[1:]``).

        Parameters
        ----------
        survey : array_like
            Survey stations as (md, inc_rad, azi_rad) rows.

        Returns
        -------
        numpy.ndarray
            Shape (n, 3) array of NEV derivatives.
        """
        _, inc1, azi1 = np.array(survey[:-1]).T
        _, inc2, azi2 = np.array(survey[1:]).T
        si1, si2 = np.sin(inc1), np.sin(inc2)
        ca1, ca2 = np.cos(azi1), np.cos(azi2)
        sa1, sa2 = np.sin(azi1), np.sin(azi2)
        NEV = 0.5 * np.stack((
            si1 * ca1 + si2 * ca2,       # N
            si1 * sa1 + si2 * sa2,       # E
            np.cos(inc1) + np.cos(inc2), # V
        ), axis=-1)
        md0, i0, a0 = np.array(survey[0]).T
        if md0 == 0.0:
            row0 = np.array([
                np.sin(i0) * np.cos(a0),
                np.sin(i0) * np.sin(a0),
                np.cos(i0),
            ])
        else:
            row0 = np.zeros(3)
        return np.vstack((row0, NEV))

    def drk_dInc(self, survey):
        """Derivative of position with respect to inclination at each station.

        Parameters
        ----------
        survey : array_like
            Survey stations as (md, inc_rad, azi_rad) rows.

        Returns
        -------
        numpy.ndarray
            Shape (n, 3) array of NEV derivatives.

        Note
        ----
        The N/E columns are azimuth-dependent even at ``inc == 0`` (½·Δmd·cos(inc)
        ·{cos,sin}(azi)). ``Survey`` canonicalises azimuth to 0 at vertical
        stations (see ``Survey._make_angles``); a consumer feeding this model
        WITHOUT that preprocessing must apply ``azi = where(inc == 0, 0, azi)``
        first, or the covariance diverges at vertical stations.
        """
        md1, inc1, azi1 = np.array(survey[:-1]).T
        md2, inc2, azi2 = np.array(survey[1:]).T
        delta_md = md2 - md1

        N = np.array(0.5 * ((delta_md) * np.cos(inc2) * np.cos(azi2)))
        E = np.array(0.5 * ((delta_md) * np.cos(inc2) * np.sin(azi2)))
        V = np.array(0.5 * (-delta_md * np.sin(inc2)))

        # Rev5+ surface tie-on (Def. of ISCWSA Error Model §4.7.1.1, eq. 32):
        # double the FIRST station's full inc column (N,E,V), not just inc-N.
        # ISCWSA well #1 is vertical-North at station 1 so only N is non-zero
        # there, but E/V bite for a deviated slot. Only at a true surface root
        # (md[0] == 0); a below-datum tie-on carries station 0 externally. rev4
        # predates the tie-on.
        if md1[0] == 0.0 and self.error_model.lower().split()[-1] != 'rev4':
            N[0] *= 2
            E[0] *= 2
            V[0] *= 2

        return np.vstack(
            (
                np.array(np.zeros((1, 3))),
                np.stack((N, E, V), axis=-1)
            )
        )

    def drk_dAz(self, survey):
        """Derivative of position with respect to azimuth at each station.

        Parameters
        ----------
        survey : array_like
            Survey stations as (md, inc_rad, azi_rad) rows.

        Returns
        -------
        numpy.ndarray
            Shape (n, 3) array of NEV derivatives.
        """
        md1, inc1, azi1 = np.array(survey[:-1]).T
        md2, inc2, azi2 = np.array(survey[1:]).T
        delta_md = md2 - md1

        N = np.array(-0.5 * ((delta_md) * np.sin(inc2) * np.sin(azi2)))
        E = np.array(0.5 * ((delta_md) * np.sin(inc2) * np.cos(azi2)))
        V = np.zeros_like(N)

        # Rev5+ surface tie-on (Def. of ISCWSA Error Model §4.7.1.1, eq. 32):
        # double the FIRST station's azi column (N,E; azi-V is identically 0).
        # Only at a true surface root (md[0] == 0); a below-datum tie-on carries
        # station 0 externally.
        if md1[0] == 0.0 and self.error_model.lower().split()[-1] != 'rev4':
            N[0] *= 2
            E[0] *= 2

        return np.vstack(
            (
                np.array(np.zeros((1, 3))),
                np.stack((N, E, V), axis=-1)
            )
        )

    def drkplus1_dDepth(self, survey):
        """Derivative of next-station position with respect to measured depth.

        Parameters
        ----------
        survey : array_like
            Survey stations as (md, inc_rad, azi_rad) rows.

        Returns
        -------
        numpy.ndarray
            Shape (n, 3) array of NEV derivatives.
        """
        return np.vstack(
            (
                self.drk_dDepth(survey)[1:] * -1,
                np.array(np.zeros((1, 3)))
            )
        )

    def drkplus1_dInc(self, survey):
        """Derivative of next-station position with respect to inclination.

        Parameters
        ----------
        survey : array_like
            Survey stations as (md, inc_rad, azi_rad) rows.

        Returns
        -------
        numpy.ndarray
            Shape (n, 3) array of NEV derivatives.
        """

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
        """Derivative of next-station position with respect to azimuth.

        Parameters
        ----------
        survey : array_like
            Survey stations as (md, inc_rad, azi_rad) rows.

        Returns
        -------
        numpy.ndarray
            Shape (n, 3) array of NEV derivatives.
        """
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

    @staticmethod
    def _tangents(inc, azi):
        """Unit wellbore tangents in NEV for angle arrays (radians)."""
        return np.stack((
            np.sin(inc) * np.cos(azi),
            np.sin(inc) * np.sin(azi),
            np.cos(inc),
        ), axis=-1)

    @classmethod
    def _mc_leg_disp(cls, md1, inc1, azi1, md2, inc2, azi2):
        """Exact minimum-curvature NEV displacement of the leg (md1,inc1,azi1) ->
        (md2,inc2,azi2).

        The minimum-curvature displacement is the balanced-tangential chord
        ``0.5 * dmd * (t1 + t2)`` scaled by the ratio factor
        ``RF = (2/DL) * tan(DL/2)``, where ``DL`` is the dogleg (subtended angle
        between the two unit tangents). ``RF -> 1`` as ``DL -> 0`` (the straight
        leg), so a straight interval reduces to the balanced-tangential chord.
        This reproduces the survey's reconstructed min-curvature positions to
        machine precision -- it is the trajectory the well is defined on.

        Arrays are accepted (one row per leg) so the Jacobian below can be
        differenced in a vectorised sweep.
        """
        t1 = cls._tangents(inc1, azi1)
        t2 = cls._tangents(inc2, azi2)
        dmd = np.asarray(md2, float) - np.asarray(md1, float)
        cos_dl = np.clip(np.sum(t1 * t2, axis=-1), -1.0, 1.0)
        dl = np.arccos(cos_dl)
        # RF = (2/DL) tan(DL/2); removable singularity at DL=0 (RF -> 1).
        safe = dl > 1e-7
        rf = np.where(safe, (2.0 / np.where(safe, dl, 1.0)) * np.tan(dl / 2.0), 1.0)
        return 0.5 * dmd[..., None] * rf[..., None] * (t1 + t2)

    @staticmethod
    def _rf_and_k(dl):
        """The minimum-curvature ratio factor ``RF = (2/DL) tan(DL/2)`` and the
        scalar ``K = RF'(DL) / sin(DL)`` that appears in its angle-derivative,
        evaluated stably through the removable singularity at ``DL = 0``.

        ``RF -> 1`` and ``K -> 1/6`` as ``DL -> 0``. Below a small threshold both
        are taken from their leading Taylor series (``RF = 1 + DL^2/12``,
        ``K = 1/6 + 11 DL^2/180``) to avoid the ``0/0`` in ``RF`` and the
        catastrophic cancellation in ``RF'(DL) = (sec^2(DL/2) - RF)/DL``; above it
        the closed forms are well-conditioned. No branch on the Jacobian itself --
        only this scalar pair is series-guarded.
        """
        small = dl < 1e-3
        dl_safe = np.where(small, 1.0, dl)
        rf_exact = (2.0 / dl_safe) * np.tan(dl_safe / 2.0)
        rf = np.where(small, 1.0 + dl * dl / 12.0, rf_exact)
        rf_prime = (1.0 / dl_safe) * (1.0 / np.cos(dl_safe / 2.0) ** 2 - rf_exact)
        k = np.where(
            small, 1.0 / 6.0 + 11.0 * dl * dl / 180.0,
            rf_prime / np.sin(dl_safe),
        )
        return rf, k

    def _drdp_min_curve(self, survey):
        """Minimum-curvature dp basis: the analytic Jacobian of the exact
        min-curvature position wrt the survey measurements (``dp_basis='min_curve'``).

        The default balanced-tangential ``_drdp`` weights a leg by the tangent
        average alone; the minimum-curvature basis weights it by the actual arc --
        the ratio factor ``RF`` AND its angle-derivatives (the leg direction
        shifts, not just its magnitude). It is the dp basis CONSISTENT with the
        min-curvature trajectory the survey is reconstructed on, differs from
        balanced-tangential by O(interval^2) (~0.1% of the total covariance at
        30 m, growing with dogleg -- within the ISCWSA inter-implementation band),
        and is validated against welleng-assay's symbolic min-curve Jacobian and
        its Monte-Carlo. Opt-in: the default ``'balanced_tangent'`` reproduces the
        published ISCWSA numbers exactly.

        A leg's displacement is ``D = 0.5 * dmd * RF * (t1 + t2)``
        (:meth:`_mc_leg_disp`).
        Differentiating wrt an angle ``th`` of a bounding station, with
        ``u = t1.t2 = cos(DL)`` so ``dDL/dth = -(du/dth)/sin(DL)``:

            dD/dth = 0.5 * dmd * ( -K * (du/dth) * (t1 + t2) + RF * dt/dth )

        where ``K = RF'(DL)/sin(DL)`` (:meth:`_rf_and_k`). At a straight leg
        ``du/dth -> 0`` (the derivative of a unit dot product at coincidence), so
        the arc terms vanish and the basis reduces to balanced-tangential SMOOTHLY
        -- no DL->0 special case is needed. Depth derivatives carry no RF term
        (``RF`` is angle-only): ``dD/dmd2 = 0.5 * RF * (t1 + t2)``.

        The surface-boundary conventions (station-0 direct depth seed and the Rev5
        surface tie-on doubling) are dp-basis-independent and applied identically
        to :meth:`_drdp`.
        """
        survey = np.asarray(survey, float)
        n = len(survey)
        md, inc, azi = survey.T
        si, ci = np.sin(inc), np.cos(inc)
        sa, ca = np.sin(azi), np.cos(azi)
        t = self._tangents(inc, azi)                              # (n,3) tangents
        dt_di = np.stack((ci * ca, ci * sa, -si), axis=-1)        # dt/dinc
        dt_da = np.stack((-si * sa, si * ca, np.zeros_like(si)), axis=-1)  # dt/dazi

        t1, t2 = t[:-1], t[1:]
        dmd = np.diff(md)
        u = np.clip(np.sum(t1 * t2, axis=-1), -1.0, 1.0)
        dl = np.arccos(u)
        rf, kk = self._rf_and_k(dl)
        tsum = t1 + t2
        half = 0.5 * dmd[:, None]
        rf_c, kk_c = rf[:, None], kk[:, None]

        result = np.zeros((n, 18))

        # own leg [k-1 -> k], wrt station k (the '2' side); rows 1..n-1
        du_di2 = np.sum(t1 * dt_di[1:], axis=-1)[:, None]
        du_da2 = np.sum(t1 * dt_da[1:], axis=-1)[:, None]
        result[1:, 0:3] = 0.5 * rf_c * tsum                       # drk_dDepth
        result[1:, 3:6] = half * (-kk_c * du_di2 * tsum + rf_c * dt_di[1:])
        result[1:, 6:9] = half * (-kk_c * du_da2 * tsum + rf_c * dt_da[1:])

        # next leg [k -> k+1], wrt station k (the '1' side); rows 0..n-2
        du_di1 = np.sum(dt_di[:-1] * t2, axis=-1)[:, None]
        du_da1 = np.sum(dt_da[:-1] * t2, axis=-1)[:, None]
        result[:-1, 9:12] = -0.5 * rf_c * tsum                    # drkplus1_dDepth
        result[:-1, 12:15] = half * (-kk_c * du_di1 * tsum + rf_c * dt_di[:-1])
        result[:-1, 15:18] = half * (-kk_c * du_da1 * tsum + rf_c * dt_da[:-1])

        # surface-boundary conventions (identical to _drdp)
        if md[0] == 0.0:
            result[0, 0:3] = [si[0] * ca[0], si[0] * sa[0], ci[0]]
        if md[0] == 0.0 and self.error_model.lower().split()[-1] != 'rev4':
            result[1, 3:9] *= 2
        return result

    def _drdp(self, survey):
        '''
        Jacobian of position wrt survey parameters, computed in a single trig
        pass.  Returns array of shape (n, 18): columns 0-2 drk_dDepth,
        3-5 drk_dInc, 6-8 drk_dAz, 9-11 drkplus1_dDepth,
        12-14 drkplus1_dInc, 15-17 drkplus1_dAz.

        The ``dp_basis`` header selects the leg weighting: the default
        ``'balanced_tangent'`` (this method, the ISCWSA standard, exact-published)
        or ``'min_curve'`` (:meth:`_drdp_min_curve`, the trajectory-consistent
        basis).

        Station-data convention: each station k's measurement error propagates
        through BOTH adjacent minimum-curvature segments -- [k-1->k] (the
        ``drk_*`` columns) and [k->k+1] (the ``drkplus1_*`` columns), summed
        in ``_e_NEV`` as ``(drdp[:,0]+drdp[:,9])*D + ...``. This is the N+/-1
        ("station above and below") variant, NOT the N-2/N-1 ("two previous")
        variant. It is the variant that reproduces the ISCWSA MWD reference to
        ~5e-5 (tests/test_iscwsa_mwd_error.py); SPE 90408 gyro Appendix E may
        use the other interpretation, contributing to its ~0.6% inter-
        implementation residual. See docs/dev/VALIDATION.md ("Known
        differences") and ISCWSA "Test Profile Differences" (CDR-SM-03).
        '''
        if getattr(self.survey.header, 'dp_basis', 'balanced_tangent') \
                == 'min_curve':
            return self._drdp_min_curve(survey)
        survey = np.array(survey)
        n = len(survey)
        md, inc, azi = survey.T
        si, ci = np.sin(inc), np.cos(inc)
        sa, ca = np.sin(azi), np.cos(azi)
        half_dmd = 0.5 * np.diff(md)          # shape (n-1,)

        result = np.zeros((n, 18))

        # drk_dDepth: rows 1..n-1
        dc_N = 0.5 * (si[:-1] * ca[:-1] + si[1:] * ca[1:])
        dc_E = 0.5 * (si[:-1] * sa[:-1] + si[1:] * sa[1:])
        dc_V = 0.5 * (ci[:-1] + ci[1:])
        result[1:, 0] = dc_N
        result[1:, 1] = dc_E
        result[1:, 2] = dc_V

        # Station-0 direct depth-measurement derivative -- ONLY for a survey
        # that starts at the zero datum (md[0] == 0, a true surface root). A
        # depth error at that first station shifts the along-hole position by
        # the FULL wellbore tangent (not the half-segment min-curvature average
        # -- there is no segment above station 0), so the ISCWSA random-depth
        # source carries its full variance at the surface: DRFR cov_VV(0) =
        # mag^2. Only the depth column is seeded -- inc/azi errors have zero
        # lever arm at station 0.
        #
        # When md[0] != 0 the survey starts BELOW the datum: station 0 is a
        # tie-on whose uncertainty is carried externally (a composed section's
        # freeze-carry tie, or a hierarchy sidetrack inheriting its parent), so
        # NO fresh direct-depth term is added -- the row-0 depth column stays 0,
        # exactly as the pre-0.25.0 datum. This keeps the station-0 term local
        # to the random branch AND preserves the composition/hierarchy tie
        # invariant that a sub-run's station 0 injects no new error.
        if md[0] == 0.0:
            result[0, 0] = si[0] * ca[0]
            result[0, 1] = si[0] * sa[0]
            result[0, 2] = ci[0]

        # drk_dInc: rows 1..n-1 (wrt inc at station i+1)
        result[1:, 3] = half_dmd * ci[1:] * ca[1:]
        result[1:, 4] = half_dmd * ci[1:] * sa[1:]
        result[1:, 5] = -half_dmd * si[1:]

        # drk_dAz: rows 1..n-1 (wrt azi at station i+1)
        result[1:, 6] = -half_dmd * si[1:] * sa[1:]
        result[1:, 7] = half_dmd * si[1:] * ca[1:]

        # Rev5+ surface tie-on (Definition of ISCWSA Error Model §4.7.1.1,
        # eq. 32): from Revision 5 the slot attitude is allowed its own
        # measurement error, of the same magnitude as a downhole survey. This is
        # modelled by doubling the FIRST station's inc AND azi weighting columns
        # -- the full middle and right-hand columns of eq. (10), i.e. inc-{N,E,V}
        # and azi-{N,E} (azi-V is identically 0). Applied after both columns are
        # populated. rev4 predates the surface tie-on, so it is excluded.
        #
        # NB the ISCWSA reference well #1 is vertical-North at station 1
        # (inc=azi=0), so only the inc-N term is non-zero there; doubling inc-N
        # alone reproduces the reference, but the E/V/azi terms matter for a
        # deviated first survey (see tests/test_iscwsa_surface_tieon.py).
        #
        # Only at a true surface root (md[0] == 0). A sub-survey starting below
        # the datum (md[0] != 0 -- a composed/hierarchy tie-on) carries its
        # station-0 attitude uncertainty externally, so no fresh slot allowance
        # is injected. Mirrors the station-0 depth (DRFR) gate above and the
        # composition/hierarchy tie invariant.
        if md[0] == 0.0 and self.error_model.lower().split()[-1] != 'rev4':
            result[1, 3:9] *= 2

        # drkplus1_dDepth: rows 0..n-2 (negated dc)
        result[:-1, 9]  = -dc_N
        result[:-1, 10] = -dc_E
        result[:-1, 11] = -dc_V

        # drkplus1_dInc: rows 0..n-2 (wrt inc at station i)
        result[:-1, 12] = half_dmd * ci[:-1] * ca[:-1]
        result[:-1, 13] = half_dmd * ci[:-1] * sa[:-1]
        result[:-1, 14] = -half_dmd * si[:-1]

        # drkplus1_dAz: rows 0..n-2 (wrt azi at station i)
        result[:-1, 15] = -half_dmd * si[:-1] * sa[:-1]
        result[:-1, 16] = half_dmd * si[:-1] * ca[:-1]

        return result

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
    """Extract the six unique covariance components from a 3x3 NEV matrix.

    Parameters
    ----------
    error : numpy.ndarray
        A 3x3 covariance matrix in NEV coordinates.

    Returns
    -------
    list
        [nn, ee, vv, ne, nv, ev] covariance components.
    """
    nn, ne, nv = error[0]
    _, ee, ev = error[1]
    _, __, vv = error[2]

    return [nn, ee, vv, ne, nv, ev]


def make_diagnostic_data(survey):
    """Build a per-station diagnostic breakdown of all error model components.

    Parameters
    ----------
    survey : welleng.survey.Survey
        A welleng Survey with an attached ErrorModel (survey.err).

    Returns
    -------
    dict
        Nested dict keyed by MD, then error code, containing the six
        unique covariance components and a TOTAL row summing all codes.
    """
    diagnostic = {}
    dia = np.stack((survey.md, survey.inc_deg, survey.azi_grid_deg), axis=1)
    for i, d in enumerate(survey.md):
        diagnostic[d] = {}
        total = []
        for k, v in survey.err.errors.errors.items():
            diagnostic[d][k] = get_errors(v.cov_NEV[i])
            total.extend(diagnostic[d][k])
        diagnostic[d]['TOTAL'] = np.sum((np.array(
            total
        ).reshape(-1, len(diagnostic[d][k]))), axis=0)
    return diagnostic
