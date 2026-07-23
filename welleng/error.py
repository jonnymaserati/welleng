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

    def _drdp(self, survey):
        '''
        Jacobian of position wrt survey parameters, computed in a single trig
        pass.  Returns array of shape (n, 18): columns 0-2 drk_dDepth,
        3-5 drk_dInc, 6-8 drk_dAz, 9-11 drkplus1_dDepth,
        12-14 drkplus1_dInc, 15-17 drkplus1_dAz.

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
