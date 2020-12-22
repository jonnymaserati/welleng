import numpy as np
import math

from .utils import (
    MinCurve,
    get_nev,
    get_vec,
    get_angles,
    HLA_to_NEV,
    NEV_to_HLA,
    get_sigmas
)

from welleng.error import ErrorModel
from welleng.exchange.wbp import TurnPoint

class Survey:
    def __init__(
        self,
        md,
        inc,
        azi,
        n=None,
        e=None,
        tvd=None,
        x=None,
        y=None,
        z=None,
        vec=None,
        radius=None,
        cov_nev=None,
        cov_hla=None,
        error_model=None,
        well_ref_params=None,
        start_xyz=[0,0,0],
        start_nev=[0,0,0],
        start_cov_nev=None,
        deg=True,
        unit="meters"
    ):
        """
        Initialize a welleng.Survey object.

        Parameters
        ----------
            md: (,n) list or array of floats
                List or array of well bore measured depths.
            inc: (,n) list or array of floats
                List or array of well bore survey inclinations
            azi: (,n) list or array of floats
                List or array of well bore survey azimuths
            n: (,n) list or array of floats (default: None)
                List or array of well bore northings
            e: (,n) list or array of floats (default: None)
                List or array of well bore eastings
            tvd: (,n) list or array of floats (default: None)
                List or array of local well bore z coordinates, i.e. depth
                and usually relative to surface or mean sea level.
            x: (,n) list or array of floats (default: None)
                List or array of local well bore x coordinates, which is
                usually aligned to the east direction.
            y: (,n) list or array of floats (default: None)
                List or array of local well bore y coordinates, which is
                usually aligned to the north direction.
            z: (,n) list or array of floats (default: None)
                List or array of well bore true vertical depths relative
                to the well surface datum (usually the drill floor 
                elevation DFE, so not always identical to tvd).
            vec: (n,3) list or array of (,3) floats (default: None)
                List or array of well bore unit vectors that describe the
                inclination and azimuth of the well relative to (x,y,z)
                coordinates.
            radius: float or (,n) list or array of floats (default: None)
                If a single float is specified, this value will be
                assigned to the entire well bore. If a list or array of
                floats is provided, these are the radii of the well bore.
                If None, a well bore radius of 12" or approximately 0.3 m
                is applied.
            cov_nev: (n,3,3) list or array of floats (default: None)
                List or array of covariance matrices in the (n,e,v)
                coordinate system.
            cov_hla: (n,3,3) list or array of floats (default: None)
                List or array of covariance matrices in the (h,l,a)
                well bore coordinate system (high side, lateral, along
                hole).
            error_model: str (default: None)
                If specified, this model is used to calculate the
                covariance matrices if they are not present. Currently,
                only the "ISCWSA_MWD" model is provided.
            well_ref_params: dict (default: None)
                If an error_model is set, these well reference params
                are provided to the welleng.error.ErrorModel class. The
                defaults are:
                    dict(
                        Latitude = -40,     # degrees
                        G = 9.80665,        # m/s2
                        BTotal = 61000,     # nT
                        Dip = -70,          # degrees
                        Declination = 13,   # degrees  
                        Convergence = 0,    # degrees
                    )
            start_xyz: (,3) list or array of floats (default: [0,0,0])
                The start position of the well bore in (x,y,z) coordinates.
            start_nev: (,3) list or array of floats (default: [0,0,0])
                The start position of the well bore in (n,e,v) coordinates.
            start_cov_nev: (,3,3) list or array of floats (default: None)
                The covariance matrix for the start position of the well
                bore in (n,e,v) coordinates.
            deg: boolean (default: True)
                Indicates whether the provided angles are in degrees
                (True), else radians (False).
            unit: str (default: 'meters')
                Indicates whether the provided lengths and distances are
                in 'meters' or 'feet', which impacts the calculation of
                the dls (dog leg severity).

        Returns
        -------
        A welleng.survey.Survey object.
        """
        self.unit = unit
        self.deg = deg
        self.start_xyz = start_xyz
        self.start_nev = start_nev
        self.md = np.array(md)
        self.start_cov_nev = start_cov_nev
        self._make_angles(inc, azi, deg)
        self._get_radius(radius)
        
        self.survey_deg = np.array([self.md, self.inc_deg, self.azi_deg]).T
        self.survey_rad = np.array([self.md, self.inc_rad, self.azi_rad]).T

        self.n = n
        self.e = e
        self.tvd = tvd
        self.x = x
        self.y = y
        self.z = z
        self.vec = vec

        self._min_curve()
        self._get_toolface_and_rates()

        # initialize errors
        # TODO: read this from a yaml file in errors
        error_models = ["ISCWSA_MWD"]
        if error_model is not None:
            assert error_model in error_models, "Unrecognized error model"
        self.error_model = error_model
        self.well_ref_params = well_ref_params

        self.cov_hla = cov_hla
        self.cov_nev = cov_nev

        self._get_errors()

    def _get_radius(self, radius=None):
        if radius is None:
            self.radius = np.full_like(self.md.astype(float), 0.3048)
        elif np.array([radius]).shape[-1] == 1:
            self.radius = np.full_like(self.md.astype(float), radius)
        else:
            assert len(radius) == len(self.md), "Check radius"
            self.radius = np.array(radius)
    
    def _min_curve(self):
        """
        Get the (x,y,z), (n,e,v), doglegs, rfs, delta_mds, dlss and
        vectors for the well bore if they were not provided, using the
        minimum curvature method.
        """
        mc = MinCurve(self.md, self.inc_rad, self.azi_rad, self.start_xyz, self.unit)
        self.dogleg = mc.dogleg
        self.rf = mc.rf
        self.delta_md = mc.delta_md
        self.dls = mc.dls
        self.pos = mc.poss

        if self.x is None:
            # self.x, self.y, self.z = (mc.poss + self.start_xyz).T
            self.x, self.y, self.z = (mc.poss).T
        if self.n is None:
            self._get_nev()
        if self.vec is None:
            self.vec = get_vec(self.inc_rad, self.azi_rad, deg=False)

    def _get_nev(self):
        self.n, self.e, self.tvd = get_nev(
            np.array([
                self.x,
                self.y,
                self.z
            ]).T,
            start_xyz=self.start_xyz,
            start_nev=self.start_nev
        ).T

    def _make_angles(self, inc, azi, deg=True):
        """
        Calculate angles in radians if they were provided in degrees or
        vice versa.
        """
        if deg:
            self.inc_rad = np.radians(inc)
            self.azi_rad = np.radians(azi)
            self.inc_deg = np.array(inc)
            self.azi_deg = np.array(azi)
        else:
            self.inc_rad = np.array(inc)
            self.azi_rad = np.array(azi)
            self.inc_deg = np.degrees(inc)
            self.azi_deg = np.degrees(azi)

    def _get_errors(self):
        """
        Initiate a welleng.error.ErrorModel object and calculate the
        covariance matrices with the specified error model.
        """
        if self.error_model:
            if self.error_model == "ISCWSA_MWD":
                if self.well_ref_params is None:
                    self.err = ErrorModel(
                        survey=self.survey_deg,
                        surface_loc=self.start_xyz,
                    )
                else:
                    self.err = ErrorModel(
                        survey=self.survey_deg,
                        surface_loc=self.start_xyz,
                        well_ref_params=self.well_ref_params,
                    )
                self.cov_hla = self.err.errors.cov_HLAs.T
                self.cov_nev = self.err.errors.cov_NEVs.T
        else:
            if self.cov_nev is not None and self.cov_hla is None:
                self.cov_hla = NEV_to_HLA(self.survey_rad, self.cov_nev.T).T
            elif self.cov_nev is None and self.cov_hla is not None:
                self.cov_nev = HLA_to_NEV(self.survey_rad, self.cov_hla.T).T
            else:
                pass
        
        if (
            self.start_cov_nev is not None
            and self.cov_nev is not None
        ):
            self.cov_nev += self.start_cov_nev
            self.cov_hla = NEV_to_HLA(self.survey_rad, self.cov_nev.T).T

    def _curvature_to_rate(self, curvature):
        with np.errstate(divide='ignore', invalid='ignore'):
            radius = 1 / curvature
        circumference = 2 * np.pi * radius
        if self.unit == 'meters':
            x = 30
        else:
            x = 100
        rate = np.absolute(np.degrees(2 * np.pi / circumference) * x)

        return rate
    
    def _get_toolface_and_rates(self):
        """
        Reference SPE-84246.
        theta is inc, phi is azi
        """
        # split the survey
        s = SplitSurvey(self)

        if self.unit == 'meters':
            x = 30
        else:
            x = 100

        # this is lazy I know, but I'm using this mostly for flags
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = np.arctan(
                np.sin(s.inc2) * np.sin(s.delta_azi) /
                (
                    np.sin(s.inc2) * np.cos(s.inc1) * np.cos(s.delta_azi)
                    - np.sin(s.inc1) * np.cos(s.inc2)
                )
            )
            t1 = np.nan_to_num(
                np.where(t1 < 0, t1 + 2 * np.pi, t1),
                nan=np.nan
            )
            t2 = np.arctan(
                np.sin(s.inc1) * np.sin(s.delta_azi) /
                (
                    np.sin(s.inc2) * np.cos(s.inc1)
                    - np.sin(s.inc1) * np.cos(s.inc2) * np.cos(s.delta_azi)
                )
            )
            t2 = np.nan_to_num(
                np.where(t2 < 0, t2 + 2 * np.pi, t2),
                nan=np.nan
            )
            self.curve_radius = (360 / self.dls * x) / (2 * np.pi)
        
            curvature_dls = 1 / self.curve_radius

            self.toolface = np.concatenate((t1, np.array([t2[-1]])))

            curvature_turn = curvature_dls * (np.sin(self.toolface) / np.sin(self.inc_rad))
            self.turn_rate = self._curvature_to_rate(curvature_turn)

            curvature_build = curvature_dls * np.cos(self.toolface)
            self.build_rate = self._curvature_to_rate(curvature_build)

        # calculate plan normals
        n12 = np.cross(s.vec1, s.vec2)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.normals = n12 / np.linalg.norm(n12, axis=1).reshape(-1,1)

def interpolate_survey(survey, x=0, index=0):
    """
    Interpolates a point distance x between two survey stations
    using minimum curvature.

    Parameters
    ----------
        survey: welleng.Survey
            A survey object with at least two survey stations.
        x: float
            Length along well path from indexed survey station to
            perform the interpolate at. Must be less than length
            to the next survey station.
        index: int
            The index of the survey station from which to interpolate
            from.

    Returns
    -------
        survey: welleng.Survey
            A survey object consisting of the two survey stations
            between which the interpolation has been made (index 0 and
            2), with the interpolated station between them (index 1)

    """
    index = int(index)

    assert index < len(survey.md) - 1, "Index is out of range"

    # assert x <= survey.delta_md[index + 1], "x is out of range"


    # check if it's just a tangent section
    if survey.dogleg[index + 1] == 0:
        azi = survey.azi_rad[index]
        inc = survey.inc_rad[index]

    else:
        # get the vector
        t1 = survey.vec[index]
        t2 = survey.vec[index + 1]

        total_dogleg = survey.dogleg[index + 1]

        dogleg = x * (survey.dogleg[index + 1] / survey.delta_md[index + 1])

        t = (
            (math.sin(total_dogleg - dogleg) / math.sin(total_dogleg)) * t1
            + (math.sin(dogleg) / math.sin(total_dogleg)) * t2
        )

        inc, azi = get_angles(t)[0]

    s = Survey(
        md=np.array([survey.md[index], survey.md[index] + x]),
        inc=np.array([survey.inc_rad[index], inc]),
        azi=np.array([survey.azi_rad[index], azi]),
        start_xyz=np.array([survey.x, survey.y, survey.z]).T[index],
        start_nev=np.array([survey.n, survey.e, survey.tvd]).T[index],
        deg=False
    )
    s._min_curve()

    return s

def slice_survey(survey, start, stop=None):
    """
    Take a slice from a welleng.survey.Survey object.

    Parameters
    ----------
        survey: welleng.survey.Survey object
        start: int
            The start index of the desired slice.
        stop: int (default: None)
            The stop index of the desired slice, else the remainder of
            the well bore TD is the default.

    Returns
    -------
        s: welleng.survey.Survey object
            A survey object of the desired slice is returned.
    """
    if stop is None:
        stop = start + 2
    md, inc, azi = survey.survey_rad[start:stop].T
    nevs = np.array([survey.n, survey.e, survey.tvd]).T[start:stop]
    n, e, tvd = nevs.T
    vec = survey.vec[start:stop]
        
    s = Survey(
        md=md,
        inc=inc,
        azi=azi,
        n=n,
        e=e,
        tvd=tvd,
        radius=survey.radius[start:stop],
        cov_hla=survey.cov_hla[start:stop],
        cov_nev=survey.cov_nev[start:stop],
        start_nev=[n[0], e[0], tvd[0]],
        deg=False,
        unit=survey.unit,
    )

    return s

def make_cov(a, b, c, diag=False):
    """
    Make a covariance matrix from the 1-sigma errors.

    Parameters
    ----------
        a: (,n) list or array of floats
            Errors in H or N/y axis.
        b: (,n) list or array of floats
            Errors in L or E/x axis.
        c: (,n) list or array of floats
            Errors in A or V/TVD axis.
        diag: boolean (default=False)
            If true, only the lead diagnoal is calculated
            with zeros filling the remainder of the matrix.

    Returns
    -------
        cov: (n,3,3) np.array
    """

    if diag:
        z = np.zeros_like(np.array([a]).reshape(-1))
        cov = np.array([
            [a * a, z, z],
            [z, b * b, z],
            [z, z, c * c]
        ]).T
    else:
        cov = np.array([
            [a * a, a * b, a * c],
            [a * b, b * b, b * c],
            [a * c, b * c, c * c]
        ]).T

    return cov

def make_long_cov(arr):
    """
    Make a covariance matrix from the half covariance 1sigma data.
    """
    aa, ab, ac, bb, bc, cc = np.array(arr).T
    cov = np.array([
        [aa, ab, ac],
        [ab, bb, bc],
        [ac, bc, cc]
    ]).T

    return cov

class SplitSurvey:
    def __init__(
        self,
        survey,
    ):
        self.md1, self.inc1, self.azi1 = survey.survey_rad[:-1].T
        self.md2, self.inc2, self.azi2 = survey.survey_rad[1:].T
        self.delta_azi = self.azi2 - self.azi1
        self.delta_inc = self.inc2 - self.inc1

        self.vec1 = survey.vec[:-1]
        self.vec2 = survey.vec[1:]
        self.dogleg = survey.dogleg[1:]   

def get_circle_radius(survey, **targets):
    # TODO: add target data to sections
    ss = SplitSurvey(survey)

    x1, y1, z1 = np.cross(ss.vec1, survey.normals).T
    x2, y2, z2 = np.cross(ss.vec2, survey.normals).T

    b1 = np.array([y1, x1, z1]).T
    b2 = np.array([y2, x2, z2]).T
    nev = np.array([survey.n, survey.e, survey.tvd]).T

    cc1 = (
        nev[:-1] - b1
        / np.linalg.norm(b1, axis=1).reshape(-1,1)
        * survey.curve_radius[:-1].reshape(-1,1)
    )
    cc2 = (
        nev[1:] - b2
        / np.linalg.norm(b2, axis=1).reshape(-1,1)
        * survey.curve_radius[1:].reshape(-1,1)
    )

    starts = np.vstack((cc1, cc2))
    ends = np.vstack((nev[:-1], nev[1:]))

    n = 1


    return (starts, ends)

def get_sections(survey, rtol=1e-1, atol=1e-2, **targets):
    """
    Tries to discretize a survey file into hold or curve sections. These
    sections can then be used to generate a WellPlan object to generate a
    .wbp format file for import into Landmark COMPASS, thus converting a
    survey file to an editable well trajectory.

    Note that this is in development and only tested on output from planning
    software. In its current form it likely won't be too successful on
    "as drilled" surveys (but optimizing the tolerances may help).

    Parameters
    ----------
        survey: welleng.survey.Survey object
        rtol: float (default: 1e-1)
            The relative tolerance when comparing the normals using the
            numpy.isclose() function.
        atol: float (default: 1e-2)
            The absolute tolerance when comparing the normals using the
            numpy.isclose() function.
        **targets: list of Target objects
            Not supported yet...

    Returns:
        sections: list of welleng.exchange.wbp.TurnPoint objects
    """

    # TODO: add target data to sections
    ss = SplitSurvey(survey)
    
    continuous = np.all(
        np.isclose(
            survey.normals[:-1],
            survey.normals[1:],
            rtol=rtol, atol=atol,
            equal_nan=True
        ), axis=-1
    )

    ends = np.concatenate(
        (np.where(continuous == False)[0] + 1, np.array([len(continuous)-1]))
    )
    starts = np.concatenate(([0], ends[:-1]))
    
    actions = [
        "hold" if a else "curve"
        for a in np.isnan(survey.toolface[starts])
    ]

    sections = []
    tie_on = True
    for s, e, a in zip(starts, ends, actions):
        md = survey.md[s]
        inc = survey.inc_deg[s]
        azi = survey.azi_deg[s]
        x = survey.e[s]
        y = survey.n[s]
        z = -survey.tvd[s]
        location = [x, y, z]

        target = ""
        if survey.unit == 'meters':
            denominator = 30
        else:
            denominator = 100

        if a == "hold":
            dls = 0.0
            toolface = 0.0
            build_rate = 0.0
            turn_rate = 0.0
            method = ""
        else:
            method = "0"
            dls = survey.dls[e]
            toolface = np.degrees(survey.toolface[s])
            # looks like the toolface is in range -180 to 180 in the .wbp file
            toolface = toolface - 360 if toolface > 180 else toolface
            delta_md = survey.md[e] - md

            # TODO: should sum this line by line to avoid issues with long sections
            build_rate = abs(
                (survey.inc_deg[e] - survey.inc_deg[s])
                / delta_md * denominator
            )

            # TODO: should sum this line by line to avoid issues with long sections
            # need to be careful with azimuth straddling north
            delta_azi_1 = abs(survey.azi_deg[e] - survey.azi_deg[s])
            delta_azi_2 = abs(360 - delta_azi_1)
            delta_azi = min(delta_azi_1, delta_azi_2)
            turn_rate = delta_azi / delta_md * denominator

        section = TurnPoint(
            md=md,
            inc=inc,
            azi=azi,
            build_rate=build_rate,
            turn_rate=turn_rate,
            dls=dls,
            toolface=toolface,
            method=method,
            target=None,
            tie_on=tie_on,
            location=location
        )

        sections.append(section)

        # Repeat the first section so that creating .wbp works
        if tie_on:
            sections.append(section)
            sections[-1].tie_on = False

        tie_on = False
    
    return sections