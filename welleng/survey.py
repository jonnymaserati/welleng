import numpy as np
import math

from welleng.utils import (
    MinCurve,
    get_nev,
    get_vec,
    get_angles,
    HLA_to_NEV,
    NEV_to_HLA,
    get_sigmas
)

from welleng.error import ErrorModel

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
        """
        self.unit = unit
        self.deg = deg
        self.start_xyz = start_xyz
        self.start_nev = start_nev
        self.md = np.array(md)
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

        # initialize errors
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

        """
        mc = MinCurve(self.md, self.inc_rad, self.azi_rad, self.start_xyz, self.unit)
        self.dogleg = mc.dogleg
        self.rf = mc.rf
        self.delta_md = mc.delta_md
        self.dls = mc.dls
        if self.x is None:
            self.x, self.y, self.z = mc.poss.T
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
            self.start_xyz,
            self.start_nev
        ).T

    def _make_angles(self, inc, azi, deg=True):
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

def interpolate_survey(survey, x, index=0):
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

    assert x <= survey.delta_md[index + 1], "x is out of range"


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

def slice_survey(survey, index):
        i = index
        md, inc, azi = survey.survey_rad[i-1:i+1].T
        nevs = np.array([survey.n, survey.e, survey.tvd]).T[i-1:i+1]
        n, e, tvd = nevs.T
        vec = survey.vec[i-1:i+1]
            
        s = Survey(
            md=md,
            inc=inc,
            azi=azi,
            n=n,
            e=e,
            tvd=tvd,
            radius=survey.radius[i-1:i+1],
            cov_hla=survey.cov_hla[i-1:i+1],
            cov_nev=survey.cov_nev[i-1:i+1],
            start_nev=[n[0], e[0], tvd[0]],
            deg=False,
            unit=survey.unit,
        )

        return s

def make_cov(a, b, c, diag=False):
    """
    Make a covariance matrix from the 1sigma errors.

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

def make_long_cov(aa, ab, ac, bb, bc, cc):
    """
    Make a covariance matrix from the half covariance
    1sigma data.
    """
    cov = np.array([
        [aa, ab, ac],
        [ab, bb, bc],
        [ac, bc, cc]
    ]).T

    return cov

    