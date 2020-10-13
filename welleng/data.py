import numpy as np, math

from welleng.utils import *

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
        sigmaH=None,
        sigmaL=None,
        sigmaA=None,
        sigmaN=None,
        sigmaE=None,
        sigmaV=None,
        error_model=None,
        start_xyz=[0,0,0],
        start_nev=[0,0,0],
        deg=True,
        unit="meters"
    ):
        self.unit = unit
        self.deg = deg
        self.start_xyz = start_xyz
        self.start_nev = start_nev
        self.md = np.array(md)
        self._make_angles(inc, azi, deg)
        self.radius = radius
        
        self.survey_deg = np.array([self.md, self.inc_deg, self.azi_deg]).T
        self.survey_rad = np.array([self.md, self.inc_rad, self.azi_rad]).T

        self.n = n
        self.e = e
        self.tvd = tvd
        self.x = x
        self.y = y
        self.z = z
        self.vec=vec

        self._min_curve()

        # initialize errors
        self.error_model = error_model
        self.sigmaH = sigmaH
        self.sigmaL = sigmaL
        self.sigmaA = sigmaA
        self.sigmaN = sigmaN
        self.sigmaE = sigmaE
        self.sigmaV = sigmaV
        self._get_errors()

    def _min_curve(self):
        """
        Params
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
        # e, n, v = (
        #     np.array([self.x, self.y, self.z]).T - np.array([self.start_xyz])
        # ).T
        # self.n, self.e, self.tvd = (np.array([n, e, v]).T + np.array([self.start_nev])).T

    def _make_angles(self, inc, azi, deg):
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
            print("Need to add the ErrorModel module")

        try:
            error_HLA = self.sigmaH and self.sigmaL and self.sigmaA
        except:
            error_HLA = False

        try:
            error_NEV = self.sigmaN and self.sigmaE and self.sigmaV
        except:
            error_NEV = False
        
        if error_HLA and error_NEV:
            return
        elif error_HLA:
            print("Need to add the HLA_to_NEV function")
        else:
            print("Need to add the NEV_to_HLA function")    
            

def interpolate_survey(survey, x, index=0):
    """
    Interpolates a point distance x between two survey stations
    using minimum curvature.

    Params:
        survey: object
            A survey object with at least two survey stations.
        x: float
            Length along well path from indexed survey station to
            perform the interpolate at. Must be less than length
            to the next survey station.
        index: int
            The index of the survey station from which to interpolate
            from.

    Returns:
        survey: object
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
