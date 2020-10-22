import numpy as np
from numpy import sin, cos, tan, pi, sqrt
from welleng.utils import MinCurve, NEV_to_HLA

class ErrorModel():
    """
    """

    class Error:
        '''
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
        surface_loc = np.array([0.0, 0.0, 0.0]),    # [x, y, z]
        survey = np.array([0.0, 0.0, 0.0]),         # [md, inc, azi] in radians
        well_ref_params=dict(
            Latitude = -40,                             # degrees
            G = 9.80665,                                # m/s2
            Btotal = 61000,                             # nT
            Dip = -70,                                  # degrees
            Declination = 13,                           # degrees  
            Convergence = 0,                            # degrees
        ),
        AzimuthRef = True,
        DepthUnits = 'meters',
        VerticalIncLimit = 0.0001,                  # degrees
        FeetToMeters = 0.3048,
        errors_mag = dict(
            DRFR_mag = 0.35,
            DSFS_mag = 0.00056,
            DSTG_mag = 2.5e-7,
            AB_mag = 0.004,
            AS_mag = 0.0005,
            MB_mag = 70,
            MS_mag = 0.0016,
            DECG_mag = np.radians(0.36),
            DECR_mag = np.radians(0.1),
            DBHG_mag = np.radians(5000),
            DBHR_mag = np.radians(3000),
            AMIL_mag = 220,
            SAG_mag = np.radians(0.2),
            XYM_mag = np.radians(0.1)
        ),
        error_model="ISCWSA_MWD",
    ):

        error_models = [
            "ISCWSA_MWD"
        ]
        assert error_model in error_models, "Unrecognized error model"
        self.error_model = error_model

        p = well_ref_params

        self.surface_loc = surface_loc
        self.Latitude = np.radians(p["Latitude"])
        self.G = p["G"]
        self.Btotal = p["BTotal"]
        self.Dip = np.radians(p["Dip"])
        self.Declination = np.radians(p["Declination"])
        self.Convergence = np.radians(p["Convergence"])
        self.AzimuthRef = AzimuthRef
        self.DepthUnits = DepthUnits
        self.VerticalIncLimit = np.radians(VerticalIncLimit)
        self.FeetToMeters = FeetToMeters
        self.errors_mag = errors_mag

        self.survey = np.array(survey)
        self.IncRad, self.AziRad = np.radians(self.survey[:,1:]).T
        self.AziTrue = self.AziRad + np.full(len(self.AziRad), self.Convergence)
        self.AziG = self.AziTrue - np.full(len(self.AziRad), self.Convergence)
        self.AziMag = self.AziTrue - np.full(len(self.AziRad), self.Declination)

        self.survey = np.stack((
            self.survey[:,0],
            self.IncRad,
            self.AziTrue
        ), axis = -1)
        self.survey_drdp = self.survey

        self.POS = [np.array(self.surface_loc)]
        self.DLS = [np.array(0)]
        md, inc, azi = self.survey_drdp.T
        self.mc = MinCurve(
            md=md,
            inc=inc,
            azi=azi,
            start_xyz=self.surface_loc,
            unit=self.DepthUnits
            )
        # self.pos, self.dls = min_curve(self.survey_drdp, degrees=False)

        # self.TVD = np.around(np.array(self.pos[:,2]), 2)
        self.TVD = self.mc.poss[:,2]

        self.drdp = self._drdp(self.survey_drdp)
        self.drdp_sing = self._drdp_sing(self.survey_drdp)

        self.errors = {}
        self.cov_NEVs = np.zeros((3,3,len(self.survey)))

        if self.error_model == "ISCWSA_MWD":
            self.iscwsa_mwd()


    def _e_NEV(self, e_DIA):
        D, I, A = e_DIA.T
        arr = np.array([
            (self.drdp[:,0] + self.drdp[:,9]) * D
            + (self.drdp[:,3] + self.drdp[:,12]) * I
            + (self.drdp[:,6] + self.drdp[:,15]) * A,

            (self.drdp[:,1] + self.drdp[:,10]) * D
            + (self.drdp[:,4] + self.drdp[:,13]) * I
            + (self.drdp[:,7] + self.drdp[:,16]) * A,

            (self.drdp[:,2] + self.drdp[:,11]) * D
            + (self.drdp[:,5] + self.drdp[:,14]) * I
            + (self.drdp[:,8] + self.drdp[:,17]) * A,
        ]).T

        arr[0] = 0

        return arr

    def _e_NEV_star(self, e_DIA):
        D, I, A = e_DIA.T
        arr = np.array([
            self.drdp[:,0] * D
            + self.drdp[:,3] * I
            + self.drdp[:,6] * A,

            self.drdp[:,1] * D
            + self.drdp[:,4] * I
            + self.drdp[:,7] * A,

            self.drdp[:,2] * D
            + self.drdp[:,5] * I
            + self.drdp[:,8] * A
        ]).T

        arr[0] = 0

        return arr

    def _cov(self, arr):
        '''
        Returns a covariance matrix from an n * 3 array.
        '''
        x, y, z = np.array(arr).T
        return np.array([
            [x*x, x*y, x*z],
            [y*x, y*y, y*z],
            [z*x, z*y, z*z]
        ])

    def _sigma_e_NEV_systematic(self, e_NEV, e_NEV_star):
        return e_NEV_star + np.vstack((np.zeros((1,3)), np.cumsum(e_NEV, axis=0)[:-1]))

    def _generate_error(self, code, e_DIA, propagation='systematic', NEV=True, e_NEV=None, e_NEV_star=None):
        if not NEV:
            return e_DIA
        else:   
            cov_DIA = self._cov(e_DIA)
            if e_NEV is None:
                e_NEV = self._e_NEV(e_DIA)
                e_NEV_star = self._e_NEV_star(e_DIA)
            if propagation == 'systematic':
                sigma_e_NEV = self._sigma_e_NEV_systematic(e_NEV, e_NEV_star)
                cov_NEV = self._cov(sigma_e_NEV)
            elif propagation == 'random':
                sigma_e_NEV = np.cumsum(self._cov(e_NEV), axis=-1)
                cov_NEV = np.add(
                    self._cov(e_NEV_star),
                    np.concatenate((np.array(np.zeros((3,3,1))), np.array(sigma_e_NEV[:,:,:-1])), axis=-1)
                    )
            else:
                return

            return ErrorModel.Error(code, propagation, e_DIA, cov_DIA, e_NEV, e_NEV_star, sigma_e_NEV, cov_NEV)
            # return dict(
            #     code = code,
            #     cov_NEV = cov_NEV
            # )
    
    def _DRFR(self, survey, mag=0.35, propagation='random', NEV=True):
        dpde = np.full((len(survey), 3), [1,0,0])
        e_DIA = dpde * mag
        
        return self._generate_error('DRFR', e_DIA, propagation, NEV)

    def _DSFS(self, survey, mag=0.00056, propagation='systematic', NEV=True):
        dpde = np.full((len(survey), 3), [1,0,0])
        dpde = dpde * np.array(survey)
        e_DIA = dpde * mag

        return self._generate_error('DSFS', e_DIA, propagation, NEV)

    def _DSTG(self, survey, TVD, mag=0.00000025, propagation='systematic', NEV=True):
        dpde = np.full((len(survey), 3), [1,0,0])
        dpde[:,0] = TVD
        dpde = dpde * np.array(survey)
        e_DIA = dpde * mag

        return self._generate_error('DSTG', e_DIA, propagation, NEV)

    def _ABXY_TI1S(self, survey, AziMag, gfield, dip, mag=0.0040, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = -cos(survey[:,1]) / gfield
        dpde[:,2] = (cos(survey[:,1]) * tan(dip) * sin(AziMag)) / gfield
        e_DIA = dpde * mag

        return self._generate_error('ABXY_TI1S', e_DIA, propagation, NEV)

    def _ABXY_TI2S(self, survey, AziMag, gfield, dip, mag=0.004, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        with np.errstate(divide='ignore', invalid='ignore'):
            dpde[:,2] = np.nan_to_num(((tan(-(survey[:,1]) + (pi/2)) - tan(dip) * cos(AziMag)) / gfield),
                                posinf = 0.0,
                                neginf = 0.0
                                )
            # dpde[:,2] = ((tan(-(survey[:,1]) + (pi/2)) - tan(dip) * cos(AziMag)) / gfield)
        # dpde[:,2][np.where(np.array(survey)[:,1] < self.VerticalIncLimit)] = 0
        e_DIA = dpde * mag

        sing = np.where(self.survey[:,1] < self.VerticalIncLimit)
        if len(sing[0]) < 1:
            return self._generate_error('ABXY_TI2S', e_DIA, propagation, NEV)
        else: 
            e_NEV = self._e_NEV(e_DIA)
            N = np.array(0.5 * self.drdp_sing['double_delta_md'] * -sin(self.drdp_sing['azi2']) * mag) / gfield
            E = np.array(0.5 * self.drdp_sing['double_delta_md'] * cos(self.drdp_sing['azi2']) * mag) / gfield
            V = np.zeros_like(N)
            e_NEV_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV[sing] = e_NEV_sing[sing]

            e_NEV_star = self._e_NEV_star(e_DIA)
            N = np.array(0.5 * self.drdp_sing['delta_md'] * -sin(self.drdp_sing['azi2']) * mag) / gfield
            E = np.array(0.5 * self.drdp_sing['delta_md'] * cos(self.drdp_sing['azi2']) * mag) / gfield
            V = np.zeros_like(N)
            e_NEV_star_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV_star[sing] = e_NEV_star_sing[sing]

            return self._generate_error('ABXY_TI2S', e_DIA, propagation, NEV, e_NEV, e_NEV_star)

    def _ABZ(self, survey, AziMag, gfield, dip, mag=0.004, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = -sin(np.array(survey)[:,1]) / gfield
        dpde[:,2] = (sin(np.array(survey)[:,1]) * tan(dip) * sin(AziMag)) / gfield
        e_DIA = dpde * mag

        return self._generate_error('ABZ', e_DIA, propagation, NEV)

    def _ASXY_TI1S(self, survey, AziMag, dip, mag=0.0005, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = sin(np.array(survey)[:,1]) * cos(np.array(survey)[:,1]) / sqrt(2)
        dpde[:,2] = (sin(np.array(survey)[:,1]) * -tan(dip) * cos(np.array(survey)[:,1]) * sin(AziMag)) / sqrt(2)
        e_DIA = dpde * mag

        return self._generate_error('ASXY_TI1S', e_DIA, propagation, NEV)

    def _ASXY_TI2S(self, survey, AziMag, dip, mag=0.0005, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = sin(np.array(survey)[:,1]) * cos(np.array(survey)[:,1]) / 2
        dpde[:,2] = (sin(np.array(survey)[:,1]) * -tan(dip) * cos(np.array(survey)[:,1]) * sin(AziMag)) / 2
        e_DIA = dpde * mag

        return self._generate_error('ASXY_TI2S', e_DIA, propagation, NEV)

    def _ASXY_TI3S(self, survey, AziMag, dip, mag=0.0005, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = (sin(np.array(survey)[:,1]) * tan(dip) * cos(AziMag) - cos(np.array(survey)[:,1])) / 2
        e_DIA = dpde * mag

        return self._generate_error('ASXY_TI3S', e_DIA, propagation, NEV)

    def _ASZ(self, survey, AziMag, dip, mag=0.0005, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = -sin(np.array(survey)[:,1]) * cos(np.array(survey)[:,1])
        dpde[:,2] = (sin(np.array(survey)[:,1]) * tan(dip) * cos(np.array(survey)[:,1]) * sin(AziMag))
        e_DIA = dpde * mag

        return self._generate_error('ASZ', e_DIA, propagation, NEV)

    def _MBXY_TI1S(self, survey, AziMag, dip, bfield, mag=70.0, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = (-cos(np.array(survey)[:,1]) * sin(AziMag)) / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self._generate_error('MBXY_TI1S', e_DIA, propagation, NEV)

    def _MBXY_TI2S(self, survey, AziMag, dip, bfield, mag=70.0, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = cos(AziMag) / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self._generate_error('MBXY_TI2S', e_DIA, propagation, NEV)

    def _MBZ(self, survey, AziMag, dip, bfield, mag=70.0, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = (-sin(np.array(survey)[:,1]) * sin(AziMag)) / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self._generate_error('MBZ', e_DIA, propagation, NEV)

    def _MSXY_TI1S(self, survey, AziMag, dip, mag=0.0016, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = sin(np.array(survey)[:,1]) * sin(AziMag) * (tan(dip) * cos(np.array(survey)[:,1])
        + sin(np.array(survey)[:,1]) * cos(AziMag)) / sqrt(2)
        e_DIA = dpde * mag

        return self._generate_error('MSXY_TI1S', e_DIA, propagation, NEV)


    def _MSXY_TI2S(self, survey, AziMag, dip, mag=0.0016, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = sin(AziMag) * (tan(dip) * sin(np.array(survey)[:,1]) * cos(np.array(survey)[:,1])
        - cos(np.array(survey)[:,1]) * cos(np.array(survey)[:,1]) * cos(AziMag) -cos(AziMag)) / 2
        e_DIA = dpde * mag

        return self._generate_error('MSXY_TI2S', e_DIA, propagation, NEV)

    def _MSXY_TI3S(self, survey, AziMag, dip, mag=0.0016, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = (cos(np.array(survey)[:,1]) * cos(AziMag) * cos(AziMag)
                - cos(np.array(survey)[:,1]) * sin(AziMag) * sin(AziMag) 
                - tan(dip) * sin(np.array(survey)[:,1]) * cos(AziMag)) / 2
        e_DIA = dpde * mag

        return self._generate_error('MSXY_TI3S', e_DIA, propagation, NEV)

    def _MSZ(self, survey, AziMag, dip, mag=0.0016, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = -(sin(np.array(survey)[:,1]) * cos(AziMag)
        + tan(dip) * cos(np.array(survey)[:,1])) * sin(np.array(survey)[:,1]) * sin(AziMag)
        e_DIA = dpde * mag

        return self._generate_error('MSZ', e_DIA, propagation, NEV)

    def _DECG(self, survey, mag=0.00628, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = 1
        e_DIA = dpde * mag

        return self._generate_error('DECG', e_DIA, propagation, NEV)

    def _DECR(self, survey, mag=0.00175, propagation='random', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = 1
        e_DIA = dpde * mag

        return self._generate_error('DECR', e_DIA, propagation, NEV)

    def _DBHG(self, survey, dip, bfield, mag=np.radians(5000), propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = 1 / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self._generate_error('DBHG', e_DIA, propagation, NEV)

    def _DBHR(self, survey, dip, bfield, mag=np.radians(3000), propagation='random', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = 1 / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self._generate_error('DBHR', e_DIA, propagation, NEV)

    def _AMIL(self, survey, AziMag, dip, bfield, mag=220.0, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = -sin(np.array(survey)[:,1]) * sin(AziMag) / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self._generate_error('AMIL', e_DIA, propagation, NEV)

    def _SAG(self, survey, mag=0.00349, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = sin(np.array(survey)[:,1])
        e_DIA = dpde * mag

        return self._generate_error('SAG', e_DIA, propagation, NEV)

    def _XYM1(self, survey, mag=0.00175, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = np.absolute(sin(np.array(survey)[:,1]))
        e_DIA = dpde * mag

        return self._generate_error('XYM1', e_DIA, propagation, NEV)

    def _XYM2(self, survey, mag=0.00175, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = -1
        e_DIA = dpde * mag

        return self._generate_error('XYM2', e_DIA, propagation, NEV)

    def _XYM3(self, survey, AziTrue, mag=0.00175, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        # e_DIA = np.zeros_like(survey)
        # e_DIA[:,1] = np.absolute(np.cos(survey[:,1])) * np.cos(AziTrue)
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     e_DIA[:,2] = np.nan_to_num(
        #         -(np.absolute(np.cos(survey[:,1])) * np.sin(AziTrue)) / np.sin(survey[:,1]),
        #         posinf = 0.0,
        #         neginf = 0.0
        #     )
        # e_DIA = e_DIA * mag

        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = np.absolute(cos(np.array(survey)[:,1])) * cos(AziTrue)
        with np.errstate(divide='ignore', invalid='ignore'):
            dpde[:,2] = np.nan_to_num(
                -(np.absolute(cos(np.array(survey)[:,1])) * sin(AziTrue)) / sin(np.array(survey)[:,1]),
                posinf = 0.0,
                neginf = 0.0
            )
        e_DIA = dpde * mag

        sing = np.where(self.survey[:,1] < self.VerticalIncLimit)
        if len(sing[0]) < 1:
            return self._generate_error('XYM3', e_DIA, propagation, NEV)
        else: 
            e_NEV = self._e_NEV(e_DIA)
            N = np.array(0.5 * self.drdp_sing['double_delta_md'] * mag)
            E = np.zeros(len(self.drdp_sing['double_delta_md']))
            V = np.zeros_like(N)
            e_NEV_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV[sing] = e_NEV_sing[sing]

            e_NEV_star = self._e_NEV_star(e_DIA)
            N = np.array(0.5 * self.drdp_sing['delta_md'] * mag)
            E = np.zeros(len(self.drdp_sing['delta_md']))
            V = np.zeros_like(N)
            e_NEV_star_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV_star[sing] = e_NEV_star_sing[sing]

            return self._generate_error('XYM3', e_DIA, propagation, NEV, e_NEV, e_NEV_star)

        # return self._generate_error('XYM3', e_DIA, propagation, NEV)

    def _XYM4(self, survey, AziTrue, mag=0.00175, propagation='systematic', NEV=True):
        # mag = np.array([mag])
        # if not NEV:
        #     return e_DIA
        # else:  
            
            # e_DIA = np.zeros_like(survey)
            # e_DIA[:,1] = np.absolute(np.cos(survey[:,1])) * np.sin(AziTrue)
            # with np.errstate(divide='ignore', invalid='ignore'):
            #     e_DIA[:,2] =  np.nan_to_num((np.absolute(np.cos(survey[:,1])) * np.cos(AziTrue)) / np.sin(survey[:,1]),
            #                             posinf = 0.0,
            #                             neginf = -0.0
            #                         )

        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = np.absolute(cos(np.array(survey)[:,1])) * sin(AziTrue)
        with np.errstate(divide='ignore', invalid='ignore'):
            dpde[:,2] = np.nan_to_num((np.absolute(np.cos(np.array(survey)[:,1])) * cos(AziTrue))
            / sin(np.array(survey)[:,1]),
                posinf = 0.0,
                neginf = 0.0
            )
        e_DIA = dpde * mag

        sing = np.where(self.survey[:,1] < self.VerticalIncLimit)
        if len(sing[0]) < 1:
            return self._generate_error('XYM4', e_DIA, propagation, NEV)
        else: 
            e_NEV = self._e_NEV(e_DIA)
            N = np.zeros(len(self.drdp_sing['double_delta_md']))
            E = np.array(0.5 * self.drdp_sing['double_delta_md'] * mag)
            V = np.zeros_like(N)
            e_NEV_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV[sing] = e_NEV_sing[sing]

            e_NEV_star = self._e_NEV_star(e_DIA)
            N = np.zeros(len(self.drdp_sing['delta_md']))
            E = np.array(0.5 * self.drdp_sing['delta_md'] * mag)
            V = np.zeros_like(N)
            e_NEV_star_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV_star[sing] = e_NEV_star_sing[sing]

            return self._generate_error('XYM4', e_DIA, propagation, NEV, e_NEV, e_NEV_star)

    def iscwsa_mwd(self):
        # self.errors.append(
        #     (
        #         self._DRFR(self.survey, mag=self.errors_mag['DRFR_mag']),
        #         self._DSFS(self.survey, mag=self.errors_mag['DSFS_mag']),
        #         self._DSTG(self.survey, self.TVD, mag=self.errors_mag['DSTG_mag']),
        #         self._ABXY_TI1S(self.survey, self.AziMag, self.G, self.Dip, mag=self.errors_mag['AB_mag']),
        #         self._ABXY_TI2S(self.survey, self.AziMag, self.G, self.Dip, mag=self.errors_mag['AB_mag']),
        #         self._ABZ(self.survey, self.AziMag, self.G, self.Dip, mag=self.errors_mag['AB_mag']),
        #         self._ASXY_TI1S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['AS_mag']),
        #         self._ASXY_TI2S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['AS_mag']),
        #         self._ASXY_TI3S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['AS_mag']),
        #         self._ASZ(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['AS_mag']),
        #         self._MBXY_TI1S(self.survey, self.AziMag, self.Dip, self.Btotal, mag=self.errors_mag['MB_mag']),
        #         self._MBXY_TI2S(self.survey, self.AziMag, self.Dip, self.Btotal, mag=self.errors_mag['MB_mag']),
        #         self._MBZ(self.survey, self.AziMag, self.Dip, self.Btotal, mag=self.errors_mag['MB_mag']),
        #         self._MSXY_TI1S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['MS_mag']),
        #         self._MSXY_TI2S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['MS_mag']),
        #         self._MSXY_TI3S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['MS_mag']),
        #         self._MSZ(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['MS_mag']),
        #         self._DECG(self.survey, mag=self.errors_mag['DECG_mag']),
        #         self._DECR(self.survey, mag=self.errors_mag['DECR_mag']),
        #         self._DBHG(self.survey, self.Dip, self.Btotal, mag=self.errors_mag['DBHG_mag']),
        #         self._DBHR(self.survey, self.Dip, self.Btotal, mag=self.errors_mag['DBHR_mag']),
        #         self._AMIL(self.survey, self.AziMag, self.Dip, self.Btotal, mag=self.errors_mag['AMIL_mag']),
        #         self._SAG(self.survey, mag=self.errors_mag['SAG_mag']),
        #         self._XYM1(self.survey, mag=self.errors_mag['XYM_mag']),
        #         self._XYM2(self.survey, mag=self.errors_mag['XYM_mag']),
        #         self._XYM3(self.survey, self.AziTrue, mag=self.errors_mag['XYM_mag']),
        #         self._XYM4(self.survey, self.AziTrue, mag=self.errors_mag['XYM_mag']),
        #     )
            
        # )
        self.errors = {
            "DRFR": self._DRFR(self.survey, mag=self.errors_mag['DRFR_mag']),
            "DSFS": self._DSFS(self.survey, mag=self.errors_mag['DSFS_mag']),
            "DSTG": self._DSTG(self.survey, self.TVD, mag=self.errors_mag['DSTG_mag']),
            "ABXY_TI1S": self._ABXY_TI1S(self.survey, self.AziMag, self.G, self.Dip, mag=self.errors_mag['AB_mag']),
            "ABXY_TI2S": self._ABXY_TI2S(self.survey, self.AziMag, self.G, self.Dip, mag=self.errors_mag['AB_mag']),
            "ABZ": self._ABZ(self.survey, self.AziMag, self.G, self.Dip, mag=self.errors_mag['AB_mag']),
            "ASXY_TI1S": self._ASXY_TI1S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['AS_mag']),
            "_ASXY_TI2S": self._ASXY_TI2S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['AS_mag']),
            "ASXY_TI3S": self._ASXY_TI3S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['AS_mag']),
            "ASZ": self._ASZ(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['AS_mag']),
            "MBXY_TI1S": self._MBXY_TI1S(self.survey, self.AziMag, self.Dip, self.Btotal, mag=self.errors_mag['MB_mag']),
            "MBXY_TI2S": self._MBXY_TI2S(self.survey, self.AziMag, self.Dip, self.Btotal, mag=self.errors_mag['MB_mag']),
            "MBZ": self._MBZ(self.survey, self.AziMag, self.Dip, self.Btotal, mag=self.errors_mag['MB_mag']),
            "MSXY_TI1S": self._MSXY_TI1S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['MS_mag']),
            "MSXY_TI2S": self._MSXY_TI2S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['MS_mag']),
            "MSXY_TI3S": self._MSXY_TI3S(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['MS_mag']),
            "MSZ": self._MSZ(self.survey, self.AziMag, self.Dip, mag=self.errors_mag['MS_mag']),
            "DECG": self._DECG(self.survey, mag=self.errors_mag['DECG_mag']),
            "DECR": self._DECR(self.survey, mag=self.errors_mag['DECR_mag']),
            "DBHG": self._DBHG(self.survey, self.Dip, self.Btotal, mag=self.errors_mag['DBHG_mag']),
            "DBHR": self._DBHR(self.survey, self.Dip, self.Btotal, mag=self.errors_mag['DBHR_mag']),
            "AMIL": self._AMIL(self.survey, self.AziMag, self.Dip, self.Btotal, mag=self.errors_mag['AMIL_mag']),
            "SAG": self._SAG(self.survey, mag=self.errors_mag['SAG_mag']),
            "XYM1": self._XYM1(self.survey, mag=self.errors_mag['XYM_mag']),
            "XYM2": self._XYM2(self.survey, mag=self.errors_mag['XYM_mag']),
            "XYM3": self._XYM3(self.survey, self.AziTrue, mag=self.errors_mag['XYM_mag']),
            "XYM4": self._XYM4(self.survey, self.AziTrue, mag=self.errors_mag['XYM_mag']),            
        }
        
        # self.errors = self.errors[0]
        # self.delta_cov_NEVs = np.zeros_like(self.cov_NEVs)

        for key, value in self.errors.items():
            self.cov_NEVs += value.cov_NEV
        # for error in self.errors:
        #     self.cov_NEVs += error.cov_NEV
        # self.delta_cov_NEVs[:,:,1:] = ((self.cov_NEVs[:,:,1:] - self.cov_NEVs[:,:,:-1]))

        self.cov_HLAs = NEV_to_HLA(self.survey, self.cov_NEVs)



    def drk_dDepth(self, survey):
        '''
        survey1 is previous survey station (with inc and azi in radians)
        survey2 is current survey station (with inc and azi in radians)
        '''
        md1, inc1, azi1 = np.array(survey[:-1]).T
        md2, inc2, azi2 = np.array(survey[1:]).T
        # delta_md = md2 - md1
        delta_md = 1

        dogleg = np.arccos(
            np.cos(inc2 - inc1)
            - (np.sin(inc1) * np.sin(inc2)) * (1 - np.cos(azi2 - azi1))
        )
        
        # manage discontinuity
        with np.errstate(divide='ignore', invalid='ignore'):
            rf = np.ones_like(dogleg)
            rf = np.where(dogleg == 0, rf, 2 / dogleg * np.tan(dogleg / 2))
        
        N = np.array(
            delta_md / 2 * (
            np.sin(inc1) * np.cos(azi1)
            + np.sin(inc2) * np.cos(azi2)
            ) * rf
        )
                    
        E = np.array(
            delta_md / 2 * (
            np.sin(inc1) * np.sin(azi1)
            + np.sin(inc2) *np.sin(azi2)
            ) * rf
        )
        
        V = np.array(
            delta_md / 2 * (
            np.cos(inc1) + np.cos(inc2)) * rf
        )

        return np.vstack((np.array(np.zeros((1,3))), np.stack((N, E, V), axis=-1)))

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
        V = np.array(0.5 * (-(md2 - md1) * np.sin(inc2)))
        
        return np.vstack((np.array(np.zeros((1,3))), np.stack((N, E, V), axis=-1)))

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
        
        return np.vstack((np.array(np.zeros((1,3))), np.stack((N, E, V), axis=-1)))

    def drkplus1_dDepth(self, survey):
        '''
        survey2 is current survey station (with inc and azi in radians)
        survey3 is next survey station (with inc and azi in radians)
        '''
        return np.vstack((self.drk_dDepth(survey)[1:] * -1, np.array(np.zeros((1,3)))))

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
        
        return np.vstack((np.stack((N, E, V), axis=-1), np.array(np.zeros((1,3)))))

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
        
        return np.vstack((np.stack((N, E, V), axis=-1), np.array(np.zeros((1,3)))))

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
        
        # N = np.array(0.5 * delta_md)
        # E = np.array(0.5 * delta_md)
        # V = np.zeros_like(N)
        
        # return np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))

        return dict(
            double_delta_md = double_delta_md,
            delta_md = delta_md,
            azi2 = azi2
        )

    # def _NEV_to_HLA(self, survey, cov_NEV):
    #     inc = np.array(survey[:,1])
    #     azi = np.array(survey[:,2])

    #     trans = np.array([
    #         [cos(inc) * cos(azi), -sin(azi), sin(inc) * cos(azi)],
    #         [cos(inc) * sin(azi), cos(azi), sin(inc) * sin(azi)],
    #         [-sin(inc), np.zeros_like(inc), cos(inc)]
    #     ]).T

    #     cov_HLAs = [
    #         np.dot(np.dot(mat, cov_NEV.T[i]), mat.T) for i, mat in enumerate(trans)
    #         ]

    #     # cov_HLAs = []

    #     # for i, mat in enumerate(trans):
    #     #     cov_HLAs.append(
    #     #         np.dot(np.dot(mat, cov_NEV.T[i]), mat.T)
    #     #     )

    #     return np.vstack(cov_HLAs).reshape(-1,3,3).T

    # def _HLA_to_NEV(self, survey, cov_HLA):
    #     inc = np.array(survey[:,1])
    #     azi = np.array(survey[:,2])

    #     trans = np.array([
    #         [cos(inc) * cos(azi), -sin(azi), sin(inc) * cos(azi)],
    #         [cos(inc) * sin(azi), cos(azi), sin(inc) * sin(azi)],
    #         [-sin(inc), np.zeros_like(inc), cos(inc)]
    #     ]).T

    #     cov_NEVs = [
    #         np.dot(np.dot(mat, cov_NEV.T[i]), mat.T) for i, mat in enumerate(trans)
    #         ]

        

    #     return np.vstack(cov_HLAs).reshape(-1,3,3).T


