import numpy as np
from numpy import sin, cos, tan, pi, sqrt

import welleng.error
from welleng.utils import NEV_to_HLA

class iscwsaMwd:
    def __init__(
        self,
        error,
    ):
        """
        Class using the ISCWSA MWD (Rev4) model to determine
        well bore uncertatinty.
        """
        error.__init__
        self.e = error

        self.errors = {
                "DRFR": self._DRFR(self.e.survey, mag=self.e.errors_mag['DRFR']),
                "DSFS": self._DSFS(self.e.survey, mag=self.e.errors_mag['DSFS']),
                "DSTG": self._DSTG(self.e.survey, self.e.TVD, mag=self.e.errors_mag['DSTG']),
                "ABXY_TI1S": self._ABXY_TI1S(self.e.survey, self.e.AziMag, self.e.G, self.e.Dip, mag=self.e.errors_mag['AB']),
                "ABXY_TI2S": self._ABXY_TI2S(self.e.survey, self.e.AziMag, self.e.G, self.e.Dip, mag=self.e.errors_mag['AB']),
                "ABZ": self._ABZ(self.e.survey, self.e.AziMag, self.e.G, self.e.Dip, mag=self.e.errors_mag['AB']),
                "ASXY_TI1S": self._ASXY_TI1S(self.e.survey, self.e.AziMag, self.e.Dip, mag=self.e.errors_mag['AS']),
                "_ASXY_TI2S": self._ASXY_TI2S(self.e.survey, self.e.AziMag, self.e.Dip, mag=self.e.errors_mag['AS']),
                "ASXY_TI3S": self._ASXY_TI3S(self.e.survey, self.e.AziMag, self.e.Dip, mag=self.e.errors_mag['AS']),
                "ASZ": self._ASZ(self.e.survey, self.e.AziMag, self.e.Dip, mag=self.e.errors_mag['AS']),
                "MBXY_TI1S": self._MBXY_TI1S(self.e.survey, self.e.AziMag, self.e.Dip, self.e.BTotal, mag=self.e.errors_mag['MB']),
                "MBXY_TI2S": self._MBXY_TI2S(self.e.survey, self.e.AziMag, self.e.Dip, self.e.BTotal, mag=self.e.errors_mag['MB']),
                "MBZ": self._MBZ(self.e.survey, self.e.AziMag, self.e.Dip, self.e.BTotal, mag=self.e.errors_mag['MB']),
                "MSXY_TI1S": self._MSXY_TI1S(self.e.survey, self.e.AziMag, self.e.Dip, mag=self.e.errors_mag['MS']),
                "MSXY_TI2S": self._MSXY_TI2S(self.e.survey, self.e.AziMag, self.e.Dip, mag=self.e.errors_mag['MS']),
                "MSXY_TI3S": self._MSXY_TI3S(self.e.survey, self.e.AziMag, self.e.Dip, mag=self.e.errors_mag['MS']),
                "MSZ": self._MSZ(self.e.survey, self.e.AziMag, self.e.Dip, mag=self.e.errors_mag['MS']),
                "DECG": self._DECG(self.e.survey, mag=self.e.errors_mag['DECG']),
                "DECR": self._DECR(self.e.survey, mag=self.e.errors_mag['DECR']),
                "DBHG": self._DBHG(self.e.survey, self.e.Dip, self.e.BTotal, mag=self.e.errors_mag['DBHG']),
                "DBHR": self._DBHR(self.e.survey, self.e.Dip, self.e.BTotal, mag=self.e.errors_mag['DBHR']),
                "AMIL": self._AMIL(self.e.survey, self.e.AziMag, self.e.Dip, self.e.BTotal, mag=self.e.errors_mag['AMIL']),
                "SAG": self._SAG(self.e.survey, mag=self.e.errors_mag['SAG']),
                "XYM1": self._XYM1(self.e.survey, mag=self.e.errors_mag['XYM']),
                "XYM2": self._XYM2(self.e.survey, mag=self.e.errors_mag['XYM']),
                "XYM3": self._XYM3(self.e.survey, self.e.AziTrue, mag=self.e.errors_mag['XYM']),
                "XYM4": self._XYM4(self.e.survey, self.e.AziTrue, mag=self.e.errors_mag['XYM']),            
            }

        self.cov_NEVs = np.zeros((3,3,len(self.e.survey)))
        for _, value in self.errors.items():
            self.cov_NEVs += value.cov_NEV

        self.cov_HLAs = NEV_to_HLA(self.e.survey, self.cov_NEVs)

    ### error functions ###
    def _DRFR(self, survey, mag=0.35, propagation='random', NEV=True):
        dpde = np.full((len(survey), 3), [1,0,0])
        e_DIA = dpde * mag
        
        return self.e._generate_error('DRFR', e_DIA, propagation, NEV)

    def _DSFS(self, survey, mag=0.00056, propagation='systematic', NEV=True):
        dpde = np.full((len(survey), 3), [1,0,0])
        dpde = dpde * np.array(survey)
        e_DIA = dpde * mag

        return self.e._generate_error('DSFS', e_DIA, propagation, NEV)

    def _DSTG(self, survey, TVD, mag=0.00000025, propagation='systematic', NEV=True):
        dpde = np.full((len(survey), 3), [1,0,0])
        dpde[:,0] = TVD
        dpde = dpde * np.array(survey)
        e_DIA = dpde * mag

        return self.e._generate_error('DSTG', e_DIA, propagation, NEV)

    def _ABXY_TI1S(self, survey, AziMag, gfield, dip, mag=0.0040, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = -cos(survey[:,1]) / gfield
        dpde[:,2] = (cos(survey[:,1]) * tan(dip) * sin(AziMag)) / gfield
        e_DIA = dpde * mag

        return self.e._generate_error('ABXY_TI1S', e_DIA, propagation, NEV)

    def _ABXY_TI2S(self, survey, AziMag, gfield, dip, mag=0.004, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        with np.errstate(divide='ignore', invalid='ignore'):
            dpde[:,2] = np.nan_to_num(((tan(-(survey[:,1]) + (pi/2)) - tan(dip) * cos(AziMag)) / gfield),
                                posinf = 0.0,
                                neginf = 0.0
                                )
        e_DIA = dpde * mag

        sing = np.where(survey[:,1] < self.e.VerticalIncLimit)
        if len(sing[0]) < 1:
            return self.e._generate_error('ABXY_TI2S', e_DIA, propagation, NEV)
        else: 
            e_NEV = self.e._e_NEV(e_DIA)
            N = np.array(0.5 * self.e.drdp_sing['double_delta_md'] * -sin(self.e.drdp_sing['azi2']) * mag) / gfield
            E = np.array(0.5 * self.e.drdp_sing['double_delta_md'] * cos(self.e.drdp_sing['azi2']) * mag) / gfield
            V = np.zeros_like(N)
            e_NEV_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV[sing] = e_NEV_sing[sing]

            e_NEV_star = self.e._e_NEV_star(e_DIA)
            N = np.array(0.5 * self.e.drdp_sing['delta_md'] * -sin(self.e.drdp_sing['azi2']) * mag) / gfield
            E = np.array(0.5 * self.e.drdp_sing['delta_md'] * cos(self.e.drdp_sing['azi2']) * mag) / gfield
            V = np.zeros_like(N)
            e_NEV_star_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV_star[sing] = e_NEV_star_sing[sing]

            return self.e._generate_error('ABXY_TI2S', e_DIA, propagation, NEV, e_NEV, e_NEV_star)

    def _ABZ(self, survey, AziMag, gfield, dip, mag=0.004, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = -sin(np.array(survey)[:,1]) / gfield
        dpde[:,2] = (sin(np.array(survey)[:,1]) * tan(dip) * sin(AziMag)) / gfield
        e_DIA = dpde * mag

        return self.e._generate_error('ABZ', e_DIA, propagation, NEV)

    def _ASXY_TI1S(self, survey, AziMag, dip, mag=0.0005, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = sin(np.array(survey)[:,1]) * cos(np.array(survey)[:,1]) / sqrt(2)
        dpde[:,2] = (sin(np.array(survey)[:,1]) * -tan(dip) * cos(np.array(survey)[:,1]) * sin(AziMag)) / sqrt(2)
        e_DIA = dpde * mag

        return self.e._generate_error('ASXY_TI1S', e_DIA, propagation, NEV)

    def _ASXY_TI2S(self, survey, AziMag, dip, mag=0.0005, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = sin(np.array(survey)[:,1]) * cos(np.array(survey)[:,1]) / 2
        dpde[:,2] = (sin(np.array(survey)[:,1]) * -tan(dip) * cos(np.array(survey)[:,1]) * sin(AziMag)) / 2
        e_DIA = dpde * mag

        return self.e._generate_error('ASXY_TI2S', e_DIA, propagation, NEV)

    def _ASXY_TI3S(self, survey, AziMag, dip, mag=0.0005, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = (sin(np.array(survey)[:,1]) * tan(dip) * cos(AziMag) - cos(np.array(survey)[:,1])) / 2
        e_DIA = dpde * mag

        return self.e._generate_error('ASXY_TI3S', e_DIA, propagation, NEV)

    def _ASZ(self, survey, AziMag, dip, mag=0.0005, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = -sin(np.array(survey)[:,1]) * cos(np.array(survey)[:,1])
        dpde[:,2] = (sin(np.array(survey)[:,1]) * tan(dip) * cos(np.array(survey)[:,1]) * sin(AziMag))
        e_DIA = dpde * mag

        return self.e._generate_error('ASZ', e_DIA, propagation, NEV)

    def _MBXY_TI1S(self, survey, AziMag, dip, bfield, mag=70.0, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = (-cos(np.array(survey)[:,1]) * sin(AziMag)) / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self.e._generate_error('MBXY_TI1S', e_DIA, propagation, NEV)

    def _MBXY_TI2S(self, survey, AziMag, dip, bfield, mag=70.0, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = cos(AziMag) / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self.e._generate_error('MBXY_TI2S', e_DIA, propagation, NEV)

    def _MBZ(self, survey, AziMag, dip, bfield, mag=70.0, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = (-sin(np.array(survey)[:,1]) * sin(AziMag)) / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self.e._generate_error('MBZ', e_DIA, propagation, NEV)

    def _MSXY_TI1S(self, survey, AziMag, dip, mag=0.0016, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = sin(np.array(survey)[:,1]) * sin(AziMag) * (tan(dip) * cos(np.array(survey)[:,1])
        + sin(np.array(survey)[:,1]) * cos(AziMag)) / sqrt(2)
        e_DIA = dpde * mag

        return self.e._generate_error('MSXY_TI1S', e_DIA, propagation, NEV)


    def _MSXY_TI2S(self, survey, AziMag, dip, mag=0.0016, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = sin(AziMag) * (tan(dip) * sin(np.array(survey)[:,1]) * cos(np.array(survey)[:,1])
        - cos(np.array(survey)[:,1]) * cos(np.array(survey)[:,1]) * cos(AziMag) -cos(AziMag)) / 2
        e_DIA = dpde * mag

        return self.e._generate_error('MSXY_TI2S', e_DIA, propagation, NEV)

    def _MSXY_TI3S(self, survey, AziMag, dip, mag=0.0016, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = (cos(np.array(survey)[:,1]) * cos(AziMag) * cos(AziMag)
                - cos(np.array(survey)[:,1]) * sin(AziMag) * sin(AziMag) 
                - tan(dip) * sin(np.array(survey)[:,1]) * cos(AziMag)) / 2
        e_DIA = dpde * mag

        return self.e._generate_error('MSXY_TI3S', e_DIA, propagation, NEV)

    def _MSZ(self, survey, AziMag, dip, mag=0.0016, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = -(sin(np.array(survey)[:,1]) * cos(AziMag)
        + tan(dip) * cos(np.array(survey)[:,1])) * sin(np.array(survey)[:,1]) * sin(AziMag)
        e_DIA = dpde * mag

        return self.e._generate_error('MSZ', e_DIA, propagation, NEV)

    def _DECG(self, survey, mag=0.00628, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = 1
        e_DIA = dpde * mag

        return self.e._generate_error('DECG', e_DIA, propagation, NEV)

    def _DECR(self, survey, mag=0.00175, propagation='random', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = 1
        e_DIA = dpde * mag

        return self.e._generate_error('DECR', e_DIA, propagation, NEV)

    def _DBHG(self, survey, dip, bfield, mag=np.radians(5000), propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = 1 / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self.e._generate_error('DBHG', e_DIA, propagation, NEV)

    def _DBHR(self, survey, dip, bfield, mag=np.radians(3000), propagation='random', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = 1 / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self.e._generate_error('DBHR', e_DIA, propagation, NEV)

    def _AMIL(self, survey, AziMag, dip, bfield, mag=220.0, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = -sin(np.array(survey)[:,1]) * sin(AziMag) / (bfield * cos(dip))
        e_DIA = dpde * mag

        return self.e._generate_error('AMIL', e_DIA, propagation, NEV)

    def _SAG(self, survey, mag=0.00349, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = sin(np.array(survey)[:,1])
        e_DIA = dpde * mag

        return self.e._generate_error('SAG', e_DIA, propagation, NEV)

    def _XYM1(self, survey, mag=0.00175, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = np.absolute(sin(np.array(survey)[:,1]))
        e_DIA = dpde * mag

        return self.e._generate_error('XYM1', e_DIA, propagation, NEV)

    def _XYM2(self, survey, mag=0.00175, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,2] = -1
        e_DIA = dpde * mag

        return self.e._generate_error('XYM2', e_DIA, propagation, NEV)

    def _XYM3(self, survey, AziTrue, mag=0.00175, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = np.absolute(cos(np.array(survey)[:,1])) * cos(AziTrue)
        with np.errstate(divide='ignore', invalid='ignore'):
            dpde[:,2] = np.nan_to_num(
                -(np.absolute(cos(np.array(survey)[:,1])) * sin(AziTrue)) / sin(np.array(survey)[:,1]),
                posinf = 0.0,
                neginf = 0.0
            )
        e_DIA = dpde * mag

        sing = np.where(survey[:,1] < self.e.VerticalIncLimit)
        if len(sing[0]) < 1:
            return self.e._generate_error('XYM3', e_DIA, propagation, NEV)
        else: 
            e_NEV = self.e._e_NEV(e_DIA)
            N = np.array(0.5 * self.e.drdp_sing['double_delta_md'] * mag)
            E = np.zeros(len(self.e.drdp_sing['double_delta_md']))
            V = np.zeros_like(N)
            e_NEV_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV[sing] = e_NEV_sing[sing]

            e_NEV_star = self.e._e_NEV_star(e_DIA)
            N = np.array(0.5 * self.e.drdp_sing['delta_md'] * mag)
            E = np.zeros(len(self.e.drdp_sing['delta_md']))
            V = np.zeros_like(N)
            e_NEV_star_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV_star[sing] = e_NEV_star_sing[sing]

            return self.e._generate_error('XYM3', e_DIA, propagation, NEV, e_NEV, e_NEV_star)


    def _XYM4(self, survey, AziTrue, mag=0.00175, propagation='systematic', NEV=True):
        dpde = np.zeros((len(survey), 3))
        dpde[:,1] = np.absolute(cos(np.array(survey)[:,1])) * sin(AziTrue)
        with np.errstate(divide='ignore', invalid='ignore'):
            dpde[:,2] = np.nan_to_num((np.absolute(np.cos(np.array(survey)[:,1])) * cos(AziTrue))
            / sin(np.array(survey)[:,1]),
                posinf = 0.0,
                neginf = 0.0
            )
        e_DIA = dpde * mag

        sing = np.where(survey[:,1] < self.e.VerticalIncLimit)
        if len(sing[0]) < 1:
            return self.e._generate_error('XYM4', e_DIA, propagation, NEV)
        else: 
            e_NEV = self.e._e_NEV(e_DIA)
            N = np.zeros(len(self.e.drdp_sing['double_delta_md']))
            E = np.array(0.5 * self.e.drdp_sing['double_delta_md'] * mag)
            V = np.zeros_like(N)
            e_NEV_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV[sing] = e_NEV_sing[sing]

            e_NEV_star = self.e._e_NEV_star(e_DIA)
            N = np.zeros(len(self.e.drdp_sing['delta_md']))
            E = np.array(0.5 * self.e.drdp_sing['delta_md'] * mag)
            V = np.zeros_like(N)
            e_NEV_star_sing = np.vstack((np.zeros((1,3)), np.stack((N, E, V), axis=-1), np.zeros((1,3))))
            e_NEV_star[sing] = e_NEV_star_sing[sing]

            return self.e._generate_error('XYM4', e_DIA, propagation, NEV, e_NEV, e_NEV_star)