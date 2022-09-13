from typing import Union

import numpy as np
from numpy.linalg import norm
from scipy import optimize
from scipy.signal import argrelmin
from scipy.spatial.distance import cdist

from .survey import Survey, _interpolate_survey
from .utils import NEV_to_HLA


class Clearance:
    def __init__(
        self,
        reference: Survey,
        offset: Survey,
        k: float = 3.5,
        sigma_pa: float = 0.5,
        Sm: float = 0.3,
        Rr: Union[np.ndarray, float] = 0.4572,
        Ro: Union[np.ndarray, float] = 0.3048,
        kop_depth: float = -np.inf
    ):
        """
        Initialize a welleng.clearance.Clearance object.

        Parameters
        ----------
        reference : welleng.survey.Survey object
            The current well from which other wells are referenced.
        offset : welleng.survey.Survey object
            The other well.
        k : float
            The dimensionless scaling factor that determines the probability
            of well crossing.
        sigma_pa : float
            Quantifies the 1-SD uncertainty in the projection ahead of the
            current survey station. Its value is partially correlated with
            the projection distance, determined as the current survey depth to
            the bit plus the next survey interval. The magnitude of the actual
            uncertainty also depends on the planned curvature and on the actual
            BHA performance at the wellbore attitude in the formation being
            drilled. The project-ahead uncertainty is only an approximation,
            and although it is predominantly oriented normal to the reference
            well, it is mathematically convenient to define sigma_pa as being
            the radius of a sphere.
        Sm : float
            The surface margin term increases the effective radius of the
            offset well. It accommodates small, unidentified errors and helps
            overcome one of the geometric limitations of the separation rule,
            described in the Separation-Rule Limitations section. It also
            defines the minimum acceptable slot separation during facility
            design and ensures that the separation rule will prohibit the
            activity before nominal contact between the reference and offset
            wells, even if the position uncertainty is zero.
        Rr : float
            The openhole radius of the reference borehole (in meters).
        Ro : float
            The openhole radius of the offset borehole (in meters).
        kop_depth: float
            The kick-off point (measured) depth along the well bore - the
            default value assures that the first survey station is utilized.

        References
        ----------
        Sawaryn, S. J., Wilson, H.. , Bang, J.. , Nyrnes, E.. , Sentance,
        A.. , Poedjono, B.. , Lowdon, R.. , Mitchell, I.. , Codling, J.. ,
        Clark, P. J., and W. T. Allen. "Well-Collision-Avoidance Separation
        Rule." SPE Drill & Compl 34 (2019): 01–15.
        doi: https://doi.org/10.2118/187073-PA
        """
        self.reference = reference
        self.offset = offset
        self.k = k
        self.sigma_pa = sigma_pa
        self.Sm = Sm

        self._get_kop_index(kop_depth)
        self._get_ref()
        self._get_radii(Rr, Ro)

        self.ref_nevs = self._get_nevs(self.ref)
        self.offset_nevs = self._get_nevs(self.offset)

    def _get_kop_index(self, kop_depth):
        self.kop_depth = kop_depth
        self.kop_index = np.searchsorted(
            self.reference.md, self.kop_depth, side="left"
        )

    def _get_nevs(self, survey: Survey) -> np.ndarray:
        return np.array([
            survey.n,
            survey.e,
            survey.tvd
        ]).T

    def _get_radii(self, Rr: float, Ro: float):
        """
        If well bore radius data is provided, use this, otherwise use the
        defaults values.
        """
        if self.ref.radius is not None:
            self.Rr = self.ref.radius
        else:
            self.Rr = np.full(len(self.ref.md), Rr)

        if self.offset.radius is not None:
            self.Ro = self.offset.radius
        else:
            self.Ro = np.full(len(self.offset.md), Ro)

    def _get_ref(self):
        if self.kop_index == 0:
            self.ref = self.reference
        else:
            try:
                cov_nev = self.reference.cov_nev[self.kop_index:]
                cov_hla = self.reference.cov_hla[self.kop_index:]
            except IndexError:
                cov_nev, cov_hla = (None, None)
            self.ref = Survey(
                md=self.reference.md[self.kop_index:],
                inc=self.reference.inc_rad[self.kop_index:],
                azi=self.reference.azi_grid_rad[self.kop_index:],
                n=self.reference.n[self.kop_index:],
                e=self.reference.e[self.kop_index:],
                tvd=self.reference.tvd[self.kop_index:],
                vec=self.reference.vec_nev[self.kop_index:],
                radius=self.reference.radius[self.kop_index:],
                header=self.reference.header,
                cov_nev=cov_nev,
                cov_hla=cov_hla,
                error_model=self.reference.error_model,
                start_xyz=[
                    self.reference.x[self.kop_index],
                    self.reference.y[self.kop_index],
                    self.reference.z[self.kop_index],
                ],
                start_nev=[
                    self.reference.n[self.kop_index],
                    self.reference.e[self.kop_index],
                    self.reference.tvd[self.kop_index],
                ],
                deg=self.reference.deg,
                unit=self.reference.unit,
                nev=True
            )


def get_ref_sigma(sigma1, sigma2, sigma3, kop_index):
    sigma = np.array([sigma1, sigma2, sigma3]).T
    sigma_diff = np.diff(sigma, axis=0)

    sigma_above = np.cumsum(sigma_diff[:kop_index][::-1], axis=0)[::-1]
    sigma_below = np.cumsum(sigma_diff[kop_index:], axis=0)

    sigma_new = np.vstack((sigma_above, np.array([0, 0, 0]), sigma_below))

    return sigma_new


class ISCWSA:

    def __init__(
        self,
        clearance: Clearance,
        minimize_SF: bool = True
    ):
        """
        Class to calculate the clearance between two well bores using the
        standard method documented by ISCWSA.

        See https://www.iscwsa.net/articles/standard-wellpath-revision-4-word/
        """
        # TODO rewrite to inherit Clearance.

        clearance.__init__
        self.c = clearance
        self.minimize_SF = minimize_SF

        # get closest survey station in offset well for each survey
        # station in the reference well
        self.idx = np.argmin(
            cdist(
                self.c.ref_nevs, self.c.offset_nevs
            ), axis=-1
        )

        # iterate to find closest point on offset well between
        # survey stations
        self._get_closest_points()
        self.off_nevs = self.c._get_nevs(self.off)

        # get the unit vectors and horizontal bearing between the wells
        self._get_delta_nev_vectors()

        # make the covariance matrices
        self._get_covs()

        # get the PCRs
        self._get_PCRs()

        # calculate sigmaS
        self.sigmaS = np.sqrt(self.ref_PCR ** 2 + self.off_PCR ** 2)

        # calculate combined hole radii
        self._get_calc_hole()

        # calculate Ellipse of Uncertainty Boundary
        self.eou_boundary = (
            self.c.k * np.sqrt(self.sigmaS ** 2 + self.c.sigma_pa ** 2)
        )

        # calculate distance between well bores
        self.wellbore_separation = (
            self.dist_CC_Clr.T - self.calc_hole - self.c.Sm
        )

        # calculate the Ellipse of Uncertainty Separation
        self.eou_separation = (
            self.wellbore_separation - self.eou_boundary
        )

        # calculate the Minimum Allowable Separation Distance
        self.masd = (
            self.eou_boundary + self.calc_hole + self.c.Sm
        )

        # calculate SF (renamed from ISCWSA_ACR)
        self.SF = np.stack((
            self.c.ref.md,
            np.array(
                self.wellbore_separation / self.eou_boundary,
            ).reshape(-1),
            np.zeros_like(self.c.ref.md)
        ), axis=1)

        # check for minima
        if self.minimize_SF:
            self.get_sf_mins()

        # for debugging
        # self.pc_method()

    def _get_sf_min(self, x, i, delta_md):
        if x == 0.0:
            return self.SF[i][1]
        if x == -delta_md[0]:
            return self.SF[i - 1][1]
        if x == delta_md[1]:
            return self.SF[i + 1][1]

        if x < 0:
            ii = i - 1
            xx = delta_md[0] + x
            mult = xx / delta_md[0]
        else:
            ii = i
            xx = x
            mult = xx / delta_md[1]

        node = self.c.ref.interpolate_md(
            self.c.ref.md[ii] + xx
        )

        cov_nev = (
            self.c.ref.cov_nev[ii] + (
                np.full(shape=(1, 3, 3), fill_value=mult)
                * (self.c.ref.cov_nev[ii + 1] - self.c.ref.cov_nev[ii])
            )
        ).reshape(-1, 3, 3)

        sh = self.c.ref.header
        sh.azi_reference = 'grid'

        survey = Survey(
            md=np.insert(
                self.c.ref.md[ii: ii + 2], 1, node.md
            ),
            inc=np.insert(
                self.c.ref.inc_rad[ii: ii + 2], 1, node.inc_rad
            ),
            azi=np.insert(
                self.c.ref.azi_grid_rad[ii: ii + 2], 1, node.azi_rad
            ),
            cov_nev=np.insert(
                self.c.ref.cov_nev[ii: ii + 2], 1, cov_nev, axis=0
            ),
            start_nev=self.c.ref.pos_nev[ii],
            # start_xyz=self.c.ref.pos_xyz[ii],
            deg=False
        )

        clearance = Clearance(
            survey,
            self.c.offset,
            k=self.c.k,
            sigma_pa=self.c.sigma_pa,
            Sm=0.0,
            Rr=np.insert(
                self.c.Rr[ii: ii + 2], 1, self.c.Rr[ii + 1]
            ),
            Ro=self.c.Ro,
            kop_depth=self.c.kop_depth
        )

        SF_interpolated = ISCWSA(clearance, minimize_SF=False).SF[1, 1]

        return SF_interpolated

    def get_sf_mins(self):
        """
        Method for assessing whether a minima has occurred between survey
        station SF values on the reference well and if so calculates the
        minimum SF value between stations (between the previous and next
        station relative to the identified station).
        Modifies the SF property to include the interpolated minimum SF values.
        """
        minimas = argrelmin(self.SF[:, 1])

        sf_interpolated = []

        for minima in minimas[0].tolist():
            delta_md = self.c.ref.delta_md[minima: minima + 2]
            bounds = [[-delta_md[0], delta_md[1]]]
            # x0 = (np.diff(bounds) / 2)
            x0 = [0]
            args = (minima, delta_md)
            options = {
                'eps': np.sum(delta_md) / 10
            }

            # SLSQP and L-BFGS-B don't work when using neg to pos ranges in
            # this example, but Powell seems to do the job.
            result = optimize.minimize(
                self._get_sf_min,
                x0,
                method='Powell',
                bounds=bounds,
                args=args,
                options=options
            )

            if any((
                    result.x == 0,
                    result.x in delta_md
            )):
                continue

            else:
                sf_interpolated.append((
                    self.c.ref.md[minima] + result.x[0], result.fun
                ))

        if bool(sf_interpolated):
            for md, sf in sf_interpolated:
                i = np.searchsorted(self.SF[:, 0], md, side='right')
                self.SF = np.insert(
                    self.SF, i, np.array([md, sf, 1]), axis=0
                )

    def get_lines(self):
        """
        Extracts the closest points between wells for each survey section.
        """
        points = [
            [
                c[0],
                [
                    c[1].n[1],
                    c[1].e[1],
                    c[1].tvd[1]
                ]
            ]
            for c in self.closest
        ]

        start_points = []
        end_points = []

        [(start_points.append(p[0]), end_points.append(p[1])) for p in points]

        return np.array([
            np.vstack(start_points).tolist(), np.vstack(end_points).tolist()
        ])

    def _get_closest_points(self):
        """
        Determines the point between pairs of subsequent survey stations on
        the offset well that is closest to each survey stations on the
        reference well.
        """
        closest = []
        for j, (i, station) in enumerate(zip(self.idx, self.c.ref_nevs.tolist())):
            if i > 0:
                bnds = [(0, self.c.offset.md[i] - self.c.offset.md[i - 1])]
                res_1 = optimize.minimize(
                    self._fun,
                    bnds[0][1],
                    method='SLSQP',
                    bounds=bnds,
                    args=(self.c.offset, i - 1, station)
                )
                mult = res_1.x[0] / (bnds[0][1] - bnds[0][0])
                sigma_new_1 = self._interpolate_covs(i, mult)
            else:
                res_1 = False

            if i < len(self.c.offset_nevs) - 1:
                bnds = [(0, self.c.offset.md[i + 1] - self.c.offset.md[i])]
                res_2 = optimize.minimize(
                    self._fun,
                    bnds[0][0],
                    method='SLSQP',
                    bounds=bnds,
                    args=(self.c.offset, i, station)
                )
                mult = res_2.x[0] / (bnds[0][1] - bnds[0][0])
                sigma_new_2 = self._interpolate_covs(i + 1, mult)
            else:
                res_2 = False

            if res_1 and res_2 and res_1.fun < res_2.fun or not res_2:
                closest.append((
                    station,
                    _interpolate_survey(self.c.offset, res_1.x[0], i - 1),
                    res_1, sigma_new_1
                ))
            else:
                closest.append((
                    station,
                    _interpolate_survey(self.c.offset, res_2.x[0], i),
                    res_2,
                    sigma_new_2
                ))

        self.closest = closest
        md, inc, azi, n, e, tvd, x, y, z = np.array([
            [
                r[1].md[1],
                r[1].inc_rad[1],
                r[1].azi_grid_rad[1],
                r[1].n[1],
                r[1].e[1],
                r[1].tvd[1],
                r[1].x[1],
                r[1].y[1],
                r[1].z[1],
            ]
            for r in self.closest
        ]).T

        cov_hla = np.array([
            [
                r[3][0],
            ]
            for r in self.closest
        ])

        cov_nev = np.array([
            [
                r[3][1],
            ]
            for r in self.closest
        ])

        self.off = Survey(
            md=md,
            inc=inc,
            azi=azi,
            n=n,
            e=e,
            tvd=tvd,
            header=self.c.offset.header,
            error_model=None,
            cov_hla=list(cov_hla),
            cov_nev=list(cov_nev),
            start_xyz=[x[0], y[0], z[0]],
            start_nev=[n[0], e[0], tvd[0]],
            deg=False,
            unit=self.c.offset.unit
        )

    def _interpolate_covs(self, i, mult):
        """
        Returns the interpolated covariance matrices for the interpolated
        survey points representing the closest points on the offset well
        relative to each reference well survey station.
        """
        cov_hla_new = (
            self.c.offset.cov_hla[i - 1]
            + mult * (self.c.offset.cov_hla[i] - self.c.offset.cov_hla[i - 1])
        )

        cov_nev_new = (
            self.c.offset.cov_nev[i - 1]
            + mult * (self.c.offset.cov_nev[i] - self.c.offset.cov_nev[i - 1])
        )

        return (cov_hla_new, cov_nev_new)

    def _fun(self, x, survey, index, station):
        """
        Optimization function used to find the closest point between pairs of
        offset well survey stations.
        """
        s = _interpolate_survey(survey, x[0], index)
        new_pos = np.array([s.n, s.e, s.tvd]).T[1]
        dist = norm(new_pos - station, axis=-1)

        return dist

    def _get_delta_nev_vectors(self):
        temp = self.off_nevs - self.c.ref_nevs

        # Center to Center distance between reference survey station and
        # closest point in the offset well
        self.dist_CC_Clr = norm(temp, axis=-1).reshape(-1, 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.ref_delta_nevs = np.nan_to_num(
                temp / self.dist_CC_Clr,
                posinf=0.0,
                neginf=0.0
            )

        temp = self.c.ref_nevs - self.off_nevs
        with np.errstate(divide='ignore', invalid='ignore'):
            self.off_delta_nevs = np.nan_to_num(
                temp / self.dist_CC_Clr,
                posinf=0.0,
                neginf=0.0
            )

        self.hoz_bearing = np.arctan2(
            self.ref_delta_nevs[:, 1], self.ref_delta_nevs[:, 0]
        )
        self.hoz_bearing_deg = (np.degrees(self.hoz_bearing) + 360) % 360

        self._get_delta_hla_vectors()
        self.toolface_bearing = np.arctan2(
            self.ref_delta_hlas[:, 1], self.ref_delta_hlas[:, 0]
        )
        self.toolface_bearing_deg = (
            np.degrees(self.toolface_bearing) + 360
        ) % 360

        self._traveling_cylinder()

        self.dist_CC_Clr = self.dist_CC_Clr.reshape(-1)

    def _get_delta_hla_vectors(self):
        self.ref_delta_hlas = np.vstack([
            NEV_to_HLA(s, nev, cov=False)
            for s, nev in zip(
                self.c.ref.survey_rad, self.ref_delta_nevs
            )
        ])
        self.off_delta_hlas = np.vstack([
            NEV_to_HLA(s, nev, cov=False)
            for s, nev in zip(
                self.off.survey_rad, self.off_delta_nevs
            )
        ])

    def _get_covs(self):
        self.ref_cov_hla = self.c.ref.cov_hla
        self.ref_cov_nev = self.c.ref.cov_nev
        self.off_cov_hla = self.off.cov_hla
        self.off_cov_nev = self.off.cov_nev

    def _get_PCRs(self):
        self.ref_PCR = np.hstack([
            np.sqrt(np.dot(np.dot(vec, cov), vec.T))
            for vec, cov in zip(self.ref_delta_nevs, self.ref_cov_nev)
        ])
        self.off_PCR = np.hstack([
            np.sqrt(np.dot(np.dot(vec, cov), vec.T))
            for vec, cov in zip(self.off_delta_nevs, self.off_cov_nev)
        ])

    def _get_calc_hole(self):
        self.calc_hole = self.c.Rr + self.c.Ro[self.idx]

    def _traveling_cylinder(self):
        """
        Calculates the azimuthal data for a traveling cylinder plot.
        """
        self.trav_cyl_azi_deg = (
            self.c.reference.azi_grid_deg[self.c.kop_index:]
            + self.toolface_bearing_deg + 360
        ) % 360
