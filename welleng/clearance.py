import numpy as np
from numpy.linalg import norm
import trimesh

from scipy import optimize
from scipy.spatial.distance import cdist

from welleng.survey import Survey, interpolate_survey, make_cov
from welleng.utils import NEV_to_HLA, HLA_to_NEV
from welleng.mesh import WellMesh

class Clearance:
    def __init__(
        self,
        reference,
        offset,
        k=3.5,
        sigma_pa=0.5,
        Sm=0.3,
        Rr=0.4572,
        Ro=0.3048,
        kop_depth=0,
        # N_verts=12,
    ):
        """
        Initialize a welleng Clearance object.

        Parameters
        ----------

            method: str (default="ISCWSA")
                The method used to calculate the clearance.
                Either "ISCWSA", "mesh_ellipse" or "mesh_pedal_curve".

        reference: object

        """
        self.reference = reference
        self.offset = offset
        self.k = k
        self.sigma_pa = sigma_pa
        self.Sm = Sm
        # self.N_verts = N_verts
        
        self._get_kop_index(kop_depth)
        self._get_ref()
        self._get_radii(Rr, Ro)

        self.ref_nevs = self._get_nevs(self.ref)
        self.offset_nevs = self._get_nevs(self.offset)

    def _get_kop_index(self, kop_depth):
        self.kop_depth = kop_depth
        self.kop_index = np.searchsorted(self.reference.md, self.kop_depth, side="left")

    def _get_nevs(self, survey):
        return np.array([
            survey.n,
            survey.e,
            survey.tvd
        ]).T

    def _get_radii(self, Rr, Ro):
        """
        If well bore radius data is provided, use this, otherwise use the
        defaults values.
        """
        if self.ref.radius:
            self.Rr = self.ref.radius
        else:
            self.Rr = np.full(len(self.ref.md), Rr)

        if self.offset.radius:
            self.Ro = self.offset.radius
        else:
            self.Ro = np.full(len(self.offset.md), Ro)

    def _get_ref(self):
        if self.kop_index == 0:
            self.ref = self.reference
        else:
            # H, L, A = get_ref_sigma(
            #     self.reference.sigmaH,
            #     self.reference.sigmaL,
            #     self.reference.sigmaA,
            #     self.kop_index
            # ).T
            self.ref = Survey(
                md=self.reference.md[self.kop_index:],
                inc=self.reference.inc_rad[self.kop_index:],
                azi=self.reference.azi_rad[self.kop_index:],
                n=self.reference.n[self.kop_index:],
                e=self.reference.e[self.kop_index:],
                tvd=self.reference.tvd[self.kop_index:],
                vec=self.reference.vec,
                radius=self.reference.radius,
                well_ref_params=self.reference.well_ref_params,
                error_model=self.reference.error_model,
                # sigmaH=H[self.kop_index:],
                # sigmaL=L[self.kop_index:],
                # sigmaA=A[self.kop_index:],
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
                unit=self.reference.unit
            )

def get_ref_sigma(sigma1, sigma2, sigma3, kop_index):
    sigma = np.array([sigma1, sigma2, sigma3]).T
    sigma_diff = np.diff(sigma, axis=0)

    sigma_above = np.cumsum(sigma_diff[:kop_index][::-1], axis=0)[::-1]
    sigma_below = np.cumsum(sigma_diff[kop_index:], axis=0)

    sigma_new = np.vstack((sigma_above, np.array([0,0,0]), sigma_below))

    return sigma_new

class ISCWSA:
    def __init__(
        self,
        clearance,
    ):
        """
        Class to calculate the clearance between two well bores using the
        standard method documented by ISCWSA.
        """
        Clearance.__init__
        self.c = clearance
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

        # transform to HLA coordinates
        self._get_delta_hla_vectors()

        # make the covariance matrices
        self._get_covs()

        # get the PCRs
        self._get_PCRs()

        # calculate sigmaS
        self.sigmaS = np.sqrt(self.ref_PCR ** 2 + self.off_PCR ** 2)

        # calculate combined hole radii
        self._get_calc_hole()

        # calculate SF
        self.ISCWSA_ACR = np.around(
            (self.dist_CC_Clr.T - self.calc_hole - self.c.Sm)
            / (self.c.k * np.sqrt(self.sigmaS ** 2 + self.c.sigma_pa ** 2)),
            decimals=2
            ).reshape(-1)

        # for debugging
        # self.pc_method()

    def _get_closest_points(self):
        closest = []
        for j, (i, station) in enumerate(zip(self.idx, self.c.ref_nevs.tolist())):
            if i > 0:
                bnds = [(0, self.c.offset.md[i] - self.c.offset.md[i - 1])]
                res_1 = optimize.minimize(
                    self._fun,
                    bnds[0][1],
                    method='SLSQP',
                    bounds=bnds,
                    args=(self.c.offset, i-1, station)
                    )
                mult = res_1.x[0] / (bnds[0][1] - bnds[0][0])
                sigma_new_1 = self._interpolate_covs(i, mult)
            else: res_1 = False

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
            else: res_2 = False

            if res_1 and res_2 and res_1.fun < res_2.fun or not res_2:
                closest.append((station, interpolate_survey(self.c.offset, res_1.x[0], i - 1), res_1, sigma_new_1))
            else:
                closest.append((station, interpolate_survey(self.c.offset, res_2.x[0], i), res_2, sigma_new_2))

        self.closest = closest
        md, inc, azi, n, e, tvd, x, y, z,  = np.array([
            [
                r[1].md[1],
                r[1].inc_rad[1],
                r[1].azi_rad[1],
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
            well_ref_params=self.c.offset.well_ref_params,
            error_model=None,
            cov_hla=cov_hla,
            cov_nev=cov_nev,
            # sigmaH=sigmaH,
            # sigmaL=sigmaL,
            # sigmaA=sigmaA,
            # sigmaN=sigmaN,
            # sigmaE=sigmaE,
            # sigmaV=sigmaV,
            start_xyz=[x[0],y[0],z[0]],
            start_nev=[n[0],e[0],tvd[0]],
            deg=False,
            unit=self.c.offset.unit
        )

    def _interpolate_covs(self, i, mult):
        cov_hla_new = (
            self.c.offset.cov_hla[i - 1]
            + mult * (self.c.offset.cov_hla[i] - self.c.offset.cov_hla[i-1])
        )

        cov_nev_new = (
            self.c.offset.cov_nev[i - 1]
            + mult * (self.c.offset.cov_nev[i] - self.c.offset.cov_nev[i-1])
        )
        # sigmaH_new = (
        #     self.c.offset.sigmaH[i - 1]
        #     + mult * (self.c.offset.sigmaH[i] - self.c.offset.sigmaH[i - 1])
        # )
        # sigmaL_new = (
        #     self.c.offset.sigmaL[i - 1]
        #     + mult * (self.c.offset.sigmaL[i] - self.c.offset.sigmaL[i - 1])
        # )
        # sigmaA_new = (
        #     self.c.offset.sigmaA[i - 1]
        #     + mult * (self.c.offset.sigmaA[i] - self.c.offset.sigmaA[i - 1])
        # )
        # sigmaN_new = (
        #     self.c.offset.sigmaN[i - 1]
        #     + mult * (self.c.offset.sigmaN[i] - self.c.offset.sigmaN[i - 1])
        # )
        # sigmaE_new = (
        #     self.c.offset.sigmaE[i - 1]
        #     + mult * (self.c.offset.sigmaE[i] - self.c.offset.sigmaE[i - 1])
        # )
        # sigmaV_new = (
        #     self.c.offset.sigmaV[i - 1]
        #     + mult * (self.c.offset.sigmaV[i] - self.c.offset.sigmaV[i - 1])
        # )
        return (cov_hla_new, cov_nev_new)
        # return (self.c.offset.sigmaH[i], self.c.offset.sigmaL[i], self.c.offset.sigmaA[i])

    def _fun(self, x, survey, index, station):
        s = interpolate_survey(survey, x[0], index)
        new_pos = np.array([s.n, s.e, s.tvd]).T[1]
        dist = norm(new_pos - station, axis=-1)

        return dist

    def _get_delta_nev_vectors(self):
        temp = self.off_nevs - self.c.ref_nevs
        self.dist_CC_Clr = norm(temp, axis=-1).reshape(-1,1)
        self.ref_delta_nevs = temp / self.dist_CC_Clr

        temp = self.c.ref_nevs - self.off_nevs
        self.off_delta_nevs = temp / self.dist_CC_Clr

        self.hoz_bearing = np.arctan2(
            self.ref_delta_nevs[:,1], self.ref_delta_nevs[:,0]
        )
        self.hoz_bearing_deg = (np.degrees(self.hoz_bearing) + 360) % 360

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
        # self.ref_cov_hla = self._get_cov_HLA(self.c.ref)
        # self.ref_cov_nev = self.c.ref.error_NEV
        # self.ref_cov_nev = self._get_cov_NEV(self.c.ref)
        
        self.off_cov_hla = self.off.cov_hla
        self.off_cov_nev = self.off.cov_nev
        # self.off_cov_hla = self._get_cov_HLA(self.c.offset)
        # self.off_cov_nev = self.c.offset.error_NEV
        # self.off_cov_nev = HLA_to_NEV(self.off.survey_rad, self.off_cov_hla.T, cov=True).T
        # self.off_cov_nev = self._get_cov_NEV(self.c.offset)

    # def _get_cov_HLA(self, survey, diag=True):
    #     H, L, A = np.array([survey.sigmaH, survey.sigmaL, survey.sigmaA])
    #     cov_hla = make_cov(H, L, A, diag=diag)

    #     return cov_hla

    # def _get_cov_NEV(self, survey, diag=False):
    #     N, E, V = np.array([survey.sigmaN, survey.sigmaE, survey.sigmaV])
    #     cov_nev = make_cov(N, E, V, diag=diag)

    #     return cov_nev

    def _get_PCRs(self):
        self.ref_PCR = np.hstack([
            np.sqrt(np.dot(np.dot(vec, cov), vec.T))
            for vec, cov in zip(self.ref_delta_nevs, self.ref_cov_nev)
            # np.sqrt(np.dot(np.dot(vec.T, cov), vec))
            # for vec, cov in zip(self.ref_delta_nevs, self.ref_cov_nev)
        ])
        self.off_PCR = np.hstack([
            np.sqrt(np.dot(np.dot(vec, cov), vec.T))
            for vec, cov in zip(self.off_delta_nevs, self.off_cov_nev)
            # np.sqrt(np.dot(np.dot(vec.T, cov), vec))
            # for vec, cov in zip(self.off_delta_nevs, self.off_cov_nev)
        ])

    def _get_calc_hole(self):
        self.calc_hole = self.c.Rr + self.c.Ro[self.idx]

    # def pc_method(self):
    #     """
    #     Different ways to calculate the pedal curve radius for debugging.
    #     """
    #     # from the COMPASS manual
    #     self.ref_sigmaA = np.sqrt(np.absolute(
    #         self.ref_cov_nev[:,0,0] * np.cos(self.hoz_bearing) ** 2
    #         + self.ref_cov_nev[:,0,1]
    #         * np.sin(2 * self.hoz_bearing)
    #         + self.ref_cov_nev[:,1,1] * np.sin(self.hoz_bearing)
    #     ))
    #     self.off_sigmaA = np.sqrt(np.absolute(
    #         self.off_cov_nev[:,0,0] * np.cos(self.hoz_bearing + np.pi) ** 2
    #         + self.off_cov_nev[:,0,1]
    #         * np.sin(2 * (self.hoz_bearing + np.pi))
    #         + self.off_cov_nev[:,1,1] * np.sin(self.hoz_bearing)
    #     ))
    #     # for the NE plane
    #     p_ref = np.array([self.c.ref.n, self.c.ref.e]).T
    #     p_off = np.array([self.off.n, self.off.e]).T
    #     p_delta = p_off - p_ref
    #     d = norm(p_delta, axis=-1)
    #     ref_cov_nev = self.ref_cov_nev[:,:2,:2]
    #     off_cov_nev = self.off_cov_nev[:,:2,:2]
    #     self.ne_ref_PCR = np.sqrt(np.absolute(np.array([
    #         np.dot(np.dot(v.T, c), v) for v, c in zip(p_delta, ref_cov_nev)
    #     ]) / d ** 2))
    #     self.ne_off_PCR = np.sqrt(np.absolute(np.array([
    #         np.dot(np.dot(v.T, c), v) for v, c in zip(p_delta, off_cov_nev)
    #     ]) / d ** 2))

    #     # for the EV plane
    #     p_ref = np.array([self.c.ref.e, self.c.ref.tvd]).T
    #     p_off = np.array([self.off.e, self.off.tvd]).T
    #     p_delta = p_off - p_ref
    #     d = norm(p_delta, axis=-1)
    #     ref_cov_nev = self.ref_cov_nev[:,1:,1:]
    #     off_cov_nev = self.off_cov_nev[:,1:,1:]
    #     self.ev_ref_PCR = np.sqrt(np.absolute(np.array([
    #         np.dot(np.dot(v.T, c), v) for v, c in zip(p_delta, ref_cov_nev)
    #     ]) / d ** 2))
    #     self.ev_off_PCR = np.sqrt(np.absolute(np.array([
    #         np.dot(np.dot(v.T, c), v) for v, c in zip(p_delta, off_cov_nev)
    #     ]) / d ** 2))

    #     # for the NV plane
    #     p_ref = np.array([self.c.ref.n, self.c.ref.tvd]).T
    #     p_off = np.array([self.off.n, self.off.tvd]).T
    #     p_delta = p_off - p_ref
    #     d = norm(p_delta, axis=-1)
    #     ref_cov_nev = self.ref_cov_nev[:,[0,0,-1,-1],[0,-1,0,-1]].reshape(-1,2,2)
    #     off_cov_nev = self.off_cov_nev[:,[0,0,-1,-1],[0,-1,0,-1]].reshape(-1,2,2)
    #     self.nv_ref_PCR = np.sqrt(np.absolute(np.array([
    #         np.dot(np.dot(v.T, c), v) for v, c in zip(p_delta, ref_cov_nev)
    #     ]) / d ** 2))
    #     self.nv_off_PCR = np.sqrt(np.absolute(np.array([
    #         np.dot(np.dot(v.T, c), v) for v, c in zip(p_delta, off_cov_nev)
    #     ]) / d ** 2))

    #     self.mean_ref_PCR = np.sqrt(self.ne_ref_PCR ** 2 + self.ev_ref_PCR ** 2 + self.nv_ref_PCR ** 2)
    #     self.mean_off_PCR = np.sqrt(self.ne_off_PCR ** 2 + self.ev_off_PCR ** 2 + self.nv_off_PCR ** 2)

    #     self.max_ref_PCR = [max(a, b, c) for a, b, c, in zip(self.ne_ref_PCR, self.ev_ref_PCR, self.nv_ref_PCR)]
    #     self.max_off_PCR = [max(a, b, c) for a, b, c, in zip(self.ne_off_PCR, self.ev_off_PCR, self.nv_off_PCR)]

class MeshClearance:
    def __init__(
        self,
        clearance,
        n_verts=12,
        sigma=3,
    ):
        """
        Class to calculate the clearance between two well bores using the
        standard method documented by ISCWSA.
        """
        Clearance.__init__
        self.c = clearance
        self.n_verts = n_verts
        self.sigma = sigma

        # generate mesh for offset well
        self.off_mesh = self._get_off_mesh(self.c.offset).mesh

        # make a CollisionManager object and add the offset well mesh
        self.cm = trimesh.collision.CollisionManager
        self.cm.add_object(self.off_mesh)


    def _get_mesh(self, survey):        
        mesh = WellMesh(
            survey=survey,
            n_verts=self.n_verts,
            sigma=self.sigma,
            surface_margin=self.c.Sm,
        )

    def _process_well(self):
        ref = self.c.ref
        nev = np.array([ref.n, ref.e, ref.tvd]).T
        hla = np.array([
                ref.sigmaH, ref.sigmaL, ref.sigmaA
            ]).T
        
        for i, _ in enumerate(ref):
            if i == 0: continue
            
            # slice a well section and create section survey
            md, inc, azi = ref.survey_rad[i-1:i+1].T
            n, e, tvd = nev[i-1:i+1].T
            vec = ref.vec[i-1:i+1]
            sigmaH, sigmaL, sigmaA = hla[i-1:i+1].T
            s = Survey(
                md=md,
                inc=inc,
                azi=azi,
                n=n,
                e=e,
                tvd=tvd,
                sigmaH=sigmaH,
                sigmaL=sigmaL,
                sigmaA=sigmaA,
                start_nev=[n[0], e[0], tvd[0]],
                deg=False,
                unit=self.c.ref.unit,
            )

            # generate a mesh for the section slice
            m = self._get_mesh(s).mesh

            distance_absolute = cm.min_distance_single(m, return_name=True, return_data=True)
            closest_point_reference = distance_absolute[2].point("__external")
            name_offset_absolute = distance_absolute[1]
            closest_point_offset = distance_absolute[2].point(name_offset_absolute)
            reference_nev = reference_NEVs[KDTree(reference_NEVs).query(closest_point_reference)[1]]







        
