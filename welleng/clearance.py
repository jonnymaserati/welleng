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

        return (cov_hla_new, cov_nev_new)

    def _fun(self, x, survey, index, station):
        s = interpolate_survey(survey, x[0], index)
        new_pos = np.array([s.n, s.e, s.tvd]).T[1]
        dist = norm(new_pos - station, axis=-1)

        return dist

    def _get_delta_nev_vectors(self):
        temp = self.off_nevs - self.c.ref_nevs
        self.dist_CC_Clr = norm(temp, axis=-1).reshape(-1,1)
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







        
