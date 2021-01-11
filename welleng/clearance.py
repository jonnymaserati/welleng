import numpy as np
from numpy.linalg import norm
import trimesh

from scipy import optimize
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from .survey import Survey, interpolate_survey, slice_survey
from .utils import NEV_to_HLA
from .mesh import WellMesh


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
        kop_depth=-np.inf,
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
        self.kop_index = np.searchsorted(
            self.reference.md, self.kop_depth, side="left"
        )

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

        # calculate SF (renamed from ISCWSA_ACR)
        self.SF = np.array(
            (self.dist_CC_Clr.T - self.calc_hole - self.c.Sm)
            / (self.c.k * np.sqrt(self.sigmaS ** 2 + self.c.sigma_pa ** 2)),
            ).reshape(-1)

        # for debugging
        # self.pc_method()

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
        closest = []
        for j, (i, station) in enumerate(zip(
            self.idx, self.c.ref_nevs.tolist()
        )):
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
                    interpolate_survey(self.c.offset, res_1.x[0], i - 1),
                    res_1, sigma_new_1
                ))
            else:
                closest.append((
                    station,
                    interpolate_survey(self.c.offset, res_2.x[0], i),
                    res_2,
                    sigma_new_2
                ))

        self.closest = closest
        md, inc, azi, n, e, tvd, x, y, z,  = np.array([
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
            cov_hla=cov_hla,
            cov_nev=cov_nev,
            start_xyz=[x[0], y[0], z[0]],
            start_nev=[n[0], e[0], tvd[0]],
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
        sigma=2.445,
        return_data=True,
        return_meshes=False,
    ):
        """
        Class to calculate the clearance between two well bores using the
        standard method documented by ISCWSA.
        """
        Clearance.__init__
        self.c = clearance
        self.n_verts = n_verts
        self.sigma = sigma
        self.Rr = self.c.ref.radius
        self.Ro = self.c.offset.radius

        # if you're only interesting in a binary "go/no-go" decision
        # then you can forfeit the expensive ISCWSA calculations by
        # setting return_data to False.
        self.return_data = return_data

        if self.return_data:
            self.distance_CC = []
            self.distance = []
            self.collision = []
            self.off_index = []
            self.SF = []
            self.nev = []
            self.hoz_bearing_deg = []
            self.ref_PCR = []
            self.off_PCR = []
            self.calc_hole = []
            self.ref_md = []
            self.off_md = []

        self.return_meshes = return_meshes
        if self.return_meshes:
            self.meshes = []

        # generate mesh for offset well
        self.off_mesh = self._get_mesh(self.c.offset, offset=True).mesh

        # make a CollisionManager object and add the offset well mesh
        self.cm = trimesh.collision.CollisionManager()

        self.cm.add_object("offset", self.off_mesh)

        self._process_well()

    def get_lines(self):
        """
        Extracts the closest points between wells for each survey section.
        """
        points = [
            list(d[2]._points.values())
            for d in self.distance
        ]

        start_points = []
        end_points = []

        [(start_points.append(p[0]), end_points.append(p[1])) for p in points]

        return np.array([
            np.vstack(start_points).tolist(),
            np.vstack(end_points).tolist()
        ])

    def _get_mesh(self, survey, offset=False):
        """
        Generates a mesh object from the survey object.
        """
        if offset:
            Sm = self.c.Sm
        else:
            Sm = 0.0

        mesh = WellMesh(
            survey=survey,
            n_verts=self.n_verts,
            sigma=self.sigma,
            Sm=Sm,
        )

        return mesh

    def _process_well(self):
        """
        Iterates through the reference well survey, determining for
        each section whether a collision has occurred with the offset
        well and optionally calculates separation data.
        """
        ref = self.c.ref
        off = self.c.offset
        off_nevs = self.c.offset_nevs

        for i in range(len(ref.md) - 1):
            # slice a well section and create section survey
            s = slice_survey(ref, i)

            # generate a mesh for the section slice
            m = self._get_mesh(s).mesh

            # see if there's a collision
            collision = self.cm.in_collision_single(
                m, return_names=self.return_data, return_data=self.return_data
            )

            if self.return_data:
                distance = self.cm.min_distance_single(
                    m, return_name=True, return_data=True
                )
                closest_point_reference = distance[2].point("__external")
                name_offset_absolute = distance[1]
                closest_point_offset = distance[2].point(name_offset_absolute)

                ref_nev = self._get_closest_nev(s, closest_point_reference)
                ref_md = ref.md[i-1] + ref_nev[1].x[0]

                # find the closest point on the well trajectory to the closest
                # points on the mesh surface
                off_index = KDTree(off_nevs).query(closest_point_offset)[1]
                if off_index < len(off.md) - 1:
                    s = slice_survey(off, off_index)
                    off_nev_1 = self._get_closest_nev(s, closest_point_offset)
                else:
                    off_nev_1 = False

                if off_index > 0:
                    s = slice_survey(off, off_index - 1)
                    off_nev_0 = self._get_closest_nev(s, closest_point_offset)
                else:
                    off_nev_0 = False

                if off_nev_0 and off_nev_1:
                    if off_nev_0[1].fun < off_nev_1[1].fun:
                        off_nev = off_nev_0
                        off_md = off.md[off_index-1] + off_nev_0[1].x[0]
                    else:
                        off_nev = off_nev_1
                        off_md = off.md[off_index] + off_nev_1[1].x[0]
                elif off_nev_0:
                    off_nev = off_nev_0
                    off_md = off.md[off_index-1] + off_nev_0[1].x[0]
                else:
                    off_nev = off_nev_1
                    off_md = off.md[off_index] + off_nev_1[1].x[0]

                vec = off_nev[0] - ref_nev[0]
                distance_CC = norm(vec)
                hoz_bearing_deg = (
                    np.degrees(np.arctan2(vec[1], vec[0])) + 360
                ) % 360

                if collision[0] is True:
                    depth = norm(
                        closest_point_offset - closest_point_reference
                    )
                    # prevent divide by zero
                    if distance_CC != 0 and depth != 0:
                        SF = distance_CC / (distance_CC + depth)
                    else:
                        SF = 0
                else:
                    SF = distance_CC / (distance_CC - distance[0])

                # data for ISCWSA method comparison
                self.collision.append(collision)
                self.off_index.append(off_index)
                self.distance.append(distance)
                self.distance_CC.append(distance_CC)
                self.SF.append(round(SF, 2))
                self.nev.append((ref_nev, off_nev))
                self.hoz_bearing_deg.append(hoz_bearing_deg)
                self.ref_PCR.append(
                    (ref_nev[1].fun - self.c.sigma_pa / 2 - self.Rr[i])
                    / self.sigma
                )
                self.off_PCR.append(
                    (
                        off_nev[1].fun - self.c.sigma_pa / 2
                        - self.Ro[off_index] - self.c.Sm
                    ) / self.sigma
                )
                self.calc_hole.append(ref.radius[i] + off.radius[off_index])
                self.ref_md.append(ref_md)
                self.off_md.append(off_md)

            if self.return_meshes:
                self.meshes.append(m)

            else:
                self.collision.append(collision)

    def _fun(self, x, survey, pos):
        """
        Interpolates a point on a well trajectory and returns
        the distance between the interpolated point and the
        position provided.
        """
        s = interpolate_survey(survey, x[0])
        new_pos = np.array([s.n, s.e, s.tvd]).T[1]
        dist = norm(new_pos - pos, axis=-1)

        return dist

    def _get_closest_nev(self, survey, pos):
        """
        Using an optimization function to determine the closest
        point along a well trajectory to the position provided.
        """
        bnds = [(0, survey.md[1] - survey.md[0])]
        res = optimize.minimize(
            self._fun,
            bnds[0][1] / 2,
            method='SLSQP',
            bounds=bnds,
            args=(survey, pos)
            )

        s = interpolate_survey(survey, res.x[0])

        nev = np.array([s.n, s.e, s.tvd]).T[-1]

        return (nev, res)
