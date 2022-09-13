from typing import Tuple

import numpy as np
from numpy.linalg import norm

try:
    import trimesh
    MESH_MODE = True
except ImportError:
    MESH_MODE = False

from scipy import optimize
from scipy.spatial import KDTree

from .clearance import Clearance
from .mesh import WellMesh
from .survey import Survey, _interpolate_survey, slice_survey


class MeshClearance:
    def __init__(
        self,
        clearance: Clearance,
        n_verts: int = 12,
        sigma: float = 2.445,
        return_data: bool = True,
        return_meshes: bool = False,
    ):
        """
        Class to calculate the clearance between two well bores using a novel
        mesh clearnace method. This method is experimental and was developed
        to provide a fast method for determining if well bores are potentially
        colliding.

        This class requires that `trimesh` is installed along with
        `python-fcl`.

        clearance : `welleng.clearnance.Clearance` object
        n_verts : int
            The number of points (vertices) used to generate the uncertainty
            ellipses which are used to generate a `trimesh` representation of
            the well bores. The default is 12 which is a good balance between
            accuracy and speed.
        sigma : float
            The required/desired sigma value representation of the generated
            mesh. The default value of 2.445 represents about 98.5% confidence
            of the well bore being located within the volume of the generated
            mesh.
        """
        assert MESH_MODE, "ImportError: try pip install welleng[all]"
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

    def get_lines(self) -> np.ndarray:
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

    def _get_mesh(self, survey: Survey, offset: bool = False) -> WellMesh:
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
                ref_md = ref.md[i - 1] + ref_nev[1].x[0]

                # find the closest point on the well trajectory to the closest
                # points on the mesh surface
                off_index = KDTree(off_nevs).query(closest_point_offset)[1]

                off_md, off_nev = self.get_offset_md_and_nev(
                    off,
                    off_index,
                    closest_point_offset
                )

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

    def _fun(self, x, survey: Survey, pos: np.ndarray):
        """
        Interpolates a point on a well trajectory and returns
        the distance between the interpolated point and the
        position provided.
        """
        s = _interpolate_survey(survey, x[0])
        new_pos = np.array([s.n, s.e, s.tvd]).T[1]
        dist = norm(new_pos - pos, axis=-1)

        return dist

    def _get_closest_nev(self, survey: Survey, pos: np.ndarray) -> Tuple:
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

        s = _interpolate_survey(survey, res.x[0])

        nev = np.array([s.n, s.e, s.tvd]).T[-1]

        return (nev, res)

    def get_offset_md_and_nev(
            self,
            offset: Survey,
            offset_index: int,
            closest_point_offset: np.ndarray
    ) -> Tuple[float, np.ndarray]:

        if offset_index < len(offset.md) - 1:
            s = slice_survey(offset, offset_index)
            off_nev_1 = self._get_closest_nev(s, closest_point_offset)
        else:
            off_nev_1 = False

        if offset_index > 0:
            s = slice_survey(offset, offset_index - 1)
            off_nev_0 = self._get_closest_nev(s, closest_point_offset)
        else:
            off_nev_0 = False

        if off_nev_0 and off_nev_1:
            if off_nev_0[1].fun < off_nev_1[1].fun:
                offset_nev = off_nev_0
                offset_md = offset.md[offset_index - 1] + off_nev_0[1].x[0]
            else:
                offset_nev = off_nev_1
                offset_md = offset.md[offset_index] + off_nev_1[1].x[0]
        elif off_nev_0:
            offset_nev = off_nev_0
            offset_md = offset.md[offset_index - 1] + off_nev_0[1].x[0]
        else:
            offset_nev = off_nev_1
            offset_md = offset.md[offset_index] + off_nev_1[1].x[0]

        return offset_md, offset_nev
