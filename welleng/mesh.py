import trimesh
import numpy as np
from numpy import sin, cos, pi
from scipy.spatial import KDTree

from .utils import HLA_to_NEV, get_sigmas
from .survey import slice_survey


class WellMesh:

    def __init__(
        self,
        survey,
        n_verts=12,
        sigma=3.0,
        sigma_pa=0.5,
        Sm=0,
        method="ellipse",
    ):
        """
        Create a WellMesh object from a welleng Survey object and a
        welleng Clearance object.

        Parameters
        ----------
            survey: welleng.Survey
            clearance: welleng.Clearance
            n_verts: int
                The number of vertices along the uncertainty ellipse
                edge from which to construct the mesh. Recommended minimum is
                12 and that the number is a multiple of 4.
            sigma: float (default: 3.0)
                The desired standard deviation sigma value of the well bore
                uncertainty.
            sigma_pa: float (default: 0.5)
                The desired "project ahead" value. A remnant of the ISCWSA
                method but may be used in the future to accomodate for well
                bore curvature that is not captured by the mesh.
            Sm: float
                From the ISCWSA method, this is an additional factor applied to
                the well bore radius of the offset well to oversize the hole.
            method: str (default="ellipse")
                The method for constructing the uncertainty edge.
                Either "ellipse" or "pedal_curve".
        """
        self.s = survey
        # self.c = clearance
        self.n_verts = int(n_verts)
        self.sigma = sigma
        self.radius = self.s.radius
        self.Sm = Sm
        self.sigma_pa = sigma_pa

        assert method in ["ellipse", "pedal_curve", "circle"], \
            "Invalid method (ellipse or pedal_curve)"
        self.method = method

        if self.method != 'circle':
            self.sigmaH, self.sigmaL, self.sigmaA = get_sigmas(
                self.s.cov_hla
            )
        self.nevs = np.array([self.s.n, self.s.e, self.s.tvd]).T.reshape(-1, 3)
        self._get_vertices()
        self._align_verts()

        self._get_faces()
        self._make_trimesh()

    # Helper functions #
    def _get_faces(self):
        '''
        Construct a mesh of triangular faces (n,3) on the well bore
        uncertainty edge for the given well bore.
        '''
        step = self.n_verts
        faces = []
        total_verts = len(self.vertices.reshape(-1, 3))
        rows = int(total_verts / self.n_verts)

        # make first end
        B = np.arange(1, step - 1, 1)
        C = np.arange(2, step, 1)
        A = np.zeros_like(B)
        temp = np.array([A, B, C]).T
        faces.extend(temp.tolist())

        # make cylinder
        temp = ([np.array([
            step, 0, step - 1, 2 * step - 1, step, step - 1
        ])])
        # TODO: this can probably be vectorized to speed it up
        for row in range(0, rows - 1):
            verts = row * self.n_verts

            A_start = verts
            B_start = verts + step
            C_start = verts + step + 1
            D_start = verts + 1

            A_stop = verts + step - 1
            B_stop = verts + step + step - 1
            C_stop = verts + step + step
            D_stop = verts + step

            A = np.arange(A_start, A_stop)
            B = np.arange(B_start, B_stop)
            C = np.arange(C_start, C_stop)
            D = np.arange(D_start, D_stop)

            temp.extend(np.array([C, D, A, B, C, A]).T)

            last = np.array([
                B_start,
                A_start,
                D_stop - 1,
                C_stop - 1,
                B_start,
                D_stop - 1
            ])

            temp.extend([last])

        faces.extend(np.stack(temp, axis=0).reshape(-1, 3).tolist())

        # make final end
        B = np.arange(total_verts - step + 1, (total_verts - 1), 1)
        C = np.arange(total_verts - step + 2, (total_verts), 1)
        A = np.full_like(B, total_verts - step)
        temp = np.array([A, B, C]).T
        faces.extend(temp.tolist())

        self.faces = faces

    def _get_vertices(
        self,
    ):
        '''
        Determine the positions of the vertices on the desired uncertainty
        circumference.
        '''
        if self.method == "circle":
            h = self.s.radius.reshape(-1, 1)
            l = h

        else:
            h = (
                np.array(self.sigmaH) * self.sigma
                + self.radius + self.Sm
                + self.sigma_pa / 2
            ).reshape(-1, 1)

            l = (
                np.array(self.sigmaL) * self.sigma
                + self.radius
                + self.Sm
                + self.sigma_pa / 2
            ).reshape(-1, 1)
            # a = self.s.sigmaA * self.c.k + self.s.radius + self.c.Sm

        if self.method in ["ellipse", "circle"]:
            lam = np.linspace(0, 2 * pi, self.n_verts, endpoint=False)
            # theta = np.zeros_like(lam)

            x = h * cos(lam)
            y = l * sin(lam)
            z = np.zeros_like(x)
            vertices = np.stack((x, y, z), axis=-1)

        else:
            # make the vertices evenly spaced around the circumference
            lam = np.concatenate((
                np.arccos(
                    np.linspace(
                        1., -1., int(self.n_verts / 2),
                        endpoint=False
                    )
                ),
                np.arccos(
                    np.linspace(
                        1., -1., int(self.n_verts / 2),
                        endpoint=False)) + np.pi
                )
            )
            f = h * (
                (l ** 2 * np.cos(lam))
                /
                (l ** 2 * (np.cos(lam)) ** 2 + h ** 2 * (np.sin(lam)) ** 2)
            )

            g = l * (
                (h ** 2 * np.sin(lam))
                /
                (l ** 2 * (np.cos(lam)) ** 2 + h ** 2 * (np.sin(lam)) ** 2)
            )
            z = np.zeros_like(f)
            vertices = np.stack((f, g, z), axis=-1)

        vertices = HLA_to_NEV(self.s.survey_rad, vertices, cov=False)

        self.vertices = (
            vertices
            + self.nevs.reshape(tuple(
                (
                    self.nevs.shape[0],
                    1,
                    self.nevs.shape[1]
                )
            ))
        )

    def _get_vertices_vectors(self, vertices, pos_center):
        '''
        Determine the vectors of the vertices relative the the well path
        position.
        '''
        vectors = vertices - pos_center
        normals = np.linalg.norm(vectors, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vectors = vectors / normals.reshape(-1, 1)
        unit_vectors[np.where(normals == 0)] = vectors[np.where(normals == 0)]

        return (vectors, normals, unit_vectors)

    def _align_verts(self):
        """
        The meshing gets unstable when the inclination is close to vertical
        (since high side can quickly shift azimuth). This code cycles through
        the ellipses and makes sure that they're all lined up relative to each
        other.
        """
        verts_new_list = []
        for i, verts in enumerate(self.vertices):
            if i == 0:
                verts_new_list.append([verts])
                continue
            else:
                vecs_1, norms_1, units_1 = self._get_vertices_vectors(
                    verts_new_list[-1][0],
                    self.nevs[i - 1]
                )
                vecs_2, norms_2, units_2 = self._get_vertices_vectors(
                    self.vertices[i],
                    self.nevs[i]
                )

                nearest = KDTree(units_2)

                key = units_1[0]
                idx_key = nearest.query(key)[1]

                rotation = units_1[3]
                idx_rotation = nearest.query(rotation)[1]

                # re-order vecs_2 to align with vecs_1
                verts_new = np.vstack((verts[idx_key:], verts[:idx_key]))

                if (
                    (idx_rotation - idx_key) < 0
                    and abs(idx_rotation - idx_key) < len(verts) / 2
                ):
                    verts_new = verts_new[::-1]

                verts_new_list.append([verts_new])

        self.vertices = np.vstack(
            verts_new_list
        ).reshape(self.vertices.shape)

    def _make_trimesh(self):
        '''
        Consturct a trimesh.Trimesh object from the well vertices and faces.
        '''
        vertices = self.vertices.reshape(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=self.faces,
            process=False,
            validate=True,
        )

        # self.mesh = fix_mesh(mesh)
        self.mesh = mesh


def make_trimesh_scene(data):
    """
    Construct a trimesh scene. A collision manager can't be saved, but a scene
    can and a scene can be imported into a collision manager.

    Parameters
    ----------
        data: list of welleng.mesh.WellEng objects.

    Returns
    -------
        scene: a trimesh.scene.scene.Scene object.
    """
    scene = trimesh.scene.scene.Scene()
    for well in data:
        mesh = data[well].mesh
        scene.add_geometry(
            mesh, node_name=well, geom_name=well, parent_node_name=None
        )

    return scene


def transform_trimesh_scene(scene, origin=None, scale=100, redux=0.25):
    """
    Transforms a scene by scaling it, reseting the origin/datum and performing
    a reduction in the number of triangles to reduce the file size.

    Parameters
    ----------
        scene: trimesh.scene.scene.Scene object
            A trimesh scene of well meshes
        origin: 3d array [x, y, z]
            The origin of the scene from which the new scene will reset to
            [0, 0, 0].
        scale: float
            A scalar reduction will be performed using this float
        redux: float
            The desired reduction ratio for the number of triangles in each
            mesh.

    Returns
    -------
        scene_transformed: trimesh.scene.scene.Scene object
            A transformed, scaled and reprocessed scene.
    """
    i = 0
    if not origin:
        T = np.array([0, 0, 0])
    else:
        T = np.array(origin)
    scene_transformed = trimesh.scene.scene.Scene()
    for well in scene.geometry.items():
        name, mesh = well
        mesh_new = mesh.copy()

        mesh_new.simplify_quadratic_decimation(
            int(len(mesh_new.triangles) * redux)
        )
        mesh_new.vertices -= T

        # change axis convention for visualisation #
        x, y, z = mesh_new.vertices.T
        mesh_new.vertices = (np.array([x, z * -1, y]) / scale).T
        scene_transformed.add_geometry(
            mesh_new, node_name=name, geom_name=name
        )

    return scene_transformed


def sliced_mesh(
    survey,
    n_verts=12,
    sigma=3.0,
    sigma_pa=0.5,
    Sm=0,
    start=0,
    stop=-1,
    step=1,
    method="mesh_ellipse",
):
    """
    Generates a list of mesh objects of a user defined length.

    Parameters
    ----------
        survey: welleng.Survey
        n_verts: int
            The number of vertices along the uncertainty ellipse
            edge from which to construct the mesh. Recommended minimum is
            12 and that the number is a multiple of 4.
        sigma: float (default: 3.0)
            The desired standard deviation sigma value of the well bore
            uncertainty.
        sigma_pa: float (default: 0.5)
            The desired "project ahead" value. A remnant of the ISCWSA method
            but may be used in the future to accomodate for well bore
            curvature that is not captured by the mesh.
        Sm: float
            From the ISCWSA method, this is an additional factor applied to
            the well bore radius of the offset well to oversize the hole.
        method: str (default="ellipse")
            The method for constructing the uncertainty edge.
            Either "ellipse" or "pedal_curve".

    Returns
    -------
        list of trimesh.Trimesh mesh objects.

    """
    meshes = []
    l = len(survey.md)
    i = start
    assert stop <= l, f"Out of range, {stop} > {l}"
    if stop == -1:
        stop = l

    # TODO: set this up for multiprocessing

    while True:
        j = i + step + 1
        if j > stop:
            j = stop
        if j - i < 2 and len(meshes) > 0:
            del meshes[-1]
            i -= step

        # slice a well section and create section survey
        s = slice_survey(survey, i, j)

        # generate a mesh for the section slice
        m = WellMesh(
            s,
            n_verts=n_verts,
            sigma=sigma,
            sigma_pa=sigma_pa,
            Sm=Sm,
            method=method,
        )

        meshes.append(m.mesh)

        if j == stop:
            break
        i += step

    return meshes


def fix_mesh(mesh):
    '''
    For whatever reason, the first pass meshing often results in a mesh
    that is not watertight. This function fixes that, as well as
    repairs the windings and normals.

    Parameters
    ----------
        mesh: trimesh object

    Returns
    -------
        good_mesh: trimesh object
    '''
    # it makes two outputs (unique_idx, inverse)
    unique_indices = trimesh.grouping.unique_rows(mesh.faces)[0]
    unique_faces = mesh.faces[unique_indices]

    flat_faces = (
        (unique_faces[:, 0] == unique_faces[:, 1])
        | (unique_faces[:, 0] == unique_faces[:, 2])
        | (unique_faces[:, 1] == unique_faces[:, 2])
    )

    good_faces = unique_faces[~flat_faces]
    good_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=good_faces, process=False
    )

    trimesh.repair.fix_winding(good_mesh)
    trimesh.repair.fix_normals(good_mesh)

    return good_mesh
