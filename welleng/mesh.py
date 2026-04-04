"""Wellbore mesh generation from survey data and positional uncertainty."""

try:
    import trimesh
    TRIMESH = True
except ImportError:
    TRIMESH = False
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from numpy import sin, cos, pi
from scipy.spatial import KDTree

from .utils import HLA_to_NEV, get_sigmas
from .survey import slice_survey, Survey
from .visual import figure


class WellMesh:
    """Triangular mesh representing a wellbore's positional uncertainty envelope.

    Attributes
    ----------
    s : welleng.survey.Survey
        The input Survey object.
    vertices : numpy.ndarray
        Vertex positions array, shape (n_stations, n_verts, 3).
    faces : numpy.ndarray
        Triangle face index array, shape (n_faces, 3).
    mesh : types.SimpleNamespace
        Lightweight mesh container with ``vertices`` and ``faces`` attributes.
    n_verts : int
        Number of vertices per station cross-section.
    sigma : float
        Sigma multiplier for the uncertainty envelope.
    radius : numpy.ndarray
        Wellbore radius at each station.
    nevs : numpy.ndarray
        Station positions in NEV coordinates, shape (n_stations, 3).

    Methods
    -------
    figure()
        Create a plotly 3D figure of the well mesh.
    """

    def __init__(
        self,
        survey: Survey,
        n_verts: int = 12,
        sigma: float = 3.0,
        sigma_pa: float = 0.5,
        Sm: float = 0,
        method: str = "ellipse",
    ):
        """Create a WellMesh object from a welleng Survey object.

        Parameters
        ----------
        survey : welleng.survey.Survey
            The survey from which to build the mesh.
        n_verts : int, optional
            The number of vertices along the uncertainty ellipse
            edge from which to construct the mesh. Recommended minimum is
            12 and that the number is a multiple of 4.
        sigma : float, optional
            The desired standard deviation sigma value of the well bore
            uncertainty.
        sigma_pa : float, optional
            The desired "project ahead" value. A remnant of the ISCWSA
            method but may be used in the future to accommodate for well
            bore curvature that is not captured by the mesh.
        Sm : float, optional
            From the ISCWSA method, this is an additional factor applied to
            the well bore radius of the offset well to oversize the hole.
        method : str, optional
            The method for constructing the uncertainty edge.
            Either "ellipse", "pedal_curve" or "circle".
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
        total_verts = len(self.vertices.reshape(-1, 3))
        rows = int(total_verts / self.n_verts)

        self.faces = get_faces(self.n_verts, rows)

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

        elif self.method == "ellipse":
            # Use eigenvalue decomposition of the 2x2 H-L covariance submatrix
            # to correctly represent the uncertainty ellipse orientation and
            # magnitude, rather than assuming alignment with h and l axes.
            cov_hl = self.s.cov_hla[:, :2, :2]  # shape (n, 2, 2)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_hl)
            # eigh returns eigenvalues in ascending order; clip for numerical safety
            sigma_major = np.sqrt(np.maximum(eigenvalues[:, 1], 0))
            sigma_minor = np.sqrt(np.maximum(eigenvalues[:, 0], 0))
            h = (
                sigma_major * self.sigma
                + self.radius + self.Sm
                + self.sigma_pa / 2
            ).reshape(-1, 1)
            l = (
                sigma_minor * self.sigma
                + self.radius + self.Sm
                + self.sigma_pa / 2
            ).reshape(-1, 1)
            # Rotation angle of the major axis in the H-L plane
            ellipse_theta = np.arctan2(
                eigenvectors[:, 1, 1], eigenvectors[:, 0, 1]
            ).reshape(-1, 1)

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
            temp = np.linspace(0, 2 * pi, self.n_verts, endpoint=False)

            temp = np.broadcast_to(temp, (len(self.s.md), len(temp)))
            lam = deepcopy(temp)

            x_ell = h * cos(lam)
            y_ell = l * sin(lam)

            if self.method == "ellipse":
                cos_t = np.cos(ellipse_theta)
                sin_t = np.sin(ellipse_theta)
                x = x_ell * cos_t - y_ell * sin_t
                y = x_ell * sin_t + y_ell * cos_t
            else:
                x = x_ell
                y = y_ell

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

                if bool(np.isnan(units_2).any()):
                    raise ValueError('Missing uncertainty data')

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
        Store well vertices and faces as a lightweight mesh namespace.

        A plain SimpleNamespace is used so that WellMesh has no trimesh
        dependency.  Code that needs a trimesh.Trimesh (e.g. MeshClearance)
        should call ``welleng.mesh.to_trimesh(well_mesh)`` explicitly.

        Two center vertices (first and last wellpath positions) are appended
        so that end-cap triangles fan from the wellpath centerline rather
        than from an arbitrary circumference vertex.
        '''
        ring_vertices = self.vertices.reshape(-1, 3)
        center_top = self.nevs[0:1]       # shape (1, 3)
        center_bottom = self.nevs[-1:]     # shape (1, 3)
        vertices = np.vstack([ring_vertices, center_top, center_bottom])
        self.mesh = SimpleNamespace(vertices=vertices, faces=self.faces)

    def figure(self, type='mesh3d', **kwargs):
        """Create a plotly figure of this mesh.

        Parameters
        ----------
        type : str, optional
            Plotly figure type, default 'mesh3d'.
        **kwargs
            Passed to welleng.visual.figure().

        Returns
        -------
        plotly.graph_objects.Figure
            A plotly Figure instance.
        """
        fig = figure(self, type, **kwargs)
        return fig


def to_trimesh(well_mesh):
    """
    Convert a :class:`WellMesh` to a ``trimesh.Trimesh`` object.

    This is the single point where trimesh is constructed from the stored
    geometry arrays.  Call this explicitly wherever a true trimesh object is
    required (e.g. collision detection in :class:`~welleng.clearance.MeshClearance`).

    Parameters
    ----------
    well_mesh : WellMesh

    Returns
    -------
    trimesh.Trimesh
    """
    assert TRIMESH, "ImportError: try pip install welleng[easy]"
    return trimesh.Trimesh(
        vertices=well_mesh.mesh.vertices,
        faces=well_mesh.mesh.faces,
        process=False,
        validate=True,
    )


def make_trimesh_scene(data):
    """
    Construct a trimesh scene. A collision manager can't be saved, but a scene
    can and a scene can be imported into a collision manager.

    Parameters
    ----------
    data : list
        List of welleng.mesh.WellMesh objects.

    Returns
    -------
    trimesh.scene.scene.Scene
        A trimesh scene containing all well meshes.
    """
    assert TRIMESH, "ImportError: try pip install welleng[easy]"
    scene = trimesh.scene.scene.Scene()
    for well in data:
        mesh = to_trimesh(data[well])
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
    scene : trimesh.scene.scene.Scene
        A trimesh scene of well meshes.
    origin : array_like, optional
        3D array [x, y, z]. The origin of the scene from which the new
        scene will reset to [0, 0, 0].
    scale : float, optional
        A scalar reduction will be performed using this float.
    redux : float, optional
        The desired reduction ratio for the number of triangles in each
        mesh.

    Returns
    -------
    trimesh.scene.scene.Scene
        A transformed, scaled and reprocessed scene.
    """
    assert TRIMESH, "ImportError: try pip install welleng[easy]"
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
        mesh_new.vertices = np.column_stack([x, -z, y]) / scale
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
    survey : welleng.survey.Survey
        The survey from which to build the meshes.
    n_verts : int, optional
        The number of vertices along the uncertainty ellipse
        edge from which to construct the mesh. Recommended minimum is
        12 and that the number is a multiple of 4.
    sigma : float, optional
        The desired standard deviation sigma value of the well bore
        uncertainty.
    sigma_pa : float, optional
        The desired "project ahead" value. A remnant of the ISCWSA method
        but may be used in the future to accommodate for well bore
        curvature that is not captured by the mesh.
    Sm : float, optional
        From the ISCWSA method, this is an additional factor applied to
        the well bore radius of the offset well to oversize the hole.
    method : str, optional
        The method for constructing the uncertainty edge.
        Either "ellipse" or "pedal_curve".

    Returns
    -------
    list
        List of mesh namespace objects.

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
    """Fix a non-watertight mesh by removing duplicate and degenerate faces,
    then repairing windings and normals.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to repair.

    Returns
    -------
    trimesh.Trimesh
        A repaired mesh with correct windings and normals.
    """
    assert TRIMESH, "ImportError: try pip install welleng[easy]"
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


def get_ends(n_verts, rows):
    """Build cap faces for the first and last cross-section rings.

    End-cap triangles fan from center vertices (the wellpath positions)
    appended after all ring vertices, rather than from circumference
    vertex 0.

    Parameters
    ----------
    n_verts : int
        Number of vertices per cross-section ring.
    rows : int
        Number of cross-section rings along the wellbore.

    Returns
    -------
    tuple of numpy.ndarray
        (top_faces, bottom_faces), each of shape (n_verts, 3).
    """
    total_ring_verts = rows * n_verts
    center_top = total_ring_verts       # index of first center vertex
    center_bottom = total_ring_verts + 1  # index of second center vertex

    # Top cap: fan from center_top to first ring
    ring_idx = np.arange(n_verts)
    ring_next = np.roll(ring_idx, -1)
    top = np.column_stack([
        np.full(n_verts, center_top), ring_idx, ring_next
    ])

    # Bottom cap: fan from center_bottom to last ring
    last_ring_start = (rows - 1) * n_verts
    ring_idx_bottom = last_ring_start + np.arange(n_verts)
    ring_next_bottom = last_ring_start + np.roll(np.arange(n_verts), -1)
    bottom = np.column_stack([
        np.full(n_verts, center_bottom), ring_next_bottom, ring_idx_bottom
    ])

    return top, bottom


def get_faces(n_verts, rows):
    """Build triangular face indices for a tubular mesh.

    Parameters
    ----------
    n_verts : int
        Number of vertices per cross-section ring.
    rows : int
        Number of cross-section rings along the wellbore.

    Returns
    -------
    numpy.ndarray
        Face index array of shape (n_faces, 3).
    """
    A = np.arange(n_verts).reshape(-1, 1)
    B = np.roll(A, -1).reshape(-1, 1)

    lower = np.arange(rows) + np.arange(rows) * (n_verts - 1)
    upper = lower[1:]
    lower = lower[:-1]


    faces = np.array([
        A + lower, A + upper, B + upper,
        A + lower, B + lower, B + upper
    ]).T.reshape(-1, 3)

    top, bottom = get_ends(n_verts, rows)

    return np.vstack([top, faces, bottom])
