import trimesh
from vedo import show, Box, Axes, trimesh2vedo, Lines, Sphere
import numpy as np

from .version import __version__ as VERSION


class World:
    def __init__(
        self,
        bb_center,
        length,
        width,
        height
    ):
        self.bb_center = bb_center
        self.length = length
        self.width = width
        self.height = height
        self.world = Box(
            bb_center,
            length,
            width,
            height
        ).wireframe()


def plot(
    data,
    names=None,
    colors=None,
    lines=None,
    targets=None,
    arrows=None,
    text=None,
    boxes=None,
    points=None,
    interactive=True,
):
    """
    A vedo wrapper for quickly visualizing well trajectories for QAQC purposes.

    Parameters
    ----------
        data: a trimesh.Trimesh object or a list of trimesh.Trimesh
        objects or a trmiesh.scene object
        names: list of strings (default: None)
            A list of names, index aligned to the list of well meshes.
        colors: list of strings (default: None)
            A list of color or colors. If a single color is listed then this is
            applied to all meshes in data, otherwise the list of colors is
            indexed to the list of meshes.
    """
    if isinstance(data, trimesh.scene.scene.Scene):
        meshes = [v for k, v in data.geometry.items()]
        if names is None:
            names = list(data.geometry.keys())

    # handle a single mesh being passed
    elif isinstance(data, trimesh.Trimesh):
        meshes = [data]

    else:
        meshes = data
        if names is not None:
            assert len(names) == len(data), \
                "Names must be length of meshes list else None"

    if colors is not None:
        if len(colors) == 1:
            colors = colors * len(meshes)
        else:
            assert len(colors) == len(meshes), \
                "Colors must be length of meshes list, 1 else None"

    if points is not None:
        points = [
            Sphere(p, r=30, c='grey')
            for p in points
        ]

    meshes_vedo = []
    for i, mesh in enumerate(meshes):
        if i == 0:
            vertices = np.array(mesh.vertices)
            start_locations = np.array([mesh.vertices[0]])
        else:
            vertices = np.concatenate(
                (vertices, np.array(mesh.vertices)),
                axis=0
            )
            start_locations = np.concatenate(
                (start_locations, np.array([mesh.vertices[0]])),
                axis=0
            )

        # convert to vedo mesh
        m_vedo = trimesh2vedo(mesh)
        if colors is not None:
            m_vedo.c(colors[i])
        if names is not None:
            m_vedo.name = names[i]
            m_vedo.flag()
        meshes_vedo.append(m_vedo)

    w = get_bb(vertices)

    axes = get_axes(w.world)

    # try and figure out a nice start camera position
    pos = w.bb_center
    vec1 = pos - [w.length, w.width, 0]
    vec2 = np.array([vec1[1], vec1[0], 0])
    pos_new = [pos[0], pos[1], -4000] + vec2 * 3
    camera_opts = dict(
        pos=pos_new,
        focalPoint=pos,
        viewup=[0., 0., -1.]
    )

    show(
        meshes_vedo,
        w.world,
        lines,
        targets,
        arrows,
        boxes,
        axes,
        points,
        bg='lightgrey',
        bg2='lavender',
        camera=camera_opts,
        interactorStyle=10,
        resetcam=True,
        interactive=True,
        # verbose=True,
        title=f'welleng {VERSION}'
    )


def get_start_location(start_locations):
    start_location = np.average(start_locations, axis=0)
    start_location[2] = np.amin(start_locations[:, 2], axis=0)
    return start_location


def get_bb(vertices, min_size=[1000., 1000., 0.]):
    bb_max = np.amax(vertices, axis=0)
    bb_min = np.amin(vertices, axis=0)

    l, w, h = np.amax(np.array([(bb_max - bb_min), min_size]), axis=0)
    bb_center = bb_min + np.array(bb_max - bb_min) / 2

    world = World(
        bb_center,
        l,
        w,
        h
    )

    return world


# make a dictionary of axes options
def get_axes(world):
    axes = Axes(
        world,
        xtitle='y: North (m)',  # swap axis to plot correctly
        ytitle='x: East (m)',
        ztitle='z: TVD (m)',
        xTitleJustify='bottom-right',
        yTitleJustify='top-right',
        zTitleJustify='top-right',
        xyGrid2=True, xyGrid=False,
        zxGrid=True, yzGrid2=True,
        zxGridTransparent=True, yzGrid2Transparent=True,
        yzGrid=False,
        xLabelRotation=-1,
        yLabelRotation=1,
        zLabelRotation=1,
    )

    for a in axes.unpack():  # unpack the Assembly to access its elements
        if 'title' in a.name or 'NumericLabel' in a.name:
            a.mirror('y')
        if 'yNumericLabel' in a.name:
            a.scale(0.8)
    return axes


def get_lines(clearance):
    """
    Add lines per reference well interval between the closest points on the
    reference well and the offset well and color them according to the
    calculated Separation Factor (SF) between the two wells at these points.

    Parameters
    ----------
        clearance: welleng.clearance object

    Returns
    -------
        lines: vedo.Lines object
            A vedo.Lines object colored by the object's SF values.
    """
    c = clearance.SF
    start_points, end_points = clearance.get_lines()
    lines = Lines(start_points, end_points).cmap('hot_r', c, on='cells')
    lines.addScalarBar(title='SF')

    return lines
