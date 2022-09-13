try:
    import trimesh
    TRIMESH = True
except ImportError:
    TRIMESH = False
try:
    from vedo import Lines, Sphere, trimesh2vedo
    VEDO = True
except ImportError:
    VEDO = False
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False

from vtk import vtkCubeAxesActor, vtkInteractorStyleTerrain, vtkNamedColors
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

from .version import __version__ as VERSION


class Plotter(vtkRenderer):
    def __init__(self, data, **kwargs):  # noqa C901
        super().__init__()

        """
        A vtk wrapper for quickly visualizing well trajectories for QAQC
        purposes. Initiates a vtkRenderer instance and populates it with
        mesh data and a suitable axes for viewing well trajectory data.

        Parameters
        ----------
        data: obj or list(obj)
            A trimesh.Trimesh object or a list of trimesh.Trimesh
            objects or a trimesh.scene object
        names: list of strings (default: None)
            A list of names, index aligned to the list of well meshes.
        colors: list of strings (default: None)
            A list of color or colors. If a single color is listed then this is
            applied to all meshes in data, otherwise the list of colors is
            indexed to the list of meshes.
        """
        assert all((VEDO, TRIMESH)), \
            "ImportError: try pip install welleng[easy]"

        names = kwargs.get('names')

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

        colors = kwargs.get('colors')
        if colors is not None:
            if len(colors) == 1:
                colors = colors * len(meshes)
            else:
                assert len(colors) == len(meshes), \
                    "Colors must be length of meshes list, 1 else None"

        points = kwargs.get('points')
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

        self.namedColors = vtkNamedColors()

        self.colors = {}
        self.colors['background'] = kwargs.get('background', 'LightGrey')
        self.colors['background2'] = kwargs.get('background', 'Lavender')

        for mesh in meshes_vedo:
            if mesh is None:
                continue
            self.AddActor(mesh)

        for obj in kwargs.values():
            if isinstance(obj, list):
                for item in obj:
                    try:
                        self.AddActor(item)
                    except TypeError:
                        pass
            else:
                try:
                    self.AddActor(obj)
                except TypeError:
                    pass

        axes = CubeAxes(self, **kwargs)
        self.AddActor(axes)

        self.GetActiveCamera().Azimuth(30)
        self.GetActiveCamera().Elevation(30)
        self.GetActiveCamera().SetViewUp(0, 0, -1)
        self.GetActiveCamera().SetFocalPoint(axes.GetCenter())

        self.ResetCamera()
        self.SetBackground(
            self.namedColors.GetColor3d(self.colors['background'])
        )
        self.SetBackground2(
            self.namedColors.GetColor3d(self.colors['background2'])
        )

    def show(self, **kwargs):
        """
        Convenient method for opening a window to view the rendered scene.
        """
        setSize = kwargs.get('setSize', (1200, 900))

        renderWindow = vtkRenderWindow()

        renderWindow.AddRenderer(self)
        renderWindow.SetSize(*(setSize))
        renderWindow.SetWindowName(f'welleng {VERSION}')

        renderWindowInteractor = vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderWindowInteractor.SetInteractorStyle(vtkInteractorStyleTerrain())

        renderWindow.Render()
        self.GetActiveCamera().Zoom(0.8)

        interactive = kwargs.get('interactive', True)
        if interactive:
            renderWindowInteractor.Start()


class CubeAxes(vtkCubeAxesActor):
    def __init__(self, renderer, **kwargs):
        super().__init__()

        # Determine the bounds from the meshes/actors being plotted rounded
        # up to the nearest 100 units.
        bounds = np.array(renderer.ComputeVisiblePropBounds())

        with np.errstate(divide='ignore', invalid='ignore'):
            self.bounds = tuple(np.nan_to_num(
                (
                    np.ceil(np.abs(bounds) / 100) * 100
                ) * (bounds / np.abs(bounds))
            ))

        namedColors = vtkNamedColors()
        self.colors = {}

        for n in range(1, 4):
            self.colors[f'axis{n}Color'] = namedColors.GetColor3d(
                kwargs.get('axis1Color', 'DarkGrey')
            )

        self.SetLabelScaling(0, 0, 0, 0)

        # Try and prevent scientific numbering on axes
        self.SetXLabelFormat("%.0f")
        self.SetYLabelFormat("%.0f")
        self.SetZLabelFormat("%.0f")

        self.SetUseTextActor3D(1)

        self.SetBounds(self.bounds)
        self.SetCamera(renderer.GetActiveCamera())

        self.GetTitleTextProperty(0).SetColor(self.colors['axis1Color'])
        self.GetLabelTextProperty(0).SetColor(self.colors['axis1Color'])
        self.GetLabelTextProperty(0).SetOrientation(45.0)

        self.GetTitleTextProperty(1).SetColor(self.colors['axis2Color'])
        self.GetLabelTextProperty(1).SetColor(self.colors['axis2Color'])
        self.GetLabelTextProperty(1).SetOrientation(45.0)

        self.GetTitleTextProperty(2).SetColor(self.colors['axis3Color'])
        self.GetLabelTextProperty(2).SetColor(self.colors['axis3Color'])
        self.GetLabelTextProperty(2).SetOrientation(45.0)

        self.DrawXGridlinesOn()
        self.DrawYGridlinesOn()
        self.DrawZGridlinesOn()
        self.SetGridLineLocation(self.VTK_GRID_LINES_FURTHEST)

        self.XAxisMinorTickVisibilityOff()
        self.YAxisMinorTickVisibilityOff()
        self.ZAxisMinorTickVisibilityOff()

        units = kwargs.get('units', 'meters')
        self.SetXTitle('East')
        self.SetXUnits(units)
        self.SetYTitle('North')
        self.SetYUnits(units)
        self.SetZTitle('TVD')
        self.SetZUnits(units)

        self.SetFlyModeToClosestTriad()

        self.ForceOpaqueOff()


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
    **kwargs
):
    """
    A deprecated wrapper for the Plotter class, maintained only for
    compatability with older versions.

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
    for k, v in locals().items():
        if k != 'data' and k[0] != '_':
            kwargs.update({k: v})

    plt = Plotter(data, **kwargs)

    plt.show()


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
    assert VEDO, "ImportError: try pip install welleng[easy]"
    c = clearance.SF
    start_points, end_points = clearance.get_lines()
    lines = Lines(start_points, end_points).cmap('hot_r', c, on='cells')
    lines.addScalarBar(title='SF')

    return lines


def figure(obj, type='scatter3d', **kwargs):
    assert PLOTLY, "ImportError: try pip install plotly"
    func = {
        'scatter3d': _scatter3d,
        'mesh3d': _mesh3d,
        'panel': _panel
    }
    fig = func[type](obj, **kwargs)
    return fig


def _panel(survey, **kwargs):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Plan",
            f"Vertical Section: {np.degrees(survey.header.vertical_section_azimuth)} deg",
            "WE Section", "NS Section"
        )
        # shared_xaxes=True, shared_yaxes=True
    )

    fig.add_trace(
        go.Scatter(
            x=survey.e,
            y=survey.n,
            mode='lines',
            name='NE Plan',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=survey.vertical_section,
            y=survey.tvd,
            mode='lines',
            name='Plan',
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=survey.n,
            y=survey.tvd,
            mode='lines',
            name='NS Section',
            showlegend=False,
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=survey.e,
            y=survey.tvd,
            mode='lines',
            name='WE Section',
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_layout(
        xaxis=dict(
            title='West-East'
        ),
        yaxis=dict(
            title='North-South'
        ),
        xaxis2=dict(
            title='Outstep'
        ),
        yaxis2=dict(
            title='TVD',
            autorange="reversed",
            matches='y3'
        ),
        xaxis3=dict(
            title='West-East',
            matches='x'
        ),
        yaxis3=dict(
            title='TVD',
            autorange="reversed"
        ),
        xaxis4=dict(
            title='North-South',
            matches='y'
        ),
        yaxis4=dict(
            title='TVD',
            autorange="reversed",
            matches='y3'
        ),
    )

    return fig


def _scatter3d(survey, **kwargs):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=survey.e,
            y=survey.n,
            z=survey.tvd,
            name='survey',
            mode='lines',
            hoverinfo='skip'
        )
    )
    if not hasattr(survey, "interpolated"):
        survey.interpolated = [False] * len(survey.md)
    if survey.interpolated is None:
        survey.interpolated = [False] * len(survey.md)
    try:
        n, e, v, md, inc, azi = np.array([
            [n, e, v, md, inc, azi]
            for n, e, v, i, md, inc, azi in zip(
                survey.n, survey.e, survey.tvd, survey.interpolated,
                survey.md, survey.inc_deg, survey.azi_grid_deg
            )
            if bool(i) is True
        ]).T
        if n.size:
            text = [
                f"N: {n:.2f}m<br>E: {e:.2f}m<br>TVD: {v:.2f}m<br>"
                + f"MD: {md:.2f}m<br>INC: {inc:.2f}\xb0<br>AZI: {azi:.2f}\xb0"
                for n, e, v, md, inc, azi in zip(n, e, v, md, inc, azi)
            ]
            fig.add_trace(
                go.Scatter3d(
                    x=e,
                    y=n,
                    z=v,
                    name='interpolated',
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='blue',
                    ),
                    text=text,
                    hoverinfo='text'
                )
            )
    except ValueError:
        pass

    try:
        n, e, v, md, inc, azi = np.array([
            [n, e, v, md, inc, azi]
            for n, e, v, i, md, inc, azi in zip(
                survey.n, survey.e, survey.tvd, survey.interpolated,
                survey.md, survey.inc_deg, survey.azi_grid_deg
            )
            if bool(i) is False
        ]).T
        text = [
            f"N: {n:.2f}m<br>E: {e:.2f}m<br>TVD: {v:.2f}m<br>"
            + f"MD: {md:.2f}m<br>INC: {inc:.2f}\xb0<br>AZI: {azi:.2f}\xb0"
            for n, e, v, md, inc, azi in zip(n, e, v, md, inc, azi)
        ]
        fig.add_trace(
            go.Scatter3d(
                x=e,
                y=n,
                z=v,
                name='survey_point',
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                ),
                text=text,
                hoverinfo='text'
            )
        )
    except ValueError:
        pass

    fig = _update_fig(fig, kwargs)

    return fig


def _mesh3d(mesh, **kwargs):
    fig = go.Figure()
    vertices = mesh.vertices.reshape(-1, 3)
    faces = np.array(mesh.faces)
    n, e, v = vertices.T
    i, j, k = faces.T
    fig.add_trace(
        go.Mesh3d(
            x=e, y=n, z=v,
            i=i, j=j, k=k,
        )
    )
    if kwargs.get('edges'):
        tri_points = np.array([
            vertices[i] for i in faces.reshape(-1)
        ])
        n, e, v = tri_points.T
        fig.add_trace(
            go.Scatter3d(
                x=e, y=n, z=v,
                mode='lines',
            )
        )
    fig = _update_fig(fig, kwargs)

    return fig


def _update_fig(fig, kwargs):
    """
    Update the fig axis along with any user defined kwargs.
    """
    fig.update_scenes(
        zaxis_autorange="reversed",
        aspectmode='data',
        xaxis=dict(
            title='East (m)'
        ),
        yaxis=dict(
            title='North (m)',
        ),
        zaxis=dict(
            title="TVD (m)"
        )
    )
    for k, v in kwargs.items():
        if k == "layout":
            fig.update_layout(v)
        elif k == "traces":
            fig.update_traces(v)
        else:
            continue

    return fig
