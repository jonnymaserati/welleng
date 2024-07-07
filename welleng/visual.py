try:
    import trimesh
    TRIMESH = True
except ImportError:
    TRIMESH = False
try:
    import vedo
    from vedo import Lines, Mesh
    from vedo import Plotter as VedoPlotter
    from vedo import Point, Text2D, mag, plotter_instance
    from vedo.addons import Icon, compute_visible_bounds
    from vedo.utils import buildPolyData
    VEDO = True
except ImportError:
    VEDO = False
import numpy as np
from scipy.spatial import KDTree

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False

from vtk import vtkAxesActor, vtkCubeAxesActor, vtkNamedColors

from . import mesh
from .version import __version__ as VERSION

# VEDO = False


class Plotter(VedoPlotter):
    def __init__(self, *args, **kwargs):
        """
        Notes
        -----
        On account of Z or TVD pointing down in the drilling world, the
        coordinate system is right handed. In order to map coordinates in the
        NEV (or North, East, Vertical) reference correctly, North coordinates
        are plotted on the X axis and East on the Y axis. Be mindful of this
        adding objects to a scene.
        """
        super().__init__(*args, **kwargs)

        self.wells = []

        pass

    def add(self, obj, *args, **kwargs) -> None:
        """Modified method to support direct plotting of
        ``welleng.mesh.WellMesh`` instances and for processing the callback
        to print well data when the pointer is hovered of a well trajectory.

        If the ``obj`` is a ``welleng.mesh.WellMesh`` instance, then the args
        and kwargs will be passed to the `vedo.Mesh` instance to facilate e.g.
        color options etc.

        Notes
        -----
        ``welleng.mesh.WellMesh`` creates ``trimesh.Mesh`` instances, a legacy
        of using the ``trimesh`` library for detecting mesh collisions when
        developing automated well trajectory planning. Therefore, to visualize
        the meshes with ``vedo`` and ``vtk``, the meshes need to be converted.

        Meshes in ``welleng`` typically reference an 'NEV' coordinate system,
        which is [North, East, Vertical]. To map correctly to ``vtk``, North
        needs to be mapped to X and East to Y on account of Z pointing down.
        """
        if isinstance(obj, mesh.WellMesh):
            poly = buildPolyData(obj.mesh.vertices, obj.mesh.faces)
            vedo_mesh = Mesh(poly, *args, **kwargs)
            setattr(obj, 'vedo_mesh', vedo_mesh)
            self.wells.append(obj)
            super().add(obj.vedo_mesh)
        else:
            super().add(obj)

        pass

    def _initiate_axes_actor(self):
        plt = vedo.plotter_instance
        r = plt.renderers.index(plt.renderer)

        axact = vtkAxesActor()
        axact.SetShaftTypeToCylinder()
        axact.SetCylinderRadius(0.03)
        axact.SetXAxisLabelText("N")
        axact.SetYAxisLabelText("E")
        axact.SetZAxisLabelText("V")
        axact.GetXAxisShaftProperty().SetColor(1, 0, 0)
        axact.GetYAxisShaftProperty().SetColor(0, 1, 0)
        axact.GetZAxisShaftProperty().SetColor(0, 0, 1)
        axact.GetXAxisTipProperty().SetColor(1, 0, 0)
        axact.GetYAxisTipProperty().SetColor(0, 1, 0)
        axact.GetZAxisTipProperty().SetColor(0, 0, 1)
        bc = np.array(plt.renderer.GetBackground())
        if np.sum(bc) < 1.5:
            lc = (1, 1, 1)
        else:
            lc = (0, 0, 0)
        axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
        axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
        axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().BoldOff()
        axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
        axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
        axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().ItalicOff()
        axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        axact.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
        axact.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
        axact.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(lc)
        axact.PickableOff()
        icn = Icon(axact, size=0.1)
        plt.axes_instances[r] = icn
        icn.SetInteractor(plt.interactor)
        icn.EnabledOn()
        icn.InteractiveOff()
        plt.widgets.append(icn)

    def _pointer_callback(self, event):
        i = event.at
        pt2d = event.picked2d
        objs = self.at(i).objects
        pt3d = self.at(i).compute_world_coordinate(pt2d, objs=objs)
        if mag(pt3d) < 0.01:
            if self.pointer is None:
                return
            # if self.pointer in self.at(i).actors:
            self.at(i).remove(self.pointer)
            self.pointer = None
            self.pointer_text.text('')
            self.render()
            return
        if self.pointer is None:
            self.pointer = Point().color('red').pos(pt3d)
        else:
            self.pointer.pos(pt3d)
        self.at(i).add(self.pointer)

        well_data = self._get_closest_well(pt3d, objs)
        if well_data is None:
            self.pointer_text.text(f'point coordinates: {np.round(pt3d, 3)}')
        else:
            survey = well_data.get('well').s
            idx = well_data.get('idx_survey')
            name = survey.header.name
            md = survey.md[idx]
            inc = survey.inc_deg[idx]
            azi_grid = survey.azi_grid_deg[idx]
            dls = survey.dls[idx]
            self.pointer_text.text(f'''
                well name: {name}\n
                md: {md:.2f}\t inc: {inc:.2f}\t azi: {azi_grid:.2f}\t dls: {dls:.2f}\n
                point coordinates: {np.round(pt3d, 3)}
            ''')
        self.render()

    def _well_vedo_meshes(self):
        return [well.vedo_mesh for well in self.wells]

    def _get_closest_well(self, pos, objs) -> dict:
        wells = [
            well for well in self.wells
            if well.vedo_mesh in objs
        ]
        if not bool(wells):
            return

        results = np.zeros((len(wells), 3))
        for i, well in enumerate(wells):
            tree = KDTree(well.vertices.reshape(-1, 3))
            distance, idx_vertices = tree.query(pos)
            results[i] = np.array([distance, idx_vertices, well.n_verts])

        winner = np.argmin(results[:, 0])
        distance, idx_vertices, n_verts = results[winner]

        return {
            'well': wells[winner],
            'distance': distance,
            'idx_vertices': int(idx_vertices),
            'n_verts': int(n_verts),
            'idx_survey': int(idx_vertices // n_verts)
        }

    def show(self, axes=None, *args, **kwargs):
        # check if there's an axes and if so remove them
        if self.axes is not None:
            self.remove(self.axes)

        self._initiate_axes_actor()

        self.add_callback('mouse move', self._pointer_callback)
        self.pointer_text = Text2D("", pos='bottom-right', s=0.5, c='black')
        self.add(self.pointer_text)
        self.pointer = None

        self = super().show(
            viewup=[0, 0, -1], mode=8,
            axes=CubeAxes() if axes is None else axes,
            title=f'welleng {VERSION}',
            *args, **kwargs
        )

        return self


class CubeAxes(vtkCubeAxesActor):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        # # Determine the bounds from the meshes/actors being plotted rounded
        # # up to the nearest 100 units.
        # bounds = np.array(renderer.ComputeVisiblePropBounds())

        # with np.errstate(divide='ignore', invalid='ignore'):
        #     self.bounds = tuple(np.nan_to_num(
        #         (
        #             np.ceil(np.abs(bounds) / 100) * 100
        #         ) * (bounds / np.abs(bounds))
        #     ))

        plt = vedo.plotter_instance
        r = plt.renderers.index(plt.renderer)
        vbb = compute_visible_bounds()[0]
        self.SetBounds(vbb)
        self.SetCamera(plt.renderer.GetActiveCamera())

        namedColors = vtkNamedColors()
        self.colors = {}

        for n in range(1, 4):
            self.colors[f'axis{n}Color'] = namedColors.GetColor3d(
                kwargs.get(f'axis{n}Color', 'Black')
            )

        # self.SetUseTextActor3D(1)

        # self.SetBounds(self.bounds)
        # self.SetCamera(renderer.GetActiveCamera())

        self.GetTitleTextProperty(0).SetColor(self.colors['axis1Color'])
        self.GetLabelTextProperty(0).SetColor(self.colors['axis1Color'])
        self.GetLabelTextProperty(0).SetOrientation(45.0)

        self.GetTitleTextProperty(1).SetColor(self.colors['axis2Color'])
        self.GetLabelTextProperty(1).SetColor(self.colors['axis2Color'])
        self.GetLabelTextProperty(1).SetOrientation(45.0)

        self.GetTitleTextProperty(2).SetColor(self.colors['axis3Color'])
        self.GetLabelTextProperty(2).SetColor(self.colors['axis3Color'])
        self.GetLabelTextProperty(2).SetOrientation(45.0)

        self.SetGridLineLocation(self.VTK_GRID_LINES_FURTHEST)
        for a in ('X', 'Y', 'Z'):
            getattr(self, f'Get{a}AxesLinesProperty')().SetColor(
                namedColors.GetColor3d('Black')
            )
            getattr(self, f'SetDraw{a}Gridlines')(1)
            getattr(self, f'Get{a}AxesGridlinesProperty')().SetColor(
                namedColors.GetColor3d('Grey')
            )
            getattr(self, f'{a}AxisMinorTickVisibilityOff')()

        # self.DrawXGridlinesOn()
        # self.DrawYGridlinesOn()
        # self.DrawZGridlinesOn()
        # self.SetGridLineLocation(self.VTK_GRID_LINES_FURTHEST)

        # self.XAxisMinorTickVisibilityOff()
        # self.YAxisMinorTickVisibilityOff()
        # self.ZAxisMinorTickVisibilityOff()

        units = kwargs.get('units', None)
        self.SetXTitle('N')
        self.SetXUnits(units)
        self.SetYTitle('E')
        self.SetYUnits(units)
        self.SetZTitle('TVD')
        self.SetZUnits(units)
        # self.SetTickLocation(self.VTK_GRID_LINES_FURTHEST)

        # self.ForceOpaqueOff()
        self.SetFlyModeToClosestTriad()
        # Try and prevent scientific numbering on axes
        self.SetLabelScaling(0, 0, 0, 0)
        self.SetXLabelFormat("%.0f")
        self.SetYLabelFormat("%.0f")
        self.SetZLabelFormat("%.0f")

        plt.axes_instances[r] = self
        plt.renderer.AddActor(self)


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
    c = clearance.sf
    start_points, end_points = clearance.get_lines()
    lines = Lines(start_points, end_points).cmap('hot_r', c, on='cells')
    lines.add_scalarbar(title='SF')

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
