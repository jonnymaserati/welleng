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

try:
    from vtk import vtkAxesActor, vtkCubeAxesActor, vtkNamedColors
    VTK = True
except ImportError:
    VTK = False

from . import mesh
from .version import __version__ as VERSION


if VEDO:
    class Plotter(VedoPlotter):
        def __init__(self, *args, **kwargs):
            """
            Notes
            -----
            On account of Z or TVD pointing down in the drilling world, the
            coordinate system is right handed. In order to map coordinates in the
            NEV (or North, East, Vertical) reference correctly, North coordinates
            are plotted on the X axis and East on the Y axis. Be mindful of this
            when adding objects to a scene.
            """
            super().__init__(*args, **kwargs)
            self.wells = []

        def add(self, obj, *args, **kwargs) -> None:
            """Modified method to support direct plotting of
            ``welleng.mesh.WellMesh`` instances and for processing the callback
            to print well data when the pointer is hovered over a well trajectory.

            If the ``obj`` is a ``welleng.mesh.WellMesh`` instance, then the args
            and kwargs will be passed to the `vedo.Mesh` instance to facilitate e.g.
            color options etc.

            Notes
            -----
            ``welleng.mesh.WellMesh`` stores geometry as a lightweight namespace
            with ``.vertices`` and ``.faces`` arrays.  These are converted to a
            vedo/VTK polydata representation here for rendering.

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

        def _initiate_axes_actor(self):
            if not VTK:
                return
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
            lc = (1, 1, 1) if np.sum(bc) < 1.5 else (0, 0, 0)
            for axis in ('X', 'Y', 'Z'):
                cap = getattr(axact, f'Get{axis}AxisCaptionActor2D')()
                prop = cap.GetCaptionTextProperty()
                prop.BoldOff()
                prop.ItalicOff()
                prop.ShadowOff()
                prop.SetColor(lc)
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
                self.pointer_text.text(
                    f'well name: {name}\n'
                    f'md: {md:.2f}\t inc: {inc:.2f}\t'
                    f' azi: {azi_grid:.2f}\t dls: {dls:.2f}\n'
                    f'point coordinates: {np.round(pt3d, 3)}'
                )
            self.render()

        def _well_vedo_meshes(self):
            return [well.vedo_mesh for well in self.wells]

        def _get_closest_well(self, pos, objs) -> dict:
            wells = [
                well for well in self.wells
                if well.vedo_mesh in objs
            ]
            if not bool(wells):
                return None

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
            if self.axes is not None:
                self.remove(self.axes)

            self._initiate_axes_actor()

            self.add_callback('mouse move', self._pointer_callback)
            self.pointer_text = Text2D("", pos='bottom-right', s=0.5, c='black')
            self.add(self.pointer_text)
            self.pointer = None

            if axes is None:
                axes = CubeAxes() if VTK else 0

            return super().show(
                viewup=[0, 0, -1], mode=8,
                axes=axes,
                title=f'welleng {VERSION}',
                *args, **kwargs
            )


if VTK:
    class CubeAxes(vtkCubeAxesActor):
        def __init__(self, **kwargs):
            super().__init__()

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

            units = kwargs.get('units', None)
            self.SetXTitle('N')
            self.SetXUnits(units)
            self.SetYTitle('E')
            self.SetYUnits(units)
            self.SetZTitle('TVD')
            self.SetZUnits(units)

            self.SetFlyModeToClosestTriad()
            self.SetLabelScaling(0, 0, 0, 0)
            self.SetXLabelFormat("%.0f")
            self.SetYLabelFormat("%.0f")
            self.SetZLabelFormat("%.0f")

            plt.axes_instances[r] = self
            plt.renderer.AddActor(self)


def plot(data, **kwargs):
    """
    Deprecated wrapper for the Plotter class, maintained only for
    compatibility with older versions.

    Parameters
    ----------
    data: WellMesh or list of WellMesh
    """
    assert VEDO, "ImportError: try pip install welleng[easy]"
    plt = Plotter()
    if isinstance(data, (list, tuple)):
        for item in data:
            plt.add(item, **kwargs)
    else:
        plt.add(data, **kwargs)
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
    )

    fig.add_trace(
        go.Scatter(x=survey.e, y=survey.n, mode='lines',
                   name='NE Plan', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=survey.vertical_section, y=survey.tvd, mode='lines',
                   name='Plan', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=survey.n, y=survey.tvd, mode='lines',
                   name='NS Section', showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=survey.e, y=survey.tvd, mode='lines',
                   name='WE Section', showlegend=False),
        row=2, col=1
    )

    fig.update_layout(
        xaxis=dict(title='West-East'),
        yaxis=dict(title='North-South'),
        xaxis2=dict(title='Outstep'),
        yaxis2=dict(title='TVD', autorange="reversed", matches='y3'),
        xaxis3=dict(title='West-East', matches='x'),
        yaxis3=dict(title='TVD', autorange="reversed"),
        xaxis4=dict(title='North-South', matches='y'),
        yaxis4=dict(title='TVD', autorange="reversed", matches='y3'),
    )

    return fig


def _scatter3d(survey, **kwargs):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=survey.e, y=survey.n, z=survey.tvd,
            name='survey', mode='lines', hoverinfo='skip'
        )
    )

    if not hasattr(survey, "interpolated") or survey.interpolated is None:
        survey.interpolated = [False] * len(survey.md)

    for is_interpolated, trace_name, color in (
        (True, 'interpolated', 'blue'),
        (False, 'survey_point', 'red'),
    ):
        try:
            n, e, v, md, inc, azi = np.array([
                [n, e, v, md, inc, azi]
                for n, e, v, i, md, inc, azi in zip(
                    survey.n, survey.e, survey.tvd, survey.interpolated,
                    survey.md, survey.inc_deg, survey.azi_grid_deg
                )
                if bool(i) is is_interpolated
            ]).T
            if n.size:
                text = [
                    f"N: {n:.2f}m<br>E: {e:.2f}m<br>TVD: {v:.2f}m<br>"
                    + f"MD: {md:.2f}m<br>INC: {inc:.2f}\xb0<br>AZI: {azi:.2f}\xb0"
                    for n, e, v, md, inc, azi in zip(n, e, v, md, inc, azi)
                ]
                fig.add_trace(
                    go.Scatter3d(
                        x=e, y=n, z=v,
                        name=trace_name, mode='markers',
                        marker=dict(size=5, color=color),
                        text=text, hoverinfo='text'
                    )
                )
        except ValueError:
            pass

    return _update_fig(fig, kwargs)


def _mesh3d(mesh, **kwargs):
    fig = go.Figure()
    vertices = mesh.vertices.reshape(-1, 3)
    faces = np.array(mesh.faces)
    n, e, v = vertices.T
    i, j, k = faces.T
    fig.add_trace(go.Mesh3d(x=e, y=n, z=v, i=i, j=j, k=k))
    if kwargs.get('edges'):
        tri_points = vertices[faces.reshape(-1)]
        n, e, v = tri_points.T
        fig.add_trace(go.Scatter3d(x=e, y=n, z=v, mode='lines'))
    return _update_fig(fig, kwargs)


def _update_fig(fig, kwargs):
    """Update figure axes and apply any user-defined layout/trace overrides."""
    fig.update_scenes(
        zaxis_autorange="reversed",
        aspectmode='data',
        xaxis=dict(title='East (m)'),
        yaxis=dict(title='North (m)'),
        zaxis=dict(title='TVD (m)')
    )
    if "layout" in kwargs:
        fig.update_layout(kwargs["layout"])
    if "traces" in kwargs:
        fig.update_traces(kwargs["traces"])
    return fig
