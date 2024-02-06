import numpy as np

from welleng.connector import Connector
from welleng.survey import Survey, from_connections
from welleng.utils import get_arc, radius_from_dls
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

DEBUG = True

if DEBUG:
    import welleng as we
    import vedo


def clc_generator(
    radius: float, toolface1: float, dogleg1: float,
    toolface2: float, dogleg2: float,
    pos1: NDArray | None = None, vec1: NDArray | None = None,
    md1: float | None = None, tangent: float | None = None,
) -> dict:
    """Generates a curve, line, curve trajectory from the provided params.
    """
    pos1 = np.array([0.0, 0.0, 0.0]) if pos1 is None else pos1
    vec1 = np.array([0.0, 0.0, 1.0]) if vec1 is None else vec1
    md1 = 0 if md1 is None else md1
    pos2, vec2, delta_md = get_arc(
        dogleg1, radius, toolface1, pos1, vec1
    )
    md2 = md1 + delta_md

    pos3 = pos2 if tangent is None else pos2 + tangent * vec2
    md3 = md2 if tangent is None else md2 + tangent
    vec3 = vec2

    pos4, vec4, delta_md = get_arc(
        dogleg2, radius, toolface2, pos3, vec3
    )
    md4 = md3 + delta_md

    return dict(
        radius=radius,
        pos1=pos1, vec1=vec1, md1=md1, toolface1=toolface1, dogleg1=dogleg1,
        pos2=pos2, vec2=vec2, md2=md2,
        pos3=pos3, vec3=vec3, md3=md3, toolface2=toolface2, dogleg2=dogleg2,
        pos4=pos4, vec4=vec4, md4=md4,
    )


def reference_plot(data, arrow_length=100):
    starts = np.array([v for k, v in data.items() if 'pos' in k])
    ends = starts + np.array([v for k, v in data.items() if 'vec' in k]) * arrow_length
    plot = we.visual.Plotter()
    plot.add(vedo.Spheres(starts, r=10))
    plot.add(vedo.Arrows(starts, ends, thickness=5))
    plot.add(vedo.Torus(pos=data.get('pos1'), r1=data.get('radius'), r2=data.get('radius'), alpha=0.1))
    inc, azi = we.utils.get_angles(data.get('vec2'), nev=True)[0]
    plot.add(
        vedo.Torus(
            r1=data.get('radius'), r2=data.get('radius'), alpha=0.1
        ).rotate_y(inc, rad=True).rotate_z(azi, rad=True).pos(data.get('pos2'))
    )
    survey = get_mincurve(data)
    plot.add(
        we.mesh.WellMesh(survey, method='circle')
    )
    return plot


def get_mincurve(data):
    md = np.array([v for k, v in data.items() if 'md' in k])
    vec = np.array([v for k, v in data.items() if 'vec' in k])
    inc, azi = we.utils.get_angles(vec, nev=True).T
    survey = we.survey.Survey(md, inc, azi, deg=False, radius=10).interpolate_survey(step=30)

    return survey


def debug_plot(data, connector):
    plot = reference_plot(data)
    connections = []
    connections.append(Connector(
        pos1=data.get('pos1'), vec1=data.get('vec1'),
        pos2=data.get('pos2'), vec2=data.get('vec2'),
        dls_design=1e-8,
        # degrees=True,
        # force_min_curve=True
    ))
    if not np.allclose(data.get('pos2'), data.get('pos3')):
        connections.append(Connector(
            node1=connections[-1].node_end,
            pos2=data.get('pos3'),
            vec2=data.get('vec3'),
            dls_design=1e-8,
            # degrees=True,
            # force_min_curve=True
        ))
    connections.append(Connector(
        node1=connections[-1].node_end,
        pos2=data.get('pos4'),
        vec2=data.get('vec4'),
        dls_design=1e-8,
        # degrees=True,
        # force_min_curve=True
    ))
    survey_reference = we.survey.from_connections(
        connections, survey_header=we.survey.SurveyHeader(name='reference'),
        step=30
    )
    survey = we.survey.from_connections(
        [connector],
        survey_header=we.survey.SurveyHeader(
            name='connector', azi_reference='grid'
        ),
        step=30
    )
    plot.add(
        we.mesh.WellMesh(survey_reference, method='circle'), c='blue'
    )
    plot.add(
        we.mesh.WellMesh(survey, method='circle'), c='red'
    )
    plot.show()


def test_clc(n: int = 1, radius: float | None =None):
    rng = np.random.default_rng(seed=42)

    radius = radius_from_dls(3.0) if radius is None else radius

    dogleg1, dogleg2 = (rng.random(n * 2) * np.pi * 2).reshape((2, -1))
    toolface1, toolface2 = (
        (rng.random(n * 2) * np.pi * 2) - np.pi
    ).reshape((2, -1))

    reference = [
        clc_generator(radius, tf1, dg1, tf2, dg2)
        for tf1, dg1, tf2, dg2 in zip(toolface1, dogleg1, toolface2, dogleg2)
    ]

    for data in reference:
        connector = Connector(
            pos1=data.get('pos1'),
            vec1=data.get('vec1'),
            pos2=data.get('pos4'),
            vec2=data.get('vec4'),
            dls_design=2.0
        )
        if DEBUG:
            debug_plot(data, connector)

        pass

    return


def test_md_hold():
    # test hold with only md provided
    c = Connector(
        vec1=[0, 0, 1],
        md2=500,
    )
    assert (
        c.inc_target == c.inc1
        and c.azi_target == c.azi1
        and c.pos_target[2] == c.md_target
    ), "Failed c1"
    assert c.method == 'hold', "Unexpected method"

    assert isinstance(from_connections(c), Survey)

def test_md_and_vec():
    # test with md2 and vec2 provided (minimum curvature)
    c = Connector(
        vec1=[0, 0, 1],
        md2=1000,
        vec2=[0, 1, 0]
    )
    assert c.method == 'min_curve'

def test_pos():
    # test with pos2 provided (minimum distance)
    c = Connector(
        vec1=[0, 0, 1],
        pos2=[100, 100, 1000],
    )
    assert c.md_target > c.pos1[2], "Failed c3"

def test_pos_and_dls():
    # test with pos2 needing more aggressive dls (minimum curvature)
    c = Connector(
        vec1=[0, 0, 1],
        pos2=[200, 400, 200]
    )
    assert c.method == 'min_curve_to_target'

def test_pos_and_vec():
    # test with pos2 and vec2 provided
    vec1 = [-1, -1, 1]
    vec2 = [1, -1, 0]
    c = Connector(
        pos1=[0., 0., 0],
        vec1=vec1 / np.linalg.norm(vec1),
        pos2=[0., 1000., 500.],
        vec2=vec2 / np.linalg.norm(vec2),
    )
    assert c.method == 'curve_hold_curve'

    # test if interpolator and survey functions are working
    assert isinstance(from_connections(c, step=30), Survey)

def test_pos_inc_azi():
    # test with pos2, inc1 and azi1 provided
    c = Connector(
        pos1=[0., 0., 0],
        inc1=0.,
        azi1=90,
        pos2=[1000., 1000., 1000.],
        vec2=[0., 0., 1.],
    )
    assert c.method == 'curve_hold_curve'

def test_dls2():
    # test with different dls for second curve section
    c = Connector(
        pos1=[0., 0., 0],
        vec1=[0., 0., 1.],
        pos2=[0., 100., 1000.],
        vec2=[0., 0., 1.],
        dls_design2=5
    )
    assert c.radius_design2 < c.radius_design

def test_radius_critical():
    # test with dls_critical requirement (actual dls < dls_design)
    c = Connector(
        pos1=[0., 0., 0],
        vec1=[0., 0., 1.],
        pos2=[0., 100., 100.],
        vec2=[0., 0., 1.],
    )
    assert c.radius_critical < c.radius_design

def test_min_curve():
    # test min_curve (inc2 provided)
    c = Connector(
        pos1=[0., 0., 0],
        vec1=[0., 0., 1.],
        inc2=30,
    )
    assert c.method == 'min_curve'

def test_radius_critical_with_min_curve():
    # test min_curve with md less than required radius
    c = Connector(
        pos1=[0., 0., 0],
        inc1=0,
        azi1=0,
        md2=500,
        inc2=90,
        azi2=0,
    )
    assert c.radius_critical < c.radius_design


def main():
    test_clc()


if __name__ == "__main__":
    main()
