import inspect
import sys
from welleng.connector import Connector
from welleng.survey import Survey
import numpy as np


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

    c.survey()


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
    assert isinstance(c.survey(), Survey)


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


def one_function_to_run_them_all():
    """
    Function to gather the test functions so that they can be tested by
    running this module.

    https://stackoverflow.com/questions/18907712/python-get-list-of-all-
    functions-in-current-module-inspecting-current-module
    """
    test_functions = [
        obj for name, obj in inspect.getmembers(sys.modules[__name__])
        if (inspect.isfunction(obj)
            and name.startswith('test')
            and name != 'all')
    ]

    [f() for f in test_functions]


if __name__ == '__main__':
    one_function_to_run_them_all()
