"""Contract tests for welleng.target.Target (geometry primitive).

Target is a pure data/geometry object: it must construct without any optional
dependency (vedo), back the documented viz signature, expose a solver-facing
shape/dim/position/cov contract, and be the single class re-exported by the WBP
exchange module.
"""
import numpy as np
import pytest

import welleng.target as target_module
from welleng.target import SHAPES, Target


def test_wbp_reexports_the_same_class():
    from welleng.exchange.wbp import Target as WbpTarget
    assert WbpTarget is Target


def test_constructs_without_vedo(monkeypatch):
    # geometry construction must not depend on vedo being importable
    monkeypatch.setattr(target_module, "VEDO", False)
    t = Target('t', 0, 0, 100, 'circle', radius=10)
    assert t.geometry['radius'] == 10


def test_backward_compatible_viz_signature():
    t = Target('t1', 100, 50, 400, 'circle', radius=30)
    assert t.shape == 'circle'
    assert t.geometry['radius'] == 30
    assert np.allclose(t.position, [100, 50, 400])
    assert t.dim == 2


def test_location_kwarg_and_property_mirror():
    t = Target('t2', location=[1, 2, 3], shape='rectangle', pos1=[0, 0], pos2=[10, 5])
    assert np.allclose(t.location, [1, 2, 3])
    assert np.allclose(t.position, [1, 2, 3])
    # setter round-trips
    t.location = [5, 6, 7]
    assert np.allclose(t.position, [5, 6, 7])


def test_name_only_construction():
    # the WBP parser path: Target(name), then attributes set later
    t = Target('t3')
    assert t.shape is None
    assert t.geometry == {}
    assert t.position is None
    assert t.dim is None


def test_explicit_geometry_dict_not_aliased_across_instances():
    def geom():
        return {'color': {'color': None}}
    a = Target('a', geometry=geom())
    b = Target('b', geometry=geom())
    a.geometry['color']['color'] = 7
    assert b.geometry['color']['color'] is None


def test_gaussian_target_carries_mean_and_cov():
    cov = np.diag([4.0, 9.0, 16.0])
    g = Target('g', 0, 0, 500, 'gaussian', cov=cov)
    assert g.shape == 'gaussian'
    assert g.dim == 3
    assert np.allclose(g.mean, [0, 0, 500])
    assert np.allclose(g.cov, cov)


def test_hard_target_has_no_mean_or_cov():
    t = Target('t', 0, 0, 0, 'circle', radius=5)
    assert t.mean is None
    assert t.cov is None


def test_shape_dim_mapping_covers_all_shapes():
    for shape in SHAPES:
        t = Target('t', 0, 0, 0, shape) if shape != 'gaussian' else \
            Target('t', 0, 0, 0, shape, cov=np.eye(3))
        assert isinstance(t.dim, int)


def test_invalid_shape_rejected():
    with pytest.raises(ValueError):
        Target('x', 0, 0, 0, 'banana')
