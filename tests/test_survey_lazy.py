"""Tests for the lazy (method-model) derivation of Survey toolface/rates +
vertical section (increment C).

These are deferred out of __init__ and computed on first access via __getattr__.
The tests pin: (1) they are NOT computed eagerly, (2) first access computes the
whole group, (3) the lazily-computed values equal the eager computation, and
(4) pickle/deepcopy still work.
"""
import copy
import pickle

import numpy as np
import pytest

from welleng.survey import Survey, SurveyHeader

_TOOLFACE = ("toolface", "turn_rate", "build_rate", "normals", "vec_radius_nev")


def _survey():
    h = SurveyHeader()
    h.azi_reference = "grid"
    return Survey(
        md=[0.0, 100.0, 250.0, 500.0, 800.0],
        inc=[0.0, 15.0, 45.0, 90.0, 90.0],
        azi=[0.0, 30.0, 60.0, 90.0, 120.0],
        header=h,
    )


def test_toolface_group_not_computed_eagerly():
    s = _survey()
    for attr in _TOOLFACE:
        assert attr not in s.__dict__, f"{attr} should be lazy, not eager"


def test_first_access_computes_whole_toolface_group():
    s = _survey()
    _ = s.toolface                       # trigger
    for attr in _TOOLFACE:
        assert attr in s.__dict__        # whole group populated in one shot


def test_vertical_section_is_eager_and_canonicalises_azi():
    """vertical_section stays EAGER because it canonicalises the azimuth at
    vertical stations (azi is undefined when inc==0), which interpolation relies
    on. Deferring it would leave the raw azimuth and diverge interpolate paths."""
    s = _survey()
    assert "vertical_section" in s.__dict__ and "hypot" in s.__dict__   # eager
    # a vertical-start survey: azi at the inc==0 stations is canonicalised to 0
    v = Survey(md=[0.0, 500.0, 1000.0, 2000.0], inc=[0.0, 0.0, 30.0, 90.0],
               azi=[45.0, 45.0, 45.0, 90.0], radius=10)
    assert np.allclose(v.azi_grid_rad[:2], 0.0)      # not the raw 45 deg


def test_lazy_values_equal_eager():
    """Lazily-computed toolface/rates/VS equal a direct eager computation."""
    lazy = _survey()
    tf, tr, br = lazy.toolface, lazy.turn_rate, lazy.build_rate
    vs = lazy.vertical_section

    eager = _survey()
    eager._get_toolface_and_rates()      # force eager
    eager._get_vertical_section()
    assert np.allclose(tf, eager.toolface, equal_nan=True)
    assert np.allclose(tr, eager.turn_rate, equal_nan=True)
    assert np.allclose(br, eager.build_rate, equal_nan=True)
    assert np.allclose(vs, eager.vertical_section, equal_nan=True)


def test_pickle_roundtrip_lazy_and_computed():
    # lazy (never accessed) survives pickle then computes on access
    s = _survey()
    r = pickle.loads(pickle.dumps(s))
    assert r.toolface.shape == s.toolface.shape
    # already-computed survives pickle
    s2 = _survey(); _ = s2.toolface
    r2 = pickle.loads(pickle.dumps(s2))
    assert np.allclose(r2.toolface, s2.toolface, equal_nan=True)


def test_deepcopy_lazy():
    s = _survey()
    c = copy.deepcopy(s)
    assert c.build_rate.shape == s.build_rate.shape


def test_getattr_raises_for_unknown():
    s = _survey()
    with pytest.raises(AttributeError):
        _ = s.definitely_not_an_attribute


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
