"""Tests for Survey.highside_vec_nev (wellpath-designer high-side lock)."""
import numpy as np
import pytest

from welleng.survey import Survey
from welleng.utils import get_transform


def _survey():
    return Survey(
        md=[0.0, 500.0, 1000.0, 1500.0, 2000.0],
        inc=[0.0, 15.0, 45.0, 90.0, 90.0],
        azi=[0.0, 30.0, 120.0, 210.0, 300.0],
    )


def test_highside_is_unit_length():
    hs = _survey().highside_vec_nev()
    assert np.allclose(np.linalg.norm(hs, axis=1), 1.0)


def test_highside_perpendicular_to_axis():
    """High side is perpendicular to the wellbore axis (unit direction vector)."""
    s = _survey()
    hs = s.highside_vec_nev()
    axis = s.vec_nev                      # (n, 3) unit axis vectors in NEV
    dots = np.einsum('ij,ij->i', hs, axis)
    assert np.allclose(dots, 0.0, atol=1e-9)


def test_highside_points_up_when_horizontal():
    """A horizontal well's high side points straight up: V-component = -1."""
    s = _survey()
    hs = s.highside_vec_nev()
    # stations 3 and 4 are inc=90 deg
    assert hs[3] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
    assert hs[4] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_highside_matches_hla_transform_row():
    """It is exactly the H (high-side) row of the NEV->HLA transform."""
    s = _survey()
    trans = get_transform(s.survey_rad)   # (n, 3, 3); row 0 is the H basis in NEV
    assert np.allclose(s.highside_vec_nev(), trans[:, 0, :])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
