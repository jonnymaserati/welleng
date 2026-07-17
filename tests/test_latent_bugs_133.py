"""Regression tests for the latent bugs surfaced while typing for issue #133.

Each test pins a bug that was found (and flagged in-comment) during the
survey/connector type-annotation pass and then fixed.
"""
from datetime import datetime

import numpy as np
import pytest

from welleng.node import Node
from welleng.survey import Survey, SurveyHeader, get_node
import welleng.connector as connector


def test_node_without_angles_sets_azi_deg_none():
    """node.py: the all-None branch used to set azi_rad twice and never set
    azi_deg, so accessing .azi_deg raised AttributeError."""
    node = Node()  # no inc/azi/vec
    assert node.inc_rad is None
    assert node.inc_deg is None
    assert node.azi_rad is None
    assert node.azi_deg is None  # previously AttributeError


def test_node_interpolated_is_first_class():
    """node.py: `interpolated` is now a declared attribute with a default."""
    assert Node().interpolated is False
    assert Node(interpolated=True).interpolated is True


def test_get_node_propagates_interpolated():
    """survey.get_node passes `interpolated` through to the Node."""
    s = Survey(
        md=[0.0, 100.0, 200.0],
        inc=[0.0, 5.0, 10.0],
        azi=[10.0, 15.0, 20.0],
    )
    assert get_node(s, 0, interpolated=True).interpolated is True
    assert get_node(s, 0).interpolated is False


def test_get_date_fallback_uses_today():
    """SurveyHeader._get_date sets survey_date; the magnetic-lookup fallback
    now reads that attribute instead of the function's (None) return value."""
    header = SurveyHeader.__new__(SurveyHeader)
    header._get_date(date=None)
    assert header.survey_date == datetime.today().strftime('%Y-%m-%d')


def test_survey_to_plan_dead_code_removed():
    """connector.survey_to_plan / _get_section were dead + broken (called a
    non-existent Connector.survey); they have been removed."""
    assert not hasattr(connector, "survey_to_plan")
    assert not hasattr(connector, "_get_section")


def test_transform_coordinates_still_works():
    """transform_coordinates no longer forwards *args (the pyproj itransform
    direction= collision); the documented call still works."""
    from welleng.survey import SurveyParameters
    calc = SurveyParameters('EPSG:23031')
    result = calc.transform_coordinates(
        coords=[(588319.02, 5770571.03)], to_projection='EPSG:32631'
    )
    assert np.asarray(result).shape[-1] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
