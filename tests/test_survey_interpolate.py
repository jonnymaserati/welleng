import numpy as np
import welleng as we
from welleng.survey import (
    _interpolate_survey, _interpolate_pos_nev, slice_survey,
)

SURVEY = we.survey.Survey(
    md=[0, 500, 1000, 2000, 2500, 3500],
    inc=[0, 0, 30, 90, 100, 80],
    azi=[45, 45, 45, 90, 90, 180],
    radius=10,
)


def test_survey_interpolate_survey(step=30):

    survey_interp = we.survey.interpolate_survey(SURVEY, step=step)
    assert isinstance(survey_interp, we.survey.Survey)

    survey_interp = SURVEY.interpolate_survey(step=step)
    assert isinstance(survey_interp, we.survey.Survey)

def test_survey_interpolate_survey_tvd(step=10):

    survey_interp = SURVEY.interpolate_survey(step=30)
    survey_interp_tvd = we.survey.interpolate_survey_tvd(
        survey_interp, step=step
    )
    assert isinstance(survey_interp_tvd, we.survey.Survey)

    survey_interp_tvd = survey_interp.interpolate_survey_tvd(step=step)
    assert isinstance(survey_interp_tvd, we.survey.Survey)

def test_interpolate_md(md=800):

    node = SURVEY.interpolate_md(md=md)
    assert isinstance(node, we.node.Node)

def test_interpolate_tvd(tvd=800):

    node = SURVEY.interpolate_tvd(tvd=tvd)
    assert isinstance(node, we.node.Node)


def test_interpolate_pos_nev_matches_interpolate_survey():
    """The lightweight, position-only `_interpolate_pos_nev` (used by
    MeshClearance's closest-point search for speed, instead of building a full
    Survey) must return the same NEV position as `_interpolate_survey`. Guards
    the clearance performance optimisation (covers both the curved and the
    tangent/dogleg-zero branches)."""
    for i in range(len(SURVEY.md) - 1):
        s = slice_survey(SURVEY, i, i + 2)
        dmd = float(s.md[1] - s.md[0])
        for frac in (0.0, 0.25, 0.5, 0.9, 1.0):
            x = frac * dmd
            full = _interpolate_survey(s, x)
            pos_full = np.array([full.n, full.e, full.tvd]).T[1]
            pos_light = _interpolate_pos_nev(s, x, 0)
            assert np.allclose(pos_light, pos_full, atol=1e-7), (i, frac)
