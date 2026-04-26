import welleng as we
import numpy as np

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

def test_survey_interpolate_survey_vs_interpolate_mds(step=30):
    global SURVEY
    mds = np.arange(SURVEY.md[0], SURVEY.md[-1], step)
    survey_interp = we.survey.interpolate_mds(SURVEY, mds)
    assert isinstance(survey_interp, we.survey.Survey)
    
    survey_interp_1 = we.survey.interpolate_survey(SURVEY, step=step)
    assert np.allclose(survey_interp.md, survey_interp_1.md)
    assert np.allclose(survey_interp.azi_grid_rad, survey_interp_1.azi_grid_rad)
    assert np.allclose(survey_interp.inc_rad, survey_interp_1.inc_rad)
    assert np.allclose(survey_interp.pos_xyz, survey_interp_1.pos_xyz)
    assert np.allclose(survey_interp.pos_nev, survey_interp_1.pos_nev)
    assert np.allclose(survey_interp.dogleg, survey_interp_1.dogleg)
    assert np.all(survey_interp.interpolated == survey_interp_1.interpolated)

    survey_interp = SURVEY.interpolate_mds(mds)
    assert isinstance(survey_interp, we.survey.Survey)
    
    survey_interp_1 = SURVEY.interpolate_survey(step=step)
    assert np.allclose(survey_interp.md, survey_interp_1.md)
    assert np.allclose(survey_interp.azi_grid_rad, survey_interp_1.azi_grid_rad)
    assert np.allclose(survey_interp.inc_rad, survey_interp_1.inc_rad)
    assert np.allclose(survey_interp.pos_xyz, survey_interp_1.pos_xyz)
    assert np.allclose(survey_interp.pos_nev, survey_interp_1.pos_nev)
    assert np.allclose(survey_interp.dogleg, survey_interp_1.dogleg)
    assert np.all(survey_interp.interpolated == survey_interp_1.interpolated)

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

test_survey_interpolate_survey_vs_interpolate_mds()