import welleng as we

SURVEY = we.survey.Survey(
    md=[0, 500, 1000, 2000, 2500, 3500],
    inc=[0, 0, 30, 90, 100, 80],
    azi=[45, 45, 45, 90, 90, 180],
    radius=10,
)


def test_survey_interpolate_survey(step=30):
    global SURVEY
    survey_interp = we.survey.interpolate_survey(SURVEY, step=step)
    assert isinstance(survey_interp, we.survey.Survey)

    survey_interp = SURVEY.interpolate_survey(step=step)
    assert isinstance(survey_interp, we.survey.Survey)

def test_survey_interpolate_survey_tvd(step=10):
    global SURVEY
    survey_interp = SURVEY.interpolate_survey(step=30)
    survey_interp_tvd = we.survey.interpolate_survey_tvd(
        survey_interp, step=step
    )
    assert isinstance(survey_interp_tvd, we.survey.Survey)

    survey_interp_tvd = survey_interp.interpolate_survey_tvd(step=step)
    assert isinstance(survey_interp_tvd, we.survey.Survey)

def test_interpolate_md(md=800):
    global SURVEY
    node = SURVEY.interpolate_md(md=md)
    assert isinstance(node, we.node.Node)

def test_interpolate_tvd(tvd=800):
    global SURVEY
    node = SURVEY.interpolate_tvd(tvd=tvd)
    assert isinstance(node, we.node.Node)
