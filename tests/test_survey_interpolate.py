import welleng as we
import inspect
import sys

survey = we.survey.Survey(
    md=[0, 500, 1000, 2000, 2500, 3500],
    inc=[0, 0, 30, 90, 100, 80],
    azi=[45, 45, 45, 90, 90, 180],
    radius=10,
)


def test_survey_interpolate_survey(step=30):
    global survey
    survey_interp = we.survey.interpolate_survey(survey, step=step)
    assert isinstance(survey_interp, we.survey.Survey)

    survey_interp = survey.interpolate_survey(step=step)
    assert isinstance(survey_interp, we.survey.Survey)


def test_survey_interpolate_survey_tvd(step=10):
    global survey
    survey_interp = survey.interpolate_survey(step=30)
    survey_interp_tvd = we.survey.interpolate_survey_tvd(
        survey_interp, step=step
    )
    assert isinstance(survey_interp_tvd, we.survey.Survey)

    survey_interp_tvd = survey_interp.interpolate_survey_tvd(step=step)
    assert isinstance(survey_interp_tvd, we.survey.Survey)


def test_interpolate_md(md=800):
    global survey
    node = survey.interpolate_md(md=md)
    assert isinstance(node, we.node.Node)


def test_interpolate_tvd(tvd=800):
    global survey
    node = survey.interpolate_tvd(tvd=tvd)
    assert isinstance(node, we.node.Node)


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
