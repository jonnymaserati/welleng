
"""
test_minimal.py
--------------
Test things that should work with a *minimal* welleng install.
"""
import unittest
import welleng as we


class MinimalTest(unittest.TestCase):

    def test_survey(self):
        survey = we.survey.interpolate_survey(
            survey=we.survey.Survey(
                md=[0, 500, 1000, 2000, 3000],
                inc=[0, 0, 30, 90, 90],
                azi=[90, 90, 90, 135, 180],
                error_model='iscwsa_mwd_rev5'
            ),
            step=30.
        )
        return survey


if __name__ == '__main__':
    unittest.main()
