import unittest

import welleng as we

reference = {
    'x': 588319.02, 'y': 5770571.03, 'northing': 5770571.03,
    'easting': 588319.02, 'latitude': 52.077583926214494,
    'longitude': 4.288694821453205, 'convergence': 1.0166440347220762,
    'scale_factor': 0.9996957469340422, 'magnetic_field_intensity': 49381,
    'declination': 2.213, 'dip': -67.199, 'date': '2023-12-16',
    'srs': 'EPSG:23031'
}


class SurveyParamsTest(unittest.TestCase):
    def test_known_location(self):
        calculator = we.survey.SurveyParameters(reference.get('srs'))
        survey_parameters = calculator.get_factors_from_x_y(
            x=reference.get('x'), y=reference.get('y'), date=reference.get('date')
        )

        for k, v in survey_parameters.items():
            try:
                assert round(v, 3) == round(reference.get(k), 3)
            except TypeError:
                assert v == reference.get(k)

        pass


# def one_function_to_run_them_all():
#     """
#     Function to gather the test functions so that they can be tested by
#     running this module.

#     https://stackoverflow.com/questions/18907712/python-get-list-of-all-
#     functions-in-current-module-inspecting-current-module
#     """
#     test_functions = [
#         obj for name, obj in inspect.getmembers(sys.modules[__name__])
#         if (inspect.isfunction(obj)
#             and name.startswith('test')
#             and name != 'all')
#     ]

#     for f in test_functions:
#         f()

#         pass


if __name__ == '__main__':
    unittest.main()
    # one_function_to_run_them_all()
