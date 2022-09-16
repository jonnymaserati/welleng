from welleng.error_formula_extractor.formula_utils import function_builder

from pathlib import Path
import unittest
import math

filename = "Volve F.edm.xml"
file_path = (Path(__file__) / ".." / ".." / "resources"/filename).resolve()


class TestErrorFormulaExtractor(unittest.TestCase):

    def test_formula_builder(self):
        """
        Test the functionality to convert text functions to mathematical functions.
        """
        inc = math.radians(45)
        azi = math.radians(60)
        test_strings = [
            "sin(inc)",
            "sin(inc) * cos(azi)",
            "(sin(inc))^2 + (cos(inc))^2",
            "5 * (tan(inc))^2"
        ]

        input_map = {
            "inc": inc,
            "azi": azi
        }
        expected_outputs = [
            math.sin(inc),
            math.sin(inc) * math.cos(azi),
            (math.sin(inc)) ** 2 + (math.cos(inc)) ** 2,
            5 * (math.tan(inc)) ** 2
        ]

        for idx, string in enumerate(test_strings):
            function, args, _ = function_builder(string, "func1")

            calculated = function(*[input_map[arg] for arg in args])

            assert calculated == expected_outputs[idx]
