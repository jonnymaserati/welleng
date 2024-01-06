import inspect
import sys
import unittest

import numpy as np
from numpy.typing import NDArray

from welleng.units import ureg
from welleng.utils import annular_volume, decimal2dms, dms2decimal

LAT, LON = (52, 4, 43.1868, 'N'), (4, 17, 19.6368, 'E')


def _generate_random_dms(n: int, ndigits: int = None) -> NDArray:
    """Generates a bunch of lat, lon coordinates.
    """
    assert n % 2 == 0, "n must be an even int"
    deg = np.random.randint(0, 180, n)
    min = np.random.randint(0, 60, n)
    sec = np.random.uniform(0, 60, n)

    if ndigits is not None:
        sec = np.around(sec, ndigits)

    data = {
        0: 'N', 1: 'S', 2: 'E', 3: 'W'
    }

    direction = np.array([
        data.get(d)
        for d in np.ravel(np.stack((
            np.random.randint(0, 2, int(n/2)),
            np.random.randint(2, 4, int(n/2))
        ), axis=1
        ))
    ])

    return np.stack(
        (deg, min, sec, direction),
        axis=-1,
        dtype=object
    ).reshape((-1, 2, 4))


class UtilsTest(unittest.TestCase):
    def test_annular_volume(self):
        av = annular_volume(
            od=ureg('12.25 inch').to('meter'),
            id=ureg(f'{9+5/8} inch').to('meter'),
            length=ureg('1000 meter')
        )

        assert av.m == 3.491531223156194
        assert str(av.u) == 'meter ** 3'

        pass

    def test_decimal2dms(self):
        degrees, minutes, seconds, direction = decimal2dms(
            (LAT[0] + LAT[1] / 60 + LAT[2] / 3600, LAT[3])
        )
        assert (degrees, minutes, round(seconds, 4), direction) == LAT

        dms = decimal2dms(np.array([
            (LAT[0] + LAT[1] / 60 + LAT[2] / 3600, LAT[3]),
            (LON[0] + LON[1] / 60 + LON[2] / 3600, LON[3])
        ]), ndigits=4)
        assert np.all(np.equal(
            dms,
            np.array((np.array(LAT, dtype=object), np.array(LON, dtype=object)))
        ))

    def test_dms2decimal(self):
        decimal = dms2decimal((-LAT[0], LAT[1], LAT[2], LAT[3]))  # check it handles westerly
        assert np.all(np.equal(
            decimal,
            np.array([
                -(LAT[0] + LAT[1] / 60 + LAT[2] / 3600),
                LAT[3]
            ], dtype=object)
        ))

        decimals = dms2decimal((LAT, LON))
        assert np.all(np.equal(
            decimals,
            np.array((dms2decimal(LAT), dms2decimal(LON)))
        ))

        decimals = dms2decimal(((LAT, LON), (LON, LAT)))
        assert np.all(np.equal(
            decimals,
            np.array((
                (dms2decimal(LAT), dms2decimal(LON)),
                (dms2decimal(LON), dms2decimal(LAT))
            ))
        ))

    def test_dms2decimal2dms(self):
        _dms = _generate_random_dms(int(1e3), 8)
        decimal = dms2decimal(_dms)
        dms = decimal2dms(decimal, 8)

        assert np.all(np.equal(_dms, dms))


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
