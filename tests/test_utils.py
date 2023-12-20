import inspect
import sys

import numpy as np
from numpy.typing import NDArray

from welleng.units import ureg
from welleng.utils import annular_volume, decimal2dms, dms2decimal

LAT, LON = (52, 4, 43.1868), (4, 17, 19.6368)


def _generate_random_dms(n: int) -> NDArray:
    """Generates a bunch of lat, lon coordinates.
    """
    deg = np.random.randint(-180, 180, n)
    min = np.random.randint(0, 60, n)
    sec = np.random.uniform(0, 60, n)

    return np.stack((deg, min, sec), axis=-1).reshape((-1, 2, 3))


def test_annular_volume():
    av = annular_volume(
        od=ureg('12.25 inch').to('meter'),
        id=ureg(f'{9+5/8} inch').to('meter'),
        length=ureg('1000 meter')
    )

    assert av.m == 3.491531223156194
    assert str(av.u) == 'meter ** 3'

    pass


def test_decimal2dms():
    degrees, minutes, seconds = decimal2dms(LAT[0] + LAT[1] / 60 + LAT[2] / 3600)
    assert (degrees, minutes, round(seconds, 4)) == LAT

    dms = decimal2dms(np.array([
        -(LAT[0] + LAT[1] / 60 + LAT[2] / 3600),
        LON[0] + LON[1] / 60 + LON[2] / 3600
    ]))
    assert np.allclose(
        dms,
        np.array((np.array(LAT) * np.array((-1, 1, 1)), np.array(LON)))
    )


def test_deg2decimal():
    decimal = dms2decimal((-LAT[0], LAT[1], LAT[2]))  # check it handles westerly
    assert decimal == -(LAT[0] + LAT[1] / 60 + LAT[2] / 3600)

    decimals = dms2decimal((LAT, LON))
    assert np.allclose(decimals, np.array((dms2decimal(LAT), dms2decimal(LON))))

    decimals = dms2decimal(((LAT, LON), (LON, LAT)))
    assert np.allclose(
        decimals,
        np.array((
            (dms2decimal(LAT), dms2decimal(LON)),
            (dms2decimal(LON), dms2decimal(LAT))
        ))
    )


def test_dms2decimal2dms():
    _dms = _generate_random_dms(int(1e3))
    decimal = dms2decimal(_dms)
    dms = decimal2dms(decimal)

    assert np.allclose(_dms, dms)


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

    for f in test_functions:
        f()

        pass


if __name__ == '__main__':
    one_function_to_run_them_all()
