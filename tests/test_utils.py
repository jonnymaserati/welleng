import inspect
import sys

from welleng.units import ureg
from welleng.utils import annular_volume


def test_annular_volume():
    av = annular_volume(
        od=ureg('12.25 inch').to('meter'),
        id=ureg(f'{9+5/8} inch').to('meter'),
        length=ureg('1000 meter')
    )

    assert av.m == 3.491531223156194
    assert str(av.u) == 'meter ** 3'

    pass


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
