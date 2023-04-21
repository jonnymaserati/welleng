from re import search

from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

# TODO import custom units from file instead of defining here
ureg.define('ft_lbf = ft * lbf')


def manage_unitized_unit(string):
    """
    Due to equal precedence of multiplication and division in Python, ``pint``
    won't always handle a unitized unit as expected (e.g. dls in deg/30m).
    This function processes units and ensures that these unitized units are
    handled as expected.

    Parameters
    ----------
    string : str
        A string consisting of a scalar and a unit, e.g. ``"3.0 deg/30m"``.

    Returns
    -------
    new_string : str
        A modified string that is more explicit in precedence of unitized
        units.

    Example
    -------
    >>> from pint import Quantity as Q_
    >>> string = "3.0 deg/30m"
    >>> print(f"Unprocessed result {Q_(string) = }")
    >>> print(f"Processed result {Q_(manage_unitized_unit(string)) = }")
    Unprocessed result Q_(string) = <Quantity(0.1, 'degree * meter')>
    Processed result Q_(manage_unitized_unit(string)) = <Quantity(0.1, 'degree / meter')>
    """
    scalar = search(r'[-+]?(?:\d*\.*\d+)', string)
    unit = string[scalar.span()[-1]:].replace(' ', '')

    try:
        numerator, denominator = unit.split('/')
    except ValueError:
        numerator = unit
        denominator = 1

    new_string = f"({scalar.group()} {numerator}) / ({denominator})"

    return new_string


if __name__ == "__main__":
    from pint import Quantity as Q_
    string = "3.0 deg/30m"
    print(f"Unprocessed result {Q_(string) = }")
    print(f"Processed result {Q_(manage_unitized_unit(string)) = }")
