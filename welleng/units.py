import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

# TODO import custom units from file instead of defining here
ureg.define('ft_lbf = ft * lbf')
