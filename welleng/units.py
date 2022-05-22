import pint

ureg = pint.UnitRegistry()

# TODO import custom units from file instead of defining here
ureg.define('ft_lbf = ft * lbf')
ureg.define('Nm = N * m')
