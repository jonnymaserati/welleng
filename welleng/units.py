import pint

ureg = pint.UnitRegistry()

# TODO import custom units from file instead of defining here
ureg.define('ft_lbf = ft * lbf')
ureg.define('Nm = N * m')


METER_TO_FOOT = 3.28084
DEGREE_TO_RAD = 0.0174533

TORTUOSITY_DEG_PER_100_FT = 1

TORTUOSITY_RAD_PER_M = TORTUOSITY_DEG_PER_100_FT * DEGREE_TO_RAD * METER_TO_FOOT / 100
