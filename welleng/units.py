import pint

ureg = pint.UnitRegistry()

# add some oilfiled units
ureg.define('_30m = 30 * meters')
ureg.define('_30ft = 30 * ft')
ureg.define('_100ft = 100 * ft')
ureg.define('_100m = 100 * meters')
ureg.define('_10m = 10 * meters')

# define the standard welleng units used by the engine
units_default = {
    'angles': 'deg',
    'depths': 'meters',
    'diameters': 'meters',
    'lateral_distances': 'meters',
    'dls': "deg / _30m",
    'turn_rates': 'deg / _30m'
}

# map the parameters to the unit groups
parameters = {
    'angles': ['dogleg', 'toolface'],
    'depths': ['md', 'tvd', 'z', 'curve_radius'],
    'diameters': ['radius'],
    'lateral_distances': ['n', 'e', 'x', 'y'],
    'turn_rates': ['build', 'turn']
}
