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
    'turn_rates': 'deg / _30m',
    'magnetic_field': 'nT',
    'angular_velocity': 'rad / hr',
    'acceleration': 'meters / sec ** 2'
}

# map the parameters to the unit groups
parameters = {
    'angles': [
        'dogleg', 'toolface', 'dip', 'declination', 'convergence',
        'vertical_inc_limit', 'vertical_section_azimuth'
    ],
    'depths': ['md', 'tvd', 'z', 'curve_radius', 'altitude'],
    'diameters': ['radius'],
    'lateral_distances': ['n', 'e', 'x', 'y', 'vertical_section'],
    'turn_rates': ['build', 'turn'],
    'magnetic_field': ['b_total'],
    'angular_velocity': ['earth_rate'],
    'acceleration': ['G']
}
