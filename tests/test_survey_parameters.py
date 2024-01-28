import welleng as we
import numpy as np

REFERENCE = {
    'x': 588319.02, 'y': 5770571.03, 'northing': 5770571.03,
    'easting': 588319.02, 'latitude': 52.077583926214494,
    'longitude': 4.288694821453205, 'convergence': 1.0166440347220762,
    'scale_factor': 0.9996957469340422, 'magnetic_field_intensity': 49381,
    'declination': 2.213, 'dip': -67.199, 'date': '2023-12-16',
    'srs': 'EPSG:23031',
    'wgs84-utm31': [588225.162, 5770360.512]
}

CALCULATOR = we.survey.SurveyParameters(REFERENCE.get('srs'))


def test_known_location():
    survey_parameters = CALCULATOR.get_factors_from_x_y(
        x=REFERENCE.get('x'), y=REFERENCE.get('y'),
        date=REFERENCE.get('date')
    )
    for k, v in survey_parameters.items():
        try:
            assert round(v, 3) == round(REFERENCE.get(k), 3)
        except TypeError:
            assert v == REFERENCE.get(k)

    pass

def test_transform_projection_coordinates():
    # Convert survey coordinates from UTM31_ED50 to UTM31_WGS84
    coords = np.array((REFERENCE.get('easting'), REFERENCE.get('northing')))
    result = CALCULATOR.transform_coordinates(coords, 'EPSG:32631')
    assert np.allclose(
        result,
        np.array(REFERENCE.get('wgs84-utm31'))
    )

    # Try as a list
    result = CALCULATOR.transform_coordinates(
        coords.tolist(), 'EPSG:32631'
    )
    assert np.allclose(
        result,
        np.array(REFERENCE.get('wgs84-utm31'))
    )

    # Try as a tuple
    result = CALCULATOR.transform_coordinates(
        tuple(coords.tolist()), 'EPSG:32631'
    )
    assert np.allclose(
        result,
        np.array(REFERENCE.get('wgs84-utm31'))
    )

    result = CALCULATOR.transform_coordinates(
        np.array([coords, coords]),
        'EPSG:32631'
    )
    assert np.allclose(
        result,
        np.full_like(result, REFERENCE.get('wgs84-utm31'))
    )

    # Try as a list
    result = CALCULATOR.transform_coordinates(
        [coords.tolist(), coords.tolist()],
        'EPSG:32631'
    )
    assert np.allclose(
        result,
        np.full_like(result, REFERENCE.get('wgs84-utm31'))
    )

    # Try as a tuple
    result = CALCULATOR.transform_coordinates(
        (tuple(coords.tolist()), tuple(coords.tolist())),
        'EPSG:32631'
    )
    assert np.allclose(
        result,
        np.full_like(result, REFERENCE.get('wgs84-utm31'))
    )
