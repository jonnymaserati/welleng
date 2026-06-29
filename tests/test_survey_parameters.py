import welleng as we
import numpy as np
import pytest

REFERENCE = {
    'x': 588319.02, 'y': 5770571.03, 'northing': 5770571.03,
    'easting': 588319.02, 'latitude': 52.077583926214494,
    'longitude': 4.288694821453205, 'convergence': 1.0166440347220762,
    'scale_factor': 0.9996957469340422, 'magnetic_field_intensity': 49421,
    'declination': 2.356, 'dip': -67.224, 'date': '2025-06-01',
    'srs': 'EPSG:23031',
    'wgs84-utm31': [588225.162, 5770360.512]
}

CALCULATOR = we.survey.SurveyParameters(REFERENCE.get('srs'))


def test_known_location(monkeypatch):
    # Always runs -- no optional 'magnetic_field_calculator' install and no live BGS
    # network call needed. Stub the API client to return the known BGS response for
    # this location/date, so we deterministically validate welleng's own code: the
    # projection factors (real pyproj) and the magnetic-field processing (dip sign
    # from the "down" units, nested-field extraction). The external service's values
    # are not welleng's to test.
    class _StubMagCalc:
        def calculate(self, **kwargs):
            return {"field-value": {
                "total-intensity": {"value": REFERENCE["magnetic_field_intensity"]},
                "declination": {"value": REFERENCE["declination"]},
                # welleng negates a "down" inclination -> dip; feed +intensity so the
                # sign handling is what's under test.
                "inclination": {"value": -REFERENCE["dip"], "units": "deg (down)"},
            }}
    monkeypatch.setattr(we.survey, "MAG_CALC", True)
    monkeypatch.setattr(we.survey, "MagneticFieldCalculator", _StubMagCalc, raising=False)

    survey_parameters = CALCULATOR.get_factors_from_x_y(
        x=REFERENCE.get('x'), y=REFERENCE.get('y'),
        date=REFERENCE.get('date')
    )
    for k, v in survey_parameters.items():
        if REFERENCE.get(k) is None:
            continue
        try:
            assert round(v, 3) == round(REFERENCE.get(k), 3)
        except TypeError:
            assert v == REFERENCE.get(k)

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
