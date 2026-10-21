import welleng as we
import numpy as np

REFERENCE = {
    'x': 588319.02, 'y': 5770571.03, 'northing': 5770571.03,
    'easting': 588319.02, 'latitude': 52.077583926214494,
    'longitude': 4.288694821453205, 'convergence': 1.0166440347220762,
    'scale_factor': 0.9996957469340422, 'magnetic_field_intensity': 49421,
    'declination': 2.356, 'dip': 67.224, 'date': '2025-06-01',
    'srs': 'EPSG:23031',
    'wgs84-utm31': [588225.162, 5770360.512]
}

CALCULATOR = we.survey.SurveyParameters(REFERENCE.get('srs'))


def test_known_location(monkeypatch):
    # Always runs -- no live BGS network call needed. Stub the BGS client to
    # return the known response for this location/date, so we deterministically
    # validate welleng's own code: the projection factors (real pyproj) and the
    # magnetic-field processing (dip positive-DOWN, the ISCWSA/BGS
    # convention; nested-field extraction). The external service's values
    # are not welleng's to test.
    def _stub_lookup(**kwargs):
        return {"field-value": {
            "total-intensity": {"value": REFERENCE["magnetic_field_intensity"]},
            "declination": {"value": REFERENCE["declination"]},
            # dip stored positive-down, straight from the service
            "inclination": {"value": REFERENCE["dip"], "units": "deg (down)"},
        }}
    monkeypatch.setattr(we.survey, "lookup_field", _stub_lookup)

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


def test_transform_refuses_when_no_operation_is_valid_there():
    """A point outside ED50's area of use (mid-Atlantic) has no operation of
    stated accuracy to ETRS89, directly or through WGS 84; PROJ's fallback
    ("ballpark") would return it unchanged, so the transform refuses."""
    import pytest
    geo = we.survey.SurveyParameters('EPSG:4230')             # ED50 lat/lon
    with pytest.raises(ValueError, match="unchanged"):
        geo.transform_coordinates([(40.0, -40.0)], 'EPSG:4258')


def test_transform_routes_through_wgs84_where_no_direct_operation_is_valid():
    """Onshore Netherlands, ED50 -> ETRS89: no direct operation of stated
    accuracy is valid there without grids, so it goes through WGS 84 -- not
    PROJ's fallback, which returned the input unchanged (~130 m)."""
    geo = we.survey.SurveyParameters('EPSG:4230')
    out = geo.transform_coordinates([(52.0, 4.5)], 'EPSG:4258')
    op = geo.last_operation
    assert "ED50 to WGS 84 (18)" in op["name"] and "ETRS89 to WGS 84" in op["name"]
    assert op["accuracy_m"] == 2.0
    moved = np.hypot((out[0][0] - 52.0) * 111_000,
                     (out[0][1] - 4.5) * 111_000 * np.cos(np.radians(52.0)))
    assert 50.0 < moved < 200.0                       # a real datum shift


def test_transform_uses_an_operation_valid_where_the_points_are():
    geo = we.survey.SurveyParameters('EPSG:4230')
    # Dutch North Sea: covered by ED50 to ETRS89 (15), stated 1 m
    out = geo.transform_coordinates([(54.0, 4.5)], 'EPSG:4258')
    assert "ED50 to ETRS89 (15)" in geo.last_operation["name"]
    assert geo.last_operation["accuracy_m"] == 1.0
    moved = np.hypot((out[0][0] - 54.0) * 111_000,
                     (out[0][1] - 4.5) * 111_000 * np.cos(np.radians(54.0)))
    assert 50.0 < moved < 200.0                       # a real datum shift
    # onshore Netherlands to WGS 84: ED50 to WGS 84 (18), not an operation
    # whose area of use ends south of the point
    CALCULATOR.transform_coordinates(
        (REFERENCE['easting'], REFERENCE['northing']), 'EPSG:32631')
    assert "ED50 to WGS 84 (18)" in CALCULATOR.last_operation["name"]


def test_transform_same_datum_is_a_conversion():
    CALCULATOR.transform_coordinates(
        (REFERENCE['easting'], REFERENCE['northing']), 'EPSG:4230')
    assert CALCULATOR.last_operation["accuracy_m"] == 0.0
