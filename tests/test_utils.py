import numpy as np
import pytest
from numpy.typing import NDArray
from welleng.units import ureg
from welleng.utils import (
    annular_volume,
    decimal2dms,
    dms2decimal,
    pprint_dms,
    dms_from_string,
    radius_from_dls,
    get_toolface,
    get_arc,
    MinCurve
)
import json

LAT, LON = (52, 4, 43.1868, "N"), (4, 17, 19.6368, "E")


def test_mincurve(decimals=2):
    with open("tests/test_data/clearance_iscwsa_well_data.json") as f:
        data = json.load(f)

    for well, survey in data.get('wells').items():
        def _get_start_xyz(survey):
            return np.array([
                survey.get('E')[0],
                survey.get('N')[0],
                survey.get('TVD')[0]
            ])

        mc = MinCurve(
            md=survey.get('MD'),
            inc=np.radians(survey.get('IncDeg')),
            azi=np.radians(survey.get('AziDeg')),
            start_xyz=_get_start_xyz(survey)
        )

        assert np.allclose(
            np.round(mc.poss, decimals), np.round(np.array([
                survey.get('E'), survey.get('N'), survey.get('TVD')
            ]).T, decimals)
        ), "Unexpected position."
    
    pass


def _generate_random_dms(n: int, ndigits: int = None) -> NDArray:
    assert n % 2 == 0, "n must be an even int"
    deg = np.random.randint(0, 180, n)
    min = np.random.randint(0, 60, n)
    sec = np.random.uniform(0, 60, n)

    if ndigits is not None:
        sec = np.around(sec, ndigits)

    data = {0: "N", 1: "S", 2: "E", 3: "W"}

    direction = np.array(
        [
            data.get(d)
            for d in np.ravel(
                np.stack(
                    (
                        np.random.randint(0, 2, int(n / 2)),
                        np.random.randint(2, 4, int(n / 2)),
                    ),
                    axis=1,
                )
            )
        ]
    )

    return np.stack((deg, min, sec, direction), axis=-1, dtype='object').reshape(
        (-1, 2, 4)
    )


def are_tuples_identical(tuple1, tuple2):
    return all(x == y for x, y in zip(tuple1, tuple2))


@pytest.mark.parametrize(
    "od, id, length, expected",
    [
        (
            ureg("12.25 inch").to("meter"),
            ureg(f"{9+5/8} inch").to("meter"),
            ureg("1000 meter"),
            3.491531223156194,
        ),
        # Add more test cases as needed
    ],
)
def test_annular_volume(od, id, length, expected):
    av = annular_volume(od=od, id=id, length=length)
    assert av.m == expected
    assert str(av.u) == "meter ** 3"


def test_decimal2dms():
    degrees, minutes, seconds, direction = decimal2dms(
        (LAT[0] + LAT[1] / 60 + LAT[2] / 3600, LAT[3])
    )
    assert (degrees, minutes, round(seconds, 4), direction) == LAT

    dms = decimal2dms((LAT[0] + LAT[1] / 60 + LAT[2] / 3600), ndigits=4)
    assert np.all(np.equal(dms, np.array((np.array(LAT[:3])))))

    dms = decimal2dms(
        np.array(
            [
                (LAT[0] + LAT[1] / 60 + LAT[2] / 3600),
                (LON[0] + LON[1] / 60 + LON[2] / 3600),
            ]
        ).reshape((-1, 1)),
        ndigits=4,
    )
    assert np.all(np.equal(dms, np.array((np.array(LAT[:3]), np.array(LON[:3])))))

    dms = decimal2dms(
        np.array(
            [
                (LAT[0] + LAT[1] / 60 + LAT[2] / 3600),
                (LON[0] + LON[1] / 60 + LON[2] / 3600),
            ]
        ),
        ndigits=4,
    )
    assert np.all(np.equal(dms, np.array((np.array(LAT[:3]), np.array(LON[:3])))))

    dms = decimal2dms(
        np.array(
            [
                (LAT[0] + LAT[1] / 60 + LAT[2] / 3600, LAT[3]),
                (LON[0] + LON[1] / 60 + LON[2] / 3600, LON[3]),
            ]
        ),
        ndigits=4,
    )
    assert np.all(
        np.equal(
            dms, np.array((np.array(LAT, dtype=object), np.array(LON, dtype=object)))
        )
    )


def test_dms2decimal():
    decimal = dms2decimal(
        (-LAT[0], LAT[1], LAT[2], LAT[3])
    )  # check it handles westerly
    assert np.all(
        np.equal(
            decimal,
            np.array([-(LAT[0] + LAT[1] / 60 + LAT[2] / 3600), LAT[3]], dtype=object),
        )
    )

    decimal = dms2decimal((LAT[:3]))
    assert decimal == LAT[0] + LAT[1] / 60 + LAT[2] / 3600

    decimals = dms2decimal((LAT[:3], LON[:3]))
    assert np.all(
        np.equal(decimals, np.array((dms2decimal(LAT[:3]), dms2decimal(LON[:3]))))
    )

    decimals = dms2decimal((LAT, LON))
    assert np.all(
        np.equal(
            decimals,
            np.array((dms2decimal(LAT), dms2decimal(LON))).reshape(decimals.shape),
        )
    )

    decimals = dms2decimal(((LAT, LON), (LON, LAT)))
    assert np.all(
        np.equal(
            decimals,
            np.array(
                (
                    (dms2decimal(LAT), dms2decimal(LON)),
                    (dms2decimal(LON), dms2decimal(LAT)),
                )
            ),
        )
    )


def test_dms2decimal2dms():
    _dms = _generate_random_dms(int(1e3), 8)
    decimal = dms2decimal(_dms)
    dms = decimal2dms(decimal, 8)
    assert np.all(np.equal(_dms, dms))


def test_pprint_dms():
    result = pprint_dms(LAT, return_data=True)
    data = dms_from_string(result)

    assert are_tuples_identical(LAT, data)

    result = pprint_dms(LAT, return_data=True, symbols=False)
    data = dms_from_string(result)

    assert are_tuples_identical(LAT, data)

    result = pprint_dms(LAT[:3], return_data=True)
    data = dms_from_string(result)

    assert are_tuples_identical(LAT[:3], data)

    result = pprint_dms(LAT[:3], return_data=True, symbols=False)
    data = dms_from_string(result)

    assert are_tuples_identical(LAT[:3], data)


def test_get_arc():
    pos1 = np.array([0, 0, 0])
    vec1 = np.array([1, 0, 0])
    dls_design = 2.5
    radius = radius_from_dls(dls_design)
    toolface = np.pi / 2
    dogleg = np.pi / 2

    pos2, vec2, arc_length = get_arc(dogleg, radius, toolface, pos1, vec1)

    assert all(
        (
            np.allclose(pos2, pos1 + np.array([radius, radius, 0])),
            np.allclose(vec2, np.array([0, 1, 0])),
        )
    )


def test_get_toolface():
    pos1 = np.array([0, 0, 0])
    vec1 = np.array([1, 0, 0])
    dls_design = 2.5
    radius = radius_from_dls(dls_design)

    pos2 = pos1 + np.array([radius, radius, 0])
    toolface = get_toolface(pos1, vec1, pos2)
    assert np.isclose(toolface, np.pi / 2)

    pos2 = pos1 + np.array([radius, -radius, 0])
    toolface = get_toolface(pos1, vec1, pos2)
    assert np.isclose(toolface, -np.pi / 2)

    pos2 = pos1 + np.array([radius, 0, radius])
    toolface = get_toolface(pos1, vec1, pos2)
    assert np.isclose(toolface, np.pi)

    pos2 = pos1 + np.array([radius, 0, -radius])
    toolface = get_toolface(pos1, vec1, pos2)
    assert np.isclose(toolface, 0)

    pos1 = np.array([0, 0, 0])
    vec1 = np.array([0, 0, 1])

    pos2 = pos1 + np.array([radius, 0, radius])
    toolface = get_toolface(pos1, vec1, pos2)
    assert np.isclose(toolface, 0)


def main():
    test_mincurve()
    test_dms2decimal2dms()
    pass


if __name__ == "__main__":
    main()
