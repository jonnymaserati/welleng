import numpy as np
import pytest
from numpy.typing import NDArray
from welleng.units import ureg
from welleng.utils import (
    annular_volume,
    cov_from_vec,
    decimal2dms,
    dms2decimal,
    pprint_dms,
    dms_from_string,
    radius_from_dls,
    dls_from_radius,
    get_toolface,
    get_arc,
    get_angles,
    get_vec,
    get_transform,
    get_sigmas,
    make_cov,
    errors_from_cov,
    NEV_to_HLA,
    HLA_to_NEV,
    MinCurve,
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
            29.096093526301622,
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


def test_dls_radius_inverse():
    """dls_from_radius and radius_from_dls should be exact inverses."""
    for dls in [1.0, 2.5, 5.0, 10.0]:
        assert np.isclose(dls_from_radius(radius_from_dls(dls)), dls)
    for radius in [100.0, 500.0, 1718.87]:
        assert np.isclose(radius_from_dls(dls_from_radius(radius)), radius)
    # zero radius → infinite dls and vice versa
    assert dls_from_radius(0) == np.inf
    assert radius_from_dls(0) == np.inf


def test_get_vec_get_angles_roundtrip():
    """get_vec and get_angles should be exact inverses."""
    incs = np.array([0., 30., 60., 90., 45.])
    azis = np.array([0., 45., 90., 180., 270.])

    vecs = get_vec(incs, azis, nev=True, deg=True)
    result = get_angles(vecs, nev=True)

    assert np.allclose(result[:, 0], np.radians(incs), atol=1e-10)
    assert np.allclose(result[:, 1], np.radians(azis), atol=1e-10)


def test_get_vec_unit_length():
    """get_vec should always return unit vectors."""
    incs = np.random.uniform(0, 180, 20)
    azis = np.random.uniform(0, 360, 20)
    vecs = get_vec(incs, azis)
    norms = np.linalg.norm(vecs, axis=-1)
    assert np.allclose(norms, 1.0)


def test_get_transform_orthogonal():
    """get_transform should return orthogonal rotation matrices (R @ R.T = I)."""
    survey = np.column_stack([
        np.zeros(5),
        np.radians([0., 30., 60., 90., 45.]),
        np.radians([0., 45., 90., 180., 270.]),
    ])
    trans = get_transform(survey)  # (n,3,3)
    for i in range(len(survey)):
        product = trans[i] @ trans[i].T
        assert np.allclose(product, np.eye(3), atol=1e-10)


def test_NEV_to_HLA_roundtrip():
    """HLA_to_NEV(NEV_to_HLA(cov)) should recover the original cov."""
    survey = np.column_stack([
        np.zeros(10),
        np.radians(np.linspace(0, 90, 10)),
        np.radians(np.linspace(0, 180, 10)),
    ])
    # build a simple diagonal (n,3,3) cov
    cov_nev = np.zeros((10, 3, 3))
    cov_nev[:, 0, 0] = 1.0
    cov_nev[:, 1, 1] = 0.5
    cov_nev[:, 2, 2] = 0.25

    cov_hla = NEV_to_HLA(survey, cov_nev, cov=True)
    cov_recovered = HLA_to_NEV(survey, cov_hla, cov=True)

    assert np.allclose(cov_recovered, cov_nev, atol=1e-10)


def test_NEV_to_HLA_preserves_trace():
    """Rotation preserves the trace (sum of eigenvalues) of the cov matrix."""
    survey = np.column_stack([
        np.zeros(5),
        np.radians([10., 30., 60., 80., 45.]),
        np.radians([20., 60., 120., 200., 300.]),
    ])
    cov_nev = np.random.rand(5, 3, 3)
    # make symmetric positive semi-definite
    cov_nev = cov_nev @ cov_nev.swapaxes(-1, -2)

    cov_hla = NEV_to_HLA(survey, cov_nev, cov=True)

    traces_nev = np.trace(cov_nev, axis1=-2, axis2=-1)
    traces_hla = np.trace(cov_hla, axis1=-2, axis2=-1)
    assert np.allclose(traces_nev, traces_hla, atol=1e-10)


def test_NEV_to_HLA_cov_false():
    """NEV_to_HLA with cov=False should transform coordinate vectors."""
    survey = np.column_stack([
        np.zeros(3),
        np.zeros(3),      # vertical well: inc=0
        np.zeros(3),
    ])
    # For a vertical well, NEV → HLA should be identity-like
    nev = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    hla = NEV_to_HLA(survey, nev, cov=False)
    assert hla.shape == nev.shape


def test_make_cov_get_sigmas_roundtrip():
    """make_cov then get_sigmas should recover the diagonal values."""
    a = np.array([2.0, 3.0, 1.5])
    b = np.array([1.0, 0.5, 2.0])
    c = np.array([0.5, 1.5, 1.0])

    cov = make_cov(a, b, c, long=False)  # diagonal only, shape (n,3,3)
    sigmas = get_sigmas(cov)

    assert np.allclose(sigmas[0], np.abs(a))
    assert np.allclose(sigmas[1], np.abs(b))
    assert np.allclose(sigmas[2], np.abs(c))


def test_make_cov_long_symmetric():
    """make_cov with long=True should produce symmetric matrices."""
    a, b, c = 2.0, 1.0, 0.5
    cov = make_cov(a, b, c, long=True)  # (1,3,3) or (3,3)
    cov = cov.reshape(-1, 3, 3)
    for m in cov:
        assert np.allclose(m, m.T)


def test_cov_from_vec():
    """cov_from_vec should return the outer product of each row with itself."""
    arr = np.array([[1., 2., 3.], [4., 0., 1.]])
    cov = cov_from_vec(arr)
    assert cov.shape == (2, 3, 3)
    # each (3,3) slice should equal the outer product v @ v.T
    for i, v in enumerate(arr):
        assert np.allclose(cov[i], np.outer(v, v))
    # result must be symmetric
    assert np.allclose(cov, cov.swapaxes(-1, -2))


def test_errors_from_cov():
    """errors_from_cov should return 6 unique cov elements per station."""
    n = 5
    cov = np.zeros((n, 3, 3))
    cov[:, 0, 0] = 4.   # nn
    cov[:, 1, 1] = 1.   # ee
    cov[:, 2, 2] = 0.25  # vv
    cov[:, 0, 1] = cov[:, 1, 0] = 0.5   # ne
    cov[:, 0, 2] = cov[:, 2, 0] = 0.25  # nv
    cov[:, 1, 2] = cov[:, 2, 1] = 0.1   # ev

    result = errors_from_cov(cov)
    assert result.shape == (n, 6)
    # diagonal elements should match squared values: [nn, ne, nv, ee, ev, vv]
    assert np.allclose(result[:, 0], 4.)    # nn
    assert np.allclose(result[:, 3], 1.)    # ee
    assert np.allclose(result[:, 5], 0.25)  # vv


def main():
    test_mincurve()
    test_dms2decimal2dms()
    pass


if __name__ == "__main__":
    main()
