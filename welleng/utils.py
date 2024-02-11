import re
from typing import Annotated, Literal, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

try:
    from numba import njit
    NUMBA = True
except ImportError:
    NUMBA = False


def numbafy(func):
    func = njit(func)


class MinCurve:
    def __init__(
        self,
        md,
        inc,
        azi,
        start_xyz=[0., 0., 0.],
        unit="meters"
    ):
        """
        Generate geometric data from a well bore survey.

        Parameters
        ----------
        md: list or 1d array of floats
            Measured depth along well path from a datum.
        inc: list or 1d array of floats
            Well path inclination (relative to z/tvd axis where 0
            indicates down), in radians.
        azi: list or 1d array of floats
            Well path azimuth (relative to y/North axis),
            in radians.
        unit: str
            Either "meters" or "feet" to determine the unit of the dogleg
            severity.

        """

        assert unit == "meters" or unit == "feet", (
            'Unknown unit, please select "meters" of "feet"'
        )

        self.md = md
        survey_length = len(self.md)
        assert survey_length > 1, "Survey must have at least two rows"

        self.inc = inc
        self.azi = azi
        self.start_xyz = start_xyz
        self.unit = unit

        # make two slices with a difference or 1 index to enable array
        # calculations
        md_1, md_2 = self._split_arr(np.array(md))
        inc_1, inc_2 = self._split_arr(np.array(inc))
        azi_1, azi_2 = self._split_arr(np.array(azi))

        self.dogleg = self._get_dogleg(survey_length, inc_1, azi_1, inc_2, azi_2)

        # calculate rf and assume rf is 1 where dogleg is 0
        self.rf = self._get_rf(survey_length, self.dogleg)

        # calculate the change in md between survey stations
        self.delta_md = np.diff(self.md, prepend=0)

        args = (
            self.delta_md, inc_1, azi_1, inc_2, azi_2, self.rf, survey_length
        )

        # calculate change in y direction (north)
        self.delta_y = self._get_delta_y(*args)

        # calculate change in x direction (east)
        self.delta_x = self._get_delta_x(*args)

        # calculate change in z direction
        self.delta_z = self._get_delta_z(*args)

        # calculate the dog leg severity
        self.dls = self._get_dls(
            self.dogleg, self.delta_md, survey_length, unit
        )

        # cumulate the coordinates and add surface coordinates
        self.poss = np.vstack(
            np.cumsum(
                np.array([self.delta_x, self.delta_y, self.delta_z]).T, axis=0
            ) + self.start_xyz
        )

    @staticmethod
    def _get_dogleg(survey_length, inc1, azi1, inc2, azi2):
        dogleg = np.zeros(survey_length)
        dogleg[1:] = np.arccos(
            np.cos(inc2 - inc1)
            - (np.sin(inc1) * np.sin(inc2))
            * (1 - np.cos(azi2 - azi1))
        )
        return dogleg

    @staticmethod
    def _split_arr(arr):
        return (arr[:-1], arr[1:])

    @staticmethod
    def _get_rf(survey_length, dogleg):
        rf = np.ones(survey_length)
        idx = np.where(dogleg != 0)
        rf[idx] = 2 / dogleg[idx] * np.tan(dogleg[idx] / 2)
        return rf

    @staticmethod
    def _get_delta_y(delta_md, inc_1, azi_1, inc_2, azi_2, rf, survey_length):
        delta_y = np.zeros(survey_length)
        delta_y[1:] = (
            delta_md[1:]
            / 2
            * (
                np.sin(inc_1) * np.cos(azi_1)
                + np.sin(inc_2) * np.cos(azi_2)
            )
            * rf[1:]
        )
        return delta_y

    @staticmethod
    def _get_delta_x(delta_md, inc_1, azi_1, inc_2, azi_2, rf, survey_length):
        delta_x = np.zeros(survey_length)
        delta_x[1:] = (
            delta_md[1:]
            / 2
            * (
                np.sin(inc_1) * np.sin(azi_1)
                + np.sin(inc_2) * np.sin(azi_2)
            )
            * rf[1:]
        )
        return delta_x

    @staticmethod
    def _get_delta_z(delta_md, inc_1, azi_1, inc_2, azi_2, rf, survey_length):
        delta_z = np.zeros(survey_length)
        delta_z[1:] = (
            delta_md[1:]
            / 2
            * (np.cos(inc_1) + np.cos(inc_2))
            * rf[1:]
        )
        return delta_z
    
    @staticmethod
    def _get_dls(dogleg, delta_md, survey_length, unit):
        dls = np.zeros(survey_length)
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = np.degrees(dogleg[1:]) / delta_md[1:]
        mask = np.where(temp != np.nan)
        dls[1:][mask] = temp[mask]

        if unit == "meters":
            dls *= 30
        else:
            dls *= 100
        return dls


def get_vec(inc, azi, nev=False, r=1, deg=True):
    """
    Convert inc and azi into a vector.

    Parameters
    ----------
    inc: array of n floats
        Inclination relative to the z-axis (up)
    azi: array of n floats
        Azimuth relative to the y-axis
    r: float or array of n floats
        Scalar to return a scaled vector

    Returns
    -------
    vec: arraylike
        An (n,3) array of vectors
    """
    if deg:
        inc_rad, azi_rad = np.radians(np.array([inc, azi]))
    else:
        inc_rad = inc
        azi_rad = azi
    y = r * np.sin(inc_rad) * np.cos(azi_rad)
    x = r * np.sin(inc_rad) * np.sin(azi_rad)
    z = r * np.cos(inc_rad)

    if nev:
        vec = np.array([y, x, z]).T
    else:
        vec = np.array([x, y, z]).T

    return vec / np.linalg.norm(vec, axis=-1).reshape(-1, 1)


def get_nev(
    pos, start_xyz=np.array([0., 0., 0.]), start_nev=np.array([0., 0., 0.])
):
    """
    Convert [x, y, z] coordinates to [n, e, tvd] coordinates.

    Parameters
    ----------
    pos: (n,3) array of floats
        Array of [x, y, z] coordinates
    start_xyz: (,3) array of floats
        The datum of the [x, y, z] cooardinates
    start_nev: (,3) array of floats
        The datum of the [n, e, tvd] coordinates

    Returns
    -------
        An (n,3) array of [n, e, tvd] coordinates.
    """
    # e, n, v = (
    #     np.array([pos]).reshape(-1,3) - np.array([start_xyz])
    # ).T
    e, n, v = (
        np.array([pos]).reshape(-1, 3) - np.array([start_xyz])
    ).T

    return (np.array([n, e, v]).T + np.array([start_nev]))


def get_xyz(pos, start_xyz=[0., 0., 0.], start_nev=[0., 0., 0.]):
    y, x, z = (
        np.array([pos]).reshape(-1, 3) - np.array([start_nev])
    ).T

    return (np.array([x, y, z]).T + np.array([start_xyz]))


def _get_angles(vec):
    xy = vec[:, 0] ** 2 + vec[:, 1] ** 2
    inc = np.arctan2(np.sqrt(xy), vec[:, 2])  # for elevation angle defined from Z-axis down
    azi = (np.arctan2(vec[:, 0], vec[:, 1]) + (2 * np.pi)) % (2 * np.pi)

    return np.stack((inc, azi), axis=1)


def get_angles(
    vec: Annotated[NDArray, Literal["N", 3]], nev: bool = False
):
    '''
    Determines the inclination and azimuth from a vector.

    Parameters
    ----------
    vec: (n,3) array of floats
    nev: boolean (default: False)
        Indicates if the vector is in (x,y,z) or (n,e,v) coordinates.

    Returns
    -------
    [inc, azi]: (n,2) array of floats
        A numpy array of incs and axis in radians

    '''
    # make sure it's a unit vector
    vec = vec / np.linalg.norm(vec, axis=-1).reshape(-1, 1)
    vec = vec.reshape(-1, 3)

    # if it's nev then need to do the shuffle
    if nev:
        y, x, z = vec.T
        vec = np.array([x, y, z]).T

    return _get_angles(vec)


def _get_transform(inc, azi):
    trans = np.array([
        [np.cos(inc) * np.cos(azi), -np.sin(azi), np.sin(inc) * np.cos(azi)],
        [np.cos(inc) * np.sin(azi), np.cos(azi), np.sin(inc) * np.sin(azi)],
        [-np.sin(inc), np.zeros_like(inc), np.cos(inc)]
    ]).T

    return trans


def get_transform(
    survey
):
    """
    Determine the transform for transforming between NEV and HLA coordinate
    systems.

    Parameters
    ----------
    survey: (n,3) array of floats
        The [md, inc, azi] survey listing array.

    Returns
    -------
    transform: (n,3,3) array of floats
    """
    survey = survey.reshape(-1, 3)
    inc = np.array(survey[:, 1])
    azi = np.array(survey[:, 2])

    return _get_transform(inc, azi)


def NEV_to_HLA(
    survey: Annotated[NDArray, Literal["N", 3]],
    NEV: Union[
        Annotated[NDArray, Literal["N", 3]],
        Annotated[NDArray, Literal[3, 3, "N"]]
    ],
    cov: bool = True
) -> Union[
        Annotated[NDArray, Literal['..., 3']],
        Annotated[NDArray, Literal['3, 3, ...']]
]:
    """
    Transform from NEV to HLA coordinate system.

    Parameters
    ----------
    survey: (n,3) array of floats
        The [md, inc, azi] survey listing array.
    NEV: (d,3) or (3,3,d) array of floats
        The NEV coordinates or covariance matrices.
    cov: boolean
        If cov is True then a (3,3,d) array of covariance matrices
        is expected, else a (d,3) array of coordinates.

    Returns
    -------
    HLAs: NDArray
        Either a transformed (n,3) array of HLA coordinates or an
        (3,3,n) array of HLA covariance matrices.
    """

    trans = get_transform(survey)

    if cov:
        HLAs = np.einsum(
            '...ik,...jk',
            np.einsum(
                '...ik,...jk', trans, NEV.T
            ),
            trans
        ).T

    else:
        NEV = NEV.reshape(-1, 3)
        HLAs = np.einsum(
            '...k,...jk', NEV, trans
        )

    return HLAs


def HLA_to_NEV(survey, HLA, cov=True, trans=None):
    if trans is None:
        trans = get_transform(survey)

    if cov:
        NEVs = np.einsum(
            '...ik,jk...',
            np.einsum(
                '...ki,...jk', trans, HLA.T
            ),
            trans.T
        ).T

    else:
        NEVs = np.einsum(
            'k...,jk...', HLA.T, trans.T
        )

    return np.swapaxes(NEVs, 0, 1)


def get_sigmas(cov, long=False):
    """
    Extracts the sigma values of a covariance matrix along the principle axii.

    Parameters
    ----------
    cov: (n,3,3) array of floats

    Returns
    -------
    arr: (n,3) array of floats
    """

    assert cov.shape[-2:] == (3, 3), "Cov is the wrong shape"

    cov = cov.reshape(-1, 3, 3)

    aa, ab, ac = cov[:, :, 0].T
    ab, bb, bc = cov[:, :, 1].T
    ac, bc, cc = cov[:, :, 2].T

    if long:
        return (aa, bb, cc, ab, ac, bc)
    else:
        return (np.sqrt(aa), np.sqrt(bb), np.sqrt(cc))


def get_unit_vec(vec):
    vec = vec / np.linalg.norm(vec)

    return vec


def linear_convert(data, factor):
    flag = False
    if not isinstance(data, list):
        flag = True
        data = [data]
    converted = [d * factor if d is not None else None for d in data]
    if flag:
        return converted[0]
    else:
        return converted


def make_cov(a, b, c, long=False):
    # a, b, c = np.sqrt(np.array([a, b, c]))
    if long:
        cov = np.array([
            [a * a, a * b, a * c],
            [a * b, b * b, b * c],
            [a * c, b * c, c * c]
        ])

    else:
        cov = np.array([
            [a * a, np.zeros_like(a), np.zeros_like(a)],
            [np.zeros_like(a), b * b, np.zeros_like(a)],
            [np.zeros_like(a), np.zeros_like(a), c * c]
        ])

    return cov.T


def dls_from_radius(radius):
    """
    Returns the dls in degrees from a radius.
    """
    if isinstance(radius, np.ndarray):
        circumference = np.full_like(radius, np.inf)
        circumference = np.where(
            radius != 0,
            2 * np.pi * radius,
            circumference
        )
    else:
        if radius == 0:
            return np.inf
        circumference = 2 * np.pi * radius
    dls = 360 / circumference * 30

    return dls


def radius_from_dls(dls):
    """
    Returns the radius in meters from a DLS in deg/30m.
    """
    if isinstance(dls, np.ndarray):
        circumference = np.full_like(dls, np.inf)
        circumference = np.where(
            dls != 0,
            (30 / dls) * 360,
            circumference
        )
    else:
        if dls == 0:
            return np.inf
        circumference = (30 / dls) * 360
    radius = circumference / (2 * np.pi)

    return radius


def errors_from_cov(cov, data=False):
    """
    Parameters
    ----------
    cov: (n, 3, 3) array
        The error covariance matrices.
    data: bool (default: False)
        If True returns a dictionary, else returns a list.
    """
    nn, ne, nv, _, ee, ev, _, _, vv = (
        cov.reshape(-1, 9).T
    )

    if data:
        return {
            i: {
                'nn': _nn, 'ne': _ne, 'nv': _nv,
                'ee': _ee, 'ev': _ev, 'vv': _vv
            }
            for i, (_nn, _ne, _nv, _ee, _ev, _vv)
            in enumerate(zip(nn, ne, nv, ee, ev, vv))
        }

    return np.array([nn, ne, nv, ee, ev, vv]).T


def _get_arc_pos_and_vec(dogleg, radius):
    pos = np.array([
        np.cos(dogleg),
        0.,
        np.sin(dogleg)
    ]) * radius
    pos[0] = radius - pos[0]

    vec = np.array([
        np.sin(dogleg),
        0.,
        np.cos(dogleg)
    ])
    return (pos, vec)


class Arc:
    def __init__(self, dogleg, radius):
        """
        Generates a generic arc that can be transformed with a specific pos
        and vec via a transform method. The arc is initialized at a local
        origin and kicks off down and to the north (assuming an NEV coordinate
        system).

        Parameters
        ----------
        dogleg: float
            The sweep angle of the arc in radians.
        radius: float
            The radius of the arc in meters.

        Returns
        -------
        arc: Arc object
        """
        self.dogleg = dogleg
        self.radius = radius
        self.delta_md = dogleg * radius

        self.pos, self.vec = _get_arc_pos_and_vec(dogleg, radius)


    def transform(self, toolface, pos=None, vec=None, target=False):
        """
        Transforms an Arc to a position and orientation.

        Parameters
        ----------
        pos: (,3) array
        The desired position to transform the arc.
        vec: (,3) array
            The orientation unit vector to transform the arc.
        target: bool
            If true, returned arc vector is reversed.

        Returns
        -------
        tuple (pos_new, vec_new)
        pos_new: (,3) array
            The position at the end of the arc post transform.
        vec_new: (,3) array
            The unit vector at the end of the arc post transform.
        """
        if vec is None:
            vec = np.array([0., 0., 1.])
        if target:
            vec *= -1
        inc, azi = get_angles(vec, nev=True).reshape(2)
        angles = [
            toolface,
            inc,
            azi
        ]
        r = R.from_euler('zyz', angles, degrees=False)

        pos_new, vec_new = r.apply(np.vstack((self.pos, self.vec)))

        # make sure vec_new is a unit vector:
        vec_new = get_unit_vec(vec_new)

        if pos is not None:
            pos_new += pos
        if target:
            vec_new *= -1

        return (pos_new, vec_new)


def get_arc(
    dogleg, radius, toolface, pos=None, vec=None, target=False
) -> tuple:
    """Creates an Arc instance and transforms it to the desired position
    and orientation.

    Parameters
    ----------
    dogleg: float
        The swept angle of the arc (arc angle) in radians.
    radius: float
        The radius of the arc (in meters).
    toolface: float
        The toolface angle in radians (relative to the high side) to rotate the
        arc at the desired position and orientation.
    pos: (,3) array
        The desired position to transform the arc.
    vec: (,3) array
        The orientation unit vector to transform the arc.
    target: bool
        If true, returned arc vector is reversed.

    Returns
    -------
    tuple of (pos_new, vec_new, arc.delta_md)
    pos_new: (,3) array
        The position at the end of the arc post transform.
    vec_new: (,3) array
        The unit vector at the end of the arc post transform.
    arc.delta_md: int
        The arc length of the arc.
    """
    arc = Arc(dogleg, radius)
    pos_new, vec_new = arc.transform(toolface, pos, vec, target)

    return (pos_new, vec_new, arc.delta_md)


def annular_volume(od: float, id: float = None, length: float = None):
    """
    Calculate an annular volume.

    If no ``id`` is provided then circular volume is calculated. If no
    ``length`` is provided, then the unit volume is calculated (i.e. the
    area).

    Units are assumed consistent across input parameters, i.e. the
    calculation is dimensionless.

    Parameters
    ----------
    od: float
        The outer diameter.
    id: float | None, optional
        The inner diameter, default is 0.
    length : float | None, optional
        The length of the annulus.

    Returns
    -------
    annular_volume: float
        The (unit) volume of the annulus or cylinder.

    Examples
    --------
    In the following example we calculate annular volume along a 1,000 meter
    section length of 9 5/8" casing inside 12 1/4" hole.

    >>> from welleng.utils import annular_volume
    >>> from welleng.units import ureg
    >>> av = annular_volume(
    ...     od=ureg('12.25 inch').to('meters),
    ...     id=ureg(f'{9+5/8} inch').to('meter'),
    ...     length=ureg('1000 meter')
    ... )
    >>> print(av)
    3.491531223156194 meter ** 3
    """
    length = 1 if length is None else length
    id = 0 if id is None else id
    annular_unit_volume = (np.pi * (od - id)**2) / 4
    annular_volume = annular_unit_volume * length

    return annular_volume


def _decimal2dms(decimal: tuple, ndigits: int = None) -> tuple:
    try:
        _decimal, direction = decimal
    except (TypeError, ValueError):
        _decimal = decimal[0] if isinstance(decimal, np.ndarray) else decimal
        direction = None
    _decimal = float(_decimal)
    minutes, seconds = divmod(abs(_decimal) * 3600, 60)
    _, minutes = divmod(minutes, 60)

    return np.array([
        int(_decimal),
        int(minutes),
        seconds if ndigits is None else round(seconds, ndigits)
    ]) if direction is None else np.array([
        int(_decimal),
        int(minutes),
        seconds if ndigits is None else round(seconds, ndigits),
        direction
    ], dtype=object)


def decimal2dms(decimal: tuple | NDArray, ndigits: int = None) -> tuple | NDArray:
    """Converts a decimal lat, lon to degrees, minutes and seconds.

    Parameters
    ----------
    decimal : tuple | arraylike
        A tuple of (lat, direction) or (lon, direction) or arraylike of
        ((lat, direction), (lon, direction)) coordinates.
    ndigits: int (default is None)
        If specified, rounds the seconds decimal to the desired number of
        digits.

    Returns
    -------
    dms: arraylike
        An array of (degrees, minutes, seconds, direction).

    Examples
    --------
    If you want to convert the lat/lon coordinates for Den Haag from decimals
    to degrees, minutes and seconds:

    >>> LAT, LON = [(52.078663, 'N'), (4.288788, 'E')]
    >>> dms = decimal2dms((LAT, LON), ndigits=6)
    >>> print(dms)
    [[52 4 43.1868 'N']
     [4 17 19.6368 'E']]
    """
    flag = False
    _decimal = np.array(decimal)
    if _decimal.dtype == np.float64:
        _decimal = _decimal.reshape((-1, 1))
        flag = True
    try:
        dms = np.apply_along_axis(_decimal2dms, -1, _decimal, ndigits)
    except np.exceptions.AxisError:
        dms = _decimal2dms(_decimal, ndigits)

    if dms.shape == (4,):
        return tuple(dms)
    else:
        return dms.reshape((-1, 3)) if flag else dms


def _dms2decimal(dms: NDArray, ndigits: int = None) -> NDArray:
    try:
        degrees, minutes, seconds, direction = dms
    except ValueError:
        degrees, minutes, seconds = dms
        direction = None

    decimal = abs(degrees) + minutes / 60 + seconds / 3600

    return np.array([
        np.copysign(
            decimal if ndigits is None else round(decimal, ndigits),
            degrees
        )
    ]) if direction is None else np.array([
        np.copysign(
            decimal if ndigits is None else round(decimal, ndigits),
            degrees
        ),
        direction
    ], dtype=object)


def dms2decimal(dms: tuple | NDArray, ndigits: int = None) -> NDArray:
    """Converts a degrees, minutes and seconds lat, lon to decimals.

    Parameters
    ----------
    dms : tuple | arraylike
        A tuple or arraylike of (degrees, minutes, seconds, direction) lat
        and/or lon or arraylike of lat, lon coordinates.
    ndigits: int (default is None)
        If specified, rounds the decimal to the desired number of digits.

    Returns
    -------
    degrees: arraylike
        A tuple or array of lats and/or longs in decimals.

    Examples
    --------
    If you want to convert the lat/lon coordinates for Den Haag from degrees,
    minutes and seconds to decimals:

    >>> LAT, LON = (52, 4, 43.1868, 'N'), (4, 17, 19.6368, 'E')
    >>> decimal = dms2decimal((LAT, LON), ndigits=6)
    >>> print(decimal)
    [[52.078663 'N']
     [4.288788 'E']]
    """
    result = np.apply_along_axis(
        _dms2decimal, -1, np.array(dms, dtype=object), ndigits
    )

    if result.shape == ():
        return float(result)
    elif result.shape == (1,):
        return float(result[0])
    elif result.shape[-1] == 1:
        return result.reshape(-1)
    else:
        return result


def pprint_dms(dms, symbols: bool = True, return_data: bool = False):
    """Pretty prints a (decimal, minutes, seconds) tuple or list.

    Parameters
    ----------
    dms: tuple | list
        An x or y or northing or easting (degree, minute, second).
    symbols: bool (default: True)
        Whether to print symbols for (deg, min, sec).
    return_data: bool (default: False)
        If True then will return the string rather than print it.
    """
    if symbols:
        try:
            deg, min, sec = dms
            text = f"{deg}\N{DEGREE SIGN}, {min}', {sec}\""
        except ValueError:
            deg, min, sec, _ = dms
            text = f"{deg}\N{DEGREE SIGN}, {min}', {sec}\" {_}"

    else:
        try:
            deg, min, sec = dms
            text = f"{deg} deg, {min} min, {sec} sec"
        except ValueError:
            deg, min, sec, _ = dms
            text = f"{deg} deg, {min} min, {sec} sec {_}"

    if return_data:
        return text
    else:
        print(text)


def dms_from_string(text):
    """Extracts the values from a string dms x or y or northing or easting.
    """
    pattern = re.compile(r'(\d+)\s*(?:°|deg)?,\s*(\d+)\s*(?:\'|min)?,\s*(\d+(?:\.\d+)?)\s*(sec)?\s*.*?(\S+)?$', re.IGNORECASE)
    matches = pattern.findall(text)

    if matches:
        deg, min, sec_str = matches[0][:3]
        sec = float(sec_str)
        final_data = matches[0][-1] if matches[0][-1] else None

        if final_data:
            return (int(deg), int(min), sec, final_data)
        else:
            return (int(deg), int(min), sec)

    else:
        return


def get_toolface(pos1: NDArray, vec1: NDArray, pos2: NDArray) -> float:
    """Returns the toolface of an offset position relative to a reference
    position and vector.

    Parameters
    ----------
    pos1: ndarray
        The reference NEV coordinate, e.g. current location.
    vec1: ndarray
        The reference NEV unit vector, e.g. current vector heading.
    pos2: ndarray
        The offset NEV coordinate, e.g. a target position.

    Returns
    -------
    toolface: float
        The toolface (bearing or required heading) in radians to pos2 from pos1
        with vec1.
    """
    inc, azi = get_angles(vec1, nev=True)[0]
    r = R.from_euler('zy', [-azi, -inc], degrees=False)
    pos = r.apply(pos2 - pos1)

    return np.arctan2(*(np.flip(pos[:2])))


if NUMBA:
    NUMBA_FUNCS = (
        _get_angles, _get_arc_pos_and_vec
    )
    for func in NUMBA_FUNCS:
        numbafy(func)
