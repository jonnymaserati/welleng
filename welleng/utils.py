from typing import List, Union

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from numba import njit
    NUMBA = True
except ImportError:
    NUMBA = False


class MinCurve:
    def __init__(
        self,
        md,
        inc,
        azi,
        start_xyz=None,
        unit="meters"
    ):
        """
        Generate geometric data from a well bore survey.

        Params:
            md: list or 1d array of floats
                Measured depth along well path from a datum.
            inc: list or 1d array of floats
                Well path inclincation (relative to z/tvd axis where 0
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
        self.start_xyz = start_xyz if start_xyz is not None else np.array([0., 0., 0.])
        self.unit = unit

        # make two slices with a difference or 1 index to enable array
        # calculations
        md_1 = md[:-1]
        md_2 = md[1:]

        inc_1 = inc[:-1]
        inc_2 = inc[1:]

        azi_1 = azi[:-1]
        azi_2 = azi[1:]

        self.dogleg = np.zeros(survey_length)
        self.dogleg[1:] = get_dogleg(inc_1, azi_1, inc_2, azi_2)

        # calculate rf and assume rf is 1 where dogleg is 0
        self.rf = np.ones(survey_length)
        idx = np.where(self.dogleg != 0)
        self.rf[idx] = 2 / self.dogleg[idx] * np.tan(self.dogleg[idx] / 2)

        # calculate the change in md between survey stations
        temp = np.array(md_2) - np.array(md_1)
        self.delta_md = np.zeros(survey_length)
        self.delta_md[1:] = temp

        # calculate change in y direction (north)
        temp = (
            self.delta_md[1:]
            / 2
            * (
                np.sin(inc_1) * np.cos(azi_1)
                + np.sin(inc_2) * np.cos(azi_2)
            )
            * self.rf[1:]
        )
        self.delta_y = np.zeros(survey_length)
        self.delta_y[1:] = temp

        # calculate change in x direction (east)
        temp = (
            self.delta_md[1:]
            / 2
            * (
                np.sin(inc_1) * np.sin(azi_1)
                + np.sin(inc_2) * np.sin(azi_2)
            )
            * self.rf[1:]
        )
        self.delta_x = np.zeros(survey_length)
        self.delta_x[1:] = temp

        # calculate change in z direction
        temp = (
            self.delta_md[1:]
            / 2
            * (np.cos(inc_1) + np.cos(inc_2))
            * self.rf[1:]
        )
        self.delta_z = np.zeros(survey_length)
        self.delta_z[1:] = temp

        # calculate the dog leg severity
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = np.degrees(self.dogleg[1:]) / self.delta_md[1:]
        self.dls = np.zeros(survey_length)
        mask = np.where(temp != np.nan)
        self.dls[1:][mask] = temp[mask]

        if unit == "meters":
            self.dls *= 30
        else:
            self.dls *= 100

        # cumulate the coordinates and add surface coordinates
        self.poss = np.vstack(
            np.cumsum(
                np.array([self.delta_x, self.delta_y, self.delta_z]).T, axis=0
            ) + self.start_xyz
        )


def get_dogleg(inc1, azi1, inc2, azi2):
    """
    Calculate DLS (Curvature) between two stations in rad/m.
    """

    dogleg = np.arccos(
        np.cos(inc2 - inc1)
        - (np.sin(inc1) * np.sin(inc2))
        * (1 - np.cos(azi2 - azi1))
    )
    return dogleg


def get_vec(inc, azi, nev=False, r=1, deg=True):
    """
    Convert inc and azi into a vector.

    Params:
        inc: array of n floats
            Inclination relative to the z-axis (up)
        azi: array of n floats
            Azimuth relative to the y-axis
        r: float or array of n floats
            Scalar to return a scaled vector

    Returns:
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
        pos: List,
        start_xyz: Union[List, None] = None,
        start_nev: Union[List, None] = None
) -> np.ndarray:
    """
    Convert [x, y, z] coordinates to [n, e, tvd] coordinates.

    Params:
        pos: (n,3) array of floats
            Array of [x, y, z] coordinates
        start_xyz: (,3) array of floats
            The datum of the [x, y, z] cooardinates
        start_nev: (,3) array of floats
            The datum of the [n, e, tvd] coordinates

    Returns:
        An (n,3) array of [n, e, tvd] coordinates.
    """

    start_xyz = start_xyz if start_xyz is not None else np.array([0., 0., 0.])
    start_nev = start_nev if start_nev is not None else np.array([0., 0., 0.])

    e, n, v = (
        np.array([pos]).reshape(-1, 3) - np.array([start_xyz])
    ).T

    return np.array([n, e, v]).T + np.array([start_nev])


def get_xyz(pos, start_xyz=None, start_nev=None):

    start_xyz = start_xyz if start_xyz is not None else np.array([0., 0., 0.])
    start_nev = start_nev if start_nev is not None else np.array([0., 0., 0.])

    y, x, z = (
        np.array([pos]).reshape(-1, 3) - np.array([start_nev])
    ).T

    return np.array([x, y, z]).T + np.array([start_xyz])


def _get_angles(vec):
    xy = vec[:, 0] ** 2 + vec[:, 1] ** 2

    # for elevation angle defined from Z-axis down
    inc = np.arctan2(np.sqrt(xy), vec[:, 2])

    azi = (np.arctan2(vec[:, 0], vec[:, 1]) + (2 * np.pi)) % (2 * np.pi)

    return np.stack((inc, azi), axis=1)


if NUMBA:
    _get_angles = njit(_get_angles)


def get_angles(vec, nev=False):
    '''
    Determines the inclination and azimuth from a vector.

    Params:
        vec: (n,3) array of floats
        nev: boolean (default: False)
            Indicates if the vector is in (x,y,z) or (n,e,v) coordinates.

    Returns:
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

    Params:
        survey: (n,3) array of floats
        The [md, inc, azi] survey listing array.

    Returns:
        transform: (n,3,3) array of floats
    """
    survey = survey.reshape(-1, 3)
    inc = np.array(survey[:, 1])
    azi = np.array(survey[:, 2])

    return _get_transform(inc, azi)


def NEV_to_HLA(survey, NEV, cov=True):
    """
    Transform from NEV to HLA coordinate system.

    Params:
        survey: (n,3) array of floats
        The [md, inc, azi] survey listing array.
        NEV: (d,3) or (3,3,d) array of floats
            The NEV coordinates or covariance matrices.
        cov: boolean
            If cov is True then a (3,3,d) array of covariance matrices
            is expected, else a (d,3) array of coordinates.

    Returns:
        Either a transformed (n,3) array of HLA coordinates or an
        (3,3,n) array of HLA covariance matrices.
    """

    trans = get_transform(survey)

    if cov:
        HLAs = [
            np.dot(np.dot(t, NEV.T[i]), t.T) for i, t in enumerate(trans)
        ]
        HLAs = np.vstack(HLAs).reshape(-1, 3, 3).T

    else:
        NEV = NEV.reshape(-1, 3)
        HLAs = [
            np.dot(NEV[i], t.T) for i, t in enumerate(trans)
        ]

    return HLAs


def HLA_to_NEV(survey, HLA, cov=True, trans=None):
    if trans is None:
        trans = get_transform(survey)

    if cov:
        NEVs = [
            np.dot(np.dot(t.T, HLA.T[i]), t) for i, t in enumerate(trans)
        ]
        NEVs = np.vstack(NEVs).reshape(-1, 3, 3).T

    else:
        NEVs = [
            np.dot(hla, t) for hla, t in zip(HLA, trans)
        ]

    return np.vstack(NEVs).reshape(HLA.shape)


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

    # a, b, c = np.array([
    #     np.sqrt(cov[:, 0, 0]),
    #     np.sqrt(cov[:, 1, 1]),
    #     np.sqrt(cov[:, 2, 2])
    # ])

    # return (a, b, c)


def get_unit_vec(vec):
    vec = vec / np.linalg.norm(vec)

    return vec


def linear_convert(data, factor):
    flag = False
    if type(data) != list:
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
                'nn': _nn, 'ne': _ne, 'nv': _nv, 'ee': _ee, 'ev': _ev, 'vv': _vv
            }
            for i, (_nn, _ne, _nv, _ee, _ev, _vv)
            in enumerate(zip(nn, ne, nv, ee, ev, vv))
        }

    return np.array([nn, ne, nv, ee, ev, vv]).T


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

        self.pos = np.array([
            np.cos(dogleg),
            0.,
            np.sin(dogleg)
        ]) * radius
        self.pos[0] = radius - self.pos[0]

        self.vec = np.array([
            np.sin(dogleg),
            0.,
            np.cos(dogleg)
        ])

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

        if pos is not None:
            pos_new += pos
        if target:
            vec_new *= -1

        return (pos_new, vec_new)


def get_arc(dogleg, radius, toolface, pos=None, vec=None, target=False):
    """
    Creates an Arc instance and transforms it to the desired position and
    orientation.

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
