import numpy as np
from numba import njit


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
        self.start_xyz = start_xyz
        self.unit = unit

        # make two slices with a difference or 1 index to enable array
        # calculations
        md_1 = md[:-1]
        md_2 = md[1:]

        inc_1 = inc[:-1]
        inc_2 = inc[1:]

        azi_1 = azi[:-1]
        azi_2 = azi[1:]

        # calculate the dogleg
        temp = np.arccos(
            np.cos(inc_2 - inc_1)
            - (np.sin(inc_1) * np.sin(inc_2))
            * (1 - np.cos(azi_2 - azi_1))
        )

        self.dogleg = np.zeros(survey_length)
        self.dogleg[1:] = temp

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


def get_nev(pos, start_xyz=[0., 0., 0.], start_nev=[0., 0., 0.]):
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
    e, n, v = (
        np.array([pos]).reshape(-1,3) - np.array([start_xyz])
    ).T

    return (np.array([n, e, v]).T + np.array([start_nev]))


def get_xyz(pos, start_xyz=[0., 0., 0.], start_nev=[0., 0., 0.]):
    y, x, z = (
        np.array([pos]).reshape(-1, 3) - np.array([start_nev])
    ).T

    return (np.array([x, y, z]).T + np.array([start_xyz]))


@njit
def _get_angles(vec):
    xy = vec[:, 0] ** 2 + vec[:, 1] ** 2
    inc = np.arctan2(np.sqrt(xy), vec[:, 2])  # for elevation angle defined from Z-axis down
    azi = (np.arctan2(vec[:, 0], vec[:, 1]) + (2 * np.pi)) % (2 * np.pi)

    return np.stack((inc, azi), axis=1)


def get_angles(vec, nev=False):
    '''
    Determines the inclination and azimuth from a vector.

    Params:
        vec: (n,3) array of floats
        nev: boolean (default: False)
            Indicates if the vector is in (x,y,z) or (n,e,v) coordinates.

    Returns:
        [inc, azi]: (n,2) array of floats
            A numpy array of incs and azis in radians

    '''
    # make sure it's a unit vector
    vec = vec / np.linalg.norm(vec, axis=-1).reshape(-1, 1)
    vec = vec.reshape(-1, 3)

    # if it's nev then need to do the shuffle
    if nev:
        y, x, z = vec.T
        vec = np.array([x, y, z]).T

    return _get_angles(vec)

    # xy = vec[:, 0] ** 2 + vec[:, 1] ** 2
    # inc = np.arctan2(np.sqrt(xy), vec[:, 2])  # for elevation angle defined from Z-axis down
    # azi = (np.arctan2(vec[:, 0], vec[:, 1]) + (2 * np.pi)) % (2 * np.pi)

    # return np.stack((inc, azi), axis=1)


# @njit
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

    # trans = np.array([
    #     [np.cos(inc) * np.cos(azi), -np.sin(azi), np.sin(inc) * np.cos(azi)],
    #     [np.cos(inc) * np.sin(azi), np.cos(azi), np.sin(inc) * np.sin(azi)],
    #     [-np.sin(inc), np.zeros_like(inc), np.cos(inc)]
    # ]).T

    # return trans

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
            is expecte, else a (d,3) array of coordinates.

    Returns:
        Either a transformed (n,3) array of HLA coordinates or an
        (3,3,n) array of HLA covariance matrices. 
    """

    trans = get_transform(survey)

    if cov:
        HLAs = [
            np.dot(np.dot(t, NEV.T[i]), t.T) for i, t in enumerate(trans)
        ]

        HLAs = np.vstack(HLAs).reshape(-1,3,3).T

    else:
        NEV = NEV.reshape(-1,3)
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

        NEVs = np.vstack(NEVs).reshape(-1,3,3).T

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
