import numpy as np
from scipy.spatial.transform import Rotation as R

from numpy.typing import ArrayLike, NDArray
from typing import Annotated, Literal, Union, List

try:
    from numba import njit
    NUMBA = True
except ImportError:
    NUMBA = False


class MinCurve:
    def __init__(
        self,
        md: List[float],
        inc: List[float],
        azi: List[float],
        start_xyz: List[float] = None,
        dls_denominator: float = None
    ):
        """
        Generate geometric data from a well bore survey.

        Parameters
        ----------
        md: array_like
            1D list or array of measured depth along well path from a datum.
        inc: array_like
            1D list or array of well path inclination (relative to z/tvd axis
            where 0 indicates down), in radians.
        azi: array_like
            1D array of well path azimuth (relative to y/North axis) in radians.
        dls_denominator : float, optional
            The denominator used to calculate the DLS, e.g. for a DLS in
            degrees per 30 meters, the default value ``dls_denominator=30``
            should be used.


        Variables
        ---------
        md : ndarray
            The 1D array of measured depth.
        """

        self.md = np.array(md)
        survey_length = len(self.md)
        assert survey_length > 1, "Survey must have at least two rows"

        self.inc = np.array(inc)
        self.azi = np.array(azi)
        self.start_xyz = (
            np.zeros(3) if start_xyz is None
            else np.array(start_xyz)
        )
        self.dls_denominator = (
            30 if dls_denominator is None
            else dls_denominator
        )

        # make two slices with a difference or 1 index to enable array
        # calculations
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
        self.delta_md = np.diff(self.md, prepend=0)

        # calculate change in y direction (north)
        self.delta_y = np.zeros(survey_length)
        self.delta_y[1:] = (
            self.delta_md[1:] / 2
            * (
                np.sin(inc_1) * np.cos(azi_1)
                + np.sin(inc_2) * np.cos(azi_2)
            )
            * self.rf[1:]
        )

        # calculate change in x direction (east)
        self.delta_x = np.zeros(survey_length)
        self.delta_x[1:] = (
            self.delta_md[1:] / 2
            * (
                np.sin(inc_1) * np.sin(azi_1)
                + np.sin(inc_2) * np.sin(azi_2)
            )
            * self.rf[1:]
        )

        # calculate change in z direction
        self.delta_z = np.zeros(survey_length)
        self.delta_z[1:] = (
            self.delta_md[1:]
            / 2
            * (np.cos(inc_1) + np.cos(inc_2))
            * self.rf[1:]
        )

        # calculate the dog leg severity
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = np.degrees(self.dogleg[1:]) / self.delta_md[1:]

        self.dls = np.zeros(survey_length)
        mask = np.where(temp != np.nan)
        self.dls[1:][mask] = temp[mask]
        self.dls *= self.dls_denominator

        # cumulate the coordinates and add surface coordinates
        self.poss = np.vstack(
            np.cumsum(
                np.array([self.delta_x, self.delta_y, self.delta_z]).T, axis=0
            ) + self.start_xyz
        )


def get_dogleg(inc1, azi1, inc2, azi2):
    """
    Calculated the dogleg (arc angle) of a well path between two points.

    Parameters
    ----------
    inc1 : float | ndarray
        A float or 1D array of inclination (in radians) at the current station.
    azi1 : float | ndarray
        A float or 1D array of azimuth (in radians) at the current station.
    inc2 : float | ndarray
        A float or 1D array of inclination (in radians) at the next station.
    azi2 : float | ndarray
        A float or 1D array of azimuth (in radians) at the next station.

    Returns
    -------
    dogleg : float | ndarray
        A float or 1D array of doglegs (in radians) between the stations.
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

    Parameters
    ----------
    inc : float or array_like
        Float or 1D list or array of inclination relative to the z-axis (up).
    azi : float or array_like
        Float or 1D list or array of azimuth relative to the y-axis.
    r : float or array_like, optional
        Float or 1D list or array of scalar to return a scaled vector.
    deg : bool, optional
        Indicates whether the inclination and azimuth angles are in the default
        degrees (``True``) or radians (``False``).

    Returns
    -------
    vec : ndarray
        An (n, 3) array of vectors.
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


def _get_start_nev_and_xyz(start_nev, start_xyz):
    start_xyz = np.array([0., 0., 0.]) if start_xyz is None else start_xyz
    start_nev = np.array([0., 0., 0.]) if start_nev is None else start_nev

    return (start_nev, start_xyz)


def get_nev(
    pos, start_xyz=None, start_nev=None
):
    """
    Convert [x, y, z] coordinates to [n, e, tvd] coordinates.

    Parameters
    ----------
    pos : array_like
        An [x, y, z] list or array or an (n, 3) array of [x, y, z]
        coordinates.
    start_xyz : array_like | None, optional
        A list or array of the datum of the [x, y, z] coordinates. The default
        is [0, 0, 0].
    start_nev : array_like | None, optional
        A list or array of the datum of the [n, e, tvd] coordinates. The
        default is [0, 0, 0].

    Returns
    -------
    pos_nev : ndarray
        An (n, 3) array of [n, e, tvd] coordinates.

    See Also
    --------
    welleng.utils.get_xyz
    """
    start_xyz, start_nev = _get_start_nev_and_xyz(start_nev, start_xyz)

    e, n, v = (
        np.array([pos]).reshape(-1, 3) - np.array([start_xyz])
    ).T

    pos_nev = np.array([n, e, v]).T + np.array([start_nev])

    return pos_nev


def get_xyz(pos, start_xyz=None, start_nev=None):
    """
    Convert [n, e, tvd] coordinates to [x, y, z] coordinates.

    Parameters
    ----------
    pos : array_like
        An [n, e, v] list or array or an (n, 3) array of [n, e, v]
        coordinates.
    start_xyz : array_like | None, optional
        A list or array of the datum of the [n, e, v] coordinates. The default
        is [0, 0, 0].
    start_nev : array_like | None, optional
        A list or array of the datum of the [x, y, z] coordinates. The
        default is [0, 0, 0].

    Returns
    -------
    pos_xyz : ndarray
        An (n, 3) array of [x, y, z] coordinates.

    See Also
    --------
    welleng.utils.get_nev
    """
    start_xyz, start_nev = _get_start_nev_and_xyz(start_nev, start_xyz)

    y, x, z = (
        np.array([pos]).reshape(-1, 3) - np.array([start_nev])
    ).T

    return (np.array([x, y, z]).T + np.array([start_xyz]))


def _get_angles(vec):
    xy = vec[:, 0] ** 2 + vec[:, 1] ** 2
    inc = np.arctan2(np.sqrt(xy), vec[:, 2])  # for elevation angle defined from Z-axis down
    azi = (np.arctan2(vec[:, 0], vec[:, 1]) + (2 * np.pi)) % (2 * np.pi)

    return np.stack((inc, azi), axis=1)


if NUMBA:
    _get_angles = njit(_get_angles)


def get_angles(
    vec: Annotated[NDArray, Literal["N", 3]], nev: bool = False
) -> Annotated[NDArray, Literal["N", 2]]:
    '''
    Determines the inclination and azimuth from a vector.

    Parameters
    ----------
    vec : ndarray
        An (n, 3) array of floats of vectors.
    nev : bool, optional
        Indicates if the vectors are in default (x, y, z) when ``nev=False``
        or (n, e, v) coordinates.

    Returns
    -------
    angles : ndarray
        An (n, 2) array of inclinations and azimuths in radians.

    See Also
    --------
    welleng.utils.get_vec
    '''
    # make sure it's a unit vector
    vec = vec / np.linalg.norm(vec, axis=-1).reshape(-1, 1)
    vec = vec.reshape(-1, 3)

    # if it's nev then need to do the shuffle
    if nev:
        y, x, z = vec.T
        vec = np.array([x, y, z]).T

    angles = _get_angles(vec)

    return angles


def _get_transform(inc, azi):
    trans = np.array([
        [np.cos(inc) * np.cos(azi), -np.sin(azi), np.sin(inc) * np.cos(azi)],
        [np.cos(inc) * np.sin(azi), np.cos(azi), np.sin(inc) * np.sin(azi)],
        [-np.sin(inc), np.zeros_like(inc), np.cos(inc)]
    ]).T

    return trans


def get_transform(
    survey: Annotated[NDArray, Literal["N", 3]]
) -> Annotated[NDArray, Literal["N", 3, 3]]:
    """
    Determine the transform for transforming between NEV and HLA coordinate
    systems.

    Parameters
    ----------
    survey : ndarray
        An (n, 3) array of the [md, inc, azi] survey listing.

    Returns
    -------
    transform: ndarray
        An (n, 3, 3) array of floats representing the transform matrix for
        each survey station.
    """
    survey = survey.reshape(-1, 3)
    inc = np.array(survey[:, 1])
    azi = np.array(survey[:, 2])

    transform = _get_transform(inc, azi)

    return transform


def nev_to_hla(
    survey: Annotated[NDArray, Literal["N", 3]],
    nev: Union[
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
    survey : ndarray
        An (n, 3) array of an [md, inc, azi] survey listing array.
    nev : ndarray
        An (n, 3) or (3, 3, n) array of the NEV coordinates or covariance
        matrices.
    cov : bool, optional
        If cov is True then a (3, 3, n) array of covariance matrices
        is expected, else an (n, 3) array of coordinates.

    Returns
    -------
    hla : ndarray
        Either a transformed (n, 3) array of HLA coordinates or an
        (3, 3, n) array of HLA covariance matrices.

    See Also
    --------
    welleng.utils.hla_to_nev
    """

    trans = get_transform(survey)

    if cov:
        hla = np.einsum(
            '...ik,...jk',
            np.einsum(
                '...ik,...jk', trans, nev.T
            ),
            trans
        ).T

    else:
        nev = nev.reshape(-1, 3)

        hla = np.einsum(
            '...k,...jk', nev, trans
        )

    return hla


def hla_to_nev(survey, hla, cov=True, trans=None):
    """
    Transform from HLA to NEV coordinate system.

    Parameters
    ----------
    survey : ndarray
        An (n, 3) array of the [md, inc, azi] survey listing.
    hla : ndarray
        An (n, 3) or (3, 3, n) array of the NEV coordinates or covariance
        matrices.
    cov : bool, optional
        If cov is ``True`` then a (3, 3, n) array of covariance matrices
        is expected, else a (n, 3) array of coordinates.
    trans : ndarray | None, optional
        An (n, 3, 3) array of transforms. If default ``None`` then they will be
        calculated.

    Returns
    -------
    nev : NDArray
        Either a transformed (n, 3) array of nev coordinates or an
        (3, 3, n) array of nev covariance matrices.

    See Also
    --------
    welleng.utils.nev_to_hla
    """
    if trans is None:
        trans = get_transform(survey)

    if cov:
        nev = np.einsum(
            '...ik,jk...',
            np.einsum(
                '...ki,...jk', trans, hla.T
            ),
            trans.T
        ).T

    else:
        nev = np.einsum(
            'k...,jk...', hla.T, trans.T
        )

    return np.swapaxes(nev, 0, 1)


def get_sigmas(cov: NDArray, long: bool = False):
    """
    Extracts the sigma values of a covariance matrix along the principle axii.

    Parameters
    ----------
    cov : ndarray
        An (n, 3 ,3) array of covariance matrices.

    Returns
    -------
    sigmas : ndarray
        An (n, 3) array of sigma values.
    """

    assert cov.shape[-2:] == (3, 3), "Cov is the wrong shape"

    cov = cov.reshape(-1, 3, 3)

    aa, ab, ac = cov[:, :, 0].T
    ab, bb, bc = cov[:, :, 1].T
    ac, bc, cc = cov[:, :, 2].T

    if long:
        sigmas = (aa, bb, cc, ab, ac, bc)
    else:
        sigmas = (np.sqrt(aa), np.sqrt(bb), np.sqrt(cc))

    return sigmas


def get_unit_vec(vec: ArrayLike) -> NDArray:
    """
    Returns the unit vector of a vector.

    Parameters
    ----------
    vec : array_like
        A vector or an array of vectors.

    Returns
    -------
    unit_vector : ndarray
        A vector or array of 3D vectors.
    """
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


def make_cov(
        a: Union[float, NDArray],
        b: Union[float, NDArray],
        c: Union[float, NDArray],
        long: bool = False
    ) -> NDArray:
    """
    Generates a covariance matrix from orthogonal error magnitudes (sigmas) in
    the HLA or NEV coordinate systems.

    Parameters
    ----------
    a : float | ndarray
        The 1D array of (1-sigma) error in the H or N direction.
    b : float | ndarray
        The 1D array of (1-sigma) error in the L or E direction.
    c : float | ndarray
        The 1D array of (1-sigma) error in the A or V direction.
    long : bool, optional
        Constructs a covariance matrix of only the lead diagonal for the
        default ``False``, otherwise a full matrix is calculated.

    Returns
    -------
    cov : ndarray
        An [n, 3, 3] array of covariance matrices.

    See Also
    --------
    welleng.utils.get_sigmas
    """
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


def dls_from_radius(radius: Union[float, ArrayLike]) -> Union[float, NDArray]:
    """
    Returns the dls in degrees from a radius.

    Parameters
    ----------
    radius : float | array_like
        A float or list or array of the radius of curvature in meters.

    Returns
    -------
    dls : float | ndarray
        The equivalent Dog Leg Severity (DLS) in degrees / 30m.

    See Also
    --------
    welleng.utils.radius_from_dls
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


def radius_from_dls(dls: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Returns the radius in meters from a DLS in degrees / 30m.

    Parameters
    ----------
    dls : float | ndarray
        A float or array of Dog Leg Severity (DLS) in degrees / 30m.

    Returns
    -------
    radius : float | ndarray
        A float or array of radius of curvature in meters.

    See Also
    --------
    welleng.utils.dls_from_radius
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


def errors_from_cov(cov: NDArray, data: bool = False) -> Union[dict, NDArray]:
    """
    Extracts one half of a covariance matrix (along the lead diagonal). Since
    a covariance matrix is reflected along the lead diagonal, it can be
    more efficient to only save half the values.

    Parameters
    ----------
    cov : ndarray
        The (n, 3, 3) array of error covariance matrices.
    data : bool, optional
        If ``True`` returns a dictionary, else returns an array (default).

    Returns
    -------
    data : dict | ndarray
        A dict or (n, 6) ndarray of covariance matrix values.
    """
    nn, ne, nv, _, ee, ev, _, _, vv = (
        cov.reshape(-1, 9).T
    )

    if data:
        return {
            i: {
                'nn': _nn, 'ne': _ne, 'nv': _nv, 'ee': _ee, 'ev': _ev,
                'vv': _vv
            }
            for i, (_nn, _ne, _nv, _ee, _ev, _vv)
            in enumerate(zip(nn, ne, nv, ee, ev, vv))
        }

    return np.array([nn, ne, nv, ee, ev, vv]).T


class Arc:
    def __init__(self, dogleg: float, radius: float):
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
        pos: (3) array
        The desired position to transform the arc.
        vec: (3) array
            The orientation unit vector to transform the arc.
        target: bool, optional
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
    od : float
        The outer diameter.
    id : float | None, optional
        The inner diameter, default is 0.
    length : float | None, optional
        The length of the annulus, the default is 1 (i.e. unit volume or area)

    Returns
    -------
    annular_volume : float
        The (unit) volume of the annulus or cylinder.
    """
    length = 1 if length is None else length
    id = 0 if id is None else id
    annular_unit_volume = np.pi * ((od - id)**2) / 4
    annular_volume = annular_unit_volume * length

    return annular_volume
