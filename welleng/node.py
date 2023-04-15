import numpy as np

from .utils import get_angles, get_nev, get_unit_vec, get_vec, get_xyz


class Node:
    """
    Represents a single point or station on a well path.

    Parameters
    ----------
    pos : array_like
        1D list or array of coordinates representing the position of the point.
        e.g. [x, y, z] or [n, e, v].
    vec : array_like
        1D list or array of coordinates representing the unit vector at the
        point, e.g. [x, y, z] or [n, e, v].
    md : float
        The measured depth in meters at the point.
    inc: float
        The inclination in degrees or radians at the point.
    azi : float
        The azimuth referenced to grid north in degrees or radians at the
        point.
    degrees : bool
        Indicates whether angles (``inc``, ``azi``) are input as default
        degrees ``degrees=True`` or radians ``degrees=False``.
    nev: bool
        Indicates whether ``pos`` and ``vec`` are input the ``[x, y, z]`` or
        ``[n, e, v]`` coordinate system.
    cov_nev : array_like, optional
        The covariance matrix for the point can optionally be provided in the
        ``[n, e, v]`` coordinate system.

    Note
    ----
    A ``Node`` requires direction data which can be provided as either an
    ``inc`` and ``azi`` or a ``vec``, but not both.

    Example
    -------
    """
    def __init__(
        self,
        pos=None,
        vec=None,
        md=None,
        inc=None,
        azi=None,
        degrees=None,
        nev=None,
        cov_nev=None,
        **kwargs
    ):
        degrees = True if degrees is None else degrees
        nev = True if nev is None else nev

        self._check_angle_inputs(inc, azi, vec, nev, degrees)
        self._get_pos(pos, nev)
        self.md = 0 if md is None else md
        self.cov_nev = (
            np.zeros(shape=(1, 3, 3)) if cov_nev is None
            else cov_nev
        )

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _check_angle_inputs(self, inc, azi, vec, nev, degrees):
        if all(v is None for v in [inc, azi, vec]):
            self.vec_xyz = None
            self.vec_nev = None
            self.inc_rad = None
            self.inc_deg = None
            self.azi_rad = None
            self.azi_rad = None
            return
        elif vec is None:
            assert inc is not None and azi is not None, (
                "vec or (inc, azi) must not be None"
            )
            self.vec_xyz = get_vec(inc, azi, deg=degrees).reshape(3).tolist()
            self.vec_nev = get_nev(self.vec_xyz).reshape(3).tolist()
        else:
            if nev:
                self.vec_nev = get_unit_vec(vec).reshape(3).tolist()
                self.vec_xyz = get_xyz(self.vec_nev).reshape(3).tolist()
            else:
                self.vec_xyz = get_unit_vec(vec).reshape(3).tolist()
                self.vec_nev = get_nev(self.vec_xyz).reshape(3).tolist()
        self.inc_rad, self.azi_rad = (
            get_angles(self.vec_xyz).T
        ).reshape(2)
        self.inc_deg, self.azi_deg = (
            np.degrees(np.array([self.inc_rad, self.azi_rad]))
        ).reshape(2)

    def _get_pos(self, pos, nev):
        if pos is None:
            self.pos_xyz = None
            self.pos_nev = None
            return
        if nev:
            self.pos_nev = np.array(pos).reshape(3).tolist()
            self.pos_xyz = get_xyz(np.array(pos)).reshape(3).tolist()
        else:
            self.pos_xyz = np.array(pos).reshape(3).tolist()
            self.pos_nev = get_nev(np.array(pos)).reshape(3).tolist()

    def properties(self):
        return vars(self)


def get_node_params(node):
    pos = node.pos_nev
    vec = node.vec_nev
    md = node.md

    return pos, vec, md
