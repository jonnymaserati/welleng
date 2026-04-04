"""Wellbore survey node representing a position and direction in a well trajectory."""

import numpy as np
from .utils import (
    get_unit_vec, get_vec, get_angles, get_nev, get_xyz,
)
# from .connector import Connector


class Node:
    """A survey station in a wellbore trajectory.

    Stores position, direction vector, measured depth, and covariance
    for a single point along a well path. Coordinates can be specified
    in either NEV (north-east-vertical) or XYZ convention.

    Attributes
    ----------
    pos_nev : list
        Position as [north, east, vertical].
    pos_xyz : list
        Position as [x, y, z].
    vec_nev : list
        Unit direction vector in NEV.
    vec_xyz : list
        Unit direction vector in XYZ.
    inc_rad : float
        Inclination in radians.
    inc_deg : float
        Inclination in degrees.
    azi_rad : float
        Azimuth in radians.
    azi_deg : float
        Azimuth in degrees.
    md : float
        Measured depth along the wellbore.
    unit : str
        Unit of measurement (default 'meters').
    cov_nev : ndarray
        3x3 covariance matrix in NEV coordinates.
    """

    def __init__(
        self,
        pos=None,
        vec=None,
        md=None,
        inc=None,
        azi=None,
        unit='meters',
        degrees=True,
        nev=True,
        cov_nev=None,
        **kwargs
    ):
        """Initialize a Node with position and direction.

        Parameters
        ----------
        pos : array_like, optional
            Position as a 3-element array. Interpreted as NEV or XYZ
            depending on the ``nev`` flag.
        vec : array_like, optional
            Unit direction vector (3-element). If provided, ``inc``
            and ``azi`` are ignored.
        md : float, optional
            Measured depth along the wellbore.
        inc : float, optional
            Inclination angle.
        azi : float, optional
            Azimuth angle.
        unit : str
            Length unit, default ``'meters'``.
        degrees : bool
            If True, ``inc`` and ``azi`` are in degrees.
        nev : bool
            If True, ``pos`` and ``vec`` are in NEV coordinates;
            otherwise XYZ.
        cov_nev : ndarray, optional
            Covariance matrix (1, 3, 3). Defaults to zeros.
        **kwargs
            Additional attributes set on the instance.
        """
        self.check_angle_inputs(inc, azi, vec, nev, degrees)
        self.get_pos(pos, nev)
        self.md = md
        self.unit = unit
        self.cov_nev = np.zeros(shape=(1, 3, 3)) if cov_nev is None else cov_nev

        for k, v in kwargs.items():
            setattr(self, k, v)

    def check_angle_inputs(self, inc, azi, vec, nev, degrees):
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

    def get_pos(self, pos, nev):
        if pos is None:
            self.pos_xyz = None
            self.pos_nev = None
            return
        if nev:
            self.pos_nev = np.array(pos).reshape(3).tolist()
            self.pos_xyz = get_xyz(pos).reshape(3).tolist()
        else:
            self.pos_xyz = np.array(pos).reshape(3).tolist()
            self.pos_nev = get_nev(pos).reshape(3).tolist()

    def properties(self):
        """Return all instance attributes as a dictionary.

        Returns
        -------
        dict
            Mapping of attribute names to their values.
        """
        return vars(self)


def get_node_params(node):
    """Extract position, direction, and measured depth from a Node.

    Parameters
    ----------
    node : Node
        A Node instance.

    Returns
    -------
    tuple
        A tuple of (pos_nev, vec_nev, md).
    """
    pos = node.pos_nev
    vec = node.vec_nev
    md = node.md

    return pos, vec, md
