import numpy as np

from .utils import get_angles, get_nev, get_unit_vec, get_vec, get_xyz

# from .connector import Connector


class Node:
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
        **kwargs
    ):
        self.check_angle_inputs(inc, azi, vec, nev, degrees)
        self.get_pos(pos, nev)
        self.md = md
        self.unit = unit
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

    @staticmethod
    def get_unit_vec(vec):
        vec = vec / np.linalg.norm(vec)

        return vec

    def properties(self):
        return vars(self)


def get_node_params(node):
    pos = node.pos_nev
    vec = node.vec_nev
    md = node.md

    return pos, vec, md
