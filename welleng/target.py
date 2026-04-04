"""Drilling target definitions for wellbore trajectory visualization."""

import numpy as np
try:
    from vedo import Circle
    VEDO = True
except ImportError:
    VEDO = False

class Target:
    """A geometric target zone in 3D space for wellbore trajectory planning.

    Represents a target area (circle, ellipse, rectangle, or polygon) at a
    given subsurface location, with optional orientation and dip. Requires
    vedo for visualization.

    Attributes
    ----------
    name : str
        Identifier for the target.
    n : float
        Northing coordinate.
    e : float
        Easting coordinate.
    tvd : float
        True vertical depth.
    shape : str
        Target geometry type.
    locked : int
        Lock state of the target.
    orientation : float
        Rotation angle about the vertical axis in degrees.
    dip : float
        Dip angle of the target plane in degrees.
    color : str
        Display color for rendering.
    alpha : float
        Opacity for rendering (0.0 to 1.0).
    geometry : dict
        Shape-specific dimensional parameters.
    """

    def __init__(
        self,
        name,
        n,
        e,
        tvd,
        shape,
        locked=0,
        orientation=0,
        dip=0,
        color='green',
        alpha=0.5,
        **geometry
    ):
        """Initialize a Target.

        Parameters
        ----------
        name : str
            Identifier for the target.
        n : float
            Northing coordinate (meters).
        e : float
            Easting coordinate (meters).
        tvd : float
            True vertical depth (meters).
        shape : str
            Target geometry type. One of 'circle', 'ellipse',
            'rectangle', or 'polygon'.
        locked : int, optional
            Lock state of the target (0 = unlocked).
        orientation : float, optional
            Rotation angle about the vertical axis in degrees.
        dip : float, optional
            Dip angle of the target plane in degrees.
        color : str, optional
            Display color for rendering.
        alpha : float, optional
            Opacity for rendering (0.0 to 1.0).
        **geometry : dict
            Shape-specific parameters. For 'circle': radius.
            For 'ellipse': radius_1, radius_2, res. For 'rectangle':
            pos1, pos2.

        Raises
        ------
        AssertionError
            If vedo is not installed, shape is invalid, or geometry keys
            do not match the expected keys for the shape.
        """
        assert VEDO, "ImportError: try pip install welleng[easy]"

        SHAPES = [
            'circle',
            'ellipse',
            'rectangle',
            'polygon',
        ]

        GEOMETRY = dict(
            rectangle = ['pos1', 'pos2'],
            circle = ['radius'],
            ellipse = {'radius_1': 0, 'radius_2': 0, 'res': 120},
        )

        assert shape in SHAPES, "shape not in SHAPES"
        assert set(geometry.keys()) == set(GEOMETRY[shape]), 'wrong geometry'

        self.name = name
        self.n = n
        self.e = e
        self.tvd = tvd
        self.shape = shape
        self.locked = locked
        self.orientation = orientation
        self.dip = dip
        self.color = color
        self.alpha = alpha
        self.geometry = geometry

    def plot_data(self):
        """Generate a vedo mesh object for rendering the target.

        Currently supports the 'circle' shape. The target is positioned at
        (n, e, tvd) and rotated according to dip and orientation.

        Returns
        -------
        vedo object
            A vedo geometry object representing the target, with the
            target name assigned to its ``flag`` attribute.
        """
        pos = [self.n, self.e, self.tvd]
        if self.shape == "circle":
            g = Circle(
                pos=pos,
                r=self.geometry['radius'],
                c=self.color,
                alpha=self.alpha,
                # res=self.geometry['res']
            )
            g.flag = self.name
            g.pos=[self.n, self.e, 0]
            g.rotate(self.dip, point=pos, axis=(0,1,0))
            g.rotate(self.orientation, point=pos, axis=(1,0,0))

        return g
