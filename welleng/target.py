"""Drilling target definitions for wellbore trajectory planning and visualization.

A :class:`Target` is a region in 3D space that a trajectory aims to reach. It
carries a position, a shape, shape geometry, and an orientation (the plane the
shape lies in). Shapes range from a bare point through 1D (line), 2D (circle,
ellipse, rectangle, polygon) to a 3D ``volume``, and a probabilistic
``gaussian`` target defined by a covariance about its mean rather than a hard
boundary.

Construction is geometry-only and requires no optional dependencies; only
:meth:`Target.plot_data` (rendering) needs ``vedo``.
"""
import numpy as np

try:
    from vedo import Circle
    VEDO = True
except ImportError:
    VEDO = False

# Recognised target shapes, ordered by the positional degrees of freedom a
# landing point has within the shape (used by the trajectory solver).
SHAPES = (
    'point',      # 0 DOF
    'line',       # 1 DOF (along the line/segment)
    'circle',     # 2 DOF (planar, on the target frame, |r| <= radius)
    'ellipse',    # 2 DOF (planar)
    'rectangle',  # 2 DOF (planar)
    'polygon',    # 2 DOF (planar)
    'volume',     # 3 DOF (extruded region: 2D shape + thickness)
    'cube',       # 3 DOF (axis-aligned box in NEV: geometry half_extents)
    'sphere',     # 3 DOF (ball in NEV: geometry radius)
    'gaussian',   # 3 DOF (Mahalanobis ellipsoid: mean + covariance cov)
)

# Positional DOF a landing point has within each shape.
_SHAPE_DIM = {
    'point': 0, 'line': 1,
    'circle': 2, 'ellipse': 2, 'rectangle': 2, 'polygon': 2,
    'volume': 3, 'cube': 3, 'sphere': 3, 'gaussian': 3,
}


class Target:
    """A geometric or probabilistic target zone in 3D space.

    Represents a target region at a given subsurface location, on a plane set by
    ``orientation``/``dip``/``azimuth``, with a shape-specific geometry. A
    ``gaussian`` target instead carries a covariance ``cov`` about its mean
    (the position), for probabilistic (uncertainty-aware) targeting.

    The object is a pure data/geometry primitive — it constructs without any
    optional dependency. :meth:`plot_data` (vedo rendering) is the only method
    that requires ``vedo``.

    Attributes
    ----------
    name : str
        Identifier for the target.
    n, e, tvd : float
        Position (northing, easting, true vertical depth), metres. Also exposed
        as :attr:`position` and :attr:`location` (a 3-vector).
    shape : str or None
        One of :data:`SHAPES`, or ``None`` if unspecified.
    locked : int
        Lock state of the target.
    orientation : float
        Rotation about the vertical axis, degrees.
    dip : float
        Dip of the target plane, degrees.
    azimuth : float
        Azimuth of the target plane, degrees.
    cov : ndarray of shape (3, 3) or None
        NEV covariance for a ``gaussian`` target (mean = position); ``None`` for
        hard-boundary shapes.
    color : str
        Display colour for rendering.
    alpha : float
        Opacity for rendering (0.0 to 1.0).
    geometry : dict
        Shape-specific dimensional parameters (e.g. ``radius`` for a circle;
        ``radius_1``/``radius_2`` for an ellipse; ``pos1``/``pos2`` for a
        rectangle; ``vertices`` for a polygon; ``thickness_up``/
        ``thickness_down`` to extrude a 2D shape into a ``volume``).
    """

    def __init__(
        self,
        name,
        n=None,
        e=None,
        tvd=None,
        shape=None,
        *,
        location=None,
        geometry=None,
        locked=0,
        orientation=0.0,
        dip=0.0,
        azimuth=0.0,
        cov=None,
        color='green',
        alpha=0.5,
        **geometry_kwargs,
    ):
        """Initialize a Target.

        Parameters
        ----------
        name : str
            Identifier for the target.
        n, e, tvd : float, optional
            Position (northing, easting, TVD) in metres. Alternatively pass
            ``location=[n, e, tvd]``.
        shape : str, optional
            One of :data:`SHAPES`. May be left ``None`` and set later.
        location : sequence of float, optional
            Position as ``[n, e, tvd]``; an alternative to ``n``/``e``/``tvd``.
        geometry : dict, optional
            Shape-specific parameters as a dict. Alternatively pass them as
            keyword arguments (e.g. ``radius=30``), which are collected into
            :attr:`geometry`.
        locked : int, optional
            Lock state (0 = unlocked).
        orientation, dip, azimuth : float, optional
            Target-plane orientation angles, degrees.
        cov : array_like of shape (3, 3), optional
            NEV covariance for a ``gaussian`` target (mean = position).
        color : str, optional
            Display colour.
        alpha : float, optional
            Opacity (0.0 to 1.0).
        **geometry_kwargs
            Shape-specific parameters collected into :attr:`geometry` when
            ``geometry`` is not given explicitly. For ``circle``: ``radius``.
            For ``ellipse``: ``radius_1``, ``radius_2``, optional ``res``. For
            ``rectangle``: ``pos1``, ``pos2``.

        Raises
        ------
        ValueError
            If ``shape`` is given but not one of :data:`SHAPES`.
        """
        if location is not None:
            location = np.asarray(location, dtype=float).reshape(3)
            n, e, tvd = float(location[0]), float(location[1]), float(location[2])

        if shape is not None and shape not in SHAPES:
            raise ValueError(f"shape {shape!r} not in {SHAPES}")

        self.name = name
        self.n = n
        self.e = e
        self.tvd = tvd
        self.shape = shape
        self.locked = locked
        self.orientation = orientation
        self.dip = dip
        self.azimuth = azimuth
        self.cov = None if cov is None else np.asarray(cov, dtype=float).reshape(3, 3)
        self.color = color
        self.alpha = alpha
        # geometry precedence: an explicit dict wins; otherwise collect kwargs.
        self.geometry = (
            dict(geometry) if geometry is not None else dict(geometry_kwargs)
        )

    @property
    def position(self):
        """Position as an ``ndarray([n, e, tvd])``, or ``None`` if unset."""
        if self.n is None or self.e is None or self.tvd is None:
            return None
        return np.array([self.n, self.e, self.tvd], dtype=float)

    # ``location`` mirrors ``position`` for callers that used the WBP name.
    @property
    def location(self):
        """Alias of :attr:`position` (the ``[n, e, tvd]`` 3-vector)."""
        return self.position

    @location.setter
    def location(self, value):
        if value is None:
            self.n = self.e = self.tvd = None
            return
        value = np.asarray(value, dtype=float).reshape(3)
        self.n, self.e, self.tvd = float(value[0]), float(value[1]), float(value[2])

    @property
    def mean(self):
        """Mean of a ``gaussian`` target (its position); ``None`` otherwise."""
        return self.position if self.shape == 'gaussian' else None

    @property
    def dim(self):
        """Positional DOF a landing point has within the shape (see :data:`SHAPES`)."""
        return _SHAPE_DIM.get(self.shape)

    def plot_data(self):
        """Generate a vedo mesh object for rendering the target.

        Currently supports the ``circle`` shape. The target is positioned at
        (n, e, tvd) and rotated according to dip and orientation.

        Returns
        -------
        vedo object
            A vedo geometry object representing the target, with the target name
            assigned to its ``flag`` attribute.

        Raises
        ------
        AssertionError
            If ``vedo`` is not installed.
        """
        assert VEDO, "ImportError: try pip install welleng[easy]"

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
            g.pos = [self.n, self.e, 0]
            g.rotate(self.dip, point=pos, axis=(0, 1, 0))
            g.rotate(self.orientation, point=pos, axis=(1, 0, 0))

        return g
