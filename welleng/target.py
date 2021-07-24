import numpy as np
try:
    from vedo import Circle
    VEDO = True
except ImportError:
    VEDO = False

class Target:
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
        """
        Parameters
        ----------
            geometry: 
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
