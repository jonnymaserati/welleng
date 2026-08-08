"""Gridded surfaces (e.g. seismic horizons) — a light geometry primitive.

A :class:`Surface` is a 2.5-D structural surface: a single value ``Z`` (depth or
two-way time) over a regular ``(inline, crossline)`` grid, with the map
projection ``(X, Y)`` of every node. It answers the query a trajectory planner
needs — *what is the surface depth at an arbitrary map location* — by bilinear
interpolation, and supports above/below/within tests against one or a pair of
surfaces (a formation interval = top + base).

This is the open, scalar reference primitive; consumers (e.g. the trajectory
planner) use it for target surfaces and geosteering corridors. Populate one from
an exchange reader such as :func:`welleng.exchange.openworks.read_ow_horizon`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["Surface"]


@dataclass
class Surface:
    """A gridded 2.5-D surface on a regular ``(inline, crossline)`` grid.

    Build via :meth:`from_nodes` (scattered node records) rather than the
    constructor directly.

    Parameters
    ----------
    z : ndarray, shape (n_il, n_xl)
        Surface value at each grid node (NaN where a node is absent).
    il, xl : ndarray
        Sorted, regularly-spaced inline / crossline coordinate vectors.
    affine : ndarray, shape (2, 3)
        Maps ``[il, xl, 1] -> [X, Y]`` (the 3-D survey geometry).
    name : str
        Horizon / surface name.
    domain : str
        ``'DEPTH'`` or ``'TWT'``.
    crs : str, optional
        Source cartographic system, if known.
    """

    z: np.ndarray
    il: np.ndarray
    xl: np.ndarray
    affine: np.ndarray
    name: str = ""
    domain: str = "DEPTH"
    crs: Optional[str] = None
    _inv: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        # inverse of the linear part, to map (X, Y) -> (il, xl)
        self._inv = np.linalg.inv(self.affine[:, :2])

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_nodes(cls, il, xl, x, y, z, *, name="", domain="DEPTH", crs=None):
        """Build a :class:`Surface` from scattered node records.

        Parameters
        ----------
        il, xl, x, y, z : array_like
            Per-node inline, crossline, easting, northing and surface value.
        """
        il = np.asarray(il, float)
        xl = np.asarray(xl, float)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.asarray(z, float)

        il_vals = np.unique(il)
        xl_vals = np.unique(xl)
        n_il, n_xl = il_vals.size, xl_vals.size
        if n_il < 2 or n_xl < 2:
            raise ValueError("a surface needs at least a 2x2 (inline, crossline) grid")

        grid = np.full((n_il, n_xl), np.nan)
        i_idx = np.searchsorted(il_vals, il)
        x_idx = np.searchsorted(xl_vals, xl)
        grid[i_idx, x_idx] = z

        # affine [il, xl, 1] -> [X, Y] (regular survey -> exact least-squares fit)
        A = np.column_stack([il, xl, np.ones_like(il)])
        coef, *_ = np.linalg.lstsq(A, np.column_stack([x, y]), rcond=None)
        affine = coef.T  # (2, 3)

        return cls(z=grid, il=il_vals, xl=xl_vals, affine=affine,
                   name=name, domain=domain, crs=crs)

    # -- queries --------------------------------------------------------------
    def _il_xl(self, x, y):
        """Map map coordinates (X, Y) -> fractional (inline, crossline)."""
        rhs = np.stack([np.asarray(x, float) - self.affine[0, 2],
                        np.asarray(y, float) - self.affine[1, 2]], axis=-1)
        ic = rhs @ self._inv.T          # (..., 2)
        return ic[..., 0], ic[..., 1]

    def z_at(self, x, y):
        """Bilinearly interpolated surface value at map location(s) (X, Y).

        Returns NaN outside the grid or where any surrounding node is absent.
        Accepts scalars or arrays (one surface, many query points).
        """
        il_q, xl_q = self._il_xl(x, y)
        scalar = np.ndim(il_q) == 0
        il_q = np.atleast_1d(il_q)
        xl_q = np.atleast_1d(xl_q)

        di = self.il[1] - self.il[0]
        dx = self.xl[1] - self.xl[0]
        fi = (il_q - self.il[0]) / di
        fx = (xl_q - self.xl[0]) / dx

        n_il, n_xl = self.z.shape
        out = np.full(il_q.shape, np.nan)
        eps = 1e-6  # tolerate fp error from the (X,Y)->(il,xl) inversion at edges
        inb = ((fi >= -eps) & (fi <= n_il - 1 + eps)
               & (fx >= -eps) & (fx <= n_xl - 1 + eps))
        if inb.any():
            fii = np.clip(fi[inb], 0.0, n_il - 1)
            fxi = np.clip(fx[inb], 0.0, n_xl - 1)
            # clamp the cell origin so the +1 corners always exist; a query
            # exactly on the far edge then lands with t == 1 on that node.
            i0 = np.clip(np.floor(fii).astype(int), 0, n_il - 2)
            j0 = np.clip(np.floor(fxi).astype(int), 0, n_xl - 2)
            ti = fii - i0
            tj = fxi - j0
            z00 = self.z[i0, j0]
            z10 = self.z[i0 + 1, j0]
            z01 = self.z[i0, j0 + 1]
            z11 = self.z[i0 + 1, j0 + 1]
            # nan-safe: a zero-weight corner never poisons the sum (nan*0 -> 0);
            # a corner with weight > 0 that is absent correctly yields nan.
            def _c(w, zc):
                return np.where(w > 0, w * zc, 0.0)
            out[inb] = (_c((1 - ti) * (1 - tj), z00) + _c(ti * (1 - tj), z10)
                        + _c((1 - ti) * tj, z01) + _c(ti * tj, z11))
        return float(out[0]) if scalar else out

    def is_below(self, x, y, tvd):
        """True where the point (X, Y, tvd) lies below (deeper than) the surface.

        NaN surface (off-grid / absent) -> False. Depth increases downward.
        """
        z = np.asarray(self.z_at(x, y), float)
        return np.asarray(tvd, float) > z

    def within(self, base: "Surface", x, y, tvd):
        """True where (X, Y, tvd) is between this surface (top) and ``base``.

        A formation interval: ``top.z_at <= tvd <= base.z_at``.
        """
        zt = np.asarray(self.z_at(x, y), float)
        zb = np.asarray(base.z_at(x, y), float)
        t = np.asarray(tvd, float)
        return (t >= zt) & (t <= zb)
