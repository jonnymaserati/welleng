"""Catalogue-backed well geometry for the gas-migration kick-tolerance engine.

Builds :class:`~welleng.kick_tolerance.migration.WellSection` list with the TRUE
ANNULAR capacity -- the annulus between the bore (casing ID or open-hole
diameter) and the drill/work string inside it -- rather than a full-bore
capacity. Casing IDs are resolved from the API-5CT catalogue
(:mod:`welleng.catalog`) from (OD, nominal weight[, grade]); the string OD is
subtracted. This fixes the over-capacity error you get from an EDM
``linear_capacity`` (which is full-bore): a full-bore capacity makes the gas
bubble shorter than reality and UNDER-states the pressure it imposes on a
shallow formation -- the wrong direction for a fracture barrier.

``welleng.catalog`` is imported lazily so the kick-tolerance package stays
importable without it; the builders raise a clear error if it is unavailable.
"""
from __future__ import annotations

from typing import Optional

from .migration import WellSection

# Oilfield annular-capacity constant: capacity [bbl/ft] = (ID^2 - OD^2) / 1029.4
# with diameters in inches.
_CAP_CONST = 1029.4


def annular_capacity(inner_id_in: float, pipe_od_in: float = 0.0) -> float:
    """Annulus capacity [bbl/ft] between a bore ``inner_id_in`` (casing ID or
    open-hole diameter) and the string ``pipe_od_in`` inside it::

        capacity = (inner_id^2 - pipe_od^2) / 1029.4

    ``pipe_od_in = 0`` gives the full-bore capacity (no string)."""
    if pipe_od_in >= inner_id_in:
        raise ValueError(
            f"pipe OD ({pipe_od_in}) must be smaller than the bore ID ({inner_id_in})"
        )
    return (inner_id_in ** 2 - pipe_od_in ** 2) / _CAP_CONST


def cased_section(
    top_tvd: float,
    bottom_tvd: float,
    *,
    casing_od_in: float,
    casing_weight_ppf: float,
    pipe_od_in: float,
    grade: Optional[str] = None,
) -> WellSection:
    """A cased WellSection with true annular capacity (casing ID from the
    API-5CT catalogue, minus the string OD). ``is_open_hole=False`` (protected;
    its formation is not exposed to fracture)."""
    try:
        from welleng.catalog import resolve
    except ImportError as exc:  # pragma: no cover - catalog is a core subpackage
        raise ImportError(
            "cased_section needs welleng.catalog (the API-5CT tubular catalogue)."
        ) from exc
    spec = resolve(casing_od_in, casing_weight_ppf, grade, kind="casing")
    return WellSection(
        top_tvd=top_tvd,
        bottom_tvd=bottom_tvd,
        annular_capacity_bbl_per_ft=annular_capacity(spec.id_in, pipe_od_in),
        is_open_hole=False,
    )


def open_hole_section(
    top_tvd: float,
    bottom_tvd: float,
    *,
    hole_size_in: float,
    pipe_od_in: float,
) -> WellSection:
    """An open-hole WellSection with true annular capacity (bit diameter minus
    the string OD). ``is_open_hole=True`` -- its formation is fracture-exposed."""
    return WellSection(
        top_tvd=top_tvd,
        bottom_tvd=bottom_tvd,
        annular_capacity_bbl_per_ft=annular_capacity(hole_size_in, pipe_od_in),
        is_open_hole=True,
    )
