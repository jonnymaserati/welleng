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

from dataclasses import replace
from typing import Optional

import numpy as np

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


#: Default burst design factor -- the published minimum internal yield pressure
#: is reduced to this fraction to get the allowable. 0.80 is the conventional
#: setting and is what Volve's own kill sheets carry
#: (``WP_KILL_SHEET_GEN.casing_burst_safety_factor = 80.0``, on all 362 records).
BURST_DESIGN_FACTOR = 0.80


def cased_section(
    top_tvd: float,
    bottom_tvd: float,
    *,
    casing_od_in: float,
    casing_weight_ppf: float,
    pipe_od_in: float,
    grade: Optional[str] = None,
    burst_design_factor: float = BURST_DESIGN_FACTOR,
) -> WellSection:
    """A cased WellSection with true annular capacity (casing ID from the
    API-5CT catalogue, minus the string OD). ``is_open_hole=False`` -- its
    formation is protected and is not exposed to fracture.

    When a ``grade`` is given the catalogue also yields the minimum internal
    yield pressure (API TR 5C3 Eq. 10, Barlow, already derated by minimum wall),
    and the section carries ``burst_design_factor`` times that as its allowable
    internal pressure. Without a grade there is no rating and the section is
    left unchecked -- the catalogue flags an absent grade rather than guessing,
    and so does this.

    The burst check is INDICATIVE, not a casing design: no external backup is
    credited, and no axial, bending, temperature, wear or connection effects.
    See :class:`~welleng.kick_tolerance.migration.WellSection`.
    """
    try:
        from welleng.catalog import resolve
    except ImportError as exc:  # pragma: no cover - catalog is a core subpackage
        raise ImportError(
            "cased_section needs welleng.catalog (the API-5CT tubular catalogue)."
        ) from exc
    spec = resolve(casing_od_in, casing_weight_ppf, grade, kind="casing")
    rating = spec.internal_yield_pressure_psi
    return WellSection(
        top_tvd=top_tvd,
        bottom_tvd=bottom_tvd,
        annular_capacity_bbl_per_ft=annular_capacity(spec.id_in, pipe_od_in),
        is_open_hole=False,
        burst_pressure_psi=(
            None if rating is None else float(rating) * burst_design_factor
        ),
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


# ============================================================================
# Survey-coupled construction
# ============================================================================
# welleng's Survey and architecture.String are canonical SI (metres); the
# kick-tolerance engines are oilfield (feet). Conversion happens here, at the
# boundary, and nowhere else.
_M_TO_FT = 1.0 / 0.3048


def _tvds_ft(survey, mds_m) -> np.ndarray:
    """TVD [ft] at each along-hole depth in ``mds_m`` [m], in one pass.

    Uses the survey's batch interpolation rather than a per-depth call: the
    breakpoint set is known up front, so there is no reason to walk the
    minimum-curvature interpolation once per cut.
    """
    mds_m = np.asarray(mds_m, dtype=float)
    interp = survey.interpolate_mds(mds_m)
    interp_md = np.asarray(interp.md, dtype=float)
    interp_tvd = np.asarray(interp.pos_nev, dtype=float)[:, 2] * _M_TO_FT

    # interpolate_mds may return the union of the requested depths and the
    # original stations, so locate each requested depth rather than assuming
    # positional correspondence.
    idx = np.searchsorted(interp_md, mds_m)
    idx = np.clip(idx, 0, interp_md.size - 1)
    if not np.allclose(interp_md[idx], mds_m, atol=1e-6):
        missing = mds_m[~np.isclose(interp_md[idx], mds_m, atol=1e-6)]
        raise ValueError(f"the survey did not return depths {missing.tolist()} m")

    return interp_tvd[idx]


def _od_at(string, md_m: float) -> float:
    """String OD [in] at ``md_m``; 0.0 where no string is modelled (above the
    string top or below the bit) -- i.e. a full-bore annulus there."""
    if string is None:
        return 0.0
    if not (string.top <= md_m <= string.bottom):
        return 0.0
    return float(string.at(md_m)['od'])


def sections_from_architecture(
    wellbore,
    string,
    survey,
    *,
    shoe_md: float,
    top_md: Optional[float] = None,
    bottom_md: Optional[float] = None,
) -> list:
    """Build the elementary :class:`WellSection` list from an MD-domain well
    architecture and a survey.

    Annular volume lives in the MD domain (capacity is bbl per foot of
    along-hole length) and pressure lives in the TVD domain. The two are
    coupled by the survey. This builder cuts the well at the UNION of every
    depth at which either domain changes character:

      * every hole-geometry change (``wellbore`` section tops/bottoms),
      * every string-geometry change (``string`` section tops/bottoms -- BHA
        component boundaries, the bit),
      * every survey station,
      * the casing shoe.

    Each resulting piece therefore has a CONSTANT annular capacity and lies
    inside a SINGLE minimum-curvature leg, so its along-hole and vertical
    extents are both exact and their ratio is the piece's mean ``sec(inc)``.
    No fixed-step march and no chunk size: an interface that lands inside a
    piece is placed exactly with :func:`split_at_tvd`.

    Parameters
    ----------
    wellbore : welleng.architecture.WellBore
        Hole geometry, MD [m]; each section carries ``'id'`` [in].
    string : welleng.architecture.BHA or welleng.architecture.String or None
        Drill/work string, MD [m]; each section carries ``'od'`` [in]. Where
        the string is absent the annulus is taken as full bore.
    survey : welleng.survey.Survey
        The trajectory, supplying MD [m] -> TVD [m] and the station set.
    shoe_md : float
        Casing-shoe MD [m]. Pieces below it are ``is_open_hole=True`` and are
        checked against the pore/fracture envelope; pieces above it are cased
        and protected.
    top_md, bottom_md : float, optional
        Restrict the build to this along-hole interval [m]. Defaults to the
        wellbore's own extent.

    Returns
    -------
    list of WellSection
        Ordered shallow to deep, each carrying both its MD and TVD extent.

    Raises
    ------
    ValueError
        If a piece has a non-increasing TVD extent -- a TVD reversal (a well
        that drops then builds back up). The TVD-domain engines are not
        formulated for a depth that is reached twice.
    """
    top = wellbore.top if top_md is None else top_md
    bottom = wellbore.bottom if bottom_md is None else bottom_md

    cuts = set(wellbore.breakpoints())
    if string is not None:
        cuts.update(string.breakpoints())
    cuts.update(float(md) for md in survey.md)
    # Cut where the well passes through horizontal, so no piece spans a TVD
    # turning point and every piece is single-valued in TVD.
    cuts.update(float(md) for md in survey.tvd_turning_points())
    cuts.add(float(shoe_md))
    cuts.update((top, bottom))

    mds = sorted(md for md in cuts if top <= md <= bottom)

    tvds = _tvds_ft(survey, mds)

    sections = []
    for i, (a, b) in enumerate(zip(mds, mds[1:])):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        cap = annular_capacity(float(wellbore.at(mid)['id']), _od_at(string, mid))
        tvd_a, tvd_b = float(tvds[i]), float(tvds[i + 1])
        if tvd_b <= tvd_a:
            # Turning points are already cut, so this is a piece that is
            # horizontal (no TVD extent) or heading back up -- neither can be
            # expressed as a capacity per foot of TVD.
            what = "horizontal" if abs(tvd_b - tvd_a) < 1e-9 else "upward-turning"
            raise ValueError(
                f"the interval md {a}-{b} m is {what} "
                f"(TVD {tvd_a:.2f}-{tvd_b:.2f} ft): the TVD-domain kick engines "
                "hold volume against TVD and cannot represent it."
            )
        sections.append(WellSection(
            top_tvd=tvd_a,
            bottom_tvd=tvd_b,
            annular_capacity_bbl_per_ft=cap,
            is_open_hole=mid >= shoe_md,
            top_md=a * _M_TO_FT,
            bottom_md=b * _M_TO_FT,
        ))

    return sections


def split_at_tvd(section: WellSection, tvd: float, survey) -> tuple:
    """Split ``section`` at ``tvd`` [ft], placing the cut EXACTLY.

    The along-hole depth of the cut comes from the survey's closed-form TVD
    interpolation (Sawaryn & Thorogood 2005, SPE-84246-PA), so both halves
    carry exact MD and TVD extents and neither is a linearisation of the
    parent. This is what removes the need for a fine march: an interface is
    not stepped up to, it is solved for and the piece it lands in is split.

    Parameters
    ----------
    section : WellSection
        The piece the interface lands in. Must carry an MD extent.
    tvd : float
        The true vertical depth of the interface [ft], strictly inside the
        section.
    survey : welleng.survey.Survey
        The trajectory the section was built against.

    Returns
    -------
    tuple of (WellSection, WellSection)
        The upper and lower halves.

    Raises
    ------
    ValueError
        If ``tvd`` is not inside the section, if the section carries no MD
        extent, or if the survey crosses ``tvd`` more than once within it (a
        TVD reversal inside a single piece).
    """
    if section.top_md is None or section.bottom_md is None:
        raise ValueError(
            "split_at_tvd needs a section with an MD extent -- build it with "
            "sections_from_architecture."
        )
    if not (section.top_tvd < tvd < section.bottom_tvd):
        raise ValueError(
            f"tvd {tvd} is not inside the section "
            f"({section.top_tvd}-{section.bottom_tvd} ft)"
        )

    top_md_m, bottom_md_m = section.top_md / _M_TO_FT, section.bottom_md / _M_TO_FT
    inside = [
        float(node.md) for node in survey.interpolate_tvd(tvd / _M_TO_FT)
        if top_md_m - 1e-9 <= float(node.md) <= bottom_md_m + 1e-9
    ]
    if not inside:
        raise ValueError(f"the survey has no crossing of tvd {tvd} ft in this section")
    if len(inside) > 1:
        raise ValueError(
            f"the survey crosses tvd {tvd} ft {len(inside)} times within one "
            "section -- a TVD reversal inside a single piece"
        )

    cut_md = inside[0] * _M_TO_FT
    upper = replace(section, bottom_tvd=tvd, bottom_md=cut_md)
    lower = replace(section, top_tvd=tvd, top_md=cut_md)
    return upper, lower
