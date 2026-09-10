"""OSDU-aligned lean input models for well schematics (pydantic).

These are deliberately *lean* Data Transfer Objects: just enough of the
OSDU / WITSML-2.0 well-construction vocabulary to drive a 2D schematic
drawing, no more. Field names map onto OSDU concepts (noted per class) so a
richer OSDU record can be projected down onto these without renaming.

The model is a **tree of wellbores** so that a future increment can render
multilaterals: a :class:`WellSchematic` owns a list of :class:`Wellbore`
branches, each carrying its own survey / casings / completion plus a
``parent_id`` + ``kickoff_md`` (OSDU ``Wellbore`` / ``KickOffWellbore``).
This increment is single-bore only -- exactly one branch -- but the shape
does not preclude adding laterals later. Multilateral *rendering* is a TODO
(see ``column.py`` / ``section.py``: both operate on ``schematic.primary``).

No renderer imports here -- this module is pure data + validation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Gravitational acceleration used by the EMW->hydrostatic-pressure helper.
_G = 9.81


def _osdu_check(list_name: str, code: Optional[str], field_name: str) -> None:
    """Validate an optional OSDU reference-data code on a schematic field.

    Strictness comes from the list's own governance model, so nothing is
    re-decided here: a FIXED list refuses an unknown code, an OPEN one warns
    and lets it through (operators may extend it), a LOCAL one is silent.
    ``None`` is always fine -- these fields annotate, they are not required.
    """
    if code is None:
        return
    from ..osdu_ref import validate           # lazy: keeps models import light
    validate(list_name, code, field=field_name)


class _Base(BaseModel):
    """Shared config: allow population by field name or OSDU alias."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Well(_Base):
    """Top-level well identity. OSDU ``master-data--Well``."""

    name: str
    uwi: Optional[str] = None  # OSDU FacilityId / unique well identifier
    # --- datum system: all elevations referenced to MSL = 0, which is what
    # enables inter-datum conversion ---
    # primary MD datum (RKB/RT/KB/DF/MSL/mudline); labels the axis, e.g. "MD RKB"
    depth_reference: str = "RKB"
    # RKB/rig-floor elevation above MSL (air gap included)
    datum_elevation_m: Optional[float] = None
    water_depth_m: Optional[float] = None  # MSL to seabed/mudline (offshore)
    ground_elevation_m: Optional[float] = None  # ground level above MSL (onshore)
    # NOTE: RKB<->MSL<->mudline<->TVDSS<->ground conversion is handled in
    # welleng.schematic.depth. Every entered depth defaults to
    # `depth_reference`; per-value datum overrides are still to be added, so
    # mixed-datum inputs (e.g. TVDSS formation tops) convert correctly.


class SurveyRef(_Base):
    """Directional survey stations. OSDU ``WellboreTrajectory`` (md/inc/azi).

    Arrays are kept parallel (equal length); minimum-curvature MD<->TVD and
    N/E are resolved downstream by :mod:`welleng.schematic.depth`.
    """

    md: List[float]
    inc: List[float]
    azi: List[float]

    @model_validator(mode="after")
    def _equal_length(self) -> "SurveyRef":
        if not (len(self.md) == len(self.inc) == len(self.azi)):
            raise ValueError("md, inc and azi must have equal length")
        if len(self.md) < 2:
            raise ValueError("survey needs at least two stations")
        return self


class HoleSection(_Base):
    """A drilled open-hole interval. OSDU ``HoleSection`` (bit size + depths).

    ``radial_scale`` is the *horizontal* (radial) exaggeration applied to this
    section only -- shallow, tightly-nested sections need more exaggeration to
    read (e.g. x50) than sparse deep sections (e.g. x30). It must be
    **monotonic non-increasing with depth** (see :class:`Wellbore`). ``None``
    means "use the global fallback" (backward-compatible with the prototypes).
    """

    bit_in: float = Field(..., description="bit diameter, inches")
    top_md: float
    base_md: float
    radial_scale: Optional[float] = Field(
        None, description="per-section radial exaggeration (H-scale multiplier)"
    )


class TubularSection(_Base):
    """One diameter of a TAPERED / COMBINATION string.

    A combination string is ONE string on ONE hanger with ONE shoe -- e.g.
    10-3/4in to 191.6 m then 9-5/8in to the shoe at 1997.6 m. Modelling that as
    two :class:`Casing` objects makes the renderer draw a shoe at the
    crossover, which asserts the string ends there and the annulus opens below
    it. Both are false, and a fabricated shoe is a false statement about well
    architecture, where a missing diameter step is only a cosmetic loss.

    ``id_in`` is auto-filled from the API 5CT catalogue on the same terms as
    :class:`Tubular` when ``nominal_weight_ppf`` is given.
    """

    od_in: float = Field(..., description="outer diameter, inches")
    id_in: Optional[float] = Field(
        None, description="inner diameter, inches (auto-filled from catalogue)"
    )
    top_md: float
    base_md: float
    nominal_weight_ppf: Optional[float] = None
    grade: Optional[str] = None


class Tubular(_Base):
    """Base for run steel tubulars. OSDU ``Tubular`` / WITSML ``tubular``.

    ``id_in`` is optional: when it is omitted but ``od_in`` +
    ``nominal_weight_ppf`` are given, ``id_in`` and ``drift_in`` are filled
    from the API 5CT catalogue (:mod:`welleng.catalog`) by the subclass'
    ``kind`` (``Casing`` -> ``"casing"``, else ``"tubing"``). An explicit
    ``id_in`` always wins. OSDU aliases follow ``TubularComponent.1.0.0``.
    """

    name: str
    od_in: float = Field(..., description="outer diameter, inches")
    id_in: Optional[float] = Field(
        None, description="inner diameter, inches (auto-filled from catalogue)"
    )
    top_md: float = 0.0
    shoe_md: float = Field(..., description="setting/shoe depth, MD")
    # NOT 0.0-by-default, and not required: 0.0 means "cemented to surface",
    # which is a claim about a BARRIER. Defaulting to it made an unstated TOC
    # render as the MAXIMUM POSSIBLE cement -- silently, on exactly the strings
    # least likely to have any (a driven conductor, a screen, a tie-back on a
    # packer, junk left in hole). A schematic is read as a barrier drawing, so
    # the default has to fail toward "nothing recorded", never toward "sealed".
    toc_md: Optional[float] = Field(
        None,
        description="top of cement, MD (annulus). None = no cement recorded "
                    "(nothing is drawn); 0.0 = cemented to surface",
    )
    sections: List[TubularSection] = Field(
        default_factory=list,
        description="diameters of a tapered/combination string, top to shoe; "
                    "empty = a single-diameter string",
    )
    # --- catalogue key + auto-filled dimensions (OSDU TubularComponent) ---
    nominal_weight_ppf: Optional[float] = Field(
        None, description="nominal weight, lb/ft (OSDU TubularComponentNominalWeight)"
    )
    grade: Optional[str] = Field(
        None,
        description="material grade, e.g. L80 "
                    "(OSDU TubularComponentTubingGradeID)",
    )
    connection: Optional[str] = Field(
        None, description="thread/connection type, e.g. BTC/LTC/STC/NUE/EUE"
    )
    drift_in: Optional[float] = Field(
        None, description="drift diameter, inches (OSDU DriftDiameter; auto-filled)"
    )
    coupling_od_in: Optional[float] = Field(
        None, description="coupling OD (W), inches (auto-filled from API 5CT when "
        "connection is a known API type)"
    )
    coupling_length_in: Optional[float] = Field(
        None, description="minimum coupling length (NL), inches (auto-filled)"
    )

    def _catalog_kind(self) -> str:
        return "casing" if isinstance(self, Casing) else "tubing"

    @model_validator(mode="after")
    def _fill_from_catalogue(self) -> "Tubular":
        """Fill ``id_in``/``drift_in`` from the API 5CT catalogue.

        Only runs when ``id_in`` is missing and ``(od_in,
        nominal_weight_ppf)`` are both present. An explicit ``id_in`` is never
        overridden; ``drift_in`` is filled only when absent.
        """
        if self.id_in is None and self.nominal_weight_ppf is not None:
            from welleng.catalog import resolve  # lazy: keep import light

            spec = resolve(
                self.od_in, self.nominal_weight_ppf,
                grade=self.grade, kind=self._catalog_kind(),
            )
            # bypass validation re-entry; assign resolved dims in-place.
            object.__setattr__(self, "id_in", spec.id_in)
            if self.drift_in is None:
                object.__setattr__(self, "drift_in", spec.drift_in)
        if self.id_in is None:
            raise ValueError(
                "id_in is required unless (od_in, nominal_weight_ppf) resolve "
                "from the catalogue"
            )
        # optional coupling dims from API 5CT when a known API connection is
        # given (best-effort; a non-API/premium label or untabulated size just
        # leaves the coupling fields None - never breaks an existing schematic).
        if self.connection is not None and self.coupling_od_in is None:
            from welleng.catalog import CatalogError, resolve_coupling

            try:
                cpl = resolve_coupling(
                    self.od_in, self.connection, kind=self._catalog_kind()
                )
            except CatalogError:
                cpl = None
            if cpl is not None:
                object.__setattr__(self, "coupling_od_in", cpl.coupling_od_in)
                if self.coupling_length_in is None:
                    object.__setattr__(
                        self, "coupling_length_in", cpl.coupling_length_in
                    )
        return self

    @model_validator(mode="after")
    def _check_sections(self) -> "Tubular":
        """A combination string's sections must tile ``top_md``..``shoe_md``.

        Contiguity is checked, not assumed: a gap would draw the string as two
        pieces with open annulus between them, and an overlap would draw two
        walls at the same depth. Both are the fabricated-geometry class of
        defect this field exists to remove.
        """
        if not self.sections:
            return self
        secs = sorted(self.sections, key=lambda s: s.top_md)
        object.__setattr__(self, "sections", secs)
        for s in secs:
            if s.base_md <= s.top_md:
                raise ValueError(
                    f"{self.name}: section {s.od_in}in has base_md "
                    f"{s.base_md} at or above top_md {s.top_md}"
                )
            if s.id_in is None and s.nominal_weight_ppf is not None:
                from welleng.catalog import resolve

                spec = resolve(s.od_in, s.nominal_weight_ppf, grade=s.grade,
                               kind=self._catalog_kind())
                object.__setattr__(s, "id_in", spec.id_in)
            if s.id_in is None:
                raise ValueError(
                    f"{self.name}: section {s.od_in}in needs id_in unless "
                    "(od_in, nominal_weight_ppf) resolve from the catalogue"
                )
        for a, b in zip(secs, secs[1:]):
            if abs(a.base_md - b.top_md) > 1e-6:
                raise ValueError(
                    f"{self.name}: sections are not contiguous -- "
                    f"{a.od_in}in ends at {a.base_md} but {b.od_in}in starts "
                    f"at {b.top_md}. A combination string is one continuous "
                    "string; a gap or overlap draws geometry that is not there."
                )
        if abs(secs[0].top_md - self.top_md) > 1e-6:
            raise ValueError(
                f"{self.name}: sections start at {secs[0].top_md} but the "
                f"string top_md is {self.top_md}"
            )
        if abs(secs[-1].base_md - self.shoe_md) > 1e-6:
            raise ValueError(
                f"{self.name}: sections end at {secs[-1].base_md} but the "
                f"shoe_md is {self.shoe_md}"
            )
        # The deepest section is the one the shoe, any perforation naming this
        # string, and the annulus keying all resolve against, so od_in/id_in
        # must BE that section rather than quietly disagreeing with it.
        deep = secs[-1]
        if abs(deep.od_in - self.od_in) > 1e-6:
            raise ValueError(
                f"{self.name}: od_in is {self.od_in}in but the section at the "
                f"shoe is {deep.od_in}in. Set od_in/id_in to the shoe "
                "section -- that is the diameter the shoe and any perforation "
                "naming this string resolve against."
            )
        return self

    # -- depth-aware geometry --------------------------------------------- #
    def profile(self) -> List[Tuple[float, float, float, float]]:
        """``(top_md, base_md, od_in, id_in)`` per diameter, top to shoe."""
        if not self.sections:
            return [(self.top_md, self.shoe_md, self.od_in, float(self.id_in))]
        return [(s.top_md, s.base_md, s.od_in, float(s.id_in))
                for s in self.sections]

    def crossovers(self) -> List[float]:
        """Depths where the diameter changes (empty for a single diameter)."""
        return [s.base_md for s in self.sections[:-1]]

    def od_at(self, md: float) -> float:
        """OD (in) at ``md`` -- the shoe section's OD outside the string."""
        for top, base, od, _id in self.profile():
            if top <= md <= base:
                return od
        return self.od_in

    def id_at(self, md: float) -> float:
        """ID (in) at ``md`` -- the shoe section's ID outside the string."""
        for top, base, _od, id_ in self.profile():
            if top <= md <= base:
                return id_
        return float(self.id_in)

    def has_od(self, od_in: float, tol: float = 1e-6) -> bool:
        """True when ANY of the string's diameters is ``od_in``.

        A caller naming an annulus or a perforated string by OD is naming a
        wall, and a combination string presents more than one.
        """
        return any(abs(od - od_in) <= tol for _t, _b, od, _i in self.profile())


class Casing(Tubular):
    """A tubular run in the hole. OSDU ``Tubular``.

    ``kind`` says what it is, and the renderer takes the terminating glyph from
    it. **A shoe is a barrier-relevant symbol**: drawing one on a string that
    has none asserts a barrier that is not there. Sand screens hung on a
    packer, and a drill-in assembly left in hole, are both correctly tubulars
    in the hole and belong on a P&A drawing -- neither has a shoe.

    ``casing`` / ``liner`` terminate in a shoe; ``screen`` and ``tubular``
    do not. ``tubular`` is the honest catch-all for a string in the hole that
    is not a cased string -- a fish, junk, a drill-in assembly.

    Maps to OSDU ``TubularComponentType`` (``CAS.CAS``, ``CAS.L*``,
    ``GPSCRN.GPSCRN``); a screen's own construction is OSDU ``LinerType``
    (``Slotted`` / ``GravelPacked`` / ``PrePerforated``).
    """

    kind: Literal["casing", "liner", "screen", "tubular"] = "casing"

    @property
    def has_shoe(self) -> bool:
        """True when this string terminates in a shoe."""
        return self.kind in ("casing", "liner")


class CementPlug(_Base):
    """A set cement plug in the bore. OSDU well-activity ``CementJob`` output."""

    name: str = "Cement plug"
    top_md: float
    base_md: float
    plug_type: Optional[str] = Field(
        None,
        description="OSDU CementPlugType: Abandonment | KickOff | "
                    "LostCirculation | PlugBack | Suspension",
    )

    @model_validator(mode="after")
    def _check_plug_type(self) -> "CementPlug":
        _osdu_check("CementPlugType", self.plug_type, "plug_type")
        return self


class Perforation(_Base):
    """A perforated interval. OSDU ``WellboreCompletionInterval`` (perforated).

    ``casing_od_in`` names the string that was shot, so the marks can be drawn
    through the right wall when several strings are present at that depth; left
    None the innermost string there is assumed.
    """

    name: str = "Perforations"
    top_md: float
    base_md: float
    casing_od_in: Optional[float] = Field(
        None, description="OD (in) of the string shot; None = innermost there"
    )
    shots_per_m: Optional[float] = Field(
        None, description="shot density, per metre; presentation only"
    )
    method: Optional[str] = Field(
        None,
        description="OSDU PerforationIntervalType: JetPerforate | BulletPerf | "
                    "HPFluidJet | Abrasive-HP-FluidJet | T-C-Punch | Undefined",
    )

    @model_validator(mode="after")
    def _check_method(self) -> "Perforation":
        _osdu_check("PerforationIntervalType", self.method, "method")
        return self


class AnnulusFluid(_Base):
    """Fluid standing in an annulus -- what is there when it is not cement.

    The annulus is identified by ``inside_od_in``: the OD of the casing forming
    its INNER wall (so the 9-5/8in annulus is ``inside_od_in=9.625``). The outer
    wall is resolved the same way cement is -- the next-outer casing ID, or the
    drilled hole where none confines it -- so a fluid and the cement below it
    share one geometry.

    ``density_sg`` is the fluid density; it is NOT inferred from ``name``,
    because an annulus routinely holds something other than what its label
    suggests (a "packer fluid" annulus may hold seawater after a workover).
    Leave it None when unknown rather than assuming: an unstated density is a
    data gap, and for a barrier argument it is the number that matters.
    """

    name: str = "Annulus fluid"
    inside_od_in: float = Field(
        ..., description="OD (in) of the casing forming the annulus inner wall"
    )
    top_md: float
    base_md: float
    density_sg: Optional[float] = Field(
        None, description="fluid density (sg); None = not known, not assumed"
    )
    colour: Optional[str] = Field(
        None, description="hex fill; None = picked from the fluid name"
    )
    fluid_type: Optional[str] = Field(
        None,
        description="OSDU AnnularFluidType: CaCl2-Water | Diesel | Fresh-Water "
                    "| DrillingMud-Water | DrillingMud-Oil | CrudeOil | ...",
    )

    @model_validator(mode="after")
    def _check_fluid_type(self) -> "AnnulusFluid":
        _osdu_check("AnnularFluidType", self.fluid_type, "fluid_type")
        return self


class CompletionItem(_Base):
    """A completion component. OSDU ``WellboreCompletion`` element.

    Point items (``packer``/``sssv``/``nipple``) use ``md``; run items
    (``tubing``) use ``top_md`` + ``base_md``.
    """

    type: Literal["tubing", "packer", "sssv", "nipple"]
    name: Optional[str] = None
    od_in: float
    md: Optional[float] = None
    top_md: Optional[float] = None
    base_md: Optional[float] = None

    @model_validator(mode="after")
    def _depths(self) -> "CompletionItem":
        if self.type == "tubing":
            if self.top_md is None or self.base_md is None:
                raise ValueError("tubing needs top_md and base_md")
        else:
            if self.md is None:
                raise ValueError(f"{self.type} needs md")
        return self


class Formation(_Base):
    """A geological marker + lithology band. OSDU ``WellboreMarkerSet`` + litho.

    ``color`` defaults to WHITE, so a caller who builds ``Formation`` objects
    without one gets a blank lithology track and no warning. The NL colours are
    one call away: :func:`welleng.lithology.colour` takes an RGD unit code and
    rolls it up to its group.
    """

    name: str = ""
    top_md: float
    #: Base of the band. ``None`` = the next formation's top, or TD for the
    #: deepest -- which is what makes the deepest band (usually the reservoir)
    #: draw at all. Set it explicitly where the base is KNOWN and is not the
    #: next top: across an unconformity, or where the column has a gap.
    base_md: Optional[float] = None
    litho: str = ""
    color: str = "#ffffff"
    seal: bool = False  # caprock / barrier
    flow: bool = False  # permeable / flowing


class PressureProfile(_Base):
    """Pore- and fracture-**pressure** profile vs depth.

    Stored/displayed quantity is *pressure* (bar or psi), NOT mud weight.
    Use :meth:`from_emw` to convert an equivalent-mud-weight (sg) profile to
    hydrostatic pressure given TVD.
    """

    md: List[float]
    pore: List[float]
    frac: List[float]
    unit: Literal["bar", "psi"] = "bar"

    @model_validator(mode="after")
    def _equal_length(self) -> "PressureProfile":
        if not (len(self.md) == len(self.pore) == len(self.frac)):
            raise ValueError("md, pore and frac must have equal length")
        return self

    @classmethod
    def from_emw(
        cls,
        md: List[float],
        tvd: List[float],
        pore_sg: List[float],
        frac_sg: List[float],
        unit: Literal["bar", "psi"] = "bar",
    ) -> "PressureProfile":
        """Build from equivalent mud weight (sg) using P = emw * rho_w * g * TVD.

        Hydrostatic head of a column of density ``emw * 1000`` kg/m^3:
        ``P[Pa] = emw * 1000 * g * tvd`` -> bar (/1e5) or psi (*0.0145038).
        """

        def conv(sg: float, h: float) -> float:
            p_pa = sg * 1000.0 * _G * h
            p_bar = p_pa / 1.0e5
            return p_bar if unit == "bar" else p_bar * 14.5037738
        return cls(
            md=list(md),
            pore=[conv(s, h) for s, h in zip(pore_sg, tvd)],
            frac=[conv(s, h) for s, h in zip(frac_sg, tvd)],
            unit=unit,
        )


class Wellbore(_Base):
    """A single bore (branch). OSDU ``Wellbore`` / ``KickOffWellbore``.

    ``parent_id`` + ``kickoff_md`` position a lateral off its parent. For a
    single-bore well ``parent_id`` is ``None`` and ``kickoff_md`` is 0.
    """

    id: str = "main"
    parent_id: Optional[str] = None
    kickoff_md: float = 0.0
    survey: SurveyRef
    hole_sections: List[HoleSection] = Field(default_factory=list)
    casings: List[Casing] = Field(default_factory=list)
    cement_plugs: List[CementPlug] = Field(default_factory=list)
    annulus_fluids: List[AnnulusFluid] = Field(default_factory=list)
    perforations: List[Perforation] = Field(default_factory=list)
    completion: List[CompletionItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _radial_scale_monotonic(self) -> "Wellbore":
        """Radial scale must not increase with depth (shallow >= deep).

        Guarantees concentric nesting: a deeper/inner string can never render
        wider than a shallower/outer one. Only the sections that actually set
        ``radial_scale`` are checked, in depth order.
        """
        scaled = sorted(
            (s for s in self.hole_sections if s.radial_scale is not None),
            key=lambda s: s.top_md,
        )
        for shallow, deep in zip(scaled, scaled[1:]):
            if deep.radial_scale > shallow.radial_scale:
                raise ValueError(
                    "radial_scale must be monotonic non-increasing with depth: "
                    f"section at {deep.top_md} m has {deep.radial_scale} > "
                    f"{shallow.radial_scale} at {shallow.top_md} m"
                )
        return self

    @model_validator(mode="after")
    def _plugs_clear_of_tubing(self) -> "Wellbore":
        """A cement plug cannot occupy the same depths as a tubing run.

        You cannot set a plug in the bore with the completion still in the
        hole -- the tubing has to be pulled (or cut) first. Left unchecked the
        schematic will happily draw a plug with tubing running through it,
        which is a physically impossible well and reads as if it were a real
        one, so it is refused here rather than rendered.

        Cut-and-pull, where the tubing above a cut is removed and a plug set in
        the vacated bore, is expressible: shorten the tubing run to the cut
        depth and the overlap disappears.
        """
        runs = [(c.top_md or 0.0, c.base_md)
                for c in self.completion
                if c.type == "tubing" and c.base_md is not None]
        for p in self.cement_plugs:
            for top, base in runs:
                if p.top_md < base and p.base_md > top:
                    raise ValueError(
                        f"cement plug {p.name!r} ({p.top_md}-{p.base_md} m) "
                        f"overlaps a tubing run ({top}-{base} m): a plug cannot "
                        "be set through a completion that is still in the hole. "
                        "Pull or cut the tubing (shorten the run) first."
                    )
        return self


class WellSchematic(_Base):
    """Aggregate schematic: a well, a tree of wellbores, and shared geology.

    ``formations`` and ``pressures`` are well-level (geology is shared across
    branches). Accepts either the tree form (``wellbores=[...]``) or a flat
    single-bore form (survey/casings/... at top level, auto-wrapped into one
    :class:`Wellbore`).
    """

    well: Well
    wellbores: List[Wellbore]
    formations: List[Formation] = Field(default_factory=list)
    pressures: Optional[PressureProfile] = None

    @model_validator(mode="before")
    @classmethod
    def _wrap_flat(cls, data):
        """Fold a flat single-bore dict into ``wellbores=[<one bore>]``."""
        if not isinstance(data, dict):
            return data
        if "wellbores" in data:
            return data
        data = dict(data)  # don't mutate the caller's dict
        bore_keys = (
            "survey", "hole_sections", "casings", "cement_plugs",
            "annulus_fluids", "perforations", "completion",
            "id", "parent_id", "kickoff_md",
        )
        bore = {k: data.pop(k) for k in list(data) if k in bore_keys}
        if bore:
            data["wellbores"] = [bore]
        return data

    @property
    def primary(self) -> Wellbore:
        """The main bore -- the only branch this increment renders."""
        return self.wellbores[0]

    # --- loaders -----------------------------------------------------------
    def formation_bands(self) -> List[Tuple["Formation", float, float]]:
        """``(formation, top_md, base_md)`` for EVERY formation, deepest included.

        Taking each band's base from the NEXT formation's top leaves the
        deepest formation with no successor, so it is silently omitted -- and
        the deepest unit is the reservoir on almost every well. On one P&A
        sheet the oil zone was missing from the lithology track, the seal/flow
        track AND the formation background, nothing errored, and the sheet
        looked complete.

        The deepest band's base comes from TD (the deepest hole section, else
        the deepest survey station) unless the formation carries an explicit
        ``base_md``. A zero- or negative-thickness band is dropped: that is a
        data error, not a band.

        This lives on the model because three renderers each derived it
        independently and each got it wrong the same way.
        """
        forms = sorted(self.formations, key=lambda f: f.top_md)
        if not forms:
            return []
        bore = self.primary
        td = max(
            [h.base_md for h in (bore.hole_sections or [])]
            + [max(bore.survey.md) if bore.survey and bore.survey.md else 0.0]
        )
        out: List[Tuple["Formation", float, float]] = []
        for i, f in enumerate(forms):
            base = f.base_md
            if base is None:
                base = forms[i + 1].top_md if i + 1 < len(forms) else td
            if base > f.top_md:
                out.append((f, f.top_md, base))
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "WellSchematic":
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "WellSchematic":
        return cls.model_validate(json.loads(text))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WellSchematic":
        return cls.from_json(Path(path).read_text())

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)
