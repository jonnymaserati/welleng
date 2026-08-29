"""Separation-factor acceptance criteria — ISCWSA/OWSG anti-collision.

welleng owns the separation-factor MATHS (:mod:`welleng.clearance`). This module
adds the small typed POLICY layer the standard also defines, so consumers
of the SF stop each inventing "acceptable" privately.

It is NOT a policy engine. It carries the standard's graded thresholds, the action
each triggers, the HSE-risk parameter set that accompanies them, and a
``classify(sf) -> Verdict``. What a consumer DOES with a verdict — reject a
candidate, record it, maximise a target — is the consumer's business, not this
module's.

Source
------
Sodling, Clark & Allen, *"The Development and Testing of an Enhanced Anti-Collision
Rule"*, SPE-187073 (SPE Drill & Compl 34, 2019), DOI 10.2118/187073-PA -- the same
paper whose separation rule :class:`~welleng.clearance.IscwsaClearance` implements.
The graded criteria and their actions are the paper's own; the HSE-risk parameter
set (``k = 3.5`` Williamson 1998; the crossing-probability caution Bang 2017) is the
recommendation it makes for HSE-risk wells.

The standard's own escape hatch is respected: *"where local regulations are more
conservative they take precedence"* -- an operator override may only TIGHTEN the
critical floor (raise it), never loosen it below the mandatory ``SF = 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

# --- The standard's graded thresholds (SPE-187073) --------------------------
SF_CRITICAL = 1.0
#   SF < 1 is a critical condition -- STOP DRILLING for HSE-risk offset wells;
#   SF >= 1 is the mandatory minimum acceptable separation.
SF_REVIEW = 1.25
#   Planning: a prompt for detailed engineering review. Execution: triggers
#   preventive measures and a Management of Change (MOC).
SF_EXCLUDE = 5.0
#   Offset wells beyond this are excluded from further scanning (planning
#   convenience where hundreds of wells are involved).

# --- The HSE-risk parameter set that accompanies them -----------------------
K_HSE = 3.5                     # position-error multiplier (Williamson 1998)
SURFACE_MARGIN_M = 0.3          # Sm, surface margin [m] (~1 ft)
PROJECT_AHEAD_SIGMA_M = 0.5     # sigma_pa, project-ahead uncertainty [m] (~1.6 ft)

_SOURCE = "SPE-187073 (Sodling, Clark & Allen, 2019); DOI 10.2118/187073-PA"

# The bands, worst -> best. A consumer may key on `band` (these exact strings)
# or on the richer Verdict.
CRITICAL = "critical"       # SF below the (mandatory or operator) floor -- STOP
REVIEW = "review"           # acceptable, but below the review/MOC prompt
ACCEPTABLE = "acceptable"   # clear
EXCLUDE = "exclude"         # so far apart it is dropped from scanning


@dataclass(frozen=True)
class Verdict:
    """The classification of a single separation factor. Read-only."""

    sf: float               # the separation factor classified
    band: str                 # one of CRITICAL / REVIEW / ACCEPTABLE / EXCLUDE
    acceptable: bool          # sf >= the effective critical floor (band != CRITICAL)
    action: str               # the standard's action for this band, human-readable
    criterion: "AcceptanceCriteria"   # the criterion that produced this verdict

    def __bool__(self) -> bool:
        """``bool(verdict)`` is its acceptability -- so ``if not verdict:`` reads."""
        return self.acceptable

    def to_dict(self) -> dict:
        """Canonical JSON-serialisable form, for a provenance stamp.

        The BLESSED serialisation -- every consumer stamps a verdict identically
        rather than each hand-rolling :func:`dataclasses.asdict`, so a stored
        result's record of what "acceptable" meant is byte-identical across repos.
        Fields are listed explicitly (not reflected) so the stamp is a stable
        contract: a new internal field cannot silently change it. The nested
        criterion recurses through its own :meth:`AcceptanceCriteria.to_dict`.
        """
        return {
            "sf": self.sf,
            "band": self.band,
            "acceptable": self.acceptable,
            "action": self.action,
            "criterion": self.criterion.to_dict(),
        }


@dataclass(frozen=True)
class AcceptanceCriteria:
    """The standard's graded acceptance criteria, with an explicit operator override.

    Construct the standard via :meth:`standard`; tighten it via
    :meth:`with_operator_floor`. ``classify`` maps an SF to a :class:`Verdict`.
    """

    sf_critical: float = SF_CRITICAL          # STOP-drilling floor [SF]
    sf_review: float = SF_REVIEW              # engineering-review/MOC prompt [SF]
    sf_exclude: float = SF_EXCLUDE            # drop-from-scanning threshold [SF]
    k: float = K_HSE                          # HSE-risk position-error multiplier
    surface_margin_m: float = SURFACE_MARGIN_M       # Sm, surface margin [m]
    project_ahead_sigma_m: float = PROJECT_AHEAD_SIGMA_M  # project-ahead sigma [m]
    source: str = _SOURCE     # citation string; carries the override note if tightened
    operator_override: bool = False
    #   True when `with_operator_floor` has raised the critical floor above the
    #   standard's SF = 1. A result carrying this must NOT present the number as
    #   the standard's -- the override is the operator's, deliberately stamped.

    @classmethod
    def standard(cls) -> "AcceptanceCriteria":
        """The SPE-187073 criteria, unmodified."""
        return cls()

    def with_operator_floor(self, sf_min: float) -> "AcceptanceCriteria":
        """A tightened copy whose critical floor is ``sf_min``.

        An override may only TIGHTEN: ``sf_min`` below the mandatory ``SF = 1``
        is refused, because more-conservative local regulation takes precedence
        over the standard but nothing may drop below its mandatory minimum.
        """
        if sf_min < SF_CRITICAL:
            raise ValueError(
                f"an operator floor may only TIGHTEN the standard: {sf_min} is "
                f"below the mandatory SF = {SF_CRITICAL}. More-conservative local "
                f"regulation takes precedence, but nothing may sit below SF = 1."
            )
        return replace(
            self, sf_critical=float(sf_min), operator_override=True,
            source=f"{_SOURCE}; operator floor SF >= {sf_min}",
        )

    def classify(self, sf: float) -> Verdict:
        """Classify a separation factor into a :class:`Verdict`.

        Bands, checked in order so an operator floor above the review threshold
        still behaves: ``sf < sf_critical`` -> CRITICAL; ``sf > sf_exclude`` ->
        EXCLUDE; ``sf < sf_review`` -> REVIEW; else ACCEPTABLE.
        """
        sf = float(sf)
        if sf < self.sf_critical:
            band, action = CRITICAL, (
                "critical -- below the mandatory separation; STOP DRILLING for "
                "HSE-risk offset wells (SPE-187073)"
            )
        elif sf > self.sf_exclude:
            band, action = EXCLUDE, (
                "exclude -- separation so large the offset may be dropped from "
                "further scanning"
            )
        elif sf < self.sf_review:
            band, action = REVIEW, (
                "review -- acceptable but below the review threshold; prompt a "
                "detailed engineering review (planning) / MOC (execution)"
            )
        else:
            band, action = ACCEPTABLE, "acceptable -- clear of the review threshold"
        return Verdict(
            sf=sf, band=band, acceptable=(band != CRITICAL), action=action,
            criterion=self,
        )

    def to_dict(self) -> dict:
        """Canonical JSON-serialisable form, for a provenance stamp.

        The BLESSED serialisation (see :meth:`Verdict.to_dict`) -- fields listed
        explicitly so the stamp is a stable contract, not whatever
        :func:`dataclasses.asdict` happens to reflect.
        """
        return {
            "sf_critical": self.sf_critical,
            "sf_review": self.sf_review,
            "sf_exclude": self.sf_exclude,
            "k": self.k,
            "surface_margin_m": self.surface_margin_m,
            "project_ahead_sigma_m": self.project_ahead_sigma_m,
            "source": self.source,
            "operator_override": self.operator_override,
        }


def classify(sf: float, criteria: Optional[AcceptanceCriteria] = None) -> Verdict:
    """Classify ``sf`` against ``criteria`` (the SPE-187073 standard by default)."""
    return (criteria or AcceptanceCriteria.standard()).classify(sf)
