"""welleng -- well engineering: survey, error models, clearance, trajectory.

Submodules load ON DEMAND (PEP 562), and that is a correctness decision, not a
micro-optimisation.

Every submodule used to be imported eagerly here, so ``from welleng.survey
import MinCurve`` -- a pure-numpy arithmetic kernel -- cost **1.7 s** and
dragged in VTK, trimesh, plotly, pandas and scipy, because importing any
submodule runs this file first. A consumer wanting minimum curvature had to
take a renderer with it.

That is not merely slow: it is a standing invitation to FORK the kernel rather
than import it. A family member did exactly that, hand-rolling minimum
curvature and a bisection inverse of ``interpolate_tvd``, and the stated reason
was a dependency boundary this cost is what makes attractive. A principle that
is expensive to obey gets disobeyed, and the disobedience then gets a
rationale. Making the cheap thing correct is core's job, not the consumer's
discipline.

``welleng.visual`` / ``welleng.mesh`` still work exactly as before -- attribute
access triggers the import -- so ``we.visual.figure(...)`` after ``import
welleng`` is unchanged.
"""
import importlib
from typing import TYPE_CHECKING

from welleng.version import __version__

#: Submodules resolved lazily on attribute access.
_SUBMODULES = (
    "architecture", "clearance", "composition", "conditioning", "connector",
    "error", "errors", "exchange", "fluid", "geomag", "hierarchy", "io",
    "kick_tolerance", "lint", "lithology", "mesh", "node", "osdu", "osdu_ref",
    "schematic", "survey", "target", "torque_drag", "units", "utils",
    "version", "visual",
)

#: Names re-exported from a submodule: ``name -> submodule``. Kept working
#: because they were part of the public surface before this file went lazy.
_REEXPORTS = {
    "EDMReader": "exchange.edm_stream",
    "open_edm": "exchange.edm_stream",
    "classify_tool": "exchange.edm_stream",
    "ToolKind": "exchange.edm_stream",
    "SurveyTool": "exchange.edm_stream",
    "Wellbore": "exchange.edm_stream",
    "EDMSurveyHeader": "exchange.edm_stream",
    "ProgramInterval": "exchange.edm_stream",
    "SurveyStation": "exchange.edm_stream",
    "WellboreSurvey": "exchange.edm_stream",
    "SurveyComposition": "composition",
    "SurveySection": "composition",
}

#: The one re-export whose exposed name differs from its source name.
_RENAMED = {"EDMSurveyHeader": "SurveyHeader"}

__all__ = sorted(set(_SUBMODULES) | set(_REEXPORTS) | {"__version__"})


def __getattr__(name: str):
    """Resolve a submodule or re-exported name on first access (PEP 562)."""
    if name in _SUBMODULES:
        mod = importlib.import_module(f"welleng.{name}")
        globals()[name] = mod
        return mod
    if name in _REEXPORTS:
        mod = importlib.import_module(f"welleng.{_REEXPORTS[name]}")
        obj = getattr(mod, _RENAMED.get(name, name))
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'welleng' has no attribute {name!r}")


def __dir__():
    return __all__


if TYPE_CHECKING:       # keep static analysis and IDEs seeing the real thing
    from welleng import (  # noqa: F401
        architecture, clearance, composition, conditioning, connector, error,
        errors, exchange, fluid, geomag, hierarchy, io, kick_tolerance, lint,
        lithology, mesh, node, osdu, osdu_ref, schematic, survey, target,
        torque_drag, units, utils, visual,
    )
    from welleng.composition import SurveyComposition, SurveySection  # noqa: F401
    from welleng.exchange.edm_stream import (  # noqa: F401
        EDMReader, ProgramInterval, SurveyHeader as EDMSurveyHeader,
        SurveyStation, SurveyTool, ToolKind, Wellbore, classify_tool, open_edm,
    )
