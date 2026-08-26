"""EDM IPM import — build error models from an EDM export's survey-tool layer.

An EDM (Landmark Engineer's Data Model) export can carry the COMPASS
error-model layer alongside the well data:

- ``CD_SURVEY_TOOL`` — the named survey-tool definitions;
- ``DP_TOOL_TERM``  — the full instrument performance model (IPM) per tool:
  one row per error term with its magnitude (``c_value``), units
  (``c_units``), weighting function as a text formula (``c_formula``),
  vector direction (``vector_type``) and tie-on/propagation (``tie_type``);
- ``CD_SURVEY_HEADER.survey_tool_id`` — which tool ran each survey interval;
- ``DP_MAGNETIC`` — the per-wellbore geomagnetic reference used at the time.

This module parses those tables and converts each tool's IPM into the
ISCWSA-JSON-shaped model dict that welleng's formula-interpreter error engine
evaluates (`welleng.errors.tool_errors`), so a survey's uncertainty can be
computed with the *actual tool models the operator ran* instead of a
nearest-standard-model guess:

>>> # doctest: +SKIP
>>> ipm = parse_edm_ipm("data/Volve.xml")
>>> model = ipm.error_model("5pod5")          # a survey_tool_id or tool name
>>> s = Survey(md, inc, azi, header=header, error_model=model)

COMPASS IPM conventions (COMPASS 5000 manual, ch. 3 "Survey Tool Editor"):

- ``vector_type``: ``a`` azimuth, ``i`` inclination (highside), ``e`` depth
  (ISCWSA), ``d`` depth (Wolff & de Wardt), ``l`` lateral (equivalent to an
  azimuth error divided by sin(inclination)), ``b``/``j`` azimuth/inclination
  bias, ``m`` misalignment, ``n`` intermediate (see below).
- ``tie_type``: ``r`` random, ``s`` systematic, ``w`` well-by-well,
  ``g`` global, ``n`` **not used in accumulation** — an intermediate value
  other formulas reference by name (e.g. ``latsf``, ``tvdsf``).
- Formulas are Excel-style text over lower-case variables: ``inc``, ``azm``
  (magnetic azimuth), ``azt`` (true), ``azi`` (grid), ``tmd`` (measured
  depth), ``dmd`` (course length to the previous station), ``tvd``,
  ``gtot`` (total gravity), ``mtot`` (total B-field, nT), ``dip``, ``lat``
  (latitude), ``erot`` (earth rate), ``mtf`` (metres-to-feet — identity
  here, welleng evaluates in SI).

Intermediate names may contain ``-`` (e.g. ``ngxy-b1``), which would parse
as subtraction — they are mangled to ``_`` in both definitions and
referencing formulas before evaluation.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List

from ..exchange.edm_stream import ToolKind, classify_tool
from .interpreter import _rewrite_excel_to_python, _validate

_FORMULA_KEYS = ("depth_formula", "inclination_formula", "azimuth_formula")


def _first_unparseable_formula(entry):
    """Return ``(formula_key, error)`` for the first term formula that does not
    parse (syntax typo / unknown function), else None. Uses the same rewrite +
    AST whitelist the evaluator does, so it catches at import what would
    otherwise raise mid-covariance."""
    for key in _FORMULA_KEYS:
        f = entry.get(key)
        if not f or f == "0":
            continue
        try:
            _validate(ast.parse(_rewrite_excel_to_python(str(f)), mode="eval"))
        except (SyntaxError, ValueError) as e:
            return key, str(e)
    return None

# a formula identifier token (a survey variable, math function, or an EDM
# intermediate name); used by ``normalise_edm_model`` to inline intermediates
# by whole-token replacement (never a substring, so ``dinit`` is not hit
# inside ``deltad`` and ``ainit`` is not hit inside a longer name).
_IDENT = re.compile(r"[A-Za-z_]\w*")

# COMPASS c_units -> the ISCWSA-JSON units enum welleng's engine converts
# from (see tool_errors._MAG_UNIT_TO_BASE). COMPASS 'm'/'im' encode the
# metres<->feet conversions of its feet-internal engine; welleng evaluates
# in SI so lengths stay metres.
_COMPASS_UNITS = {
    "d": "deg",
    "m": "m",
    "im": "1/m",
    "nt": "nT",
    "dnt": "deg/nT",
    "-": "-",
    "": "-",
}

_TIE_TO_PROPAGATION = {
    "r": "Random",
    "s": "Systematic",
    "w": "Well",
    "g": "Global",
}

# regex helpers for the streaming XML scan (the Volve export is ~200 MB;
# a DOM parse is not an option and only flat attribute rows are needed)
_ROW_PATTERNS = {
    tag: re.compile(rf"<{tag}\s([^>]*?)/?>")
    for tag in (
        "CD_SURVEY_TOOL", "DP_TOOL_TERM", "DP_MAGNETIC", "CD_SURVEY_HEADER",
    )
}
_ATTR = re.compile(r'([\w-]+)="([^"]*)"')


@dataclass
class IPMTerm:
    """One ``DP_TOOL_TERM`` row."""

    name: str
    sequence_no: int
    vector_type: str            # a | i | e | d | l | b | j | m | n
    tie_type: str               # r | s | w | g | n
    value: float
    units: str                  # raw COMPASS units code
    formula: str
    min_inc: float = 0.0        # inclination range gate (deg); 0/0 = none
    max_inc: float = 0.0


@dataclass
class IPMTool:
    """A survey tool and its full IPM."""

    tool_id: str
    name: str
    description: str = ""
    kind: ToolKind = ToolKind.OTHER
    terms: List[IPMTerm] = field(default_factory=list)

    @property
    def intermediates(self) -> List[IPMTerm]:
        return [t for t in self.terms if t.tie_type == "n"]

    @property
    def error_terms(self) -> List[IPMTerm]:
        return [t for t in self.terms if t.tie_type != "n"]


def _mangle(name: str) -> str:
    """COMPASS term names may contain '-', invalid in a parsed formula."""
    return name.replace("-", "_")


def _mangle_formula(formula: str, hyphenated: List[str]) -> str:
    """Replace hyphenated intermediate NAMES before '-' means subtraction.

    Longest-first so ``ngxy-gd1`` is replaced before any shorter overlap.
    """
    for name in sorted(hyphenated, key=len, reverse=True):
        formula = formula.replace(name, _mangle(name))
    return formula


class EDMIPMError(ValueError):
    """The EDM export lacks the requested tool / IPM content."""


@dataclass
class EDMIPM:
    """The parsed error-model layer of an EDM export."""

    tools: Dict[str, IPMTool]                 # keyed by survey_tool_id
    run_tool_map: Dict[str, str]              # survey_header_id -> tool_id
    magnetics: List[Dict[str, str]]           # DP_MAGNETIC rows, raw

    def tool(self, key: str) -> IPMTool:
        """Fetch a tool by ``survey_tool_id`` or (case-insensitive) name."""
        if key in self.tools:
            return self.tools[key]
        lowered = key.lower()
        matches = [
            t for t in self.tools.values() if t.name.lower() == lowered
        ]
        if not matches:
            raise EDMIPMError(f"no survey tool {key!r} in the EDM export")
        if len(matches) > 1:
            raise EDMIPMError(
                f"tool name {key!r} is ambiguous; use the survey_tool_id "
                f"({', '.join(t.tool_id for t in matches)})"
            )
        return matches[0]

    def error_model(self, key: str, normalise: bool = False,
                    compass_gyro_parity: bool = False) -> dict:
        """Build the welleng error-model dict for a tool (see
        :func:`ipm_to_error_model`).

        ``normalise=True`` inlines the EDM intermediates into the term formulas
        (:func:`normalise_edm_model`), yielding a self-contained ISCWSA-JSON
        model for a generic/symbolic formula engine; the covariance is
        unchanged.

        ``compass_gyro_parity=True`` (gyro tools only) appends a systematic
        vertical depth-scale term (:data:`COMPASS_GYRO_TVDSF` per TVD) that
        COMPASS applies to gyro definitives but does NOT export in
        ``DP_TOOL_TERM`` — an empirical well-level vertical term, back-calculated
        from public Volve; its exact COMPASS source is not established (a
        depth-term substitution / wireline hypothesis was tested against F-12 and
        did not hold). OFF by default — the default model is OWSG-standard-faithful;
        enable ONLY to reproduce COMPASS's stored gyro covariances. Non-standard.
        """
        return ipm_to_error_model(
            self.tool(key), normalise=normalise,
            compass_gyro_parity=compass_gyro_parity,
        )


def tool_from_ipm_model(model) -> IPMTool:
    """Bridge a parsed ``.IPM`` file (:class:`welleng.exchange.ipm.IPMModel`)
    to an :class:`IPMTool`, so file-based tool models run the same
    conversion/engine path as EDM-embedded ones:

    >>> # doctest: +SKIP
    >>> from welleng.exchange.ipm import read_ipm
    >>> tool = tool_from_ipm_model(read_ipm("MWD+SAG.IPM"))
    >>> s = Survey(md, inc, azi, header=h, error_model=ipm_to_error_model(tool))
    """
    name = model.short_name or model.name
    return IPMTool(
        tool_id=name,
        name=name,
        description=model.description,
        kind=classify_tool(name, model.description),
        terms=[
            IPMTerm(
                name=t.name,
                sequence_no=i,
                vector_type=t.vector.lower(),
                tie_type=t.tie_on.lower(),
                value=t.value,
                units=t.unit.lower(),
                formula=t.formula,
            )
            for i, t in enumerate(model.terms)
        ],
    )


def parse_edm_ipm(path: str) -> EDMIPM:
    """Stream-parse an EDM export's survey-tool error-model layer.

    Parameters
    ----------
    path : str
        The EDM XML export (e.g. the public Volve dataset's ``Volve.xml``).

    Returns
    -------
    EDMIPM
    """
    rows: Dict[str, List[Dict[str, str]]] = {t: [] for t in _ROW_PATTERNS}
    with open(path, encoding="utf-8", errors="replace") as fh:
        tail = ""
        for chunk in iter(lambda: fh.read(1 << 20), ""):
            buf = tail + chunk
            for tag, pattern in _ROW_PATTERNS.items():
                for m in pattern.finditer(buf):
                    rows[tag].append(dict(_ATTR.findall(m.group(1))))
            # keep a tail longer than any row so no match spans a boundary
            tail = buf[-(1 << 14):]
    # de-duplicate rows the overlapping tail scanned twice
    for tag in rows:
        seen, unique = set(), []
        for r in rows[tag]:
            k = tuple(sorted(r.items()))
            if k not in seen:
                seen.add(k)
                unique.append(r)
        rows[tag] = unique

    tools: Dict[str, IPMTool] = {}
    for r in rows["CD_SURVEY_TOOL"]:
        tool_id = r.get("survey_tool_id", "")
        name = r.get("tool_name", "") or r.get("name", "")
        desc = r.get("description", "")
        tools[tool_id] = IPMTool(
            tool_id=tool_id, name=name, description=desc,
            kind=classify_tool(name, desc),
        )

    for r in rows["DP_TOOL_TERM"]:
        tool_id = r.get("survey_tool_id", "")
        if tool_id not in tools:
            tools[tool_id] = IPMTool(tool_id=tool_id, name=tool_id)
        tools[tool_id].terms.append(IPMTerm(
            name=r.get("term_name", ""),
            sequence_no=int(float(r.get("sequence_no", "0"))),
            vector_type=r.get("vector_type", "").lower(),
            tie_type=r.get("tie_type", "").lower(),
            value=float(r.get("c_value", "0") or 0),
            units=r.get("c_units", "").lower(),
            formula=r.get("c_formula", "0"),
            min_inc=float(r.get("min_range", "0") or 0),
            max_inc=float(r.get("max_range", "0") or 0),
        ))
    for t in tools.values():
        t.terms.sort(key=lambda term: term.sequence_no)

    run_tool_map = {
        r["survey_header_id"]: r["survey_tool_id"]
        for r in rows["CD_SURVEY_HEADER"]
        if r.get("survey_header_id") and r.get("survey_tool_id")
    }
    return EDMIPM(
        tools=tools, run_tool_map=run_tool_map,
        magnetics=rows["DP_MAGNETIC"],
    )


#: COMPASS-parity gyro vertical depth-scale (per TVD), systematic. NOT an OWSG
#: standard term — back-calculated from public Volve gyro definitives (consistent
#: ~2.73e-4 across 8 wells / 4 gyro tools, CV ~3%); an empirical well-level
#: vertical term COMPASS applies but does not export in DP_TOOL_TERM (exact source
#: not established — a wireline/depth-term-substitution hypothesis was tested
#: against F-12 and did not hold). Applied only via ``compass_gyro_parity=True``.
#: See docs/dev/EDM_ERROR_MODEL_CONTRACT.md.
COMPASS_GYRO_TVDSF = 2.73e-4


def ipm_to_error_model(tool: IPMTool, normalise: bool = False,
                       compass_gyro_parity: bool = False,
                       strict: bool = True) -> dict:
    """Convert one tool's IPM to the welleng (ISCWSA-JSON-shaped) model dict.

    The dict can be passed straight to ``Survey(..., error_model=model)`` /
    ``ErrorModel(survey, error_model=model)``.

    ``strict=True`` (default) raises :class:`EDMIPMError` on a term whose
    weighting-function formula does not parse (a syntax typo, an unknown
    function). Real-world COMPASS IPM files that users have hand-edited can carry
    such malformed weighting functions; ``strict=False`` instead **warns and
    skips** the offending term, so one bad term does not abort the whole import.
    (Structural problems -- an unsupported vector/tie type, a zero reference with
    a non-zero component -- always raise: they are data errors, not typos.)

    With ``normalise=True`` the EDM intermediates are inlined into the term
    formulas (:func:`normalise_edm_model`) so the model is self-contained for a
    generic/symbolic formula engine; the covariance is unchanged.

    ``compass_gyro_parity=True`` (gyro tools only) appends the non-standard
    :data:`COMPASS_GYRO_TVDSF` systematic vertical depth-scale term (see
    :meth:`IPM.error_model`). OFF by default.

    Conversion rules
    ----------------
    - vector ``a``/``b`` -> azimuth formula; ``i``/``j`` -> inclination;
      ``e``/``d`` -> depth; ``l`` (lateral) -> azimuth formula divided by
      ``sin(inc)`` (clamped near vertical — the ``1/sin`` cancels against
      the ``sin(inc)`` in the position sensitivity to azimuth, which is the
      standard ISCWSA lateral treatment); ``m`` (misalignment disc) -> equal
      inclination and lateral components.
    - tie ``r``/``s``/``w``/``g`` -> Random/Systematic/Well/Global.
    - tie ``n`` rows become named intermediates evaluated (in sequence
      order) into the formula namespace: ``name = value * formula``.
    """
    if compass_gyro_parity and tool.kind is ToolKind.GYRO:
        # append the non-standard COMPASS gyro wireline depth-scale (systematic,
        # depth vector, per TVD) — reproduces COMPASS's stored gyro sigma_V.
        seq = max((t.sequence_no for t in tool.terms), default=0) + 1
        tool = replace(tool, terms=tool.terms + [IPMTerm(
            name="DTVDSF_COMPASS", sequence_no=seq, vector_type="e",
            tie_type="s", value=COMPASS_GYRO_TVDSF, units="-", formula="tvd",
        )])

    hyphenated = [t.name for t in tool.intermediates if "-" in t.name]

    intermediates = [
        {
            "name": _mangle(t.name),
            "value": t.value * _unit_multiplier(t.units),
            "formula": _mangle_formula(t.formula, hyphenated),
        }
        for t in tool.intermediates
    ]

    # Rows sharing a NAME are components of the SAME error source (the
    # COMPASS convention: reuse the name to add onto the same source — e.g.
    # a misalignment with an inclination row AND a lateral row). They share
    # one random realisation, so they must land in ONE term whose per-axis
    # weight formulas carry each component — not independent terms (and not
    # clobber each other in a name-keyed dict). Rows sharing a name but NOT
    # a tie type (e.g. a gyro's 'grexy' with a random and a systematic leg)
    # cannot share a realisation across propagation modes — they become
    # separate terms disambiguated by tie type.
    grouped: Dict[tuple, List[IPMTerm]] = {}
    for t in tool.error_terms:
        grouped.setdefault((t.name, t.tie_type), []).append(t)
    name_counts: Dict[str, int] = {}
    for (nm, _tie) in grouped:
        name_counts[nm] = name_counts.get(nm, 0) + 1

    terms = []
    for (name, tie), rows in grouped.items():
        ref = rows[0]
        if tie not in _TIE_TO_PROPAGATION:
            raise EDMIPMError(
                f"tool {tool.name!r} term {name!r}: unsupported "
                f"tie_type {tie!r}"
            )
        if name_counts[name] > 1:
            name = f"{name}({tie})"
        axes = {"d": [], "i": [], "a": []}
        # vertical-hole singularity weights (ISCWSA Rev5.13 Sec 11.5): at a
        # vertical station the inclination component acts along the (canonical
        # azi=0 ->) NORTH axis and the lateral component along EAST — the
        # standard XYM3/XYM4 treatment. Without this, the "east" disc mode of
        # a near-vertical misalignment silently vanishes (dr/dAz -> 0).
        sing_n, sing_e = [], []
        for t in rows:
            formula = _mangle_formula(t.formula, hyphenated)
            # same source, possibly different magnitude per component —
            # scale relative to the group's reference value
            if t.value != ref.value:
                if ref.value == 0:
                    raise EDMIPMError(
                        f"tool {tool.name!r} term {name!r}: zero reference "
                        "value with non-zero component"
                    )
                formula = f"({t.value / ref.value}) * ({formula})"
            vector = t.vector_type
            if vector in ("a", "b"):
                axes["a"].append(formula)
            elif vector in ("i", "j"):
                axes["i"].append(formula)
                sing_n.append(formula)
            elif vector in ("e", "d"):
                axes["d"].append(formula)
            elif vector == "l":
                axes["a"].append(
                    f"({formula}) / maximum(sin(inc), 1e-9)"
                )
                sing_e.append(formula)
            elif vector == "m":
                axes["i"].append(formula)
                sing_n.append(formula)
                axes["a"].append(
                    f"({formula}) / maximum(sin(inc), 1e-9)"
                )
                sing_e.append(formula)
            else:
                raise EDMIPMError(
                    f"tool {tool.name!r} term {name!r}: unsupported "
                    f"vector_type {vector!r}"
                )

        def _sum(parts: List[str]) -> str:
            return " + ".join(f"({p})" for p in parts) if parts else "0"

        # The singularity substitution REPLACES the term's whole position
        # error at vertical stations, so it is only emitted for purely
        # angular terms; a mixed depth+angular source would lose its depth
        # part there (no such tool exists in the Volve set).
        has_sing = (sing_n or sing_e) and not axes["d"]
        entry = {
            "name": name,
            "value": ref.value,
            "units": _COMPASS_UNITS.get(ref.units, "-"),
            "propagation_mode": _TIE_TO_PROPAGATION[tie],
            "depth_formula": _sum(axes["d"]),
            "inclination_formula": _sum(axes["i"]),
            "azimuth_formula": _sum(axes["a"]),
            "north_singularity": _sum(sing_n) if has_sing else None,
            "east_singularity": _sum(sing_e) if has_sing else None,
            "vertical_singularity": "0" if has_sing else None,
        }
        if ref.max_inc > ref.min_inc:
            # COMPASS min_range/max_range: the term is active only within
            # this inclination window (deg) -- e.g. mutually-exclusive gyro
            # mode terms. Emit the keys the evaluation engine's per-term
            # gating reads (window semantics, no carry); outside the window
            # the term contributes zero.
            entry["inc_min_deg"] = ref.min_inc
            entry["inc_max_deg"] = ref.max_inc
        bad = _first_unparseable_formula(entry)
        if bad is not None:
            fname, err = bad
            msg = (f"tool {tool.name!r} term {name!r}: unparseable "
                   f"{fname} {entry[fname]!r} ({err})")
            if strict:
                raise EDMIPMError(msg)
            warnings.warn(msg + " -- term skipped (strict=False)",
                          RuntimeWarning, stacklevel=2)
            continue
        terms.append(entry)

    model = {
        "metadata": {
            "model_id": tool.tool_id,
            "short_name": tool.name,
            "long_name": f"EDM IPM: {tool.name}",
            "tool_type": tool.kind.value,
            "source": "EDM export DP_TOOL_TERM",
            "framework": "COMPASS IPM",
        },
        "parameters": {"inc_min": 0, "inc_max": 180},
        "edm_intermediates": intermediates,
        "terms": terms,
    }
    return normalise_edm_model(model) if normalise else model


# the formula-bearing keys of a term dict, in the order the engine reads them
_FORMULA_FIELDS = (
    "depth_formula", "inclination_formula", "azimuth_formula",
    "north_singularity", "east_singularity", "vertical_singularity",
)


def normalise_edm_model(model: dict) -> dict:
    """Inline a model's EDM intermediates into its term formulas.

    :func:`ipm_to_error_model` emits COMPASS ``tie_type='n'`` rows as a
    separate ``edm_intermediates`` list — per-station computed sub-expressions
    (e.g. ``deltad = abs(tmd - dinit)``, ``ainit = 45 deg``) that the term
    formulas reference by name and welleng's own engine evaluates once into the
    formula namespace (``tool_errors``). That is an EDM-specific convention: the
    ISCWSA-JSON/OSDU schema has no per-station intermediate mechanism — a term
    is a *self-contained* inclination/azimuth/depth formula plus scalar
    parameters. A generic engine reading the model (e.g. the symbolic compiler
    behind the vectorised EOU path) therefore cannot resolve ``ainit`` and
    friends, and rejects the model.

    This function returns an equivalent model in which every intermediate is
    inlined into the formulas that reference it, and the ``edm_intermediates``
    section is dropped. The result references only base survey variables
    (``inc``, ``azi``, ``gtot``, ``erot`` …) and numeric constants — a
    self-contained ISCWSA-JSON model any standard formula engine can consume.

    The transform is an exact algebraic substitution, so the covariance is
    unchanged: welleng's own engine gives the *identical* result on the
    normalised model (validated bit-for-bit on every Volve gyro/MWD-gyro tool;
    see ``tests/test_edm_ipm_normalise.py``). Degree→radian conversion is
    carried by each intermediate's already-SI ``value`` (set from its unit code
    in :func:`ipm_to_error_model`), so inlining ``(value)*(formula)`` preserves
    it. Intermediates are resolved in sequence order — a later one may
    reference an earlier one (``w_34 = sqrt(1 - w_12**2)``) — so each
    expansion is itself already free of intermediate names before it is
    substitved onward.

    Parameters
    ----------
    model : dict
        A model dict from :func:`ipm_to_error_model` /
        :meth:`EDMIPM.error_model`. A model without an ``edm_intermediates``
        section is returned unchanged (a shallow copy).

    Returns
    -------
    dict
        The equivalent model with intermediates inlined and no
        ``edm_intermediates`` key.
    """
    intermediates = model.get("edm_intermediates") or []
    if not intermediates:
        return {k: v for k, v in model.items() if k != "edm_intermediates"}

    def _inline(formula: str, resolved: Dict[str, str]) -> str:
        return _IDENT.sub(
            lambda m: resolved.get(m.group(0), m.group(0)), formula
        )

    # resolve each intermediate against the ones already resolved before it,
    # so every stored expansion references only base variables/constants
    resolved: Dict[str, str] = {}
    for interm in intermediates:
        body = _inline(interm["formula"], resolved)
        resolved[interm["name"]] = f"(({interm['value']!r})*({body}))"

    out = {k: v for k, v in model.items() if k != "edm_intermediates"}
    out["terms"] = []
    for term in model["terms"]:
        new_term = dict(term)
        for field_name in _FORMULA_FIELDS:
            formula = new_term.get(field_name)
            if formula not in (None, ""):
                new_term[field_name] = _inline(formula, resolved)
        out["terms"].append(new_term)
    return out


def _unit_multiplier(units: str) -> float:
    """COMPASS unit code -> multiplier into welleng's SI evaluation base."""
    import numpy as np

    return {
        "d": np.pi / 180.0,
        "dnt": np.pi / 180.0,
    }.get(units, 1.0)
