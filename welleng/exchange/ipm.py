"""
Reader for **IPM** (Instrument Performance Model) survey-tool error-model files.

The ``.IPM`` format is the de-facto industry exchange format for directional-survey
error models (Landmark Engineer's Desktop / COMPASS, Schlumberger Drilling Office and
others export it). Each file describes one survey tool as a set of ISCWSA-style
weight-function terms: an error source, the axis it perturbs, its propagation
(correlation) mode, a 1-sigma magnitude and the weight-function *formula*.

File structure::

    #Tool Name  : MWD+SAG
    #ShortName  : MWD+SAG
    #Description: ...
    #Remarks    : ...
    #Name<TAB>Vector<TAB>Tie-On<TAB>Unit<TAB>Value<TAB>Formula
    abx<TAB>i<TAB>s<TAB>-<TAB>0.004<TAB>(-cos(inc)*sin(tfo))/gtot
    ...

Columns
-------
- **Name** — error-source code (``abx``, ``mbz``, ``sag``, ``decg`` ...).
- **Vector** — axis the term perturbs: ``i`` inclination, ``a`` azimuth, ``l`` lateral
  (and ``d``/``e``/``f`` depth-family in some dialects).
- **Tie-On** — propagation/correlation mode: ``s`` systematic, ``r`` random,
  ``g`` global, ``w`` well-by-well.
- **Unit** — magnitude unit (``-`` dimensionless, ``nt``, ``deg`` ...).
- **Value** — the 1-sigma error magnitude.
- **Formula** — the weight function (an expression in ``inc``, ``azm``, ``tfo``,
  ``dip``, ``gtot``, ``mtot``, ``tmd``, ``tvd`` ...). Stored verbatim; this reader does
  not evaluate it.

This module only *parses* the file into :class:`IPMModel` / :class:`IPMTerm`;
mapping the
terms onto a propagation engine is left to the caller.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

__all__ = ["IPMTerm", "IPMModel", "read_ipm", "loads_ipm"]


@dataclass
class IPMTerm:
    """A single error-model term (one row of an IPM file)."""

    name: str
    vector: str
    tie_on: str
    unit: str
    value: float
    formula: str


@dataclass
class IPMModel:
    """A parsed IPM survey-tool error model."""

    name: str = ""
    short_name: str = ""
    description: str = ""
    remarks: str = ""
    header: Dict[str, str] = field(default_factory=dict)
    terms: List[IPMTerm] = field(default_factory=list)

    def sources(self) -> List[str]:
        """Sorted, de-duplicated error-source names (each may span i/a/l rows)."""
        return sorted({t.name for t in self.terms})

    def by_tie_on(self, tie_on: str) -> List[IPMTerm]:
        """All terms with the given propagation mode
        (``'s'``/``'r'``/``'g'``/``'w'``)."""
        return [t for t in self.terms if t.tie_on == tie_on]

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return {
            "name": self.name,
            "short_name": self.short_name,
            "description": self.description,
            "remarks": self.remarks,
            "header": dict(self.header),
            "terms": [vars(t) for t in self.terms],
        }


def _parse_lines(lines) -> IPMModel:
    header: Dict[str, str] = {}
    terms: List[IPMTerm] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        if line.startswith("#"):
            body = line[1:]
            # the column-header row (starts with 'Name', contains
            # 'Vector') is not metadata
            if re.match(r"\s*Name\b", body) and "Vector" in body:
                continue
            if ":" in body:
                key, _, val = body.partition(":")
                header[key.strip()] = val.strip()
            continue
        # data row — tab-separated, fall back to runs of whitespace
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 6:
            parts = [p for p in re.split(r"\t|\s{2,}", line.strip()) if p != ""]
        if len(parts) < 6:
            continue
        name, vector, tie_on, unit, value = parts[:5]
        formula = "\t".join(parts[5:]).strip()
        try:
            magnitude = float(value)
        except ValueError:
            continue
        terms.append(IPMTerm(name, vector, tie_on, unit, magnitude, formula))
    return IPMModel(
        name=header.get("Tool Name", ""),
        short_name=header.get("ShortName", ""),
        description=header.get("Description", ""),
        remarks=header.get("Remarks", ""),
        header=header,
        terms=terms,
    )


def read_ipm(path, encoding: str = "latin-1") -> IPMModel:
    """Parse an ``.IPM`` file from ``path`` into an :class:`IPMModel`.

    Parameters
    ----------
    path : str or path-like
        Path to the ``.IPM`` file.
    encoding : str, default ``'latin-1'``
        Text encoding (IPM exports are typically latin-1; degree signs etc.).

    Returns
    -------
    IPMModel

    Examples
    --------
    >>> model = read_ipm("MWD+SAG.IPM")            # doctest: +SKIP
    >>> model.name, len(model.terms)               # doctest: +SKIP
    ('MWD+SAG', 30)
    """
    with open(path, "r", encoding=encoding) as f:
        return _parse_lines(f)


def loads_ipm(text: str) -> IPMModel:
    """Parse an IPM model from an in-memory string (see :func:`read_ipm`)."""
    return _parse_lines(text.splitlines())
