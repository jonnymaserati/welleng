"""Source-level checks for ways of misusing a survey that no test will catch.

Why this is a LINT and not a test
---------------------------------
The defect this module exists for is ``np.interp(md, survey.md, survey.tvd)``:
a straight chord through (MD, TVD) where the trajectory is a circular arc. It
cannot be caught numerically by the repo that commits it, because the error
scales with the CALLER's station spacing, not with anything the committing repo
controls. On a densely-sampled survey (~30 m stations) it is right to a few
centimetres; on a 13-station design survey with 480 m gaps the same line is out
by **14.4 m of TVD**. So every fixture passes, nothing errors, and the answer
is plausible, in range, and monotonic. The only place to catch it is the source.

Why it lives in CORE
--------------------
Because the alternative is nine copies. The first consumer to hit this wrote an
excellent regex guard for its own repo -- and that guard missed one live
instance in its own source (``np.interp(md, p_md, parent.inc)``, because the
pattern looked for ``inc_deg``/``inc_rad`` and this was ``.inc``). A guard
forked into every repo is the same duplication problem one level up, with each
fork carrying its own holes. This one is AST-based, so it sees the attribute
regardless of what the object is called, and a hole fixed here is fixed for
every consumer at once.

The rule
--------
Flag a call when **the interpolated quantity** -- ``fp``, the third argument --
is a trajectory axis (md, tvd, inc, azi, n/e/northing/easting). That is the
precise statement of the defect, and it is what separates it from the ordinary,
correct thing right next to it:

    np.interp(tvd, self.tvd, self.sigma_V)     # OK  -- a PROPERTY against depth
    np.interp(md, survey.md, survey.tvd)       # BAD -- the TRAJECTORY itself
    np.interp(md, p_md, parent.inc)            # BAD -- and an ANGLE at that
    np.interp(mid[2], target.tvd, target.md)   # BAD -- the inverse, same error

Interpolating a property (pore pressure, temperature, a log curve) against a
depth axis is correct and must not be flagged: a lint that cries wolf gets
switched off, and then it protects nothing.

Usage::

    python -m welleng.lint src/            # exits 1 on findings
    python -m welleng.lint --quiet pkg/    # exit code only

In a consumer's suite::

    from welleng.lint import find_linear_survey_interpolation

    def test_no_linear_survey_interpolation():
        assert find_linear_survey_interpolation("src") == []
"""
from __future__ import annotations

import ast
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

__all__ = [
    "Finding",
    "TRAJECTORY_AXES",
    "find_linear_survey_interpolation",
    "main",
]

#: Attribute/variable names that ARE the trajectory. Interpolating one of these
#: linearly against MD is the defect; interpolating something else against one
#: of them as an axis is ordinary and correct.
TRAJECTORY_AXES = frozenset({
    "md", "mds", "measured_depth",
    "tvd", "tvds", "tvd_m", "true_vertical_depth",
    "inc", "incs", "inc_deg", "inc_rad", "inclination", "inclination_deg",
    "azi", "azis", "azi_deg", "azi_rad", "azimuth", "azimuth_deg",
    "n", "e", "northing", "easting", "pos_nev", "pos_xyz",
})

#: What to reach for instead. Keyed by the axis being interpolated.
_REPLACEMENT = {
    "tvd": "survey.interpolate_md(md).pos_nev[2] (or interpolate_mds for many)",
    "md": "survey.interpolate_tvd(tvd) -- closed form, returns EVERY crossing "
          "where TVD reverses, which a linear or bisection inverse cannot",
    "inc": "survey.interpolate_md(md).inc_deg -- an ANGLE on an arc",
    "azi": "survey.interpolate_md(md).azi_deg -- an ANGLE, and azimuth WRAPS "
           "at 0/360, so a linear blend across the wrap swings the tangent "
           "through the opposite heading",
}

_INTERP_FUNCS = {"interp"}          # np.interp / numpy.interp / bare interp

#: Explicit, greppable opt-out. A suppression you have to TYPE is a decision on
#: the record; a lint that quietly exempts a pattern is one nobody revisits.
#: Use it only where the interpolated points are already exact minimum-curvature
#: positions and the residual is stated.
NOQA = "# lint: trajectory-interp ok"


@dataclass(frozen=True)
class Finding:
    """One linear interpolation of a trajectory axis."""

    path: str
    line: int
    axis: str
    """The trajectory axis being interpolated (the ``fp`` argument)."""
    source: str
    """The offending source line, stripped."""

    @property
    def replacement(self) -> str:
        """What to use instead."""
        for key, text in _REPLACEMENT.items():
            if self.axis.startswith(key) or key in self.axis:
                return text
        return "the survey's own minimum-curvature interpolators"

    def __str__(self) -> str:
        return (f"{self.path}:{self.line}: linear interpolation of "
                f"{self.axis!r} -- a survey between stations is a circular "
                f"ARC.\n    {self.source}\n    use: {self.replacement}")


def _axis_name(node: ast.AST) -> Optional[str]:
    """The trajectory-axis name this expression yields, if it is one.

    Looks through ``np.asarray(...)``/``np.array(...)``/``list(...)`` and
    subscripts, because ``np.interp(x, xp, np.asarray(survey.tvd, float))`` is
    the same defect wearing a coat.
    """
    if isinstance(node, ast.Attribute):
        return node.attr if node.attr in TRAJECTORY_AXES else None
    if isinstance(node, ast.Name):
        return node.id if node.id in TRAJECTORY_AXES else None
    if isinstance(node, ast.Call):
        # np.asarray(survey.tvd, float) -> look at the first argument
        if node.args:
            return _axis_name(node.args[0])
        return None
    if isinstance(node, ast.Subscript):
        return _axis_name(node.value)
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: Sequence[str]) -> None:
        self.path = path
        self.lines = lines
        self.findings: List[Finding] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 (ast API)
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None)
        if name in _INTERP_FUNCS and len(node.args) >= 3:
            axis = _axis_name(node.args[2])          # fp -- the VALUE
            if axis is not None:
                src = ""
                if 0 < node.lineno <= len(self.lines):
                    src = self.lines[node.lineno - 1].strip()
                if NOQA in src:
                    self.generic_visit(node)
                    return
                self.findings.append(
                    Finding(path=self.path, line=node.lineno,
                            axis=axis, source=src)
                )
        self.generic_visit(node)


def find_linear_survey_interpolation(
    paths: Iterable[str] | str,
    exclude: Sequence[str] = (".venv", "site-packages", "build", ".git"),
) -> List[Finding]:
    """Every linear interpolation of a trajectory axis under ``paths``.

    ``paths`` may be a single path or an iterable of them; directories are
    walked for ``*.py``. Returns an empty list when clean, so a consumer's
    suite can assert on it directly.

    A file that does not parse is REPORTED, not skipped -- a lint that silently
    passes over what it could not read is indistinguishable from one that found
    nothing.
    """
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    out: List[Finding] = []
    for p in paths:
        root = pathlib.Path(p)
        files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for f in files:
            sp = str(f)
            if any(x in sp for x in exclude):
                continue
            try:
                text = f.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except (OSError, SyntaxError, UnicodeDecodeError) as exc:
                out.append(Finding(path=sp, line=0, axis="<unreadable>",
                                   source=f"could not parse: {exc}"))
                continue
            v = _Visitor(sp, text.splitlines())
            v.visit(tree)
            out.extend(v.findings)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m welleng.lint [--quiet] PATH...``. Exit 1 on findings."""
    args = list(sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in args
    args = [a for a in args if not a.startswith("-")]
    if not args:
        print("usage: python -m welleng.lint [--quiet] PATH...", file=sys.stderr)
        return 2
    findings = find_linear_survey_interpolation(args)
    if not quiet:
        for f in findings:
            print(f)
        print(f"\n{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
