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
depth axis is not flagged: a lint that cries wolf gets switched off, and then it
protects nothing.

Why that exemption is safe -- and what it does NOT say
-----------------------------------------------------
It is NOT that a property is exempt from the geometry. A log sample is a point in
space like any other, so the chord is wrong there too. The deviation of a chord
from the arc is the sagitta, ``R(1 - cos(kappa*h/2))``, which goes as
**(kappa*h)^2** -- so the STEP decides the magnitude, not the kind of data:

    at 4.16 deg/30 m (R = 413 m), chord-vs-arc at the midpoint of a step

        h = 1 m    3.0e-04 m          h = 20 m   1.2e-01 m
        h = 5 m    7.6e-03 m          h = 30 m   2.7e-01 m

The exemption is safe because of WHO OWNS THE ERROR, not because of the data
type. Interpolating the TRAJECTORY hands the caller a position this library is
responsible for, wrong by an amount set by *their* station spacing -- which this
repo does not control and no test here can fail on. Interpolating a property is
a modelling choice the caller makes about data the caller sampled, at a spacing
the caller can see.

⚠️ So the exemption transfers badly, and knowingly: a COARSELY sampled property
-- a 20-30 m formation-tops list, a sparse pressure survey, an MDT point set --
passes this lint while carrying 10-30 cm of position error. That cannot be
detected statically, because the step is a property of the caller's data and not
of the source text. If you are placing something in space from a sparse depth
axis, this lint is silent and it is not evidence that you are right.

(Magnitudes verified against the closed-form sagitta.)

Usage::

    python -m welleng.lint src/            # exits 1 on findings
    python -m welleng.lint --quiet pkg/    # exit code only

In a consumer's suite::

    from welleng.lint import find_linear_survey_interpolation

    def test_no_linear_survey_interpolation():
        assert find_linear_survey_interpolation("src") == []

⛔ **Do NOT write that test as ``skipif`` on ImportError.** A repo that cannot
import this check and skips has a check that cannot run reading as a pass.
Declare the dependency, or let the test error loudly. **A green suite must mean "clean",
never "not checked".**

⚠️ **What a clean run does and does not mean.** This walks PACKAGE SOURCE. It
does not see notebooks, loose scripts, or spec files, so **a pass says the
PACKAGE is clean, not the REPO.** Worth knowing before quoting a zero.

⭐ Why a check rather than a convention: this defect was fixed **nine times as
CHANGES**, and reminders and conventions were both tried. The check
found **twelve more instances in core the day it existed**, and later three in
a consumer that had never called a guarded function -- it reimplemented the
decision, which is invisible to any guard on the function. A corrected line is
invisible to the next author; a red suite is not.
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
    "n", "e", "northing", "easting", "pos_nev", "pos_xyz", "poss",
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

#: Suppresses the arc-tangent rule on one line. For a symbolic derivation or a
#: test oracle that states the published form (Sawaryn & Thorogood 2005, Eq. 13)
#: on purpose -- never for numeric code that runs.
SLERP_NOQA = "# lint: slerp ok"


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
        if self.axis == "slerp":
            return (f"{self.path}:{self.line}: hand-written arc tangent "
                    f"-- a second implementation of the arc kernel.\n"
                    f"    {self.source}\n"
                    f"    use: MinCurve.interpolate(md, angles=True)")
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
        # d["tvd"] -- the axis is in the KEY, not the object
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            if sl.value in TRAJECTORY_AXES:
                return sl.value
        return _axis_name(node.value)
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: Sequence[str]) -> None:
        self.path = path
        self.lines = lines
        self.findings: List[Finding] = []
        #: local name -> the trajectory axis it was bound FROM.
        #: ``tvd_s = np.asarray(survey.tvd, float)`` is the most natural line
        #: anyone writes, and matching the argument's NAME alone missed it:
        #: a consumer's three real instances scored ZERO until the variables
        #: were renamed, at which point the same file scored five. Name
        #: matching cannot be the whole rule.
        self.bound: dict = {}

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 (ast API)
        axis = _axis_name(node.value)
        if axis is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.bound[target.id] = axis
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is not None:
            axis = _axis_name(node.value)
            if axis is not None and isinstance(node.target, ast.Name):
                self.bound[node.target.id] = axis
        self.generic_visit(node)

    def _rebound_axis(self, node: Optional[ast.AST]) -> Optional[str]:
        """The axis a REBOUND local carries, if any. Looks through the same
        wrappers as :func:`_axis_name` so ``tvd_s[ok]`` resolves too."""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return self.bound.get(node.id)
        if isinstance(node, ast.Subscript):
            return self._rebound_axis(node.value)
        if isinstance(node, ast.Call) and node.args:
            return self._rebound_axis(node.args[0])
        return None

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 (ast API)
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None)
        if name in _INTERP_FUNCS:
            # `fp` may be positional (args[2]) OR a keyword. Reading args[2]
            # alone let `np.interp(q, md, fp=survey.tvd)` through untouched --
            # a STRUCTURAL miss, independent of what the argument is called.
            fp = None
            if len(node.args) >= 3:
                fp = node.args[2]
            for kw in node.keywords:
                if kw.arg == "fp":
                    fp = kw.value
            axis = _axis_name(fp) if fp is not None else None
            if axis is None:
                axis = self._rebound_axis(fp)
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


def _is_sin(node: ast.AST) -> bool:
    f = getattr(node, "func", None)
    name = getattr(f, "attr", None) or getattr(f, "id", None)
    return isinstance(node, ast.Call) and name == "sin" and len(node.args) == 1


def _is_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (1, 1.0)


def _complement(arg: ast.AST) -> bool:
    """``A - B``, or ``1 - f`` anywhere in the argument (``(1 - f) * A``,
    ``(1 - f)[:, None] * A``) -- the ``sin(alpha - alpha*)`` weight of a SLERP
    blend in its usual spellings."""
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Sub):
        return True
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
               and _is_one(n.left) for n in ast.walk(arg))


def _has_sin(node: ast.AST) -> bool:
    return any(_is_sin(n) for n in ast.walk(node))


def _has_cos(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and (getattr(n.func, "attr", None)
               or getattr(n.func, "id", None)) == "cos" for n in ast.walk(node))


#: The one place the arc tangent is written out: the kernel itself.
_KERNEL = ("welleng/utils.py", ("_arc_tangent", "_arc_inplane"))


class _SlerpVisitor(ast.NodeVisitor):
    """A hand-written arc tangent, in either spelling:

    - SLERP blend: ``sin(complement) ... / sin(...)``;
    - Rodrigues in-plane vector: ``(t2 - cos(a) * t1) / sin(a)``.

    The denominator may be the sine itself or a name assigned from an
    expression containing one (``s = np.where(straight, 1.0, np.sin(dl))``),
    optionally indexed. The kernel (``MinCurve``'s ``_arc_tangent``) is the one
    exempt place.
    """

    def __init__(self, path: str, lines: Sequence[str]) -> None:
        self.path, self.lines, self.findings = path, lines, []
        self._sin_names: set = set()
        self._func: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._func.append(node.name)
        self.generic_visit(node)
        self._func.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _in_kernel(self) -> bool:
        return (self.path.replace("\\", "/").endswith(_KERNEL[0])
                and any(f in _KERNEL[1] for f in self._func))

    def _flag(self, node: ast.AST) -> None:
        line = self.lines[node.lineno - 1]  # type: ignore[attr-defined]
        if SLERP_NOQA not in line and not self._in_kernel():
            self.findings.append(Finding(
                path=self.path, line=node.lineno,  # type: ignore[attr-defined]
                axis="slerp", source=line.strip()))

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802 (ast API)
        if _has_sin(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    self._sin_names.add(tgt.id)
        self.generic_visit(node)

    def _sin_denominator(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Subscript):
            node = node.value
        return _is_sin(node) or (isinstance(node, ast.Name)
                                 and node.id in self._sin_names)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802 (ast API)
        if isinstance(node.op, ast.Div) and self._sin_denominator(node.right):
            left = node.left
            if (isinstance(left, ast.BinOp) and isinstance(left.op, ast.Sub)
                    and isinstance(left.right, ast.BinOp)
                    and isinstance(left.right.op, ast.Mult)
                    and _has_cos(left.right)):
                # cos(a) TIMES a vector: ``1 - cos(x)`` (half-angle) is not one
                self._flag(node)                  # Rodrigues u-form
            else:
                for sub in ast.walk(left):
                    if _is_sin(sub) and _complement(sub.args[0]):  # type: ignore[attr-defined]
                        self._flag(node)          # SLERP blend
                        break
        self.generic_visit(node)


def find_slerp_arc_tangent(
    paths: Iterable[str] | str,
    exclude: Sequence[str] = (".venv", "site-packages", "build", ".git"),
) -> List[Finding]:
    """Every hand-written arc tangent under ``paths``, in either spelling.

    The SLERP blend ``(sin(a - d) t1 + sin(d) t2) / sin(a)`` and the Rodrigues
    in-plane vector ``(t2 - cos(a) t1) / sin(a)`` both re-implement the
    minimum-curvature arc tangent that :meth:`welleng.utils.MinCurve.interpolate`
    (``angles=True``) and :meth:`~welleng.utils.MinCurve.inc_azi_at` provide.
    A second implementation agrees today and drifts the first time either is
    changed. Call the ``MinCurve`` methods instead. The kernel itself is the
    one exempt place; ``(1 - cos x) / sin x`` (``tan(x/2)``) is not a tangent
    and is not flagged. The function name predates the Rodrigues form.

    Same contract as :func:`find_linear_survey_interpolation`: empty when
    clean; an unparseable file is reported, not skipped.
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
            v = _SlerpVisitor(sp, text.splitlines())
            v.visit(tree)
            out.extend(v.findings)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m welleng.lint [--quiet] PATH...``. Exit 1 on findings.

    Runs both rules: linear interpolation of a trajectory axis, and a
    hand-written arc tangent.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in args
    args = [a for a in args if not a.startswith("-")]
    if not args:
        print("usage: python -m welleng.lint [--quiet] PATH...", file=sys.stderr)
        return 2
    findings = find_linear_survey_interpolation(args) + find_slerp_arc_tangent(args)
    if not quiet:
        for f in findings:
            print(f)
        print(f"\n{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
