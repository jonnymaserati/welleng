"""ISCWSA error-model formula-string interpreter — spike implementation.

This is the interpreter for the new ISCWSA JSON error-model schema
(github.com/iscwsa/error-models, schemas/draft.json), where each
error term carries its weight function as Excel-style formula strings
(``"Sin(Inc) / Gfield"``, ``"Abs(Cos(Inc)) * Cos(AzT)"``) rather than
references to hand-coded Python functions.

Status: spike. Goal of this iteration is to prove the architecture by
reproducing the existing welleng weight-function output to machine
precision on three trivial-to-moderate terms (DRFR, DSFS, ABZ) on the
ISCWSA standard test wells. Once that holds, the interpreter is
expanded to cover the full operator set the OWSG xlsx + ISCWSA JSON
schemas require.

Safety: formulas are parsed via ``ast.parse(..., mode='eval')`` and
walked with a whitelist of node types. No ``eval()`` of raw input.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Excel-style → Python-style formula text rewriting
# ---------------------------------------------------------------------------

_EXCEL_FUNCS = {
    "Sin": "sin", "Cos": "cos", "Tan": "tan",
    "Asin": "asin", "Acos": "acos", "Atan": "atan",
    "Abs": "abs",        # built-in, fine
    "Sqr": "sqrt",       # Excel/VBA convention: Sqr = square root
    "Exp": "exp",
    "Log": "log",
}


def _rewrite_excel_to_python(formula: str) -> str:
    """Make an Excel-style formula evaluable as Python.

    - Function names mapped to Python equivalents (Sin → sin, Sqr → sqrt …).
    - Power operator ``^`` → ``**``.
    - Bare ``pi`` left alone (we provide it in the namespace).
    - Variable / function names are case-sensitive in Python, so we map
      conservatively only the known Excel function set above; everything
      else is passed through and resolved at evaluation time.
    """
    s = formula
    # Function-name substitutions: only when followed by '(' so we don't
    # rewrite inside identifiers (e.g. "MinIncidence" survives).
    for excel, py in _EXCEL_FUNCS.items():
        s = re.sub(rf"\b{excel}\s*\(", f"{py}(", s)
    # Power operator
    s = s.replace("^", "**")
    return s


# ---------------------------------------------------------------------------
# Whitelisted-AST evaluator
# ---------------------------------------------------------------------------

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Constant,
    ast.Name, ast.Load,
    ast.Call,
)
_ALLOWED_FUNCS = {"sin", "cos", "tan", "asin", "acos", "atan",
                  "abs", "sqrt", "exp", "log", "pi"}


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"disallowed node {type(node).__name__} in formula"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only direct function calls allowed")
            if node.func.id not in _ALLOWED_FUNCS:
                raise ValueError(f"function {node.func.id!r} not whitelisted")
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            raise ValueError("__builtins__ access forbidden")


def evaluate_formula(
    formula: str | int | float,
    bindings: dict[str, Any],
) -> Any:
    """Evaluate one formula string in the given variable namespace.

    ``bindings`` provides scalar or vector values for variables (Inc,
    AzT, Gfield, MD, etc.). NumPy ufuncs (``np.sin``…) handle vector
    arguments naturally, so passing arrays for ``Inc``/``AzT``/``MD``
    yields per-station results in one call.
    """
    if isinstance(formula, (int, float)):
        return float(formula)
    s = _rewrite_excel_to_python(formula)
    tree = ast.parse(s, mode="eval")
    _validate(tree)
    # Build the eval namespace: numpy ufuncs for trig (vectorised),
    # math.pi for pi, and the user bindings on top.
    ns: dict[str, Any] = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
        "pi": math.pi,
    }
    ns.update(bindings)
    # Eval-on-whitelisted-AST: compile and call eval against the
    # restricted namespace. Safe because _validate ensures only the
    # allowed node/name set is present.
    code = compile(tree, "<formula>", "eval")
    return eval(code, {"__builtins__": {}}, ns)
