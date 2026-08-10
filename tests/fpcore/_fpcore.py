"""Minimal FPCore reader + dual-backend evaluator for the conditioning gate.

FPCore (Herbie / FPBench) is a Lisp-y format: each kernel is
``(FPCore name (args...) :name "..." :pre (...) body-expr)``. We parse the
kernels out of ``welleng.fpcore`` (contributed via welleng#307), evaluate each
body expression TWICE -- once at 200-bit ``mpmath`` (the oracle, i.e. the math
the expression *means*) and once in float64 (the naive transcription) -- and
sample its declared ``:pre`` domain. The conditioning test then compares
welleng's own implementation of the same quantity against the mpmath oracle.

This is deliberately tiny and dependency-light: only the handful of operators
the welleng kernels use (+ - * / sqrt pow sin cos asin acos atan2) are wired.
"""
import re

import mpmath as mp

# ---------------------------------------------------------------------------
# S-expression tokeniser + parser
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r'"[^"]*"|[()]|[^\s()]+')


def _strip_comments(text):
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def _tokenize(text):
    return _TOKEN.findall(_strip_comments(text))


def _parse_all(tokens):
    """Parse a flat token list into a list of nested (list/atom) S-expressions."""
    pos = 0

    def parse():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out = []
            while tokens[pos] != ")":
                out.append(parse())
            pos += 1  # consume ')'
            return out
        return tok

    forms = []
    while pos < len(tokens):
        forms.append(parse())
    return forms


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------
class Kernel:
    def __init__(self, name, args, pretty, bounds, body):
        self.name = name            # symbol, e.g. "theta_chord"
        self.args = args            # ordered arg symbols
        self.pretty = pretty        # :name string
        self.bounds = bounds        # {arg: (lo, hi)} from :pre
        self.body = body            # parsed S-expression


def _num(tok):
    return mp.mpf(tok)


def _collect_bounds(pre, args):
    """Extract per-arg [lo, hi] from a ``(and (<= lo x hi) ...)`` precondition.

    Handles the two shapes used in the file: ``(<= lo x hi)`` (three operands)
    and ``(< lo x hi)``. Args not individually bounded (e.g. appear only inside
    a derived expression) are left out and must be supplied by the sampler.
    """
    bounds = {}

    def walk(node):
        if not isinstance(node, list):
            return
        if node and node[0] in ("<=", "<") and len(node) == 4:
            lo, var, hi = node[1], node[2], node[3]
            if isinstance(var, str) and var in args:
                bounds[var] = (float(_num(lo)), float(_num(hi)))
        for child in node:
            walk(child)

    walk(pre)
    return bounds


def parse_kernels(path):
    text = open(path).read()
    forms = _parse_all(_tokenize(text))
    kernels = {}
    for form in forms:
        if not (isinstance(form, list) and form and form[0] == "FPCore"):
            continue
        name = form[1]
        args = list(form[2])
        # remaining: alternating :key value ... then the final body expression
        rest = form[3:]
        pretty, pre, body = name, None, None
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == ":name":
                pretty = rest[i + 1].strip('"')
                i += 2
            elif tok == ":pre":
                pre = rest[i + 1]
                i += 2
            else:
                body = rest[i]      # the body is the lone non-keyword form
                i += 1
        bounds = _collect_bounds(pre, args) if pre is not None else {}
        kernels[name] = Kernel(name, args, pretty, bounds, body)
    return kernels


# ---------------------------------------------------------------------------
# Evaluation (backend-parameterised: mpmath oracle or float64)
# ---------------------------------------------------------------------------
def _make_ops(m):
    return {
        "+": lambda *a: sum(a[1:], a[0]),
        "*": lambda *a: _prod(a),
        "-": lambda *a: (-a[0] if len(a) == 1 else a[0] - a[1]),
        "/": lambda x, y: x / y,
        "sqrt": m["sqrt"], "pow": lambda x, y: m["pow"](x, y),
        "sin": m["sin"], "cos": m["cos"],
        "asin": m["asin"], "acos": m["acos"], "atan2": m["atan2"],
    }


def _prod(a):
    out = a[0]
    for x in a[1:]:
        out = out * x
    return out


_MP = {"sqrt": mp.sqrt, "pow": mp.power, "sin": mp.sin, "cos": mp.cos,
       "asin": mp.asin, "acos": mp.acos, "atan2": mp.atan2}


def eval_mp(kernel, values, prec=200):
    """Evaluate the kernel body at ``prec`` bits given {arg: python float}."""
    with mp.workprec(prec):
        ops = _make_ops(_MP)
        env = {a: mp.mpf(values[a]) for a in kernel.args}

        def ev(node):
            if isinstance(node, list):
                fn = ops[node[0]]
                return fn(*[ev(c) for c in node[1:]])
            if node in env:
                return env[node]
            return mp.mpf(node)          # numeric literal

        return ev(kernel.body)


def ulp_err(approx, oracle, scale):
    """Absolute error of ``approx`` vs the high-precision ``oracle``, in ulps of
    ``scale`` (a representative float64 magnitude for the quantity)."""
    import numpy as np
    diff = abs(mp.mpf(approx) - mp.mpf(oracle))
    return float(diff) / np.spacing(abs(scale) if scale else 1.0)
