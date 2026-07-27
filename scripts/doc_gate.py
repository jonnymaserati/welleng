#!/usr/bin/env python3
"""DOC GATE — every new public API in a diff must be documented and reachable.

Run before a release (and ideally before a PR):

    python scripts/doc_gate.py                 # vs origin/main
    python scripts/doc_gate.py v0.26.0         # vs a tag
    python scripts/doc_gate.py --json          # machine-readable

WHAT IT CHECKS, for every PUBLIC name ADDED by the diff:

1. **Functions and classes have a docstring.** A public callable with no docstring
   is undocumented no matter what the release notes say.
2. **Dataclass fields are annotated.** These are public API too — they land in a
   caller's response model — and Sphinx renders nothing useful for a bare
   ``x: float``, so we require a ``#`` comment on or immediately after the field.
   This is the case that actually bit: three result flags shipped with the field
   list changing and nothing in the rendered docs describing them.
3. **The containing module is in the Sphinx build.** A perfect docstring in a
   module no ``.rst`` automodules is invisible on the published site.

WHY THIS EXISTS (TA0, 2026-07-27). In the 0.27 cycle new public surface was added
several times — three result flags, then ``maasp`` — and each time the docs were
brought up to date only because someone asked. The pattern this release keeps
re-learning is that **a check nobody runs is indistinguishable from a check that
passes**, so this is executable and exits non-zero rather than being a line in a
checklist.

Deliberately NOT clever: it parses the diff for added lines and uses ``ast`` on the
new file content. It will not catch a name added by metaprogramming, and it does not
try to. Exit 0 = pass, 1 = findings, 2 = could not run (which is a FAILURE to
report, not a pass).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = "welleng"
DOCS = REPO / "docs"


def _run(*args: str) -> str:
    out = subprocess.run(args, cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def _merge_base(base: str) -> str:
    """Fork point, so commits landed on `base` since we branched are not read as
    deletions on our side."""
    try:
        return _run("git", "merge-base", base, "HEAD").strip()
    except RuntimeError:
        return base


def changed_py_files(base: str) -> list[str]:
    """Files changed vs `base` INCLUDING the working tree.

    Deliberately diffs against the working tree rather than HEAD: a doc gate that
    only sees committed changes cannot catch the thing you just wrote, which is
    exactly when you want to be told. (First cut used `base...HEAD` and silently
    passed an undocumented function sitting uncommitted in the tree.)
    """
    diff = _run("git", "diff", "--name-only", _merge_base(base), "--", f"{PKG}/")
    return [f for f in diff.split() if f.endswith(".py")]


def added_lines(base: str, path: str) -> set[str]:
    """The literal added lines for one file, stripped. Includes the working tree."""
    try:
        d = _run("git", "diff", "-U0", _merge_base(base), "--", path)
    except RuntimeError:
        return set()
    return {
        ln[1:].strip()
        for ln in d.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    }


def sphinx_modules() -> set[str]:
    """Every module named in an ``automodule`` directive."""
    mods: set[str] = set()
    for rst in DOCS.rglob("*.rst"):
        for m in re.finditer(r"^\.\.\s+automodule::\s+([\w.]+)", rst.read_text(), re.M):
            mods.add(m.group(1))
    return mods


def module_covered(modname: str, covered: set[str]) -> bool:
    """A module is covered directly, or by a parent package's automodule."""
    parts = modname.split(".")
    return any(".".join(parts[: i + 1]) in covered for i in range(len(parts)))


def is_public(name: str) -> bool:
    return not name.startswith("_")


def audit(base: str) -> list[dict]:
    covered = sphinx_modules()
    findings: list[dict] = []

    for path in changed_py_files(base):
        p = REPO / path
        if not p.exists():                       # deleted in this diff
            continue
        added = added_lines(base, path)
        if not added:
            continue
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError as e:
            findings.append({"file": path, "name": "<module>", "kind": "parse",
                             "problem": f"could not parse: {e}"})
            continue

        modname = path[:-3].replace("/", ".")
        mod_ok = module_covered(modname, covered)

        for node in ast.walk(tree):
            # --- functions and classes -------------------------------------
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not is_public(node.name):
                    continue
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                sig = f"class {node.name}" if kind == "class" else f"def {node.name}("
                if not any(ln.startswith(sig) for ln in added):
                    continue                      # not added by THIS diff
                if not ast.get_docstring(node):
                    findings.append({"file": path, "name": node.name, "kind": kind,
                                     "line": node.lineno,
                                     "problem": "public, added, and has NO docstring"})
                if not mod_ok:
                    findings.append({"file": path, "name": node.name, "kind": kind,
                                     "line": node.lineno,
                                     "problem": f"module {modname} has no Sphinx "
                                                f"automodule -- the docs will never "
                                                f"render this"})

            # --- dataclass fields ------------------------------------------
            if isinstance(node, ast.ClassDef) and is_public(node.name):
                deco = {ast.unparse(d).split("(")[0] for d in node.decorator_list}
                if "dataclass" not in {d.split(".")[-1] for d in deco}:
                    continue
                src = p.read_text().splitlines()
                for item in node.body:
                    if not isinstance(item, ast.AnnAssign) or not isinstance(
                            item.target, ast.Name):
                        continue
                    fname = item.target.id
                    if not is_public(fname):
                        continue
                    decl = src[item.lineno - 1]
                    if not any(ln.startswith(f"{fname}:") for ln in added):
                        continue                  # not added by THIS diff
                    trailing = "#" in decl.split(":", 1)[-1]
                    following = (item.end_lineno < len(src)
                                 and src[item.end_lineno].strip().startswith("#"))
                    if not (trailing or following):
                        findings.append(
                            {"file": path, "name": f"{node.name}.{fname}",
                             "kind": "dataclass field", "line": item.lineno,
                             "problem": "public dataclass field added with no "
                                        "explanatory comment -- it lands in a "
                                        "consumer's response model undocumented"})
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", nargs="?", default="origin/main",
                    help="ref to diff against (default: origin/main)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    try:
        _run("git", "rev-parse", "--verify", a.base)
    except RuntimeError:
        print(f"!! DOC GATE: cannot resolve base ref {a.base!r}. NOT a pass -- fix "
              f"the ref and re-run.", file=sys.stderr)
        return 2
    if not DOCS.exists():
        print("!! DOC GATE: no docs/ directory. NOT a pass.", file=sys.stderr)
        return 2

    try:
        findings = audit(a.base)
    except RuntimeError as e:
        print(f"!! DOC GATE: could not run ({e}). NOT a pass.", file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(findings, indent=2))
        return 1 if findings else 0

    print(f"== DOC GATE: {PKG} vs {a.base} ==")
    if not findings:
        print("  ok: every public function, class and dataclass field added by this "
              "diff is documented and inside the Sphinx build")
        return 0
    print(f"  !! {len(findings)} finding(s):\n")
    for f in findings:
        loc = f"{f['file']}:{f.get('line', '?')}"
        print(f"  {loc}\n      {f['kind']} {f['name']}\n      {f['problem']}\n")
    print("  >> add the docstring/comment, and if a module is missing from Sphinx add "
          "an automodule entry under docs/.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
