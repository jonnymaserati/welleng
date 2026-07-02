"""Guard: every public welleng module is included in the Sphinx API docs.

Catches the case where a new module is added but never wired into docs/*.rst,
so it silently never appears in the published API documentation (as happened for
welleng.sawaryn_analytical in v0.14.0).
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "welleng"
DOCS = ROOT / "docs"

# Modules intentionally NOT in the public API docs (internal / metadata only).
EXCLUDE = {"__init__", "version", "_version"}


def _documented_modules():
    """All `welleng.<x>` names with an `.. automodule::` in any docs/*.rst."""
    documented = set()
    for rst in DOCS.glob("*.rst"):
        for m in re.finditer(r"^\.\.\s+automodule::\s+welleng\.([A-Za-z0-9_]+)",
                              rst.read_text(), re.MULTILINE):
            documented.add(m.group(1))
    return documented


@pytest.mark.skipif(not DOCS.exists(), reason="docs/ not present")
def test_every_public_module_is_documented():
    modules = {p.stem for p in PKG.glob("*.py")
               if not p.stem.startswith("_") and p.stem not in EXCLUDE}
    documented = _documented_modules()
    missing = sorted(modules - documented)
    assert not missing, (
        "welleng modules missing an `.. automodule::` in docs/*.rst "
        f"(add them so they appear in the API docs): {missing}"
    )
