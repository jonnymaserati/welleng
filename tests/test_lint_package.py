"""This package contains no linear interpolation of a trajectory axis and no
hand-written SLERP arc tangent.

1. Fails, never skips, when ``welleng.lint`` cannot be imported -- a check that
   cannot run must not read as a pass.
2. Pins the rule set: a rule added to ``welleng.lint`` fails this test until it
   is asserted here too, because a tightened lint leaves earlier clean runs
   unverified.
3. Pins a known positive per rule, so a rule that is loosened, or matches on
   the wrong thing, fails here rather than going quietly green.

Bound: it lints package source, not notebooks, loose scripts or spec files. A
clean run means the package is clean, not the repo.
"""
import pathlib
import textwrap

# An ImportError here is a test ERROR, deliberately -- never a skip.
import welleng.lint as lint

REPO = pathlib.Path(__file__).resolve().parents[1]

# The rules this test asserts. Update BOTH lists together when welleng.lint
# gains a rule.
EXPECTED_RULES = {"find_linear_survey_interpolation", "find_slerp_arc_tangent"}

KNOWN_POSITIVES = {
    "find_linear_survey_interpolation": """
        import numpy as np
        def f(survey, q):
            tvd_s = np.asarray(survey.tvd, float)
            return np.interp(q, survey.md, tvd_s)
    """,
    "find_slerp_arc_tangent": """
        import numpy as np
        def f(t1, t2, total, d):
            return t1 * (np.sin(total - d) / np.sin(total))[:, None] + t2
    """,
}


def _package_dirs():
    """Top-level (or src/) package directories; never tests or venvs."""
    roots = [REPO / "src"] if (REPO / "src").is_dir() else [REPO]
    skip = {"tests", "test", ".venv", "venv", "build", "dist", "docs", "scripts",
            "notebooks"}
    found = [p for r in roots for p in r.iterdir()
             if p.is_dir() and p.name not in skip and (p / "__init__.py").exists()]
    assert found, (f"no package directory found under {roots} -- cannot lint, "
                   "refusing to pass")
    return found


def test_rule_set_is_pinned():
    present = {n for n in dir(lint)
               if n.startswith("find_") and callable(getattr(lint, n))}
    assert present == EXPECTED_RULES, (
        f"welleng.lint rules changed: new={sorted(present - EXPECTED_RULES)} "
        f"gone={sorted(EXPECTED_RULES - present)}. A TIGHTENED lint means earlier "
        f"green runs are unverified: add the rule here AND re-scan this repo.")


def test_each_rule_still_fires_on_a_known_positive(tmp_path):
    for rule, src in KNOWN_POSITIVES.items():
        f = tmp_path / f"{rule}.py"
        f.write_text(textwrap.dedent(src))
        assert getattr(lint, rule)(str(f)), (
            f"{rule} no longer fires on its known positive")


def test_package_is_clean():
    for pkg in _package_dirs():
        for rule in sorted(EXPECTED_RULES):
            hits = getattr(lint, rule)(str(pkg))
            assert hits == [], f"{rule} in {pkg.name}: {hits}"
