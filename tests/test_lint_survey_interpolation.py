"""The lint, and core held to it.

The guard exists because this defect class cannot be caught numerically by the
repo that commits it: the error scales with the CALLER's station spacing and
dogleg, so every fixture passes. See :mod:`welleng.lint`.
"""
import pathlib

import pytest

from welleng.lint import (
    NOQA, Finding, find_linear_survey_interpolation, main,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_core_does_not_linearly_interpolate_a_trajectory():
    """Core shipped 12 of these -- in the schematic module, which ignored
    core's OWN welleng.survey.interpolate_mds. A lint its author fails is not
    a standard, it is an opinion."""
    findings = find_linear_survey_interpolation(str(ROOT / "welleng"))
    assert findings == [], "\n".join(str(f) for f in findings)


@pytest.mark.parametrize("code,flagged", [
    ("np.interp(md, survey.md, survey.tvd)", True),
    ("np.interp(u, p_md, np.asarray(parent.inc, float))", True),
    ("np.interp(u, p_md, parent.azi)", True),
    ("np.interp(mid[2], np.asarray(t.tvd), np.asarray(t.md))", True),
    ("np.interp(m, grid, self.northing)", True),
    # the ordinary, correct thing sitting right next to it
    ("np.interp(t, self.tvd, self.sigma_V)", False),
    ("np.interp(tvd_m, self.tvd, self.temperature)", False),
    ("np.interp(d, tr.tvd, tr.pressure)", False),
    ("np.interp(x, xp, fp)", False),
])
def test_it_separates_the_trajectory_from_a_property(tmp_path, code, flagged):
    """The distinction that decides whether anyone keeps the lint switched on:
    interpolating a PROPERTY against a depth axis is correct."""
    f = tmp_path / "m.py"
    f.write_text(f"import numpy as np\nx = {code}\n")
    assert bool(find_linear_survey_interpolation(str(tmp_path))) is flagged


def test_the_angle_case_a_regex_guard_missed(tmp_path):
    """A consumer's regex guard looked for inc_deg/inc_rad and walked past
    `.inc`. Matching on the ATTRIBUTE, not the spelling, is the point of
    doing this on the AST."""
    f = tmp_path / "m.py"
    f.write_text("import numpy as np\nupper_inc = np.interp(u, p_md, parent.inc)\n")
    got = find_linear_survey_interpolation(str(tmp_path))
    assert len(got) == 1 and got[0].axis == "inc"
    assert "ANGLE" in got[0].replacement


def test_a_suppression_must_be_written_out(tmp_path):
    """Suppression is explicit and greppable. A lint that exempts a pattern
    silently is one nobody ever revisits."""
    f = tmp_path / "m.py"
    f.write_text(f"import numpy as np\nx = np.interp(m, g, s.tvd)  {NOQA}\n")
    assert find_linear_survey_interpolation(str(tmp_path)) == []


def test_an_unparseable_file_is_reported_not_skipped(tmp_path):
    """A gate that silently passes over what it could not read is
    indistinguishable from one that found nothing."""
    (tmp_path / "broken.py").write_text("def (:\n")
    got = find_linear_survey_interpolation(str(tmp_path))
    assert len(got) == 1 and got[0].axis == "<unreadable>"


def test_the_finding_names_the_replacement():
    f = Finding(path="x.py", line=1, axis="md", source="")
    assert "interpolate_tvd" in f.replacement
    assert "crossing" in f.replacement          # why bisection is not enough


def test_cli_exit_codes(tmp_path, capsys):
    (tmp_path / "ok.py").write_text("import numpy as np\nx = np.interp(a, b, c)\n")
    assert main([str(tmp_path), "--quiet"]) == 0
    (tmp_path / "bad.py").write_text(
        "import numpy as np\nx = np.interp(a, s.md, s.tvd)\n")
    assert main([str(tmp_path), "--quiet"]) == 1
    assert main([]) == 2
