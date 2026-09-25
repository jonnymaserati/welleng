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


class TestCoverageMatrix:
    """The shapes the rule does and does not reach, pinned.

    Origin: a consumer wired this check against a file with three CONFIRMED
    instances and got ZERO. The same file, with a `_s` suffix dropped from
    three variable names, scored five. Matching the argument's NAME cannot be
    the whole rule, and a green run that is green only because of a naming
    accident is worse than no run -- it is believed, because this check is the
    remedy for a famous defect.

    ⭐ Every CAUGHT case below is a known positive. Without them a zero from
    this module is indistinguishable from the module doing nothing, which is
    exactly what it was doing.
    """

    @staticmethod
    def _findings(src: str) -> int:
        import tempfile
        from welleng.lint import find_linear_survey_interpolation
        with tempfile.TemporaryDirectory() as d:
            pathlib.Path(d, "m.py").write_text(src)
            return len(find_linear_survey_interpolation(d))

    @pytest.mark.parametrize("src", [
        "import numpy as np\nnp.interp(q, md, survey.tvd)\n",
        "import numpy as np\nnp.interp(q, md, np.asarray(survey.tvd, float))\n",
        "import numpy as np\nnp.interp(q, md, tvd[ok])\n",
        "import numpy as np\nnp.interp(q, md, pos_nev[:, 2])\n",
        # structural: fp as a KEYWORD bypassed args[2] entirely
        "import numpy as np\nnp.interp(q, md, fp=survey.tvd)\n",
        # what MinCurve actually returns -- the shape a consumer is likeliest
        # to write was the one slipping through
        "import numpy as np\nnp.interp(q, md, poss[:, 2])\n",
        'import numpy as np\nnp.interp(q, md, d["tvd"])\n',
        # local rebinding -- the case that scored zero on real defective code
        "import numpy as np\ntvd_s = np.asarray(survey.tvd, float)\n"
        "np.interp(q, md, tvd_s)\n",
        "import numpy as np\nn_s = np.asarray(survey.n, float)\n"
        "np.interp(q, md, n_s)\n",
        "import numpy as np\nz = survey.tvd\nnp.interp(q, md, fp=z)\n",
    ])
    def test_caught(self, src):
        assert self._findings(src) >= 1, f"MISSED:\n{src}"

    @pytest.mark.parametrize("src", [
        # a PROPERTY against a depth axis is correct and must not be flagged --
        # a lint that cries wolf gets switched off
        "import numpy as np\nnp.interp(tvd, self.tvd, self.sigma_V)\n",
        "import numpy as np\nsigma = np.asarray(model.sigma_V)\n"
        "np.interp(tvd, self.tvd, sigma)\n",
    ])
    def test_not_flagged(self, src):
        assert self._findings(src) == 0, f"FALSE POSITIVE:\n{src}"

    def test_known_miss_is_pinned_not_forgotten(self):
        """A name bound from something the rule cannot trace still escapes.

        Pinned as an EXPECTED miss so that if the rule is later tightened this
        test fails and says so. ⭐ A silently-tightened lint leaves every
        earlier green run unverified -- a previous clean scan was clean only
        against the looser rule, and nobody is told to re-scan.
        """
        src = ("import numpy as np\n"
               "def f(rows):\n"
               "    depths = [r.tvd for r in rows]\n"   # comprehension: untraced
               "    return np.interp(q, md, depths)\n")
        assert self._findings(src) == 0, (
            "the rule now reaches list comprehensions -- GOOD, but every repo "
            "scanned under the looser rule must be re-scanned; update this pin"
        )
