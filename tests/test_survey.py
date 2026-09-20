"""Tests for `welleng.survey` that are about the MODULE, not the maths.

The trajectory maths is covered across the suite by the cases that use it. What
lives here is the module's contract as a dependency: what importing it costs a
consumer, and how it behaves when an optional dependency is absent.
"""
import builtins
import subprocess
import sys

import numpy as np
import pytest

from welleng.survey import Survey, export_csv


class TestPandasIsOptional:
    """pandas is a CONVENIENCE in survey.py -- two DataFrame exporters and
    nothing in the trajectory maths. It must not be a hard import, or a
    consumer wanting only minimum curvature inherits it.

    Origin: `import welleng.survey` pulled pandas eagerly, which ALSO made
    export_csv's own `except ImportError` unreachable -- a guard that could
    never fire, because the import it guarded had already succeeded above.
    """

    @staticmethod
    def _without_pandas():
        real = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "pandas" or name.startswith("pandas."):
                raise ImportError("pandas blocked for this test")
            return real(name, *args, **kwargs)
        return real, blocked

    def test_trajectory_path_does_not_import_pandas(self):
        # a subprocess, because pandas is already in sys.modules for this suite
        code = (
            "import sys; import welleng.survey, welleng.utils; "
            "print('pandas' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False", (
            "importing welleng.survey pulled pandas -- it is a convenience "
            "dependency of two exporters, not of the trajectory maths"
        )

    def test_export_csv_refuses_clearly_without_pandas(self, monkeypatch):

        survey = Survey(md=np.array([0., 30.]), inc=np.array([0., 3.]),
                        azi=np.array([0., 45.]))
        real, blocked = self._without_pandas()
        monkeypatch.setattr(builtins, "__import__", blocked)
        # filename=None means "give me a DataFrame". Without pandas there is no
        # answer, so it must REFUSE -- previously it printed and fell through to
        # np.savetxt(None), failing with "fname must be a string or file handle",
        # an error about something else entirely.
        with pytest.raises(ImportError, match="pandas is not installed"):
            export_csv(survey, None)


class TestDeferredConnectorImport:
    """`Connector` is imported inside four helpers, not at module level.

    connector -> sawaryn_analytical -> scipy.optimize, which the trajectory
    maths does not need. Two of the four helpers -- get_node_tvd and
    project_to_target -- had NO test coverage at all when the imports were
    moved, so a misplaced import (inside a docstring, say) would have gone
    unnoticed. These execute the deferred path.
    """

    @staticmethod
    def _survey():
        return Survey(
            md=np.array([0., 30., 60., 90.]),
            inc=np.array([0., 3., 6., 9.]),
            azi=np.array([0., 45., 45., 45.]),
        )

    def test_module_does_not_import_connector_or_scipy_optimize(self):
        code = (
            "import sys; import welleng.survey; "
            "print('connector' in ''.join(sys.modules), "
            "'scipy.optimize' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, check=True)
        assert out.stdout.split()[-1] == "False", (
            "importing welleng.survey pulled scipy.optimize -- it reaches the "
            "module only via connector, which the trajectory maths does not need"
        )

    @pytest.mark.parametrize("call", ["get_node_tvd", "project_to_target"])
    def test_deferred_import_resolves(self, call):
        """The risk is a NameError, not a wrong answer.

        An import misplaced inside a docstring is inert text: the module still
        imports, the suite still passes, and the function fails at runtime with
        `NameError: Connector`. So the assertion is narrow and exact -- these
        may raise on the inputs below, but never NameError.
        """
        from welleng.node import Node
        from welleng import survey as S

        survey = self._survey()
        n1 = Node(pos=[0., 0., 0.], vec=[0., 0., 1.], md=0.)
        n2 = Node(pos=[0., 0., 60.], vec=[0., 0., 1.], md=60.)
        args = {
            "get_node_tvd": (survey, n1, n2, 30.0, n1),
            "project_to_target": (survey, n2),
        }[call]
        try:
            getattr(S, call)(*args)
        except NameError as exc:                      # the failure under test
            pytest.fail(f"{call} has an unresolved deferred import: {exc}")
        except Exception:
            # any other failure is about the inputs, not the import
            pass


class TestFromMinCurve:
    """MinCurve is the geometry; Survey is that plus a header.

    Survey subclasses MinCurve, so promotion is adding the missing data -- but
    the two disagree on units, and getting that wrong flattens a well instead
    of raising.
    """

    @staticmethod
    def _profile():
        md = np.arange(0., 901., 30.)
        inc = np.clip((md - 300.) * 3.0 / 30.0, 0., 90.)
        azi = np.full_like(md, 45.)
        return md, inc, azi

    def test_promotion_is_identical_to_building_directly(self):
        from welleng.utils import MinCurve
        md, inc, azi = self._profile()
        mc = MinCurve(md, np.radians(inc), np.radians(azi))

        promoted = Survey.from_min_curve(mc)
        direct = Survey(md=md, inc=inc, azi=azi)

        for attr in ("md", "inc_rad", "azi_grid_rad", "n", "e", "tvd"):
            np.testing.assert_array_equal(
                np.asarray(getattr(promoted, attr)),
                np.asarray(getattr(direct, attr)),
                err_msg=f"{attr} differs between promotion and direct build",
            )

    def test_geometry_survives_the_promotion(self):
        from welleng.utils import MinCurve
        md, inc, azi = self._profile()
        mc = MinCurve(md, np.radians(inc), np.radians(azi))
        promoted = Survey.from_min_curve(mc)
        np.testing.assert_array_equal(promoted.inc_rad, mc.inc)
        # MinCurve.poss is env; Survey.pos_nev is nev -- same numbers, swapped
        np.testing.assert_allclose(
            np.asarray(promoted.pos_nev)[:, [1, 0, 2]], mc.poss, atol=1e-12)

    def test_radians_are_not_silently_read_as_degrees(self):
        """The failure this guards is a QUIET one.

        MinCurve is radians, Survey defaults to degrees. Hand the radians over
        without deg=False and a 60 degree hold becomes 1.05 degrees -- the well
        flattens, every value stays in range, and nothing raises.
        """
        from welleng.utils import MinCurve
        md, inc, azi = self._profile()
        mc = MinCurve(md, np.radians(inc), np.radians(azi))

        correct = Survey.from_min_curve(mc)
        assert np.degrees(correct.inc_rad[-1]) == pytest.approx(60.0, abs=1e-9)

        naive = Survey(md=mc.md, inc=mc.inc, azi=mc.azi)      # deg defaults True
        assert np.degrees(naive.inc_rad[-1]) == pytest.approx(1.047, abs=1e-3)

    def test_header_kwargs_reach_the_survey(self):
        from welleng.utils import MinCurve
        md, inc, azi = self._profile()
        mc = MinCurve(md, np.radians(inc), np.radians(azi))
        promoted = Survey.from_min_curve(mc, start_nev=[100., 200., 50.])
        assert promoted.n[0] == pytest.approx(100.)
        assert promoted.tvd[0] == pytest.approx(50.)
