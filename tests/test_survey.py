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
