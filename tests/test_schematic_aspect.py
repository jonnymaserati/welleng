"""``to_matplotlib(aspect=)``: equal by default; ``"auto"`` keeps the caller's box."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from welleng.schematic.backends import to_matplotlib  # noqa: E402
from welleng.schematic.drawing import Drawing, Rect  # noqa: E402


def _tall_drawing():
    d = Drawing()
    d.add(Rect(corner=(0.0, 0.0), width=1.0, height=10.0))
    return d


def _box_after_render(**kw):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    before = ax.get_position().bounds
    to_matplotlib(_tall_drawing(), ax=ax, **kw)
    fig.canvas.draw()
    after = ax.get_position().bounds
    aspect = ax.get_aspect()
    plt.close(fig)
    return np.array(before), np.array(after), aspect


def test_default_is_equal_and_resizes_the_box():
    """A 1 x 10 drawing in a 6 x 4 figure: equal aspect must narrow the box."""
    before, after, aspect = _box_after_render()
    assert aspect == 1.0
    assert after[2] < 0.5 * before[2]      # width given up to hold the ratio


def test_auto_keeps_the_callers_box():
    before, after, aspect = _box_after_render(aspect="auto")
    assert aspect == "auto"
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-12)
