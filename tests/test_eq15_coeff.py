"""``_eq15_coeff`` — one Eq. 15 coefficient without paying for the other ten.

The critical-radius solve needs only ``c0``. Evaluating all eleven cost a
c0-only caller 2.4x (welleng-api: 33.3 vs 14.0 ms at N = 10,000; 1004.6 vs
386.5 ms at N = 200,000), which was 90% of their batched critical radius when
they first built it. They had worked around it by extracting ``c0`` into
generated modules of their own that had to be regenerated on every pin bump.

The eleven expressions are machine-generated ~4000-term resultant expansions
that no human edits by hand, so splitting them out of the single ``return [...]``
was SCRIPTED and verified rather than typed. This module is that verification:
``_eq15_coeff(k, ...)`` must equal ``_eq15_coeffs(...)[k]`` BIT-EXACTLY, scalar
and array, or the split corrupted a coefficient.
"""

import numpy as np
import pytest

from welleng.sawaryn_analytical import _eq15_coeff, _eq15_coeffs


def _invariants(rng, n=None):
    """Scale-normalised invariants (psi2 = 1) as the solver evaluates them."""
    size = () if n is None else n
    return [
        np.ones(size) if n else 1.0,
        rng.uniform(-2.0, 2.0, size),      # g1 = eta1 / L
        rng.uniform(-2.0, 2.0, size),      # g4 = eta4 / L
        rng.uniform(-0.99, 0.99, size),    # l  = mu
        rng.uniform(0.05, 50.0, size),     # R1 / L
        rng.uniform(0.05, 50.0, size),     # R2 / L
    ]


@pytest.mark.parametrize("k", range(11))
def test_single_coefficient_is_bit_exact_scalar(k):
    rng = np.random.default_rng(11)
    for _ in range(200):
        args = _invariants(rng)
        assert _eq15_coeff(k, *args) == _eq15_coeffs(*args)[k]


@pytest.mark.parametrize("k", range(11))
def test_single_coefficient_is_bit_exact_over_arrays(k):
    """The solver evaluates these broadcast over poses, so the array path is the
    one that ships."""
    rng = np.random.default_rng(5)
    args = _invariants(rng, 1000)
    assert np.array_equal(
        np.asarray(_eq15_coeff(k, *args)),
        np.asarray(_eq15_coeffs(*args)[k]),
    )


def test_out_of_range_raises_rather_than_returning_none():
    """A silent None here would propagate into a polynomial solve as a NaN."""
    args = _invariants(np.random.default_rng(0))
    for k in (-1, 11, 99):
        with pytest.raises(IndexError, match="c0..c10"):
            _eq15_coeff(k, *args)


def test_c0_only_is_cheaper_than_all_eleven():
    """The point of the entry point. Not a tight bound -- just proof that a
    c0-only caller no longer pays for c1..c10."""
    import timeit

    rng = np.random.default_rng(1)
    args = _invariants(rng, 20_000)
    all_11 = timeit.timeit(lambda: _eq15_coeffs(*args), number=3) / 3
    c0_only = timeit.timeit(lambda: _eq15_coeff(0, *args), number=3) / 3
    assert c0_only < 0.75 * all_11, f"c0 {c0_only:.4f}s vs all {all_11:.4f}s"
