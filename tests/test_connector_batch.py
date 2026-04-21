"""Tests for the vectorised ``solve_curve_hold_batch`` helper.

Verifies that the batch solver's outputs match, row-by-row, the scalar
``Connector`` in ``curve_hold`` mode across a fuzz of random inputs. Also
checks that the array-safety patch on ``get_vec_target`` preserves scalar
behaviour exactly.
"""
import numpy as np
import pytest

from welleng.connector import (
    Connector,
    get_vec_target,
    solve_curve_hold_batch,
)


def _random_batch(n, seed):
    """Generate random (pos1, vec1, pos_target, radius) tuples that reliably
    resolve to the ``curve_hold`` (``min_dist_to_target``) branch of the
    scalar Connector.
    """
    rng = np.random.default_rng(seed)
    pos1 = rng.uniform(-500, 500, (n, 3))

    v = rng.standard_normal((n, 3))
    vec1 = v / np.linalg.norm(v, axis=-1, keepdims=True)

    # Target well forward of the start, with moderate lateral offset — kept
    # small relative to the forward drift so the geometry stays in the
    # ``min_dist_to_target`` regime (where ``radius_design <= radius_critical``).
    forward = rng.uniform(600, 1200, (n, 1)) * vec1
    lateral = rng.uniform(-150, 150, (n, 3))
    pos_target = pos1 + forward + lateral

    radius = rng.uniform(200, 800, n)
    return pos1, vec1, pos_target, radius


def _dls_deg_per_30m(radius):
    """DLS in deg/30m for a given radius in metres — matches Connector's
    internal conversion (``dls_design = denom / radius`` with denom=30 for
    meters)."""
    return np.degrees(30.0 / radius)


def test_batch_matches_scalar_connector():
    """Fuzz 20 random cases, compare batch output to per-row scalar Connector."""
    pos1, vec1, pos_target, radius = _random_batch(n=20, seed=42)
    batch = solve_curve_hold_batch(pos1, vec1, pos_target, radius)

    for i in range(len(radius)):
        c = Connector(
            pos1=pos1[i].tolist(),
            vec1=vec1[i].tolist(),
            pos2=pos_target[i].tolist(),
            dls_design=float(_dls_deg_per_30m(radius[i])),
        )
        # Sanity: scalar Connector resolved to the curve-hold branch.
        assert c.method == 'min_dist_to_target', (
            f"row {i}: scalar connector picked method={c.method!r}, "
            "test inputs are off-regime"
        )

        np.testing.assert_allclose(
            batch['pos2'][i], c.pos2, rtol=1e-9, atol=1e-8,
            err_msg=f"pos2 mismatch at row {i}",
        )
        np.testing.assert_allclose(
            batch['vec_target'][i], c.vec_target, rtol=1e-9, atol=1e-8,
            err_msg=f"vec_target mismatch at row {i}",
        )
        np.testing.assert_allclose(
            batch['tangent_length'][i], c.tangent_length,
            rtol=1e-9, atol=1e-8,
            err_msg=f"tangent_length mismatch at row {i}",
        )
        np.testing.assert_allclose(
            batch['dogleg'][i], c.dogleg, rtol=1e-9, atol=1e-8,
            err_msg=f"dogleg mismatch at row {i}",
        )
        np.testing.assert_allclose(
            batch['dist_curve'][i], c.dist_curve, rtol=1e-9, atol=1e-8,
            err_msg=f"dist_curve mismatch at row {i}",
        )


def test_batch_scalar_input_shapes():
    """Passing unbatched inputs must still return the unbatched shapes."""
    out = solve_curve_hold_batch(
        pos1=[0.0, 0.0, 0.0],
        vec1=[0.0, 0.0, 1.0],
        pos_target=[100.0, 100.0, 1000.0],
        radius=300.0,
    )
    assert out['pos2'].shape == (3,)
    assert out['vec_target'].shape == (3,)
    # Scalars may come back as either 0-d ndarrays or Python floats
    # (``check_dogleg`` returns a Python float for scalar input — matching
    # its pre-batch scalar API). Use ``np.ndim`` which handles both.
    assert np.ndim(out['tangent_length']) == 0
    assert np.ndim(out['dogleg']) == 0
    assert np.ndim(out['dist_curve']) == 0
    assert np.ndim(out['md']) == 0

    # And the values should match a scalar Connector run on the same inputs.
    c = Connector(
        pos1=[0.0, 0.0, 0.0],
        vec1=[0.0, 0.0, 1.0],
        pos2=[100.0, 100.0, 1000.0],
        dls_design=float(_dls_deg_per_30m(300.0)),
    )
    np.testing.assert_allclose(out['pos2'], c.pos2, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(out['vec_target'], c.vec_target, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(out['tangent_length'], c.tangent_length, rtol=1e-9, atol=1e-8)


def test_get_vec_target_scalar_preserved():
    """The array-safety patch must not change scalar-input behaviour."""
    pos1 = np.array([0.0, 0.0, 0.0])
    vec1 = np.array([0.0, 0.0, 1.0])
    pos_target = np.array([100.0, 100.0, 1000.0])

    out = get_vec_target(pos1, vec1, pos_target,
                         tangent_length=850.0, dist_curve=150.0, func_dogleg=0.99)
    assert out.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, rtol=1e-10)


def test_get_vec_target_zero_curve_scalar():
    """Scalar input with dist_curve == 0 returns vec1 unchanged."""
    vec1 = np.array([0.0, 0.0, 1.0])
    out = get_vec_target(
        np.zeros(3), vec1, np.array([0.0, 0.0, 100.0]),
        tangent_length=100.0, dist_curve=0.0, func_dogleg=1.0,
    )
    np.testing.assert_array_equal(out, vec1)


def test_get_vec_target_zero_curve_batch_mixed():
    """Batched input with some zero-curve rows returns vec1 on those rows."""
    n = 5
    pos1 = np.zeros((n, 3))
    vec1 = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    pos_target = np.array([[i * 10.0, i * 10.0, 100.0 + i] for i in range(n)])
    tangent_length = np.full(n, 100.0)
    dist_curve = np.array([0.0, 50.0, 0.0, 75.0, 0.0])
    func_dogleg = np.full(n, 0.99)

    out = get_vec_target(pos1, vec1, pos_target,
                         tangent_length, dist_curve, func_dogleg)
    assert out.shape == (n, 3)

    # Zero-curve rows match vec1; non-zero rows are unit vectors (possibly
    # different from vec1).
    for i in range(n):
        if dist_curve[i] == 0:
            np.testing.assert_array_equal(out[i], vec1[i])
        else:
            np.testing.assert_allclose(np.linalg.norm(out[i]), 1.0, rtol=1e-10)


def test_batch_wraps_negative_dogleg_like_scalar():
    """Geometries that produce a raw negative dogleg must be wrapped by +2π
    (matching ``check_dogleg`` inside the scalar ``_min_dist_to_target``)
    — otherwise ``dist_curve`` goes negative and total MD is nonsense.
    """
    # Construct a case that is known to produce negative raw dogleg:
    # a near-vertical start reaching a target that's lateral and deeper by a
    # small amount, with a small radius. The scalar Connector handles this
    # via check_dogleg; the batch must do the same.
    pos1 = np.array([0.0, 0.0, 1500.0])
    vec1 = np.array([0.0, 0.0, 1.0])
    pos_target = np.array([-60.0, -350.0, 1800.0])   # same as demo KOP→intermediate
    radius = 250.0

    out = solve_curve_hold_batch(pos1, vec1, pos_target, radius)
    # If check_dogleg is applied, dogleg ∈ [0, 2π) and dist_curve ≥ 0,
    # so md ≥ 0.
    assert out["dogleg"] >= 0.0, f"dogleg not wrapped: {out['dogleg']}"
    assert out["dist_curve"] >= 0.0, f"negative dist_curve: {out['dist_curve']}"
    assert out["md"] >= 0.0, f"negative md: {out['md']}"

    # And the scalar Connector on the same inputs agrees.
    c = Connector(
        pos1=pos1.tolist(), vec1=vec1.tolist(),
        pos2=pos_target.tolist(),
        dls_design=float(_dls_deg_per_30m(radius)),
    )
    assert c.method == 'min_dist_to_target'
    np.testing.assert_allclose(out["dogleg"], c.dogleg, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(out["dist_curve"], c.dist_curve, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(out["tangent_length"], c.tangent_length, rtol=1e-9, atol=1e-8)


def test_check_dogleg_scalar_and_array():
    """Scalar input → scalar output; array input → array output."""
    from welleng.connector import check_dogleg

    # Scalar
    assert check_dogleg(-1.0) == -1.0 + 2 * np.pi
    assert check_dogleg(1.0) == 1.0
    assert isinstance(check_dogleg(0.5), float)

    # Array
    arr = np.array([-1.0, 0.0, 1.0, -0.5])
    out = check_dogleg(arr)
    np.testing.assert_allclose(
        out, np.array([-1.0 + 2 * np.pi, 0.0, 1.0, -0.5 + 2 * np.pi])
    )


def test_batch_shape_preserved_2d():
    """Batch solver handles >1 leading dim correctly (e.g., a grid sweep)."""
    shape = (4, 6)
    rng = np.random.default_rng(7)
    pos1 = rng.uniform(-100, 100, shape + (3,))
    v = rng.standard_normal(shape + (3,))
    vec1 = v / np.linalg.norm(v, axis=-1, keepdims=True)
    pos_target = pos1 + rng.uniform(500, 900, shape + (1,)) * vec1
    radius = rng.uniform(300, 600, shape)

    out = solve_curve_hold_batch(pos1, vec1, pos_target, radius)
    assert out['pos2'].shape == shape + (3,)
    assert out['vec_target'].shape == shape + (3,)
    assert out['tangent_length'].shape == shape
    assert out['md'].shape == shape
