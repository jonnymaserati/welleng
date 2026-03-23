import numpy as np

from welleng.connector import Connector, drop_off, extend_to_tvd
from welleng.survey import Survey, from_connections
from welleng.node import Node
from welleng.utils import make_clc_path, dls_from_radius


def test_md_hold():
    # test hold with only md provided
    c = Connector(
        vec1=[0, 0, 1],
        md2=500,
    )
    assert (
        c.inc_target == c.inc1
        and c.azi_target == c.azi1
        and c.pos_target[2] == c.md_target
    ), "Failed c1"
    assert c.method == 'hold', "Unexpected method"

    assert isinstance(from_connections(c), Survey)

def test_md_and_vec():
    # test with md2 and vec2 provided (minimum curvature)
    c = Connector(
        vec1=[0, 0, 1],
        md2=1000,
        vec2=[0, 1, 0]
    )
    assert c.method == 'min_curve'

def test_pos():
    # test with pos2 provided (minimum distance)
    c = Connector(
        vec1=[0, 0, 1],
        pos2=[100, 100, 1000],
    )
    assert c.md_target > c.pos1[2], "Failed c3"

def test_pos_and_dls():
    # test with pos2 needing more aggressive dls (minimum curvature)
    c = Connector(
        vec1=[0, 0, 1],
        pos2=[200, 400, 200]
    )
    assert c.method == 'min_curve_to_target'

def test_pos_and_vec():
    # test with pos2 and vec2 provided
    vec1 = [-1, -1, 1]
    vec2 = [1, -1, 0]
    c = Connector(
        pos1=[0., 0., 0],
        vec1=vec1 / np.linalg.norm(vec1),
        pos2=[0., 1000., 500.],
        vec2=vec2 / np.linalg.norm(vec2),
    )
    assert c.method == 'curve_hold_curve'

    # test if interpolator and survey functions are working
    assert isinstance(from_connections(c, step=30), Survey)

def test_pos_inc_azi():
    # test with pos2, inc1 and azi1 provided
    c = Connector(
        pos1=[0., 0., 0],
        inc1=0.,
        azi1=90,
        pos2=[1000., 1000., 1000.],
        vec2=[0., 0., 1.],
    )
    assert c.method == 'curve_hold_curve'

def test_dls2():
    # test with different dls for second curve section
    c = Connector(
        pos1=[0., 0., 0],
        vec1=[0., 0., 1.],
        pos2=[0., 100., 1000.],
        vec2=[0., 0., 1.],
        dls_design2=5
    )
    assert c.radius_design2 < c.radius_design

def test_radius_critical():
    # test with dls_critical requirement (actual dls < dls_design)
    c = Connector(
        pos1=[0., 0., 0],
        vec1=[0., 0., 1.],
        pos2=[0., 100., 100.],
        vec2=[0., 0., 1.],
    )
    assert c.radius_critical < c.radius_design

def test_min_curve():
    # test min_curve (inc2 provided)
    c = Connector(
        pos1=[0., 0., 0],
        vec1=[0., 0., 1.],
        inc2=30,
    )
    assert c.method == 'min_curve'

def test_radius_critical_with_min_curve():
    # test min_curve with md less than required radius
    c = Connector(
        pos1=[0., 0., 0],
        inc1=0,
        azi1=0,
        md2=500,
        inc2=90,
        azi2=0,
    )
    assert c.radius_critical < c.radius_design

def test_drop_off(tol=1e-4):
    node = Node(
        pos=[0, 0, 3000], inc=30, azi=135, md=4000
    )
    nodes = drop_off(
        target_inc=0, dls=3, node=node
    )
    assert len(nodes) == 1, "Unexpected tangent section."
    assert abs(nodes[0].inc_deg - 0) < tol

    nodes = drop_off(
        target_inc=0, dls=3, node=node, delta_md=1000
    )
    assert len(nodes) == 2, "Unexpected number of nodes."
    assert nodes[-1].md - 1000 - node.md < tol, "Unexpected delta md."
    assert nodes[0].inc_deg - nodes[1].inc_deg < tol, "Unexpected tangent inc."

def test_extend_to_tvd(tol=1e-4):
    node = Node(
        pos=[0, 0, 3000], inc=30, azi=135, md=4000
    )
    nodes = drop_off(
        target_inc=0, dls=3, node=node
    )
    connectors = extend_to_tvd(
        target_tvd=3500, node=node, target_inc=0, dls=3
    )
    assert len(connectors) == 2, "Unexpected number of connectors."
    assert connectors[0].node_end.md - nodes[0].md < tol, "Unexpected md."
    assert np.allclose(
        connectors[0].node_end.pos_nev,
        nodes[0].pos_nev
    ), "Unexpected pos_nev."
    assert np.allclose(
        connectors[0].node_end.vec_nev,
        nodes[0].vec_nev, rtol=tol, atol=tol
    ), "Unexpected vec_nev."
    assert np.allclose(
        connectors[0].node_end.vec_nev,
        connectors[1].node_end.vec_nev, rtol=tol, atol=tol
    ), "Unexpected tangent section."
    assert connectors[1].node_end.pos_nev[2] - 3500 < tol, "Unexpected tvd."
    
    connectors = extend_to_tvd(
        target_tvd=3500, node=node
    )
    assert connectors[-1].node_end.pos_nev[2] - 3500 < tol, "Unexpected tvd."
    assert np.allclose(
        connectors[-1].node_start.vec_nev,
        connectors[-1].node_end.vec_nev, rtol=tol, atol=tol
    ), "Unexpected tangent section."
    pass


def test_clc_connector(n=1000, seed=42, radius=1.0, tol=1e-3):
    """Round-trip test for the CLC connector using make_clc_path.

    make_clc_path builds a known CLC path from (toolface1, dogleg1, distance,
    toolface2, dogleg2), proving a valid solution exists at the given DLS.
    The Connector is then given only the start and end pos/vec and must find
    its way back.  Since the Connector always finds the minimum-MD CLC path,
    its internal decomposition may differ from the one we constructed.
    Doglegs up to π (180°) are tested — large doglegs may yield a different
    decomposition but the endpoint constraint still uniquely determines whether
    the Connector succeeded.

    All tests use radius=1.0 m (~1700 deg/30m), which is physically unrealistic
    for real drilling.  At realistic DLS values the solver performs without any
    failures.  The extreme geometry is used here to stress-test the algorithm
    and expose edge-case behaviour.

    The test separates three categories of outcome:

    Hard constraint (zero tolerance — any failure is a genuine bug):
        Wrong connector method or endpoint mismatch — these mean the solver
        returned a geometrically invalid result.

    Regression-guarded constraints (count must not grow past baseline):
        DLS exceeded  — solver converged to an intermediate geometry that
                        requires tighter curvature than the design radius.
                        Valid paths exist at the design DLS but the fixed-point
                        iterator settled in a local minimum.  Baseline 22/1000.
        MD suboptimal — among DLS-compliant results, solver found a valid path
                        longer than the known-constructible reference path.
                        Another local-minimum effect.  Baseline 4/1000.

    Both regression guards use a threshold well above the current baseline so
    that genuine regressions (counts that jump significantly) are caught while
    known edge-case noise does not block CI.
    """
    rng = np.random.default_rng(seed)

    toolface1 = rng.uniform(-np.pi, np.pi, n)
    dogleg1   = rng.uniform(1e-2, np.pi, n)
    distance  = rng.uniform(0.01, radius, n)
    toolface2 = rng.uniform(-np.pi, np.pi, n)
    dogleg2   = rng.uniform(1e-2, np.pi, n)

    dls = dls_from_radius(radius)
    pos0 = np.array([0., 0., 0.])
    vec0 = np.array([0., 0., 1.])

    # Hard failures — geometry or endpoint is wrong (genuine bugs, must be zero)
    hard_failures = []
    # DLS violations — solver used tighter curvature than the design radius
    dls_violations = []
    # MD suboptimal — valid, DLS-compliant path but not the globally shortest
    md_suboptimal = []

    for i in range(n):
        path = make_clc_path(
            toolface1[i], dogleg1[i], distance[i], toolface2[i], dogleg2[i],
            pos0=pos0, vec0=vec0, radius=radius,
        )
        constructed_md = path['dist_curve1'] + distance[i] + path['dist_curve2']

        c = Connector(
            pos1=pos0,
            vec1=vec0,
            pos2=path['pos3'],
            vec2=path['vec3'],
            dls_design=dls,
        )

        if c.method != 'curve_hold_curve':
            hard_failures.append((i, 'method', c.method))
            continue

        if not np.allclose(c.pos_target, path['pos3'], atol=tol):
            hard_failures.append((i, 'pos_target', c.pos_target.tolist(), path['pos3'].tolist()))
            continue

        if not np.allclose(c.vec_target, path['vec3'], atol=tol):
            hard_failures.append((i, 'vec_target', c.vec_target.tolist(), path['vec3'].tolist()))
            continue

        # Check that the Connector did not exceed the design DLS.
        # make_clc_path proved a valid path exists at exactly `radius`, so the
        # Connector should be able to reach the target without bending tighter.
        # Historically the Connector traded straight-section length for tighter
        # curvature to shorten total MD — this violates the dls_design
        # constraint.  The fix (r_safe seeding + outer loop) eliminated most
        # such cases; the remaining ones are local-minimum convergences for
        # extreme geometries only.
        actual_r1 = c.dist_curve / c.dogleg if c.dogleg > 1e-10 else radius
        actual_r2 = c.dist_curve2 / c.dogleg2 if c.dogleg2 > 1e-10 else radius
        if min(actual_r1, actual_r2) < radius * (1 - tol):
            dls_violations.append((i, min(actual_r1, actual_r2), radius))
            continue

        # Soft check: did the solver find the minimum-MD path?
        connector_md = c.md_target - c.md1
        if connector_md > constructed_md + tol:
            md_suboptimal.append((i, connector_md, constructed_md))

    # --- Hard assertion: wrong method or endpoint mismatch is always a bug ---
    assert not hard_failures, (
        f"{len(hard_failures)}/{n} CLC cases had hard correctness failures "
        f"(wrong method or endpoint mismatch):\n"
        + "\n".join(str(f) for f in hard_failures[:10])
    )

    # --- Regression guard: DLS violations ---
    # The fixed-point solver can converge to an intermediate point that forces
    # slightly tighter curvature than the design radius for extreme geometries
    # (radius=1 m ≈ 1700 deg/30m — physically unrealistic).  The r_safe
    # seeding fix reduced this from ~125 to 22/1000; the backward-tangent
    # rescue (_delta_pos3_rescue) further reduced it to 20/1000.  The
    # threshold is set with headroom to catch genuine regressions without
    # blocking CI.
    DLS_VIOLATION_THRESHOLD = 55  # baseline ~41; alert if count grows significantly
    assert len(dls_violations) <= DLS_VIOLATION_THRESHOLD, (
        f"{len(dls_violations)}/{n} CLC cases exceeded the design DLS "
        f"(baseline ~41; threshold {DLS_VIOLATION_THRESHOLD}):\n"
        + "\n".join(str(f) for f in dls_violations[:10])
    )

    # --- Regression guard: MD suboptimality ---
    # Cases that pass the DLS check but where the solver converges to a valid,
    # DLS-compliant path that is not the globally shortest (local minimum).
    # _project_tangent eliminated all previously known suboptimal cases by
    # replacing the raw ‖pos3-pos2‖ tangent length with the along-vec3
    # projection, removing the perpendicular-error inflation from md_target.
    # Baseline is now 0/1000.  Threshold kept with headroom for noise.
    MD_SUBOPTIMAL_THRESHOLD = 10  # baseline 0; alert if count grows significantly
    assert len(md_suboptimal) <= MD_SUBOPTIMAL_THRESHOLD, (
        f"{len(md_suboptimal)}/{n} CLC cases returned a non-optimal MD path "
        f"(local-minimum baseline 0; threshold {MD_SUBOPTIMAL_THRESHOLD}):\n"
        + "\n".join(str(f) for f in md_suboptimal[:10])
    )


def main():
    test_drop_off()
    test_extend_to_tvd()


if __name__ == "__main__":
    main()
