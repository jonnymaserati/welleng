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
    The test verifies:

    1. Method is 'curve_hold_curve'.
    2. Connector reaches the correct endpoint within tolerance.
    3. Connector's total MD ≤ constructed path MD (it found the min-MD solution).
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

    failures = []

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
            failures.append((i, 'method', c.method))
            continue

        if not np.allclose(c.pos_target, path['pos3'], atol=tol):
            failures.append((i, 'pos_target', c.pos_target.tolist(), path['pos3'].tolist()))
        elif not np.allclose(c.vec_target, path['vec3'], atol=tol):
            failures.append((i, 'vec_target', c.vec_target.tolist(), path['vec3'].tolist()))
        elif c.md_target - c.md1 > constructed_md + tol:
            failures.append((i, 'md', c.md_target - c.md1, constructed_md))

    assert not failures, (
        f"{len(failures)}/{n} CLC cases failed:\n"
        + "\n".join(str(f) for f in failures[:10])
    )


def main():
    test_drop_off()
    test_extend_to_tvd()


if __name__ == "__main__":
    main()
