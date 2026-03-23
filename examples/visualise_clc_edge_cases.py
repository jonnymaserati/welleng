"""
examples/visualise_clc_edge_cases.py
--------------------------------------
Visualise the CLC connector edge cases captured by
tests/test_connector.py::test_clc_connector.

Runs the identical 1000-sample random sweep (seed=42, radius=1 m) and
separates the results into:

  DLS violations  (red mesh)
      The solver converged to a geometry where at least one arc uses a
      radius smaller than the design radius.  A valid path at the design
      radius exists (proved by the blue reference lines) but the fixed-point
      iterator settled in a tighter local minimum.

  MD-suboptimal   (orange mesh)
      The solver produced a valid, DLS-compliant path but one that is longer
      than the reference path.  Another local-minimum effect.

Both groups are shown alongside their blue reference paths from
make_clc_path, laid out in a grid so every case can be inspected at once.
Cases are labelled with their test-index and the key metric (min arc radius
for DLS violations, excess MD for suboptimal cases).

Usage
-----
    python examples/visualise_clc_edge_cases.py
"""

import numpy as np
from vedo import Lines, Arrows, Text3D

import welleng as we
from welleng.connector import Connector
from welleng.survey import from_connections
from welleng.utils import make_clc_path, dls_from_radius, get_arc


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_arc(dogleg_total, radius, toolface, pos_start, vec_start, n=40):
    """Return (n+1, 3) positions sampled uniformly along an arc."""
    pts = []
    for k in range(n + 1):
        pos, _, _ = get_arc(
            dogleg_total * k / n, radius, toolface, pos_start, vec_start
        )
        pts.append(pos)
    return np.array(pts)


def reference_path_lines(case, radius, pos0, vec0, offset, n=40):
    """Vedo Lines tracing the reference CLC path, shifted by *offset*."""
    path = case['path']

    arc1 = _sample_arc(
        case['dl1'], radius, case['tf1'], pos0, vec0, n
    )
    hold = np.array([
        path['pos1'] + path['vec1'] * (case['d'] * k / n)
        for k in range(n + 1)
    ])
    arc2 = _sample_arc(
        case['dl2'], radius, case['tf2'], path['pos2'], path['vec2'], n
    )

    pts = np.vstack([arc1, hold[1:], arc2[1:]]) + offset
    return Lines(pts[:-1], pts[1:], c='deepskyblue', lw=2)


def start_end_arrows(case, offset, scale=0.4):
    """Small arrows showing start/end direction vectors."""
    c = case['connector']
    starts = np.array([c.pos1, c.pos_target]) + offset
    ends   = np.array([
        c.pos1      + scale * c.vec1,
        c.pos_target + scale * c.vec_target,
    ]) + offset
    return Arrows(starts, ends, c='yellow', s=0.3, res=8)


# ─────────────────────────────────────────────────────────────────────────────
# Edge-case collection  (mirrors test_clc_connector exactly)
# ─────────────────────────────────────────────────────────────────────────────

def collect_edge_cases(n=1000, seed=42, radius=1.0, tol=1e-3):
    rng = np.random.default_rng(seed)

    tf1 = rng.uniform(-np.pi, np.pi, n)
    dl1 = rng.uniform(1e-2,   np.pi, n)
    d   = rng.uniform(0.01,   radius, n)
    tf2 = rng.uniform(-np.pi, np.pi, n)
    dl2 = rng.uniform(1e-2,   np.pi, n)

    dls  = dls_from_radius(radius)
    pos0 = np.array([0., 0., 0.])
    vec0 = np.array([0., 0., 1.])

    dls_violations = []
    md_suboptimal  = []

    for i in range(n):
        path = make_clc_path(
            tf1[i], dl1[i], d[i], tf2[i], dl2[i],
            pos0=pos0, vec0=vec0, radius=radius,
        )
        constructed_md = path['dist_curve1'] + d[i] + path['dist_curve2']

        c = Connector(
            pos1=pos0, vec1=vec0,
            pos2=path['pos3'], vec2=path['vec3'],
            dls_design=dls,
        )

        if c.method != 'curve_hold_curve':
            continue
        if not np.allclose(c.pos_target, path['pos3'], atol=tol):
            continue
        if not np.allclose(c.vec_target, path['vec3'], atol=tol):
            continue

        actual_r1 = c.dist_curve  / c.dogleg  if c.dogleg  > 1e-10 else radius
        actual_r2 = c.dist_curve2 / c.dogleg2 if c.dogleg2 > 1e-10 else radius
        min_r = min(actual_r1, actual_r2)

        base = dict(
            idx=i, path=path, connector=c,
            tf1=tf1[i], dl1=dl1[i], d=d[i], tf2=tf2[i], dl2=dl2[i],
            connector_md=c.md_target - c.md1,
            constructed_md=constructed_md,
        )

        if min_r < radius * (1 - tol):
            dls_violations.append({**base, 'min_r': min_r})
        elif c.md_target - c.md1 > constructed_md + tol:
            md_suboptimal.append(base)

    return dls_violations, md_suboptimal


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def main():
    radius   = 1.0
    well_r   = 0.07   # rendered wellbore radius (m)
    step     = 0.02   # survey interpolation step (m)
    spacing  = 8.0    # grid spacing between cases (m)
    cols     = 8      # cases per row
    pos0     = np.array([0., 0., 0.])
    vec0     = np.array([0., 0., 1.])

    print("Collecting edge cases (runs the 1000-sample CLC test, ~18 s)...")
    dls_v, md_sub = collect_edge_cases(radius=radius)
    print(f"  DLS violations : {len(dls_v)}")
    print(f"  MD suboptimal  : {len(md_sub)}")

    # Row-group 0 = DLS violations, row-group 1 = MD suboptimal.
    # Within each group: columns → case index, rows → overflow.
    n_dls_rows = max(1, (len(dls_v) - 1) // cols + 1)
    group_row_gap = n_dls_rows + 2   # blank rows between the two groups

    def grid_offset(k, group):
        row = k // cols
        col = k % cols
        n_row = (group * group_row_gap + row) * spacing
        e_col = col * spacing
        return np.array([n_row, e_col, 0.])

    plt = we.visual.Plotter()

    # ── DLS violations ────────────────────────────────────────────────────────
    for k, case in enumerate(dls_v):
        offset = grid_offset(k, group=0)

        try:
            s = from_connections(
                case['connector'], step=step, radius=well_r,
                start_nev=list(offset),
            )
            m = we.mesh.WellMesh(s, method='circle', n_verts=12)
            plt.add(m, c='red', alpha=0.8)
        except Exception as exc:
            print(f"  DLS case #{case['idx']}: mesh failed ({exc})")

        plt.add(reference_path_lines(case, radius, pos0, vec0, offset))
        plt.add(start_end_arrows(case, offset))

        lbl_pos = offset + np.array([0., 0., -0.6])
        plt.add(Text3D(
            f"#{case['idx']}\nr={case['min_r']:.3f}",
            pos=lbl_pos, s=0.18, c='red', depth=0,
        ))

    # ── MD suboptimal ─────────────────────────────────────────────────────────
    for k, case in enumerate(md_sub):
        offset = grid_offset(k, group=1)

        try:
            s = from_connections(
                case['connector'], step=step, radius=well_r,
                start_nev=list(offset),
            )
            m = we.mesh.WellMesh(s, method='circle', n_verts=12)
            plt.add(m, c='orange', alpha=0.8)
        except Exception as exc:
            print(f"  MD case #{case['idx']}: mesh failed ({exc})")

        plt.add(reference_path_lines(case, radius, pos0, vec0, offset))
        plt.add(start_end_arrows(case, offset))

        excess = case['connector_md'] - case['constructed_md']
        lbl_pos = offset + np.array([0., 0., -0.6])
        plt.add(Text3D(
            f"#{case['idx']}\n+{excess:.3f}m",
            pos=lbl_pos, s=0.18, c='darkorange', depth=0,
        ))

    print()
    print("Legend:")
    print("  RED mesh      — connector path (DLS violated)")
    print("  ORANGE mesh   — connector path (MD suboptimal but DLS compliant)")
    print("  BLUE lines    — reference path from make_clc_path")
    print("  YELLOW arrows — start and end direction vectors")
    print()
    print("Labels: #<test_index>  r=<min arc radius>  or  +<excess MD in metres>")
    print()
    print("Launching VTK viewer...")
    plt.show(axes=1)


if __name__ == "__main__":
    main()
