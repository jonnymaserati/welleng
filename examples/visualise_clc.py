"""Visual QAQC of welleng's closed-form CLC solver, with vedo.

Constructs a random asymmetric (R1 != R2) curve-line-curve path, solves it back
with ``welleng.sawaryn_analytical.solve_clc`` and draws every solution overlaid
on the ground-truth construction (the arc drawing handles the major-arc geometry
correctly).

    python visualise_clc.py [seed]

    Right / Space : next problem    Left : previous    Q : quit

    White = ground-truth construction   Green = shortest (min-MD)   Grey = alternatives
"""
import sys
import numpy as np
from numpy.linalg import norm

from welleng.sawaryn_analytical import solve_clc

O = np.zeros(3)
V = np.array([0.0, 0.0, 1.0])


# ── geometry / drawing helpers (from review_asymmetric.py) ──────────────────

def arc_points(centre, e_r, axis, radius, phi_start, phi_end, n=50):
    return [centre + radius * (np.cos(a) * e_r + np.sin(a) * axis)
            for a in np.linspace(phi_start, phi_end, n)]


def find_turning_circle(pos_on, pos_start, vec_start, radius, subtended=None):
    diff = pos_on - pos_start
    along = np.dot(diff, vec_start)
    perp = diff - along * vec_start
    perp_norm = norm(perp)
    if perp_norm < 1e-10:
        return pos_start, vec_start, vec_start, 0.0
    radial = perp / perp_norm
    centre = pos_start + radius * radial
    e_r = -radial
    v_start = pos_start - centre
    v_on = pos_on - centre
    cos_phi = np.clip(np.dot(v_start, v_on) / (radius * radius), -1, 1)
    phi = np.arccos(cos_phi)                          # minor angle in [0, pi]
    if subtended is not None:
        if abs(subtended) > np.pi:                    # solver says it's the major arc
            phi = 2 * np.pi - phi
    elif np.dot(np.cross(v_start, v_on), np.cross(v_start, vec_start)) < 0:
        phi = 2 * np.pi - phi                         # geometric fallback
    return centre, e_r, vec_start, phi


def make_rotated_torus(pos, vec, r, color, alpha=0.15, res=36):
    from vedo import Torus
    t = Torus(pos=(0, 0, 0), r1=r, r2=r, res=res, c=color, alpha=alpha)
    rot_axis = np.cross(V, vec)
    rn = norm(rot_axis)
    if rn > 1e-12:
        t.rotate(np.degrees(np.arccos(np.clip(np.dot(V, vec), -1, 1))),
                 axis=rot_axis / rn, point=(0, 0, 0))
    elif np.dot(V, vec) < 0:
        t.rotate(180, axis=[1, 0, 0], point=(0, 0, 0))
    t.pos(pos)
    return t


def draw_clc_generic(pos0, vec0, p1, p2, pos3, vec3, r1, color, lw=4, alpha=1.0, r2=None,
                     dog1=None, dog2=None):
    from vedo import Line
    r2 = r1 if r2 is None else r2
    actors = []
    c1, er1, ax1, phi1 = find_turning_circle(p1, pos0, vec0, r1, subtended=dog1)
    if phi1 > 1e-6:
        actors.append(Line(arc_points(c1, er1, ax1, r1, 0, phi1), c=color, lw=lw, alpha=alpha))
    actors.append(Line([p1, p2], c=color, lw=lw, alpha=alpha))
    hold = p2 - p1
    hl = norm(hold)
    vec2 = hold / hl if hl > 1e-10 else vec0
    c2, er2, ax2, phi2 = find_turning_circle(pos3, p2, vec2, r2, subtended=dog2)
    if phi2 > 1e-6:
        actors.append(Line(arc_points(c2, er2, ax2, r2, 0, phi2), c=color, lw=lw, alpha=alpha))
    return actors


# ── welleng problem + solver data ───────────────────────────────────────────

def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def problem(seed):
    """Random asymmetric CLC problem -> target, target tangent, radii, ground truth."""
    rng = np.random.default_rng(seed)
    dl1, tf1 = rng.uniform(0.3, 2.2), rng.uniform(-np.pi, np.pi)
    dl2, tf2 = rng.uniform(0.3, 2.2), rng.uniform(-np.pi, np.pi)
    dist = rng.uniform(0.4, 1.8)
    R1, R2 = rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)
    c1, s1 = np.cos(dl1), np.sin(dl1)
    ct1, st1 = np.cos(tf1), np.sin(tf1)
    p_arc1 = R1 * np.array([(1 - c1) * ct1, (1 - c1) * st1, s1])
    v = np.array([s1 * ct1, s1 * st1, c1])
    p_line = p_arc1 + dist * v
    R = _Rz(tf1) @ _Ry(dl1) @ _Rz(tf2)
    c2, s2 = np.cos(dl2), np.sin(dl2)
    p4 = p_line + R @ (R2 * np.array([1 - c2, 0, s2]))
    t4 = R @ np.array([s2, 0, c2])
    return p4, t4, R1, R2, dist, (p_arc1, p_line, dl1, dl2)


def welleng_sols(p4, t4, R1, R2):
    """solve_clc -> sol dicts with reconstructed pos1 (end arc1), pos2 (end line)."""
    out = []
    for s in solve_clc(O, V, p4, t4, R1, R2, return_all=True):
        b, a1, a2 = s['beta'], s['alpha1'], s['alpha2']
        T1, T2 = np.tan(a1 / 2), np.tan(a2 / 2)
        t2 = (p4 - R1 * T1 * V - R2 * T2 * t4) / (R1 * T1 + b + R2 * T2)
        pos1 = R1 * T1 * (V + t2)
        pos2 = pos1 + b * t2
        out.append(dict(pos1=pos1, pos2=pos2, phi1=a1, phi2=a2, d=b,
                        total_length=s['total_md']))
    out.sort(key=lambda s: s['total_length'])
    return out


def main():
    from vedo import Arrow, Point, Plotter, Text2D
    seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cur = [seed0]
    plt = Plotter(size=(1200, 900), title="welleng CLC -- closed-form solver QAQC")
    plt.background('lightgrey')

    def render():
        for a in list(plt.actors):
            plt.remove(a)
        seed = cur[0]
        p4, t4, R1, R2, gt_d, (gtA, gtB, gdl1, gdl2) = problem(seed)
        sols = welleng_sols(p4, t4, R1, R2)
        gt_found = any(abs(s['d'] - gt_d) < 1e-2 for s in sols)
        status, sc = ("OK", "green4") if gt_found else ("MISS", "red")

        cap = (f"seed {seed}  [{status}]  r1={R1:.2f} r2={R2:.2f}  "
               f"{len(sols)} solution(s)")
        if sols:
            cap += f"  best_MD={sols[0]['total_length']:.3f}"
        cap += "\nRight/Space : next    Left : back    Q : quit"
        plt.add(Text2D(cap, pos='top-left', s=0.75, c=sc, bg='k8', alpha=0.85))

        plt.add(make_rotated_torus(O, V, R1, "lightblue"))
        plt.add(make_rotated_torus(p4, t4, R2, "lightsalmon"))
        plt.add(Point(O, c="green4", r=12))
        plt.add(Arrow(O, O + V * R1 * 0.5, c="green4", s=0.003))
        plt.add(Point(p4, c="tomato", r=12))
        plt.add(Arrow(p4, p4 + t4 * R2 * 0.5, c="tomato", s=0.003))

        # ground-truth reference (white)
        for a in draw_clc_generic(O, V, gtA, gtB, p4, t4, R1, "white", 3, 0.8, R2,
                                  dog1=gdl1, dog2=gdl2):
            plt.add(a)
        plt.add(Point(gtA, c="white", r=6)); plt.add(Point(gtB, c="white", r=6))

        # alternatives (grey), shortest (green)
        for s in reversed(sols[1:]):
            for a in draw_clc_generic(O, V, s['pos1'], s['pos2'], p4, t4, R1, "grey", 2, 0.4, R2,
                                      dog1=s['phi1'], dog2=s['phi2']):
                plt.add(a)
        if sols:
            best = sols[0]
            for a in draw_clc_generic(O, V, best['pos1'], best['pos2'], p4, t4, R1, "green", 4, 1.0, R2,
                                      dog1=best['phi1'], dog2=best['phi2']):
                plt.add(a)
            plt.add(Point(best['pos1'], c="green", r=8)); plt.add(Point(best['pos2'], c="green", r=8))
            lines = []
            for si, s in enumerate(sols[:5]):
                pre = ">>>" if si == 0 else "   "
                lines.append(f"{pre} Sol {si+1}: phi1={np.degrees(s['phi1']):.1f}  "
                             f"phi2={np.degrees(s['phi2']):.1f}  d={s['d']:.3f}  L={s['total_length']:.3f}")
            plt.add(Text2D("\n".join(lines), pos=(0.01, 0.85), s=0.6, c="green", font="Courier"))
        for li, (txt, col) in enumerate([("ground truth (constructed path)", "white"),
                                          ("shortest (min-MD) solution", "green5"),
                                          ("alternative solutions", "grey")]):
            plt.add(Text2D(txt, pos=(0.62, 0.96 - 0.045 * li), s=0.85, c=col,
                           bg='k3', alpha=0.9))
        plt.reset_camera(); plt.render()

    def on_key(evt):
        if evt.keypress in ('space', 'n', 'Right'):
            cur[0] += 1; render()
        elif evt.keypress in ('b', 'p', 'Left'):
            cur[0] = max(0, cur[0] - 1); render()

    plt.add_callback('KeyPress', on_key)
    render()
    plt.show(axes=0)


if __name__ == "__main__":
    main()
