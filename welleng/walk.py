import numpy as np
from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

from welleng.utils import (
    get_vec, toolface_vec, toolface_from_vec, get_angles,
    real_radius_from_toolface_radius
)
from welleng.connector import Node, Connector


def get_toolface_apparent(toolface_design_vec, toolface_walk_vec):
    toolface_apparent_vec = (
        toolface_design_vec + toolface_walk_vec
    )
    return toolface_apparent_vec


def bit_walk(
    pos, vec, dls_design, toolface_design,
    dls_walk, toolface_walk,
    # step=30.,
    nev=True, deg=True
):
    # pos_nev, pos_xyz = utils.process_coords(pos, nev)
    # vec_nev, vec_xyz = utils.process_coords(vec, nev)

    coeff = 30

    toolface_design_vec = toolface_vec(toolface_design, dls_design)
    toolface_walk_vec = toolface_vec(toolface_walk, dls_walk)

    toolface_apparent_vec = get_toolface_apparent(
        toolface_design_vec, toolface_walk_vec
    )

    t, r = toolface_from_vec(toolface_apparent_vec)

    return (t, r)


def _get_delta_azi(connector):
    delta_azi = connector.azi_target - connector.azi1
    if delta_azi < -np.pi:
        delta_azi += 2 * np.pi
    if delta_azi > np.pi:
        delta_azi -= 2 * np.pi

    return delta_azi


def _get_toolface_and_radius(connector):
    delta_azi = _get_delta_azi(connector)
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.arctan2(
            np.sin(connector.inc_target) * np.sin(delta_azi),
            (
                np.sin(connector.inc_target)
                * np.cos(connector.inc1)
                * np.cos(delta_azi)
                - np.sin(connector.inc1)
                * np.cos(connector.inc_target)
            )
        )
        t1 = np.nan_to_num(
            t1,
            # np.where(t1 < 0, t1 + 2 * np.pi, t1),
            nan=np.nan
        )
        t2 = np.arctan2(
            np.sin(connector.inc1) * np.sin(delta_azi),
            (
                np.sin(connector.inc_target)
                * np.cos(connector.inc1)
                - np.sin(connector.inc1)
                * np.cos(connector.inc_target)
                * np.cos(delta_azi)
            )
        )

        t2 = np.nan_to_num(
            np.where(t2 < 0, t2 + 2 * np.pi, t2),
            nan=np.nan
        )

        if connector.method == 'hold':
            radius = np.inf
        else:
            radius = min(
                connector.radius_critical, connector.radius_design
            )

        # toolface = np.degrees(np.concatenate((t1, np.array([t2[-1]]))))
        toolface = np.degrees(t1)

    return (toolface, radius)


def bit_walk_survey(survey, toolface_walk, dls_walk, dls_design=3.0):
    if not isinstance(dls_design, (list, tuple, np.ndarray)):
        dls_design = np.full_like(survey.md, dls_design)
    coords = np.array([
        survey.n, survey.e, survey.tvd
    ]).T

    vecs = [survey.vec_nev[0]]
    nodes = [
        Node(
            md=survey.md[0],
            pos=coords[0],
            vec=vecs[0]
        )
    ]
    connectors = []

    for i, (p, d) in enumerate(zip(coords[1:], dls_design[:-1])):
        node = Node(
            pos=p
        )
        connector = Connector(
            node1=nodes[-1],
            node2=node
        )

        toolface_initial, radius_initial = (
            _get_toolface_and_radius(connector)
        )
        if np.isinf(radius_initial):
            dls_survey = 0.
        else:
            dls_survey = np.degrees(
                connector.dogleg
                / (connector.md_target - connector.md1)
                * 30
            )
        toolface_initial_apparent, radius_initial_apparent = bit_walk(
            coords[i], vecs[-1], dls_survey, toolface_initial,
            dls_walk, toolface_walk,
            # step=30.,
            nev=True, deg=True
        )
        toolface_anti, radius_anti = toolface_from_vec(
            toolface_vec(toolface_initial_apparent - 180, d)
            + toolface_vec(toolface_walk, dls_walk)
        )

        radius_anti_real = real_radius_from_toolface_radius(radius_anti)
        radius_initial_apparent_real = real_radius_from_toolface_radius(
            radius_initial_apparent
        )

        bounds = [[0., np.pi / 2], [0., np.pi / 2]]
        args=(
            toolface_initial_apparent,
            # 180,
            radius_initial_apparent_real,
            coords[i],
            vecs[-1],
            toolface_anti,
            radius_anti_real,
            p
        )

        res = minimize(
            function,
            x0=[0., 0.],
            args=args,
            method='SLSQP',
            # method='Nelder-Mead',
            bounds=bounds,
            # tol=1e-2000,
            options={
                # 'eps': 1e-12,
                # 'ftol': 1e-2000
                # 'xtol': 1e-6
            }
        )
        result = function(
            res.x,
            *args,
            result=True
        )

        continue

    return


def function(
    x0, toolface, radius, pos, vec, toolface_anti, radius_anti, pos_end,
    result=False
):
    pos1, vec1, delta_md1 = get_arc(
        x0[0], radius_anti, np.radians(toolface_anti), pos, vec
    )
    pos2, vec2, delta_md2 = get_arc(
        x0[1], radius,
        np.radians(toolface),
        pos1, vec1
    )
    distance = np.linalg.norm(pos2 - pos_end)

    if result:
        result = namedtuple(
            'result', 'pos1 vec1 delta_md1 pos2 vec2 delta_md2'
        )
        return result(
            pos1=pos1,
            vec1=vec1,
            delta_md1=delta_md1,
            pos2=pos2,
            vec2=vec2,
            delta_md2=delta_md2
        )
    else:
        return distance


def get_arc(dogleg, radius, toolface, pos, vec):
    # delta_md = dogleg * np.pi * radius * 0.5
    delta_md = dogleg * radius
    pos_temp = np.array([
        np.cos(dogleg),
        0.,
        np.sin(dogleg)
    ]) * radius
    pos_temp[0] = radius - pos_temp[0]

    vec_temp = np.array([
        np.sin(dogleg),
        0.,
        np.cos(dogleg)
    ])

    inc, azi = get_angles(vec, nev=True).reshape(2)
    # if inc == 0.:
    #     azi = toolface
    #     toolface = 0.
    angles = [
        toolface,
        inc,
        azi
    ]

    # Use the code from those clever chaps over at scipy, I'm sure they're
    # better at writing this code than I am.
    r = R.from_euler('zyz', angles, degrees=False)

    pos_new, vec_new = r.apply(np.vstack((pos_temp, vec_temp)))
    pos_new += pos

    return (pos_new, vec_new, delta_md)


def get_random_position(
    pos, vec, dls=3.0, probability=0.5, nev=True, unit='meters', step_min=30.
):
    """
    Get a random position based on the current position and vector and
    bound by a function of the Dog Leg Severity. The result is discrete in
    being either straight or along the path the DLS radius, the probability
    of which is defined by probability.

    Parameters
    ----------
        pos: (1, 3) array of floats
            The current position.
        vec: (1, 3) array of floats
            The current vector.
        dls: float
            The desired Dog Leg Severity in degrees per 30 meters or 100 feet.
        probability: float between 0 and 1.
            The probability of the returned position being straight ahead
            versus being along a circular path with the radius defined by dls.
        nev: bool (default: True)
            Indicates whether the pos and vec parameters are provided in the
            [North, East, Vertical] or [x, y, z] coordinate system.
        unit: string (default: 'meters')
            The unit, either 'meters' or 'feet'.
        step_min: float (default: 30)
            The minimum allowed delta_md

    Returns
    -------
        pos_new: (1, 3) array of floats
            A new position.
        vec_new: (1, 3) array of floats
            A new vector
    """
    assert unit in ['meters', 'feet'], "Unit must be 'meters' or 'feet'"
    pos_nev, pos_xyz = utils.process_coords(pos, nev)
    vec_nev, vec_xyz = utils.process_coords(vec, nev)

    if unit == 'meters':
        coeff = 30
    else:
        coeff = 100

    radius = (360 * coeff) / (dls * 2 * np.pi)
    # delta_md = random.random() * radius
    factor_min = step_min / (math.pi * radius * 0.5)
    factor = random.random()
    factor = factor_min if factor < factor_min else factor
    dogleg = factor * math.pi / 2
    delta_md = factor * math.pi * radius * 0.5
    # if delta_md < step_min:
    #     delta_md = step_min
    action = 0 if random.random() < probability else 1

    if action:
        # do some directional drilling
        # dogleg = math.atan(delta_md / radius)
        pos_temp = np.array([
            math.cos(dogleg),
            0.,
            math.sin(dogleg)
        ]) * radius
        pos_temp[0] = radius - pos_temp[0]

        vec_temp = np.array([
            math.sin(dogleg),
            0.,
            math.cos(dogleg)
        ])

        # spin that roulette wheel to see which direction we're heading
        toolface = random.random() * 2 * np.pi
        inc, azi = utils.get_angles(vec_nev, nev=True).reshape(2)
        angles = [
            toolface,
            inc,
            azi
        ]

        # Use the code from those clever chaps over at scipy, I'm sure they're
        # better at writing this code than I am.
        r = R.from_euler('zyz', angles, degrees=False)

        pos_new, vec_new = r.apply(np.vstack((pos_temp, vec_temp)))
        pos_new += pos_nev

    else:
        # hold section
        vec_new = vec_nev
        pos_new = pos_nev + (vec_nev * delta_md)

    if not nev:
        pos_new, vec_new = utils.get_xyz(
            np.vstack((pos_new, vec_new))
        )

    return (pos_new, vec_new, delta_md)


if __name__ == '__main__':
    bit_walk(
        pos=[0., 0., 0.],
        vec=[0., 0., 1.],
        dls_design=3.0,
        toolface_design=0.,
        dls_walk=1.0,
        toolface_walk=270
    )

    print("Done")