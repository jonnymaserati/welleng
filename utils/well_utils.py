import numpy as np
from numpy import cos, sin, tan, radians, degrees, arccos, arctan, arctan2

from well_utils import MinCurve

class Survey:
    def __init__(self, md, inc, azi, start_xyz=[0,0,0], start_nev=[0,0,0], deg=True, unit="meters"):
        self.unit = unit
        self.deg = deg
        self.start_xyz = start_xyz
        self.start_nev = start_nev
        self.md = md
        if self.deg:
            self.inc_rad = radians(inc)
            self.azi_rad = radians(azi)
            self.inc_deg = inc
            self.azi_deg = azi
        else:
            self.inc_rad = inc
            self.azi_rad = azi
            self.inc_deg = degrees(inc)
            self.azi_deg = degrees(azi)
        self.survey_deg = np.array([self.md, self.inc_deg, self.azi_deg]).T
        self.survey_rad = np.array([self.md, self.inc_rad, self.azi_rad]).T

        self._min_curve()
        self._get_nev()
        self.vec = get_vec(self.inc_rad, self.azi_rad, deg=False)

    def _min_curve(self):
        """
        Params
        """
        mc = MinCurve(self.md, self.inc_rad, self.azi_rad, self.start_xyz, self.unit)
        self.dogleg = mc.dogleg
        self.rf = mc.rf
        self.delta_md = mc.delta_md
        self.dls = mc.dls
        self.x, self.y, self.z = mc.poss.T

        # survey_length = len(self.md)

        # md_1 = self.md[:-1]
        # md_2 = self.md[1:] 

        # inc_1 = self.inc_rad[:-1]
        # inc_2 = self.inc_rad[1:]

        # azi_1 = self.azi_rad[:-1]
        # azi_2 = self.azi_rad[1:]

        # dogleg = arccos(
        #     cos(inc_2 - inc_1)
        #     - (sin(inc_1) * sin(inc_2))
        #     * (1 - cos(azi_2 - azi_1))
        # )
        # self.dogleg = np.zeros(survey_length)
        # self.dogleg[1:] = dogleg

        # self.rf = np.ones(survey_length)
        # idx = np.where(self.dogleg != 0)
        # self.rf[idx] = 2 / self.dogleg[idx] * tan(self.dogleg[idx] / 2)

        # delta_md = np.array(md_2) - np.array(md_1)
        # self.delta_md = np.zeros(survey_length)
        # self.delta_md[1:] = delta_md

        # delta_y = (
        #     delta_md
        #     / 2
        #     * (
        #         sin(inc_1) * cos(azi_1)
        #         + sin(inc_2) * cos(azi_2)
        #     )
        #     * self.rf[1:]
        # )
        # self.delta_y = np.zeros(survey_length)
        # self.delta_y[1:] = delta_y
        
        # y = np.cumsum(delta_y) + self.start_xyz[1]
        # self.y = np.hstack((self.start_xyz[1], y))

        # delta_x = (
        #     delta_md
        #     / 2
        #     * (
        #         sin(inc_1) * sin(azi_1)
        #         + sin(inc_2) * sin(azi_2)
        #     )
        #     * self.rf[1:]
        # )
        # self.delta_x = np.zeros(survey_length)
        # self.delta_x[1:] = delta_x

        # x = np.cumsum(delta_x) + self.start_xyz[0]
        # self.x = np.hstack((self.start_xyz[0], x))

        # delta_z = (
        #     delta_md
        #     / 2
        #     * (cos(inc_1) + cos(inc_2))
        #     * self.rf[1:]
        # )
        # self.delta_z = np.zeros(survey_length)
        # self.delta_z[1:] = delta_z

        # z = np.cumsum(delta_z) + self.start_xyz[2]
        # self.z = np.hstack((self.start_xyz[2], z))

        # dls = degrees(dogleg) / delta_md
        # self.dls = np.zeros(survey_length)
        # self.dls[1:] = dls

        # if self.unit == "meters":
        #     self.dls *= 30
        # elif self.unit == "feet":
        #     self.dls *= 100
        # else:
        #     print("Unknown unit, please select meters or feet")

    def _get_nev(self):
        e, n, v = (
            np.array([self.x, self.y, self.z]).T - np.array([self.start_xyz])
        ).T
        self.n, self.e, self.tvd = (np.array([n, e, v]).T + np.array([self.start_nev])).T


def HLA_to_NEV(
    survey, HLA, cov=True
    ):

    trans = transform(survey)

    if cov:
        NEVs = [
            np.dot(np.dot(mat.T, HLA.T[i]), mat) for i, mat in enumerate(trans.T)
        ]
    
        NEVs = np.vstack(NEVs).reshape(-1,3,3).T
    
    else:
        NEVs = [
            np.dot(hla, mat) for hla, mat in list(zip(HLA, trans.T))
        ]
        
    return np.vstack(NEVs).reshape(HLA.shape)


def transform(
    survey
    ):

    inc = np.array(survey[:,1])
    azi = np.array(survey[:,2])
    
    trans = np.array([
        [cos(inc) * cos(azi), -sin(azi), sin(inc) * cos(azi)],
        [cos(inc) * sin(azi), cos(azi), sin(inc) * sin(azi)],
        [-sin(inc), np.zeros_like(inc), cos(inc)]
    ])

    return trans

# def interpolate_survey(survey, index, step=30, unit='meters', deg=True):
#     '''
#     Bla Bla
#     '''
#     if index >= len(survey.md) - 1:
#         return "survey out of range"

#     if step > survey.delta_md[index + 1]:
#         return "x out of range"

#     poss = np.array([survey.x, survey.y, survey.z]).T

#     toolface, radius_temp, dogleg_temp, step_temp = use_magnet(
#         pos=poss[index],
#         vec=survey.vec[index],
#         dls=survey.dls[index + 1],
#         target=poss[index + 1],
#         step=step,
#         unit=unit,
#         deg=True
#         )

#     pos, vec = take_action(
#         poss[index],
#         survey.vec[index],
#         toolface,
#         dogleg_temp,
#         radius_temp,
#         step_temp
#         )

#     if toolface == -1:
#         dls = 0
    
#     inc, azi = get_angles(np.array(vec))[0].T
            
#     return {
#         'poss': pos.reshape(-1,3)[0],
#         'vecs': vec.reshape(-1,3)[0],
#         'dlss': survey.dls[index + 1],
#         'mds': survey.md[index] + step,
#         'incs': inc,
#         'azis': azi,
#         'index': index,
#         'x': step,
#     }

def interpolate(survey, x, index=0):
    """
    Interpolates a point distance x between two survey stations
    using minimum curvature.

    Params
        survey: object
            A survey object with at least two survey stations.
        x: float
            Length along well path from indexed survey station to
            perform the interpolate at. Must be less than length
            to the next survey station.
        index: int
            The index of the survey station from which to interpolate
            from. 
    """
    assert index >= len(survey.md) - 1, "Index is out of range"

    assert x > survey.delta_md[index + 1], "x isout of range"

    # check if it's just a tangent section
    if survey.dogleg[index + 1] == 0:
        azi = survey.azi_rad[index]
        inc = survey.inc_rad[index]

    else:
        # get the vector
        t1 = survey.vec[index]
        t2 = survey.vec[index + 1]

        total_dogleg = survey.dogleg[index + 1]

        dogleg = x * (survey.dogleg[index + 1] / survey.delta_md[index + 1])

        t = (
            (sin(total_dogleg - dogleg) / sin(total_dogleg)) * t1
            + (sin(dogleg) / sin(total_dogleg)) * t2
        )

        inc, azi = get_angles(t)[0]

    s = Survey(
        md=np.array([survey.md[index], survey.md[index] + x], survey.md[index + 1]),
        inc=np.array([survey.inc_rad[index], inc], survey.inc_rad[index + 1]),
        azi=np.array([survey.azi_rad[index], azi, survey.azi_rad[index + 1]]),
        start_xyz=np.array([survey.x, survey.y, survey.z]).T[index],
        start_nev=np.array([survey.n, survey.e, survey.tvd]).T[index],
        deg=False
    )

    return s

def get_vec(inc, azi, r=1, deg=True):
    if deg:
        inc_rad, azi_rad = radians(np.array([inc, azi]))
    else:
        inc_rad = inc
        azi_rad = azi
    y = r * sin(inc_rad) * cos(azi_rad)
    x = r * sin(inc_rad) * sin(azi_rad)
    z = r * cos(inc_rad)
    
    return np.array([x,y,z]).T

def get_angles(vec):
    '''
    Determines the inclination and azimuth of a direction vector.

    Args
        vec -- direction vector

    Returns
        theta - the inclination angle in radians
        phi - the azimuth angle in radians

    '''
    # make sure it's a unit vector
    vec = vec / np.linalg.norm(vec, axis=-1)

    print(f"{vec.shape = }")
    if vec.shape == (3,1):
        print("Stop")
    vec = vec.reshape(-1,3)
    if vec.dtype == 'O':
        vec = np.stack(vec, axis=1)
    xy = vec[:,0]**2 + vec[:,1]**2
    inc = arctan2(np.sqrt(xy), vec[:,2]) # for elevation angle defined from Z-axis down
    azi = (arctan2(vec[:,0], vec[:,1]) + (2 * np.pi)) % (2 * np.pi)
    return np.stack((inc, azi), axis=1)
    

def rotate_z(pos, phi):
    '''
    Rotates a 1x3 array (point or vector) around the z-axis by an angle.

    Args
        pos -- 1x3 array (position or vector) to be rotated.
        phi -- the angle (in radians) of rotations, often referred to as azimuth.

    Returns
        A new 1x3 array (point or vector).

    '''
    if type(phi) == np.ndarray:
        phi = phi[0]
    print(f"{pos = }\t{phi = }")
    Rz = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    return np.matmul(pos, Rz)

def rotate_x(pos, theta):
    '''
    Rotates a 1x3 array (point or vector) around the x-axis by an angle.

    Args
        pos -- 1x3 array (position or vector) to be rotated.
        phi -- the angle (in radians) of rotations, often referred to as inclination.

    Returns
        A new 1x3 array (point or vector).

    '''
    if type(theta) == np.ndarray:
        theta = theta[0]
    Rx = np.array([[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]])
    return np.matmul(pos, Rx)

def use_magnet(pos, vec, dls, target, step=30, unit='meters', deg=True):
    '''
    Determines the toolface required to steer towards a target point from the current position and direction.# 
    
    Args
        pos -- current position
        vec -- current direction
        target -- position of the desired target else the function tries to determine one from the rewards
        
    Returns
        phi -- the toolface angle required steer towards the target, or -1 to indicate going straight
        
    There's an issue when the inclination is small and the required direction is 180 degrees from highside.
    In this scenario, the direction can get stuck in a loop, so the action needs to switch to an azimuth
    (relative to North) direction.
    
    '''

    if dls == 0:
        return -1, 0, 0, step

    # determine the dogleg relative to step and the unit system
    if unit == 'meters':
        dls_step = 30 # assumes dls expressed in deg/30m
    else:
        dls_step = 100 # assumes dls expressed in deg/30m

    if deg:
        dls_rad = radians(dls)
    else:
        dls_rad = dls

    dogleg = dls_rad / dls_step * step
    radius = 1 / dls_rad * dls_step
    
    target = np.array(target)
    
    destiny = np.subtract(target, pos)
    unit_destiny = destiny / np.linalg.norm(destiny)

    if np.allclose(vec, unit_destiny):
        return -1, 0, 0, step
    
    # radius_temp, dogleg_temp = trim_dogleg(target, pos, vec, dls, unit=unit)
    # step_temp = radius_temp * dogleg_temp
    # if step_temp < step:
    #     step = step_temp
    #     radius = radius_temp
    #     dogleg = dogleg_temp

    inc, azi = get_angles(vec)[0].T
    unazi_destiny = rotate_z(destiny, azi)
    uninc_unazi_destiny = rotate_x(unazi_destiny, inc)
    theta, phi = get_angles(uninc_unazi_destiny)[0].T
    
    return np.degrees(phi), radius, dogleg, step


def get_toolface(pos, vec, target):
    target = np.array(target)
    
    destiny = np.subtract(target, pos)
    unit_destiny = destiny / np.linalg.norm(destiny)

    # if np.allclose(vec, unit_destiny):
    #     return -1

    inc, azi = get_angles(vec)[0].T
    unazi_destiny = rotate_z(destiny, azi)
    uninc_unazi_destiny = rotate_x(unazi_destiny, inc)
    theta, phi = get_angles(uninc_unazi_destiny)[0].T

    return phi


def go_straight(pos, vec, step):
    '''
    Move from the current position in a straight line, maintaining the current direction.

    Args
        pos -- the current position
        vec -- the current direction vector

    Returns
        A new position and its direction vector
    '''

    theta, phi = get_angles(vec)[0].T

    # we follow a 3 step approach:
    # -- 1. in a rotated coordinate system, move down
    delta_pos = np.array([0, 0, step])
    # -- 2. then rotate over x
    delta_pos_inc = rotate_x(delta_pos, -theta)
    # -- 3. finally rotate over z
    delta_pos_inc_azi = rotate_z(delta_pos_inc, -phi)

    # add delta vector to start position
    new_pos = pos + delta_pos_inc_azi
    # direction vector remains the same
    new_vec = vec

    return new_pos, new_vec

def turn(pos, vec, dogleg_rad, radius, toolface):
    '''
    Move from the current position in a circular arc.

    Args
        pos -- the current position
        vec -- the current direction vector
        dogleg_rad -- the dogleg of the turn
        radius -- the curvature radius of the turn
        toolface -- the desired toolface setting of the drilling assembly

    Returns
        A new position and its direction vector

    '''

    theta, phi = get_angles(vec)[0].T

    # we follow a 3 step approach:
    # -- 1. assume current position is at the origin and the current direction vector is [0,0,1]
    # --    then the new position is along an arc in the negative y direction
    # --    so first, determine new position and direction vector relative to current position

    ### optimised for this env ###
    delta_pos = np.array([np.array([0.0]), np.array([-radius + radius * np.cos(dogleg_rad)]), np.array([radius * np.sin(dogleg_rad)])]).reshape(-1,3)
    delta_vec = np.array([np.array([0.0]), np.array([-np.sin(dogleg_rad)]), np.array([np.cos(dogleg_rad)])]).reshape(-1,3)
#     delta_pos = delta_pos
#     delta_vec = delta_vec

    # -- 2. next, rotate the new point to the requested toolface setting
    # --    then rotate and move the new position from object space to local space
    # --    so that it aligns with the current point direction vector

    delta_pos_toolface = rotate_z(delta_pos, np.radians(-toolface) + np.pi)

#     delta_pos_toolface = delta_pos_toolface_lookup[np.where(self.all_actions == toolface)][0]
    delta_pos_toolface_inc = rotate_x(delta_pos_toolface, -theta)
    delta_pos_toolface_inc_azi = rotate_z(delta_pos_toolface_inc, -phi)

    # -- 3. finally, determine the new direction vector of new position in local space
    # --    and apply the same rotations and movements as carried out for the new position above

    delta_vec_toolface = rotate_z(delta_vec, np.radians(-toolface) + np.pi)

#     delta_vec_toolface = delta_vec_toolface_lookup[np.where(self.all_actions == toolface)][0]
    delta_vec_toolface_inc = rotate_x(delta_vec_toolface, -theta)
    delta_vec_toolface_inc_azi = rotate_z(delta_vec_toolface_inc, -phi)

    # add delta vector to start position
    new_pos = pos + delta_pos_toolface_inc_azi
    # and update direction vector
    new_vec = delta_vec_toolface_inc_azi

    return new_pos, new_vec

def take_action(pos, vec, action, dogleg_rad, radius, step):
    '''
    Takes an action
    '''

    # adjust moves counter
#     self.moves_remaining -= 1

    # go straight
    if action < 0:
        new_pos, new_vec = go_straight(pos, vec, step)
    # turn
    else:
        # new_pos, new_vec = turn(pos, vec, dogleg_rad/30*step, radius, action)
        new_pos, new_vec = turn(pos, vec, dogleg_rad, radius, action)

    return new_pos, new_vec


def NEV_to_HLA(survey, NEV, cov=True):

    trans = transform(survey)

    if cov:
        HLAs = [
            np.dot(np.dot(t, NEV.T[i]), t.T) for i, t in enumerate(trans.T)
        ]
        
        HLAs = np.vstack(HLAs).reshape(-1,3,3).T
        
    else:
        HLAs = [
            np.dot(NEV, t.T) for i, t in enumerate(trans.T)
        ]
        
    return HLAs