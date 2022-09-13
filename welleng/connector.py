from copy import copy, deepcopy

import numpy as np
from scipy.spatial import distance

try:
    from numba import njit
    NUMBA = True
except ImportError:
    NUMBA = False

from .node import Node, get_node_params
from .utils import _get_angles, dls_from_radius, get_angles, get_vec


class Connector:
    def __init__(
        self,
        node1=None,
        node2=None,
        pos1=None,
        vec1=None,
        inc1=None,
        azi1=None,
        md1=0,
        dls_design=3.0,
        dls_design2=None,
        md2=None,
        pos2=None,
        vec2=None,
        inc2=None,
        azi2=None,
        degrees=True,
        unit='meters',
        min_error=1e-5,
        delta_dls=0.1,
        min_tangent=0.,
        max_iterations=1_000,
        force_min_curve=False,
        closest_approach=False
    ):

        """
        A class to provide a fast, efficient method for determining well
        bore section geometry using the appropriate method based upon the
        provided parameters, with the intent of assisting machine learning
        fitness functions. Interpolation between the returned control points
        can be performed posthumously - attempts are made to keep processing to
        a minimum in the event that the Connector is being used for machine
        learning.

        Only specific combinations of input data are permitted, e.g. if you
        input both a start vector AND a start inc and azi you'll get an error
        (which one is correct after all?). Think about what you're trying to
        achieve and provide the data that will facilitate that outcome.

        Parameters
        ----------
        pos1: (3) list or array of floats (default: [0,0,0])
            Start position in NEV coordinates.
        vec1: (3) list or array of floats or None (default: None)
            Start position unit vector in NEV coordinates.
        inc1: float or None (default: None)
            Start position inclination.
        azi2: float or None (default: None)
            Start position azimuth.
        md1: float or None (default: None)
            Start position measured depth.
        dls_design: float (default: 3.0)
            The desired Dog Leg Severity (DLS) for the (first) curved
            section in degrees per 30 meters or 100 feet.
        dls_design2: float or None (default: None)
            The desired DLS for the second curve section in degrees per
            30 meters or 100 feet. If set to None then `dls_design` will
            be the default value.
        md2: float or None (default: None)
            The measured depth of the target position.
        pos2: (3) list or array of floats or None (default: None)
            The position of the target in NEV coordinates.
        vec2: (3) list or array of floats or None (default: None)
            The target unit vector in NEV coordinates.
        inc1: float or None (default: None)
            The inclination at the target position.
        azi2: float or None (default: None)
            The azimuth at the target position.
        degrees: boolean (default: True)
            Indicates whether the input angles (inc, azi) are in degrees
            (True) or radians (False).
        unit: string (default: 'meters')
            Indicates the distance unit, either 'meters' or 'feet'.
        min_error: float (default: 1e-5):
            Infers the error tolerance of the results and is used to set
            iteration stops when the desired error tolerance is met. Value
            must be less than 1. Use with caution as the code may
            become unstable if this value is changed.
        delta_radius: float (default: 20)
            The delta radius (first curve and second curve sections) used
            as an iteration stop when balancing radii. If the resulting
            delta radius yielded from `dls_design` and `dls_design2` is
            larger than `delta_radius`, then `delta_radius` defaults to
            the former.
        delta_dls: float (default: 0.1)
            The delta dls (first curve and second curve sections) used as an
            iteration stop when balancing radii, i.e. if the dls of the second
            section is within 0.1 deg/30m of the first curve section then the
            section is considered balanced and no further iterations are
            performed. Setting this value too low will likely result in hitting
            the recursion limit.
        min_tangent: float (default: 10)
            The minimum tangent length in the `curve_hold_curve` method
            used to mitigate instability during iterations (where the
            tangent section approaches or equals 0).
        max_iterations: int (default: 1000)
            The maximum number of iterations before giving up trying to
            fit a `curve_hold_curve`. This number is limited by Python's
            depth of recursion, but if you're hitting the default stop
            then consider changing `delta_radius` and `min_tangent` as
            your expectations may be unrealistic (this is open source
            software after all!)

        Results
        -------
        connector: welleng.connector.Connector object
        """
        pos1 = pos1 or [0., 0., 0.]

        if node1 is not None:
            pos1, vec1, md1 = get_node_params(
                node1
            )
        if node2 is not None:
            pos2, vec2, md2 = get_node_params(
                node2
            )

        # Set up a lookup dictionary to use with the logic to determine
        # what connector method to deploy. Use a binary string to
        # represent the inputs in the order:
        # (md2, inc2, azi2, pos2, vec2)
        # Initially I used boolean logic, but it quickly became non-
        # transparent and difficult to debug.
        self._get_initial_methods()

        # METHODS = [
        #     'hold',
        #     'curve_hold',
        #     'min_dist_to_target',
        #     'min_curve_to_target',
        #     'curve_hold_curve',
        #     'min_curve'
        # ]

        # quick check that inputs are workable and if not some steer to
        # the user.
        self._check_input_data(
            data_station_1=(pos1, vec1, md1, inc1, azi1),
            data_station_2=(pos2, vec2, md2, inc2, azi2),
            dls_design=dls_design,
            min_error=min_error
        )

        # figure out what method is required to connect the points
        target_input = convert_target_input_to_booleans(
            md2, inc2, azi2, pos2, vec2
        )

        self.force_min_curve = force_min_curve
        if self.force_min_curve:
            self.initial_method = 'min_curve_or_hold'
        else:
            self.initial_method = self.initial_methods[target_input]

        # do some more initialization stuff
        self.min_error = min_error
        self.min_tangent = min_tangent
        self.iterations = 0
        self.max_iterations = max_iterations
        self.errors = []
        self.radii = []
        self.dogleg_old, self.dogleg2_old = 0, 0
        self.dist_curve2 = 0
        self.pos1 = np.array(pos1)

        self.pos2, self.vec2, self.inc2, self.azi2, self.md2 = (
            None, None, None, None, None
        )

        # fill in the input data gaps
        self._set_inc1_azi1(vec1, inc1, azi1, degrees)

        self.md1 = md1
        self.pos_target = None if pos2 is None else np.array(pos2).reshape(3)
        self.md_target = md2

        self._set_target_inc_and_azi(vec2, inc2, azi2, degrees)

        self.unit = unit
        if self.unit == 'meters':
            self.denom = 30
        else:
            self.denom = 100

        if degrees:
            self.dls_design = np.radians(dls_design)
            if dls_design2:
                self.dls_design2 = np.radians(dls_design2)
        else:
            self.dls_design = dls_design
            if dls_design2:
                self.dls_design2 = dls_design2
        if not dls_design2:
            self.dls_design2 = self.dls_design

        self.radius_design = self.denom / self.dls_design
        self.radius_design2 = self.denom / self.dls_design2

        self.delta_dls = delta_dls

        # some more initialization stuff
        self.tangent_length = None
        self.dogleg2 = None

        self.pos3, self.vec3, self.inc3, self.azi3, self.md3 = (
            None, None, None, None, None
        )
        self.radius_critical, self.radius_critical2 = np.inf, np.inf
        self.closest_approach = closest_approach

        # Things fall apart if the start and end vectors exactly equal
        # one another, so need to check for this and if this is the
        # case, modify the end vector slightly. This is a lazy way of
        # doing this, but it's fast. Probably a more precise way would
        # be to split the dogleg in two, but that's more hassle than
        # it's worth.
        if (
            self.vec_target is not None
            and np.array_equal(self.vec_target, self.vec1 * -1)
        ):
            (
                self.vec_target,
                self.inc_target,
                self.azi_target
            ) = mod_vec(self.vec_target, self.min_error)

        # properly figure out the method
        self._get_method()

        # and finally, actually do something...
        self._use_method()

        self._get_nodes()

    def _check_input_data(self, data_station_1, data_station_2, dls_design, min_error):

        pos1, vec1, md1, inc1, azi1 = data_station_1
        pos2, vec2, md2, inc2, azi2 = data_station_2

        assert vec1 is not None or (inc1 is not None and azi1 is not None), (
            "Require either vec1 or (inc1 and azi1)"
        )
        if vec1 is not None:
            assert inc1 is None and azi1 is None, (
                "Either vec1 or (inc1 and azi1)"
            )
        if (inc1 is not None or azi1 is not None):
            assert vec1 is None, "Either vec1 or (inc1 and azi1)"

        assert (
            md2 is not None
            or pos2 is not None
            or vec2 is not None
            or inc2 is not None
            or azi2 is not None
        ), "Missing target parameters"

        if vec2 is not None:
            assert not (inc2 or azi2), "Either vec2 or (inc2 and azi2)"
        if (inc2 is not None or azi2 is not None):
            assert vec2 is None, "Either vec2 or (inc2 and azi2)"
        if md2 is not None:
            assert pos2 is None, "Either md2 or pos2"
            assert md2 >= md1, "md2 must be larger than md1"

        if dls_design is None:
            dls_design = 3.0
        else:
            assert dls_design > 0, "dls_design must be greater than zero"
        assert min_error < 1, "min_error must be less than 1.0"

    def _set_inc1_azi1(self, vec1, inc1, azi1, degrees):
        if inc1 is not None and azi1 is not None:
            if degrees:
                self.inc1 = np.radians(inc1)
                self.azi1 = np.radians(azi1)
            else:
                self.inc1 = inc1
                self.azi1 = azi1
            self.vec1 = np.array(get_vec(
                self.inc1, self.azi1, nev=True, deg=False
            )).reshape(3)
        else:
            self.vec1 = np.array(vec1).reshape(3)
            self.inc1, self.azi1 = get_angles(self.vec1, nev=True).reshape(2)

    def _set_target_inc_and_azi(self, vec2, inc2, azi2, degrees):

        if vec2 is not None:
            self.vec_target = np.array(vec2).reshape(3)
            self.inc_target, self.azi_target = get_angles(
                self.vec_target,
                nev=True
            ).reshape(2)

        elif inc2 is not None and azi2 is not None:
            if degrees:
                self.inc_target = np.radians(inc2)
                self.azi_target = np.radians(azi2)
            else:
                self.inc_target = inc2
                self.azi_target = azi2
            self.vec_target = get_vec(
                self.inc_target, self.azi_target, nev=True, deg=False
            ).reshape(3)

        elif inc2 is None and azi2 is None:
            self.inc_target, self.azi_target, self.vec_target = (
                self.inc1, self.azi1, self.vec1
            )

        elif inc2 is None:
            self.inc_target = self.inc1
            if degrees:
                self.azi_target = np.radians(azi2)
            else:
                self.azi_target = azi2
            self.vec_target = get_vec(
                self.inc_target, self.azi_target, nev=True, deg=False
            ).reshape(3)

        elif azi2 is None:
            self.azi_target = self.azi1
            if degrees:
                self.inc_target = np.radians(inc2)
            else:
                self.inc_target = inc2
            self.vec_target = get_vec(
                self.inc_target, self.azi_target, nev=True, deg=False
            ).reshape(3)

        else:
            self.vec_target = vec2
            self.inc_target = inc2
            self.azi_target = azi2

    def _get_nodes(self):
        self.node_start = Node(
            pos=self.pos1.reshape(3),
            vec=self.vec1.reshape(3),
            md=self.md1
        )
        self.node_end = Node(
            pos=self.pos_target.reshape(3),
            vec=self.vec_target.reshape(3),
            md=self.md_target
        )

    def _min_dist_to_target(self):
        (
            self.tangent_length,
            self.dogleg
        ) = min_dist_to_target(self.radius_design, self.distances)
        self.dogleg = check_dogleg(self.dogleg)
        self.dist_curve, self.func_dogleg = get_curve_hold_data(
            self.radius_design, self.dogleg
        )
        self.vec_target = get_vec_target(
            self.pos1,
            self.vec1,
            self.pos_target,
            self.tangent_length,
            self.dist_curve,
            self.func_dogleg
        )
        self._get_angles_target()
        self._get_md_target()
        self.pos2 = (
            self.pos_target - (
                self.tangent_length * self.vec_target
            )
        )
        self.md2 = self.md1 + abs(self.dist_curve)
        self.md_target = self.md2 + self.tangent_length
        self.vec2 = self.vec_target

    def _min_curve_to_target(self):
        (
            self.tangent_length,
            self.radius_critical,
            self.dogleg
        ) = min_curve_to_target(self.distances)
        self.dogleg = check_dogleg(self.dogleg)
        self.dist_curve, self.func_dogleg = get_curve_hold_data(
            min(self.radius_design, self.radius_critical), self.dogleg
        )
        self.vec_target = get_vec_target(
            self.pos1,
            self.vec1,
            self.pos_target,
            self.tangent_length,
            self.dist_curve,
            self.func_dogleg
        )
        self._get_angles_target()
        self._get_md_target()

    def _use_method(self):
        if self.method == 'hold':
            self._hold()

        elif self.method == 'min_curve':
            self._min_curve()

        elif self.method == 'curve_hold_curve':
            self.pos2_list, self.pos3_list = [], [deepcopy(self.pos_target)]
            self.vec23 = [np.array([0., 0., 0.])]
            self.delta_radius_list = []
            self._target_pos_and_vec_defined()

        else:
            self.distances = self._get_distances(
                self.pos1, self.vec1, self.pos_target
            )

            if self.radius_design <= get_radius_critical(
                self.radius_design, self.distances, self.min_error
            ):
                self.method = 'min_dist_to_target'
                self._min_dist_to_target()

            else:
                if self.closest_approach:
                    self.method = 'min_curve_to_target'
                    self._closest_approach()

                else:
                    self.method = 'min_curve_to_target'
                    self._min_curve_to_target()

    def _get_method(self):
        assert self.initial_method not in [
            'no_input',
            'vec_and_inc_azi',
            'md_and_pos'
        ], f"{self.initial_method}"
        if self.initial_method == 'hold':
            self.method = 'hold'
        elif self.initial_method[-8:] == '_or_hold':
            if np.array_equal(self.vec_target, self.vec1):
                if self.pos_target is None:
                    self.method = 'hold'
                elif np.allclose(
                        self.vec_target,
                        (self.pos_target - self.pos1)
                        / np.linalg.norm(self.pos_target - self.pos1)
                ):
                    self.method = 'hold'
                else:
                    self.method = self.initial_method[:-8]
            else:
                self.method = self.initial_method[:-8]
        else:
            self.method = self.initial_method

    def _get_initial_methods(self):
        # TODO: probably better to load this in from a yaml file
        # [md2, inc2, azi2, pos2, vec2] forms the booleans
        self.initial_methods = {
            '00000': 'no_input',
            '00001': 'min_curve_or_hold',
            '00010': 'curve_hold_or_hold',
            '00011': 'curve_hold_curve_or_hold',
            '00100': 'min_curve_or_hold',
            '00101': 'vec_and_inc_azi',
            '00110': 'curve_hold',
            '00111': 'vec_and_inc_azi',
            '01000': 'min_curve_or_hold',
            '01001': 'vec_and_inc_azi',
            '01010': 'curve_hold_or_hold',
            '01011': 'vec_and_inc_azi',
            '01100': 'min_curve_or_hold',
            '01101': 'vec_and_inc_azi',
            '01110': 'curve_hold_curve_or_hold',
            '01111': 'vec_and_inc_azi',
            '10000': 'hold',
            '10001': 'min_curve_or_hold',
            '10010': 'md_and_pos',
            '10011': 'md_and_pos',
            '10100': 'min_curve_or_hold',
            '10101': 'vec_and_inc_azi',
            '10110': 'md_and_pos',
            '10111': 'md_and_pos',
            '11000': 'min_curve_or_hold',
            '11001': 'vec_and_inc_azi',
            '11010': 'md_and_pos',
            '11011': 'md_and_pos',
            '11100': 'min_curve_or_hold',
            '11101': 'vec_and_inc_azi',
            '11110': 'md_and_pos',
            '11111': 'md_and_pos'
        }

    def _closest_approach(self):
        vec_pos1_pos_target = self.pos_target - self.pos1
        vec_pos1_pos_target /= np.linalg.norm(vec_pos1_pos_target)

        cross_product = np.cross(vec_pos1_pos_target, self.vec1)
        cross_product /= np.linalg.norm(cross_product)

        factor = cross_product / vec_pos1_pos_target
        factor /= abs(factor)

        cc = (
            self.pos1 + cross_product * factor * self.radius_design
        )

        cc_pos_target = self.pos_target - cc
        cc_pos_target /= np.linalg.norm(cc_pos_target)

        self.pos_target_original = copy(self.pos_target)

        self.pos_target = cc + cc_pos_target * self.radius_design

        # recalculate self.distances with new self.pos_target
        self.distances = self._get_distances(
            self.pos1, self.vec1, self.pos_target
        )

        self._min_curve_to_target()

    def _min_curve(self):
        self.dogleg = get_dogleg(
            self.inc1, self.azi1, self.inc_target, self.azi_target
        )

        self.dogleg = check_dogleg(self.dogleg)
        if self.md_target is None:
            self.md2 = None
            self.dist_curve, self.func_dogleg = get_curve_hold_data(
                self.radius_design, self.dogleg
            )
            self.md_target = self.md1 + abs(self.dist_curve)
            self.pos_target = get_pos(
                self.pos1,
                self.vec1,
                self.vec_target,
                self.dist_curve,
                self.func_dogleg
            ).reshape(3)
        else:
            self.radius_critical = abs(
                (self.md_target - self.md1) / self.dogleg
            )
            if (
                self.radius_critical > self.radius_design
                or (
                    np.around(self.dogleg, decimals=5)
                    == np.around(np.pi, decimals=5)
                )
            ):
                self.md2 = (
                    self.md1
                    + min(self.radius_design, self.radius_critical)
                    * self.dogleg
                )
                (
                    self.inc2, self.azi2, self.vec2
                ) = self.inc_target, self.azi_target, self.vec_target
                self.dist_curve, self.func_dogleg = get_curve_hold_data(
                    min(self.radius_design, self.radius_critical),
                    self.dogleg
                )
                self.pos2 = get_pos(
                    self.pos1,
                    self.vec1,
                    self.vec2,
                    self.dist_curve,
                    self.func_dogleg
                ).reshape(3)
                self.pos_target = self.pos2 + (
                    self.vec2 * (self.md_target - self.md2)
                )
            else:
                self.dist_curve, self.func_dogleg = get_curve_hold_data(
                    self.radius_critical, self.dogleg
                )
                self.md2 = None
                self.pos_target = get_pos(
                    self.pos1,
                    self.vec1,
                    self.vec_target,
                    self.dist_curve,
                    self.func_dogleg
                ).reshape(3)

    def _hold(self):
        if self.pos_target is None:
            self.pos_target = (
                self.pos1 + self.vec1 * (self.md_target - self.md1)
            )
        if self.md_target is None:
            self.md_target = (
                np.linalg.norm(self.pos_target - self.pos1)
                + self.md1
            )
        self.dls, self.dls2 = 0.0, 0.0

    def _get_angles_target(self):
        self.inc_target, self.azi_target = get_angles(
            self.vec_target, nev=True
        ).reshape(2)

    def _get_md_target(self):
        self.md_target = (
            self.dist_curve
            + self.tangent_length
            + self.dist_curve2
            + self.md1
        )

    def _get_pos2(self, pos1, vec1, pos_target, radius):
        distances = self._get_distances(pos1, vec1, pos_target)

        radius_temp = get_radius_critical(
            self.radius_design,
            distances,
            self.min_error
        )
        if radius_temp > radius:
            radius_temp = radius
            assert self.radius_critical > 0

        (
            tangent_length,
            dogleg
        ) = min_dist_to_target(
            radius_temp,
            distances
        )

        dogleg = check_dogleg(dogleg)

        dist_curve, func_dogleg = get_curve_hold_data(
            radius_temp,
            dogleg
        )
        vec3 = get_vec_target(
            pos1,
            vec1,
            pos_target,
            tangent_length,
            dist_curve,
            func_dogleg
        )

        tangent_temp = self._get_tangent_temp(tangent_length)

        pos2 = (
            pos_target - (
                tangent_temp * vec3
            )
        )
        return pos2

    def _target_pos_and_vec_defined(self):
        """
        Function for fitting a curve, hold, curve path between a pair of
        points with vectors.

        It's written in this odd way to allow a solver function like scipy's
        optimize to solve it, but so far I've not had much success using
        these.

        If the curve sections can't be achieved with the design DLS values,
        this function will iterate until the two curve section DLSs are
        approximately balances (in DLS terms) within the prescribed delta_dls
        parameter.
        """
        args = (self, None, [0., 0., 0.])
        minimize_target_pos_and_vec_defined(
            *(
                ([
                    self.radius_design,
                    self.radius_design2
                ],)
                + args + (True,)
            )
        )
        if not all((
            self.radius_critical > self.radius_design,
            self.radius_critical2 > self.radius_design2
        )):
            while True and self.iterations <= self.max_iterations:
                self.radius_critical = (
                    (self.md_target - self.md1)
                    / (abs(self.dogleg) + abs(self.dogleg2))
                )
                assert self.radius_critical >= 0
                self.radius_critical2 = self.radius_critical
                args = (self, deepcopy(self.pos3), deepcopy(self.vec23[-1]))
                minimize_target_pos_and_vec_defined(
                    *(
                        ([
                            self.radius_design,
                            self.radius_design2
                        ],)
                        + args + (True,)
                    )
                )
                if abs(
                    self.dls - self.dls2
                ) < self.delta_dls:
                    break
            self._happy_finish()
            return

        self._happy_finish()

    def _happy_finish(self):
        # get pos1 to pos2 curve data
        self.dist_curve, self.func_dogleg = get_curve_hold_data(
            min(self.radius_critical, self.radius_design),
            self.dogleg
        )

        self.vec3 = get_vec_target(
            self.pos1,
            self.vec1,
            self.pos3,
            self.tangent_length,
            self.dist_curve,
            self.func_dogleg
        )

        self.vec2 = self.vec3

        self.pos2 = get_pos(
            self.pos1,
            self.vec1,
            self.vec3,
            self.dist_curve,
            self.func_dogleg
        ).reshape(3)

        self.md2 = self.md1 + abs(self.dist_curve)

        self.dist_curve2, self.func_dogleg2 = get_curve_hold_data(
            min(self.radius_critical2, self.radius_design2),
            self.dogleg2
        )

        self.pos3 = get_pos(
            self.pos_target,
            self.vec_target * -1,
            self.vec3 * -1,
            self.dist_curve2,
            self.func_dogleg2
        ).reshape(3)

        tangent_length = np.linalg.norm(
            self.pos3 - self.pos2
        )

        self.md3 = self.md2 + tangent_length

        self.md_target = self.md3 + abs(self.dist_curve2)

    def interpolate(self, step=30):
        return interpolate_well([self], step)

    def _get_tangent_temp(self, tangent_length):
        if np.isnan(tangent_length):
            tangent_temp = self.min_tangent
        else:
            tangent_temp = max(tangent_length, self.min_tangent)

        return tangent_temp

    def _mod_pos(self, pos):
        pos_rand = np.random.random(3)  # * self.delta_radius
        pos += pos_rand

    def _get_distances(self, pos1, vec1, pos_target):
        # When initializing a `curve_hold_curve` and pos_target is directly
        # ahead it can cause issues (it's a hold and not a curve). So this code
        # checks for that condition and if it's the case, will move the
        # target_pos a sufficient amount to prevent issues.
        vec_temp = np.array(pos_target - pos1)
        with np.errstate(divide='ignore', invalid='ignore'):
            vec_temp = vec_temp / np.linalg.norm(vec_temp)
        if np.array_equal(vec1, vec_temp):
            self._mod_pos(pos_target)
        if np.allclose(pos1, pos_target):
            return (0, 0, 0)

        else:
            dist_to_target = (
                np.linalg.norm((pos_target - pos1))
            )

            dist_perp_to_target = (
                np.dot((pos_target - pos1), vec1)
            )
            if dist_perp_to_target > dist_to_target:
                # since a tolerance is being used, occasionally things can go
                # wrong and need to be caught.
                dist_perp_to_target = dist_to_target

            dist_norm_to_target = (
                (
                    dist_to_target ** 2
                    - dist_perp_to_target ** 2
                ) ** 0.5
            )

            return (
                dist_to_target,
                dist_perp_to_target,
                dist_norm_to_target
            )


def minimize_target_pos_and_vec_defined(
    x, c, pos3=None, vec_old=None, result=False
):
    vec_old = vec_old if vec_old is not None else [0., 0., 0.]
    radius1, radius2 = x
    if pos3 is None:
        pos2_init = c._get_pos2(c.pos1, c.vec1, c.pos_target, radius1)
        pos3_init = c._get_pos2(
            c.pos_target, c.vec_target * -1, c.pos1, radius2
        )
        c.pos3 = pos2_init + (
            pos3_init - pos2_init
        ) / 2
    c.distances1 = c._get_distances(c.pos1, c.vec1, c.pos3)

    radius_temp1 = get_radius_critical(
        radius1,
        c.distances1,
        c.min_error
    )
    if radius_temp1 < c.radius_critical:
        c.radius_critical = radius_temp1
        assert c.radius_critical >= 0

    radius_effective1 = min(radius1, c.radius_critical)

    (
        c.tangent_length,
        c.dogleg
    ) = min_dist_to_target(
        radius_effective1,
        c.distances1
    )

    c.dogleg = check_dogleg(c.dogleg)

    c.dist_curve, c.func_dogleg = get_curve_hold_data(
        radius_effective1,
        c.dogleg
    )
    c.vec3 = get_vec_target(
        c.pos1,
        c.vec1,
        c.pos3,
        c.tangent_length,
        c.dist_curve,
        c.func_dogleg
    )

    tangent_temp1 = c._get_tangent_temp(c.tangent_length)

    c.pos2 = (
        c.pos3 - (
            tangent_temp1 * c.vec3
        )
    )

    c.distances2 = c._get_distances(
        c.pos_target,
        c.vec_target * -1,
        c.pos2
    )

    radius_temp2 = get_radius_critical(
        radius2,
        c.distances2,
        c.min_error
    )
    if radius_temp2 < c.radius_critical2:
        c.radius_critical2 = radius_temp2
        assert c.radius_critical2 >= 0

    radius_effective2 = min(radius2, c.radius_critical2)

    (
        c.tangent_length2,
        c.dogleg2
    ) = min_dist_to_target(
        radius_effective2,
        c.distances2
    )

    c.dogleg2 = check_dogleg(c.dogleg2)

    c.dist_curve2, c.func_dogleg2 = get_curve_hold_data(
        radius_effective2,
        c.dogleg2
    )
    c.vec2 = get_vec_target(
        c.pos_target,
        c.vec_target * -1,
        c.pos2,
        c.tangent_length2,
        c.dist_curve2,
        c.func_dogleg2
    )

    tangent_temp2 = c._get_tangent_temp(c.tangent_length2)

    c.pos3 = (
        c.pos2 - (
            tangent_temp2 * c.vec2
        )
    )

    vec23_denom = np.linalg.norm(c.pos3 - c.pos2)
    if vec23_denom == 0:
        c.vec23.append(np.array([0., 0., 0.]))
    else:
        c.vec23.append((c.pos3 - c.pos2) / vec23_denom)

    c.error = np.allclose(
        c.vec23[-1],
        vec_old,
        equal_nan=True,
        rtol=c.min_error * 10,
        atol=c.min_error * 0.1
    )

    c.errors.append(c.error)
    c.pos3_list.append(c.pos3)
    c.pos2_list.append(c.pos2)
    c.md_target = (
        c.md1 + c.dist_curve + tangent_temp2 + c.dist_curve2
    )
    c.delta_radius_list.append(
        abs(c.radius_critical - c.radius_critical2)
    )
    c.dls = max(
        dls_from_radius(c.radius_design),
        dls_from_radius(c.radius_critical)
    )
    c.dls2 = max(
        dls_from_radius(c.radius_design2),
        dls_from_radius(c.radius_critical2)
    )

    if c.error:
        if result:
            return c
        result = 0.
        if radius_temp2 < c.radius_design2:
            result += c.radius_design2 - radius_temp2
        if radius_temp1 < c.radius_design:
            result += c.radius_design - radius_temp1
        return result
    else:
        c.iterations += 1
        return minimize_target_pos_and_vec_defined(
            x, c, c.pos3, c.vec23[-1], result
        )


def check_dogleg(dogleg):
    # the code assumes angles are positive and clockwise
    if dogleg < 0:
        dogleg_new = dogleg + 2 * np.pi
        return dogleg_new
    else:
        return dogleg


def mod_vec(vec, error=1e-5):
    # if it's not working then twat it with a hammer
    vec_mod = vec * np.array([1, 1, 1 - error])
    vec_mod /= np.linalg.norm(vec_mod)
    inc_mod, azi_mod = get_angles(vec_mod, nev=True).T

    return vec_mod, inc_mod, azi_mod


def _get_xyz(pos):
    n, e, v = pos
    return np.array([e, n, v]).reshape(-1, 3)


def get_pos(pos1, vec1, vec2, dist_curve, func_dogleg):
    inc1, azi1 = _get_angles(_get_xyz(vec1)).reshape(2)
    inc2, azi2 = _get_angles(_get_xyz(vec2)).reshape(2)

    pos2 = pos1 + (
        (
            dist_curve * func_dogleg / 2
        ) * np.array([
            np.sin(inc1) * np.cos(azi1) + np.sin(inc2) * np.cos(azi2),
            np.sin(inc1) * np.sin(azi1) + np.sin(inc2) * np.sin(azi2),
            np.cos(inc1) + np.cos(inc2)
        ]).T
    )

    return pos2


def get_vec_target(
    pos1,
    vec1,
    pos_target,
    tangent_length,
    dist_curve,
    func_dogleg
):
    if dist_curve == 0:
        return vec1

    vec_target = (
        (
            pos_target - pos1 - (
                dist_curve
                * func_dogleg
            ) / 2 * vec1
        ) / (
            (
                dist_curve * func_dogleg / 2
            ) + tangent_length
        )
    )

    vec_target /= np.linalg.norm(vec_target)

    return vec_target


def get_curve_hold_data(radius, dogleg):
    dist_curve = radius * dogleg
    func_dogleg = shape_factor(dogleg)

    return (
        dist_curve,
        func_dogleg
    )


def shape_factor(dogleg):
    """
    Function to determine the shape factor of a dogleg.

    Parameters
    ----------
        dogleg: float
            The dogleg angle in radians of a curve section.
    """
    if dogleg == 0:
        return 0
    else:
        return np.tan(dogleg / 2) / (dogleg / 2)


def min_dist_to_target(radius, distances):
    """
    Calculates the control points for a curve and hold section from the
    start position and vector to the target position.
    """
    (
        dist_to_target,
        dist_perp_to_target,
        dist_norm_to_target
    ) = distances

    tangent_length = (
        dist_to_target ** 2
        - 2 * radius * dist_norm_to_target
    ) ** 0.5

    # determine the dogleg angle of the curve section
    dogleg = 2 * np.arctan2(
        (dist_perp_to_target - tangent_length),
        (
            2 * radius - dist_norm_to_target
        )
    )

    return tangent_length, dogleg


def min_curve_to_target(distances):
    """
    Calculates the control points for a curve section from the start
    position and vector to the target position which is not achievable with
    the provided dls_design.
    """
    if distances == (0., 0., 0,):
        return (
            0.,
            np.inf,
            0.
        )

    (
        dist_to_target,
        dist_perp_to_target,
        dist_norm_to_target
    ) = distances

    if dist_norm_to_target == 0.:
        radius_critical = np.inf
    else:
        radius_critical = (
            dist_to_target ** 2 / (
                2 * dist_norm_to_target
            )
        )
    assert radius_critical > 0

    dogleg = (
        2 * np.arctan2(
            dist_norm_to_target,
            dist_perp_to_target
        )
    )

    tangent_length = 0

    return (
        tangent_length,
        radius_critical,
        dogleg
    )


def get_radius_critical(radius, distances, min_error):
    (
        dist_to_target,
        dist_perp_to_target,
        dist_norm_to_target
    ) = distances

    if dist_norm_to_target == 0:
        return 0

    radius_critical = (
        dist_to_target ** 2 / (
            2 * dist_norm_to_target
        )
    ) * (1 - min_error)

    assert radius_critical > 0

    return radius_critical


def angle(vec1, vec2, acute=True):
    angle = np.arccos(
        np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
    )

    if acute:
        return angle

    else:
        return 2 * np.pi - angle


def get_dogleg(inc1, azi1, inc2, azi2):
    dogleg = (
        2 * np.arcsin(
            (
                np.sin((inc2 - inc1) / 2) ** 2
                + np.sin(inc1) * np.sin(inc2)
                * np.sin((azi2 - azi1) / 2) ** 2
            ) ** 0.5
        )
    )

    return dogleg


def interpolate_well(sections, step=30):
    """
    Constructs a well survey from a list of sections of control points.

    Parameters
    ----------
    sections: list of welleng.connector.Connector objects
    step: float (default: 30)
        The desired delta measured depth between survey points, in
        addition to the control points.

    Returns
    -------
    data: list of welleng.connector.Connector objects
    """
    method = {
        'hold': get_interpolate_hold,
        'min_dist_to_target': get_interpolate_min_dist_to_target,
        'min_curve_to_target': get_interpolate_min_curve_to_target,
        'curve_hold_curve': get_interpololate_curve_hold_curve,
        'min_curve': get_min_curve
    }

    data = []
    if type(sections) is not list:
        sections = [sections]
    for s in sections:
        data.extend(method[s.method](s, step))

    return data


def interpolate_curve(
    md1,
    pos1,
    vec1,
    vec2,
    dist_curve,
    dogleg,
    func_dogleg,
    step,
    endpoint=False
):
    # sometimes the curve section has no length
    # this if statement handles this event
    if dist_curve == 0:
        inc, azi = get_angles(vec1, nev=True).T
        data = dict(
            md=np.array([md1]),
            vec=np.array([vec1]),
            inc=inc,
            azi=azi,
            dogleg=np.array([dogleg])
        )

        return data

    end_md = abs(dist_curve)
    if step is None:
        md = np.array([0])
    else:
        start_md = step - (md1 % step)
        md = np.arange(start_md, end_md, step)
        md = np.concatenate(([0.], md))
    if endpoint:
        md = np.concatenate((md, [end_md]))
    dogleg_interp = (dogleg / dist_curve * md).reshape(-1, 1)

    vec = (
        (
            np.sin(dogleg - dogleg_interp) / np.sin(dogleg) * vec1
        ) + (
            np.sin(dogleg_interp) / np.sin(dogleg) * vec2
        )
    )
    vec = vec / np.linalg.norm(vec, axis=1).reshape(-1, 1)
    inc, azi = get_angles(vec, nev=True).T

    data = dict(
        md=md + md1,
        vec=vec,
        inc=inc,
        azi=azi,
        dogleg=np.concatenate((
            np.array([0.]), np.diff(dogleg_interp.reshape(-1))
        )),
    )

    return data


def interpolate_hold(md1, pos1, vec1, md2, step, endpoint=False):
    end_md = md2 - md1
    if step is None:
        md = np.array([0])
    else:
        start_md = step - (md1 % step)
        md = np.arange(start_md, end_md, step)
        md = np.concatenate(([0.], md))
    if endpoint:
        md = np.concatenate((md, [end_md]))
    vec = np.full((len(md), 3), vec1)
    inc, azi = get_angles(vec, nev=True).T
    dogleg = np.full_like(md, 0.)

    data = dict(
        md=md + md1,
        vec=vec,
        inc=inc,
        azi=azi,
        dogleg=dogleg,
    )

    return data


def get_min_curve(section, step=30, data=None):
    if section.md2 is None:
        result = (
            get_interpolate_min_curve_to_target(
                section, step, data
            )
        )
    else:
        result = (
            get_interpolate_min_dist_to_target(
                section, step, data
            )
        )
    return result


def get_interpolate_hold(section, step=30, data=None):
    if data is None:
        data = []

    data.append(interpolate_hold(
        md1=section.md1,
        pos1=section.pos1,
        vec1=section.vec1,
        md2=section.md_target,
        step=step,
        endpoint=True
    ))

    return data


def get_interpolate_min_curve_to_target(section, step=30, data=None):
    if data is None:
        data = []

    data.append(interpolate_curve(
        md1=section.md1,
        pos1=section.pos1,
        vec1=section.vec1,
        vec2=section.vec_target,
        dist_curve=section.dist_curve,
        dogleg=section.dogleg,
        func_dogleg=section.func_dogleg,
        step=step,
        endpoint=True
    ))

    return data


def get_interpolate_min_dist_to_target(section, step=30, data=None):
    if data is None:
        data = []

    # the first curve section
    data.append(interpolate_curve(
        md1=section.md1,
        pos1=section.pos1,
        vec1=section.vec1,
        vec2=section.vec2,
        dist_curve=section.dist_curve,
        dogleg=section.dogleg,
        func_dogleg=section.func_dogleg,
        step=step
    ))

    # the hold section
    data.append(interpolate_hold(
        md1=section.md2,
        pos1=section.pos2,
        vec1=section.vec2,
        md2=section.md_target,
        step=step,
        endpoint=True
    ))

    return data


def get_interpololate_curve_hold_curve(section, step=30, data=None):
    if data is None:
        data = []

    # the first curve section
    data.append(interpolate_curve(
        md1=section.md1,
        pos1=section.pos1,
        vec1=section.vec1,
        vec2=section.vec2,
        dist_curve=section.dist_curve,
        dogleg=section.dogleg,
        func_dogleg=section.func_dogleg,
        step=step
    ))

    # the hold section
    data.append(interpolate_hold(
        md1=section.md2,
        pos1=section.pos2,
        vec1=section.vec2,
        md2=section.md3,
        step=step
    ))

    # the second curve section
    data.append(interpolate_curve(
        md1=section.md3,
        pos1=section.pos3,
        vec1=section.vec3,
        vec2=section.vec_target,
        dist_curve=section.dist_curve2,
        dogleg=section.dogleg2,
        func_dogleg=section.func_dogleg2,
        step=step,
        endpoint=True
    ))

    return data


def convert_target_input_to_booleans(*inputs):
    input = [
        "0" if i is None else "1" for i in inputs
    ]

    return ''.join(input)


def connect_points(
        cartesians,
        vec_start=None,
        dls_design=3.0,
        nev=True,
        md_start=0.
):
    """
    Function for connecting a list or array of only Cartesian points.

    Parameters
    ----------
        cartesians: (n, 3) list or array of floats
            Either [n, e, tvd] (default) or [x, y, z]
        vec_start: (3) list or array of floats (default: [0., 0., 1.])
            Unit start vector (default is pointing down) in the nev or xyz
            coordinate system.
        dls_design: float or (n, 1) list or array of floats (default: 3.0)
            The minimum Dog Leg Severity used when attempting to connect the
            points (a high DLS will be used if necessary).
        nev: bool (default: True)
            Indicates whether the cartesians are referencing the [nev]
            (default) or [xyz] coordinate system.
        step: float (default: 30)
            The desired step interval for the returned Survey object.
        md_start: float (default: 0)
            The md at the first cartesian point (in the event of a tie-on).

    Returns
    -------
        data: list of welleng.connector.Connector objects
    """
    vec_start = vec_start or [0., 0., 1.]

    if nev:
        pos_nev = np.array(cartesians).reshape(-1, 3)
        vec_nev = np.zeros_like(pos_nev)
        vec_nev[0] = np.array(vec_start).reshape(-1, 3)
    else:
        e, n, tvd = np.array(cartesians).reshape(-1, 3).T
        pos_nev = np.array([n, e, tvd]).T
        vec_nev = np.zeros_like(pos_nev)
        e, n, tvd = np.array(vec_start).reshape(-1, 3).T
        vec_nev[0] = np.array([n, e, tvd]).T

    if type(dls_design) is float:
        dls = np.full(len(pos_nev), dls_design)
    else:
        dls = np.array(dls_design).reshape(-1, 1)

    connections = []
    for i, (p, v, d) in enumerate(zip(pos_nev, vec_nev, dls)):
        if i == 0:
            node_1 = Node(
                pos=p,
                vec=v,
                md=md_start
            )
            continue
        if i > 1:
            node_1 = connections[-1].node_end
        node_2 = Node(
            pos=p
        )
        c = Connector(
            node1=node_1,
            node2=node_2,
            dls_design=d
        )
        assert np.allclose(c.pos_target, p)
        connections.append(c)

    return connections


def survey_to_plan(survey, tolerance=0.2, dls_design=1., step=30.):
    """
    Prototype function for extracting a plan from a drilled well survey - a
    minimal number of control points (begining/end points of either hold or
    build/turn sections) required to express the well given the provided input
    parameters.

    Parameters
    ----------
    survey: welleng.survey.Survey object
    tolerance: float (default=0.2)
        Defines how tight to fit the planned well to the survey - a higher
        number will results in less control points but likely poorer fit.
    dls_design: float (default=1.0)
        The minimum DLS used to fit the planned trajectory.
    step: float (default=30)
        The desired md step in the plan survey.

    Returns
    -------
    sections: list of welleng.connector.Connection objects
    """
    assert dls_design > 0., "dls_design must be greater than 0"

    idx = [0]
    end = len(survey.md) - 1
    md = survey.md[0]
    node = None
    sections = []

    while True:
        section, i = _get_section(
            survey=survey,
            start=idx[-1],
            md=md,
            node=node,
            tolerance=tolerance,
            dls_design=dls_design
        )
        sections.append(section)
        idx.append(i)
        if idx[-1] >= end:
            break
        node = section.node_end

    return sections


def _get_section(
    survey, start, tolerance, dls_design=1., md=0., node=None
):
    idx = start + 2
    nev = survey.get_nev_arr()

    if idx > len(nev):
        idx = len(nev) - 1

    if node is None:
        node_1 = Node(
            pos=nev[start],
            vec=survey.vec_nev[start],
            md=md
        )
    else:
        node_1 = node

    scores = [0.]
    delta_scores = []
    c_old = None

    for i, (p, v) in enumerate(zip(nev[idx:], survey.vec_nev[idx:])):
        node_2 = Node(
            pos=p,
            vec=v,
        )
        c = Connector(
            node1=node_1,
            node2=node_2,
            dls_design=dls_design
        )
        s = c.survey(step=1.)
        if c_old is None:
            c_old = deepcopy(c)
        nev_new = s.get_nev_arr()

        distances = distance.cdist(
            nev[start: idx + i],
            nev_new
        )

        score = np.sum(np.amin(distances, axis=1)) / (s.md[-1] - s.md[0])

        delta_scores.append(score - scores[-1])
        scores.append(score)

        if all((
            abs(delta_scores[-1]) >= tolerance,
            idx + i < len(nev) - 2
        )):
            break
        elif idx + i == len(nev) - 1:
            c_old = deepcopy(c)
            break
        else:
            c_old = deepcopy(c)

    return (
        c_old,
        idx + i - 1 if idx + i != len(nev) - 1 else idx + i
    )


def numbafy(functions):
    for f in functions:
        f = njit(f)


if NUMBA:
    NUMBAFY = (
        _get_xyz,
        get_pos,
        get_vec_target,
        get_curve_hold_data,
        shape_factor,
        min_dist_to_target,
        get_radius_critical,
        angle,
        get_dogleg
    )
    numbafy(NUMBAFY)
