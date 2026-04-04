"""Wellbore trajectory connector for computing minimum-MD CLC paths."""

from copy import copy, deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance

from .node import Node, get_node_params
from .utils import (
    NEV_to_HLA, dls_from_radius, get_angles,
    get_dogleg, get_nev, get_rf, get_unit_vec, get_vec, get_xyz,
    radius_from_dls, get_arc
)


class Connector:
    """Solves minimum-MD wellbore trajectories between two survey stations.

    Automatically selects the appropriate geometric method (hold, curve-hold,
    min-curve, or curve-hold-curve) based on the provided start/end constraints
    and computes control points for the connecting path segment. The solver
    honours a maximum dog-leg severity (DLS) constraint where geometrically
    feasible.

    Attributes
    ----------
    method : str
        The geometric method used ('hold', 'min_curve',
        'curve_hold_curve', 'min_dist_to_target', or
        'min_curve_to_target').
    node_start : Node
        Start survey station as a Node.
    node_end : Node
        End survey station as a Node.
    pos1 : ndarray of shape (3,)
        Start position in NEV coordinates.
    vec1 : ndarray of shape (3,)
        Unit direction vector at the start position.
    inc1 : float
        Inclination at the start position (radians).
    azi1 : float
        Azimuth at the start position (radians).
    md1 : float
        Measured depth at the start position.
    pos2 : ndarray of shape (3,) or None
        Position at the end of the first arc section in NEV
        coordinates. Equal to vec2 direction at this point.
    vec2 : ndarray of shape (3,) or None
        Unit direction vector at the end of the first arc.
        Equals vec3 for curve-hold-curve solutions.
    inc2 : float or None
        Inclination at the end of the first arc (radians).
    azi2 : float or None
        Azimuth at the end of the first arc (radians).
    md2 : float or None
        Measured depth at the end of the first arc.
    pos3 : ndarray of shape (3,) or None
        Position at the start of the second arc (end of the
        hold section) in NEV coordinates. Only set for
        curve-hold-curve solutions.
    vec3 : ndarray of shape (3,) or None
        Unit direction vector at the start of the second arc.
        Only set for curve-hold-curve solutions.
    inc3 : float or None
        Inclination at the start of the second arc (radians).
    azi3 : float or None
        Azimuth at the start of the second arc (radians).
    md3 : float or None
        Measured depth at the start of the second arc.
    md_target : float
        Measured depth at the target position.
    pos_target : ndarray of shape (3,)
        Target position in NEV coordinates.
    vec_target : ndarray of shape (3,)
        Target unit direction vector in NEV coordinates.
    inc_target : float
        Target inclination (radians).
    azi_target : float
        Target azimuth (radians).
    dogleg : float
        Dogleg angle of the first arc (radians).
    dogleg2 : float or None
        Dogleg angle of the second arc (radians). Only set for
        curve-hold-curve solutions.
    dist_curve : float
        Arc length of the first curve section.
    dist_curve2 : float
        Arc length of the second curve section.
    tangent_length : float or None
        Length of the hold (tangent) section between the two
        arcs.
    dls : float
        Dogleg severity of the first arc (radians per unit
        length).
    dls2 : float
        Dogleg severity of the second arc (radians per unit
        length).
    dls_design : float
        Design DLS constraint for the first arc (radians per
        unit length).
    dls_design2 : float
        Design DLS constraint for the second arc (radians per
        unit length).
    radius_design : float
        Design turn radius derived from dls_design.
    radius_design2 : float
        Design turn radius derived from dls_design2.
    radius_critical : float
        Critical (minimum geometric) radius for the first arc.
    radius_critical2 : float
        Critical radius for the second arc.

    Methods
    -------
    interpolate(step=30)
        Interpolate the solved trajectory at regular MD intervals.
    """

    def __init__(
        self,
        node1=None,
        node2=None,
        pos1=[0., 0., 0.],
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
        """Initializes the Connector and solves the trajectory.

        Only specific combinations of input data are permitted. For example,
        providing both a start vector and start inc/azi raises an error.
        The solver determines the appropriate method from the provided
        parameters and computes the connecting path immediately.

        Parameters
        ----------
        node1 : Node or None
            Start Node. Overrides pos1, vec1, md1 if provided.
        node2 : Node or None
            End Node. Overrides pos2, vec2, md2 if provided.
        pos1 : list or ndarray
            Start position as [n, e, v] in NEV coordinates.
        vec1 : list or ndarray or None
            Start unit direction vector in NEV coordinates.
        inc1 : float or None
            Start inclination angle.
        azi1 : float or None
            Start azimuth angle.
        md1 : float
            Start measured depth.
        dls_design : float
            Design DLS for the first curve section in
            deg/30m (meters) or deg/100ft (feet).
        dls_design2 : float or None
            Design DLS for the second curve section. Defaults
            to dls_design if None.
        md2 : float or None
            Target measured depth. Mutually exclusive with pos2.
        pos2 : list or ndarray or None
            Target position in NEV coordinates.
        vec2 : list or ndarray or None
            Target unit direction vector in NEV coordinates.
            Mutually exclusive with inc2/azi2.
        inc2 : float or None
            Target inclination angle.
        azi2 : float or None
            Target azimuth angle.
        degrees : bool
            If True, angles are in degrees; if False, radians.
        unit : str
            Distance unit, either 'meters' or 'feet'.
        min_error : float
            Error tolerance for iterative convergence. Must be
            less than 1.
        delta_dls : float
            DLS tolerance (deg/30m) for balancing curve sections
            in curve-hold-curve solutions.
        min_tangent : float
            Minimum tangent length to stabilize
            curve-hold-curve iteration.
        max_iterations : int
            Maximum iteration count for curve-hold-curve fitting.
        force_min_curve : bool
            If True, forces minimum-curvature method.
        closest_approach : bool
            If True, finds the closest-approach trajectory
            when the target is inside the critical radius.

        Raises
        ------
        AssertionError
            If input parameter combinations are invalid.
        """
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

        if dls_design is not None:
            assert dls_design > 0, "dls_design must be greater than zero"
        assert min_error < 1, "min_error must be less than 1.0"

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
        if (inc1 is not None and azi1 is not None):
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

        self.md1 = md1
        self.pos_target = None if pos2 is None else np.array(pos2).reshape(3)
        self.md_target = md2

        if vec2 is not None:
            self.vec_target = np.array(vec2).reshape(3)
            self.inc_target, self.azi_target = get_angles(
                self.vec_target,
                nev=True
            ).reshape(2)
        elif (inc2 is not None and azi2 is not None):
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

        self.unit = unit
        if self.unit == 'meters':
            self.denom = 30
        else:
            self.denom = 100

        # Primary DLS / radius.  dls_design=None means "use minimum curvature
        # required by geometry" — radius_design is set to inf so that
        # min(radius_design, radius_critical) always resolves to radius_critical.
        if dls_design is None:
            self.dls_design = 0.0
            self.radius_design = np.inf
        else:
            self.dls_design = np.radians(dls_design) if degrees else dls_design
            self.radius_design = self.denom / self.dls_design

        # Secondary DLS (second arc of curve_hold_curve)
        if dls_design2:
            self.dls_design2 = np.radians(dls_design2) if degrees else dls_design2
            self.radius_design2 = self.denom / self.dls_design2
        else:
            self.dls_design2 = self.dls_design
            self.radius_design2 = self.radius_design

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
        self.dls = np.degrees(self.dogleg) / abs(self.dist_curve) * 30

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
        self.dls = np.degrees(self.dogleg) / self.dist_curve * 30

    def _use_method(self):
        if self.method == 'hold':
            self._hold()
        elif self.method == 'min_curve':
            self._min_curve()
        elif self.method == 'curve_hold_curve':
            self.pos2_list, self.pos3_list = [], [deepcopy(self.pos_target)]
            self.vec23 = [np.array([0., 0., 0.])]
            self.delta_radius_list = []
            # self._target_pos_and_vec_defined(deepcopy(self.pos_target))
            self._target_pos_and_vec_defined(
                self.pos1 + (self.pos_target - self.pos1) / 2
            )
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
            if not np.isfinite(self.radius_design):
                raise ValueError(
                    "dls_design must be specified (not None) when only a "
                    "target direction is given without a target position or "
                    "measured depth."
                )
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
            with np.errstate(divide='ignore'):
                self.radius_critical = np.nan_to_num(abs(
                    (self.md_target - self.md1) / self.dogleg
                ), nan=np.inf)
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

    def _target_pos_and_vec_defined(self, pos3, vec_old=[0., 0., 0.]):
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
        minimize_target_pos_and_vec_defined(
            [self.radius_design, self.radius_design2],
            self, None, None, True
        )
        # Re-evaluate whether the outer loop is needed using the FINAL
        # converged geometry (not the running minimum radius_critical, which
        # may have been locked too low by transient tight iteration states).
        if hasattr(self, 'distances1') and self.distances1 is not None:
            _rc1 = get_radius_critical(
                self.radius_design, self.distances1, self.min_error
            )
            _rc2 = get_radius_critical(
                self.radius_design2, self.distances2, self.min_error
            )
            _outer_needed = not all((_rc1 >= self.radius_design, _rc2 >= self.radius_design2))
        else:
            _outer_needed = not all((
                self.radius_critical > self.radius_design,
                self.radius_critical2 > self.radius_design2
            ))
        if _outer_needed:
            # Try the r_safe seeding approach first: solve at r_safe=dist/4
            # (guaranteed to converge for any geometry), then use the result
            # as seed for a design-radius re-solve.  This can escape the tight
            # fixed point that the initial call converged to.
            dist_to_target = np.linalg.norm(self.pos_target - self.pos1)
            r_safe = dist_to_target / 4.0
            if 0 < r_safe < min(self.radius_design, self.radius_design2):
                minimize_target_pos_and_vec_defined(
                    [r_safe, r_safe], self, None, None, True
                )
                pos3_seed = deepcopy(self.pos3)
                vec_seed = (
                    deepcopy(self.vec23[-1]) if len(self.vec23) > 0 else None
                )
                # Reset running minima (r_safe call left them at small value).
                self.radius_critical = np.inf
                self.radius_critical2 = np.inf
                minimize_target_pos_and_vec_defined(
                    [self.radius_design, self.radius_design2],
                    self, pos3_seed, vec_seed, True
                )
                # Re-check with final geometry
                if hasattr(self, 'distances1') and self.distances1 is not None:
                    _rc1 = get_radius_critical(
                        self.radius_design, self.distances1, self.min_error
                    )
                    _rc2 = get_radius_critical(
                        self.radius_design2, self.distances2, self.min_error
                    )
                    _outer_needed = not all(
                        (_rc1 >= self.radius_design, _rc2 >= self.radius_design2)
                    )
                else:
                    _outer_needed = not all((
                        self.radius_critical > self.radius_design,
                        self.radius_critical2 > self.radius_design2
                    ))

        if _outer_needed:
            # Design DLS still cannot be achieved; iterate to balance curvature
            # across both arcs.  Use a *separate* counter so that the inner
            # convergence iterations (tracked via self.iterations) do not
            # prematurely exhaust the outer balancing budget.
            for _outer in range(self.max_iterations):
                self.radius_critical = (
                    (self.md_target - self.md1)
                    / (abs(self.dogleg) + abs(self.dogleg2))
                )
                if not (self.radius_critical >= 0):
                    break
                self.radius_critical2 = self.radius_critical
                minimize_target_pos_and_vec_defined(
                    [self.radius_design, self.radius_design2],
                    self, deepcopy(self.pos3), deepcopy(self.vec23[-1]), True
                )

                # Position-stability check: compare against last 5 positions
                # explicitly (avoids ambiguous np.allclose broadcasting with
                # a list of arrays).
                n_hist = min(5, len(self.pos3_list))
                pos3_stable = n_hist >= 5 and all(
                    np.allclose(self.pos3, p) for p in self.pos3_list[-5:]
                )
                pos2_stable = n_hist >= 5 and all(
                    np.allclose(self.pos2, p) for p in self.pos2_list[-5:]
                )

                if any((
                    abs(self.dls - self.dls2) < self.delta_dls,
                    np.allclose(self.pos1, self.pos2),
                    np.allclose(self.pos3, self.pos_target),
                    pos3_stable and pos2_stable,
                    all((
                        self.dls == self.dls_design,
                        self.dls2 == self.dls_design2
                    ))
                )):
                    break
            self._happy_finish()
            self._delta_pos3_rescue()
            self._project_tangent()
            self._endpoint_rescue()
            self._enforce_planarity()
            return

        self._happy_finish()
        self._delta_pos3_rescue()
        self._project_tangent()
        self._endpoint_rescue()
        self._enforce_planarity()

    def _happy_finish(self):
        # Use the critical radius at the FINAL converged geometry rather than the
        # running-minimum radius_critical, which may have been locked too low by
        # transient tight iteration states.  This ensures dist_curve is computed
        # with the same effective radius that was used to derive self.dogleg in
        # the last iteration step, keeping the arc geometry self-consistent.
        if hasattr(self, 'distances1') and self.distances1 is not None:
            rc1_final = get_radius_critical(
                self.radius_design, self.distances1, self.min_error
            )
            r1 = min(rc1_final, self.radius_design)
        else:
            r1 = min(self.radius_critical, self.radius_design)
        # get pos1 to pos2 curve data
        self.dist_curve, self.func_dogleg = get_curve_hold_data(
            r1,
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

        # Resync dogleg and arc-1 length from the just-computed vec3.
        # _happy_finish may be called after _enforce_planarity changes vec3 or
        # after any iteration that updates pos3.  The stored self.dogleg may
        # therefore differ from the actual angle between vec1 and vec3, making
        # dist_curve (= r1 * old_dogleg) inconsistent with the SLERP geometry.
        # from_connections reconstructs pos2 via minimum-curvature using the
        # actual inc/azi of vec3, so it needs dist_curve to match the true arc
        # length for vec1→vec3 at radius r1.
        actual_dogleg = float(np.arccos(
            np.clip(np.dot(self.vec1, self.vec3), -1.0, 1.0)
        ))
        if abs(actual_dogleg - self.dogleg) > 1e-9:
            self.dogleg = actual_dogleg
            self.dist_curve, self.func_dogleg = get_curve_hold_data(
                r1, self.dogleg
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

        if hasattr(self, 'distances2') and self.distances2 is not None:
            rc2_final = get_radius_critical(
                self.radius_design2, self.distances2, self.min_error
            )
            r2 = min(rc2_final, self.radius_design2)
        else:
            r2 = min(self.radius_critical2, self.radius_design2)

        # Recompute dogleg2 from the actual directional change between vec3 and
        # vec_target.  The iteration stores dogleg2 = min_dist_to_target result
        # (a geometric parameter), which can differ from arccos(dot(vec3,
        # vec_target)) after _happy_finish re-derives vec3 from arc1.  Using the
        # wrong angle in the SLERP formula inside interpolate_curve produces
        # incorrect (spiralling) visualisation paths.
        self.dogleg2 = float(np.arccos(
            np.clip(np.dot(self.vec3, self.vec_target), -1.0, 1.0)
        ))
        self.dist_curve2, self.func_dogleg2 = get_curve_hold_data(
            r2,
            self.dogleg2
        )

        # ── Endpoint-consistency fix ──────────────────────────────────────────
        # The rendered path (from_connections / minimum-curvature integration)
        # starts arc2 from hold_end = pos2 + T·vec3, NOT from the backward-
        # traced pos3.  If pos3 is off the tangent ray the rendered endpoint
        # misses pos_target by exactly the perpendicular component of
        # (pos3_backward − pos2) w.r.t. vec3.
        #
        # For well-converged cases (perp < 1 mm) this is negligible and we
        # preserve the backward-trace geometry to avoid touching dist_curve2.
        #
        # For DLS-violation cases the iteration may leave a large perpendicular
        # residual.  The 2×2 projection finds T and dist_curve2 that solve:
        #   pos2 + T·vec3 + (dist_curve2·RF/2)·(vec3+vec_target) = pos_target
        # in the plane of {vec3, vec_target}, eliminating the in-plane part
        # of the endpoint error.  The remaining (out-of-plane) residual can
        # only be removed by finding a better vec3 via a tighter-radius retry.

        # Step 1: backward trace to locate pos3 (guarantees arc2 → pos_target)
        pos3_bt   = get_pos(
            self.pos_target,
            self.vec_target * -1,
            self.vec3 * -1,
            self.dist_curve2,
            self.func_dogleg2
        ).reshape(3)
        t_along   = float(np.dot(pos3_bt - self.pos2, self.vec3))
        perp_vec  = pos3_bt - self.pos2 - t_along * self.vec3
        perp_norm = float(np.linalg.norm(perp_vec))

        # Step 2: 2×2 projection only when perpendicular drift is significant
        _PERP_THRESHOLD = 1e-3   # 1 mm — noise floor for well-converged cases
        _applied_2x2 = False
        if perp_norm > _PERP_THRESHOLD:
            delta  = self.pos_target - self.pos2
            p_comp = float(np.dot(delta, self.vec3))
            q_comp = float(np.dot(delta, self.vec_target))
            c_dot  = float(np.dot(self.vec3, self.vec_target))
            sin2   = 1.0 - c_dot * c_dot
            if sin2 > 1e-6 and abs(self.func_dogleg2) > 1e-10:
                a_coef = (p_comp - c_dot * q_comp) / sin2
                b_coef = (q_comp - c_dot * p_comp) / sin2
                if b_coef >= 0.0:
                    tangent_length   = float(max(0.0, a_coef - b_coef))
                    self.dist_curve2 = float(
                        max(0.0, 2.0 * b_coef / self.func_dogleg2)
                    )
                    self.pos3        = self.pos2 + tangent_length * self.vec3
                    _applied_2x2 = True

        if not _applied_2x2:
            # Well-converged (or 2×2 degenerate): tangent-project only
            tangent_length = max(0.0, t_along)
            self.pos3      = self.pos2 + tangent_length * self.vec3

        self.md3       = self.md2 + tangent_length
        self.md_target = self.md3 + abs(self.dist_curve2)

        return self

    def _delta_pos3_rescue(self):
        """
        Final geometric-consistency check after _happy_finish.

        If the tangent section is backward (dot(pos3-pos2, vec3) < 0) the
        path doubles back on itself.  Re-seed with pos3 = pos2 (zero tangent)
        to force the iteration to find a forward path, then only keep the
        rescue result if it has a strictly better (higher) minimum arc radius.
        """
        _tangent_proj = float(np.dot(self.pos3 - self.pos2, self.vec3))
        if _tangent_proj >= -self.min_error:
            return  # tangent is already forward — nothing to fix

        # Snapshot the current _happy_finish outputs before attempting rescue
        _state_keys = (
            'pos2', 'pos3', 'vec2', 'vec3',
            'dist_curve', 'dist_curve2', 'func_dogleg', 'func_dogleg2',
            'dogleg', 'dogleg2', 'md2', 'md3', 'md_target',
        )
        _saved = {k: deepcopy(getattr(self, k)) for k in _state_keys}

        _r_before = min(
            self.dist_curve / self.dogleg if self.dogleg > 1e-10 else self.radius_design,
            self.dist_curve2 / self.dogleg2 if self.dogleg2 > 1e-10 else self.radius_design,
        )

        for _seed in (
            deepcopy(self.pos2),                  # zero-tangent seed
            0.5 * (self.pos2 + self.pos_target),  # midpoint fallback
        ):
            self.radius_critical = np.inf
            self.radius_critical2 = np.inf
            minimize_target_pos_and_vec_defined(
                [self.radius_design, self.radius_design2],
                self, _seed, None, True
            )
            self._happy_finish()
            _tp = float(np.dot(self.pos3 - self.pos2, self.vec3))
            _r_after = min(
                self.dist_curve / self.dogleg if self.dogleg > 1e-10 else self.radius_design,
                self.dist_curve2 / self.dogleg2 if self.dogleg2 > 1e-10 else self.radius_design,
            )
            if _tp >= -self.min_error and _r_after >= _r_before - self.min_error:
                return  # rescue improved or matched — keep it

        # Rescue did not improve things — restore the saved state
        for k, v in _saved.items():
            setattr(self, k, v)

    def _project_tangent(self):
        """Project pos3 onto the tangent ray from pos2 along vec3.

        _happy_finish computes pos2 (arc1 end) and pos3 (arc2 start) via
        independent minimum-curvature traces.  When the iteration has not
        fully converged — or when dogleg2 is near π so the minimum-curvature
        ratio-factor blows up — pos3 may lie far off the tangent ray
        pos2 + t*vec3.  The survey reconstruction (interpolate_hold) advances
        strictly along vec3, so the hold section ends at pos2 + t*vec3 while
        arc2 still starts at the raw pos3.  That geometric gap produces the
        visually anomalous ~90° phase difference seen in the edge-case viewer.

        Fix: replace pos3 with its projection onto the tangent ray.  The
        projected tangent length is dot(pos3-pos2, vec3), clamped to ≥ 0.
        This eliminates the hold→arc2 discontinuity.  Arc2 dogleg and arc
        length are unchanged (they depend only on vec3, vec_target, and the
        radius), so the path is geometrically self-consistent.

        Called after _delta_pos3_rescue so that rescue still operates on the
        un-projected geometry (its backward-tangent trigger requires the raw
        dot-product, which is <= 0 before projection).
        """
        # _happy_finish now projects tangent_length inline and places pos3 via
        # the backward arc2 trace (keeping arc2 geometrically consistent with
        # pos_target).  This method is retained as a safety net to update MD
        # whenever pos3 has been changed by _delta_pos3_rescue.
        tangent_length = max(
            0.0, float(np.dot(self.pos3 - self.pos2, self.vec3))
        )
        self.md3 = self.md2 + tangent_length
        self.md_target = self.md3 + abs(self.dist_curve2)

    def _rendered_endpoint_error(self):
        """Return the distance from the rendered arc2 endpoint to pos_target.

        The rendered path integrates continuously from pos1, so arc2 starts
        from hold_end = pos2 + T·vec3 (not from pos3 directly).  Any gap
        between hold_end and pos3 translates directly into an endpoint error.
        """
        t = max(0.0, float(np.dot(self.pos3 - self.pos2, self.vec3)))
        hold_end = self.pos2 + t * self.vec3
        rendered = (
            hold_end
            + (self.dist_curve2 * self.func_dogleg2 / 2)
            * (self.vec3 + self.vec_target)
        )
        return float(np.linalg.norm(self.pos_target - rendered))

    def _endpoint_rescue(self):
        """Retry with progressively tighter radii when the rendered endpoint
        still misses pos_target by more than 1 mm after _happy_finish.

        The fixed-point iterator can converge to a spurious local minimum for
        DLS-violation geometries, leaving a large out-of-plane residual that
        the 2×2 in-plane projection cannot eliminate.  Using a tighter design
        radius enlarges the set of reachable vec3 directions and typically
        allows the iterator to escape the bad basin.

        Each retry uses a fresh geometric seed (pos3=None) and keeps whichever
        result achieves the smallest rendered endpoint error.
        """
        _THRESHOLD = 1e-3   # 1 mm
        err0 = self._rendered_endpoint_error()
        if err0 <= _THRESHOLD:
            return

        _save_keys = (
            'pos2', 'pos3', 'vec2', 'vec3',
            'dist_curve', 'dist_curve2', 'func_dogleg', 'func_dogleg2',
            'dogleg', 'dogleg2', 'md2', 'md3', 'md_target',
        )
        best_err   = err0
        best_state = {k: deepcopy(getattr(self, k)) for k in _save_keys}

        r_base = min(self.radius_design, self.radius_design2)
        for shrink in [0.5, 0.25, 0.125]:
            r_try = r_base * shrink
            self.radius_critical  = np.inf
            self.radius_critical2 = np.inf
            minimize_target_pos_and_vec_defined(
                [r_try, r_try], self, None, None, True
            )
            self._happy_finish()
            err_new = self._rendered_endpoint_error()
            if err_new < best_err:
                best_err   = err_new
                best_state = {k: deepcopy(getattr(self, k)) for k in _save_keys}
            if best_err <= _THRESHOLD:
                break

        # Restore the geometry that achieved the smallest endpoint error
        for k, v in best_state.items():
            setattr(self, k, v)

    def _enforce_planarity(self, max_iters=8, tol=1e-3):
        """Adjust vec3 to lie in the {pos_target−pos2, vec_target} plane.

        After _endpoint_rescue, an out-of-plane residual may remain because the
        fixed-point iterator converged to a vec3 that is not coplanar with
        (pos_target − pos2) and vec_target.  The 2×2 projection in _happy_finish
        can only correct the in-plane part; the out-of-plane part can only be
        removed by rotating vec3 itself.

        Strategy: project the current vec3 onto span{delta, vec_target} (where
        delta = pos_target − pos2) and re-seed _happy_finish so it recovers the
        corrected direction.  Iterate until the rendered endpoint error drops
        below tol (1 mm) or max_iters is exhausted.
        """
        if self._rendered_endpoint_error() <= tol:
            return

        # Determine the effective arc-1 radius from the post-rescue state.
        if hasattr(self, 'distances1') and self.distances1 is not None:
            r1 = min(
                get_radius_critical(self.radius_design, self.distances1, self.min_error),
                self.radius_design
            )
        elif np.isfinite(self.radius_critical) and self.radius_critical > 0:
            r1 = min(self.radius_critical, self.radius_design)
        else:
            r1 = self.radius_design

        _save_keys = (
            'pos2', 'pos3', 'vec2', 'vec3',
            'dist_curve', 'dist_curve2', 'func_dogleg', 'func_dogleg2',
            'dogleg', 'dogleg2', 'md2', 'md3', 'md_target',
            'radius_critical', 'radius_critical2',
        )
        best_err = self._rendered_endpoint_error()
        best_state = {k: deepcopy(getattr(self, k)) for k in _save_keys}

        for _ in range(max_iters):
            # Compute arc-1 endpoint (pos2) from the current vec3 at r1
            dc1, fd1 = get_curve_hold_data(r1, self.dogleg)
            pos2 = get_pos(
                self.pos1, self.vec1, self.vec3, dc1, fd1
            ).reshape(3)
            delta = self.pos_target - pos2

            # Build an ONB for span{delta, vec_target}
            d_norm = float(np.linalg.norm(delta))
            if d_norm < 1e-10:
                break  # pos_target ≈ pos2 — degenerate
            d_hat = delta / d_norm

            vt_comp = float(np.dot(self.vec_target, d_hat))
            v_perp = self.vec_target - vt_comp * d_hat
            v_perp_norm = float(np.linalg.norm(v_perp))
            if v_perp_norm < 1e-8:
                break  # vec_target ∥ delta — unconstrained direction

            v_hat = v_perp / v_perp_norm

            # Project vec3 onto the {d_hat, v_hat} plane
            proj = (
                float(np.dot(self.vec3, d_hat)) * d_hat
                + float(np.dot(self.vec3, v_hat)) * v_hat
            )
            proj_norm = float(np.linalg.norm(proj))
            if proj_norm < 1e-10:
                break  # vec3 ⊥ plane — cannot project

            new_vec3 = proj / proj_norm
            if np.allclose(new_vec3, self.vec3, atol=1e-9):
                break  # already in-plane — nothing to do

            new_dogleg = float(np.arccos(np.clip(
                np.dot(self.vec1, new_vec3), -1.0, 1.0
            )))

            # Compute pos2 seed so that _happy_finish recovers new_vec3:
            #   get_vec_target(pos1, vec1, pos2_new, 0, dc1_new, fd1_new) = new_vec3
            dc1_new, fd1_new = get_curve_hold_data(r1, new_dogleg)
            pos2_new = get_pos(
                self.pos1, self.vec1, new_vec3, dc1_new, fd1_new
            ).reshape(3)

            # Seed _happy_finish with the planarity-corrected direction
            self.dogleg = new_dogleg
            self.vec3 = new_vec3
            self.pos3 = pos2_new        # tangent_length = 0 seed
            self.tangent_length = 0.0
            self.distances1 = None      # force _happy_finish to use radius_critical
            self.radius_critical = r1
            self.radius_critical2 = np.inf
            self._happy_finish()

            err_new = self._rendered_endpoint_error()
            if err_new < best_err:
                best_err = err_new
                best_state = {k: deepcopy(getattr(self, k)) for k in _save_keys}
            if best_err <= tol:
                break

        # Always restore the state that achieved the smallest endpoint error
        for k, v in best_state.items():
            setattr(self, k, v)

    def interpolate(self, step=30):
        """Interpolates the connector trajectory at regular MD intervals.

        Parameters
        ----------
        step : float
            Desired delta measured depth between survey points.

        Returns
        -------
        list
            A list of interpolated survey data dictionaries.
        """
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
    """Iteratively solves curve-hold-curve geometry between two nodes.

    Uses a damped fixed-point iteration (alpha=0.5) on the intermediate
    tangent point to find the curve-hold-curve path connecting the start
    and target positions/vectors on the Connector.

    Parameters
    ----------
    x : list
        List of [radius1, radius2] for the two curve sections.
    c : Connector
        The Connector instance whose state is updated in place.
    pos3 : ndarray or None
        Optional initial guess for the intermediate tangent point.
    vec_old : ndarray or None
        Optional previous tangent direction for convergence check.
    result : bool
        If True, returns the Connector; if False, returns a scalar
        residual for use with optimizers.

    Returns
    -------
    Connector or float
        The Connector instance if result is True, otherwise a float
        residual measuring how much the design radii were violated.
    """
    if vec_old is None:
        vec_old = np.array([0., 0., 0.])

    radius1, radius2 = x

    # Initialise the intermediate tangent point
    if pos3 is None:
        pos2_init = c._get_pos2(c.pos1, c.vec1, c.pos_target, radius1)
        pos3_init = c._get_pos2(
            c.pos_target, c.vec_target * -1, c.pos1, radius2
        )
        pos3_mid = pos2_init + (pos3_init - pos2_init) / 2

        # Check whether the midpoint is inside the critical circle of either
        # endpoint at the design DLS.  If so, the running minimum would lock
        # in a tight radius from step 1.  Fall back to a point directly ahead
        # of pos1 along vec1 (dist_norm = 0 → radius_critical = ∞), which is
        # guaranteed to be in the design-DLS basin.
        d1 = c._get_distances(c.pos1, c.vec1, pos3_mid)
        d2 = c._get_distances(c.pos_target, c.vec_target * -1, pos3_mid)
        rc1 = get_radius_critical(radius1, d1, c.min_error)
        rc2 = get_radius_critical(radius2, d2, c.min_error)
        if rc1 >= radius1 and rc2 >= radius2:
            c.pos3 = pos3_mid
        else:
            # Midpoint is inside a critical circle.  Find a fallback pos3
            # that is guaranteed to be in the design-DLS basin:
            #   pos1 + R*vec1 + 2R*perp
            # gives dist_norm = 2R → rc = 5R²/(4R) = 1.25R ≥ R.
            # perp points in the component of (pos_target - pos1)
            # perpendicular to vec1 so the guess is biased toward the target.
            chord = c.pos_target - c.pos1
            d_perp = chord - np.dot(chord, c.vec1) * c.vec1
            d_perp_norm = np.linalg.norm(d_perp)
            if d_perp_norm > 1e-10:
                d_perp_hat = d_perp / d_perp_norm
            else:
                # Target is directly ahead/behind: pick any perpendicular
                arb = np.array([1., 0., 0.])
                if abs(np.dot(c.vec1, arb)) > 0.9:
                    arb = np.array([0., 1., 0.])
                d_perp_hat = np.cross(c.vec1, arb)
                d_perp_hat /= np.linalg.norm(d_perp_hat)
            # Try four candidate positions: ±d_perp and ±d_perp_cross
            # (the 90°-rotated perpendicular).  Pick the first that is
            # also outside the critical circle of pos_target.
            d_perp_cross = np.cross(c.vec1, d_perp_hat)
            d_perp_cross_norm = np.linalg.norm(d_perp_cross)
            if d_perp_cross_norm > 1e-10:
                d_perp_cross /= d_perp_cross_norm
            else:
                d_perp_cross = d_perp_hat  # degenerate: fallback
            chosen = None
            best_min_rc = -np.inf
            best_candidate = c.pos1 + radius1 * c.vec1 + 2.0 * radius1 * d_perp_hat
            for perp in (d_perp_hat, -d_perp_hat, d_perp_cross, -d_perp_cross):
                candidate = c.pos1 + radius1 * c.vec1 + 2.0 * radius1 * perp
                dd2 = c._get_distances(c.pos_target, c.vec_target * -1, candidate)
                rc2_cand = get_radius_critical(radius2, dd2, c.min_error)
                if rc2_cand >= radius2:
                    chosen = candidate
                    break
                # Track the candidate with the highest minimum rc
                if rc2_cand > best_min_rc:
                    best_min_rc = rc2_cand
                    best_candidate = candidate
            c.pos3 = chosen if chosen is not None else best_candidate
    else:
        c.pos3 = pos3

    radius_temp1, radius_temp2 = radius1, radius2

    for _ in range(c.max_iterations):
        prev_pos3 = c.pos3.copy()

        # ── Curve 1: pos1 → pos2 ────────────────────────────────────────────
        c.distances1 = c._get_distances(c.pos1, c.vec1, c.pos3)

        radius_temp1 = get_radius_critical(radius1, c.distances1, c.min_error)
        # Use the current geometry's critical radius directly — avoids NaN in
        # min_dist_to_target and prevents the running minimum from permanently
        # locking in tight values from transient intermediate states.
        radius_effective1 = min(radius1, radius_temp1)
        # Update the running minimum only for genuine violations (not noise).
        if radius_temp1 < min(c.radius_critical, radius1 * (1.0 - c.min_error * 100)):
            c.radius_critical = radius_temp1
            assert c.radius_critical >= 0

        c.tangent_length, c.dogleg = min_dist_to_target(
            radius_effective1, c.distances1
        )
        c.dogleg = check_dogleg(c.dogleg)
        c.dist_curve, c.func_dogleg = get_curve_hold_data(
            radius_effective1, c.dogleg
        )
        c.vec3 = get_vec_target(
            c.pos1, c.vec1, c.pos3,
            c.tangent_length, c.dist_curve, c.func_dogleg
        )

        tangent_temp1 = c._get_tangent_temp(c.tangent_length)
        c.pos2 = c.pos3 - tangent_temp1 * c.vec3

        # ── Curve 2: pos_target → pos2 (reversed) ───────────────────────────
        c.distances2 = c._get_distances(
            c.pos_target, c.vec_target * -1, c.pos2
        )

        radius_temp2 = get_radius_critical(radius2, c.distances2, c.min_error)
        # Use current geometry's critical radius directly (same rationale as arc1).
        radius_effective2 = min(radius2, radius_temp2)
        # Update running minimum only for genuine violations.
        if radius_temp2 < min(c.radius_critical2, radius2 * (1.0 - c.min_error * 100)):
            c.radius_critical2 = radius_temp2
            assert c.radius_critical2 >= 0

        c.tangent_length2, c.dogleg2 = min_dist_to_target(
            radius_effective2, c.distances2
        )
        c.dogleg2 = check_dogleg(c.dogleg2)
        c.dist_curve2, c.func_dogleg2 = get_curve_hold_data(
            radius_effective2, c.dogleg2
        )
        c.vec2 = get_vec_target(
            c.pos_target, c.vec_target * -1, c.pos2,
            c.tangent_length2, c.dist_curve2, c.func_dogleg2
        )

        tangent_temp2 = c._get_tangent_temp(c.tangent_length2)
        pos3_new = c.pos2 - tangent_temp2 * c.vec2

        # Damped update: averaging old and new prevents oscillation when the
        # target is close relative to the radius of curvature.
        c.pos3 = 0.5 * pos3_new + 0.5 * prev_pos3

        vec23_denom = np.linalg.norm(c.pos3 - c.pos2)
        if vec23_denom == 0:
            c.vec23.append(np.array([0., 0., 0.]))
        else:
            c.vec23.append((c.pos3 - c.pos2) / vec23_denom)

        c.error = np.allclose(
            c.vec23[-1], vec_old,
            equal_nan=True,
            rtol=c.min_error * 10,
            atol=c.min_error * 0.1
        )

        c.errors.append(c.error)
        c.pos3_list.append(c.pos3.copy())
        c.pos2_list.append(c.pos2.copy())
        c.md_target = c.md1 + c.dist_curve + tangent_temp2 + c.dist_curve2
        c.delta_radius_list.append(abs(c.radius_critical - c.radius_critical2))
        c.dls = max(
            np.radians(dls_from_radius(c.radius_design)),
            np.radians(dls_from_radius(c.radius_critical))
        )
        c.dls2 = max(
            np.radians(dls_from_radius(c.radius_design2)),
            np.radians(dls_from_radius(c.radius_critical2))
        )

        if c.error:
            break

        c.iterations += 1
        vec_old = c.vec23[-1].copy()

    if result:
        return c
    result_val = 0.
    if radius_temp2 < c.radius_design2:
        result_val += c.radius_design2 - radius_temp2
    if radius_temp1 < c.radius_design:
        result_val += c.radius_design - radius_temp1
    return result_val


def check_dogleg(dogleg):
    """Ensures the dogleg angle is positive by wrapping negative values.

    Parameters
    ----------
    dogleg : float
        Dogleg angle in radians.

    Returns
    -------
    float
        The dogleg angle normalized to [0, 2*pi).
    """
    # the code assumes angles are positive and clockwise
    if dogleg < 0:
        dogleg_new = dogleg + 2 * np.pi
        return dogleg_new
    else:
        return dogleg


def mod_vec(vec, error=1e-5):
    """Slightly perturbs a direction vector to avoid exact antiparallel degeneracy.

    Parameters
    ----------
    vec : ndarray
        Unit direction vector in NEV coordinates.
    error : float
        Perturbation magnitude applied to the vertical component.

    Returns
    -------
    tuple
        A tuple of (perturbed_vec, inclination, azimuth).
    """
    # if it's not working then twat it with a hammer
    vec_mod = vec * np.array([1, 1, 1 - error])
    vec_mod /= np.linalg.norm(vec_mod)
    inc_mod, azi_mod = get_angles(vec_mod, nev=True).T

    return vec_mod, inc_mod, azi_mod


def get_pos(pos1, vec1, vec2, dist_curve, func_dogleg):
    """Computes the end position of a minimum-curvature arc.

    Parameters
    ----------
    pos1 : ndarray
        Start position in NEV coordinates.
    vec1 : ndarray
        Start unit direction vector in NEV coordinates.
    vec2 : ndarray
        End unit direction vector in NEV coordinates.
    dist_curve : float
        Arc length of the curve section.
    func_dogleg : float
        Shape factor (ratio factor) for the curve.

    Returns
    -------
    ndarray
        End position in NEV coordinates.
    """
    return pos1 + (dist_curve * func_dogleg / 2) * (vec1 + vec2)


def get_vec_target(
    pos1,
    vec1,
    pos_target,
    tangent_length,
    dist_curve,
    func_dogleg
):
    """Derives the target unit vector from curve geometry and target position.

    Solves for the direction vector at the end of a curve-hold section
    given the start state, curve parameters, and target position.

    Parameters
    ----------
    pos1 : ndarray
        Start position in NEV coordinates.
    vec1 : ndarray
        Start unit direction vector in NEV coordinates.
    pos_target : ndarray
        Target position in NEV coordinates.
    tangent_length : float
        Length of the tangent (hold) section.
    dist_curve : float
        Arc length of the curve section.
    func_dogleg : float
        Shape factor (ratio factor) for the curve.

    Returns
    -------
    ndarray
        Target unit direction vector in NEV coordinates.
    """
    if dist_curve == 0:
        return vec1

    vec_target = (
        (
            pos_target - pos1 - (
                    dist_curve
                    * func_dogleg
                ) / 2 * vec1
        )
        /
        (
            (
                dist_curve * func_dogleg / 2
            ) + tangent_length
        )
    )

    vec_target /= np.linalg.norm(vec_target)

    return vec_target


def get_curve_hold_data(radius, dogleg):
    """Computes arc length and shape factor for a curve section.

    Parameters
    ----------
    radius : float
        Radius of curvature.
    dogleg : float
        Dogleg angle in radians.

    Returns
    -------
    tuple
        A tuple of (dist_curve, func_dogleg) where dist_curve is the arc
        length and func_dogleg is the minimum-curvature shape factor.
    """
    dist_curve = radius * dogleg
    func_dogleg = shape_factor(dogleg)

    return (
        dist_curve,
        func_dogleg
    )


def shape_factor(dogleg):
    """Computes the minimum-curvature shape factor for a dogleg angle.

    Parameters
    ----------
    dogleg : float
        Dogleg angle in radians.

    Returns
    -------
    float
        The ratio factor (shape factor) for minimum-curvature interpolation.
    """
    return get_rf(dogleg)


def min_dist_to_target(radius, distances):
    """Computes tangent length and dogleg for a curve-hold section to a target.

    Parameters
    ----------
    radius : float
        Radius of curvature for the curve section.
    distances : tuple
        Tuple of (dist_to_target, dist_perp_to_target,
        dist_norm_to_target) geometric distances.

    Returns
    -------
    tangent_length : float
        Hold section length.
    dogleg : float
        Curve angle in radians.
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
    """Computes minimum-curvature parameters when the design DLS is insufficient.

    Used when the target cannot be reached with the design radius, so the
    curve section uses the minimum radius geometrically required.

    Parameters
    ----------
    distances : tuple
        Tuple of (dist_to_target, dist_perp_to_target,
        dist_norm_to_target) geometric distances.

    Returns
    -------
    tangent_length : float
        Always 0 (pure curve, no hold).
    radius_critical : float
        Minimum required radius of curvature.
    dogleg : float
        Curve angle in radians.
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
        if np.isnan(radius_critical):
            radius_critical = np.nan
        else:
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
    """Computes the critical radius for a given target geometry.

    The critical radius is the minimum curvature radius needed to reach
    the target with a pure curve (no tangent). Below this radius, a
    curve-hold path is possible; above it, minimum curvature is needed.

    Parameters
    ----------
    radius : float
        Design radius of curvature.
    distances : tuple
        Tuple of (dist_to_target, dist_perp_to_target,
        dist_norm_to_target) geometric distances.
    min_error : float
        Error tolerance factor applied to the result.

    Returns
    -------
    float
        The critical radius. Returns 0 if the normal distance is zero.
    """
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

    if np.isnan(radius_critical):
        radius_critical = np.nan
    else:
        assert radius_critical > 0

    return radius_critical


def interpolate_well(sections, step=30):
    """Constructs interpolated survey data from a list of Connector sections.

    Parameters
    ----------
    sections : Connector or list of Connector
        Connector objects defining the well trajectory.
    step : float
        Desired delta measured depth between interpolated survey
        points.

    Returns
    -------
    list
        A list of interpolated survey data dictionaries.
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
    """Interpolates survey points along a curve section at regular MD intervals.

    Uses Rodrigues' rotation formula for numerical stability, especially
    for near-180-degree doglegs where SLERP becomes unstable.

    Parameters
    ----------
    md1 : float
        Measured depth at the start of the curve.
    pos1 : ndarray
        Start position in NEV coordinates.
    vec1 : ndarray
        Start unit direction vector in NEV coordinates.
    vec2 : ndarray
        End unit direction vector in NEV coordinates.
    dist_curve : float
        Arc length of the curve section.
    dogleg : float
        Total dogleg angle in radians.
    func_dogleg : float
        Shape factor (ratio factor) for the curve.
    step : float
        Desired delta measured depth between interpolated points.
    endpoint : bool
        If True, includes the curve endpoint in the output.

    Returns
    -------
    dict
        Dictionary with keys 'md', 'vec', 'inc', 'azi', 'dogleg'
        containing numpy arrays of interpolated survey data.
    """
    # sometimes the curve section has no length
    # this if statement handles this event
    if any((dist_curve == 0, np.isnan(dist_curve))):
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

    # Rodrigues' rotation formula is numerically superior to SLERP when dogleg
    # is near π.  SLERP weights contain 1/sin(dogleg) which amplifies errors
    # by ~1/sin(π-ε) ≈ 1/ε for near-180° arcs (e.g. case #492 with dogleg≈179°
    # amplifies errors 44×, producing spiralling visualisation paths).
    #
    # Rodrigues: vec(t) = cos(t)*vec1 + sin(t)*in_plane
    # where in_plane is the unit vector in the arc plane, perpendicular to vec1.
    # The formula never divides by sin(dogleg) during evaluation — only during
    # the one-time setup of in_plane — so numerical errors do not accumulate.
    cross = np.cross(vec1, vec2)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-10:
        # vec1 ≈ vec2 (zero dogleg) or exactly antiparallel (degenerate arc):
        # return the start direction for all points.
        vec = np.tile(vec1, (len(md), 1))
    else:
        rot_axis = cross / cross_norm          # unit rotation axis ⊥ arc plane
        in_plane = np.cross(rot_axis, vec1)    # unit vector in arc plane, ⊥ vec1
        vec = (
            np.cos(dogleg_interp) * vec1
            + np.sin(dogleg_interp) * in_plane
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
    """Interpolates survey points along a hold (tangent) section.

    Parameters
    ----------
    md1 : float
        Measured depth at the start of the hold.
    pos1 : ndarray
        Start position in NEV coordinates.
    vec1 : ndarray
        Constant unit direction vector during the hold.
    md2 : float
        Measured depth at the end of the hold.
    step : float
        Desired delta measured depth between interpolated points.
    endpoint : bool
        If True, includes the hold endpoint in the output.

    Returns
    -------
    dict
        Dictionary with keys 'md', 'vec', 'inc', 'azi', 'dogleg'
        containing numpy arrays of interpolated survey data.
    """
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
    """Interpolates a minimum-curve section, dispatching by sub-method.

    Parameters
    ----------
    section : Connector
        A Connector object with method 'min_curve'.
    step : float
        Desired delta measured depth between interpolated points.
    data : list or None
        Optional list to append results to.

    Returns
    -------
    list
        A list of interpolated survey data dictionaries.
    """
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
    """Interpolates a hold-method Connector section.

    Parameters
    ----------
    section : Connector
        A Connector object with method 'hold'.
    step : float
        Desired delta measured depth between interpolated points.
    data : list or None
        Optional list to append results to.

    Returns
    -------
    list
        A list of interpolated survey data dictionaries.
    """
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
    """Interpolates a min-curve-to-target Connector section.

    Parameters
    ----------
    section : Connector
        A Connector object with method 'min_curve_to_target'.
    step : float
        Desired delta measured depth between interpolated points.
    data : list or None
        Optional list to append results to.

    Returns
    -------
    list
        A list of interpolated survey data dictionaries.
    """
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
    """Interpolates a min-dist-to-target Connector section (curve + hold).

    Parameters
    ----------
    section : Connector
        A Connector object with method 'min_dist_to_target'.
    step : float
        Desired delta measured depth between interpolated points.
    data : list or None
        Optional list to append results to.

    Returns
    -------
    list
        A list of interpolated survey data dictionaries.
    """
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
    """Interpolates a curve-hold-curve Connector section.

    Parameters
    ----------
    section : Connector
        A Connector object with method 'curve_hold_curve'.
    step : float
        Desired delta measured depth between interpolated points.
    data : list or None
        Optional list to append results to.

    Returns
    -------
    list
        A list of interpolated survey data dictionaries.
    """
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
    """Converts target parameters to a binary string for method lookup.

    Parameters
    ----------
    *inputs
        Variable number of target parameters (md2, inc2, azi2,
        pos2, vec2). Each is mapped to '1' if not None, '0' otherwise.

    Returns
    -------
    str
        A 5-character binary string encoding which parameters were provided.
    """
    input = [
        "0" if i is None else "1" for i in inputs
    ]

    return ''.join(input)


def connect_points(
    cartesians, vec_start=[0., 0., 1.], dls_design=3.0, nev=True,
    # step=30,
    md_start=0.
):
    """Connects a sequence of Cartesian points with Connector sections.

    Parameters
    ----------
    cartesians : list or ndarray
        Array of shape (n, 3) with positions as [n, e, tvd]
        (if nev=True) or [x, y, z] (if nev=False).
    vec_start : list or ndarray
        Unit start direction vector in the corresponding
        coordinate system.
    dls_design : float or list
        Design DLS in deg/30m (or deg/100ft). Can be a
        scalar or array of length n.
    nev : bool
        If True, cartesians are in NEV coordinates; if False, XYZ.
    md_start : float
        Measured depth at the first point.

    Returns
    -------
    list
        A list of Connector objects linking consecutive points.
    """
    if nev:
        pos_nev = np.array(cartesians).reshape(-1, 3)
        vec_nev = np.zeros_like(pos_nev)
        vec_nev[0] = np.array(vec_start).reshape(-1, 3)
    else:
        pos_nev = get_nev(cartesians)
        vec_nev = np.zeros_like(pos_nev)
        vec_nev[0] = get_nev(vec_start)

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
    """Extracts a minimal well plan from a drilled survey.

    Identifies the minimum number of control points (start/end of hold
    or build/turn sections) needed to reproduce the survey trajectory
    within the given tolerance.

    Parameters
    ----------
    survey : Survey
        A welleng Survey object representing the drilled well.
    tolerance : float
        Fit tolerance. Higher values produce fewer control
        points but a looser fit.
    dls_design : float
        Minimum design DLS in deg/30m for the planned trajectory.
    step : float
        Desired MD step interval for the output survey.

    Returns
    -------
    list
        A list of Connector objects representing the planned sections.

    Raises
    ------
    AssertionError
        If dls_design is not greater than 0.
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

    # data = interpolate_well(sections, step=step)

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


def drop_off(
    target_inc: float, dls: float, delta_md: float | None = None,
    node: Node | None = None, tol: float = 1e-5
) -> list:
    """Computes trajectory sections to drop off (or build) to a target inclination.

    Use ``extend_to_tvd`` if a specific TVD target is also required.

    Parameters
    ----------
    target_inc : float
        Target inclination in degrees.
    dls : float
        Design DLS in deg/30m.
    delta_md : float or None
        Maximum section length in meters. If None, the section
        is unconstrained.
    node : Node or None
        Starting Node. Defaults to surface pointing down.
    tol : float
        Tolerance for tangent section length; sections shorter than
        this are omitted.

    Returns
    -------
    list
        A list of Nodes describing the trajectory. Contains one Node
        (the arc endpoint) or two (arc endpoint plus tangent endpoint)
        if the target inclination was achieved within the section.
    """
    def _drop_off(
            x: tuple,
            node,
            return_data: bool = False
    ) -> tuple:
        dogleg, toolface = x
        pos2, vec2, arc_length = get_arc(
            dogleg, radius,
            0 if -np.pi / 2 <= toolface <= np.pi / 2 else np.pi,
            node.pos_nev, node.vec_nev
        )
        if return_data:
            return (pos2, vec2, arc_length)
        else:
            inc, _ = np.degrees(
                get_angles(vec2, nev=True)[0]
            )
            return abs(inc - target_inc)

    node = Node() if node is None else node
    node.md = 0 if node.md is None else node.md
    radius = radius_from_dls(dls)
    if isinstance(delta_md, np.ndarray):
        delta_md = delta_md[0]
    max_dogleg = (
        2 * np.pi if delta_md is None
        else delta_md / radius
    )

    args = (node,)
    bounds = [
        [0, min(2 * np.pi, max_dogleg)],
        [-np.pi, np.pi]
    ]
    x0 = [bounds[0][1] / 2, np.pi]
    result = minimize(
        _drop_off, x0, args=args, bounds=bounds,
        method="SLSQP"
    )
    pos2, vec2, arc_length = _drop_off(
        result.x, *args, return_data=True
    )
    tangent_length = (
        0 if delta_md is None
        else delta_md - arc_length
    )
    node2 = Node(
        pos=pos2, vec=vec2, md=node.md + arc_length
    )
    if tangent_length > tol:
        pos3 = pos2 + tangent_length * vec2
        node3 = Node(
            pos=pos3, vec=vec2, md=node2.md + tangent_length
        )
        return [node2, node3]
    else:
        return [node2]


def extend_to_tvd(
    target_tvd: float, node: Node | None = None,
    delta_md: float | None = None,
    target_inc: float | None = None, dls: float | None = None
) -> list:
    """Computes Connector sections to reach a target TVD with optional inclination change.

    Parameters
    ----------
    target_tvd : float
        Target true vertical depth in meters.
    node : Node or None
        Starting Node. Defaults to surface pointing down.
    delta_md : float or None
        Maximum section length in meters. If None, unconstrained.
    target_inc : float or None
        Target inclination in degrees at the target TVD.
        If provided, the solver attempts to achieve this inclination
        and holds tangent to the target TVD.
    dls : float or None
        Design DLS in deg/30m. Defaults to 2.5 if None and
        target_inc is provided.

    Returns
    -------
    list
        A list of Connector objects. Contains one Connector (curve only)
        or two (curve plus tangent hold) if the target inclination was
        achieved within the section.

    Examples
    --------
    A well at 30 degrees inclination dropping to vertical:

    >>> import welleng as we
    >>> node = we.node.Node(pos=[0, 0, 3000], md=4000, inc=30, azi=135)
    >>> connectors = we.connector.extend_to_tvd(
    ...     target_tvd=3200, node=node, target_inc=0, dls=3
    ... )
    """
    if node is None:
        node = Node()
    node.md = 0 if node.md is None else node.md  # default value is None which complicates things
    _delta_md = 1e-8 if delta_md is None else delta_md
    connections = []
    if target_inc is None:
        def _extend_tvd(
            delta_md, pos, vec, target_tvd, return_data=False
        ):
            pos2 = np.array(pos) + delta_md * np.array(vec)
            if return_data:
                return pos2
            else:
                return abs(pos2[2] - target_tvd)

        args = (node.pos_nev, node.vec_nev, target_tvd)
        bounds = [[0, None]]
        result = minimize(
            _extend_tvd,
            _delta_md,
            args=args,
            bounds=bounds,
            method="Powell"
        )
        pos2 = _extend_tvd(result.x, *args, return_data=True)
        connections.append(Connector(
            pos1=node.pos_nev, vec1=node.vec_nev,
            md1=0 if node.md is None else node.md,
            pos2=pos2,
            dls_design=dls, force_min_curve=True
        ))
    else:
        def _drop_off(
            delta_md, target_inc, target_tvd, dls, node,
            return_data=False
        ):
            nodes = drop_off(
                target_inc, dls, delta_md, node
            )
            if return_data:
                return nodes
            else:
                return abs(nodes[-1].pos_nev[2] - target_tvd)

        dls = 2.5 if dls is None else dls
        args = (target_inc, target_tvd, dls, node)
        bounds = [
            [min(target_tvd - node.pos_nev[2], _delta_md), None]
        ]
        x0 = _delta_md
        result = minimize(
            _drop_off,
            x0,
            args=args,
            bounds=bounds,
            method='Powell'
        )
        nodes = _drop_off(
            result.x, *args, return_data=True
        )
        connections.append(Connector(
            node1=node,
            pos2=nodes[0].pos_nev, vec2=nodes[0].vec_nev,
            dls_design=dls, force_min_curve=True
        ))
        if len(nodes) > 1:
            connections.append(Connector(
                node1=connections[-1].node_end,
                pos2=nodes[-1].pos_nev,
                dls_design=dls, force_min_curve=True
            ))
    return connections


