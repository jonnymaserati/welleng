"""Wellbore trajectory connector.

Resolves a minimum-curvature connection between two stations (each a position
and/or a direction), classifying it into the appropriate type — a straight
*hold*, a single *curve* (min-curvature), a *curve-hold*, or a *curve-hold-curve*
(the circle-line-circle, CLC, point-to-target case) — and returns the arc/hold
sections, doglegs and measured depths. The curve-hold-curve case is solved in
closed form via Sawaryn (2021, SPE-204111-PA) — see ``welleng.sawaryn_analytical``.
"""

import warnings
from copy import copy
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .node import Node, get_node_params
from .sawaryn_analytical import max_radius, solve_clc
from .utils import (
    dls_from_radius, get_angles,
    get_dogleg, get_nev, get_rf, get_vec, radius_from_dls, get_arc
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

    method: str
    node_start: Node
    node_end: Node
    unit: str
    pos1: np.ndarray
    vec1: np.ndarray
    inc1: float
    azi1: float
    md1: float
    pos2: Optional[np.ndarray]
    vec2: Optional[np.ndarray]
    inc2: Optional[float]
    azi2: Optional[float]
    md2: Optional[float]
    pos3: Optional[np.ndarray]
    vec3: Optional[np.ndarray]
    inc3: Optional[float]
    azi3: Optional[float]
    md3: Optional[float]
    pos_target: np.ndarray
    vec_target: np.ndarray
    inc_target: float
    azi_target: float
    md_target: float
    dogleg: Any
    dogleg2: Any
    dist_curve: Any
    dist_curve2: Any
    func_dogleg: Any
    func_dogleg2: Any
    tangent_length: Optional[float]
    dls: float
    dls2: float
    dls_design: float
    dls_design2: float
    radius_design: float
    radius_design2: float
    radius_critical: float
    radius_critical2: float
    distances: tuple
    radii: list

    def __init__(
        self,
        node1: Optional[Node] = None,
        node2: Optional[Node] = None,
        pos1: ArrayLike = [0., 0., 0.],
        vec1: Optional[ArrayLike] = None,
        inc1: Optional[float] = None,
        azi1: Optional[float] = None,
        md1: float = 0,
        dls_design: Optional[float] = 3.0,
        dls_design2: Optional[float] = None,
        md2: Optional[float] = None,
        pos2: Optional[ArrayLike] = None,
        vec2: Optional[ArrayLike] = None,
        inc2: Optional[float] = None,
        azi2: Optional[float] = None,
        degrees: bool = True,
        unit: str = 'meters',
        min_error: float = 1e-5,
        delta_dls: float = 0.1,
        min_tangent: float = 0.,
        max_iterations: int = 1_000,
        force_min_curve: bool = False,
        closest_approach: bool = False,
        on_infeasible: str = 'raise',
        direct_only: bool = False
    ) -> None:
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
            in curve-hold-curve solutions. Deprecated: accepted for
            backwards compatibility but unused by the analytic CHC path.
        min_tangent : float
            Minimum tangent length to stabilize curve-hold-curve
            iteration. Deprecated: accepted for backwards compatibility
            but unused by the analytic CHC path.
        max_iterations : int
            Maximum iteration count for curve-hold-curve fitting.
            Deprecated: accepted for backwards compatibility but unused
            by the analytic CHC path.
        force_min_curve : bool
            If True, forces minimum-curvature method.
        closest_approach : bool
            If True, finds the closest-approach trajectory
            when the target is inside the critical radius.
        on_infeasible : str
            Behaviour when no curve-hold-curve solution exists at the
            design radii. ``'raise'`` (default) raises ``ValueError``.
            ``'max_radius'`` falls back to the gentlest feasible curve —
            the beta=0 biarc at the largest radius admitting a valid CLC
            (see :func:`welleng.sawaryn_analytical.max_radius`) — and
            emits a ``UserWarning`` that the design DLS is exceeded.
        direct_only : bool
            Reject solutions in which either arc turns through more than
            pi (180 deg) — i.e. accept only the "direct" way round, never a
            long-way loop. Default ``False`` (any arc angle is rendered).

            .. warning::
               A successful solve is **NOT** a dogleg-severity feasibility
               test unless ``direct_only=True``. A long-way (>pi) arc exists
               at essentially any radius, so a search of the form "does a
               CLC exist at this DLS" succeeds at almost ANY DLS — it will
               happily return a multi-kilometre corkscrew for a short
               pose-to-pose move. Any code that infers reachability from
               solver success (a DLS bisection, a ``feasible`` flag, an
               input gate) must set ``direct_only=True`` or check
               :attr:`dogleg` / :attr:`dogleg2` against pi itself.

        Raises
        ------
        AssertionError
            If input parameter combinations are invalid.
        """
        if node1 is not None:
            pos1, vec1, md1 = get_node_params(  # type: ignore[assignment]  # node1 Node fields are Optional but populated here
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
        self.max_iterations = max_iterations
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
        self.pos_target = None if pos2 is None else np.array(pos2).reshape(3)  # type: ignore[assignment]  # transient None; pos_target populated before use per method
        self.md_target = md2  # type: ignore[assignment]  # md_target set from Optional md2; solve populates it

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
                self.azi_target = np.radians(azi2)  # type: ignore[arg-type]  # azi2 non-None in this input-combination branch (mypy can't narrow)
            else:
                self.azi_target = azi2  # type: ignore[assignment]  # azi2 non-None in this branch
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
            self.vec_target = vec2  # type: ignore[assignment]  # guarded branch: vec2 already validated non-None above
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
        assert on_infeasible in ('raise', 'max_radius'), (
            "on_infeasible must be 'raise' or 'max_radius'"
        )
        self.on_infeasible = on_infeasible
        self.direct_only = bool(direct_only)

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

    def _get_nodes(self) -> None:
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

    def _min_dist_to_target(self) -> None:
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

    def _min_curve_to_target(self) -> None:
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

    def _use_method(self) -> None:
        if self.method == 'hold':
            self._hold()
        elif self.method == 'min_curve':
            self._min_curve()
        elif self.method == 'curve_hold_curve':
            # Closed-form Sawaryn (2021, SPE-204111-PA) point-to-target solve.
            # Populates the state when a CLC exists at the design radii.
            if self._solve_chc_analytical():
                self._chc_solver = 'analytical'
            elif self.on_infeasible == 'max_radius' and self._solve_chc_max_radius():
                self._chc_solver = 'max_radius'
            else:
                # No CLC at the design radii: the target needs tighter curvature
                # than the design DLS allows. We do NOT silently tighten — the
                # caller decides (e.g. sweep the radius / raise dls_design, then
                # retry, or opt in to on_infeasible='max_radius'). See solve_clc
                # in welleng.sawaryn_analytical.
                raise ValueError(
                    "No curve-hold-curve solution at the design radii "
                    f"(R1={self.radius_design:.6g}, R2={self.radius_design2:.6g}) "
                    "for the given start and target. The target requires tighter "
                    "curvature than the design dogleg severity. Retry with a "
                    "smaller radius / larger dls_design (an R-sweep), or relax "
                    "the target."
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

        if self.direct_only:
            self._assert_direct()

    def _assert_direct(self) -> None:
        """Reject a solved trajectory that loops the long way round.

        Called after the solve when ``direct_only=True``. Since PR #305 the
        renderer handles arcs of any angle, so a long-way (>pi) arc is a valid
        solution at essentially any radius -- which means solver success alone
        cannot certify that a pose pair is reachable at a given dogleg
        severity. Callers using the solve as a feasibility predicate opt in
        here and get a ``ValueError`` instead of a corkscrew.
        """
        arcs = [
            a for a in (getattr(self, 'dogleg', None),
                        getattr(self, 'dogleg2', None))
            if a is not None
        ]
        # tolerance: a hair over pi is a numerical artefact of a half-turn,
        # not a deliberate loop
        over = [abs(float(a)) for a in arcs if abs(float(a)) > np.pi + 1e-9]
        if over:
            raise ValueError(
                "No DIRECT curve-hold-curve solution at the design radii "
                f"(R1={self.radius_design:.6g}, R2={self.radius_design2:.6g}): "
                "the solution turns through "
                + ", ".join(f"{np.degrees(a):.2f} deg" for a in over)
                + " (> 180 deg), i.e. the long way round. The pose pair needs "
                "a larger dls_design to be reachable directly. Pass "
                "direct_only=False to allow looping solutions."
            )

    def _get_method(self) -> None:
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

    def _get_initial_methods(self) -> None:
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

    def _closest_approach(self) -> None:
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

    def _min_curve(self) -> None:
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

    def _hold(self) -> None:
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

    def _get_angles_target(self) -> None:
        self.inc_target, self.azi_target = get_angles(
            self.vec_target, nev=True
        ).reshape(2)

    def _get_md_target(self) -> None:
        self.md_target = (
            self.dist_curve
            + self.tangent_length
            + self.dist_curve2
            + self.md1
        )

    def _solve_chc_analytical(
        self, R1: Optional[float] = None, R2: Optional[float] = None
    ) -> bool:
        """Closed-form curve-hold-curve solve — the primary CHC path.

        Solves the curve-hold-curve point-to-target problem analytically via
        the closed-form solution of Sawaryn (2021), SPE-204111-PA
        (:func:`welleng.sawaryn_analytical.solve_clc`), at the design radii
        ``radius_design`` / ``radius_design2``, and populates the full public
        CHC state (``pos2/vec2/md2``, ``pos3/vec3/md3``, ``md_target``,
        ``dogleg/dist_curve/dls/func_dogleg`` and the ``*2`` second-arc twins,
        ``tangent_length``, the ``inc*/azi*`` angles and ``radius_critical*``)
        exactly as the inherited iterative scheme would, but directly and to
        machine precision.

        Among all CLCs at the design radii it prefers the shortest whose two arc
        doglegs are each ``<= pi``. When the geometry admits only ``> pi`` arcs
        (the target tangent can be reached at the design radius only by turning
        more than 180°), that arc is still a valid circular curve and is
        rendered correctly — :func:`interpolate_curve` sweeps the long way round
        — so the shortest such solution is used rather than rejected. Only a
        genuinely empty solution set (no CLC at any turn at the design radii, so
        the target needs tighter curvature than ``dls_design``) or a degenerate
        reconstruction returns ``False``; the caller (``_use_method``) then
        raises ``ValueError``. The reconstructed state is finiteness-checked (the
        minimum-curvature shape factor is singular exactly at a 180° arc); any
        non-finite result also returns ``False``.

        Returns
        -------
        bool
            ``True`` if an analytical CLC was found and the state populated;
            ``False`` if no renderable CLC exists at the design radii (the
            caller then raises ``ValueError``).
        """
        # Radii default to the design radii — behaviour with no args is
        # identical to solving at the design DLS. A caller (the max_radius
        # fallback) may pass explicit radii to populate the state at a
        # different, feasible curvature without mutating the design radii.
        if R1 is None:
            R1 = self.radius_design
        if R2 is None:
            R2 = self.radius_design2

        # Parallel/antiparallel tangents (|mu|=1) make the general closed form
        # singular; solve_clc auto-routes those to the planar 2D form, but the
        # transient 1/(1-mu^2) still emits a benign divide warning -> silence it.
        with np.errstate(divide='ignore', invalid='ignore'):
            sols = solve_clc(
                self.pos1, self.vec1, self.pos_target, self.vec_target,
                R1, R2, return_all=True
            )
        if not sols:
            return False
        # Prefer short-way CLCs (each arc dogleg <= pi); among them take the
        # shortest. When NONE exist the geometry requires an arc that turns more
        # than pi -- still a valid circular curve, and it now renders correctly
        # (interpolate_curve sweeps the long way round), so fall back to the
        # shortest overall rather than rejecting. Preferring the <=pi set keeps
        # every previously-solved case bit-identical.
        _PI = np.pi + 1e-9
        candidates = [
            s for s in sols if s['alpha1'] <= _PI and s['alpha2'] <= _PI
        ]
        if not candidates:
            candidates = sols
        sol = min(candidates, key=lambda s: s['total_md'])

        beta = float(sol['beta'])
        alpha1 = check_dogleg(float(sol['alpha1']))
        alpha2 = check_dogleg(float(sol['alpha2']))

        t1 = self.vec1
        t4 = self.vec_target
        T1h = np.tan(alpha1 / 2)
        T2h = np.tan(alpha2 / 2)

        # Hold-section direction (Sawaryn 2021): the straight tangent that joins
        # the two arcs. dp = pos_target - pos1 keeps this frame-free (the task's
        # bare-form drops the pos1 offset, valid only at the origin).
        dp = self.pos_target - self.pos1
        denom = R1 * T1h + beta + R2 * T2h
        t2 = (dp - R1 * T1h * t1 - R2 * T2h * t4) / denom
        norm = np.linalg.norm(t2)
        if not np.isfinite(norm) or norm == 0:
            return False
        t2 = t2 / norm                       # unit by construction (~1e-9)

        # ── Arc 1: pos1 (vec1) -> pos2 (t2), dogleg alpha1 ──────────────────
        self.dogleg = alpha1
        self.dist_curve, self.func_dogleg = get_curve_hold_data(R1, alpha1)
        self.vec2 = t2
        self.pos2 = get_pos(
            self.pos1, self.vec1, self.vec2,
            self.dist_curve, self.func_dogleg
        ).reshape(3)
        self.md2 = self.md1 + abs(self.dist_curve)

        # ── Hold: pos2 -> pos3 along t2, length beta ────────────────────────
        self.tangent_length = beta
        self.vec3 = t2                       # straight hold: vec2 == vec3
        self.pos3 = (self.pos2 + beta * t2).reshape(3)
        self.md3 = self.md2 + beta

        # ── Arc 2: pos3 (t2) -> pos_target (vec_target), dogleg alpha2 ───────
        self.dogleg2 = alpha2
        self.dist_curve2, self.func_dogleg2 = get_curve_hold_data(R2, alpha2)
        self.md_target = self.md3 + abs(self.dist_curve2)

        # The design radii are used directly, so no critical-radius override is
        # active (mirrors the iterative path, which leaves these at inf when the
        # design DLS is achievable). DLS therefore equals the design DLS.
        self.radius_critical = np.inf
        self.radius_critical2 = np.inf
        self.dls = max(
            np.radians(dls_from_radius(R1)),
            np.radians(dls_from_radius(self.radius_critical))
        )
        self.dls2 = max(
            np.radians(dls_from_radius(R2)),
            np.radians(dls_from_radius(self.radius_critical2))
        )

        self.inc2, self.azi2 = get_angles(self.vec2, nev=True).reshape(2)
        self.inc3, self.azi3 = get_angles(self.vec3, nev=True).reshape(2)

        # Reject any non-finite reconstruction (e.g. a 180° arc where the
        # shape factor diverges) -> return False (caller raises ValueError).
        for arr in (self.pos2, self.pos3, self.vec2, self.vec3):
            if not np.all(np.isfinite(arr)):
                return False
        for val in (
            self.dist_curve, self.dist_curve2, self.func_dogleg,
            self.func_dogleg2, self.md2, self.md3, self.md_target
        ):
            if not np.isfinite(val):
                return False

        return True

    def _solve_chc_max_radius(self) -> bool:
        """Opt-in fallback when no CLC exists at the design radii.

        Finds the gentlest feasible curve — the ``beta=0`` biarc at the
        largest radius admitting a valid CLC (both arc doglegs ``<= pi``),
        via :func:`welleng.sawaryn_analytical.max_radius` — and populates
        the full CHC state at that critical radius. The design radii are
        left untouched; ``radius_critical``/``radius_critical2`` record the
        radii actually used and a ``UserWarning`` flags that the resulting
        DLS exceeds the design DLS.

        Returns
        -------
        bool
            ``True`` if a feasible max-radius biarc was found and the state
            populated; ``False`` otherwise (the caller then raises).
        """
        mr = max_radius(
            self.pos1, self.vec1, self.pos_target, self.vec_target,
            ratio=self.radius_design2 / self.radius_design
        )
        if mr is None:
            return False
        if not self._solve_chc_analytical(R1=mr['radius'], R2=mr['radius2']):
            return False
        self.radius_critical = mr['radius']
        self.radius_critical2 = mr['radius2']
        warnings.warn(
            "No curve-hold-curve solution exists at the design DLS "
            f"(R1={self.radius_design:.6g}, R2={self.radius_design2:.6g}); "
            "falling back to the maximum feasible radius "
            f"(R1={mr['radius']:.6g}, R2={mr['radius2']:.6g}). The resulting "
            "dogleg severity EXCEEDS the design DLS.",
            UserWarning,
        )
        return True

    def interpolate(self, step: float = 30) -> list:
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

    def _get_distances(
        self, pos1: np.ndarray, vec1: np.ndarray, pos_target: np.ndarray
    ) -> tuple:
        # Decompose the start->target vector into components along (perp) and
        # normal to the start tangent vec1. As the target approaches PRECISE
        # alignment with vec1 the connection degenerates to a pure hold
        # (dist_norm -> 0); this is handled DETERMINISTICALLY by clamping the
        # radicand to >= 0 (float rounding can otherwise make it tiny-negative
        # -> sqrt NaN) and letting the dist_norm == 0 branches downstream
        # (min_curve_to_target / get_radius_critical -> radius_critical inf/0)
        # route it as a hold. No random target nudge (the previous `_mod_pos`
        # jiggle) -- that injected non-reproducible position error to dodge this
        # exact-alignment singularity; the clamp resolves it without any error.
        if np.allclose(pos1, pos_target):
            return (0, 0, 0)

        dist_to_target = np.linalg.norm((pos_target - pos1))
        dist_perp_to_target = np.dot((pos_target - pos1), vec1)
        if dist_perp_to_target > dist_to_target:
            # a tolerance is in play; keep the radicand non-negative
            dist_perp_to_target = dist_to_target

        dist_norm_to_target = (
            max(dist_to_target ** 2 - dist_perp_to_target ** 2, 0.0)
        ) ** 0.5

        return (
            dist_to_target,
            dist_perp_to_target,
            dist_norm_to_target
        )


def check_dogleg(dogleg: ArrayLike) -> Union[float, np.ndarray]:
    """Ensures the dogleg angle is positive by wrapping negative values.

    Accepts scalar or array-like; output shape matches input.

    Parameters
    ----------
    dogleg : float or array_like
        Dogleg angle(s) in radians.

    Returns
    -------
    float or ndarray
        The dogleg angle(s) normalized to [0, 2*pi).
    """
    # the code assumes angles are positive and clockwise
    dogleg = np.asarray(dogleg, dtype=float)
    wrapped = np.where(dogleg < 0, dogleg + 2 * np.pi, dogleg)
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def mod_vec(vec: np.ndarray, error: float = 1e-5) -> tuple:
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


def get_pos(
    pos1: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray,
    dist_curve: float,
    func_dogleg: float,
) -> np.ndarray:
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
    pos1: ArrayLike,
    vec1: ArrayLike,
    pos_target: ArrayLike,
    tangent_length: ArrayLike,
    dist_curve: ArrayLike,
    func_dogleg: ArrayLike
) -> np.ndarray:
    """Derives the target unit vector from curve geometry and target position.

    Solves for the direction vector at the end of a curve-hold section
    given the start state, curve parameters, and target position. Accepts
    either scalar inputs (legacy shape-(3,) positions/vectors with scalar
    tangent_length/dist_curve/func_dogleg) or batched inputs (leading
    batch dims on all arrays, positions/vectors with trailing axis 3).

    Parameters
    ----------
    pos1 : ndarray, shape (..., 3)
        Start position in NEV coordinates.
    vec1 : ndarray, shape (..., 3)
        Start unit direction vector in NEV coordinates.
    pos_target : ndarray, shape (..., 3)
        Target position in NEV coordinates.
    tangent_length : float or ndarray, shape (...)
        Length of the tangent (hold) section.
    dist_curve : float or ndarray, shape (...)
        Arc length of the curve section. Where equal to zero, the input
        ``vec1`` is returned unchanged (pure-hold fallback).
    func_dogleg : float or ndarray, shape (...)
        Shape factor (ratio factor) for the curve.

    Returns
    -------
    ndarray, shape (..., 3)
        Target unit direction vector in NEV coordinates.
    """
    pos1 = np.asarray(pos1, dtype=float)
    vec1 = np.asarray(vec1, dtype=float)
    pos_target = np.asarray(pos_target, dtype=float)
    tangent_length = np.asarray(tangent_length, dtype=float)
    dist_curve = np.asarray(dist_curve, dtype=float)
    func_dogleg = np.asarray(func_dogleg, dtype=float)

    half = dist_curve * func_dogleg / 2
    denom = half + tangent_length

    # Broadcast per-sample scalars to match pos/vec batch shape (..., 3).
    half_b = half[..., None] if half.ndim >= 1 else half
    denom_b = denom[..., None] if denom.ndim >= 1 else denom

    vec_target = (pos_target - pos1 - half_b * vec1) / denom_b

    # Axis-aware normalise — works for scalar shape-(3,) and batched (..., 3).
    norm = np.linalg.norm(vec_target, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        vec_target = vec_target / norm

    # Pure-hold fallback: where dist_curve == 0, return vec1.
    zero_mask = (dist_curve == 0)
    if zero_mask.ndim == 0:
        if bool(zero_mask):
            return vec1
        return vec_target
    return np.where(zero_mask[..., None], vec1, vec_target)


def get_curve_hold_data(
    radius: Union[float, np.ndarray], dogleg: Union[float, np.ndarray]
) -> tuple:
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


def shape_factor(dogleg: ArrayLike) -> Any:
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


def solve_curve_hold_batch(
    pos1: ArrayLike, vec1: ArrayLike, pos_target: ArrayLike, radius: ArrayLike
) -> dict:
    """Vectorised curve-hold connector: fixed start pose, fixed target pos.

    Solves the minimum-MD curve-then-hold geometry from a start pose
    ``(pos1, vec1)`` to a target position ``pos_target`` with a given
    design radius. The target tangent vector is an OUTPUT of the solve —
    computed analytically from the geometry — not an input. Equivalent
    to ``Connector(pos1=..., vec1=..., pos2=pos_target, dls_design=...)``
    in the ``'curve_hold'`` mode (binary code ``00110`` in
    ``_get_initial_methods``), but operates element-wise on arrays so a
    large sweep is one numpy call rather than a Python loop over
    ``Connector`` instances.

    Parameters
    ----------
    pos1 : array_like, shape (..., 3)
        Start positions in NEV coordinates. Arbitrary leading batch shape.
    vec1 : array_like, shape (..., 3)
        Unit direction vectors at the start. Must share ``pos1``'s leading
        shape.
    pos_target : array_like, shape (..., 3)
        Target positions in NEV coordinates. Must share ``pos1``'s leading
        shape.
    radius : float or array_like, shape (...)
        Design radius of curvature. Broadcasts against the leading shape.

    Returns
    -------
    dict
        All entries are ndarrays whose leading shape matches the inputs.

        - ``'pos2'`` shape (..., 3) — end of the curve / start of the hold.
        - ``'vec_target'`` shape (..., 3) — computed unit tangent at target.
        - ``'tangent_length'`` shape (...) — hold-section length.
        - ``'dogleg'`` shape (...) — curve angle, radians.
        - ``'dist_curve'`` shape (...) — arc length of the curve section.
        - ``'md'`` shape (...) — total measured depth (curve + hold).

    Notes
    -----
    When the target is exactly along ``vec1`` (pure-hold degenerate case),
    the solver returns ``dogleg = 0``, ``tangent_length = dist_to_target``,
    ``vec_target = vec1``, and ``pos2 = pos1``. This matches the scalar
    ``Connector`` behaviour in that regime.

    The underlying helpers (``min_dist_to_target``, ``get_curve_hold_data``,
    ``get_vec_target``) have all been array-safe since the vectorisation
    patch; this function is just a thin wrapper that computes the three
    intermediate distance scalars and composes the helpers.
    """
    pos1 = np.asarray(pos1, dtype=float)
    vec1 = np.asarray(vec1, dtype=float)
    pos_target = np.asarray(pos_target, dtype=float)
    radius = np.asarray(radius, dtype=float)

    delta = pos_target - pos1
    dist_to_target = np.linalg.norm(delta, axis=-1)
    dist_perp_to_target = np.sum(delta * vec1, axis=-1)
    # Same clamp the scalar Connector._get_distances applies to catch fp drift
    # where the projected distance can marginally exceed the straight-line
    # distance.
    dist_perp_to_target = np.minimum(dist_perp_to_target, dist_to_target)
    dist_norm_to_target = np.sqrt(
        np.clip(dist_to_target ** 2 - dist_perp_to_target ** 2, 0.0, None)
    )

    distances = (dist_to_target, dist_perp_to_target, dist_norm_to_target)
    tangent_length, dogleg = min_dist_to_target(radius, distances)
    # Wrap negative doglegs by +2π to match the scalar Connector
    # (connector.py `_min_dist_to_target` calls `check_dogleg` on the raw
    # output before building the path). Without this, a geometry that
    # dictates a "backward" turn yields negative dist_curve and the
    # reported total MD ends up negative.
    dogleg = check_dogleg(dogleg)
    dist_curve, func_dogleg = get_curve_hold_data(radius, dogleg)

    vec_target = get_vec_target(
        pos1, vec1, pos_target, tangent_length, dist_curve, func_dogleg
    )

    # pos2 = end of arc = target - (hold-section vector)
    tl = tangent_length[..., None] if tangent_length.ndim >= 1 else tangent_length
    pos2 = pos_target - tl * vec_target
    md = dist_curve + tangent_length

    return {
        "pos2": pos2,
        "vec_target": vec_target,
        "tangent_length": tangent_length,
        "dogleg": dogleg,
        "dist_curve": dist_curve,
        "md": md,
    }


def min_dist_to_target(
    radius: Union[float, np.ndarray], distances: tuple
) -> tuple:
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


def min_curve_to_target(distances: tuple) -> tuple:
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


def get_radius_critical(
    radius: float, distances: tuple, min_error: float
) -> float:
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


def interpolate_well(
    sections: Union["Connector", list], step: float = 30
) -> list:
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
    md1: float,
    pos1: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray,
    dist_curve: float,
    dogleg: float,
    func_dogleg: float,
    step: Optional[float],
    endpoint: bool = False
) -> dict:
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

    # Tangent along the arc:  vec(t) = cos(t)*vec1 + sin(t)*u,  where u is the
    # in-plane unit vector fixed by the arc's OWN end tangent and angle:
    #
    #     u = (vec2 - cos(dogleg)*vec1) / sin(dogleg)
    #
    # This is exact and unit for ANY arc angle (vec1·vec2 == cos(dogleg) makes
    # |u| == 1), so a dogleg > pi renders correctly with no short-way/long-way
    # special case: sin(dogleg) < 0 flips u so the sweep goes the long way round
    # and still lands on vec2. (This is the Sawaryn-bridge ``_arc_axis`` form the
    # api uses to render every CLC result, incl. loops.) The 1/sin(dogleg) is a
    # one-time setup of u, not a per-point divide, so — like the previous
    # Rodrigues form — it does not amplify errors during evaluation (SLERP's
    # per-weight 1/sin was the ~44x near-180° amplifier we avoid). sin(dogleg)
    # vanishes only at a zero turn (straight -> vec1) or an exact pi turn
    # (antiparallel tangents, arc plane undetermined by the tangents alone: a
    # measure-zero singularity left as the start direction).
    sin_dl = np.sin(dogleg)
    if abs(sin_dl) < 1e-10:
        vec = np.tile(vec1, (len(md), 1))
    else:
        u = (vec2 - np.cos(dogleg) * vec1) / sin_dl
        vec = np.cos(dogleg_interp) * vec1 + np.sin(dogleg_interp) * u
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


def interpolate_hold(
    md1: float,
    pos1: np.ndarray,
    vec1: np.ndarray,
    md2: float,
    step: Optional[float],
    endpoint: bool = False
) -> dict:
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


def get_min_curve(
    section: "Connector", step: float = 30, data: Optional[list] = None
) -> list:
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


def get_interpolate_hold(
    section: "Connector", step: float = 30, data: Optional[list] = None
) -> list:
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


def get_interpolate_min_curve_to_target(
    section: "Connector", step: float = 30, data: Optional[list] = None
) -> list:
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


def get_interpolate_min_dist_to_target(
    section: "Connector", step: float = 30, data: Optional[list] = None
) -> list:
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
        vec2=section.vec2,  # type: ignore[arg-type]  # vec2 populated for this method before interpolation
        dist_curve=section.dist_curve,
        dogleg=section.dogleg,
        func_dogleg=section.func_dogleg,
        step=step
    ))

    # the hold section
    data.append(interpolate_hold(
        md1=section.md2,  # type: ignore[arg-type]  # md set for this method before interpolation
        pos1=section.pos2,  # type: ignore[arg-type]  # pos populated for this method before interpolation
        vec1=section.vec2,  # type: ignore[arg-type]  # vec populated for this method before interpolation
        md2=section.md_target,
        step=step,
        endpoint=True
    ))

    return data


def get_interpololate_curve_hold_curve(
    section: "Connector", step: float = 30, data: Optional[list] = None
) -> list:
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
        vec2=section.vec2,  # type: ignore[arg-type]  # vec2 populated for this method before interpolation
        dist_curve=section.dist_curve,
        dogleg=section.dogleg,
        func_dogleg=section.func_dogleg,
        step=step
    ))

    # the hold section
    data.append(interpolate_hold(
        md1=section.md2,  # type: ignore[arg-type]  # md set for this method before interpolation
        pos1=section.pos2,  # type: ignore[arg-type]  # pos populated for this method before interpolation
        vec1=section.vec2,  # type: ignore[arg-type]  # vec populated for this method before interpolation
        md2=section.md3,  # type: ignore[arg-type]  # md3 populated for curve-hold-curve before interpolation
        step=step
    ))

    # the second curve section
    data.append(interpolate_curve(
        md1=section.md3,  # type: ignore[arg-type]  # md3 populated for curve-hold-curve before interpolation
        pos1=section.pos3,  # type: ignore[arg-type]  # pos3 populated for curve-hold-curve before interpolation
        vec1=section.vec3,  # type: ignore[arg-type]  # vec3 populated for curve-hold-curve before interpolation
        vec2=section.vec_target,
        dist_curve=section.dist_curve2,
        dogleg=section.dogleg2,
        func_dogleg=section.func_dogleg2,
        step=step,
        endpoint=True
    ))

    return data


def convert_target_input_to_booleans(*inputs: Any) -> str:
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
    cartesians: ArrayLike, vec_start: ArrayLike = [0., 0., 1.],
    dls_design: Union[float, list] = 3.0, nev: bool = True,
    # step=30,
    md_start: float = 0.
) -> list:
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
        dls = np.array(dls_design).reshape(-1, 1)  # type: ignore[assignment]  # numpy row broadcast (1,3)->(3,)

    connections: list = []
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
            node: Node,
            return_data: bool = False
    ) -> Any:  # returns arc data (tuple) or the inc residual (float) per flag
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
            pos=pos3, vec=vec2, md=node2.md + tangent_length  # type: ignore[operator]  # Node.md is Optional but set on node2 above
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
            delta_md: float, pos: ArrayLike, vec: ArrayLike,
            target_tvd: float, return_data: bool = False
        ) -> Any:  # returns pos (ndarray) or the tvd residual (float) per flag
            pos2 = np.array(pos) + delta_md * np.array(vec)
            if return_data:
                return pos2
            else:
                return abs(pos2[2] - target_tvd)

        args: tuple = (node.pos_nev, node.vec_nev, target_tvd)
        bounds = [[0, None]]
        result = minimize(
            _extend_tvd,
            _delta_md,
            args=args,
            bounds=bounds,
            method="Powell"
        )
        pos2 = _extend_tvd(result.x, *args, return_data=True)  # type: ignore[misc]  # star-args expansion into scipy.optimize callback
        connections.append(Connector(
            pos1=node.pos_nev, vec1=node.vec_nev,  # type: ignore[arg-type]  # node.pos_nev Optional on Node but set here
            md1=0 if node.md is None else node.md,
            pos2=pos2,
            dls_design=dls, force_min_curve=True
        ))
    else:
        def _drop_off(
            delta_md: float, target_inc: float, target_tvd: float,
            dls: float, node: Node, return_data: bool = False
        ) -> Any:  # returns nodes (list) or the tvd residual (float) per flag
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
            [min(target_tvd - node.pos_nev[2], _delta_md), None]  # type: ignore[index]  # node.pos_nev Optional on Node but set here
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


