import numpy as np
from numpy.linalg import norm
from copy import deepcopy

try:
    import trimesh
    MESH_MODE = True
except ImportError:
    MESH_MODE = False

from scipy import optimize
from scipy.signal import argrelmin
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from .mesh import WellMesh
from .survey import Survey, _interpolate_survey, slice_survey
from .utils import NEV_to_HLA


class Clearance:
    """
    Initialize a `welleng.clearance.Clearance` object.

    Parameters
    ----------
    reference : `welleng.survey.Survey` object
        The current well from which other wells are referenced.
    offset : `welleng.survey.Survey` object
        The other well.
    k : float
        The dimensionless scaling factor that determines the probability
        of well crossing.
    sigma_pa : float
        Quantifies the 1-SD uncertainty in the projection ahead of the
        current survey station. Its value is partially correlated with
        the projection distance, determined as the current survey depth to
        the bit plus the next survey interval. The magnitude of the actual
        uncertainty also depends on the planned curvature and on the actual
        BHA performance at the wellbore attitude in the formation being
        drilled. The project-ahead uncertainty is only an approximation,
        and although it is predominantly oriented normal to the reference
        well, it is mathematically convenient to define sigma_pa as being
        the radius of a sphere.
    Sm : float
        The surface margin term increases the effective radius of the
        offset well. It accommodates small, unidentified errors and helps
        overcome one of the geometric limitations of the separation rule,
        described in the Separation-Rule Limitations section. It also
        defines the minimum acceptable slot separation during facility
        design and ensures that the separation rule will prohibit the
        activity before nominal contact between the reference and offset
        wells, even if the position uncertainty is zero.
    Rr : float
        The openhole radius of the reference borehole (in meters).
    Ro : float
        The openhole radius of the offset borehole (in meters).
    kop_depth: float
        The kick-off point (measured) depth along the well bore - the
        default value assures that the first survey station is utilized.

    References
    ----------
    Sawaryn, S. J., Wilson, H.. , Bang, J.. , Nyrnes, E.. , Sentance,
    A.. , Poedjono, B.. , Lowdon, R.. , Mitchell, I.. , Codling, J.. ,
    Clark, P. J., and W. T. Allen. "Well-Collision-Avoidance Separation
    Rule." SPE Drill & Compl 34 (2019): 01–15.
    doi: https://doi.org/10.2118/187073-PA
    """
    def __init__(
        self,
        reference: Survey,
        offset: Survey,
        k: float = 3.5,
        sigma_pa: float = 0.5,
        Sm: float = 0.3,
        Rr: float = 0.4572,
        Ro: float = 0.3048,
        kop_depth: float = -np.inf,
    ):
        self.reference = reference
        self.offset = offset
        self.k = k
        self.sigma_pa = sigma_pa
        self.Sm = Sm
        # self.N_verts = N_verts

        self._get_kop_index(kop_depth)
        self._get_ref()
        self._get_radii(Rr, Ro)

        self.ref_nevs = self._get_nevs(self.ref)
        self.offset_nevs = self._get_nevs(self.offset)

    def _get_kop_index(self, kop_depth):
        self.kop_depth = kop_depth
        self.kop_index = np.searchsorted(
            self.reference.md, self.kop_depth, side="left"
        )

    def _get_nevs(self, survey):
        # TODO:
        # - [ ] Take this from the `Survey` where it is already calculated.
        return np.array([
            survey.n,
            survey.e,
            survey.tvd
        ]).T

    def _get_radii(self, Rr, Ro):
        """
        If well bore radius data is provided, use this, otherwise use the
        defaults values.
        """
        if self.ref.radius is not None:
            self.Rr = self.ref.radius
        else:
            self.Rr = np.full(len(self.ref.md), Rr)

        if self.offset.radius is not None:
            self.Ro = self.offset.radius
        else:
            self.Ro = np.full(len(self.offset.md), Ro)

    def _get_ref(self):
        if self.kop_index == 0:
            self.ref = self.reference
        else:
            try:
                cov_nev = self.reference.cov_nev[self.kop_index:]
                cov_hla = self.reference.cov_hla[self.kop_index:]
            except IndexError:
                cov_nev, cov_hla = (None, None)
            self.ref = Survey(
                md=self.reference.md[self.kop_index:],
                inc=self.reference.inc_rad[self.kop_index:],
                azi=self.reference.azi_grid_rad[self.kop_index:],
                n=self.reference.n[self.kop_index:],
                e=self.reference.e[self.kop_index:],
                tvd=self.reference.tvd[self.kop_index:],
                vec=self.reference.vec_nev[self.kop_index:],
                radius=self.reference.radius[self.kop_index:],
                header=self.reference.header,
                cov_nev=cov_nev,
                cov_hla=cov_hla,
                error_model=self.reference.error_model,
                start_xyz=[
                    self.reference.x[self.kop_index],
                    self.reference.y[self.kop_index],
                    self.reference.z[self.kop_index],
                    ],
                start_nev=[
                    self.reference.n[self.kop_index],
                    self.reference.e[self.kop_index],
                    self.reference.tvd[self.kop_index],
                    ],
                deg=self.reference.deg,
                unit=self.reference.unit,
                nev=True
            )


class IscwsaClearance(Clearance):
    """
    Parameters:
    -----------
    clearance_args: List
        See 'welleng.clearance.Clearance` for args.
    minimize_sf: bool
        If `True` (default), then the closest points on the reference well
        are determined and added to the `ref` object as interpolated stations.
    clearance_kwargs: dict
         See 'welleng.clearance.Clearance` for kwargs.

    Attributes:
    -----------
    Ro : array of floats
        The radius of the offset well at each station of the off well.
    Rr : array
        The radius of the reference well at each station on the ref well.
    sf : array of floats
        The calculated Separation Factor to the closest point on the offset
        well for each station on the reference well.
    Sm : float
        The surface margin term increases the effective radius of the
        offset well. It accommodates small, unidentified errors and helps
        overcome one of the geometric limitations of the separation rule,
        described in the Separation-Rule Limitations section. It also
        defines the minimum acceptable slot separation during facility
        design and ensures that the separation rule will prohibit the
        activity before nominal contact between the reference and offset
        wells, even if the position uncertainty is zero.
    calc_hole: array of floats
        The calculated combined equivalent radius of the two well bores, i.e.
        the sum or their radii plus margins.
    closest:
        The closest point on the `off` well from each station on the ref well.
    distance_cc:
        The closest center to center distance for each station on the `ref`
        well to the `off` well.
    eou_boundary:
        The sum of the ellipse of uncertainty radii of the `ref` and `off`
        wells.
    eou_separation:
        The distance between the ellipses of uncertainty of the `ref` and `off`
        wells.
    hoz_bearing:
        The horizontal bearing between the closest points in radians.
    hoz_bearing_deg:
        The horizontal bearing between the closest points in degrees.
    idx: int
        The index of the closest point on the `off` well for each station on
        the `ref` well.
    masd:
        The Minimum Allowable Separation Distance from the `ref` well.
    off: Survey
        The offset well `Survey`.
    off_pcr:
        The Pedal Curve Radii for each station on the `off` well.
    off_cov_hla:
        The covariance matrix in the HLA domain for each station of the `off`
        well.
    off_cov_nev:
        The covariance matrix in the NEV domain for each station of the `off`
        well.
    off_nevs:
        The NEV coordinates of the `off` well.
    offset: Survey
        The initial `offset` well `Survey`.
    offset_nevs:
        The initial NEV coordinates of the `offset` well.
    ref: Survey
        The `ref` well `Survey`.
    ref_pcr:
        The Pedal Curve Radii for each station on the `ref` well.
    ref_cov_hla:
        The covariance matrix in the HLA domain for each station of the `ref`
        well.
    ref_cov_nev:
        The covariance matrix in the NEV domain for each station of the `ref`
        well.
    ref_nevs:
        The NEV coordinates of the `ref` well.
    reference: Survey
        The initial `reference` well `Survey`.
    reference_nevs:
        The initial NEV coordinates of the `reference` well.
    sf:
        The Separation Factor between the closest point on the `off` well for
        each station on the `ref` well.
    toolface_bearing:
        The toolface bearing in radians from each station on the `ref` well to
        the closest point on the `off` well.
    trav_cyl_azi_deg:
        The heading in degrees from each station on teh `ref` well to the
        closest point on the `off` well.
    wellbore_separation:
        The distance between the edge of the wellbore for each station on the
        `ref` well to the closest point on the `off` well.
    """
    def __init__(
        self,
        *clearance_args,
        minimize_sf=None,
        **clearance_kwargs
    ):
        # TODO:
        # - [ ] Can probably remove the `offset` Survey since `off` is a copy.
        # - [ ] Can probably remover the `_nev*` attrs and instead reference
        # the onces in the `Survey` instances.

        super().__init__(*clearance_args, **clearance_kwargs)

        minimize_sf = True if minimize_sf is None else minimize_sf

        # get closest survey station in offset well for each survey
        # station in the reference well
        self.idx = np.argmin(
            cdist(
                self.ref_nevs, self.offset_nevs
            ), axis=-1
        )

        # iterate to find closest point on offset well between
        # survey stations
        self._get_closest_points()
        self.off_nevs = self._get_nevs(self.off)

        # get the unit vectors and horizontal bearing between the wells
        self._get_delta_nev_vectors()

        # transform to HLA coordinates
        # self._get_delta_hla_vectors()

        # make the covariance matrices
        self._get_covs()

        # get the PCRs
        self._get_PCRs()

        # calculate sigmaS
        self.sigmaS = np.sqrt(self.ref_pcr ** 2 + self.off_pcr ** 2)

        # calculate combined hole radii
        self._get_calc_hole()

        # calculate Ellipse of Uncertainty Boundary
        self.eou_boundary = (
            self.k * np.sqrt(self.sigmaS ** 2 + self.sigma_pa ** 2)
        )

        # calculate distance between well bores
        self.wellbore_separation = (
            self.distance_cc.T - self.calc_hole - self.Sm
        )

        # calculate the Ellipse of Uncertainty Separation
        self.eou_separation = (
            self.wellbore_separation - self.eou_boundary
        )

        # calculate the Minimum Allowable Separation Distance
        self.masd = (
            self.eou_boundary + self.calc_hole + self.Sm
        )

        self.sf = np.array(
            self.wellbore_separation / self.eou_boundary,
        )

        # check for minima
        if minimize_sf:
            self.get_sf_mins()

        # for debugging
        # self.pc_method()

    def _get_sf_min(self, x, i, delta_md):
        if x == 0.0:
            return self.sf[i]
        if x == -delta_md[0]:
            return self.sf[i-1]
        if x == delta_md[1]:
            return self.sf[i+1]

        if x < 0:
            ii = i - 1
            xx = delta_md[0] + x
            mult = xx / delta_md[0]
        else:
            ii = i
            xx = x
            mult = xx / delta_md[1]

        node = self.ref.interpolate_md(
            self.ref.md[ii] + xx
        )

        cov_nev = (
            self.ref.cov_nev[ii]
            + (
                np.full(shape=(1, 3, 3), fill_value=mult)
                * (self.ref.cov_nev[ii+1] - self.ref.cov_nev[ii])
            )
        ).reshape(-1, 3, 3)

        sh = self.ref.header
        sh.azi_reference = 'grid'

        survey = Survey(
            md=np.insert(
                self.ref.md[ii: ii+2], 1, node.md
            ),
            inc=np.insert(
                self.ref.inc_rad[ii: ii+2], 1, node.inc_rad
            ),
            azi=np.insert(
                self.ref.azi_grid_rad[ii: ii+2], 1, node.azi_rad
            ),
            cov_nev=np.insert(
                self.ref.cov_nev[ii: ii+2], 1, cov_nev, axis=0
            ),
            start_nev=self.ref.pos_nev[ii],
            deg=False
        )

        clearance_args = (survey, self.offset)
        clearance_kwargs = dict(
            k=self.k,
            sigma_pa=self.sigma_pa,
            Sm=0.0,
            Rr=np.insert(
                self.Rr[ii: ii+2], 1, self.Rr[ii+1]
            ),
            Ro=self.Ro,
            kop_depth=self.kop_depth
        )

        sf_interpolated = IscwsaClearance(
            *clearance_args, **clearance_kwargs, minimize_sf=False
        ).sf[1]

        return sf_interpolated

    def get_sf_mins(self):
        """
        Method for assessing whether a minima has occurred between survey
        station SF values on the reference well and if so calculates the
        minimum SF value between stations (between the previous and next
        station relative to the identified station).

        Modifies the `sf` attribute to include the interpolated minimum `sf`
        values.
        """
        minimas = argrelmin(self.sf)

        sf_interpolated = []

        for minima in minimas[0].tolist():
            delta_md = self.ref.delta_md[minima: minima + 2]
            bounds = [[-delta_md[0], delta_md[1]]]
            # x0 = (np.diff(bounds) / 2)
            x0 = [0]
            args = (minima, delta_md)
            # options = {
            #     'eps': np.sum(delta_md) / 10
            # }

            # SLSQP and L-BFGS-B don't work when using neg to pos ranges in
            # this example, but Powell seems to do the job.
            result = optimize.minimize(
                self._get_sf_min,
                x0,
                method='Powell',
                bounds=bounds,
                args=args,
            )

            if any((
                result.x == 0,
                result.x in delta_md
            )):
                continue

            else:
                sf_interpolated.append((
                    self.ref.md[minima] + result.x[0], result.fun
                ))

        if bool(sf_interpolated):
            for md, sf in sf_interpolated:
                # i = np.searchsorted(self.sf[:, 0], md, side='right')
                i = np.searchsorted(self.ref.md, md, side='right')
                self.sf = np.insert(
                    self.sf, i, np.array([md, sf, 1]), axis=0
                )

                node = self.ref.interpolate_md(md)

                sh = self.ref.header
                sh.azi_reference = 'grid'

                survey = Survey(
                    md=np.insert(
                        self.ref.md, i, node.md
                    ),
                    inc=np.insert(
                        self.ref.inc_rad, i, node.inc_rad
                    ),
                    azi=np.insert(
                        self.ref.azi_grid_rad, i, node.azi_rad
                    ),
                    cov_nev=np.insert(
                        self.ref.cov_nev, i, node.cov_nev, axis=0
                    ),
                    start_nev=self.ref.start_nev,
                    start_xyz=self.ref.start_xyz,
                    deg=False,
                    interpolated=np.insert(
                        self.ref.interpolated, i, True
                    ),
                    radius=np.insert(
                        self.ref.radius, i, self.ref.radius[i]
                    ),
                    header=sh
                )

                self.ref = survey

                pass

            clearance_args = (deepcopy(self.ref), self.offset)
            clearance_kwargs = dict(
                k=self.k,
                sigma_pa=self.sigma_pa,
                Sm=self.Sm,
                Rr=self.Rr,
                Ro=self.Ro,
                kop_depth=self.kop_depth
            )

            # Recalculate all the class lists for the interpolated results
            clearance = IscwsaClearance(
                *clearance_args, **clearance_kwargs, minimize_sf=False
            )

            # Set self vars to clearance vars
            for k, v in clearance.__dict__.items():
                setattr(self, k, v)

            pass

    def get_lines(self):
        """
        Extracts the closest points between wells for each survey section.
        """
        points = [
            [
                c[0],
                [
                    c[1].n[1],
                    c[1].e[1],
                    c[1].tvd[1]
                ]
            ]
            for c in self.closest
        ]

        start_points = []
        end_points = []

        [(start_points.append(p[0]), end_points.append(p[1])) for p in points]

        return np.array([
            np.vstack(start_points).tolist(), np.vstack(end_points).tolist()
        ])

    def _get_closest_points(self):
        """
        Determines the point between pairs of subsequent survey stations on
        the offset well that is closest to each survey stations on the
        reference well.
        """
        closest = []
        for j, (i, station) in enumerate(zip(
            self.idx, self.ref_nevs.tolist()
        )):
            if i > 0:
                bnds = [(0, self.offset.md[i] - self.offset.md[i - 1])]
                res_1 = optimize.minimize(
                    self._fun,
                    bnds[0][1],
                    # method='SLSQP',
                    method='Powell',
                    bounds=bnds,
                    args=(self.offset, i-1, station)
                    )
                mult = res_1.x[0] / (bnds[0][1] - bnds[0][0])
                sigma_new_1 = self._interpolate_covs(i, mult)
            else:
                res_1 = False

            if i < len(self.offset_nevs) - 1:
                bnds = [(0, self.offset.md[i + 1] - self.offset.md[i])]
                res_2 = optimize.minimize(
                    self._fun,
                    bnds[0][0],
                    # method='SLSQP',
                    method='Powell',
                    bounds=bnds,
                    args=(self.offset, i, station)
                    )
                mult = res_2.x[0] / (bnds[0][1] - bnds[0][0])
                sigma_new_2 = self._interpolate_covs(i + 1, mult)
            else:
                res_2 = False

            if res_1 and res_2 and res_1.fun < res_2.fun or not res_2:
                closest.append((
                    station,
                    _interpolate_survey(self.offset, res_1.x[0], i - 1),
                    res_1, sigma_new_1
                ))
            else:
                closest.append((
                    station,
                    _interpolate_survey(self.offset, res_2.x[0], i),
                    res_2,
                    sigma_new_2
                ))

        self.closest = closest
        md, inc, azi, n, e, tvd, x, y, z,  = np.array([
            [
                r[1].md[1],
                r[1].inc_rad[1],
                r[1].azi_grid_rad[1],
                r[1].n[1],
                r[1].e[1],
                r[1].tvd[1],
                r[1].x[1],
                r[1].y[1],
                r[1].z[1],
            ]
            for r in self.closest
        ]).T

        cov_hla = np.array([
            [
                r[3][0],
            ]
            for r in self.closest
        ])

        cov_nev = np.array([
            [
                r[3][1],
            ]
            for r in self.closest
        ])

        self.off = Survey(
            md=md,
            inc=inc,
            azi=azi,
            n=n,
            e=e,
            tvd=tvd,
            header=self.offset.header,
            error_model=None,
            cov_hla=cov_hla,
            cov_nev=cov_nev,
            start_xyz=[x[0], y[0], z[0]],
            start_nev=[n[0], e[0], tvd[0]],
            deg=False,
            unit=self.offset.unit
        )

    def _interpolate_covs(self, i, mult):
        """
        Returns the interpolated covariance matrices for the interpolated
        survey points representing the closest points on the offset well
        relative to each reference well survey station.
        """
        cov_hla_new = (
            self.offset.cov_hla[i - 1]
            + mult * (self.offset.cov_hla[i] - self.offset.cov_hla[i-1])
        )

        cov_nev_new = (
            self.offset.cov_nev[i - 1]
            + mult * (self.offset.cov_nev[i] - self.offset.cov_nev[i-1])
        )

        return (cov_hla_new, cov_nev_new)

    def _fun(self, x, survey, index, station):
        """
        Optimization function used to find the closest point between pairs of
        offset well survey stations.
        """
        s = _interpolate_survey(survey, x[0], index)
        new_pos = np.array([s.n, s.e, s.tvd]).T[1]
        dist = norm(new_pos - station, axis=-1)

        return dist

    def _get_delta_nev_vectors(self):
        temp = self.off_nevs - self.ref_nevs

        # Center to Center distance between reference survey station and
        # closest point in the offset well
        self.distance_cc = norm(temp, axis=-1).reshape(-1, 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.ref_delta_nevs = np.nan_to_num(
                temp / self.distance_cc,
                posinf=0.0,
                neginf=0.0
            )

        temp = self.ref_nevs - self.off_nevs
        with np.errstate(divide='ignore', invalid='ignore'):
            self.off_delta_nevs = np.nan_to_num(
                temp / self.distance_cc,
                posinf=0.0,
                neginf=0.0
            )

        self.hoz_bearing = np.arctan2(
            self.ref_delta_nevs[:, 1], self.ref_delta_nevs[:, 0]
        )
        self.hoz_bearing = (
            (np.around(self.hoz_bearing, 6) + np.pi * 2) % (np.pi * 2)
        )

        self.hoz_bearing_deg = (np.degrees(self.hoz_bearing) + 360) % 360

        self._get_delta_hla_vectors()

        self.toolface_bearing = np.arctan2(
            self.ref_delta_hlas[:, 1], self.ref_delta_hlas[:, 0]
        )
        self.toolface_bearing = (
            (np.around(self.toolface_bearing, 6) + np.pi * 2) % (np.pi * 2)
        )

        self.toolface_bearing_deg = (
            np.degrees(self.toolface_bearing) + 360
        ) % 360

        self._traveling_cylinder()

        self.distance_cc = self.distance_cc.reshape(-1)

    def _get_delta_hla_vectors(self):
        self.ref_delta_hlas = np.vstack([
            NEV_to_HLA(s, nev, cov=False)
            for s, nev in zip(
                self.ref.survey_rad, self.ref_delta_nevs
            )
        ])
        self.off_delta_hlas = np.vstack([
            NEV_to_HLA(s, nev, cov=False)
            for s, nev in zip(
                self.off.survey_rad, self.off_delta_nevs
            )
        ])

    def _get_covs(self):
        self.ref_cov_hla = self.ref.cov_hla
        self.ref_cov_nev = self.ref.cov_nev
        self.off_cov_hla = self.off.cov_hla
        self.off_cov_nev = self.off.cov_nev

    def _get_PCRs(self):
        self.ref_pcr = np.hstack([
            np.sqrt(np.dot(np.dot(vec, cov), vec.T))
            for vec, cov in zip(self.ref_delta_nevs, self.ref_cov_nev)
        ])
        self.off_pcr = np.hstack([
            np.sqrt(np.dot(np.dot(vec, cov), vec.T))
            for vec, cov in zip(self.off_delta_nevs, self.off_cov_nev)
        ])

    def _get_calc_hole(self):
        self.calc_hole = self.Rr + self.Ro[self.idx]

    def _traveling_cylinder(self):
        """
        Calculates the azimuthal data for a traveling cylinder plot.
        """
        self.trav_cyl_azi_deg = (
            self.reference.azi_grid_deg[self.kop_index:]
            + self.toolface_bearing_deg + 360
        ) % 360


class MeshClearance(Clearance):
    """
    Class to calculate the clearance between two well bores using a novel
    mesh clearance method. This method is experimental and was developed
    to provide a fast method for determining if well bores are potentially
    colliding.

    This class requires that `trimesh` is installed along with
    `python-fcl`.

    Parameters
    ----------
    n_verts : int
        The number of points (vertices) used to generate the uncertainty
        ellipses which are used to generate a `trimesh` representation of
        the well bores. The default is 12 which is a good balance between
        accuracy and speed.
    sigma : float
        The required/desired sigma value representation of the generated
        mesh. The default value of 2.445 represents about 98.5% confidence
        of the well bore being located within the volume of the generated
        mesh.
    """
    def __init__(
        self,
        *clearance_args,
        n_verts: int = 12,
        sigma: float = 2.445,
        return_data: bool = True,
        return_meshes: bool = False,
        **clearance_kwargs
    ):
        super().__init__(*clearance_args, **clearance_kwargs)

        assert MESH_MODE, "ImportError: try pip install welleng[all]"
        self.n_verts = n_verts
        self.sigma = sigma
        self.Rr = self.ref.radius
        self.Ro = self.offset.radius

        # if you're only interesting in a binary "go/no-go" decision
        # then you can forfeit the expensive ISCWSA calculations by
        # setting return_data to False.
        self.return_data = return_data
        self.collision = []

        if self.return_data:
            self.distance_cc = []
            self.distance = []
            self.off_index = []
            self.sf = []
            self.nev = []
            self.hoz_bearing_deg = []
            self.ref_pcr = []
            self.off_pcr = []
            self.calc_hole = []
            self.ref_md = []
            self.off_md = []

        self.return_meshes = return_meshes
        if self.return_meshes:
            self.meshes = []

        # generate mesh for offset well
        self.off_mesh = self._get_mesh(self.offset, offset=True).mesh

        # make a CollisionManager object and add the offset well mesh
        self.cm = trimesh.collision.CollisionManager()

        self.cm.add_object("offset", self.off_mesh)

        self._process_well()

    def get_lines(self):
        """
        Extracts the closest points between wells for each survey section.
        """
        points = [
            list(d[2]._points.values())
            for d in self.distance
        ]

        start_points = []
        end_points = []

        [(start_points.append(p[0]), end_points.append(p[1])) for p in points]

        return np.array([
            np.vstack(start_points).tolist(),
            np.vstack(end_points).tolist()
        ])

    def _get_mesh(self, survey, offset=False):
        """
        Generates a mesh object from the survey object.
        """
        if offset:
            Sm = self.Sm
        else:
            Sm = 0.0

        mesh = WellMesh(
            survey=survey,
            n_verts=self.n_verts,
            sigma=self.sigma,
            Sm=Sm,
        )

        return mesh

    def _process_well(self):
        """
        Iterates through the reference well survey, determining for
        each section whether a collision has occurred with the offset
        well and optionally calculates separation data.
        """
        ref = self.ref
        off = self.offset
        off_nevs = self.offset_nevs

        for i in range(len(ref.md) - 1):
            # slice a well section and create section survey
            s = slice_survey(ref, i, i + 2)

            # generate a mesh for the section slice
            m = self._get_mesh(s).mesh

            # see if there's a collision
            collision = self.cm.in_collision_single(
                m, return_names=self.return_data, return_data=self.return_data
            )

            if self.return_data:
                distance = self.cm.min_distance_single(
                    m, return_name=True, return_data=True
                )
                closest_point_reference = distance[2].point("__external")
                name_offset_absolute = distance[1]
                closest_point_offset = distance[2].point(name_offset_absolute)

                ref_nev = self._get_closest_nev(s, closest_point_reference)
                ref_md = ref.md[i] + ref_nev[1].x[0]

                # find the closest point on the well trajectory to the closest
                # points on the mesh surface
                off_index = KDTree(off_nevs).query(closest_point_offset)[1]
                if off_index < len(off.md) - 1:
                    s = slice_survey(off, off_index)
                    off_nev_1 = self._get_closest_nev(s, closest_point_offset)
                else:
                    off_nev_1 = False

                if off_index > 0:
                    s = slice_survey(off, off_index - 1)
                    off_nev_0 = self._get_closest_nev(s, closest_point_offset)
                else:
                    off_nev_0 = False

                if off_nev_0 and off_nev_1:
                    if off_nev_0[1].fun < off_nev_1[1].fun:
                        off_nev = off_nev_0
                        off_md = off.md[off_index-1] + off_nev_0[1].x[0]
                    else:
                        off_nev = off_nev_1
                        off_md = off.md[off_index] + off_nev_1[1].x[0]
                elif off_nev_0:
                    off_nev = off_nev_0
                    off_md = off.md[off_index-1] + off_nev_0[1].x[0]
                else:
                    off_nev = off_nev_1
                    off_md = off.md[off_index] + off_nev_1[1].x[0]

                vec = off_nev[0] - ref_nev[0]
                distance_cc = norm(vec)
                hoz_bearing_deg = (
                    np.degrees(np.arctan2(vec[1], vec[0])) + 360
                ) % 360

                if collision[0] is True:
                    depth = norm(
                        closest_point_offset - closest_point_reference
                    )
                    # prevent divide by zero
                    if distance_cc != 0 and depth != 0:
                        sf = distance_cc / (distance_cc + depth)
                    else:
                        sf = 0
                else:
                    sf = distance_cc / (distance_cc - distance[0])

                # data for ISCWSA method comparison
                # self.collision.append(collision)
                self.off_index.append(off_index)
                self.distance.append(distance)
                self.distance_cc.append(distance_cc)
                self.sf.append(round(sf, 2))
                self.nev.append((ref_nev, off_nev))
                self.hoz_bearing_deg.append(hoz_bearing_deg)
                self.ref_pcr.append(
                    (ref_nev[1].fun - self.sigma_pa / 2 - self.Rr[i])
                    / self.sigma
                )
                self.off_pcr.append(
                    (
                        off_nev[1].fun - self.sigma_pa / 2
                        - self.Ro[off_index] - self.Sm
                    ) / self.sigma
                )
                self.calc_hole.append(ref.radius[i] + off.radius[off_index])
                self.ref_md.append(ref_md)
                self.off_md.append(off_md)

            if self.return_meshes:
                self.meshes.append(m)

            # else:
            self.collision.append(collision)

    def _fun(self, x, survey, pos):
        """
        Interpolates a point on a well trajectory and returns
        the distance between the interpolated point and the
        position provided.
        """
        s = _interpolate_survey(survey, x[0])
        new_pos = np.array([s.n, s.e, s.tvd]).T[1]
        dist = norm(new_pos - pos, axis=-1)

        return dist

    def _get_closest_nev(self, survey, pos):
        """
        Using an optimization function to determine the closest
        point along a well trajectory to the position provided.
        """
        bnds = [(0, survey.md[1] - survey.md[0])]
        res = optimize.minimize(
            self._fun,
            bnds[0][1] / 2,
            # method='SLSQP',
            method='Powell',
            bounds=bnds,
            args=(survey, pos)
            )

        s = _interpolate_survey(survey, res.x[0])

        nev = np.array([s.n, s.e, s.tvd]).T[-1]

        return (nev, res)


def get_ref_sigma(sigma1, sigma2, sigma3, kop_index):
    sigma = np.array([sigma1, sigma2, sigma3]).T
    sigma_diff = np.diff(sigma, axis=0)

    sigma_above = np.cumsum(sigma_diff[:kop_index][::-1], axis=0)[::-1]
    sigma_below = np.cumsum(sigma_diff[kop_index:], axis=0)

    sigma_new = np.vstack((sigma_above, np.array([0, 0, 0]), sigma_below))

    return sigma_new
