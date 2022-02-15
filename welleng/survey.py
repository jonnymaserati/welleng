import numpy as np
import math
from copy import copy
try:
    from magnetic_field_calculator import MagneticFieldCalculator
    MAG_CALC = True
except ImportError:
    MAG_CALC = False
from datetime import datetime
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from .version import __version__
from .utils import (
    MinCurve,
    get_nev,
    get_vec,
    get_angles,
    HLA_to_NEV,
    NEV_to_HLA,
    get_xyz,
    radius_from_dls
)
from .error import ErrorModel, ERROR_MODELS
from .node import Node
from .connector import Connector, interpolate_well
from .visual import figure


AZI_REF = ["true", "magnetic", "grid"]


class SurveyHeader:
    def __init__(
        self,
        name=None,
        longitude=None,
        latitude=None,
        altitude=None,
        survey_date=None,
        G=9.80665,
        b_total=None,
        earth_rate=0.26251614,
        dip=None,
        declination=None,
        convergence=0,
        azi_reference="true",
        vertical_inc_limit=0.0001,
        deg=True,
        depth_unit='meters',
        surface_unit='meters',
        mag_defaults={
            'b_total': 50_000.,
            'dip': 70.,
            'declination': 0.,
        },
        **kwargs
    ):
        """
        A class for storing header information about a well.

        Parameters
        ----------
        name: string (default: None)
            The assigned name of the well bore.
        longitude: float (default: None)
            The longitude of the surface location of the well. If left
            default (None) then it will be assigned to Grenwich, the
            undisputed center of the universe.
        latitude: float (default: None)
            The latitude of the surface location of the well. If left
            default (None) then it will be assigned to Grenwich, the
            undisputed center of the universe.
        altitude: float (default: None)
            The altitude of the surface location. If left defaults (None)
            then it will be assigned to 0.
        survey_date: YYYY-mm-dd (default: None)
            The date on which the survey data was recorded. If left
            default then the current date is assigned.
        G: float (default: 9.80665)
            The gravitational field strength in m/s^2.
        b_total: float (default: None)
            The gravitation field strength in nT. If left default, then
            the value is calculated from the longitude, latitude, altitude
            and survey_data properties using the magnetic_field_calculator.
        earth_rate: float (default: 0.26249751949994715)
            The rate of rotation of the earth in radians per hour.
        noise_reduction_factor: float (default: 1.0)
            A fiddle factor for random gyro noise.
        dip: float (default: None)
            The dip (inclination) of the magnetic field relative to the
            earth's horizontal. If left default, then the value is
            calculated using the magnetic_field_calculator. The unit (deg
            of rad) is determined by the deg property.
        declination: float (default: None)
            The angle between true north and magnetic north at the well
            location. If left default, then the value is calculated
            using the magnetic_field_calculator.
        convergence: float (default: 0)
            The angle of convergence between the projection meridian and
            the line from true north through the location of the well.
        azi_reference: string (default: 'true')
            The reference system for the azimuth angles in the survey data,
            either "true", "magnetic" or "grid". Note that survey
            calculations are performed in the "grid" reference and
            converted to and from the other systems.
        vertical_inc_limit: float (default 0.0001)
            For survey inclination angles less than the vertical_inc_limit
            (in degrees), calculations are approximated to avoid
            singularities and errors.
        deg: bool (default: True)
            Indicates whether the survey angles are measured in degrees
            (True) or radians (False).
        depth_unit: string (default: "meters")
            The unit of depth for the survey data, either "meters" or
            "feet".
        surface_unit: string (default: "feet")
            The unit of distance for the survey data, either "meters" or
            "feet".
        """
        if latitude is not None:
            assert 90 >= latitude >= -90, "latitude out of bounds"
        if longitude is not None:
            assert 180 >= longitude >= -180, "longitude out of bounds"
        assert azi_reference in AZI_REF

        self._validate_date(survey_date)
        self.name = name
        self.latitude = latitude if latitude is not None else 51.4934
        self.longitude = longitude if longitude is not None else 0.0098
        self.altitude = altitude if altitude is not None else 0.
        self._get_date(survey_date)
        self.b_total = b_total
        self.earth_rate = earth_rate
        self.dip = dip
        self.convergence = convergence
        self.declination = declination
        self.vertical_inc_limit = vertical_inc_limit

        self.depth_unit = get_unit(depth_unit)
        self.surface_unit = get_unit(surface_unit)
        self.G = G
        self.azi_reference = azi_reference

        self.mag_defaults = mag_defaults
        self._get_mag_data(deg)

    def _get_mag_data(self, deg):
        """
        Initiates b_total if provided, else calculates a value.
        """
        if MAG_CALC:
            calculator = MagneticFieldCalculator()
            try:
                result = calculator.calculate(
                    latitude=self.latitude,
                    longitude=self.longitude,
                    altitude=self.altitude,
                    date=self.survey_date
                )
            except:
                result = calculator.calculate(
                    latitude=self.latitude,
                    longitude=self.longitude,
                    altitude=self.altitude,
                    date=self._get_date(date=None)
                )
        else:
            result = {
                'field-value': {
                    'total-intensity': {
                        'value': self.mag_defaults.get('b_total')
                    },
                    'inclination': {
                        'value': self.mag_defaults.get('dip')
                    },
                    'declination': {
                        'value': self.mag_defaults.get('declination')
                    }
                }
            }

        if self.b_total is None:
            self.b_total = result['field-value']['total-intensity']['value']
            if not deg:
                self.b_total = math.radians(self.b_total)
        if self.dip is None:
            self.dip = -result['field-value']['inclination']['value']
            if not deg:
                self.dip = math.radians(self.dip)
        if self.declination is None:
            self.declination = result['field-value']['declination']['value']
            if not deg:
                self.declination = math.radians(self.declination)

        if deg:
            self.dip = math.radians(self.dip)
            self.declination = math.radians(self.declination)
            self.convergence = math.radians(self.convergence)
            self.vertical_inc_limit = math.radians(
                self.vertical_inc_limit
            )

    def _get_date(self, date):
        if date is None:
            date = datetime.today().strftime('%Y-%m-%d')
        self.survey_date = date

    def _validate_date(self, date):
        if date is None:
            return
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("incorrect data format, should be YYYY-MM-DD")


class Survey:
    def __init__(
        self,
        md,
        inc,
        azi,
        n=None,
        e=None,
        tvd=None,
        x=None,
        y=None,
        z=None,
        vec=None,
        nev=True,
        header=None,
        radius=None,
        cov_nev=None,
        cov_hla=None,
        error_model=None,
        start_xyz=[0., 0., 0.],
        start_nev=[0., 0., 0.],
        start_cov_nev=None,
        deg=True,
        unit="meters",
        **kwargs
    ):
        """
        Initialize a welleng.Survey object. Calculations are performed in the
        azi_reference "grid" domain.

        Parameters
        ----------
            md: (,n) list or array of floats
                List or array of well bore measured depths.
            inc: (,n) list or array of floats
                List or array of well bore survey inclinations
            azi: (,n) list or array of floats
                List or array of well bore survey azimuths
            n: (,n) list or array of floats (default: None)
                List or array of well bore northings
            e: (,n) list or array of floats (default: None)
                List or array of well bore eastings
            tvd: (,n) list or array of floats (default: None)
                List or array of local well bore z coordinates, i.e. depth
                and usually relative to surface or mean sea level.
            x: (,n) list or array of floats (default: None)
                List or array of local well bore x coordinates, which is
                usually aligned to the east direction.
            y: (,n) list or array of floats (default: None)
                List or array of local well bore y coordinates, which is
                usually aligned to the north direction.
            z: (,n) list or array of floats (default: None)
                List or array of well bore true vertical depths relative
                to the well surface datum (usually the drill floor
                elevation DFE, so not always identical to tvd).
            vec: (n,3) list or array of (,3) floats (default: None)
                List or array of well bore unit vectors that describe the
                inclination and azimuth of the well relative to (x,y,z)
                coordinates.
            header: SurveyHeader object (default: None)
                A SurveyHeader object with information about the well location
                and survey data. If left default then a SurveyHeader will be
                generated with the default properties assigned, but these may
                not be relevant and may result in incorrect data.
            radius: float or (,n) list or array of floats (default: None)
                If a single float is specified, this value will be
                assigned to the entire well bore. If a list or array of
                floats is provided, these are the radii of the well bore.
                If None, a well bore radius of 12" or approximately 0.3 m
                is applied.
            cov_nev: (n,3,3) list or array of floats (default: None)
                List or array of covariance matrices in the (n,e,v)
                coordinate system.
            cov_hla: (n,3,3) list or array of floats (default: None)
                List or array of covariance matrices in the (h,l,a)
                well bore coordinate system (high side, lateral, along
                hole).
            error_model: str (default: None)
                If specified, this model is used to calculate the
                covariance matrices if they are not present. Currently,
                only the "ISCWSA_MWD" model is provided.
            start_xyz: (,3) list or array of floats (default: [0,0,0])
                The start position of the well bore in (x,y,z) coordinates.
            start_nev: (,3) list or array of floats (default: [0,0,0])
                The start position of the well bore in (n,e,v) coordinates.
            start_cov_nev: (,3,3) list or array of floats (default: None)
                The covariance matrix for the start position of the well
                bore in (n,e,v) coordinates.
            deg: boolean (default: True)
                Indicates whether the provided angles are in degrees
                (True), else radians (False).
            unit: str (default: 'meters')
                Indicates whether the provided lengths and distances are
                in 'meters' or 'feet', which impacts the calculation of
                the dls (dog leg severity).

        Returns
        -------
        A welleng.survey.Survey object.
        """
        if header is None:
            self.header = SurveyHeader()
        else:
            assert isinstance(header, SurveyHeader)
            self.header = header
        assert unit == self.header.depth_unit, (
            "inconsistent units with header"
        )
        self.unit = unit
        self.deg = deg
        self.start_xyz = start_xyz
        self.start_nev = start_nev
        self.md = np.array(md).astype('float64')
        self.start_cov_nev = start_cov_nev

        self._process_azi_ref(inc, azi, deg)

        self._get_radius(radius)

        self.survey_deg = np.array(
            [self.md, self.inc_deg, self.azi_grid_deg]
        ).T
        self.survey_rad = np.array(
            [self.md, self.inc_rad, self.azi_grid_rad]
        ).T

        self.n = n
        self.e = e
        self.tvd = tvd
        self.x = x
        self.y = y
        self.z = z
        if vec is not None:
            if nev:
                self.vec_nev = vec
                self.vec_xyz = get_xyz(vec)
            else:
                self.vec_xyz = vec
                self.vec_nev = get_nev(vec)
        else:
            self.vec_nev, self.vec_xyz = vec, vec

        self._min_curve(vec)
        self._get_toolface_and_rates()

        # initialize errors
        # TODO: read this from a yaml file in errors
        error_models = ERROR_MODELS
        if error_model is not None:
            assert error_model in error_models, "Unrecognized error model"
        self.error_model = error_model

        self.cov_hla = cov_hla
        self.cov_nev = cov_nev

        self._get_errors()

        self.interpolated = kwargs.get('interpolated')

    def _process_azi_ref(self, inc, azi, deg):
        if self.header.azi_reference == 'grid':
            self._make_angles(inc, azi, deg)
            self.azi_true_deg = (
                self.azi_grid_deg + math.degrees(self.header.convergence)
            )
            self.azi_mag_deg = (
                self.azi_true_deg - math.degrees(self.header.declination)
            )
            self._get_azi_mag_and_true_rad()
        elif self.header.azi_reference == 'true':
            if deg:
                self.azi_true_deg = np.array(azi).astype('float64')
                self.azi_mag_deg = (
                    self.azi_true_deg - math.degrees(self.header.declination)
                )
                self._get_azi_mag_and_true_rad()
                azi_temp = self._get_azi_temp(deg)
            else:
                self.azi_true_rad = np.array(azi).astype('float64')
                self.azi_mag_rad = (
                    self.azi_true_rad - self.header.declination
                )
                self._get_azi_mag_and_true_deg()
                azi_temp = self._get_azi_temp(deg)
            self._make_angles(inc, azi_temp, deg)
        else:  # azi_reference is "magnetic"
            if deg:
                self.azi_mag_deg = np.array(azi).astype('float64')
                self.azi_true_deg = (
                    self.azi_mag_deg + math.degrees(self.header.declination)
                )
                self._get_azi_mag_and_true_rad()
                azi_temp = self._get_azi_temp(deg)
            else:
                self.azi_mag_rad = np.array(azi).astype('float64')
                self.azi_true_rad = (
                    self.azi_mag_rad + self.header.declination
                )
                self._get_azi_mag_and_true_deg()
                azi_temp = self._get_azi_temp(deg)
            self._make_angles(inc, azi_temp, deg)

    def _get_azi_temp(self, deg):
        if deg:
            azi_temp = self.azi_true_deg - math.degrees(
                self.header.convergence
            )
        else:
            azi_temp = self.azi_true_rad - self.header.convergence

        return azi_temp

    def _get_azi_mag_and_true_rad(self):
        self.azi_true_rad, self.azi_mag_rad = (
            np.radians(np.array([
                self.azi_true_deg, self.azi_mag_deg
            ]))
        )

    def _get_azi_mag_and_true_deg(self):
        self.azi_true_deg, self.azi_mag_deg = (
            np.degrees(np.array([
                self.azi_true_rad, self.azi_mag_rad
            ]))
        )

    def _get_radius(self, radius=None):
        if radius is None:
            self.radius = np.full_like(self.md.astype(float), 0.3048)
        elif np.array([radius]).shape[-1] == 1:
            self.radius = np.full_like(self.md.astype(float), radius)
        else:
            assert len(radius) == len(self.md), "Check radius"
            self.radius = np.array(radius)

    def _min_curve(self, vec):
        """
        Get the (x,y,z), (n,e,v), doglegs, rfs, delta_mds, dlss and
        vectors for the well bore if they were not provided, using the
        minimum curvature method.
        """
        mc = MinCurve(
            self.md, self.inc_rad, self.azi_grid_rad, self.start_xyz, self.unit
        )
        self.dogleg = mc.dogleg
        self.rf = mc.rf
        self.delta_md = mc.delta_md
        self.dls = mc.dls
        self.pos_xyz = mc.poss
        self.pos_nev = get_nev(self.pos_xyz)

        if self.x is None:
            # self.x, self.y, self.z = (mc.poss + self.start_xyz).T
            self.x, self.y, self.z = (mc.poss).T
        if self.n is None:
            self._get_nev()
        if vec is None:
            self.vec_xyz = get_vec(self.inc_rad, self.azi_grid_rad, deg=False)
            self.vec_nev = get_vec(
                self.inc_rad, self.azi_grid_rad, deg=False, nev=True
            )

    def _get_nev(self):
        self.n, self.e, self.tvd = get_nev(
            np.array([
                self.x,
                self.y,
                self.z
            ]).T,
            start_xyz=self.start_xyz,
            start_nev=self.start_nev
        ).reshape(-1, 3).T

    def _make_angles(self, inc, azi, deg=True):
        """
        Calculate angles in radians if they were provided in degrees or
        vice versa.
        """
        if deg:
            self.inc_rad = np.radians(inc)
            self.azi_grid_rad = np.radians(azi)
            self.inc_deg = np.array(inc)
            self.azi_grid_deg = np.array(azi)
        else:
            self.inc_rad = np.array(inc)
            self.azi_grid_rad = np.array(azi)
            self.inc_deg = np.degrees(inc)
            self.azi_grid_deg = np.degrees(azi)

    def get_error(self, error_model, return_error=False):
        assert error_model in ERROR_MODELS, "Undefined error model"

        self.error_model = error_model
        self._get_errors()

        if return_error:
            return self.err
        else:
            return self

    def _get_errors(self):
        """
        Initiate a welleng.error.ErrorModel object and calculate the
        covariance matrices with the specified error model.
        """
        if self.error_model:
            # if self.error_model == "iscwsa_mwd_rev4":
            self.err = ErrorModel(
                self,
                error_model=self.error_model
            )
            self.cov_hla = self.err.errors.cov_HLAs.T
            self.cov_nev = self.err.errors.cov_NEVs.T
        else:
            if self.cov_nev is not None and self.cov_hla is None:
                self.cov_hla = NEV_to_HLA(self.survey_rad, self.cov_nev.T).T
            elif self.cov_nev is None and self.cov_hla is not None:
                self.cov_nev = HLA_to_NEV(self.survey_rad, self.cov_hla.T).T
            else:
                pass

        if (
            self.start_cov_nev is not None
            and self.cov_nev is not None
        ):
            self.cov_nev += self.start_cov_nev
            self.cov_hla = NEV_to_HLA(self.survey_rad, self.cov_nev.T).T

    def _curvature_to_rate(self, curvature):
        with np.errstate(divide='ignore', invalid='ignore'):
            radius = 1 / curvature
        circumference = 2 * np.pi * radius
        if self.unit == 'meters':
            x = 30
        else:
            x = 100
        rate = np.absolute(np.degrees(2 * np.pi / circumference) * x)

        return rate

    def _get_toolface_and_rates(self):
        """
        Reference SPE-84246.
        theta is inc, phi is azi
        """
        # split the survey
        s = SplitSurvey(self)

        if self.unit == 'meters':
            x = 30
        else:
            x = 100

        # this is lazy I know, but I'm using this mostly for flags
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = np.arctan2(
                np.sin(s.inc2) * np.sin(s.delta_azi),
                (
                    np.sin(s.inc2) * np.cos(s.inc1) * np.cos(s.delta_azi)
                    - np.sin(s.inc1) * np.cos(s.inc2)
                )
            )
            t1 = np.nan_to_num(
                t1,
                # np.where(t1 < 0, t1 + 2 * np.pi, t1),
                nan=np.nan
            )
            t2 = np.arctan2(
                np.sin(s.inc1) * np.sin(s.delta_azi),
                (
                    np.sin(s.inc2) * np.cos(s.inc1)
                    - np.sin(s.inc1) * np.cos(s.inc2) * np.cos(s.delta_azi)
                )
            )
            t2 = np.nan_to_num(
                np.where(t2 < 0, t2 + 2 * np.pi, t2),
                nan=np.nan
            )
            self.curve_radius = (360 / self.dls * x) / (2 * np.pi)

            curvature_dls = 1 / self.curve_radius

            self.toolface = np.concatenate((t1, np.array([t2[-1]])))

            curvature_turn = curvature_dls * (
                np.sin(self.toolface) / np.sin(self.inc_rad)
            )
            self.turn_rate = self._curvature_to_rate(curvature_turn)

            curvature_build = curvature_dls * np.cos(self.toolface)
            self.build_rate = self._curvature_to_rate(curvature_build)

        # calculate plane normals
        n12 = np.cross(s.vec1_nev, s.vec2_nev)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.normals = n12 / np.linalg.norm(n12, axis=1).reshape(-1, 1)

    def _get_sections(self, rtol=0.1, atol=0.1, dls_cont=True):
        sections = get_sections(self, rtol, atol, dls_cont)

        return sections

    def get_nev_arr(self):
        return np.array([
            self.n,
            self.e,
            self.tvd
        ]).T.reshape(-1, 3)

    def save(self, filename):
        """
        Saves a minimal (control points) survey listing as a .csv file,
        including the survey header information.

        Parameters
        ----------
        filename: str
            The path and filename for saving the text file.
        """
        export_csv(self, filename)

    def interpolate_md(self, md):
        """
        Method to interpolate a position based on measured depth and return
        a node.

        Parameters
        ----------
        md: float
            The measured depth of the point of interest.

        Returns
        -------
        node: we.node.Node object
            A node with attributes describing the point at the provided
            measured depth.

        Examples
        --------
        >>> import welleng as we
        >>> survey = we.connector.interpolate_survey(
        ...    survey=we.survey.Survey(
        ...       md=[0, 500, 1000, 2000, 3000],
        ...       inc=[0, 0, 30, 90, 90],
        ...       azi=[0, 0, 45, 135, 180],
        ...    ),
        ...    step=30
        ... )
        >>> node = survey.interpolate_md(1234)
        >>> node.properties()
        {
            'vec_nev': [0.07584209568113438, 0.5840332282889957, 0.8081789187902809],
            'vec_xyz': [0.5840332282889957, 0.07584209568113438, 0.8081789187902809],
            'inc_rad': 0.6297429542197106,
            'azi_rad': 1.4416597719915565,
            'inc_deg': 36.081613454889634,
            'azi_deg': 82.60102042890875,
            'pos_nev': [141.27728744087796, 201.41424652428694, 1175.5823295305202],
            'pos_xyz': [201.41424652428694, 141.27728744087796, 1175.5823295305202],
            'md': 1234.0,
            'unit': 'meters',
            'interpolated': True
        }
        """
        s = interpolate_md(self, md)
        node = get_node(s, -1, s.interpolated[-1])

        return node

    def interpolate_tvd(self, tvd):
        node = interpolate_tvd(self, tvd=tvd)
        return node

    def interpolate_survey_tvd(self, start=None, stop=None, step=10):
        """
        Convenience method for interpolating a Survey object's TVD.
        """
        survey_interpolated = interpolate_survey_tvd(
            self, start=start, stop=stop, step=step
        )
        return survey_interpolated

    def interpolate_survey(self, step=30, dls=1e-8):
        """
        Convenience method for interpolating a Survey object's MD.
        """
        survey_interpolated = interpolate_survey(self, step=30, dls=1e-8)
        return survey_interpolated

    def figure(self, type='scatter3d', **kwargs):
        fig = figure(self, type, **kwargs)
        return fig

    def project_to_bit(self, delta_md, dls=None, toolface=None):
        """
        Convenience method to project the survey ahead to the bit.

        Parameters
        ----------
        delta_md: float
            The along hole distance from the surveying tool to the bit in
            meters.
        dls: float
            The desired dog leg severity (deg / 30m) between the surveying
            tool and the bit. Default is to project the DLS of the last
            survey section.
        toolface: float
            The desired toolface to project from at the last survey point.
            The default is to project the current toolface from the last
            survey station.

        Returns
        -------
        node: welleng.node.Node object
        """
        if dls is None:
            dls = self.dls[-1]
        if toolface is None:
            toolface = self.toolface[-1]

        node = project_ahead(
            pos=np.array([self.n, self.e, self.tvd]).T[-1],
            vec=self.vec_nev[-1],
            delta_md=delta_md,
            dls=dls,
            toolface=toolface,
            md=self.md[-1]
        )

        return node

    def project_to_target(
        self,
        node_target,
        dls_design=3.0,
        delta_md=None,
        dls=None, toolface=None,
        step=30
    ):
        survey = project_to_target(
            self,
            node_target,
            dls_design,
            delta_md,
            dls, toolface,
            step
        )
        return survey


class TurnPoint:
    def __init__(
        self,
        md=None,
        inc=None,
        azi=None,
        build_rate=None,
        turn_rate=None,
        dls=None,
        toolface=None,
        method=None,
        target=None,
        tie_on=False,
        location=None
    ):
        self.md = md
        self.inc = inc
        self.azi = azi
        self.build_rate = build_rate
        self.turn_rate = turn_rate
        self.dls = dls
        self.toolface = toolface
        self.method = method
        self.target = target
        self.tie_on = tie_on
        self.location = location


def get_node(survey, idx, interpolated=False):
    node = Node(
        pos=[survey.n[idx], survey.e[idx], survey.tvd[idx]],
        vec=survey.vec_nev[idx].tolist(),
        md=survey.md[idx],
        unit=survey.unit,
        nev=True,
        interpolated=interpolated
    )
    return node


def interpolate_md(survey, md):
    """
    Interpolates a survey at a given measured depth.
    """
    # get the closest survey stations
    idx = np.searchsorted(survey.md, md, side="left") - 1

    assert idx < len(survey.md), "The md is beyond the survey"

    x = md - survey.md[idx]
    assert x >= 0

    return _interpolate_survey(survey, x=x, index=idx)


def _interpolate_survey(survey, x=0, index=0):
    """
    Interpolates a point distance x between two survey stations
    using minimum curvature.

    Parameters
    ----------
        survey: welleng.Survey
            A survey object with at least two survey stations.
        x: float
            Length along well path from indexed survey station to
            perform the interpolate at. Must be less than length
            to the next survey station.
        index: int
            The index of the survey station from which to interpolate
            from.

    Returns
    -------
        survey: welleng.Survey
            A survey object consisting of the two survey stations
            between which the interpolation has been made (index 0 and
            2), with the interpolated station between them (index 1)

    """
    index = int(index)

    assert index < len(survey.md) - 1, "Index is out of range"

    # check if it's just a tangent section
    if survey.dogleg[index + 1] == 0:
        azi = survey.azi_grid_rad[index]
        inc = survey.inc_rad[index]

    else:
        # get the vector
        t1 = survey.vec_xyz[index]
        t2 = survey.vec_xyz[index + 1]

        total_dogleg = survey.dogleg[index + 1]

        dogleg = x * (total_dogleg / survey.delta_md[index + 1])

        t = (
            (math.sin(total_dogleg - dogleg) / math.sin(total_dogleg)) * t1
            + (math.sin(dogleg) / math.sin(total_dogleg)) * t2
        )

        t /= np.linalg.norm(t)

        inc, azi = get_angles(t)[0]

    sh = survey.header
    sh.azi_reference = 'grid'

    s = Survey(
        md=np.array([survey.md[index], survey.md[index] + x]),
        inc=np.array([survey.inc_rad[index], inc]),
        azi=np.array([survey.azi_grid_rad[index], azi]),
        start_xyz=np.array([survey.x, survey.y, survey.z]).T[index],
        start_nev=np.array([survey.n, survey.e, survey.tvd]).T[index],
        header=sh,
        deg=False,
    )
    interpolated = False if any((
        x == 0,
        x == survey.md[index + 1] - survey.md[index]
     )) else True
    s.interpolated = [False, interpolated]

    return s


def interpolate_tvd(survey, tvd, **kwargs):
    # only seem to work with relative small delta_md - re-write with minimize
    # function?

    def tidy_up_angle(d):
        """
        Helper function to handle large angles.
        """
        if abs(d) > np.pi:
            d %= (2 * np.pi)
        return d

    coeff = 1
    # find closest point assuming tvd is sorted list
    idx = np.searchsorted(survey.tvd, tvd, side="right") - 1
    if idx == len(survey.tvd) - 1:
        idx = len(survey.tvd) - 2
    elif idx == -1:
        idx = len(survey.tvd) - 2
        coeff = -1
    pos1, pos2 = np.array([survey.n, survey.e, survey.tvd]).T[idx: idx + 2]
    vec1, vec2 = survey.vec_nev[idx: idx + 2]
    dogleg = survey.dogleg[idx + 1]
    delta_md = survey.delta_md[idx + 1]

    node_origin = kwargs.get('node_origin')
    if node_origin:
        pos1, vec1 = node_origin.pos_nev, node_origin.vec_nev
        delta_md = survey.md[idx + 1] - node_origin.md
        # TODO: need to recalculate the dogleg
        s_temp = Survey(
            md=[node_origin.md, survey.md[idx + 1]],
            inc=[node_origin.inc_rad, survey.inc_rad[idx + 1]],
            azi=[node_origin.azi_rad, survey.azi_grid_rad[idx + 1]],
            deg=False
        )
        dogleg = s_temp.dogleg[-1]

    if np.isnan(dogleg):
        return _interpolate_survey(survey, x=0, index=idx)

    if dogleg == 0:
        x = (
            (
                tvd - survey.tvd[idx]
            )
            / (survey.tvd[idx + 1] - survey.tvd[idx])
        ) * (survey.md[idx + 1] - survey.md[idx])
    else:
        m = np.array([0., 0., coeff])
        a = np.dot(m, vec1) * np.sin(dogleg)
        b = np.dot(m, vec1) * np.cos(dogleg) - np.dot(m, vec2)
        # p = get_unit_vec(np.array([0., 0., tvd]) - pos1)
        p = np.array([0., 0., tvd]) - pos1
        c = (
            np.dot(m, p)
            * dogleg
            * np.sin(dogleg)
            / delta_md
        ) + b

        d1 = 2 * np.arctan2(
            (
                a + (a ** 2 + b ** 2 - c ** 2) ** 0.5
            ),
            (
                b + c
            )
        )
        d1 = tidy_up_angle(d1)

        d2 = 2 * np.arctan2(
            (
                a - (a ** 2 + b ** 2 - c ** 2) ** 0.5
            ),
            (
                b + c
            )
        )
        d2 = tidy_up_angle(d2)

        assert d1 >= 0 or d2 >= 0
        if d1 < 0:
            d = d2
        elif d2 < 0:
            d = d1
        else:
            d = min(d1, d2)

        x = d / dogleg * delta_md

        if node_origin:
            x -= survey.md[idx] - node_origin.md

        assert x <= delta_md

    interpolated_survey = _interpolate_survey(survey, x=x, index=idx)

    interpolated = True if x > 0 else False
    node = get_node(interpolated_survey, 1, interpolated=interpolated)

    return node


def slice_survey(survey, start, stop=None):
    """
    Take a slice from a welleng.survey.Survey object.

    Parameters
    ----------
        survey: welleng.survey.Survey object
        start: int
            The start index of the desired slice.
        stop: int (default: None)
            The stop index of the desired slice, else the remainder of
            the well bore TD is the default.

    Returns
    -------
        s: welleng.survey.Survey object
            A survey object of the desired slice is returned.
    """
    if stop is None:
        stop = start + 2
    md, inc, azi = survey.survey_rad[start:stop].T
    nevs = np.array([survey.n, survey.e, survey.tvd]).T[start:stop]
    n, e, tvd = nevs.T
    # vec = survey.vec[start:stop]

    # Handle `None` values:
    cov_hla = None if not bool(survey.cov_hla) else survey.cov_hla[start:stop]
    cov_nev = None if not bool(survey.cov_nev) else survey.cov_nev[start:stop]

    s = Survey(
        md=md,
        inc=inc,
        azi=azi,
        n=n,
        e=e,
        tvd=tvd,
        header=survey.header,
        radius=survey.radius[start:stop],
        cov_hla=cov_hla,
        cov_nev=cov_hla,
        start_nev=[n[0], e[0], tvd[0]],
        deg=False,
        unit=survey.unit,
    )

    return s


def make_cov(a, b, c, diag=False):
    """
    Make a covariance matrix from the 1-sigma errors.

    Parameters
    ----------
        a: (,n) list or array of floats
            Errors in H or N/y axis.
        b: (,n) list or array of floats
            Errors in L or E/x axis.
        c: (,n) list or array of floats
            Errors in A or V/TVD axis.
        diag: boolean (default=False)
            If true, only the lead diagnoal is calculated
            with zeros filling the remainder of the matrix.

    Returns
    -------
        cov: (n,3,3) np.array
    """

    if diag:
        z = np.zeros_like(np.array([a]).reshape(-1))
        cov = np.array([
            [a * a, z, z],
            [z, b * b, z],
            [z, z, c * c]
        ]).T
    else:
        cov = np.array([
            [a * a, a * b, a * c],
            [a * b, b * b, b * c],
            [a * c, b * c, c * c]
        ]).T

    return cov


def make_long_cov(arr):
    """
    Make a covariance matrix from the half covariance 1sigma data.
    """
    aa, ab, ac, bb, bc, cc = np.array(arr).T
    cov = np.array([
        [aa, ab, ac],
        [ab, bb, bc],
        [ac, bc, cc]
    ]).T

    return cov


class SplitSurvey:
    def __init__(
        self,
        survey,
    ):
        self.md1, self.inc1, self.azi1 = survey.survey_rad[:-1].T
        self.md2, self.inc2, self.azi2 = survey.survey_rad[1:].T
        self.delta_azi = self.azi2 - self.azi1
        self.delta_inc = self.inc2 - self.inc1

        self.vec1_xyz = survey.vec_xyz[:-1]
        self.vec1_nev = get_nev(self.vec1_xyz)
        self.vec2_xyz = survey.vec_xyz[1:]
        self.vec2_nev = get_nev(self.vec2_xyz)
        self.dogleg = survey.dogleg[1:]

        # for i, a in enumerate(self.delta_azi):
        #     if a < -np.pi:
        #         self.delta_azi[i] = np.pi + a
        #     elif a > np.pi:
        #         self.delta_azi[i] = np.pi - a
        #     else:
        #         continue


def get_circle_radius(survey, **targets):
    # TODO: add target data to sections
    ss = SplitSurvey(survey)

    y1, x1, z1 = np.cross(ss.vec1_nev, survey.normals).T
    y2, x2, z2 = np.cross(ss.vec2_nev, survey.normals).T

    b1 = np.array([y1, x1, z1]).T
    b2 = np.array([y2, x2, z2]).T
    nev = np.array([survey.n, survey.e, survey.tvd]).T

    cc1 = (
        nev[:-1] - b1
        / np.linalg.norm(b1, axis=1).reshape(-1, 1)
        * survey.curve_radius[:-1].reshape(-1, 1)
    )
    cc2 = (
        nev[1:] - b2
        / np.linalg.norm(b2, axis=1).reshape(-1, 1)
        * survey.curve_radius[1:].reshape(-1, 1)
    )

    starts = np.vstack((cc1, cc2))
    ends = np.vstack((nev[:-1], nev[1:]))

    # n = 1

    return (starts, ends)


def get_sections(survey, rtol=1e-1, atol=1e-1, dls_cont=False, **targets):
    """
    Tries to discretize a survey file into hold or curve sections. These
    sections can then be used to generate a WellPlan object to generate a
    .wbp format file for import into Landmark COMPASS, thus converting a
    survey file to an editable well trajectory.

    Note that this is in development and only tested on output from planning
    software. In its current form it likely won't be too successful on
    "as drilled" surveys (but optimizing the tolerances may help).

    Parameters
    ----------
    survey: welleng.survey.Survey object
    rtol: float (default: 1e-1)
        The relative tolerance when comparing the normals using the
        numpy.isclose() function.
    atol: float (default: 1e-2)
        The absolute tolerance when comparing the normals using the
        numpy.isclose() function.
    dls_cont: bool
        Whether to explicitly check for dls continuity. May results in a
        larger number of control points but a trajectory that is a closer
        fit to the survey.
    **targets: list of Target objects
        Not supported yet...

    Returns:
    --------
    sections: list of welleng.exchange.wbp.TurnPoint objects
    """
    # it turns out that since the well is being split into "holds" and "turns"
    # that the method can always be "920", since even a hold can be expressed
    # as an [md, inc, azi]. This simplifies things greatly!

    METHOD = "920"  # the COMPASS method for minimum curvature

    # TODO: add target data to sections
    # ss = SplitSurvey(survey)

    # check for DLS continuity
    if not dls_cont:
        dls_cont = [True] * (len(survey.dls) - 2)
    else:
        upper = np.around(survey.dls[1:-1], decimals=2)
        lower = np.around(survey.dls[2:], decimals=2)
        dls_cont = [
            True if u == l else False
            for u, l in zip(upper, lower)
        ]

    continuous = np.all((
        np.all(
            np.isclose(
                survey.normals[:-1],
                survey.normals[1:],
                rtol=rtol, atol=atol,
                equal_nan=True
            ), axis=-1
        ),
        dls_cont
    ), axis=0)

    starts = np.concatenate((
        np.array([0]),
        np.where(continuous == False)[0] + 1,
        np.array([len(survey.md) - 1])
    ))

    actions = ["hold"]
    actions.extend([
        "hold" if d == 0.0 else "curve"
        for d in survey.dogleg[starts[:-1] + 1]
    ])

    sections = []
    tie_on = True
    # for i, (s, e, a) in enumerate(zip(starts, ends, actions)):
    for i, (s, a) in enumerate(zip(starts, actions)):
        md = survey.md[s]
        inc = survey.inc_deg[s]
        azi = survey.azi_grid_deg[s]
        x = survey.e[s]
        y = survey.n[s]
        z = -survey.tvd[s]
        location = [x, y, z]

        # target = ""
        if survey.unit == 'meters':
            denominator = 30
        else:
            denominator = 100

        if a == "hold" or tie_on or i == 0:
            dls = 0.0
            toolface = 0.0
            build_rate = 0.0
            turn_rate = 0.0
            method = METHOD
        else:
            # COMPASS appears to look back, i.e. at a design point in the
            # well plan it looks back to what the dls and toolface was
            # required to get to that point, so need to give it the data from
            # the previous start point.
            lb = starts[i - 1]
            method = METHOD
            dls = survey.dls[s]
            toolface = abs(np.degrees(survey.toolface[starts[i - 1]]))

            azi_p = sections[-1].azi
            if azi - azi_p < -180:
                coeff = 1
            elif azi - azi_p > 180:
                coeff = -1
            else:
                with np.errstate(all='ignore'):
                    coeff = (azi - azi_p) / abs(azi - azi_p)
            if np.isnan(coeff):
                coeff = 1

            toolface *= coeff

            # looks like the toolface is in range -180 to 180 in the .wbp file
            # toolface = toolface - 360 if toolface > 180 else toolface
            delta_md = md - survey.md[lb]

            # TODO: should sum this line by line to avoid issues with long
            # sections
            build_rate = abs(
                (survey.inc_deg[s] - survey.inc_deg[lb])
                / delta_md * denominator
            )

            # TODO: should sum this line by line to avoid issues with long
            # sections need to be careful with azimuth straddling north
            delta_azi_1 = survey.azi_grid_deg[s] - survey.azi_grid_deg[lb]
            if delta_azi_1 < -180:
                delta_azi_1 += 360
            if delta_azi_1 > 180:
                delta_azi_1 -= 360

            delta_azi_2 = 360 - delta_azi_1
            delta_azi = min(delta_azi_1, delta_azi_2)

            delta_azi = delta_azi_1
            turn_rate = delta_azi / delta_md * denominator

        section = TurnPoint(
            md=md,
            inc=inc,
            azi=azi,
            build_rate=build_rate,
            turn_rate=turn_rate,
            dls=dls,
            toolface=toolface,
            method=method,
            target=None,
            tie_on=tie_on,
            location=location
        )

        sections.append(section)

        # Repeat the first section so that creating .wbp works
        if tie_on:
            section.method = '2'
            sections.append(section)
            sections[-1].tie_on = False

        tie_on = False

    return sections


def get_unit(unit):
    if unit in ['m', 'meters']:
        return 'meters'
    elif unit in ['ft', 'feet']:
        return 'feet'
    else:
        return None


def make_survey_header(data):
    """
    Takes a dictionary of survey header data with the same keys as the
    SurveyHeader class properties and returns a SurveyHeader object.
    """
    sh = SurveyHeader()

    for k, v in data.items():
        setattr(sh, k, v)

    return sh


# def save(survey, filename):
#     """
#     Saves the survey header and survey to a text file.
#     """
#     export_csv(survey, filename)


def export_csv(
    survey, filename, tolerance=0.1, dls_cont=False, decimals=3, **kwargs
):
    """
    Function to export a minimalist (only the control points - i.e. the
    begining and end points of hold and/or turn sections) survey to input into
    third party trajectory planning software.

    Parameters
    ----------
    survey: welleng.survey.Survey object
    filename: str
        The path and filename for saving the text file.
    tolerance: float (default: 0.1)
        How close the the final N, E, TVD position of the minimalist survey
        should be to the original survey point (e.g. within 1 meter)
    dls_cont: bool
        Whether to explicitly check for dls continuity. May result in a
        larger number of control points but a trajectory that is a closer
        fit to the survey.
    decimals: int (default: 3)
        Number of decimal places provided in the output file listing
    """

    start_tol = 0

    res = minimize(
        func, start_tol, args=(survey, dls_cont, tolerance), method='SLSQP',
        bounds=[[0, 1.0]], options={'eps': 0.001}
    )

    data = get_data(
        res.x[0], survey, dls_cont
    )

    headers = ','.join([
        'MD',
        'INC (deg)',
        'AZI (deg)',
        'NORTHING (m)',
        'EASTING (m)',
        'TVDSS (m)',
        'DLS',
        'TOOLFACE',
        'BUILD RATE',
        'TURN RATE'
    ])

    if filename is None:
        try:
            import pandas as pd

            df = pd.DataFrame(
                data,
                columns=headers.split(',')
            )
            return df
        except ImportError:
            print("Missing pandas dependency")

    author = kwargs.get('author', 'Jonny Corcutt')
    comments = [
        f"welleng, version: {__version__}\n"
        f"author, {author}\n"
    ]
    comments.extend([
        f"{k}, {v}\n" for k, v in vars(survey.header).items()
    ])
    comments += "\n"
    comments = ''.join(comments)

    np.savetxt(
        filename,
        data,
        delimiter=',',
        fmt=f"%.{decimals}f",
        header=headers,
        comments=comments
    )


def get_data(tol, survey, dls_cont):

    rtol = atol = tol

    sections = survey._get_sections(rtol=rtol, atol=atol, dls_cont=dls_cont)

    data = [[
        tp.md,
        tp.inc,
        tp.azi,
        tp.location[1],
        tp.location[0],
        tp.location[2],
        tp.dls,
        tp.toolface,
        tp.build_rate,
        tp.turn_rate,
    ] for tp in sections]

    data = np.vstack(data[1:])

    return data


def func(x0, survey, dls_cont, tolerance):

    data = get_data(x0, survey, dls_cont)

    md, inc, azi, n, e, tvd, dls, tf, br, tr = data.T
    nev = np.array([survey.n, survey.e, survey.tvd]).T

    s = Survey(
        md=md,
        inc=inc,
        azi=azi,
        start_nev=nev[0],
        header=survey.header
    )

    s_nev = np.array([s.n, s.e, s.tvd]).T

    diff = abs(
        tolerance - np.amax(np.absolute(s_nev[-1] - nev[-1]))
    )

    return diff


def _remove_duplicates(md, inc, azi):
    arr = np.array([md, inc, azi]).T
    upper = arr[:-1]
    lower = arr[1:]

    temp = np.vstack((
        upper[0],
        lower[lower[:, 0] != upper[:, 0]]
    ))

    return temp.T


def from_connections(
    section_data, step=None, survey_header=None,
    start_nev=[0., 0., 0.],
    start_xyz=[0., 0., 0.],
    start_cov_nev=None,
    radius=10, deg=False, error_model=None,
    depth_unit='meters', surface_unit='meters'
):
    """
    Constructs a well survey from a list of sections of control points.

    Parameters
    ----------
        section_data: list of dicts with section data
        start_nev: (3) array of floats (default: [0,0,0])
            The starting position in NEV coordinates.
        radius: float (default: 10)
            The radius is passed to the `welleng.survey.Survey` object
            and represents the radius of the wellbore. It is also used
            when visualizing the results, so can be used to make the
            wellbore *thicker* in the plot.

    Results
    -------
        survey: `welleng.survey.Survey` object
    """
    if type(section_data) is not list:
        section_data = [section_data]
    section_data_interp = interpolate_well(section_data, step)
    # generate lists for survey
    md, inc, azi = np.vstack([np.array(list(zip(
            s['md'].tolist(),
            s['inc'].tolist(),
            s['azi'].tolist(),
        )))
        for s in section_data_interp
    ]).T

    # remove duplicates
    md, inc, azi = _remove_duplicates(md, inc, azi)

    if survey_header is None:
        survey_header = SurveyHeader(
            depth_unit=depth_unit,
            surface_unit=surface_unit
        )

    survey = Survey(
        md=md,
        inc=inc,
        azi=azi,
        start_nev=section_data[0].pos1 + start_nev,
        start_xyz=start_xyz,
        start_cov_nev=start_cov_nev,
        deg=deg,
        radius=radius,
        header=survey_header,
        error_model=error_model,
        unit=depth_unit
    )

    return survey


def interpolate_survey(survey, step=30, dls=1e-8):
    '''
    Interpolate a sparse survey with the desired md step.

    Parameters
    ----------
    survey: welleng.survey.Survey object
    step: float (default=30)
        The desired delta md between stations.
    dls: float (default=0.01)
        The design DLS used to calculate the minimum curvature. This will be
        the minimum DLS used to fit a curve between stations so should be set
        to a small value to ensure a continuous curve is fit without any
        tangent sections.

    Returns
    -------
    survey_interpolated: welleng.survey.Survey object
        Note that a `interpolated` property is added indicating if the survey
        stations is interpolated (True) or not (False).
    '''
    if survey.header.azi_reference == 'true':
        azi = survey.azi_true_rad
    elif survey.header.azi_reference == 'grid':
        azi = survey.azi_grid_rad
    else:
        azi = survey.azi_mag_rad

    s = np.array([survey.md, survey.inc_rad, azi]).T

    s_upper = s[:-1]
    s_lower = s[1:]
    well = []

    for i, (u, l) in enumerate(zip(s_upper, s_lower)):
        if i == 0:
            node1 = Node(
                pos=survey.start_nev,
                md=u[0],
                inc=u[1],
                azi=u[2],
                degrees=False,
                unit=survey.unit
            )
        else:
            node1 = well[-1].node_end
        node2 = Node(
            md=l[0],
            inc=l[1],
            azi=l[2],
            degrees=False,
            unit=survey.unit
        )
        c = Connector(
            node1=node1,
            node2=node2,
            dls_design=dls,
            degrees=False,
            force_min_curve=True,
            unit=survey.unit
        )
        well.append(c)

    survey_interpolated = from_connections(
        well,
        step=step,
        start_xyz=survey.start_xyz,
        survey_header=survey.header,
        error_model=None
    )

    survey_interpolated.interpolated = [
        False if md in survey.md else True
        for md in survey_interpolated.md
    ]

    i = -1
    radii = []
    cov_nev = []
    for (md, boolean) in zip(
        survey_interpolated.md,
        survey_interpolated.interpolated
    ):
        if not boolean:
            i += 1
            if survey.error_model is not None:
                # interpolate covariance error between survey stations
                j = 1 if i < len(survey.md) - 1 else 0
                delta_md = survey.md[i + j] - survey.md[i]
                delta_cov_nev = (
                    survey.cov_nev[i + j] - survey.cov_nev[i]
                )
                unit_cov_nev = (
                    delta_cov_nev / delta_md
                    if j == 1
                    else 0
                )
        radii.append(survey.radius[i])
        if survey.error_model is not None:
            cov_nev.append(
                survey.cov_nev[i]
                + (
                    (md - survey.md[i]) * unit_cov_nev
                )
            )
    survey_interpolated.radius = np.array(radii)
    if bool(cov_nev):
        survey_interpolated.cov_nev = np.array(cov_nev)
        survey_interpolated.cov_hla = NEV_to_HLA(
            survey_interpolated.survey_rad,
            survey_interpolated.cov_nev.T
        ).T

    return survey_interpolated


def get_node_tvd(survey, node1, node2, tvd, node_origin):
    node2.pos_nev, node2.pos_xyz = None, None
    c = Connector(node1=node1, node2=node2, dls_design=1e-8)
    s = from_connections(c, step=None)
    node_new = interpolate_tvd(s, tvd, node_origin=node_origin)

    return node_new


def interpolate_survey_tvd(survey, start=None, stop=None, step=10):
    """
    """
    tvds = [start] if start is not None else [survey.tvd[0]]
    nodes = []

    for i, md2 in enumerate(survey.md):
        if i == 0:
            nodes.append(get_node(survey, i))
            continue
        node_origin = nodes[-1]
        node2_master = survey.interpolate_md(md2)

        while 1:
            node1 = nodes[-1]
            node2 = copy(node2_master)
            if np.isclose(node1.md, md2):
                node1.interpolated = False
                break

            # check if heading upwards
            if node1.pos_nev[2] > tvds[-1] >= node2.pos_nev[2]:
                tvd = tvds[-1]
                node_new = get_node_tvd(survey, node1, node2, tvd, node_origin)
                node_new.interpolated = True
                nodes.append(node_new)
                tvds.append(tvd - step)
            elif node1.pos_nev[2] < (tvds[-1] + step) <= node2.pos_nev[2]:
                tvd = tvds[-1] + step
                node_new = get_node_tvd(survey, node1, node2, tvd, node_origin)
                node_new.interpolated = True
                nodes.append(node_new)
                tvds.append(tvd)
            else:
                nodes.append(node2_master)
                tvds.append(tvds[-1])
                break

    md, inc, azi, interpolated = np.array([
        [n.md, n.inc_rad, n.azi_rad, n.interpolated]
        for n in nodes
    ]).T

    s_interp = Survey(
        md=md,
        inc=inc,
        azi=azi,
        interpolated=interpolated,
        deg=False,
        header=survey.header
    )

    return s_interp


def project_ahead(pos, vec, delta_md, dls, toolface, md=0.0):
    """
    Apply a simple arc or hold from a current position and vector.

    Parameters
    ----------
    pos: (3) array of floats
        Current position in n, e, tvd coordinates.
    vec: (3) array of floats
        Current vector in n, e, tvd coordinates.
    delta_md: float
        The desired along hole projection length.
    dls: float
        The desired dogleg severity of the projection. Entering 0.0 will
        result in a hold section.
    toolface: float
        The desired toolface for the projection.
    md: float (optional)
        The current md if applicable.

    Returns
    -------
    node: welleng.node.Node object
    """
    if dls > 0:
        radius = radius_from_dls(dls)
        dogleg = np.radians(delta_md / 30 * dls)

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

        angles = [
            toolface,
            inc,
            azi
        ]

        r = R.from_euler('zyz', angles, degrees=False)

        pos_new, vec_new = r.apply(np.vstack((pos_temp, vec_temp)))
        pos_new += pos

    else:
        # if dls is 0 then it's a hold
        pos_new = pos + vec * delta_md
        vec_new = vec

    node = Node(
        pos=pos_new,
        vec=vec_new,
        md=md + delta_md,
    )

    return node


def project_to_target(
    survey,
    node_target,
    dls_design=3.0,
    delta_md=None,
    dls=None, toolface=None,
    step=30,
):
    """
    Project a wellpath from the end of a current survey to a target, taking
    account of the location of the bit relative to the surveying tool if the
    `delta_md` property is not `None`.

    Parameters
    ----------
    survey: welleng.survey.Survey obj
    node_target: welleng.node.Node obj
    dls_design: float
        The dls from which to construct the projected wellpath.
    delta_md: float
        The along hole length from the surveying sensor to the bit.
    dls: float
        The desired dogleg severity for the projection from the survey tool
        to the bit. Entering 0.0 will result in a hold section.
    toolface: float
        The desired toolface for the projection from the survey tool to the
        bit.
    step: float
        The desired survey interval for the projected wellpath to the target.

    Returns
    -------
    node: welleng.survey.Survey obj
    """
    connectors = []
    node_start = Node(
            pos=survey.pos_nev[-1], vec=survey.vec_nev[-1], md=survey.md[-1]
        )
    if dls is None:
        dls = survey.dls[-1]
    if toolface is None:
        toolface = survey.toolface[-1]
    if survey.cov_nev is not None:
        cov_nev = survey.cov_nev[-1]
    else:
        cov_nev = None

    # first project to bit if delta_md is defined
    if delta_md is not None:
        node_bit = project_ahead(
            survey.pos_nev[-1],
            survey.vec_nev[-1],
            delta_md,
            dls,
            toolface,
            survey.md[-1]
        )
        node_bit.pos_nev, node_bit.pos_xyz = None, None
        connectors.append(
            Connector(node_start, node_bit, dls_design=dls_design)
        )
        node_bit = connectors[-1].node_end
    else:
        node_bit = node_start

    connectors.append(
        Connector(
            node_bit, node_target, dls_design
        )
    )
    survey_to_target = from_connections(
        connectors,
        step=step,
        survey_header=survey.header,
        start_cov_nev=cov_nev,
        radius=survey.radius[-1], deg=False, error_model=survey.error_model,
        depth_unit=survey.header.depth_unit,
        surface_unit=survey.header.surface_unit
    )
    return survey_to_target


class SurveyData:
    def __init__(self, survey):
        """
        A class for extracting the minimal amount of data from a `Survey`
        object, with methods for combining data from a list of surveys that
        describe an entire well path.

        Parameters
        ----------
        survey : `welleng.survey.Survey`
        """
        self.header = survey.header
        self.md = survey.md
        self.inc = survey.inc_rad
        self.azi = getattr(
            survey, f"azi_{getattr(self.header, 'azi_reference')}_rad"
        )
        self.start_nev = survey.start_nev
        self.start_xyz = survey.start_xyz
        self.cov_nev = survey.cov_nev
        self.cov_hla = survey.cov_hla
        self.radius = survey.radius

    def append_survey(self, survey):
        """
        Method to extract data from a survey and append it to
        the existing survey data existing in the instance.

        Parameters
        ----------
        survey : `welleng.survey.Survey`
        """
        self.md = np.hstack((self.md, survey.md[1:]))
        self.inc = np.hstack((self.inc, survey.inc_rad[1:]))
        self.azi = np.hstack(
            (
                self.azi,
                getattr(
                    survey, f"azi_{getattr(self.header, 'azi_reference')}_rad"
                )[1:]
            )
        )
        self.cov_nev = np.hstack(
            (
                self.cov_nev.reshape(-1),
                survey.cov_nev[1:].reshape(-1)
            )
        ).reshape(-1, 3, 3)
        self.cov_hla = np.hstack(
            (
                self.cov_hla.reshape(-1),
                survey.cov_hla[1:].reshape(-1)
            )
        ).reshape(-1, 3, 3)
        self.radius = np.hstack((self.radius, survey.radius[1:]))

    def get_survey(self):
        """
        Method to create a `welleng.survey.Survey` object from the survey
        data existing in the instance.

        Returns
        -------
        survey : `welleng.survey.Survey`
        """
        survey = Survey(
            md=self.md,
            inc=self.inc,
            azi=self.azi,
            deg=False,
            start_nev=self.start_nev,
            start_xyz=self.start_xyz,
            cov_nev=self.cov_nev,
            cov_hla=self.cov_hla,
            radius= self.radius,
            header=self.header
        )
        return survey


def splice_surveys(surveys):
    """
    Join together an ordered list of surveys for a well (for example, a list
    of surveys with a different error model for each survey).

    Parameters
    ----------
    surveys : list of `welleng.survey.Survey` objects
        The first survey in the list is assumed to be the shallowest and the
        survey `header` data is taken from this well. Subsequent surveys are
        assumed to be ordered by depth, with the first `md` of the next
        survey being equal to the last `md` of the previous survey.

    Returns
    -------
    spliced_survey : `welleng.survey.Survey` object
        A single survey consisting of the input surveys placed together.

    Notes
    -----
    The returned survey will include the covariance data describing the well
    bore uncertainty, but will not include the error models since these may
    be different for each well section.
    """
    assert type(surveys) is list, "Expected a list of surveys"
    assert type(surveys[0]) is Survey, "Expected a list of surveys"

    for i, s in enumerate(surveys):
        if i == 0:
            survey = SurveyData(s)
            continue
        survey.append_survey(s)

    return survey.get_survey()
