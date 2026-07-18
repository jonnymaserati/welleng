"""Well survey management, coordinate transforms, and trajectory analysis."""

"""Survey management for directional drilling wellbores.

Provides classes and functions for creating, manipulating, and interpolating
well surveys using minimum curvature calculations, error models, and
coordinate transformations.
"""
# Typing note (issue #133): this module is fully annotated. The code-scoped
# `# type: ignore[...]` comments below suppress residual mypy findings that are
# not annotation errors — Optional attributes/params that are populated at
# runtime (post-__init__/min-curve) and numpy dtype-narrowness on scalar
# reductions. `warn_unused_ignores` is enabled, so each ignore must catch a real
# error. Lines tagged `LATENT BUG` flag genuine pre-existing issues (left as-is,
# out of typing scope) rather than masking them silently.
import numpy as np
import math
import warnings
import pandas as pd
try:
    from magnetic_field_calculator import MagneticFieldCalculator
    MAG_CALC = True
except ImportError:
    MAG_CALC = False
from datetime import datetime
from pyproj import CRS, Proj, Transformer
from pyproj.enums import TransformDirection
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
    make_long_cov,
    min_curve_step,
    radius_from_dls,
)
from .error import ErrorModel, ERROR_MODELS
from .node import Node
from .connector import Connector, interpolate_well
from .visual import figure
from .units import ureg

from typing import Any, List, Optional, Union
from numpy.typing import NDArray, ArrayLike


AZI_REF = ["true", "magnetic", "grid"]


class SurveyParameters(Proj):
    """Class for calculating survey parameters for input to a Survey Header.

    This is a wrapper of pyproj that tries to simplify the process of getting
    convergence, declination and dip values for a survey header.

    Notes
    -----
    Requires ``pyproj`` and ``magnetic_field_calculator`` to be installed and
    access to the internet.

    For reference, here's some EPSG codes:
    {
        'UTM31_ED50': 'EPSG:23031',
        'UTM31_WGS84': 'EPSG:32631',
        'RD': 'EPSG:28992',
        'ED50-UTM31': 'EPSG:23031',
        'ED50-NEDTM': 'EPSG:23095',  # assume same as ED50-UTM31
        'ETRS89-UTM31': 'EPSG:25831',
        'ED50-UTM32': 'EPSG:23032',
        'ED50-GEOGR': 'EPSG:4230',
        'WGS84-UTM31': 'EPSG:32631'
    }

    References
    ----------
    For more info on transformations between maps, refer to the pyproj project
    [here](https://pypi.org/project/pyproj/).
    """
    def __init__(self, projection: str = "EPSG:23031") -> None:
        """Initiates a SurveyParameters object for conversion of map
        coordinates to WGS84 lat/lon for calculating magnetic field properties.

        Parameters
        ----------
        projection: str (default: "EPSG:23031")
            The EPSG code of the map of interest. The default represents
            ED50/UTM zone 31N.

        References
        ----------
        For codes refer to [EPSG](https://epsg.io).
        """
        self.crs = CRS(projection)
        super().__init__(self.crs)

    def get_factors_from_x_y(
        self, x: float, y: float, altitude: Optional[float] = None,
        date: Optional[str] = None
    ) -> dict:
        """Calculates the survey header parameters for a given map coordinate.

        Parameters
        ----------
        x: float
            The x or East/West coordinate.
        y: float
            The y or North/South coordinate.
        altitude: float (default: None)
            The altitude or z value coordinate. If none is provided this will
            default to zero (sea level).
        date: str (default: None)
            The date of the survey, used when calculating the magnetic
            parameters. Will default to the current date.

        Returns
        -------
        dict:
            x: float
                The x coordinate.
            y: float
                The y coordinate.
            northing: float
                The Northing (negative values are South).
            easting: float
                The Easting (negative values are West).
            latitude: float
                The WGS84 latitude.
            longitude: float
                The WGS84 longitude.
            convergence: float
                Te grid convergence for the provided coordinates.
            scale_factor: float
                The scale factor for the provided coordinates.
            magnetic_field_intensity: float
                The total field intensity for the provided coordinates and
                time.
            declination: float
                The declination at the provided coordinates and time.
            dip: float
                The dip angle at the provided coordinates and time.
            date:
                The date used for determining the magnetic parameters.

        Examples
        --------
        In the following example, the parameters for Den Haag in The
        Netherlands are looked up with the reference map ED50 UTM Zone 31N.

        >>> import pprint
        >>> from welleng.survey import SurveyParameters
        >>> calculator = SurveyParameters('EPSG:23031')
        >>> survey_parameters = calculator.get_factors_from_x_y(
        ...     x=588319.02, y=5770571.03
        ... )
        >>> pprint(survey_parameters)
        {'convergence': 1.01664403471959,
        'date': '2023-12-16',
        'declination': 2.213,
        'dip': -67.199,
        'easting': 588319.02,
        'latitude': 52.077583926214494,
        'longitude': 4.288694821453205,
        'magnetic_field_intensity': 49381,
        'northing': 5770571.03,
        'scale_factor': 0.9996957469340414,
        'srs': 'EPSG:23031',
        'x': 588319.02,
        'y': 5770571.03}
        """
        longitude, latitude = self(x, y, inverse=True)
        result = self.get_factors(longitude, latitude)

        date = (
            datetime.today().strftime('%Y-%m-%d') if date is None
            else date
        )
        if MAG_CALC:
            magnetic_calculator = MagneticFieldCalculator()
            try:
                result_magnetic = magnetic_calculator.calculate(
                    latitude=latitude, longitude=longitude,
                    altitude=0 if altitude is None else altitude,
                    date=date
                )
            except Exception as exc:
                warnings.warn(
                    f"Magnetic-field lookup failed ({exc}); declination, dip and "
                    "field intensity set to None (no connection to the BGS service?)."
                )
                result_magnetic = None
        else:
            warnings.warn(
                "Magnetic-field parameters (declination, dip, field intensity) need "
                "the optional 'magnetic_field_calculator' package -- install with "
                "`pip install welleng[all]` (or `pip install magnetic_field_calculator`). "
                "Returning None for those fields."
            )
            result_magnetic = None

        data = dict(
            x=x,
            y=y,
            northing=y,
            easting=x,
            latitude=latitude,
            longitude=longitude,
            convergence=result.meridian_convergence,
            scale_factor=result.meridional_scale,
            magnetic_field_intensity=(
                None if result_magnetic is None
                else result_magnetic.get('field-value').get('total-intensity').get('value')
            ),
            declination=(
                None if result_magnetic is None
                else result_magnetic.get('field-value').get('declination').get('value')
            ),
            dip=(
                None if result_magnetic is None
                else result_magnetic.get('field-value').get('inclination').get('value')
                * (
                    -1 if "down" in result_magnetic.get('field-value').get('inclination').get('units')
                    else 1
                )
            ),
            date=date,
            srs=self.crs.srs
        )

        return data

    def transform_coordinates(
        self, coords: ArrayLike, to_projection: str,
        altitude: Optional[float] = None,
        **kwargs: Any
    ) -> ArrayLike:
        """Transforms coordinates from instance's projection to another
        projection.

        Parameters
        ----------
        coords: arraylike
            A list of decimal coordinates to transform from the instance
            projection to the specified projection system. Can be 2D or 3D in
            (x, y, z) format, where x is East/West and y is North/South.
        to_projection: str
            The EPSG code of the desired coordinates.

        Returns
        -------
        result: ArrayLike
            An array of transformed coordinates in the desired projection.

        Examples
        --------
        Convert the coordinates of Den Haag from ED50-UTM31 to WGS84-UTM31:

        >>> from welleng.survey import SurveyParameters
        >>> calculator = SurveyParameters('EPSG:23031')
        >>> result = calculator.transform_coordinates(
        ...     coords=[(588319.02, 5770571.03)], to_projection='EPSG:32631'
        ... )
        >>> print(result)
        [[ 588225.93417027 5770360.56500115]]
        """
        transformer = Transformer.from_crs(
            self.crs, CRS(to_projection)
        )
        _coords = np.array(coords)
        result = list(transformer.itransform(
            (
                _coords.tolist() if len(_coords.shape) > 1
                else _coords.reshape((1, -1)).tolist()
            ),
            direction=TransformDirection('FORWARD'), **kwargs
        ))

        return np.array(result)


class SurveyHeader:
    """Metadata for a well survey including location, magnetic field, and reference systems.

    Stores the geographic position, magnetic field parameters (total field,
    dip, declination), convergence, azimuth reference system, and unit
    conventions needed to interpret and process directional survey data.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        longitude: Optional[float] = None,
        latitude: Optional[float] = None,
        altitude: Optional[float] = None,
        survey_date: Optional[str] = None,
        G: float = 9.80665,
        b_total: Optional[float] = None,
        earth_rate: float = 0.26251614,
        dip: Optional[float] = None,
        declination: Optional[float] = None,
        convergence: float = 0,
        azi_reference: str = "true",
        vertical_inc_limit: float = 0.0001,
        deg: bool = True,
        depth_unit: str = 'meters',
        surface_unit: str = 'meters',
        mag_defaults: dict = {
            'b_total': 50_000.,
            'dip': 70.,
            'declination': 0.,
        },
        vertical_section_azimuth: float = 0,
        grid_scale_factor: float = 1.0
        # **kwargs
    ) -> None:
        """A class for storing header information about a well.

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
        vertical_section_azimuth: float (default: 0.0)
            The azimuth along which to determine the vertical section data
            for the well trajectory.
        grid_scale_factor: float (default: 1.0)
            Scale factor applied during when determining the grid coordinates
            from the provided survey data.
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
        self.grid_scale_factor = grid_scale_factor

        self.depth_unit = get_unit(depth_unit)
        self.surface_unit = get_unit(surface_unit)
        self.G = G
        self.azi_reference = azi_reference
        self.vertical_section_azimuth = vertical_section_azimuth

        self.mag_defaults = mag_defaults
        self._get_mag_data(deg)

    def _get_mag_data(self, deg: bool) -> None:
        """
        Initiates b_total if provided, else calculates a value.
        """
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

        if MAG_CALC:
            calculator = MagneticFieldCalculator()
            try:
                result = calculator.calculate(
                    latitude=self.latitude,
                    longitude=self.longitude,
                    altitude=self.altitude,
                    date=self.survey_date
                )
            except Exception:
                try:
                    # retry with today's date (sets self.survey_date, then use it)
                    self._get_date(date=None)
                    result = calculator.calculate(
                        latitude=self.latitude,
                        longitude=self.longitude,
                        altitude=self.altitude,
                        date=self.survey_date
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Magnetic-field lookup failed ({exc}); using the header's "
                        "default magnetic parameters (no connection to the BGS "
                        "service?)."
                    )

        if self.b_total is None:
            self.b_total = result['field-value']['total-intensity']['value']
            # if not deg:
            #     self.b_total = math.radians(self.b_total)
        if self.dip is None:
            self.dip = -result['field-value']['inclination']['value']  # type: ignore[operator]
            if not deg:
                self.dip = math.radians(self.dip)
        if self.declination is None:
            self.declination = result['field-value']['declination']['value']
            if not deg:
                self.declination = math.radians(self.declination)  # type: ignore[arg-type]

        if deg:
            self.dip = math.radians(self.dip)
            self.declination = math.radians(self.declination)  # type: ignore[arg-type]
            self.convergence = math.radians(self.convergence)
            self.vertical_inc_limit = math.radians(
                self.vertical_inc_limit
            )
            self.vertical_section_azimuth = math.radians(
                self.vertical_section_azimuth
            )

    def _get_date(self, date: Optional[str]) -> None:
        if date is None:
            date = datetime.today().strftime('%Y-%m-%d')
        self.survey_date = date

    def _validate_date(self, date: Optional[str]) -> None:
        if date is None:
            return
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("incorrect data format, should be YYYY-MM-DD")


class Survey:
    """Directional well survey with positions, vectors, errors, and trajectory properties.

    Computes wellbore positions via minimum curvature, converts between azimuth
    reference systems (true/magnetic/grid), calculates dogleg severity, toolface,
    build/turn rates, and optionally propagates ISCWSA error model covariances.

    Attributes
    ----------
    header : SurveyHeader
        Survey metadata including location, datum, and reference information.
    md : ndarray of shape (n,)
        Measured depths along the wellbore.
    inc_deg : ndarray of shape (n,)
        Inclination angles in degrees.
    inc_rad : ndarray of shape (n,)
        Inclination angles in radians.
    azi_grid_deg : ndarray of shape (n,)
        Grid azimuth angles in degrees.
    azi_grid_rad : ndarray of shape (n,)
        Grid azimuth angles in radians.
    azi_true_deg : ndarray of shape (n,)
        True north azimuth angles in degrees.
    azi_true_rad : ndarray of shape (n,)
        True north azimuth angles in radians.
    azi_mag_deg : ndarray of shape (n,)
        Magnetic north azimuth angles in degrees.
    azi_mag_rad : ndarray of shape (n,)
        Magnetic north azimuth angles in radians.
    pos_nev : ndarray of shape (n, 3)
        Station positions in North-East-Vertical coordinates.
    pos_xyz : ndarray of shape (n, 3)
        Station positions in X-Y-Z coordinates.
    vec_nev : ndarray of shape (n, 3)
        Unit direction vectors in North-East-Vertical coordinates.
    vec_xyz : ndarray of shape (n, 3)
        Unit direction vectors in X-Y-Z coordinates.
    n : ndarray of shape (n,)
        Northing coordinates of each survey station.
    e : ndarray of shape (n,)
        Easting coordinates of each survey station.
    tvd : ndarray of shape (n,)
        True vertical depth of each survey station.
    x : ndarray of shape (n,)
        X coordinates of each survey station.
    y : ndarray of shape (n,)
        Y coordinates of each survey station.
    z : ndarray of shape (n,)
        Z coordinates (depth) of each survey station.
    dogleg : ndarray of shape (n,)
        Dogleg angles between successive stations in radians.
    dls : ndarray of shape (n,)
        Dogleg severity per 30 m (or 100 ft) interval.
    delta_md : ndarray of shape (n,)
        Measured depth intervals between successive stations.
    rf : ndarray of shape (n,)
        Ratio factors from minimum curvature calculation.
    toolface : ndarray of shape (n,)
        Toolface angles in radians at each station.
    build_rate : ndarray of shape (n,)
        Build rate (inclination change rate) per unit length.
    turn_rate : ndarray of shape (n,)
        Turn rate (azimuth change rate) per unit length.
    curve_radius : ndarray of shape (n,)
        Radius of curvature at each station.
    radius : ndarray of shape (n,)
        Wellbore radius at each station.
    cov_nev : ndarray of shape (n, 3, 3) or None
        Covariance matrices in North-East-Vertical coordinates.
    cov_hla : ndarray of shape (n, 3, 3) or None
        Covariance matrices in High-Lateral-Along-hole coordinates.
    err : ErrorModel or None
        Error model results when an error model is applied.
    survey_deg : ndarray of shape (n, 3)
        Survey data as [md, inc_deg, azi_grid_deg] columns.
    survey_rad : ndarray of shape (n, 3)
        Survey data as [md, inc_rad, azi_grid_rad] columns.
    vertical_section : ndarray of shape (n,) or None
        Vertical section lateral displacement if a VS azimuth is defined.

    Methods
    -------
    interpolate_survey(step=30)
        Interpolate survey at regular MD intervals.
    interpolate_md(md)
        Interpolate survey data at a specific measured depth.
    interpolate_tvd(tvd)
        Interpolate survey data at a specific true vertical depth.
    interpolate_survey_tvd(step=30)
        Interpolate survey at regular TVD intervals.
    get_error(error_model)
        Apply an ISCWSA/OWSG error model to the survey.
    get_nev_arr()
        Return station positions as (n, 3) NEV array.
    get_vertical_section(azimuth)
        Compute vertical section along a given azimuth.
    set_vertical_section(azimuth)
        Set the vertical section azimuth on the survey.
    project_to_bit(delta_md)
        Project the survey ahead by a given MD.
    project_to_target(target, dls_design)
        Plan a trajectory to a target location.
    figure()
        Create a plotly 3D figure of the survey.
    save(filename)
        Export survey data to file.
    maximum_curvature(dls_noise=1.0)
        Compute survey using the maximum curvature method.
    tortuosity_index()
        Calculate the tortuosity index.
    modified_tortuosity_index()
        Calculate the modified tortuosity index.
    directional_difficulty_index()
        Calculate the directional difficulty index.
    """

    header: SurveyHeader
    md: np.ndarray
    inc_deg: np.ndarray
    inc_rad: np.ndarray
    azi_grid_deg: np.ndarray
    azi_grid_rad: np.ndarray
    azi_true_deg: np.ndarray
    azi_true_rad: np.ndarray
    azi_mag_deg: np.ndarray
    azi_mag_rad: np.ndarray
    pos_nev: np.ndarray
    pos_xyz: np.ndarray
    vec_nev: np.ndarray
    vec_xyz: np.ndarray
    vec_radius_nev: np.ndarray
    n: np.ndarray
    e: np.ndarray
    tvd: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dogleg: np.ndarray
    dls: np.ndarray
    delta_md: np.ndarray
    rf: np.ndarray
    toolface: np.ndarray
    build_rate: np.ndarray
    turn_rate: np.ndarray
    curve_radius: np.ndarray
    radius: np.ndarray
    normals: np.ndarray
    survey_deg: np.ndarray
    survey_rad: np.ndarray
    cov_nev: Optional[np.ndarray]
    cov_hla: Optional[np.ndarray]
    err: Optional[ErrorModel]
    error_model: Optional[str]
    steering: Optional[np.ndarray]
    unit: str
    deg: bool

    def __init__(
        self,
        md: ArrayLike,
        inc: ArrayLike,
        azi: ArrayLike,
        n: Optional[ArrayLike] = None,
        e: Optional[ArrayLike] = None,
        tvd: Optional[ArrayLike] = None,
        x: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
        vec: Optional[ArrayLike] = None,
        nev: bool = True,
        header: Optional[SurveyHeader] = None,
        radius: Optional[ArrayLike] = None,
        cov_nev: Optional[np.ndarray] = None,
        cov_hla: Optional[np.ndarray] = None,
        error_model: Optional[str] = None,
        start_xyz: ArrayLike = [0., 0., 0.],
        start_nev: ArrayLike = [0., 0., 0.],
        start_cov_nev: Optional[ArrayLike] = None,
        deg: bool = True,
        unit: str = "meters",
        steering: Optional[Union[str, ArrayLike]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize a `welleng.Survey` object. Calculations are performed in the
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
            Name of the survey-tool error model used to compute the position
            covariance. Leave as None for no uncertainty calculation. The
            recommended/standard model is ``"ISCWSA MWD Rev5.11"`` (the validated
            ISCWSA standard); ``"ISCWSA MWD Rev4"`` is the legacy model. The OWSG
            toolcode library (``"MWD+SRGM"``, ``+SAG``, ``+AX``, ``+IFR``, gyro
            stacks ``"GYRO-NS"`` / ``"GYRO-NS-CT"`` / ``"GYRO-MWD"``, ...) is also
            selectable. List every available name with
            ``welleng.error.get_error_models()``; switch by passing a different
            name. Raises if the name is unrecognised.
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

        self.azi_ref_lookup = {
            'true': "true", 'magnetic': "mag", 'grid': "grid"
        }

        self.unit = unit
        self.deg = deg
        self.start_xyz = start_xyz
        self.start_nev = start_nev
        self.md = np.array(md).astype('float64')
        self.start_cov_nev = start_cov_nev

        # Per-station steering mode ('slide' | 'rotary'), one entry per station
        # (the mode of the leg arriving at that station; station 0 is ignored).
        # Governs whether the maximum-curvature deflection is treated as a
        # directional (slide, bent-motor) bias or is randomised (rotary/RSS,
        # where the toolface averages out). None -> unspecified.
        self.steering = None if steering is None else np.asarray(steering)

        self._process_azi_ref(inc, azi, deg)

        self._get_radius(radius)

        self.survey_deg = np.column_stack([self.md, self.inc_deg, self.azi_grid_deg])
        self.survey_rad = np.column_stack([self.md, self.inc_rad, self.azi_grid_rad])

        # n/e/tvd/x/y/z/vec_* may be None here, but are always populated as
        # ndarrays by _min_curve()/_get_nev() below; the class-level
        # annotations describe the finished object, so ignore the transient
        # None assignments during construction.
        self.n = np.array(n) if n is not None else n  # type: ignore[assignment]
        self.e = np.array(e) if e is not None else e  # type: ignore[assignment]
        self.tvd = np.array(tvd) if tvd is not None else tvd  # type: ignore[assignment]

        # start_nev will be overwritten if n, e, tvd data provided
        if not all((self.n is None, self.e is None, self.tvd is None)):
            self.start_nev = np.array(
                [self.n[0], self.e[0], self.tvd[0]]
            )
        else:
            self.start_nev = np.array(start_nev)

        self.x = np.array(x) if x is not None else x  # type: ignore[assignment]
        self.y = np.array(y) if y is not None else y  # type: ignore[assignment]
        self.z = np.array(z) if z is not None else z  # type: ignore[assignment]
        if vec is not None:
            if nev:
                self.vec_nev = vec  # type: ignore[assignment]
                self.vec_xyz = get_xyz(vec)
            else:
                self.vec_xyz = vec  # type: ignore[assignment]
                self.vec_nev = get_nev(vec)
        else:
            self.vec_nev, self.vec_xyz = vec, vec  # type: ignore[assignment]

        self._min_curve(vec)
        self._get_toolface_and_rates()

        # initialize errors (ERROR_MODELS is derived from errors/tool_index.yaml)
        error_models = ERROR_MODELS
        if error_model is not None:
            assert error_model in error_models, "Unrecognized error model"
        self.error_model = error_model

        self.cov_hla = cov_hla
        self.cov_nev = cov_nev
        self.cov_nev_random = None
        self.cov_nev_systematic = None
        self.cov_nev_global = None
        self.cov_nev_within_pad = None

        self._get_errors()

        self.interpolated = (
            np.full_like(self.md, False)
            if kwargs.get('interpolated') is None
            else kwargs.get('interpolated')
        )

        self._get_vertical_section()

    def _process_azi_ref(
        self, inc: ArrayLike, azi: ArrayLike, deg: bool
    ) -> None:
        if self.header.azi_reference == 'grid':
            self._make_angles(inc, azi, deg)
            self.azi_true_deg = (
                self.azi_grid_deg + math.degrees(self.header.convergence)
            )
            self.azi_mag_deg = (
                self.azi_true_deg - math.degrees(self.header.declination)  # type: ignore[arg-type]
            )
            self._get_azi_mag_and_true_rad()
        elif self.header.azi_reference == 'true':
            if deg:
                self.azi_true_deg = np.array(azi).astype('float64')
                self.azi_mag_deg = (
                    self.azi_true_deg - math.degrees(self.header.declination)  # type: ignore[arg-type]
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
                    self.azi_mag_deg + math.degrees(self.header.declination)  # type: ignore[arg-type]
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

    def _get_azi_temp(self, deg: bool) -> np.ndarray:
        if deg:
            azi_temp = self.azi_true_deg - math.degrees(
                self.header.convergence
            )
        else:
            azi_temp = self.azi_true_rad - self.header.convergence

        return azi_temp

    def _get_azi_mag_and_true_rad(self) -> None:
        self.azi_true_rad, self.azi_mag_rad = (
            np.radians(np.array([
                self.azi_true_deg, self.azi_mag_deg
            ]))
        )

    def _get_azi_mag_and_true_deg(self) -> None:
        self.azi_true_deg, self.azi_mag_deg = (
            np.degrees(np.array([
                self.azi_true_rad, self.azi_mag_rad
            ]))
        )

    def _get_radius(self, radius: Optional[ArrayLike] = None) -> None:
        if radius is None:
            self.radius = np.full_like(self.md.astype(float), 0.3048)
        elif np.array([radius]).shape[-1] == 1:
            self.radius = np.full_like(self.md.astype(float), radius)
        else:
            assert len(radius) == len(self.md), "Check radius"  # type: ignore[arg-type]
            self.radius = np.array(radius)

    def _min_curve(self, vec: Optional[ArrayLike]) -> None:
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
        self.pos_nev = (
            get_nev(self.pos_xyz)
            * np.full_like(
                self.pos_xyz,
                np.array([
                    self.header.grid_scale_factor,
                    self.header.grid_scale_factor,
                    1
                ])
            )
            + self.start_nev
        )

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

    def _get_nev(self) -> None:
        self.n, self.e, self.tvd = get_nev(
            np.array([
                self.x,
                self.y,
                self.z
            ]).T,
            start_xyz=self.start_xyz,
            start_nev=self.start_nev
        ).reshape(-1, 3).T

    def _make_angles(
        self, inc: ArrayLike, azi: ArrayLike, deg: bool = True
    ) -> None:
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

    def get_error(
        self, error_model: str, return_error: bool = False
    ) -> Union[ErrorModel, "Survey"]:
        """Apply an error model and compute covariance matrices.

        Parameters
        ----------
        error_model : str
            Name of the error model (e.g. ``"ISCWSA_MWD"``).
        return_error : bool
            If True, return the ErrorModel object; otherwise
            return the Survey with updated covariances.

        Returns
        -------
        ErrorModel or Survey
            The ErrorModel object if ``return_error`` is True, otherwise
            the Survey instance with updated covariance attributes.

        Raises
        ------
        AssertionError
            If ``error_model`` is not a recognized model name.
        """
        assert error_model in ERROR_MODELS, "Undefined error model"

        self.error_model = error_model
        self._get_errors()

        if return_error:
            # error_model was just applied above, so self.err is set (not None)
            return self.err  # type: ignore[return-value]
        else:
            return self

    def _get_errors(self) -> None:
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
            self.cov_hla = self.err.errors.cov_HLAs
            self.cov_nev = self.err.errors.cov_NEVs
            self.cov_nev_random = self.err.errors.cov_NEVs_random
            self.cov_nev_systematic = self.err.errors.cov_NEVs_systematic
            self.cov_nev_global = self.err.errors.cov_NEVs_global
            self.cov_nev_within_pad = self.err.errors.cov_NEVs_within_pad
        else:
            if self.cov_nev is not None and self.cov_hla is None:
                self.cov_hla = NEV_to_HLA(self.survey_rad, self.cov_nev)
            elif self.cov_nev is None and self.cov_hla is not None:
                self.cov_nev = HLA_to_NEV(self.survey_rad, self.cov_hla)
            else:
                pass

        if (
            self.start_cov_nev is not None
            and self.cov_nev is not None
        ):
            self.cov_nev += self.start_cov_nev
            self.cov_hla = NEV_to_HLA(self.survey_rad, self.cov_nev)

    def _curvature_to_rate(self, curvature: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            radius = 1 / curvature
        circumference = 2 * np.pi * radius
        if self.unit == 'meters':
            x = 30
        else:
            x = 100
        rate = np.absolute(np.degrees(2 * np.pi / circumference) * x)

        return rate

    def _get_toolface_and_rates(self) -> None:
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
        # TODO: update so its the same length as the survey - need to cascade
        n12 = np.cross(s.vec1_nev, s.vec2_nev)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.normals = n12 / np.linalg.norm(n12, axis=1).reshape(-1, 1)

        # get radius vectors
        with np.errstate(divide='ignore', invalid='ignore'):
            self.vec_radius_nev = np.cross(
                np.vstack((self.normals[0], self.normals)),  # temp fix
                self.vec_nev
            )

    def _get_sections(
        self, rtol: float = 0.1, atol: float = 0.1, dls_cont: bool = True
    ) -> list:
        sections = get_sections(self, rtol, atol, dls_cont)

        return sections

    def get_nev_arr(self) -> np.ndarray:
        """Return survey positions as an (n, 3) array of [N, E, TVD].

        Returns
        -------
        ndarray
            Array of shape (n, 3) with northing, easting, and TVD columns.
        """
        return np.array([
            self.n,
            self.e,
            self.tvd
        ]).T.reshape(-1, 3)

    def highside_vec_nev(self) -> np.ndarray:
        """Unit high-side vector at each station, as an (n, 3) [N, E, V] array.

        The high side is perpendicular to the wellbore axis, in the vertical
        plane containing it, pointing to the high side of the hole (up-dip). It
        is the high-side (H) basis vector of the NEV->HLA transform, expressed in
        NEV, using the grid azimuth (consistent with :meth:`get_nev_arr`). For a
        horizontal well it points straight up ([0, 0, -1]); for a vertical well
        the high side is undefined and this returns the azimuth direction.

        Returns
        -------
        ndarray
            Array of shape (n, 3) of unit high-side vectors in [N, E, V].
        """
        ci, si = np.cos(self.inc_rad), np.sin(self.inc_rad)
        ca, sa = np.cos(self.azi_grid_rad), np.sin(self.azi_grid_rad)
        return np.stack([ci * ca, ci * sa, -si], axis=-1)

    def save(self, filename: str) -> None:
        """
        Saves a minimal (control points) survey listing as a .csv file,
        including the survey header information.

        Parameters
        ----------
        filename: str
            The path and filename for saving the text file.
        """
        export_csv(self, filename)

    def interpolate_mds(self, md: ArrayLike) -> "Survey":
        """
        Method to interpolate positions at an array of measured depths and
        return a new `welleng.Survey` object. This is a vectorized equivalent
        of looping the scalar `interpolate_md`, and produces a survey
        equivalent to `interpolate_survey` when passed the same station
        measured depths.

        Parameters
        ----------
        md: (,n) list or array of floats
            The measured depths of the points of interest.

        Returns
        -------
        A welleng.survey.Survey object with an `interpolated` property
        indicating whether each station was interpolated (True) or is an
        original survey station (False).

        Examples
        --------
        >>> import welleng as we
        >>> import numpy as np
        >>> survey = we.survey.Survey(
        ...       md=[0, 500, 1000, 2000, 3000],
        ...       inc=[0, 0, 30, 90, 90],
        ...       azi=[0, 0, 45, 135, 180],
        ...    )
        >>> survey_interp = survey.interpolate_mds(np.arange(0, 3000, 30))
        """
        return interpolate_mds(self, md)

    def interpolate_md(self, md: float) -> Optional[Node]:
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
        if s is None:
            return None
        node = get_node(s, -1, s.interpolated[-1])  # type: ignore[index]

        return node

    def interpolate_tvd(self, tvd: float) -> list:
        """Interpolate the survey at a target true vertical depth.

        Reversal-robust (Sawaryn & Thorogood 2005, SPE-84246-PA): returns
        *every* crossing of ``tvd``, so a target hit twice by a TVD reversal
        yields two Nodes.

        Parameters
        ----------
        tvd : float
            The true vertical depth at which to interpolate.

        Returns
        -------
        list of Node
            Every crossing of ``tvd``, sorted by measured depth (normally a
            single element; empty if ``tvd`` is outside the well's TVD range).

        Notes
        -----
        Breaking change (welleng 0.15.0): returns a ``list`` of Nodes instead
        of a single Node. Use ``interpolate_tvd(tvd)[0]`` on a monotonic well
        for the previous behaviour.
        """
        return interpolate_tvd(self, tvd=tvd)

    def interpolate_survey_tvd(
        self, start: Optional[float] = None, stop: Optional[float] = None,
        step: float = 10
    ) -> "Survey":
        """
        Convenience method for interpolating a Survey object's TVD.
        """
        survey_interpolated = interpolate_survey_tvd(
            self, start=start, stop=stop, step=step
        )
        return survey_interpolated

    def interpolate_survey(
        self, step: float = 30, dls: float = 1e-8
    ) -> "Survey":
        """
        Convenience method for interpolating a Survey object's MD.
        """
        survey_interpolated = interpolate_survey(self, step=step, dls=dls)
        return survey_interpolated

    def figure(self, type: str = 'scatter3d', **kwargs: Any) -> Any:
        """Generate a plotly figure of the survey trajectory.

        Parameters
        ----------
        type : str
            Plot type passed to ``welleng.visual.figure``.
        **kwargs
            Additional keyword arguments forwarded to the plotting
            function.

        Returns
        -------
        object
            A plotly figure object.
        """
        fig = figure(self, type, **kwargs)
        return fig

    def project_to_bit(
        self, delta_md: float, dls: Optional[float] = None,
        toolface: Optional[float] = None
    ) -> Node:
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
        node_target: Node,
        dls_design: float = 3.0,
        delta_md: Optional[float] = None,
        dls: Optional[float] = None, toolface: Optional[float] = None,
        step: float = 30
    ) -> "Survey":
        """Project a wellpath from the end of this survey to a target node.

        Parameters
        ----------
        node_target : Node
            The target Node to connect to.
        dls_design : float
            Design dogleg severity (deg/30m) for the connection.
        delta_md : float or None
            Along-hole distance from survey tool to bit. If None,
            projection starts at the last survey station.
        dls : float or None
            DLS for the projection to the bit. Defaults to last survey DLS.
        toolface : float or None
            Toolface for the projection to the bit. Defaults to last
            survey toolface.
        step : float
            Survey interval (m) for the projected wellpath.

        Returns
        -------
        Survey
            A Survey object representing the projected path to the target.
        """
        survey = project_to_target(
            self,
            node_target,
            dls_design,
            delta_md,
            dls, toolface,
            step
        )
        return survey

    def _get_vertical_section(self, *args: Any, **kwargs: Any) -> None:
        """
        Internal function to initiate the vertical section by calculating
        the magnitude of the lateral displacement and, if a vertical section
        azimuth is defined, calculating the vertical section lateral
        component along the given vertical section azimuth (relative to the
        defined reference azimuth).
        """
        self.hypot = np.hypot(
            self.n[1:] - self.n[:-1],
            self.e[1:] - self.e[:-1]
        )

        if self.header.vertical_section_azimuth is not None:
            self.vertical_section = self.get_vertical_section(
                self.header.vertical_section_azimuth, deg=False
            )
        else:
            self.vertical_section = None

    def get_vertical_section(
        self, vertical_section_azimuth: float, deg: bool = True
    ) -> np.ndarray:
        """
        Calculate the vertical section.

        Parameters
        ----------
        vertical_section_azimuth: float
            The azimuth (relative to the reference azimuth defined in the
            survey header) along which to calculate the vertical section
            lateral displacement.
        deg: boolean (default: True)
            Indicates whether the vertical section azimuth parameter is in
            degrees or radians (True or False respectively).

        Returns
        -------
        result: (n, 1) ndarray
        """
        vertical_section_azimuth = (
            vertical_section_azimuth if not deg
            else math.radians(vertical_section_azimuth)
        )
        azi_temp = getattr(
            self,
            f"azi_{self.azi_ref_lookup[self.header.azi_reference]}_rad"
        )
        azi_temp[np.where(self.inc_rad == 0.0)] = (
            vertical_section_azimuth
        )
        result = np.cos(
            vertical_section_azimuth
            - (azi_temp[1:] + azi_temp[:-1]) / 2
        ) * self.hypot

        result = np.cumsum(np.hstack(([0.0], result)))

        return result

    def set_vertical_section(
        self, vertical_section_azimuth: float, deg: bool = True
    ) -> None:
        """
        Sets the vertical_section_azimuth property in the survey header and
        the vertical section data with the data calculated for the input
        azimuth.

        Parameters
        ----------
        vertical_section_azimuth: float
            The azimuth (relative to the reference azimuth defined in the
            survey header) along which to calculate the vertical section
            lateral displacement.
        deg: boolean (default: True)
            Indicates whether the vertical section azimuth parameter is in
            degrees or radians (True or False respectively).
        """
        self.header.vertical_section_azimuth = vertical_section_azimuth
        self.vertical_section = self.get_vertical_section(
            vertical_section_azimuth, deg
        )

    def modified_tortuosity_index(
        self, rtol: float = 1.0, dls_tol: Optional[float] = 1e-3,
        step: Optional[float] = 1.0, dls_noise: Optional[float] = 1.0,
        data: bool = False,
        **kwargs: Any
    ) -> Union[np.ndarray, dict]:
        """
        Convenience method for the Modified Tortuosity Index (MTI): a native-3D,
        *dimensionless* variant of the Tortuosity Index (TI) of Ashok et al.
        ([IADD presentation](https://www.iadd-intl.org/media/files/files/47d68cb4/iadd-luncheon-february-22-2018-v2.pdf))
        and D'Angelo et al. (SPE/IADC-194099-MS).

        Compared with :func:`tortuosity_index`, the MTI divides each curve
        turn's ``(L_cs / L_xs - 1)`` term by its arc length ``L_cs`` and uses
        ``L_c`` (rather than ``1 / L_c``) as the normalizing factor, which makes
        the result independent of the survey's unit of length (a survey in feet
        and the same survey in metres give the same MTI). See
        [the method post](https://jonnymaserati.github.io/2022/05/26/a-modified-tortuosity-index.html).

        .. warning::
            "MTI" here means *Modified* Tortuosity Index. In SPE/IADC-194099-MS
            "MTI" denotes the unrelated *Mapped* Tortuosity Index (planned curve
            turns mapped onto the as-drilled path); do not conflate the two.

        By default the survey is pre-processed with the maximum-curvature method
        (interpolated to ``step`` then ``dls_noise`` deg/30m added) so that the
        MTI is robust to survey-station frequency; set ``dls_noise=None`` to use
        the raw minimum-curvature survey instead.

        Parameters
        ----------
        rtol: float
            Relative tolerance when testing normal-vector continuity (passed to
            ``numpy.isclose`` as both ``rtol`` and ``atol``).
        dls_tol: float or None
            If not None, additionally require dogleg-severity continuity within
            this tolerance when sectionizing.
        step: float or None
            Step length (metres) for interpolating the survey before applying
            the maximum-curvature method. Ignored if ``dls_noise`` is None.
        dls_noise: float or None
            Incremental Dog Leg Severity (deg/30m) added by the maximum-curvature
            method. If None, no pre-processing is done and minimum curvature is
            assumed. When applied, every leg is treated as a **slide**
            (``steering='slide'``) - the maximum, worst-case tortuosity the method
            is defined to give; the slide/rotary distinction of
            :meth:`maximum_curvature` is deliberately not exposed here, since the
            MTI is a conservative geometric quality metric rather than an
            error-propagation calculation.
        data: bool
            If True, return a dict of intermediate properties instead of the
            array.

        Returns
        -------
        mti: (n,) ndarray or dict
            Per-station modified tortuosity index, or a dict of intermediate
            results (``starts``, ``mds``, ``locs``, ``l_cs``, ``l_xs``,
            ``mti``, ``survey`` ...) if ``data`` is True.

        References
        ----------
        Further details on the maximum-curvature method and survey-frequency
        robustness are
        [here](https://jonnymaserati.github.io/2022/06/19/modified-tortuosity-index-survey-frequency.html).
        """
        # Check whether to pre-process the survey to apply maximum curvature.
        if bool(dls_noise):
            survey = self.interpolate_survey(step=step)  # type: ignore[arg-type]
            # The MTI's max-curvature is the worst-case (all-slide) tortuosity by
            # definition; pass steering explicitly so maximum_curvature does not
            # demand a per-leg mode the MTI intentionally does not model.
            survey = survey.maximum_curvature(
                dls_noise=dls_noise, steering='slide'  # type: ignore[arg-type]
            )

        else:
            survey = self

        return modified_tortuosity_index(
            survey, rtol=rtol, dls_tol=dls_tol, data=data, **kwargs
        )

    def tortuosity_index(
        self, rtol: float = 0.01, dls_tol: Optional[float] = None,
        data: bool = False, **kwargs: Any
    ) -> Union[np.ndarray, dict]:
        """
        Convenience method for the Tortuosity Index (TI), a native-3D variant
        of the method presented in the [IADD presentation](https://www.iadd-intl.org/media/files/files/47d68cb4/iadd-luncheon-february-22-2018-v2.pdf)
        by Pradeep Ashok et al. and in SPE/IADC-194099-MS by D'Angelo et al.
        (itself adapted from the retinal-vessel tortuosity work of Grisan et
        al.).

        The published method computes a tortuosity index separately in the
        inclination and azimuth domains and combines them as the root of the
        sum of squares. However, the arc-length and chord-length terms used in
        each domain are the full 3D quantities (only the curve-turn *detection*
        differs between the domains), so the two components are not independent
        and the 3D curvature is effectively double-counted. This method avoids
        that by sectionizing the trajectory directly in 3D: a curve turn is
        considered continuous while its normal vector (``vec_i x vec_j``)
        remains constant, which inherently accounts for torsion. See
        :func:`tortuosity_index` for the implementation and
        :func:`modified_tortuosity_index` for the dimensionless variant.

        Note that TI is *not* dimensionless (the result scales with the unit of
        length, since the ``1 / L_c`` normalization carries length units); per
        SPE/IADC-194099-MS a scale factor of ``1e7`` is applied so values fall
        in a convenient range. Compute ``L_c`` in feet to compare with the
        published reference ranges.

        Parameters
        ----------
        rtol: float
            Relative tolerance when testing normal-vector continuity (passed to
            ``numpy.isclose`` as both ``rtol`` and ``atol``).
        dls_tol: float or None
            If not None, additionally require dogleg-severity continuity within
            this tolerance when sectionizing.
        data: bool
            If True, return a dict of intermediate properties instead of the
            array.
        **kwargs
            ``coeff`` (length unit conversion, default 0.3048 -> feet) and
            ``kappa`` (scale factor, default 1e7) may be overridden.

        Returns
        -------
        ti : ndarray or dict
            Per-station tortuosity index, or a dict of results if ``data``.
        """

        return tortuosity_index(
            self, rtol=rtol, dls_tol=dls_tol, data=data, **kwargs
        )

    def tortuosity_views(
        self, modified: bool = True, target_md: Optional[float] = None,
        **kwargs: Any
    ) -> dict:
        """Total, remaining and local readings of the tortuosity profile.

        The tortuosity index is evaluated at every station, so it is a profile;
        these are the three engineering reads of it (see the MTI paper):

        - **total**: the value at the end (the whole-well KPI scalar);
        - **remaining**: the increment still to accumulate from each station to
          the end (or to ``target_md``) — what is left to drill;
        - **local**: the along-hole gradient of the index — flags a single
          tortuous interval that the total would hide.

        Parameters
        ----------
        modified: bool
            If True (default) use the dimensionless
            :meth:`modified_tortuosity_index`; otherwise use
            :meth:`tortuosity_index`.
        target_md: float or None
            Reference depth for ``remaining``; defaults to total depth.
        **kwargs
            Passed through to the underlying index method.

        Returns
        -------
        dict
            ``{'md', 'total', 'remaining', 'local'}`` where ``md`` is the depth
            grid the profile is evaluated on (the maximum-curvature pre-processed
            grid when ``modified`` pre-processing is active).
        """
        if modified:
            d = self.modified_tortuosity_index(data=True, **kwargs)
            profile, md = d['mti'], d['survey'].md  # type: ignore[union-attr]
        else:
            d = self.tortuosity_index(data=True, **kwargs)
            profile, md = d['ti'], self.md
        views = tortuosity_views(profile, md, target_md=target_md)
        views['md'] = md
        return views

    def directional_difficulty_index(self, **kwargs: Any) -> np.ndarray:
        """
        Taken from IADC/SPE 59196 The Directional Difficulty Index - A
        New Approach to Performance Benchmarking by Alistair W. Oag et al.

        Returns
        -------
        data: (n) array of floats
            The ddi for each survey station.
        """

        return directional_difficulty_index(self, **kwargs)

    def maximum_curvature(
        self, dls_noise: float = 1.0,
        steering: Optional[Union[str, ArrayLike]] = None
    ) -> "Survey":
        """
        Create a well trajectory using the Maximum Curvature method.

        Parameters
        ----------
        survey: welleng.survey.Survey object
        dls_noise: float
            The additional Dog Leg Severity (DLS) in deg/30m used to calculate
            the curvature for the initial section of the survey interval.
        steering: {'slide', 'rotary'}, or (,n) array-like of those / bool
            The slide/rotary mode of each leg - a physical drilling property that
            is *not* inferable from the survey (inclination, azimuth, measured
            depth); it is either a single value applied to every leg, or one entry
            per survey station (the mode of the leg arriving at that station):

            - ``'slide'`` - the leg was steered by sliding a bent motor at an
              oriented toolface. The extra ``dls_noise`` curvature is applied in
              the surveyed toolface, giving a **directional** deflection (the
              survey-interval error has a consistent sign - the well lands
              shallower).
            - ``'rotary'`` - the leg was drilled rotating (RSS or rotary hold).
              The toolface averages out over rotation, so no directional
              deflection is applied (the leg keeps its minimum-curvature path);
              the survey-interval error there is *random*, not directional.

            A boolean array is read as ``True == slide``. **Defaults to**
            ``'slide'`` - the conservative choice, since folding a directional
            bias into a symmetric (rotary) treatment would under-state the error;
            **rotary is therefore opt-in** and must be positively declared. Falls
            back to ``Survey.steering`` when the argument is left as ``None``.

        Returns
        -------
        survey_new: welleng.Survey.survey object
            A revised survey object calculated using the Minimum Curvature
            method with updated survey positions and additional mid-point
            stations.

        Raises
        ------
        ValueError
            If an array is given whose length does not match the number of survey
            stations.
        """
        # slide/rotary mode per leg. Default 'slide' (conservative - the
        # directional bias is included); rotary is opt-in and must be declared,
        # because randomising a directional bias under-states the error.
        mode = steering if steering is not None else self.steering
        if mode is None:
            mode = 'slide'
        if isinstance(mode, str):
            mode = np.full(len(self.md), mode)
        mode = np.asarray(mode)
        if mode.shape[0] != len(self.md):
            raise ValueError(
                "steering must have one entry per survey station "
                f"({len(self.md)}); got {mode.shape[0]}."
            )
        if mode.dtype.kind in ('U', 'S', 'O'):
            is_slide = np.array([str(m).lower() != 'rotary' for m in mode])
        else:
            is_slide = mode.astype(bool)

        # Apply the DLS increment only on slide legs; rotary legs stay at the
        # surveyed DLS (i.e. minimum curvature - no directional deflection).
        dls_noise_arr = np.where(is_slide, dls_noise, 0.0)

        dls_effective = self.dls + dls_noise_arr
        # dls_effective == 0 on a rotary hold (no surveyed curvature, no added
        # noise) -> infinite radius / straight leg; the divide is expected.
        with np.errstate(divide='ignore'):
            radius_effective = radius_from_dls(dls_effective)

        dogleg1 = (
            (self.delta_md / radius_effective) / 2
        )

        radius_effective = np.where(
            dogleg1 > np.pi,
            self.delta_md * 4 / (2 * np.pi),
            radius_effective
        )

        # Each leg's mid-station is the end of an arc of swept angle dogleg1,
        # transformed by the leg toolface and the start-station orientation.
        # Only the arc end-vector and arc length (dogleg * radius) are needed
        # (the new Survey recomputes positions by minimum curvature), so this is
        # vectorised over all legs rather than calling get_arc per station.
        dl = dogleg1[1:]
        # arc end-vector in the local arc frame: [sin(dogleg), 0, cos(dogleg)]
        local_vec = np.column_stack((np.sin(dl), np.zeros_like(dl), np.cos(dl)))
        inc_azi = get_angles(self.vec_nev[:-1], nev=True)
        euler = np.column_stack((self.toolface[:-1], inc_azi[:, 0], inc_azi[:, 1]))
        vec_new = R.from_euler('zyz', euler, degrees=False).apply(local_vec)
        inc_azi_new = np.degrees(get_angles(vec_new, nev=True))

        _survey_new = np.column_stack((
            # mid-station MD = leg start + half the leg (arc length dogleg*radius
            # == delta_md/2 analytically; computed directly to stay finite when a
            # rotary hold leg has an infinite radius).
            self.md[:-1] + self.delta_md[1:] / 2,
            inc_azi_new[:, 0],
            inc_azi_new[:, 1],
        ))

        survey_new = np.zeros(shape=(len(_survey_new) * 2 + 1, 3))
        survey_new[:-1] = np.stack(
            (self.survey_deg[:-1].T, _survey_new.T),
            axis=1
        ).T.reshape(-1, 3)
        survey_new[-1] = self.survey_deg[-1]

        # Update the new survey header as the new azimuth reference is 'grid'.
        sh = self.header
        sh.azi_reference = 'grid'

        # Create a new Survey instance
        survey = Survey(
            md=survey_new[:, 0],
            inc=survey_new[:, 1],
            azi=survey_new[:, 2],
            header=sh,
            start_xyz=self.start_xyz,
            start_nev=self.start_nev
        )

        # Update the interpolated property to keep track of the original survey
        # stations.
        survey.interpolated = np.full_like(survey.md, True)
        survey.interpolated[::2] = self.interpolated

        return survey

    def torsion(self) -> np.ndarray:
        """Geometric torsion ``τ`` per station (radians per unit length).

        The helical rate of the wellpath — the rate at which the osculating-plane
        normal (``self.normals``, i.e. the Serret-Frenet binormal direction) rotates
        along the trajectory. **Zero on a planar (2D) trajectory**; nonzero only where
        the well turns out of plane. This is a geometric property of the path and is
        distinct from mechanical drillstring twist.

        Discrete form (Mitchell & Samuel 2009, SPE-105068-PA, Eq. 17):

            τ_j = arccos(b_{j-1} · b_j) / Δs_j

        with ``b`` the unit osculating-plane normal and ``Δs_j`` the central
        measured-depth spacing about station ``j``. This is the same normal-vector
        continuity the (modified) tortuosity index uses — a section of constant normal
        is planar and has zero torsion.

        Returns
        -------
        torsion : ndarray of shape (n,)
            Per-station geometric torsion (rad/unit length), aligned to ``self.md``.
            The two end stations (undefined) and any straight-hold sections (undefined
            osculating plane) are set to 0.

        Notes
        -----
        Consumers (e.g. the stiff-string T&D 3D contact terms, SPE-105068-PA Eq. 20)
        should prefer a smooth (spline) trajectory: a minimum-curvature survey gives a
        bending-moment discontinuity at stations (App. F), so torsion evaluated on raw
        min-curve stations is a station-spaced approximation.
        """
        b = self.normals                                    # (n-1, 3) unit osculating normals
        dot = np.clip(np.sum(b[:-1] * b[1:], axis=1), -1.0, 1.0)   # (n-2,)
        ang = np.arccos(dot)
        with np.errstate(divide='ignore', invalid='ignore'):
            ds = (self.md[2:] - self.md[:-2]) / 2.0         # central spacing at stations 1..n-2
            tau_interior = ang / ds
        torsion = np.zeros(len(self.md), dtype=float)
        torsion[1:-1] = np.nan_to_num(tau_interior, nan=0.0, posinf=0.0, neginf=0.0)
        return torsion

    def curvature_rate(self) -> np.ndarray:
        """Rate of change of curvature ``dκ/ds`` per station (rad per unit length²).

        The along-hole derivative of curvature ``κ = dogleg / Δmd``. With the torsion
        term it completes the 3D stiff-string contact force (SPE-105068-PA, Eq. 20:
        the ``EI·τ·dκ/ds`` binormal term). Zero on a constant-curvature (constant-DLS)
        section.

        Discrete form (SPE-105068-PA, Eq. 18): ``(κ_{j+1} − κ_j) / (s_{j+1} − s_j)``.

        Returns
        -------
        dkappa_ds : ndarray of shape (n,)
            Per-station ``dκ/ds`` aligned to ``self.md``; the final station (undefined
            forward difference) is 0.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            kappa = np.where(self.delta_md > 0, self.dogleg / self.delta_md, 0.0)
            dk = np.diff(kappa)
            ds = np.diff(self.md)
            rate = np.where(ds > 0, dk / ds, 0.0)
        dkappa_ds = np.zeros(len(self.md), dtype=float)
        dkappa_ds[:-1] = np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
        return dkappa_ds


def modified_tortuosity_index(
    survey: "Survey", rtol: float = 1.0, dls_tol: Optional[float] = 1e-3,
    data: bool = False, **kwargs: Any
) -> Union[np.ndarray, dict]:
    """
    Calculate the Modified Tortuosity Index (MTI): a native-3D, dimensionless
    variant of the Tortuosity Index (TI) of Ashok et al. ([IADD presentation](https://www.iadd-intl.org/media/files/files/47d68cb4/iadd-luncheon-february-22-2018-v2.pdf))
    and D'Angelo et al. (SPE/IADC-194099-MS).

    The trajectory is split into curve-turn / hold sections in 3D via
    normal-vector continuity (see :func:`_get_ti_data`). Each section's
    ``(L_cs / L_xs - 1)`` term is divided by its arc length ``L_cs`` and the
    running sum is scaled by ``n / (n + 1)`` and the curve length ``L_c``,
    making the result independent of the unit of length. ``L_cs`` is the
    along-hole (arc) distance from the section start to each station and
    ``L_xs`` the corresponding straight-line (chord) distance.

    Note: "MTI" here is the *Modified* Tortuosity Index; in SPE/IADC-194099-MS
    "MTI" is the unrelated *Mapped* Tortuosity Index.

    Parameters
    ----------
    survey : welleng.survey.Survey
    rtol : float
        Relative tolerance for normal-vector continuity (also used as atol).
    dls_tol : float or None
        If not None, also require dogleg-severity continuity within this
        tolerance.
    data : bool
        If True, return a dict of intermediate properties.
    **kwargs
        ``coeff`` (unit conversion, default 1.0) and ``kappa`` (scale factor,
        default 1) may be overridden.

    Returns
    -------
    mti : ndarray or dict
    """
    # set default params
    coeff = kwargs.get('coeff', 1.0)  # length-unit conversion (1.0 -> as-is)
    # honour the correctly-spelled 'kappa'; fall back to the legacy 'kapa' typo
    kappa = kwargs.get('kappa', kwargs.get('kapa', 1))

    continuous, starts, mds, locs, n_sections, n_sections_arr = _get_ti_data(
        survey, rtol, dls_tol
    )

    l_cs = (
        survey.md[1:] - mds[n_sections_arr - 1]
    ) / coeff
    l_xs = np.linalg.norm(
        survey.pos_nev[1:]
        - np.array(locs)[n_sections_arr - 1],
        axis=1
    ) / coeff
    b = (
        (l_cs / l_xs) - 1
    ) / l_cs

    a = _accumulate_sections(b, n_sections_arr)

    mti = np.hstack((
        np.array([0.0]),
        (
            # 1
            (n_sections_arr / (n_sections_arr + 1))
            * (kappa * ((survey.md[1:] - survey.md[0]) / coeff))
            # * (kappa / (np.cumsum(survey.dogleg)[1:] + 1))
            * a
        )
    ))

    if data:
        return {
            'continuous': continuous, 'starts': starts, 'mds': mds,
            'locs': locs, 'n_sections': n_sections,
            'n_sections_arr': n_sections_arr, 'l_cs': l_cs, 'l_xs': l_xs,
            'mti': mti, 'survey': survey
        }

    return mti


def tortuosity_index(
    survey: "Survey", rtol: float = 0.01, dls_tol: Optional[float] = None,
    data: bool = False, **kwargs: Any
) -> Union[np.ndarray, dict]:
    """
    Calculate the Tortuosity Index (TI), a native-3D variant of the method of
    Ashok et al. ([IADD presentation](https://www.iadd-intl.org/media/files/files/47d68cb4/iadd-luncheon-february-22-2018-v2.pdf))
    and D'Angelo et al. (SPE/IADC-194099-MS).

    The trajectory is split into curve-turn / hold sections in 3D via
    normal-vector continuity (see :func:`_get_ti_data`); each section's
    ``(L_cs / L_xs - 1)`` term is accumulated, scaled by ``n / (n + 1)`` and
    normalized by ``1 / L_c``, then by ``kappa`` (1e7 per SPE/IADC-194099-MS).
    ``L_cs`` is the along-hole (arc) distance from the section start to each
    station and ``L_xs`` the corresponding straight-line (chord) distance.

    TI is *not* dimensionless: the result scales with the unit of length, so
    ``coeff`` defaults to 0.3048 to express ``L_c`` in feet and match the
    published reference ranges. See :func:`modified_tortuosity_index` for the
    dimensionless variant.

    Parameters
    ----------
    survey : welleng.survey.Survey
    rtol : float
        Relative tolerance for normal-vector continuity (also used as atol).
    dls_tol : float or None
        If not None, also require dogleg-severity continuity within this
        tolerance.
    data : bool
        If True, return a dict of intermediate properties.
    **kwargs
        ``coeff`` (unit conversion, default 0.3048 -> feet) and ``kappa``
        (scale factor, default 1e7) may be overridden.

    Returns
    -------
    ti : ndarray or dict
    """
    # set default params
    coeff = kwargs.get('coeff', 0.3048)  # length-unit conversion (-> feet)
    # honour the correctly-spelled 'kappa'; fall back to the legacy 'kapa' typo
    kappa = kwargs.get('kappa', kwargs.get('kapa', 1e7))

    continuous, starts, mds, locs, n_sections, n_sections_arr = _get_ti_data(
        survey, rtol, dls_tol
    )

    l_cs = (survey.md[1:] - mds[n_sections_arr - 1]) / coeff
    l_xs = np.linalg.norm(
        survey.pos_nev[1:]
        - np.array(locs)[n_sections_arr - 1],
        axis=1
    ) / coeff
    b = (
        (l_cs / l_xs) - 1
    )

    a = _accumulate_sections(b, n_sections_arr)

    ti = np.hstack((
        np.array([0.0]),
        (
            (n_sections_arr / (n_sections_arr + 1))
            * (kappa / ((survey.md[1:] - survey.md[0]) / coeff))
            * a
        )
    ))

    if data:
        return {
            'continuous': continuous, 'starts': starts, 'mds': mds,
            'locs': locs, 'n_sections': n_sections,
            'n_sections_arr': n_sections_arr, 'l_cs': l_cs, 'l_xs': l_xs,
            'ti': ti
        }

    return ti


def tortuosity_views(
    profile: ArrayLike, md: ArrayLike, target_md: Optional[float] = None
) -> dict:
    """Derive total, remaining and local readings from a tortuosity profile.

    Parameters
    ----------
    profile: array_like
        A per-station tortuosity index profile (TI or MTI), monotonic
        non-decreasing.
    md: array_like
        Measured depth at each station, same length as ``profile``.
    target_md: float or None
        Reference depth for the ``remaining`` calculation; defaults to the last
        station (total depth).

    Returns
    -------
    dict
        ``total`` (float, the profile value at ``target_md`` / end),
        ``remaining`` (ndarray, ``total`` minus the profile — what is left to
        accumulate from each station), ``local`` (ndarray, the along-hole
        gradient ``d(profile)/d(md)`` — the rate of tortuosity accumulation).
    """
    profile = np.asarray(profile, dtype=float)
    md = np.asarray(md, dtype=float)
    end = (
        float(profile[-1]) if target_md is None
        else float(np.interp(target_md, md, profile))
    )
    return {
        'total': end,
        'remaining': end - profile,
        'local': np.gradient(profile, md),
    }


def directional_difficulty_index(survey: "Survey", **kwargs: Any) -> np.ndarray:
    """
    Taken from IADC/SPE 59196 The Directional Difficulty Index - A
    New Approach to Performance Benchmarking by Alistair W. Oag et al.
    Parameters
    ----------
    survey: welleng.survey.Survey object
    data: bool
        If True, returns the ddi at each survey station.
    Returns
    -------
    ddi: float
        The ddi for the well at well (at TD).
    data: (n) array of floats
        The ddi for each survey station.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ddi = np.nan_to_num(np.log10(
            (
                (survey.md * ureg.meters).to('ft').m
                * (
                    np.linalg.norm(
                        (survey.n, survey.e), axis=0
                    ) * ureg.meters
                ).to('ft').m
                * np.cumsum(np.degrees(survey.dogleg))
            )
            / (survey.tvd * ureg.meters).to('ft').m
        ), nan=0.0, posinf=0.0, neginf=0.0)

    return ddi


def _accumulate_sections(
    b: ArrayLike, n_sections_arr: np.ndarray
) -> np.ndarray:
    """Per-section cumulative sum used by the tortuosity indices.

    Returns ``a`` where ``a[k] = b[k] + (sum of each prior section's final b)``,
    i.e. each station's section contribution plus the running total of every
    completed section. This is the vectorized (O(n)) equivalent of the
    per-section carry loop (which was O(sections * n) because it masked the full
    array once per section).

    ``n_sections_arr`` is the per-station section index and must be
    non-decreasing (stations in measured-depth order, as produced by
    :func:`_get_ti_data`).
    """
    b = np.asarray(b, dtype=float)
    if b.size == 0:
        return b.copy()
    # last index of each contiguous section run
    run_last = np.concatenate((
        np.where(np.diff(n_sections_arr) != 0)[0],
        [b.size - 1]
    ))
    # carry into each run = cumulative sum of the previous runs' final b
    carry_per_run = np.concatenate(([0.0], np.cumsum(b[run_last])[:-1]))
    run_sizes = np.diff(np.concatenate(([-1], run_last)))
    return b + np.repeat(carry_per_run, run_sizes)


def _get_ti_data(
    survey: "Survey", rtol: float, dls_tol: Optional[float] = None
) -> tuple:
    """Sectionize a survey into curve-turn / hold sections for the TI/MTI.

    A section is continuous while successive stations share the same normal
    vector (``vec_i x vec_j``, NaN for straight holds) within ``rtol`` and,
    optionally, the same dogleg severity within ``dls_tol``. Hold (straight)
    sections are treated as discrete sections with arc length == chord length.

    Parameters
    ----------
    survey : welleng.survey.Survey
    rtol : float
        Relative (and absolute) tolerance for normal-vector continuity.
    dls_tol : float or None
        If not None, also require dogleg-severity continuity within this
        tolerance.

    Returns
    -------
    continuous : ndarray of bool
        Per-station continuity flags.
    starts : ndarray of int
        Station indices at which each section starts (always includes 0).
    mds : ndarray of float
        Measured depth at each section start.
    locs : ndarray
        NEV position at each section start.
    n_sections : ndarray of int
        1-based section numbers.
    n_sections_arr : ndarray of int
        Section number for each station (from ``survey.md[1:]``).
    """
    if dls_tol is None:
        dls_continuity = np.full(len(survey.dls) - 2, True)
    else:
        dls_continuity = np.isclose(
            survey.dls[1:-1],
            survey.dls[2:],
            equal_nan=True,
            rtol=dls_tol,
            atol=rtol
        )
    continuous = np.all((
        np.all(
            np.isclose(
                survey.normals[:-1],
                survey.normals[1:],
                equal_nan=True,
                rtol=rtol, atol=rtol
            ), axis=-1
        ),
        dls_continuity
    ), axis=0)

    starts = np.concatenate((
        np.array([0]),
        np.where(continuous == False)[0] + 1,
    ))

    mds = survey.md[starts]
    locs = survey.pos_nev[starts]
    n_sections = np.arange(1, len(starts) + 1, 1)
    n_sections_arr = np.searchsorted(mds, survey.md[1:])

    return (
        continuous, starts, mds, locs, n_sections, n_sections_arr
    )


class TurnPoint:
    """A control point in a well plan, representing a hold or curve section.

    Used when discretizing a survey into sections for export to planning
    software (e.g. Landmark COMPASS .wbp format).
    """

    def __init__(
        self,
        md: Optional[float] = None,
        inc: Optional[float] = None,
        azi: Optional[float] = None,
        build_rate: Optional[float] = None,
        turn_rate: Optional[float] = None,
        dls: Optional[float] = None,
        toolface: Optional[float] = None,
        method: Optional[str] = None,
        target: Optional[Any] = None,
        tie_on: bool = False,
        location: Optional[list] = None
    ) -> None:
        """Initialize a TurnPoint.

        Parameters
        ----------
        md : float or None
            Measured depth.
        inc : float or None
            Inclination in degrees.
        azi : float or None
            Azimuth in degrees.
        build_rate : float or None
            Build rate in deg per unit length.
        turn_rate : float or None
            Turn rate in deg per unit length.
        dls : float or None
            Dogleg severity.
        toolface : float or None
            Toolface angle in degrees.
        method : str or None
            Planning method code (e.g. ``"920"`` for minimum curvature).
        target : object or None
            Associated target, if any.
        tie_on : bool
            Whether this is the tie-on point.
        location : list or None
            Position as ``[x, y, z]``.
        """
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


def get_node(
    survey: "Survey", idx: int, interpolated: bool = False
) -> Node:
    """Extract a Node from a survey at a given index.

    Parameters
    ----------
    survey : Survey
        A Survey object.
    idx : int
        Index of the survey station.
    interpolated : bool
        Whether this station was interpolated.

    Returns
    -------
    Node
        A Node with position, vector, and MD from the survey station.
    """
    node = Node(
        pos=[survey.n[idx], survey.e[idx], survey.tvd[idx]],
        vec=survey.vec_nev[idx].tolist(),
        md=survey.md[idx],
        unit=survey.unit,
        nev=True,
        interpolated=interpolated
    )
    return node


def interpolate_mds(survey: "Survey", md: ArrayLike) -> "Survey":
    """
    Interpolates a survey at an array of measured depths, returning a new
    `welleng.survey.Survey` object that includes the original survey stations
    plus the requested (interpolated) measured depths.

    This is a vectorized equivalent of looping the scalar `interpolate_md`.
    Any requested depth that coincides with an existing survey station is
    dropped (the station is already present in the output).

    Parameters
    ----------
        survey: welleng.survey.Survey
            A survey object with at least two survey stations.
        md: (,n) list or array of floats
            The measured depths of the points of interest.

    Returns
    -------
        survey_interpolated: welleng.survey.Survey object
    """
    md = np.array(md)
    # drop requested depths that coincide with existing stations
    # (np.setdiff1d returns a sorted, unique array)
    md = np.setdiff1d(md, survey.md)

    assert md[0] >= survey.md[0], "The shortest md is not within the survey"
    assert md[-1] <= survey.md[-1], "The largest md is beyond the survey"

    # get the closest (preceding) survey stations
    idxs = np.searchsorted(survey.md, md, side="left") - 1
    idxs = np.clip(idxs, 0, len(survey.md) - 2)

    xs = md - survey.md[idxs]

    return _interpolate_surveys(survey, md, xs, idxs)


def interpolate_md(survey: "Survey", md: float) -> Optional["Survey"]:
    """
    Interpolates a survey at a given measured depth.
    """
    # get the closest survey stations
    idx = np.searchsorted(survey.md, md, side="left") - 1

    if idx >= len(survey.md) - 1:
        return None  # md is at or beyond the end of the survey

    if idx < 0:
        idx = 0  # type: ignore[assignment]
        x = 0

    else:
        x = md - survey.md[idx]

    return _interpolate_survey(survey, x=x, index=idx)  # type: ignore[arg-type]


def _interpolate_survey(
    survey: "Survey", x: float = 0, index: int = 0
) -> "Survey":
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
    index = _ensure_int_or_float(index, int)  # type: ignore[assignment]
    x = _ensure_int_or_float(x, float)

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

    mult = x / (survey.delta_md[index + 1])

    cov_nev = None if survey.cov_nev is None else (
        survey.cov_nev[index]
        + (
            np.full(shape=(1, 3, 3), fill_value=mult)
            * (survey.cov_nev[index+1] - survey.cov_nev[index])
        )
    ).reshape(3, 3)

    sh = survey.header
    sh.azi_reference = 'grid'

    s = Survey(
        md=np.array(
            [survey.md[index], survey.md[index] + x],
            dtype='float64'
        ),
        inc=np.array([survey.inc_rad[index], inc]),
        azi=np.array([survey.azi_grid_rad[index], azi]),
        cov_nev=(
            None if cov_nev is None
            else np.array([survey.cov_nev[index], cov_nev])  # type: ignore[index]
        ),
        start_xyz=np.array([survey.x, survey.y, survey.z]).T[index],
        start_nev=np.array([survey.n, survey.e, survey.tvd]).T[index],
        header=sh,
        deg=False,
        unit=sh.depth_unit,  # type: ignore[arg-type]
    )

    interpolated = False if any((
        x == 0,
        x == survey.md[index + 1] - survey.md[index]
     )) else True
    s.interpolated = [False, interpolated]

    return s


def _interpolate_surveys(
    survey: "Survey", md: np.ndarray, xs: np.ndarray, indexes: np.ndarray
) -> "Survey":
    """
    Interpolate multiple points at distances ``xs`` between their respective
    pairs of survey stations using minimum curvature. Vectorized equivalent
    of `_interpolate_survey`.

    Parameters
    ----------
        survey: welleng.Survey
            A survey object with at least two survey stations.
        md: (,n) array of floats
            The measured depths of the points of interest. Assumes that
            each value in md is not already in survey.md.
        xs: (,n) array of floats
            Lengths along the well path from each indexed survey station to
            perform the interpolation at. Must be less than the length to the
            next survey station.
        indexes: (,n) array of ints
            The indexes of the survey station from which to interpolate each
            x in xs.

    Returns
    -------
        survey_interpolated: welleng.survey.Survey object
            Note that an `interpolated` property is added indicating if the
            survey station is interpolated (True) or not (False).
    """
    assert indexes[-1] < len(survey.md) - 1, "Index is out of range"

    total_doglegs = survey.dogleg[indexes + 1]
    azi, inc = np.zeros(len(xs)), np.zeros(len(xs))

    # regions which are effectively straight (tangent sections)
    mask = np.where(total_doglegs < 1e-14)
    azi[mask] = survey.azi_grid_rad[indexes][mask]
    inc[mask] = survey.inc_rad[indexes][mask]

    # regions which are not straight
    mask = np.where(total_doglegs >= 1e-14)
    t1 = survey.vec_xyz[indexes][mask]
    t2 = survey.vec_xyz[indexes + 1][mask]

    dogleg = (
        xs[mask] * (total_doglegs[mask] / survey.delta_md[indexes + 1][mask])
    )

    t = (
        t1 * (
            np.sin(total_doglegs[mask] - dogleg)
            / np.sin(total_doglegs[mask])
        )[:, np.newaxis]
        + t2 * (np.sin(dogleg) / np.sin(total_doglegs[mask]))[:, np.newaxis]
    )

    # normalise tangent vectors
    t = t / np.linalg.norm(t, axis=-1).reshape(-1, 1)

    inc_azi = get_angles(t)
    inc[mask], azi[mask] = inc_azi[:, 0], inc_azi[:, 1]

    # merge the interpolated stations with the original stations and sort on md
    len_svy = len(survey.md)
    len_md = len(md)
    sorted_arr = np.zeros((3, len_svy + len_md))
    sorted_arr[0, 0:len_svy] = survey.md
    sorted_arr[0, len_svy:] = md
    sorted_arr[1, 0:len_svy] = survey.inc_rad
    sorted_arr[1, len_svy:] = inc
    sorted_arr[2, 0:len_svy] = survey.azi_grid_rad
    sorted_arr[2, len_svy:] = azi

    sorted_arr = sorted_arr[:, np.argsort(sorted_arr[0, :])]

    sh = survey.header
    sh.azi_reference = 'grid'

    survey_interpolated = Survey(
        md=sorted_arr[0, :],
        inc=sorted_arr[1, :],
        azi=sorted_arr[2, :],
        start_xyz=survey.start_xyz,
        start_nev=survey.start_nev,
        header=sh,
        deg=False,
        unit=sh.depth_unit,  # type: ignore[arg-type]
        error_model=None
    )

    survey_interpolated.interpolated = ~np.isin(
        survey_interpolated.md, survey.md
    )

    # carry the wellbore radius from the preceding station and, if present,
    # linearly interpolate the covariance between stations (mirrors the scalar
    # `_interpolate_survey` covariance interpolation).
    i = -1
    radii = []
    cov_nev = []
    unit_cov_nev = 0
    for (station_md, is_interpolated) in zip(
        survey_interpolated.md,
        survey_interpolated.interpolated
    ):
        if not is_interpolated:
            i += 1
            if survey.cov_nev is not None:
                j = 1 if i < len(survey.md) - 1 else 0
                if j == 1:
                    delta_md = survey.md[i + j] - survey.md[i]
                    unit_cov_nev = (
                        survey.cov_nev[i + j] - survey.cov_nev[i]
                    ) / delta_md
                else:
                    unit_cov_nev = 0
        radii.append(survey.radius[i])
        if survey.cov_nev is not None:
            cov_nev.append(
                survey.cov_nev[i]
                + ((station_md - survey.md[i]) * unit_cov_nev)
            )

    survey_interpolated.radius = np.array(radii)
    if bool(cov_nev):
        survey_interpolated.cov_nev = np.array(cov_nev)
        survey_interpolated.cov_hla = NEV_to_HLA(
            survey_interpolated.survey_rad,
            survey_interpolated.cov_nev
        )

    return survey_interpolated


def _interpolate_pos_nev(
    survey: "Survey", x: float, index: int
) -> np.ndarray:
    """
    Lightweight position-only interpolation: returns the NEV [N, E, TVD]
    position at distance ``x`` from ``survey[index]`` without constructing
    a Survey object.  Used as the inner cost function for closest-point
    optimisations in clearance calculations.
    """
    if survey.dogleg[index + 1] == 0:
        inc2 = survey.inc_rad[index]
        azi2 = survey.azi_grid_rad[index]
    else:
        t1 = survey.vec_xyz[index]
        t2 = survey.vec_xyz[index + 1]
        total_dogleg = survey.dogleg[index + 1]
        dogleg = x * (total_dogleg / survey.delta_md[index + 1])
        t = (
            (math.sin(total_dogleg - dogleg) / math.sin(total_dogleg)) * t1
            + (math.sin(dogleg) / math.sin(total_dogleg)) * t2
        )
        t /= np.linalg.norm(t)
        inc2, azi2 = get_angles(t)[0]

    pos = np.array([survey.n[index], survey.e[index], survey.tvd[index]])
    step = min_curve_step(x, survey.inc_rad[index], survey.azi_grid_rad[index], inc2, azi2)
    return pos + step


def _horizontal_tangent_delta(
    u1: float, u2: float, alpha: float
) -> Optional[float]:
    """Subtended angle at which a minimum-curvature arc's tangent becomes
    horizontal (i.e. the arc's TVD turning point), or ``None`` if that does
    not occur in the open interval ``(0, alpha)``.

    This is the vertical specialisation (target-plane normal ``m = [0, 0, 1]``)
    of the *Turning Point* construction in Sawaryn & Thorogood (2005,
    SPE-84246-PA, Eq. 31): the well goes horizontal where the vertical
    component of the unit tangent vanishes. Along the arc the tangent is the
    SLERP ``t(d) = [sin(alpha - d) t1 + sin(d) t2] / sin(alpha)``, so its
    vertical component is zero when

        ``sin(alpha - d) * u1 + sin(d) * u2 = 0``

    i.e. ``tan(d) = -sin(alpha) * u1 / (u2 - cos(alpha) * u1)``, where ``u1``,
    ``u2`` are the vertical components of the start/end unit tangents. Because
    a dogleg is at most ``pi``, an arc has at most one such point, splitting it
    into (up to) two monotonic-TVD spans.
    """
    p_term = math.sin(alpha) * u1
    q_term = u2 - math.cos(alpha) * u1
    if abs(p_term) < 1e-15 and abs(q_term) < 1e-15:
        return None
    base = math.atan2(-p_term, q_term)
    for cand in (base, base + math.pi, base - math.pi):
        if 1e-12 < cand < alpha - 1e-12:
            return cand
    return None


def _arc_tvd_crossings(
    u1: float, u2: float, alpha: float, delta_md: float, dvert: float
) -> list:
    """Subtended angles in ``[0, alpha]`` at which a minimum-curvature arc
    reaches a target true vertical depth.

    Closed-form *Interpolation at a Plane* of Sawaryn & Thorogood (2005,
    SPE-84246-PA), Eqs. 25-27 and Eq. 1, specialised to a horizontal target
    plane (normal ``m = [0, 0, 1]``). ``u1``/``u2`` are the vertical components
    of the unit tangents at the start/end of the arc, ``alpha`` the subtended
    (dogleg) angle, ``delta_md`` the arc length and ``dvert`` the target TVD
    minus the arc-start TVD.

    Returns the 0, 1 or 2 real roots. The discriminant ``A**2 + B**2 - C**2``
    is guarded: a negative value means the arc never reaches the plane, and an
    empty list is returned rather than a NaN.
    """
    a = u1 * math.sin(alpha)
    b = u1 * math.cos(alpha) - u2
    c = dvert * alpha * math.sin(alpha) / delta_md + b
    disc = a * a + b * b - c * c
    if disc < -1e-12:
        return []
    disc = max(disc, 0.0)
    root = disc ** 0.5
    out = []
    for sign in (1.0, -1.0):
        d = 2.0 * math.atan2(a + sign * root, b + c)
        # bring into [0, 2*pi); valid solutions on the arc lie in [0, alpha]
        d %= (2 * math.pi)
        if d > alpha:
            if abs(d - 2 * math.pi) < 1e-7:  # tiny negative root wrapped high
                d = 0.0
            else:
                continue
        out.append(min(max(d, 0.0), alpha))
        if root == 0.0:  # tangent: the two roots coincide
            break
    return out


def _subarc_from_node_origin(
    survey: "Survey", node_origin: Node
) -> "Survey":
    """Two-station survey from ``node_origin`` to the survey station just
    ahead of it, so a TVD interpolation can be referenced to a previously
    interpolated point rather than to a survey station."""
    j = int(np.searchsorted(survey.md, node_origin.md, side="right"))  # type: ignore[call-overload]
    j = min(max(j, 1), len(survey.md) - 1)
    return Survey(
        md=[node_origin.md, survey.md[j]],
        inc=[node_origin.inc_rad, survey.inc_rad[j]],
        azi=[node_origin.azi_rad, survey.azi_grid_rad[j]],
        deg=False,
        start_nev=node_origin.pos_nev,  # type: ignore[arg-type]
    )


def interpolate_tvd(survey: "Survey", tvd: float, **kwargs: Any) -> list:
    """Interpolate a survey at a target true vertical depth.

    Reversal-robust: does *not* assume monotonic TVD. The survey is walked
    segment by segment; each minimum-curvature arc is split at its TVD turning
    point (where the well goes horizontal) into monotonic spans, and every
    crossing of the target TVD is solved for in closed form. **All** crossings
    are returned, sorted by measured depth.

    Method: Sawaryn & Thorogood (2005), "A Compendium of Directional
    Calculations Based on the Minimum Curvature Method" (SPE-84246-PA),
    *Interpolation at a Plane* (Eqs. 25-27 and Eq. 1) with the target plane
    horizontal, plus the *Turning Point* construction (Eq. 31) to segment each
    arc into monotonic-TVD spans. See also :func:`_arc_tvd_crossings` and
    :func:`_horizontal_tangent_delta`.

    Parameters
    ----------
    survey : Survey
        A Survey object.
    tvd : float
        The target true vertical depth.
    **kwargs
        node_origin : Node, optional
            Interpolate on the sub-arc that starts at this node (rather than a
            survey station), spanning to the next survey station. Used to
            reference the interpolation to a previously interpolated point.

    Returns
    -------
    list of Node
        Every crossing of ``tvd``, sorted by measured depth (normally a single
        element; an empty list if ``tvd`` is outside the well's TVD range).

    Notes
    -----
    Breaking change (welleng 0.15.0): this returns a ``list`` of Nodes instead
    of a single Node. On a monotonic well, ``interpolate_tvd(tvd)[0]`` recovers
    the previous single-crossing behaviour.
    """
    node_origin = kwargs.get('node_origin')
    if node_origin is not None:
        survey = _subarc_from_node_origin(survey, node_origin)

    tol_md = 1e-6
    tol_ang = 1e-9
    crossings = []  # (md, index, x, interpolated)
    n_stations = len(survey.md)

    for i in range(n_stations - 1):
        alpha = survey.dogleg[i + 1]
        delta_md = survey.delta_md[i + 1]
        if delta_md == 0:
            continue
        v1 = survey.tvd[i]
        v2 = survey.tvd[i + 1]

        if np.isnan(alpha) or alpha <= tol_ang:
            # straight segment: TVD is linear in MD (hold or tangent)
            dv = v2 - v1
            if abs(dv) <= tol_ang:
                # horizontal hold: constant TVD along the whole segment
                if abs(tvd - v1) <= tol_md:
                    crossings.append((survey.md[i], i, 0.0, False))
                continue
            frac = (tvd - v1) / dv
            if -1e-9 <= frac <= 1 + 1e-9:
                frac = min(max(frac, 0.0), 1.0)
                x = frac * delta_md
                interp = not (x <= tol_md or abs(x - delta_md) <= tol_md)
                crossings.append((survey.md[i] + x, i, x, interp))
            continue

        u1 = survey.vec_nev[i][2]
        u2 = survey.vec_nev[i + 1][2]

        # split the arc into monotonic-TVD spans at its turning point
        d_tp = _horizontal_tangent_delta(u1, u2, alpha)
        breaks = [0.0, alpha] if d_tp is None else [0.0, d_tp, alpha]

        for da, db in zip(breaks[:-1], breaks[1:]):
            va = v1 if da == 0.0 else _interpolate_pos_nev(
                survey, da / alpha * delta_md, i)[2]
            vb = v2 if db == alpha else _interpolate_pos_nev(
                survey, db / alpha * delta_md, i)[2]
            lo, hi = (va, vb) if va <= vb else (vb, va)
            if not (lo - tol_md <= tvd <= hi + tol_md):
                continue
            for d in _arc_tvd_crossings(u1, u2, alpha, delta_md, tvd - v1):
                if da - 1e-7 <= d <= db + 1e-7:
                    x = min(max(d / alpha * delta_md, 0.0), delta_md)
                    interp = not (
                        x <= tol_md or abs(x - delta_md) <= tol_md
                    )
                    crossings.append((survey.md[i] + x, i, x, interp))

    # sort by MD and de-duplicate coincident crossings (shared station nodes,
    # or a target grazing a turning point from both adjoining spans)
    crossings.sort(key=lambda cr: cr[0])
    nodes = []
    last_md = None
    for md, idx, x, interp in crossings:
        if last_md is not None and abs(md - last_md) <= tol_md:
            continue
        s = _interpolate_survey(survey, x=x, index=idx)
        nodes.append(get_node(s, 1, interpolated=interp))
        last_md = md

    return nodes


def slice_survey(
    survey: "Survey", start: int, stop: Optional[int] = None
) -> "Survey":
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
    # Removing this start + 2 code - define this explicitly when making call 
    # if stop is None:
    #     stop = start + 2
    md, inc, azi = survey.survey_rad[start:stop].T
    nevs = np.array([survey.n, survey.e, survey.tvd]).T[start:stop]
    n, e, tvd = nevs.T
    # vec = survey.vec[start:stop]

    # Handle `None` values:
    cov_hla = None if survey.cov_hla is None else survey.cov_hla[start:stop]
    cov_nev = None if survey.cov_nev is None else survey.cov_nev[start:stop]

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
        cov_nev=cov_nev,
        start_nev=[n[0], e[0], tvd[0]],
        deg=False,
        unit=survey.unit,
    )

    s.error_model = survey.error_model

    return s


def _ensure_int_or_float(val: Any, required_type: type) -> int | float:
    if isinstance(val, np.ndarray):
        val = val[0]

    return required_type(val)


class SplitSurvey:
    """Split a survey into upper and lower station pairs for interval calculations.

    Provides paired arrays of inclinations, azimuths, vectors, and doglegs
    for consecutive survey stations.
    """

    def __init__(
        self,
        survey: "Survey",
    ) -> None:
        self.md1, self.inc1, self.azi1 = survey.survey_rad[:-1].T
        self.md2, self.inc2, self.azi2 = survey.survey_rad[1:].T
        self.delta_azi = self.azi2 - self.azi1
        self.delta_inc = self.inc2 - self.inc1

        self.vec1_xyz = survey.vec_xyz[:-1]
        self.vec1_nev = get_nev(self.vec1_xyz)
        self.vec2_xyz = survey.vec_xyz[1:]
        self.vec2_nev = get_nev(self.vec2_xyz)
        self.dogleg = survey.dogleg[1:]


def get_circle_radius(survey: "Survey", **targets: Any) -> tuple:
    """Compute curvature circle centers and endpoints for each survey interval.

    Parameters
    ----------
    survey : Survey
        A Survey object.
    **targets
        Reserved for future target data support.

    Returns
    -------
    tuple of ndarray
        Tuple of (starts, ends) arrays representing circle center positions
        and their corresponding survey station positions.
    """
    # TODO: add target data to sections
    ss = SplitSurvey(survey)

    b1 = np.cross(ss.vec1_nev, survey.normals)
    b2 = np.cross(ss.vec2_nev, survey.normals)
    nev = np.column_stack([survey.n, survey.e, survey.tvd])

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


def get_sections(
    survey: "Survey", rtol: float = 1e-1, atol: float = 1e-1,
    dls_cont: bool = False, **targets: Any
) -> list:
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

    Returns
    -------
    sections : list of TurnPoint
        List of TurnPoint objects representing control points.
    """
    # it turns out that since the well is being split into "holds" and "turns"
    # that the method can always be "920", since even a hold can be expressed
    # as an [md, inc, azi]. This simplifies things greatly!

    METHOD = "920"  # the COMPASS method for minimum curvature

    # TODO: add target data to sections
    # ss = SplitSurvey(survey)

    # check for DLS continuity
    if not dls_cont:
        # dls_cont = [True] * (len(survey.dls) - 2)
        dls_cont = np.full(len(survey.dls) - 2, True)  # type: ignore[assignment]
    else:
        upper = np.around(survey.dls[1:-1], decimals=2)
        lower = np.around(survey.dls[2:], decimals=2)
        # dls_cont = [
        #     True if u == l else False
        #     for u, l in zip(upper, lower)
        # ]
        dls_cont = np.equal(upper, lower)  # type: ignore[assignment]

    continuous = np.all((  # type: ignore[arg-type]
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

    sections: list = []
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


def get_unit(unit: str) -> Optional[str]:
    """Normalize a unit string to ``'meters'`` or ``'feet'``.

    Parameters
    ----------
    unit : str
        Input unit string (e.g. ``'m'``, ``'meters'``, ``'ft'``, ``'feet'``).

    Returns
    -------
    str or None
        ``'meters'``, ``'feet'``, or None if unrecognized.
    """
    if unit in ['m', 'meters']:
        return 'meters'
    elif unit in ['ft', 'feet']:
        return 'feet'
    else:
        return None


def make_survey_header(data: dict) -> SurveyHeader:
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


def survey_to_df(survey: Survey) -> pd.DataFrame:
    """Convert a Survey object to a pandas DataFrame.

    Parameters
    ----------
    survey : Survey
        A Survey object.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for MD, inclination, azimuths, positions,
        DLS, toolface, build rate, and turn rate.
    """
    data = {
        'MD (m)': survey.md,
        'INC (deg)': survey.inc_deg,
        'AZI_GN (deg)': survey.azi_grid_deg,
        'AZI_TN (deg)': survey.azi_true_deg,
        'NORTHING (m)': survey.pos_nev[:, 0],
        'EASTING (m)': survey.pos_nev[:, 1],
        'TVDSS (m)': survey.pos_nev[:, 2],
        'X (m)': survey.pos_xyz[:, 0],
        'Y (m)': survey.pos_xyz[:, 1],
        'Z (m)': survey.pos_xyz[:, 2],
        'DLS (deg/30m)': survey.dls,
        'TOOLFACE (deg)': np.degrees(survey.toolface + 2 * np.pi) % 360,
        'BUILD RATE (deg)': np.nan_to_num(survey.build_rate, nan=0.0),
        'TURN RATE (deg)': np.nan_to_num(survey.turn_rate, nan=0.0)
    }

    df = pd.DataFrame(data)

    return df


def export_csv(  # type: ignore[return]  # untyped pandas -> Optional[Any] false-positives the missing-return check
    survey: "Survey", filename: Optional[str], tolerance: float = 0.1,
    dls_cont: bool = False, decimals: int = 3, **kwargs: Any
) -> Optional[pd.DataFrame]:
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
    comments_str = ''.join(comments)

    np.savetxt(
        filename,  # type: ignore[arg-type]  # filename is not None on this path (the None case returns a DataFrame above)
        data,
        delimiter=',',
        fmt=f"%.{decimals}f",
        header=headers,
        comments=comments_str
    )


def get_data(tol: float, survey: "Survey", dls_cont: bool) -> np.ndarray:
    """Extract control-point data from a survey at a given tolerance.

    Parameters
    ----------
    tol : float
        Tolerance for section boundary detection (used as rtol and atol).
    survey : Survey
        A Survey object.
    dls_cont : bool
        Whether to check DLS continuity between sections.

    Returns
    -------
    ndarray
        Array of shape (n, 10) with MD, inc, azi, N, E, TVD, DLS,
        toolface, build rate, and turn rate for each control point.
    """
    rtol = atol = tol

    sections = survey._get_sections(rtol=rtol, atol=atol, dls_cont=dls_cont)

    rows = [[
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

    data = np.vstack(rows[1:])

    return data


def func(
    x0: float, survey: "Survey", dls_cont: bool, tolerance: float
) -> float:
    """Objective function for optimizing control-point tolerance in export_csv.

    Parameters
    ----------
    x0 : float
        Current tolerance value being optimized.
    survey : Survey
        The original Survey object.
    dls_cont : bool
        Whether to check DLS continuity.
    tolerance : float
        Target positional tolerance for the endpoint.

    Returns
    -------
    float
        Absolute difference between the target tolerance and the maximum
        endpoint position error.
    """
    data = get_data(x0, survey, dls_cont)

    md, inc, azi, n, e, tvd, dls, tf, br, tr = data.T
    nev = np.column_stack([survey.n, survey.e, survey.tvd])

    s = Survey(
        md=md,
        inc=inc,
        azi=azi,
        start_nev=nev[0],
        header=survey.header
    )

    s_nev = np.column_stack([s.n, s.e, s.tvd])

    diff = abs(
        tolerance - np.amax(np.absolute(s_nev[-1] - nev[-1]))
    )

    return diff


def _remove_duplicates(
    md: ArrayLike, inc: ArrayLike, azi: ArrayLike, decimals: int = 4
) -> np.ndarray:
    arr = np.column_stack([md, inc, azi])
    upper = arr[:-1]
    lower = arr[1:]

    temp = np.vstack((
        upper[0],
        lower[lower[:, 0].round(decimals) != upper[:, 0].round(decimals)]
    ))

    return temp.T


def from_connections(
    section_data: Any, step: Optional[float] = None,
    survey_header: Optional[SurveyHeader] = None,
    start_nev: ArrayLike = [0., 0., 0.],
    start_xyz: ArrayLike = [0., 0., 0.],
    start_cov_nev: Optional[ArrayLike] = None,
    radius: float = 10, deg: bool = False, error_model: Optional[str] = None,
    depth_unit: str = 'meters', surface_unit: str = 'meters',
    decimals: int | None = None
) -> "Survey":
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
    decimals: int (default=6)
        Round the md decimal when checking for duplicate surveys.

    Returns
    -------
    survey : Survey
        A Survey object constructed from the connections.
    """
    decimals = 6 if decimals is None else decimals
    assert isinstance(decimals, int), "decimals must be an int"

    if type(section_data) is not list:
        section_data = [section_data]

    # get reference mds
    mds_ref = []
    for s in section_data:
        mds_ref.extend([s.md1, s.md_target])

    section_data_interp = interpolate_well(section_data, step)  # type: ignore[arg-type]
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
            surface_unit=surface_unit,
            azi_reference="grid"  # since connections are typcially derived from pos
        )

    interpolated = np.array([False if m in mds_ref else True for m in md])

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
        unit=depth_unit,
        interpolated=interpolated
    )

    return survey


def interpolate_survey(
    survey: "Survey", step: float = 30, dls: float = 1e-8
) -> "Survey":
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

    s = np.column_stack([survey.md, survey.inc_rad, azi])

    s_upper = s[:-1]
    s_lower = s[1:]
    well: list = []

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
                    survey.cov_nev[i + j] - survey.cov_nev[i]  # type: ignore[index]
                )
                unit_cov_nev = (
                    delta_cov_nev / delta_md
                    if j == 1
                    else 0
                )
        radii.append(survey.radius[i])
        if survey.error_model is not None:
            cov_nev.append(
                survey.cov_nev[i]  # type: ignore[index]
                + (
                    (md - survey.md[i]) * unit_cov_nev
                )
            )
    survey_interpolated.radius = np.array(radii)
    if bool(cov_nev):
        survey_interpolated.cov_nev = np.array(cov_nev)
        survey_interpolated.cov_hla = NEV_to_HLA(
            survey_interpolated.survey_rad,
            survey_interpolated.cov_nev
        )

    return survey_interpolated


def get_node_tvd(
    survey: "Survey", node1: Node, node2: Node, tvd: float, node_origin: Node
) -> Optional[Node]:
    """Connect two nodes and interpolate to a target TVD.

    Parameters
    ----------
    survey : Survey
        The parent Survey object.
    node1 : Node
        Start node.
    node2 : Node
        End node (position is cleared and recomputed via Connector).
    tvd : float
        Target true vertical depth.
    node_origin : Node
        Origin node for the interpolation reference.

    Returns
    -------
    Node
        A Node at the target TVD between the two input nodes.
    """
    node2.pos_nev, node2.pos_xyz = None, None
    c = Connector(node1=node1, node2=node2, dls_design=1e-8)
    s = from_connections(c, step=None)
    # ``node1``/``node2`` bracket ``tvd`` (the caller checks this), so the
    # sub-arc has exactly one crossing; take it. ``interpolate_tvd`` now
    # returns a list (welleng 0.15.0).
    nodes = interpolate_tvd(s, tvd, node_origin=node_origin)

    return nodes[0] if nodes else None


def interpolate_survey_tvd(
    survey: "Survey", start: Optional[float] = None,
    stop: Optional[float] = None, step: float = 10
) -> "Survey":
    """Interpolate a survey at regular TVD intervals.

    Reversal-robust (welleng 0.15.0): builds regular TVD levels spanning the
    well's full TVD range and inserts a station at *every* crossing of each
    level (so a level revisited by a TVD reversal is represented at each pass),
    interleaved with the original survey stations. Crossings are found with the
    closed-form, turning-point-segmented :func:`interpolate_tvd` (Sawaryn &
    Thorogood 2005, SPE-84246-PA).

    Parameters
    ----------
    survey : Survey
        A Survey object.
    start : float or None
        TVD level anchor. Levels are placed at ``start + k * step``. Defaults
        to the first survey station's TVD.
    stop : float or None
        Upper TVD bound for the levels. Defaults to the well's maximum TVD.
    step : float
        TVD interval between interpolated levels.

    Returns
    -------
    Survey
        A Survey object with stations at regular TVD levels plus the original
        survey stations, ordered by measured depth.
    """
    anchor = survey.tvd[0] if start is None else start
    tvd_lo = min(float(np.min(survey.tvd)), anchor)
    tvd_hi = float(np.max(survey.tvd)) if stop is None else stop

    # regular TVD levels at anchor + k * step, within the well's TVD range
    k_lo = int(np.ceil((tvd_lo - anchor) / step))
    k_hi = int(np.floor((tvd_hi - anchor) / step))
    levels = [anchor + k * step for k in range(k_lo, k_hi + 1)]

    # collect nodes keyed by MD: original stations first, then level crossings
    nodes_by_md = {}
    for i in range(len(survey.md)):
        node = get_node(survey, i)
        nodes_by_md[round(float(node.md), 6)] = node  # type: ignore[arg-type]
    for level in levels:
        for node in interpolate_tvd(survey, level):
            nodes_by_md.setdefault(round(float(node.md), 6), node)

    ordered = [nodes_by_md[key] for key in sorted(nodes_by_md)]

    md, inc, azi, interpolated = np.array([
        [n.md, n.inc_rad, n.azi_rad, n.interpolated]
        for n in ordered
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


def project_ahead(
    pos: np.ndarray, vec: np.ndarray, delta_md: float, dls: float,
    toolface: float, md: float = 0.0
) -> Node:
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
    survey: "Survey",
    node_target: Node,
    dls_design: float = 3.0,
    delta_md: Optional[float] = None,
    dls: Optional[float] = None, toolface: Optional[float] = None,
    step: float = 30,
) -> "Survey":
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
        depth_unit=survey.header.depth_unit,  # type: ignore[arg-type]
        surface_unit=survey.header.surface_unit  # type: ignore[arg-type]
    )
    return survey_to_target


class SurveyData:
    """Lightweight container for combining survey data from multiple sections.

    Extracts the minimal data needed from Survey objects and provides methods
    to append additional sections and reconstruct a unified Survey.
    """

    def __init__(self, survey: "Survey") -> None:
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

    def append_survey(self, survey: "Survey") -> None:
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
                self.cov_nev.reshape(-1),  # type: ignore[union-attr]
                survey.cov_nev[1:].reshape(-1)  # type: ignore[index]
            )
        ).reshape(-1, 3, 3)
        self.cov_hla = np.hstack(
            (
                self.cov_hla.reshape(-1),  # type: ignore[union-attr]
                survey.cov_hla[1:].reshape(-1)  # type: ignore[index]
            )
        ).reshape(-1, 3, 3)
        self.radius = np.hstack((self.radius, survey.radius[1:]))

    def get_survey(self) -> "Survey":
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


def splice_surveys(surveys: list) -> "Survey":
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
