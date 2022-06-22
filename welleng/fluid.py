import math

from scipy import interpolate
from scipy.optimize import minimize

try:
    from numba import njit
    NJIT = True
except ImportError:
    NJIT = False
import numpy as np

# Define constants
A = [7.24032, -2.84383e-3, 2.75660e-5]
B = [8.63186, -3.31977e-3, 2.37170e-5]

# Gravitational constant
G = 0.052

# Weighting Material Density in ppg
WEIGHTING_MATERIAL_DENSITY = {
    'barite': 35.,
    'spe_11118': 24.
}


class DensityDiesel:
    def __init__(self):
        """
        An interpolation wrapper of the pressure, temperature and density diesel
        data provided in the SPE 11118 paper.
        """
        psia = np.array([15., 3_000., 7_000., 10_000., 12_500.])
        temp = np.array([100., 200., 300., 350.])

        psia_psia, temp_temp = np.meshgrid(psia, temp)

        rho = np.array([
            [6.9597, 7.0597, 7.1621, 7.2254, 7.2721],
            [6.6598, 6.7690, 6.8789, 6.9464, 6.9930],
            [6.3575, 6.4782, 6.5965, 6.6673, 6.7198],
            [6.2083, 6.3350, 6.4624, 6.5366, 6.5874]
        ])

        self.density_diesel = interpolate.interp2d(
            psia_psia, temp_temp, rho, kind='cubic'
        )

    def get_density(self, pressure, temperature):
        """
        Interpolate diesel density for given pressure and temperature using
        the lookup data provided in SPE 11118 paper.
        """
        density = self.density_diesel(
            pressure, temperature
        )

        return density


class Fluid:
    def __init__(
        self,
        fluid_density,
        reference_temp=32.,
        reference_pressure=0.,
        base_fluid_water_ratio=0.2,
        weighting_material='Barite'
    ):
        """
        Density profile calculated from SPE 11118 Mathematical Field Model
        Predicts Downhold Density Changes in Static Drilling Fluids by Roland
        R. Sorelle et al.

        This paper was written in oilfield units, so we'll convert inputs to
        ppg, ft, F and psi.

        Parameters
        ----------
        fluid_density: float
            The combined fluid density in ppg at reference conditions.
        reference_temp: float (default 32.0)
            The reference temperature in Fahrenheit
        reference_pressure: float (default 0.0)
            The reference pressure in psig.
        weighting_material: str
            The material being used to weight the drilling fluid (see the
            WEIGHTING_MATERIAL_DENSITY dictionary).
        """

        assert weighting_material.lower() in WEIGHTING_MATERIAL_DENSITY.keys()

        self.density_weighting_material = (
            WEIGHTING_MATERIAL_DENSITY.get(weighting_material.lower())
        )
        self.density_fluid_reference = fluid_density
        self.temp_reference = reference_temp
        self.base_fluid_water_ratio = base_fluid_water_ratio
        self.pressure_reference = reference_pressure

        if NJIT:
            self._get_coefficients = njit()(self._get_coefficients)
            self._func = njit()(self._func)
        else:
            self._get_coefficients = self._get_coefficients
            self._func = self._func

        self._get_density_base_fluids()
        self._get_volumes_reference()

    def _get_density_base_fluids(self):
        """
        Equation 1 and 2
        """
        def func(temperature, pressure, c):
            density = c[0] + c[1] * temperature + c[2] * pressure

            return density

        self.density_oil_reference = func(
            self.temp_reference, self.pressure_reference, A
        )
        self.density_water_reference = func(
            self.temp_reference, self.pressure_reference, B
        )

    def _get_volumes_reference(self):
        self.base_fluid_density_reference = (
            self.base_fluid_water_ratio * self.density_water_reference
            + (1 - self.base_fluid_water_ratio) * self.density_oil_reference
        )

        volume_weighting_material = (
            self.density_fluid_reference
            - self.base_fluid_density_reference
        ) / self.density_weighting_material

        volume_total = 1 + volume_weighting_material

        self.volume_water_reference_relative = (
            self.base_fluid_water_ratio / volume_total
        )
        self.volume_oil_reference_relative = (
            (1 - self.base_fluid_water_ratio) / volume_total
        )
        self.volume_weighting_material_relative = (
            volume_weighting_material / volume_total
        )

    @staticmethod
    def _get_coefficients(
        density_average, pressure_applied, temperature_top,
        fluid_thermal_gradient, A0, A1, A2, B0, B1, B2
    ):
        alpha_1 = (
            A0 + A1 * temperature_top + A2 * pressure_applied
        )
        alpha_2 = (
            A1 * fluid_thermal_gradient + G * A2 * density_average
        )
        beta_1 = (
            B0 + B1 * temperature_top + B2 * pressure_applied
        )
        beta_2 = (
            B1 * fluid_thermal_gradient + G * B2 * density_average
        )

        return alpha_1, alpha_2, beta_1, beta_2

    @staticmethod
    def _func(
        density_average, density_top, volume_water_relative,
        volume_oil_relative, depth, alpha_1, alpha_2, beta_1, beta_2
    ):
        if depth == 0:
            return density_top
        func = (
            (
                density_top * depth
                - (
                    volume_oil_relative * alpha_1 * density_average
                    / alpha_2
                )
                * math.log(
                    (alpha_1 + alpha_2 * depth) / alpha_1
                )
            ) / (depth * (1 - volume_water_relative - volume_oil_relative))
            - (
                volume_water_relative * beta_1 * density_average / beta_2
                * math.log(
                    (beta_1 + beta_2 * depth) / beta_1
                )
            ) / (depth * (1 - volume_water_relative - volume_oil_relative))
        )
        return func

    def _get_density(
        self, density_average, density_top, temperature_top,
        volume_water_relative, volume_oil_relative, pressure_applied, depth,
        fluid_thermal_gradient
    ):
        density_average = density_average[0]
        alpha_1, alpha_2, beta_1, beta_2 = self._get_coefficients(
            density_average, pressure_applied, temperature_top,
            fluid_thermal_gradient, A[0], A[1], A[2], B[0], B[1], B[2]
        )

        func = self._func(
            density_average, density_top, volume_water_relative,
            volume_oil_relative, depth, alpha_1, alpha_2, beta_1, beta_2
        )

        return abs(density_average - func)

    def get_density_profile(
        self,
        depth,
        temperature,
        pressure_applied=0.,
        density_bounds=(6., 25.)
    ):
        """
        Function that returns a density profile of the fluid, adjusted for
        temperature and compressibility and assuming that the fluid's reference
        parameters are the surface parameters.

        Parameters
        ----------
        depth: float or list or (n) array of floats
            The vertical depth of interest relative to surface in feet.
        temperature: float or list or (n) array of floats
            The temperature corresponding to the vertical depth of interest in
            Fahrenheit.
        pressure_applied: float (default=0.)
            Additional pressure applied to the fluid in psi.
        density_bounds: (2) tuple of floats (default=(6., 25.))
            Density bounds to constrain the optimization algorithm in ppg.
        """

        # Convert to (n) array to manage single float or list/array
        depth = np.array([depth]).reshape(-1)
        with np.errstate(invalid='ignore'):
            temperature_thermal_gradient = np.nan_to_num((
                temperature - self.temp_reference
            ) / depth
            )

        density_profile = [
            minimize(
                fun=self._get_density,
                x0=self.density_fluid_reference,
                args=(
                    self.density_fluid_reference,
                    self.temp_reference,
                    self.volume_water_reference_relative,
                    self.volume_oil_reference_relative,
                    pressure_applied,
                    d,
                    t,
                ),
                method='SLSQP',
                bounds=[density_bounds]
            ).x
            for d, t in zip(depth, temperature_thermal_gradient)
        ]

        return np.vstack(density_profile).reshape(-1).tolist()


def main():
    """
    An example of initiating a Fluid class and generating a density profile
    for the fluid for a range of depths and temperatures.
    """
    # Define the fluid
    fluid = Fluid(
        fluid_density=10.,  # ppg
        reference_temp=120.,  # Fahrenheit,
        weighting_material='SPE_11118',
        base_fluid_water_ratio=0.103,
    )

    # Override calculated volumes - I can't get the same values as the SPE
    # paper if I build the fluid. However, the fluid properties can be
    # overwritten if desired as indicated below:
    fluid.volume_water_reference_relative = 0.09
    fluid.volume_oil_reference_relative = 0.78
    fluid.volume_weighting_material_relative = 0.11

    depth = np.linspace(0, 10_000, 1001)
    temperature = np.linspace(120, 250, 1001)

    density_profile = fluid.get_density_profile(
        depth=depth,
        temperature=temperature
    )

    # Check we get the same answer as the SPE paper example
    assert round(density_profile[-1], 2) == 9.85

    # load dependencies for plotting results
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # construct plots
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

    fig.add_trace(go.Scatter(
        x=density_profile,
        y=depth,
        mode='lines',
        name='Density (ppg)',

    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=temperature,
        y=depth,
        mode='lines',
        name='Temperature (F)'
    ), row=1, col=2)
    fig.update_layout(
        title="Effect of Temperature and Compressibility on Mud Density",
        yaxis=dict(
            autorange='reversed',
            title="TVD (ft)",
            tickformat=",.0f"
        ),
        showlegend=False
    )
    fig.update_xaxes(
        title_text="Density (ppg)",
        tickformat=".2f",
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Temperature (\xb0F)",
        tickformat=".0f",
        row=1, col=2
    )
    fig.show()


if __name__ == '__main__':
    main()
