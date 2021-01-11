class Fluid:
    def __init__(
        self,
        density,
        unit='metric'
    ):
        """
        Parameters
        ----------
            density: float
                The fluid density in either SG or ppg.
            unit: string (default: 'metric')
                The unit system of the input parameters, either "metric" or
                "imperial" for SG or ppg respectively.
        """

        assert unit in ["imperial", "metric"]

        if unit == "imperia":
            self.density_imperial = density
            self.density_metric = density * 8.33
        else:
            self.density_metric = density
            self.density_imperial = density / 8.33
