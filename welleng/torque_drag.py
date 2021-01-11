import numpy as np
from welleng.utils import linear_convert

# This model is EXPERIMENTAL, I wrote it more as a proof of concept for
# well trajectory optimization. This model has not been validated against
# know good data and I've not cross-checked the formulas against those
# referenced in the Pro Well Plan module that I based this on.


class TorqueDrag:
    def __init__(
        self,
        survey,
        assembly,
        fluid,
        torque_on_bit=0.,
        weight_on_bit=0.,
        overpull=0.,
        fixed_depth=None,
        unit='metric'
    ):
        """
        Parameters
        ----------
            weight_on_bit: float
                If unit is 'metric' then in tonnes, else if 'imperial' then
                kips.
            torque_on_bit: float
                If unit is 'metric' then in kNm, else if 'imperial'
                then kft.lbs.
        """
        self.G = 9.81
        self.survey = survey
        self.assembly = assembly
        self.fluid = fluid
        self = get_wob(self, weight_on_bit, overpull, unit)
        self = get_tob(self, torque_on_bit, unit)

        # add BHA profiles
        self._get_buoyancy()
        self.assembly_weight = np.flip(np.full_like(
            self.survey.radius, self.assembly.weight_metric * self.G
        ) * self.survey.delta_md)
        # self.assembly_weight = (
        #     self.assembly.density_metric * 1000 * self.G * np.pi
        #     * self.assembly_area * np.flip(self.survey.delta_md)
        # )
        self.buoyancy_factor = (
            (self.assembly.density_metric - self.fluid.density_metric)
            / self.assembly.density_metric
        )
        self.assembly_bouyed_weight = (
            self.assembly_weight * self.buoyancy_factor
            # self.assembly_weight * self.buoyancy
        )

        self._get_delta_angles()
        self._get_coeffs()
        self._get_loads()

    def _get_loads(self):
        self.drag_rih = [self.wob_metric]
        self.torque_rih = [self.tob_metric]
        self.drag_neutral = [0.]
        self.torque_neutral = [0.]
        self.drag_pooh = [self.overpull_metric]
        self.torque_pooh = [0.]
        for i, params in enumerate(zip(
            self.delta_inc, self.delta_azi, self.inc_avg, self.A, self.B,
            self.C, self.friction_coeff, self.string_radius
        )):
            self._get_rih(params)
            self._get_neutral(params)
            self._get_pooh(params)

        self._cleanup_loads()

    def _get_delta_angles(self):
        self.delta_inc, self.delta_azi = (
            np.vstack(
                (
                    np.array([0., 0.]),
                    np.diff(
                        np.array([
                            self.survey.inc_rad,
                            self.survey.azi_grid_rad
                        ]).T[::-1], axis=0
                    )
                )
            )
        ).T
        self.inc_avg = np.flip(self.survey.inc_rad) - self.delta_inc / 2

    def _get_coeffs(self):
        # These coeffs are used in all the calcs... so just calculate them
        # once (efficiently with numpy) and serve them to the helper functions.
        self.A = self.delta_azi * np.sin(self.inc_avg)
        self.B = (
            self.assembly_bouyed_weight
            * np.sin(self.inc_avg)
        )
        self.C = (
            self.assembly_bouyed_weight
            * np.cos(self.inc_avg)
        )
        self.friction_coeff = np.flip(self.survey.friction_coeff)

        # TODO: evolve assembly into a string of components with a radius
        # profile
        self.string_radius = np.flip(np.full_like(
            self.survey.inc_rad,
            self.assembly.id_metric / (2 * 1000)  # convert to radius in meters
        ))

    def _get_rih(self, params):
        delta_inc, delta_azi, inc_avg, a, b, c, fc, r = params

        # calculate drag
        fn = (
            (self.drag_rih[-1] * a) ** 2
            + (self.drag_rih[-1] * delta_inc + b) ** 2
        ) ** 0.5

        delta_ft = (
            c - fc * fn
        )

        ft = self.drag_rih[-1] + delta_ft

        self.drag_rih.append(ft)

        # calculate torque
        delta_t = fc * fn * r

        t = self.torque_rih[-1] + delta_t

        self.torque_rih.append(t)

    def _get_neutral(self, params):
        delta_inc, delta_azi, inc_avg, a, b, c, fc, r = params

        # calculate drag
        fn = (
            (self.drag_neutral[-1] * a) ** 2
            + (self.drag_neutral[-1] * delta_inc + b) ** 2
        ) ** 0.5

        delta_ft = (
            c
        )

        ft = self.drag_neutral[-1] + delta_ft

        self.drag_neutral.append(ft)

        # calculate torque
        delta_t = fc * fn * r

        t = self.torque_neutral[-1] + delta_t

        self.torque_neutral.append(t)

    def _get_pooh(self, params):
        delta_inc, delta_azi, inc_avg, a, b, c, fc, r = params

        # calculate drag
        fn = (
            (self.drag_pooh[-1] * a) ** 2
            + (self.drag_pooh[-1] * delta_inc + b) ** 2
        ) ** 0.5

        delta_ft = (
            c + fc * fn
        )

        ft = self.drag_pooh[-1] + delta_ft

        self.drag_pooh.append(ft)

        # calculate torque
        delta_t = fc * fn * r

        t = self.torque_pooh[-1] + delta_t

        self.torque_pooh.append(t)

    def _cleanup_loads(self):
        loads = np.flip(np.array([
            self.drag_rih,
            self.torque_rih,
            self.drag_neutral,
            self.torque_neutral,
            self.drag_pooh,
            self.torque_pooh,
        ]) / 1000, axis=-1)

        (
            self.drag_rih,
            self.torque_rih,
            self.drag_neutral,
            self.torque_neutral,
            self.drag_pooh,
            self.torque_pooh,
        ) = loads

    def _get_buoyancy(self):
        self._get_areas()
        self.buoyancy = (
            1 - (self.fluid.density_metric * self.annulus_area)
            / (self.assembly.density_metric * (
                self.annulus_area - self.assembly_area
            ))
        )

    def _get_areas(self):
        self.annulus_area = np.flip(
            np.pi * (
                self.survey.radius ** 2
                - (self.assembly.od_metric / (2 * 1000)) ** 2
            )
        )
        self.assembly_area = np.flip(
            np.pi * (
                (self.assembly.od_metric / (2 * 1000)) ** 2
                - (self.assembly.id_metric / (2 * 1000)) ** 2
            )
        )


def get_wob(obj, wob, op, unit, factor=2.204622622):
    if unit == 'imperial':
        obj.wob_imperial = wob * 1000
        obj.overpull_imperial = op * 1000
        obj.wob_metric, obj.overpull_metric = linear_convert(
            [wob, op], 1/factor
        )
    else:
        obj.wob_metric = wob * 1000
        obj.overpull_metric = op * 1000
        obj.wob_imperial, obj.overpull_imperial = linear_convert(
            [wob, op], factor
        )

    return obj


def get_tob(obj, tob, unit, factor=1.36):
    if unit == 'imperial':
        obj.tob_imperial = tob * 1000
        obj.tob_metric = linear_convert(
            tob, factor
        )
    else:
        obj.tob_metric = tob * 1000
        obj.tob_imperial = linear_convert(
            tob, 1/factor
        )

    return obj
