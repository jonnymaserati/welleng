import numpy as np
from welleng.utils import linear_convert


class Casing:
    def __init__(
        self,
        size,
        weight,
        outer_diameter,
        drift,
        inner_diameter=None,
        wall_thickness=None,
        grade=None,
        unit='imperial',
        yield_strength={
            'min': None,
            'max': None
        },
        tensile_strength=None,
    ):
        assert unit in ["imperial", "metric"], "Invalid unit"
        assert inner_diameter is not None or wall_thickness is not None, (
            "Provide either wall_thickness or inner_diameter"
        )
        assert inner_diameter is None or wall_thickness is None, (
            "Provide either wall_thickness or inner_diameter"
        )

        self.size = size
        self = get_weights(self, weight, unit)
        self = get_diameters(
            self, wall_thickness, inner_diameter, outer_diameter, drift,
            unit
        )


# class ToolJoint:
#     def __init__(
#         self,
#         type,
#         outer_diameter,
#         inner_diameter,
#         grade=None,
#         unit="imperial",
#         pin_space=None,
#         box_space=None,
#     ):

# class Scene:
#     def __init__(
#         self,
#         survey
#     ):
#         self.survey = survey
#         self.scene = []
#         self.assembly = []

#     def add_hole_section(
#         self,
#         obj,
#         depth_from,
#         depth_to,
#     ):


class DrillPipe:
    def __init__(
        self,
        size,
        weight,
        outer_diameter,
        drift=None,
        inner_diameter=None,
        wall_thickness=None,
        grade=None,
        upset_type=None,
        range=None,
        tool_joint=None,
        density=7.8 * 8.33,
        unit='imperial',
    ):
        assert unit in ["imperial", "metric"], "Invalid unit"
        assert inner_diameter is not None or wall_thickness is not None, (
            "Provide either wall_thickness or inner_diameter"
        )
        assert inner_diameter is None or wall_thickness is None, (
            "Provide either wall_thickness or inner_diameter"
        )
        self.size = size
        self = get_weights(self, weight, unit)
        self = get_diameters(
            self, wall_thickness, inner_diameter, outer_diameter, drift,
            unit
        )
        self = get_densities(self, density, unit)


def get_diameters(obj, wall, inner, outer, drift, unit, factor=25.4):
    if unit == 'imperial':
        obj.od_imperial = outer
        if wall is None:
            obj.id_imperial = inner
            obj.wall_thickness_imperial = outer - inner
        else:
            obj.wall_thickness_imperial = wall
            obj.id_imperial = outer - wall
        obj.drift_imperial = drift
        (
            obj.wall_thickness_metric,
            obj.id_metric,
            obj.od_metric,
            obj.drift_metric
        ) = (
            linear_convert(
                [
                    obj.wall_thickness_imperial,
                    obj.id_imperial,
                    outer,
                    drift
                ], factor
            )
        )
    else:
        obj.od_metric = outer
        if wall is None:
            obj.id_metric = inner
            obj.wall_thickness_metric = outer - inner
        else:
            obj.wall_thickness_metric = wall
            obj.id_metric = outer - wall
        obj.drift_metric = drift
        (
            obj.wall_thickness_imperial,
            obj.id_imperial,
            obj.od_imperial,
            obj.drift_imperial
        ) = (
            linear_convert(
                [
                    obj.wall_thickness_metric,
                    obj.id_metric,
                    outer,
                    drift
                ], 1/factor
            )
        )

    return obj


def get_weights(obj, weight, unit, factor=1.488164):
    if unit == 'imperial':
        obj.weight_imperial = weight
        obj.weight_metric = linear_convert(weight, factor)
    else:
        obj.weight_metric = weight
        obj.weight_imperial = linear_convert(weight, 1/factor)

    return obj


def get_densities(obj, density, unit):
    if unit == 'metric':
        obj.density_metric = density
        obj.density_imperial = linear_convert(density, 8.33)
    else:
        obj.density_imperial = density
        obj.density_metric = linear_convert(density, 1/8.33)

    return obj
