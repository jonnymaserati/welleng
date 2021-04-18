from welleng.fluid import Fluid

def test_fluid_density_profile():
    """
    Test that the get_density_profile method in the Fluid class returns the
    same value as the example in the SPE 11118 paper.
    """
    fluid = Fluid(
        fluid_density=10.,  # ppg
        reference_temp=120.,  # Fahrenheit,
        weighting_material='SPE_11118',
        base_fluid_water_ratio=0.103,
    )

    # Override properties to align with ones provided in the example.
    fluid.volume_water_reference_relative = 0.09
    fluid.volume_oil_reference_relative = 0.78
    fluid.volume_weighting_material_relative = 0.11

    density_profile = fluid.get_density_profile(
        depth=10_000.,
        temperature=250.
    )

    assert round(density_profile[-1], 2) == 9.85
