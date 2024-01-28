from welleng.survey import Survey, make_survey_header
from welleng.clearance import IscwsaClearance
import numpy as np
import json

"""
Test that the ISCWSA clearance model is working within a defined tolerance,
testing against the ISCWSA standard set of wellpaths for evaluating clearance
scenarios using the MWD Rev4 error model.
"""

# Read well and validation data
filename = (
    "tests/test_data/clearance_iscwsa_well_data.json"
)
data = json.load(open(filename))


def generate_surveys(self, data=data):
    # Generate surveys for imported wells
    surveys = {}

    for well in data['wells']:
        sh = make_survey_header(data["wells"][well]["header"])

        if well == "Reference well":
            radius = 0.4572
        else:
            radius = 0.3048

        s = Survey(
            md=data["wells"][well]["MD"],
            inc=data["wells"][well]["IncDeg"],
            azi=data["wells"][well]["AziDeg"],
            n=data["wells"][well]["N"],
            e=data["wells"][well]["E"],
            tvd=data["wells"][well]["TVD"],
            radius=radius,
            header=sh,
            error_model="ISCWSA MWD Rev4",
            start_xyz=[
                data["wells"][well]["E"][0],
                data["wells"][well]["N"][0],
                data["wells"][well]["TVD"][0]
                ],
            start_nev=[
                data["wells"][well]["N"][0],
                data["wells"][well]["E"][0],
                data["wells"][well]["TVD"][0]
                ],
            deg=True,
            unit="meters"
        )
        surveys[well] = s

    return surveys


def test_minimize_sf(data=data):
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]
    offset = surveys["09 - well"]

    result = IscwsaClearance(reference, offset, minimize_sf=False)
    result_min = IscwsaClearance(reference, offset, minimize_sf=True)

    idx = np.where(result_min.ref.interpolated == False)  # noqa E712

    # Check that interpolated survey is not corrupted
    for attr in [
        'azi_grid_rad', 'azi_mag_rad', 'azi_true_rad', 'cov_hla', 'cov_nev',
        'pos_nev', 'pos_xyz', 'md', 'radius'
    ]:
        assert np.allclose(
            getattr(result.ref, attr), getattr(result_min.ref, attr)[idx]
        )

        pass

    for attr in [
        'Rr', 'calc_hole', 'distance_cc', 'eou_boundary',
        'eou_separation', 'hoz_bearing', 'idx', 'masd', 'off_cov_hla',
        'off_cov_nev', 'off_delta_hlas', 'off_delta_nevs', 'off_pcr',
        'ref_cov_hla', 'ref_cov_nev', 'ref_delta_hlas', 'ref_delta_nevs',
        'ref_nevs', 'ref_pcr', 'sf', 'wellbore_separation'
    ]:
        # `toolface_bearing` and `trav_cyl_azi_deg` are a bit unstable when
        # well paths are parallel.

        assert np.allclose(
            getattr(result, attr), getattr(result_min, attr)[idx],
            rtol=1e-01, atol=1e-02
        )

        pass


def test_clearance_iscwsa(data=data, rtol=1e-02, atol=1e-03):
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]

    # Perform clearance checks for each survey
    for well in surveys:
        if well != "09 - well":
            continue
        if well == "Reference well":
            continue
        else:
            offset = surveys[well]
            # skip well 10
            if well in ["10 - well"]:
                continue
            else:
                for b in [False, True]:
                    result = IscwsaClearance(reference, offset, minimize_sf=b)
                    assert np.allclose(
                        result.sf[np.where(result.ref.interpolated == False)],  # noqa E712
                        np.array(data["wells"][well]["SF"]),
                        rtol=rtol, atol=atol
                    )

    pass
