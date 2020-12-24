from welleng.io import import_iscwsa_collision_data
import json

# Short script to extract and serialize the well data from the
# ISCWSA clearance model spreadsheet. The intent is to save on test time
# versus having to extract the data from Excel workbook each time.
# This code will remain in the repo to demonstrate where the data is
# derived and how it's been extracted.
filename = (
    "reference/standard-set-of-wellpaths"
    "-for-evaluating-clearance-scenarios-r4-17-may-2017.xlsx"
)
well_data = import_iscwsa_collision_data(filename)

well_ref_params = dict(
    Latitude=60.000000,
    BTotal=50000.00,
    Dip=70.00,
    Declination=0.00,
    Convergence=0.0,
    G=9.80665
)

well_data['well_ref_params'] = well_ref_params

# serialize data into file:
json.dump(
    well_data,
    open("test/test_data/clearance_iscwsa_well_data.json", 'w')
)
