import json

from welleng.io import import_iscwsa_collision_data
from welleng.survey import SurveyHeader

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

for w in well_data['wells']:
    sh = SurveyHeader(
        name=w,
        latitude=60.,
        b_total=50000.,
        dip=70.,
        declination=0.,
        convergence=0.,
        azi_reference="grid"
    )
    well_data['wells'][w]['header'] = vars(sh)

# serialize data into file:
json.dump(
    well_data,
    open("test/test_data/clearance_iscwsa_well_data.json", 'w')
)
