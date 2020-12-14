import numpy as np
from ..survey import get_sections

class WellPlan:
    def __init__(
        self,
        survey,
        location_type,
        plan_method,
        plan_name,
        parent_name,
        depth_unit="meters",
        surface_unit="meters",
        dirty_flag=3,
        sidetrack_id=0,
        extension="0.00000",
        **targets
    ):

        LOCATIONS = [
            "unknown",
            "surface",
            "ow_sidetrack",
            "wp_sidetrack",
            "lookahead",
            "ow_well",
            "complex_extension",
            "site",
            "site_based_plan",
            "compass_well"
        ]

        METHODS = [
            "curve_only",
            "curve_hold",
            "opt_align"
        ]

        UNITS = [
            "feet",
            "meters"
        ]

        assert location_type in LOCATIONS, "location_type not in LOCATIONS"
        assert plan_method in METHODS, "plan_method not in METHODS"
        assert len(plan_name) < 21, "plan_name too long (max length 20)"
        assert len(parent_name) < 21, "parent_name too long (max length 20)"
        assert surface_unit in UNITS, "surface_units must be feet or meters"
        assert depth_unit in UNITS, "depth_units must be feet or meters"

        self.survey = survey
        self.location = str(LOCATIONS.index(location_type))
        self.surface_unit = surface_unit
        self.depth_unit = depth_unit
        self.method = str(METHODS.index(plan_method))
        self.flag = str(dirty_flag)
        self.st_id = str(sidetrack_id).rjust(8)
        self.plan_name = str(plan_name).ljust(21)
        self.parent_name = str(parent_name).ljust(20)
        self.dls = f"{str(np.around(self.survey.dls[1]))[:5]}".rjust(5)
        self.kickoff_dls = f"{survey.dls[1]:.2f}".rjust(7)
        self.extension = str(extension).rjust(8)
        self.targets = targets

        self._get_unit()

        self.doc = []

        self._make_header()
        # self._make_tie_on(target=0)
        self._make_wellplan()

    def _get_unit(self):
        if self.depth_unit == "meters":
            if self.surface_unit == "meters":
                self.unit = 2
            else:
                self.unit = 3
        elif self.depth_unit == "feet":
            if self.surface_unit == "feet":
                self.unit = 1
            else:
                self.unit = 4

    def _add_plan(
        self,
        section
    ):
        self.doc.append(
            f"P:"
            f"{str(section.md)[:8].rjust(8)}"
            f" {str(section.azi)[:7].rjust(7)}"
            f" {str(section.inc)[:7].rjust(7)}"
            f" {str(section.build_rate)[:7].rjust(7)}"
            f" {str(section.turn_rate)[:7].rjust(7)}"
            f" {str(np.around(section.dls, decimals=1))[:7].rjust(7)}"
            f" {str(np.degrees(section.toolface))[:7].rjust(7)}"
            f" {str(section.method).rjust(4)}"
            f" {str(section.target).rjust(10)}"
        )
        self.doc.append(
            f"L:"
            f"{str(section.x)[:13].rjust(13)}"
            f"{str(section.y)[:13].rjust(13)}"
            f"{str(section.z)[:11].rjust(11)}"
        )


    def _make_wellplan(
        self,
    ):
        sections = get_sections(self.survey)

        for s in sections:
            self._add_plan(s)
    
    def _make_header(self):
        self.doc.append(
            f"DEPTH {self.unit}"
        )

        if self.targets:
            self.doc.append(
                "TARGETS:"
            )
            # TODO: add function to retrieve and list target data and implement
            # a Target class.

        self.doc.append(
            "WELLPLANS:",
        )
        self.doc.append(
            (
                f"W:{self.location}{self.unit}{self.method}{self.flag}   0   0"
                f"{self.st_id} {self.plan_name} {self.parent_name} {self.dls}"
                f"{self.extension}{self.kickoff_dls}"
            )
        )

    # I think this is redundent as it's the same line as the first section
    def _make_tie_on(self, target):
        if self.location == "unknown":
            self.doc.append(
                f"P:0.000000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000    0"
            )
        else:
            self.doc.append((
                f"P:"
                f"{str(self.survey.md[0])[:8].rjust(8)}"
                f" {str(self.survey.azi_deg[0])[:7].rjust(7)}"
                f" {str(self.survey.inc_deg[0])[:7].rjust(7)}"
                f" {str(0.00000).rjust(7)}" # build rate
                f" {str(0.00000).rjust(7)}" # turn rate
                f" {str(0.00000).rjust(7)}" # DLS
                f" {str(np.degrees(self.survey.toolface[0]))[:7].rjust(7)}"
                f" {self.method.rjust(4)}"
                f" {str(target).rjust(10)}"
            ))

            self.doc.append(
                f"L:"
                f"{str(self.survey.x)[:13].rjust(13)}"
                f"{str(self.survey.y)[:13].rjust(13)}"
                f"{str(self.survey.z)[:11].rjust(11)}"
            )
