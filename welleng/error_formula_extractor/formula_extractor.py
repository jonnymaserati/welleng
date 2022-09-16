from welleng.error_formula_extractor.enums import Propagation, VectorType
from welleng.error_formula_extractor.formula_utils import function_builder
from welleng.error_formula_extractor.models import ErrorTerm, SurveyToolErrorModel
from welleng.exchange.edm import EDM

DEGREE_TO_RAD = 0.0174533


class ErrorFormulaExtractor:

    def __init__(self, subject_edm: EDM, well_name: str):
        self.edm = subject_edm
        self.well_id = subject_edm.get_wells().get(well_name).get('well_id')

        # extract wellbore id
        self.wellbore_id = subject_edm.get_attributes(
            tags=["CD_WELLBORE"],
            attributes={"well_id": self.well_id}
        )['CD_WELLBORE'][0]["wellbore_id"]

        # extract survey headers
        survey_headers = subject_edm.get_attributes(
            tags=["CD_DEFINITIVE_SURVEY_HEADER"],
            attributes={"wellbore_id": self.wellbore_id}
        )['CD_DEFINITIVE_SURVEY_HEADER']

        # parse survey headers to extract actual survey headers
        self.survey_headers = [header for header in survey_headers if header["phase"] == "ACTUAL"]

        # if actual survey headers are not available for the given well, use plan survey headers with maximum md.
        if not self.survey_headers:
            plan_headers = [header for header in survey_headers if header["phase"] == "PLAN"]
            md_list = [header["bh_md"] for header in plan_headers if header["phase"] == "PLAN"]
            idx_max_md = md_list.index(max(md_list))
            self.survey_headers = [plan_headers[idx_max_md]]

        # extract survey programs for the given survey header
        survey_programs = subject_edm.get_attributes(
            tags=["CD_SURVEY_PROGRAM"],
            attributes={"def_survey_header_id": self.survey_headers[0]["def_survey_header_id"]}
        )['CD_SURVEY_PROGRAM']

        # extract datum elevation for the given survey header
        datum_data = subject_edm.get_attributes(
            tags=["CD_DATUM"],
            attributes={"well_id": self.well_id}
        )['CD_DATUM'][0]
        datum_elevation = float(datum_data["datum_elevation"])

        # Adjust the depths in the survey tools for the datum elevation
        params_to_adjust = ["md_base", "md_top"]
        for program in survey_programs:
            for param in params_to_adjust:
                program[param] = float(program[param]) + datum_elevation

        self.survey_programs = sorted(survey_programs, key=lambda d: int(d['sequence_no']))

        self.survey_tools = None

        self.run()

    def run(self):

        self.initiate_survey_tools()

        for tool in self.survey_tools:
            self.extract_functions(tool)

    def initiate_survey_tools(self):

        survey_tools = []
        for program_no, program in enumerate(self.survey_programs):

            tool_id = program["survey_tool_id"]

            # Get the survey tool information from the EDM file.
            survey_tool = self.edm.get_attributes(
                tags=["CD_SURVEY_TOOL"],
                attributes={"survey_tool_id": tool_id}
            )

            if not survey_tool:
                raise KeyError(f"survey tool {tool_id} is not available in the EDM file")

            # Since the survey programs is a list of sorted surveys, start_depth = md_min of current survey header
            # the end_depth = md_min of the next survey header or
            # end_depth = md_max of the current survey header if it is the last survey.
            start_depth = float(program["md_top"])
            if program_no == len(self.survey_programs) - 1:
                end_depth = float(self.survey_programs[-1]["md_base"])
            else:
                end_depth = float(self.survey_programs[program_no + 1]["md_top"])

            survey_tool = survey_tool["CD_SURVEY_TOOL"][0]
            survey_tools.append(
                SurveyToolErrorModel(
                    survey_tool_id=tool_id,
                    survey_tool_name=survey_tool["tool_name"],
                    sequence_no=program_no,
                    error_terms=[],
                    start_depth=round(start_depth, 3),
                    end_depth=round(end_depth, 3)
                )
            )

        self.survey_tools = survey_tools

    def extract_functions(self, survey_tool: SurveyToolErrorModel):

        error_codes = self.edm.get_attributes(
            tags=["DP_TOOL_TERM"],
            attributes={"survey_tool_id": survey_tool.survey_tool_id}
        )['DP_TOOL_TERM']

        error_terms = []
        for code in error_codes:

            tie_type = self.extract_tie_type(code, survey_tool.survey_tool_id)
            vector = self.extract_vector_type(code, survey_tool.survey_tool_id)

            mag = float(code["c_value"])
            if code["c_units"] in ["d", "dnt"]:
                mag = mag * DEGREE_TO_RAD

            error_terms.append(
                ErrorTerm(
                    sequence_no=int(code["sequence_no"]),
                    term_name=code["term_name"],
                    formula=code["c_formula"],
                    error_function=(result := function_builder(code["c_formula"], code["term_name"]))[0],
                    magnitude=mag,
                    units=code["c_units"],
                    vector_type=vector,
                    tie_type=tie_type,
                    arguments=result[1],
                    func_string=result[2]
                )
            )

        survey_tool.error_terms = sorted(error_terms, key=lambda d: d.sequence_no)

    def extract_tie_type(self, code: dict, survey_tool_id: str) -> str:
        if code["tie_type"] == 'r':
            return Propagation.RANDOM
        elif code["tie_type"] == "s":
            return Propagation.SYSTEMATIC
        elif code["tie_type"] == "g":
            # since welleng only covers random and systematic errors, convert global errors to systematic.
            return Propagation.SYSTEMATIC
        elif code["tie_type"] == "n":
            return Propagation.NA

        return Propagation.SYSTEMATIC

    def extract_vector_type(self, code: dict, survey_tool_id: str) -> VectorType:
        if code["vector_type"] == VectorType.AZIMUTH.value:
            return VectorType.AZIMUTH
        elif code["vector_type"] == VectorType.INCLINATION.value:
            return VectorType.INCLINATION
        elif code["vector_type"] == VectorType.DEPTH.value:
            return VectorType.DEPTH
        elif code["vector_type"] == VectorType.LATERAL.value:
            return VectorType.LATERAL
        elif code["vector_type"] == VectorType.NA:
            return VectorType.NA

        return VectorType.DEPTH
