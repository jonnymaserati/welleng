from typing import Callable, List

from pydantic import BaseModel

from welleng.error_formula_extractor.enums import Propagation, VectorType


class ErrorTerm(BaseModel):
    sequence_no: int
    term_name: str
    formula: str
    error_function: Callable
    magnitude: float
    units: str
    tie_type: Propagation
    vector_type: VectorType
    arguments: set
    func_string: str


class SurveyToolErrorModel(BaseModel):
    survey_tool_id: str
    survey_tool_name: str
    sequence_no: int
    error_terms: List[ErrorTerm]
    start_depth: float
    end_depth: float
