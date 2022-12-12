from typing import Callable, List

from pydantic import BaseModel

from welleng.error_formula_extractor.enums import Propagation, VectorType


class ErrorTerm(BaseModel):
    sequence_no: int
    term_name: str
    formula: List[str]
    error_function: List[Callable]
    magnitude: List[float]
    units: List[str]
    tie_type: List[Propagation]
    vector_type: List[VectorType]
    arguments: List[set]
    func_string: List[str]


class SurveyToolErrorModel(BaseModel):
    survey_tool_id: str
    survey_tool_name: str
    sequence_no: int
    error_terms: List[ErrorTerm]
    start_depth: float
    end_depth: float
