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

    def __add__(self, other: "ErrorTerm") -> "ErrorTerm":
        if not isinstance(other, ErrorTerm):
            raise TypeError(f"Cannot concatenate {type(other)} to ErrorTerm")

        if self.sequence_no != other.sequence_no:
            raise ValueError(
                f"Cannot concatenate ErrorTerm with different sequence_no: "
                f"{self.sequence_no} and {other.sequence_no}"
            )

        if self.term_name != other.term_name:
            raise ValueError(
                f"Cannot concatenate ErrorTerm with different term_name: "
                f"{self.term_name} and {other.term_name}"
            )

        self.formula += other.formula
        self.error_function += other.error_function
        self.magnitude += other.magnitude
        self.units += other.units
        self.tie_type += other.tie_type
        self.vector_type += other.vector_type
        self.arguments += other.arguments
        self.func_string += other.func_string

        return self


class SurveyToolErrorModel(BaseModel):
    survey_tool_id: str
    survey_tool_name: str
    sequence_no: int
    error_terms: List[ErrorTerm]
    start_depth: float
    end_depth: float
