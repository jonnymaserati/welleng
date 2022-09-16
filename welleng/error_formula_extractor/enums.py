from enum import Enum


class Propagation(Enum):
    RANDOM: str = "random"
    SYSTEMATIC: str = "systematic"
    NA: str = "n/a"


class VectorType(Enum):
    AZIMUTH = (
        "a",
        2, 1
    )
    INCLINATION = (
        "i",
        1, 1
    )
    DEPTH = (
        "d",
        0, 1
    )
    LATERAL = (
        "l",
        2, None
    )
    NA: str = (
        "n",
        None, 1
    )

    def __new__(cls, vector_type: str, column_no: int, multiplier: int):
        obj = object.__new__(cls)

        obj._value_ = vector_type  # using the task_action as the main value
        obj.column_no = column_no
        obj.multiplier = multiplier

        return obj

    def __str__(self):
        return self.value
