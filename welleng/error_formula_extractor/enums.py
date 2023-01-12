from enum import Enum


class Propagation(Enum):
    RANDOM: str = "random"
    SYSTEMATIC: str = "systematic"
    GLOBAL: str = "global"
    NA: str = "n/a"
    WELL: str = "well_to_well"

    @staticmethod
    def extract_tie_type(tie_type: str) -> 'Propagation':
        """
        Extract propagation type (tie type) from the EDM file and assign the correct enum based on the propagation.
        """
        tie_type = tie_type.lower()

        mapping = {
            "r": Propagation.RANDOM,
            "s": Propagation.SYSTEMATIC,
            "g": Propagation.GLOBAL,
            "n": Propagation.NA,
            "w": Propagation.WELL
        }

        if tie_type in mapping.keys():
            return mapping[tie_type]

        return Propagation.SYSTEMATIC


class VectorType(Enum):
    AZIMUTH_TERMS = (
        dict(AZIMUTH="a",
             AZIMUTH_BIAS="b",
             RADIUS="c",
             MISALIGNMENT_A="m",
             INERTIAL_A="c"),
        2
    )

    INCLINATION_TERMS = (
        dict(
            INCLINATION="i",
            INCLINATION_BIAS="j",
        ),
        1
    )

    DEPTH_TERMS = (
        dict(
            DEPTH="d",
            DEPTH_ISCWSA="e",
            DEPTH_BIAS="f",
            DEPTH_ISCWSA_FL="s",
            LONG_COURSE_X="x",
            LONG_COURSE_Y="y"
        ), 0
    )

    LATERAL = ("l", 2)
    NA: str = ("n", None)

    def __new__(cls, vector_type: str, column_no: int):
        obj = object.__new__(cls)

        obj._value_ = vector_type  # using the task_action as the main value
        obj.column_no = column_no

        return obj

    def __str__(self):
        return self.value

    @classmethod
    def get_object(cls):
        return {
            "azimuth_terms": [
                vector_data
                for vector_data in cls.AZIMUTH_TERMS.value.values()
            ],
            "inclination_terms": [
                vector_data
                for vector_data in cls.INCLINATION_TERMS.value.values()
            ],
            "depth_terms": [
                vector_data
                for vector_data in cls.DEPTH_TERMS.value.values()
            ]
        }
