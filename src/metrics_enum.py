from enum import Enum


class Metric(Enum):
    ALL_SCORE = "all_score"
    VITAL_SCORE = "vital_score"
    STRICT_VITAL_SCORE = "strict_vital_score"
    STRICT_ALL_SCORE = "strict_all_score"

    @staticmethod
    def from_str(label):
        for metric in Metric:
            if metric.value == label:
                return metric
        raise ValueError(f"Unknown metric: {label}")

    def __str__(self):
        return self.value
