import json
import os
from collections import defaultdict
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


def get_prompt(row):
    message = row["messages_a"][0]
    assert message["role"] == "user"
    prompt = message["content"]
    message = row["messages_b"][0]
    assert message["role"] == "user"
    assert prompt == message["content"], "both LLMs should get the same prompt"
    return prompt


def load_skips(path_prefix):
    with open(os.path.join(path_prefix, "skips.json"), "r") as f:
        data = json.load(f)
    return set(
        data.get("nugget_creation", [])
        + data.get("nugget_assignment", [])
        + data.get("sampling", [])
        + data.get("multi_turn", [])
    )


def load_inversion_ids(path_prefix):
    inversion_ids = defaultdict(set)
    with open(os.path.join(path_prefix, "per_language_inversion_ids.json"), "r") as f:
        input_dict = json.load(f)
        data, metadata = input_dict["data"], input_dict["metadata"]
        for lang, lang_dict in data.items():
            for direction_list in lang_dict.values():
                for qid, _ in direction_list:
                    inversion_ids[lang].add(qid)
    return inversion_ids, metadata
