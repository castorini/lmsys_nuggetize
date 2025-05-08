import argparse
import json
import os
from collections import defaultdict

from datasets import load_dataset

from src.metrics_enum import Metric


def get_prompt(row):
    message = row["messages_a"][0]
    assert message["role"] == "user"
    prompt = message["content"]
    message = row["messages_b"][0]
    assert message["role"] == "user"
    assert prompt == message["content"], "both LLMs should get the same prompt"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_prefix", type=str, required=True, help="Input and Output path prefix"
    )
    parser.add_argument(
        "--inversion_metric",
        type=Metric.from_str,
        choices=list(Metric),
        required=True,
        help="Metric name to track diagram candidates for",
    )
    parser.add_argument(
        "--inversion_threshold",
        type=float,
        default=0.1,
        help="Threshold to determine if a score difference is significant",
    )
    parser.add_argument(
        "--candidates_language",
        type=str,
        required=True,
        help="Language to filter for diagram candidates",
    )
    args = parser.parse_args()

    stats = defaultdict(int)
    per_language_stats = {}
    per_language_inversions = {}
    diagram_candidates = {}
    data = []
    threshold = args.inversion_threshold

    input_df = load_dataset("lmarena-ai/search-arena-v1-7k", split="test").to_pandas()

    with open(os.path.join(args.path_prefix, "results.jsonl"), "r") as f:
        for l in f:
            data.append(json.loads(l))

    for row in data:
        id = row["question_id"]
        lang = input_df.iloc[id]["language"]
        if lang not in per_language_stats:
            per_language_stats[lang] = defaultdict(int)
        per_language_stats[lang]["total"] += 1
        if lang not in per_language_inversions:
            per_language_inversions[lang] = {}

        if "tie" in row["winner"]:
            stats["tie"] += 1
            continue

        first_stats = (
            row["metrics_a"] if row["winner"] == "model_a" else row["metrics_b"]
        )
        second_stats = (
            row["metrics_b"] if row["winner"] == "model_a" else row["metrics_a"]
        )

        # Helper for score comparisons
        def update_stats(metric_name, inversion_metric, candidates_language):
            diff = first_stats[metric_name] - second_stats[metric_name]
            if -threshold < diff < threshold:
                stats[f"{metric_name}_ties"] += 1
                per_language_stats[lang][f"{metric_name}_ties"] += 1
            elif diff >= threshold:
                stats[f"{metric_name}_matches"] += 1
                per_language_stats[lang][f"{metric_name}_matches"] += 1
                if metric_name == inversion_metric and lang == candidates_language:
                    if diff not in diagram_candidates:
                        diagram_candidates[diff] = []
                    diagram_candidates[diff].append((id, get_prompt(input_df.iloc[id])))
            else:
                stats[f"{metric_name}_inversions"] += 1
                per_language_stats[lang][f"{metric_name}_inversions"] += 1
                if metric_name != inversion_metric:
                    return
                if diff not in per_language_inversions[lang]:
                    per_language_inversions[lang][diff] = []
                per_language_inversions[lang][diff].append(
                    (id, get_prompt(input_df.iloc[id]))
                )

        inversion_metric = args.inversion_metric.value
        for metric in Metric:
            update_stats(metric.value, inversion_metric, args.candidates_language)

    with open(os.path.join(args.path_prefix, "skips.json"), "r") as in_f:
        skip_data = json.load(in_f)

    with open(os.path.join(args.path_prefix, "count_stats.json"), "w") as out_f:
        json.dump(
            {
                "success": len(data),
                "failed_nuggetize": len(skip_data.get("nugget_creation", [])),
                "failed_assignment": len(skip_data.get("nugget_assignment", [])),
                "multi_turn": len(skip_data.get("multi_turn", [])),
                "sampling": len(skip_data.get("sampling", [])),
                "zero_grounding": len(skip_data.get("zero_grounding", [])),
            },
            out_f,
        )

    print(stats)
    metadata = {"inversion_threshold": threshold, "inversion_metric": inversion_metric}
    with open(os.path.join(args.path_prefix, "aggregated_stats.json"), "w") as f:
        json.dump({"data": stats, "metadata": metadata}, f, indent=2)

    sorted_lang_stats = dict(
        sorted(
            per_language_stats.items(),
            key=lambda item: item[1][f"{inversion_metric}_inversions"]
            / item[1]["total"],
            reverse=True,
        )
    )
    with open(
        os.path.join(args.path_prefix, "per_language_aggregated_stats.json"), "w"
    ) as f:
        json.dump({"data": sorted_lang_stats, "metadata": metadata}, f, indent=2)

    sorted_inversions = {
        lang: dict(sorted(inversions.items()))
        for lang, inversions in per_language_inversions.items()
    }
    with open(
        os.path.join(args.path_prefix, f"per_language_inversion_ids.json"), "w"
    ) as f:
        json.dump({"data": sorted_inversions, "metadata": metadata}, f, indent=2)

    sorted_diagrams = dict(sorted(diagram_candidates.items(), reverse=True))
    metadata = {
        "inversion_threshold": threshold,
        "inversion_metric": inversion_metric,
        "candidates_language": args.candidates_language,
    }
    with open(os.path.join(args.path_prefix, f"diagram_candidates.json"), "w") as f:
        json.dump({"data": sorted_diagrams, "metadata": metadata}, f, indent=2)


if __name__ == "__main__":
    main()
