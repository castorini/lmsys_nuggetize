import argparse
import json
import os
from collections import defaultdict

from datasets import load_dataset


def get_prompt(row):
    message = row["messages_a"][0]
    assert message["role"] == "user"
    prompt = message["content"]
    message = row["messages_b"][0]
    assert message["role"] == "user"
    assert prompt == message["content"], "Both LLMs should get the same prompt"
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


def main():
    parser = argparse.ArgumentParser(description="Extract sample queries per category.")
    parser.add_argument(
        "--path_prefix", type=str, required=True, help="Path to skips.json"
    )
    parser.add_argument(
        "--categories_path",
        type=str,
        required=True,
        help="Path to the categorization .jsonl file",
    )
    parser.add_argument(
        "--class_threshold",
        type=int,
        default=7,
        help="Minimum category score threshold",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Target language to filter (default: English)",
    )
    args = parser.parse_args()

    input_df = load_dataset("lmarena-ai/search-arena-v1-7k", split="test").to_pandas()
    input_df = input_df.set_index("question_id")

    skips = load_skips(args.path_prefix)
    categories = defaultdict(list)

    with open(args.categories_path, "r") as f:
        for line in f:
            data = json.loads(line)
            qid = data["question_id"]
            if qid in skips or qid not in input_df.index:
                continue

            row = input_df.loc[qid]
            lang = row["language"]
            if lang != args.language:
                continue

            prompt = get_prompt(row)

            max_rate = max(data["categories"].values())
            max_cat = [
                cat for cat, val in data["categories"].items() if val == max_rate
            ]

            if max_rate >= args.class_threshold:
                for cat in max_cat:
                    categories[cat].append((max_rate, qid, prompt))

    sorted_dict = dict(
        sorted(categories.items(), key=lambda item: len(item[1]), reverse=True)
    )
    metadata = {"language": args.language, "class_threshold": args.class_threshold}
    with open(
        os.path.join(args.path_prefix, "sample_query_per_category.json"), "w"
    ) as f:
        json.dump({"data": sorted_dict, "metadata": metadata}, f, indent=2)


if __name__ == "__main__":
    main()
