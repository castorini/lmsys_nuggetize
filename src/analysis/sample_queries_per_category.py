import argparse
import json
import os
from collections import defaultdict

from datasets import load_dataset

from src.utils import get_prompt, load_skips


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
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The search arena dataset",
    )
    args = parser.parse_args()

    input_df = load_dataset(args.dataset, split="test").to_pandas()
    if "question_id" not in input_df.columns:
        input_df["question_id"] = input_df.index

    skips = load_skips(args.path_prefix)
    categories = defaultdict(list)

    with open(args.categories_path, "r") as f:
        for line in f:
            data = json.loads(line)
            qid = data["question_id"]
            if qid in skips or qid not in input_df.index:
                continue

            row = input_df.loc[qid]
            lang = row["language"] if "language" in row else row["languages"]
            if not isinstance(lang, str):
                lang = (
                    "" if len(lang) != 1 else lang[0]
                )  # skips multilingual rows and those without any assigned languages.
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
