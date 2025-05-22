import argparse
import json
import os
from collections import defaultdict

from src.utils import load_inversion_ids, load_skips


def compute_category_percentages(categories_path, skips, inversion_ids, threshold):
    categories = defaultdict(list)

    with open(categories_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["question_id"]
            if qid in skips:
                continue

            max_score = max(entry["categories"].values())
            if max_score < threshold:
                continue

            top_cats = [
                cat for cat, val in entry["categories"].items() if val == max_score
            ]
            for cat in top_cats:
                categories[cat].append(qid)
            categories["all"].append(qid)

    percentage = {
        cat: (sum(1 for qid in qids if qid in inversion_ids) / len(qids), len(qids))
        for cat, qids in categories.items()
    }

    return percentage


def main():
    parser = argparse.ArgumentParser(
        description="Compute inversion percentages per query category."
    )
    parser.add_argument(
        "--path_prefix", type=str, required=True, help="Input and Output path prefix"
    )
    parser.add_argument(
        "--categories_path",
        type=str,
        required=True,
        help="Path to categories JSONL file",
    )
    parser.add_argument(
        "--class_threshold",
        type=int,
        default=7,
        help="Category score threshold (default: 7)",
    )
    args = parser.parse_args()

    skips = load_skips(args.path_prefix)
    inversion_ids_by_lang, metadata = load_inversion_ids(args.path_prefix)
    inversion_ids = set()
    for v in inversion_ids_by_lang.values():
        inversion_ids = inversion_ids | v
    percentage = compute_category_percentages(
        args.categories_path, skips, inversion_ids, args.class_threshold
    )

    print(json.dumps(percentage, indent=2))
    metadata["class_threshold"] = args.class_threshold
    with open(
        os.path.join(args.path_prefix, "inversions_per_category_percentage.json"), "w"
    ) as f:
        json.dump({"data": percentage, "metadata": metadata}, f, indent=2)


if __name__ == "__main__":
    main()
