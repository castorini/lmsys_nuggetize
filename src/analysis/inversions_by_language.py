import argparse
import json
import os

import pandas as pd
from datasets import load_dataset

from src.utils import load_inversion_ids


def compute_language_percentages(inversion_by_lang, dataset_df, jsonl_df):
    merged_df = pd.merge(dataset_df, jsonl_df, on=["question_id"], how="inner")
    total_rows = merged_df.shape[0]
    remaining_inversions = sum(len(qids) for qids in inversion_by_lang.values())
    remaining_count = total_rows

    percentage = {}
    for lang, qids in inversion_by_lang.items():
        lang_df = merged_df[merged_df["language"] == lang]
        lang_count = lang_df.shape[0]
        if lang_count / total_rows < 0.029:  # Filter out low-frequency languages
            continue
        remaining_inversions -= len(qids)
        remaining_count -= lang_count
        percentage[lang] = (len(qids) / lang_count, lang_count)

    if remaining_count > 0:
        percentage["Others"] = (remaining_inversions / remaining_count, remaining_count)

    return percentage


def main():
    parser = argparse.ArgumentParser(
        description="Compute inversion percentages per language."
    )
    parser.add_argument(
        "--path_prefix", type=str, required=True, help="Input and Output path prefix"
    )
    args = parser.parse_args()

    inversion_by_lang, metadata = load_inversion_ids(args.path_prefix)
    dataset_df = load_dataset("lmarena-ai/search-arena-v1-7k", split="test").to_pandas()
    jsonl_df = pd.read_json(os.path.join(args.path_prefix, "results.jsonl"), lines=True)

    percentage = compute_language_percentages(inversion_by_lang, dataset_df, jsonl_df)

    print(json.dumps(percentage, indent=2))
    with open(
        os.path.join(args.path_prefix, "inversions_per_language_percentage.json"), "w"
    ) as f:
        json.dump({"data": percentage, "metadata": metadata}, f, indent=2)


if __name__ == "__main__":
    main()
