import argparse
import os

from datasets import load_dataset
from tqdm import tqdm


def get_prompt(row):
    message = row["messages_a"][0]
    assert message["role"] == "user"
    prompt = message["content"]
    message = row["messages_b"][0]
    assert message["role"] == "user"
    assert prompt == message["content"], "Both LLMs should get the same prompt"
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Export prompts from Search-Arena dataset."
    )
    parser.add_argument(
        "--path_prefix", required=True, help="Path prefix for output file."
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.path_prefix, "collections/url_corpus"), exist_ok=True)
    output_path = os.path.join(args.path_prefix, "collections/url_corpus/queries.tsv")

    data = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    data_df = data.to_pandas()

    with open(output_path, "w", encoding="utf-8") as out:
        for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
            assert index == row["question_id"]
            if row["turn"] != 1:
                continue
            prompt = get_prompt(row).replace("\n", " ").replace("\t", "    ").strip()
            out.write(f"{row['question_id']}\t{prompt}\n")


if __name__ == "__main__":
    main()
