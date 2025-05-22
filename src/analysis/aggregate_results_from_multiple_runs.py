import argparse
import json
import os


def main(input_paths, output_path):
    results = []
    successful_qids = set()
    for input_path in input_paths:
        with open(os.path.join(input_path, "results.jsonl"), "r") as f:
            for l in f:
                result = json.loads(l)
                if result["question_id"] not in successful_qids:
                    results.append(result)
                    successful_qids.add(result["question_id"])

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "results.jsonl"), "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")

    skips = {
        "nugget_creation": [],
        "nugget_assignment": [],
        "multi_turn": [],
        "sampling": [],
    }
    for input_path in input_paths:
        with open(f"{input_path}/skips.json", "r") as f:
            data = json.load(f)
            for key in skips:
                skips[key].extend(data[key])
        for key in skips:
            skips[key] = list(set(skips[key]) - successful_qids)

    with open(os.path.join(output_path, "skips.json"), "w") as f:
        json.dump(skips, f)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge results and skips from multiple input directories."
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="List of input directory paths containing results.jsonl and skips.json",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory to save the merged results.jsonl and skips.json",
    )

    args = parser.parse_args()
    main(args.input_paths, args.output_path)
