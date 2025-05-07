import json
import pandas as pd
from datasets import load_dataset, Dataset

import seaborn as sns
import os
import matplotlib.pyplot as plt

def load_query_ids(hf_dataset: Dataset, filter_single_turn: bool = True):
    query_ids = set()
    for row in hf_dataset:
        if filter_single_turn and len(row["messages_a"]) == 2:
            query_ids.add(row["question_id"])
        else:
            query_ids.add(row["question_id"])
    return query_ids

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_dataset", type=str, default="lmarena-ai/search-arena-v1-7k")
    parser.add_argument("--filter_single_turn", action="store_true", help="Filter single-turn queries")
    args = parser.parse_args()

    # Load the dataset
    hf_dataset = load_dataset(args.hf_dataset, split="test")
    print(f"Loading the test dataset ({args.hf_dataset})) with filter single-turn queries: {args.filter_single_turn}")
    query_ids = load_query_ids(hf_dataset, args.filter_single_turn)
    print(f"Total number of queries: {len(query_ids)}")

    # load the data as a DataFrame
    all_data = []
    with open(args.results_path, "r", encoding="utf-8") as f:
        for row in f:
            data = json.loads(row)
            if data["question_id"] in query_ids:
                all_data.append(data["categories"])

    df = pd.DataFrame(all_data)

    # Plotting histograms for each category
    columns = [
        'ambiguous',
        'incompleteness',
        'assumptive',
        'multi-faceted',
        'knowledge-intensive',
        'subjective',
        'reasoning-intensive',
        'harmful'
    ]

    for col in columns:
        # Create a new figure for each column
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        
        # Plot histogram
        sns.histplot(df[col], bins=10, kde=False, ax=ax)
        
        # Formatting
        ax.set_title(col, fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Tight layout
        fig.tight_layout()
        
        # Save individual pdf (you can change the filename pattern as you like)
        os.makedirs(args.output_dir, exist_ok=True)
        fig.savefig(f"{args.output_dir}/query_category_{col}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()