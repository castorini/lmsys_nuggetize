import json
import os

import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np


def main():
    results_dir = "rank_swap_hist"
    os.makedirs(results_dir, exist_ok=True)

    files = [
        "/mnt/users/s8sharif/search_arena/gpt41_i_mode_5/results.jsonl",
        "/mnt/users/s8sharif/search_arena/gpt41_i_not_mode_5/results.jsonl",
        "/mnt/users/s8sharif/search_arena/gpt41_ties/results.jsonl"
    ]

    data = []
    for file in files:
        with open(file, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    metrics = ["strict_vital_score", "strict_all_score", "vital_score", "all_score"]

    for metric in metrics:
        print(f"Rank swap evaluation for metric {metric}")

        winner_a_bool = []
        diff_values = []
        inversions = []
        for val in data:
            win = 1 if val["winner"] == "model_a" else -1
            winner_a_bool.append(win)
            diff = val["metrics_a"][metric] - val["metrics_b"][metric]
            diff_values.append(diff)
            if win * diff < 0:
                inversions.append(abs(diff) * 100)

        print(f"Len of inversions: {len(inversions)} for {metric}")

        bin_size = 5
        bins = np.arange(0, 21 * bin_size, bin_size)

        nested_dict = {int(bins[i + 1]): 0 for i in range(len(bins) - 1)}

        for idx, diff in enumerate(inversions):

            for i in range(len(bins) - 1):
                if bins[i] < diff <= bins[i + 1]:
                    nested_dict[int(bins[i + 1])] += 1

        fig, ax = plt.subplots(figsize=(16, 6))
        bar_positions = np.arange(len(nested_dict)) * bin_size

        ax.bar(
            bar_positions,
            list(nested_dict.values()),
            width=0.8 * bin_size,
        )

        plt.xticks(
            bins[:-1],
            labels=[f"{x + bin_size}" for x in bins[:-1]],
            rotation=45,
            fontsize=16,
        )
        plt.ylabel("Frequency", fontsize=16)
        plt.title(
            f"Absolute Diff for Inversions for {metric}", fontsize=28, fontweight="bold"
        )
        path = f"{results_dir}/rs_histogram_{metric}.png"
        plt.subplots_adjust(top=1.0)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print("-" * 79)


if __name__ == "__main__":
    main()
