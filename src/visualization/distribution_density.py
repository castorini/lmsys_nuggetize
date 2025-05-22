import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from src.utils import Metric


def get_data_df(results_path, metric):
    # Example data
    metric_a = []
    metric_b = []
    labels = []
    with open(results_path, "r") as f:
        for l in f:
            data = json.loads(l)
            if data["winner"] not in ["model_a", "tie", "model_b"]:
                continue
            labels.append(data["winner"])
            metric_a.append(data["metrics_a"][str(metric)])
            metric_b.append(data["metrics_b"][str(metric)])

    # Create DataFrame (as before)
    return pd.DataFrame(
        {
            "diff": [b - a for a, b in zip(metric_a, metric_b)],
            "label": labels,  # "model_a", "tie", "model_b"
        }
    )


def probability_distribution(df, metric, output_dir):
    # Set style
    sns.set_context("notebook", font_scale=1.6)
    sns.set(style="whitegrid")

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, sharey=True)

    # Define categories, titles, and colors
    categories = ["model_a", "tie", "model_b"]
    titles = [r"Model$_{A}$ Wins", "Tie", r"Model$_{B}$ Wins"]
    colors = ["tab:blue", "tab:gray", "tab:red"]

    # Plot each category
    for ax, category, title, color in zip(axes, categories, titles, colors):
        subset = df[df["label"] == category]
        bin_edges = np.linspace(-1.025, 1.025, 41)
        # Filled histogram
        sns.histplot(
            data=subset,
            x="diff",
            stat="density",
            bins=bin_edges,
            binwidth=0.05,
            element="bars",
            fill=True,
            color=color,
            ax=ax,
            binrange=(-1.025, 1.025),  # Ensure bins range from -1.025 to 1.025
        )

        # KDE overlay
        sns.kdeplot(
            data=subset, x="diff", color="green", linewidth=1.5, bw_adjust=0.5, ax=ax
        )

        ax.set_title(title, fontsize=16)
        ax.set_ylabel(r"Density", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True)

    # Shared x label
    axes[-1].set_xlabel(
        r"Nugget ($\text{score}_{B}$ $-$ $\text{score}_{A}$)", fontsize=16
    )

    # Final layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{str(metric)}_density.pdf"), dpi=300)
    plt.close()


def ks_test(df, metric, output_dir):
    # Suppose your DataFrame is already loaded
    subset_a = df[df["label"] == "model_a"]["diff"]
    subset_b = df[df["label"] == "model_b"]["diff"]
    subset_tie = df[df["label"] == "tie"]["diff"]

    # Function to plot empirical CDF
    def plot_ecdf(data, label, color, ax):
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, marker=".", linestyle="none", label=label, color=color)

    # Define comparisons
    comparisons = [
        (
            subset_a,
            subset_b,
            r"Model$_{A}$ Wins vs. Model$_{B}$ Wins",
            r"Model$_{A}$ Wins",
            r"Model$_{B}$ Wins",
        ),
        (
            subset_a,
            subset_tie,
            r"Model$_{A}$ Wins vs. Tie",
            r"Model$_{A}$ Wins",
            r"Tie",
        ),
        (
            subset_b,
            subset_tie,
            r"Model$_{B}$ Wins vs. Tie",
            r"Model$_{B}$ Wins",
            r"Tie",
        ),
    ]
    colors = {
        r"Model$_{A}$ Wins": "tab:blue",
        r"Tie": "tab:gray",
        r"Model$_{B}$ Wins": "tab:red",
    }
    # plt.rcParams['text.usetex'] = True
    # Create subplots
    _, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (data1, data2, title, label1, label2) in zip(axes, comparisons):
        # Perform KS test
        ks_stat, p_value = ks_2samp(data1, data2)

        # Plot CDFs
        plot_ecdf(data1, label1, colors[label1], ax)
        plot_ecdf(data2, label2, colors[label2], ax)

        # Annotate plot
        textstr = f"KS Statistic = {ks_stat:.3f}\n$p$-value = {p_value:.1e}"
        ax.text(
            0.95,
            0.05,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.tick_params(axis="both", labelsize=14)
        ax.set_title(title, fontsize=18)
        ax.grid(True)
        ax.legend(fontsize=14, loc="upper left")
        ax.set_xlabel(r"Nugget ($\text{score}_B$ $-$ $\text{score}_A$)", fontsize=16)

    # Shared y-label
    axes[0].set_ylabel(r"Cumulative Probability", fontsize=16)

    # Global settings
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{str(metric)}_ks_test_comparisons.pdf"),
        dpi=300,
        bbox_inches="tight",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze score differences and plot distributions and KS tests."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=lambda x: Metric(x),
        default=[Metric.ALL_SCORE],
        help="List of metrics to evaluate",
    )
    parser.add_argument(
        "--results_path", type=str, required=True, help="Path to results.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where output plots will be saved",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for metric in args.metrics:
        df = get_data_df(args.results_path, metric)
        probability_distribution(df, metric, args.output_dir)
        ks_test(df, metric, args.output_dir)


if __name__ == "__main__":
    main()
