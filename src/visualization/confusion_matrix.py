import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset

from src.utils import Metric


def draw_non_sq_confusion_matrix(gts, preds, name):
    true_labels = sorted(set(gts))
    pred_labels = sorted(set(preds))

    label_to_row = {label: i for i, label in enumerate(true_labels)}
    label_to_col = {label: i for i, label in enumerate(pred_labels)}

    cm = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for t, p in zip(gts, preds):
        if t in label_to_row and p in label_to_col:
            cm[label_to_row[t], label_to_col[p]] += 1

    plt.rcParams.update({"font.size": 16})
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[r"Model$_{A}$", "Tie", r"Model$_{B}$"],
        yticklabels=[r"Model$_{A}$", "Tie", r"Model$_{B}$", "Tie\n(both bad)"],
    )
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_verticalalignment("center")
        label.set_horizontalalignment("center")
        label.set_multialignment("center")
        label.set_x(label.get_position()[0] - 0.05)

    plt.ylabel("Human Preference", fontsize=16, fontweight="bold")
    plt.xlabel(
        "Nugget ($\\mathbf{score}_{\\mathbf{B}}$ $-$ $\\mathbf{score}_{\\mathbf{A}}$)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name)
    plt.close()


def prepare_labels(data_df, diff_thresh, metric, allowed_qids=None):
    original_labels, pred_labels = [], []
    label_mapping_inv = {"model_a": 0, "tie": 1, "model_b": 2, "tie (bothbad)": 3}

    for _, val in data_df.iterrows():
        if allowed_qids and val["question_id"] not in allowed_qids:
            continue
        original_labels.append(label_mapping_inv[val["winner"]])
        score_a = val["metrics_a"][metric]
        score_b = val["metrics_b"][metric]
        diff = score_b - score_a

        if abs(diff) > diff_thresh:
            pred = 2 if diff > 0 else 0
        else:
            pred = 1
        pred_labels.append(pred)

    assert len(original_labels) == len(pred_labels), "Length mismatch between labels"
    return original_labels, pred_labels


def get_query_categories(categories_path, class_threshold=7):
    categories = defaultdict(list)
    with open(categories_path, "r") as f:
        for l in f:
            data = json.loads(l)
            max_rate = max(data["categories"].values())
            max_cat = [
                cat for cat, val in data["categories"].items() if val == max_rate
            ]
            if max_rate >= class_threshold:
                for cat in max_cat:
                    categories[cat].append(data["question_id"])
    return categories


def conf_matrices_for_query_categories(
    diff_thresh, data_df, metrics, categories, output_dir
):
    for cat, qids in categories.items():
        for metric in metrics:
            labels = prepare_labels(data_df, diff_thresh, metric, allowed_qids=qids)
            path = os.path.join(
                output_dir,
                "per_query_category_conf_matrix",
                f"{diff_thresh}_conf_mat_{cat}_{metric}.pdf",
            )
            draw_non_sq_confusion_matrix(*labels, path)


def conf_matrices_for_languages(diff_thresh, data_df, metrics, output_dir):
    for language, freq in data_df["language"].value_counts().items():
        if freq <= 100:
            continue
        qids = list(data_df[data_df["language"] == language]["question_id"])
        for metric in metrics:
            labels = prepare_labels(data_df, diff_thresh, metric, allowed_qids=qids)
            path = os.path.join(
                output_dir,
                "per_lang_conf_matrix",
                f"{diff_thresh}_conf_mat_{language}_{metric}.pdf",
            )
            draw_non_sq_confusion_matrix(*labels, path)


def overall_conf_matrix(diff_thresh, data_df, metrics, output_dir):
    for metric in metrics:
        labels = prepare_labels(data_df, diff_thresh, metric)
        path = os.path.join(
            output_dir, "overall_conf_matrix", f"{diff_thresh}_conf_mat_{metric}.pdf"
        )
        draw_non_sq_confusion_matrix(*labels, path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate confusion matrices for LLM nugget scoring."
    )
    parser.add_argument(
        "--results_path", type=str, required=True, help="Path to the results.jsonl file"
    )
    parser.add_argument(
        "--categories_path",
        type=str,
        required=True,
        help="Path to the query categories file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save confusion matrix plots",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=Metric.from_str,
        choices=list(Metric),
        required=True,
        help="List of metrics to draw the confusion matrices for.",
    )
    parser.add_argument(
        "--preference_threshold",
        type=float,
        default=0.1,
        help="Preference difference threshold to classify ties",
    )
    parser.add_argument(
        "--class_threshold",
        type=int,
        default=7,
        help="Threshold to classify queries into a category",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_df = load_dataset("lmarena-ai/search-arena-v1-7k", split="test").to_pandas()
    jsonl_df = pd.read_json(args.results_path, lines=True)
    merged_df = pd.merge(
        dataset_df, jsonl_df, on=["question_id", "winner"], how="inner"
    )
    categories = get_query_categories(
        args.categories_path, class_threshold=args.class_threshold
    )
    metrics = [metric.value for metric in args.metrics]
    conf_matrices_for_query_categories(
        args.preference_threshold, merged_df, metrics, categories, args.output_dir
    )
    conf_matrices_for_languages(
        args.preference_threshold, merged_df, metrics, args.output_dir
    )
    overall_conf_matrix(args.preference_threshold, merged_df, metrics, args.output_dir)


if __name__ == "__main__":
    main()
