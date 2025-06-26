import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset


def get_turn_label(row):
    if row["turn"] <= 3:
        return str(row["turn"])
    return "4+"


def prepare_dataset_df(dataset, skip_multi_turn, skip_no_vote):
    input_df = load_dataset(dataset, split="test").to_pandas()
    stats = []
    for _, row in input_df.iterrows():
        if skip_multi_turn and row["turn"] > 1:
            continue
        if skip_no_vote and not row["winner"]:
            continue
        if "languages" in input_df:  # 24k version uses languages
            language = (
                row["languages"][0]
                if len(row["languages"]) == 1
                else str(row["languages"])
            )
        else:
            language = row["language"]
        stats.append(
            {
                "language": language,
                "turns": get_turn_label(row),
                "winner": row["winner"],
            }
        )
    return pd.DataFrame(stats)


def win_pie_chart(dataset_df, output_dir):
    win_counts = dataset_df["winner"].value_counts()
    categories = ["model_a", "model_b", "tie", "tie (bothbad)"]
    win_counts = win_counts.reindex(categories, fill_value=0)
    colors = {
        "model_a": "tab:blue",
        "model_b": "tab:red",
        "tie": "tab:gray",
        "tie (bothbad)": "tab:orange",
    }
    pie_colors = [colors[cat] for cat in categories]

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        win_counts,
        labels=categories,
        autopct="%1.1f%%",
        startangle=90,
        colors=pie_colors,
        wedgeprops={"edgecolor": "white"},
    )
    plt.suptitle("Win Category Distribution (Single Turn Only)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "win_category_pie_chart.pdf"), dpi=300)
    plt.close()


def lang_pie_chart(top_n, dataset_df, output_dir):
    lang_counts = dataset_df["language"].value_counts()
    top_langs = lang_counts.nlargest(top_n)
    other_count = lang_counts.iloc[top_n:].sum()
    if other_count > 0:
        top_langs["others"] = other_count

    plt.rcParams.update({"font.size": 13})
    plt.figure(figsize=(6, 6))
    top_langs.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        colors=plt.cm.Set3.colors,
        textprops={"fontsize": 12},
    )
    plt.suptitle("Language Distribution (Single Turn Only)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "language_pie_chart.pdf"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate pie charts for win category and language distribution."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output pie charts.",
    )
    parser.add_argument(
        "--top_n_languages",
        type=int,
        default=5,
        help="Number of top languages to include in language pie chart.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--skip_multi_turn", action="store_true", help="skips multi-turn queries"
    )
    parser.add_argument(
        "--skip_no_vote", action="store_true", help="skips queries with no human vote"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = prepare_dataset_df(args.dataset, args.skip_multi_turn, args.skip_no_vote)
    win_pie_chart(df, args.output_dir)
    lang_pie_chart(args.top_n_languages, df, args.output_dir)


if __name__ == "__main__":
    main()
