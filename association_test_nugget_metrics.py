import json

import pandas as pd
from scipy.stats import ttest_ind


def main():
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
        print(f"Hypothesis testing for {metric}")
        winner_a = []
        score_a = []

        for val in data:
            win = 1 if val["winner"] == "model_a" else 0
            winner_a.append(win)
            score_a.append(val["metrics_a"][metric])

        df = pd.DataFrame({"Winner_a": winner_a, "Score_a": score_a})

        # Grouping based on winner binary
        group0 = df[df["Winner_a"] == 0]["Score_a"]
        group1 = df[df["Winner_a"] == 1]["Score_a"]

        alpha = 0.05

        t_stat, p_val = ttest_ind(
            group0, group1, equal_var=False, alternative="two-sided"
        )
        test_used = "Welch's t-test"

        print(f"\nTest used: {test_used}")
        print(f"P-value: {p_val:.8f}")

        if p_val < alpha:
            print(
                f"Result: Reject the null hypothesis — {metric}  is associated with Winner_a."
            )
        else:
            print(
                f"Result: Fail to reject the null hypothesis — no significant evidence that {metric} is associated with Winner_a."
            )
        print("-" * 79)


if __name__ == "__main__":
    main()
