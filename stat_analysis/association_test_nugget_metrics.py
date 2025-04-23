"""Script with association test for nugget metrics."""

import pandas as pd
from scipy.stats import ttest_ind

import nuggets_utils


def main():
    files = [
        "/mnt/users/s8sharif/search_arena/gpt41_i_mode_5/results.jsonl",
        "/mnt/users/s8sharif/search_arena/gpt41_i_not_mode_5/results.jsonl",
        "/mnt/users/s8sharif/search_arena/gpt41_ties/results.jsonl",
    ]

    data = nuggets_utils.load_results(files)
    metrics = ["strict_vital_score", "strict_all_score", "vital_score", "all_score"]

    for metric in metrics:
        print(f"Testing for {metric}")
        winner_a = []
        score_a = []

        for val in data:
            win = 1 if val["winner"] == "model_a" else 0
            winner_a.append(win)
            score_a.append(val["metrics_a"][metric])

        df = pd.DataFrame({"winner_a": winner_a, "score_a": score_a})

        # Grouping based on winner binary
        group0 = df[df["winner_a"] == 0]["score_a"]
        group1 = df[df["winner_a"] == 1]["score_a"]

        alpha = 0.05

        _, p_val = ttest_ind(group0, group1, equal_var=False, alternative="two-sided")
        test_used = "Welch's t-test"

        print(f"Test used: {test_used}")
        print(f"P-value: {p_val:.8f}")

        if p_val < alpha:
            print(
                f"Result: Reject the null hypothesis — {metric}  is associated with winner_a."
            )
        else:
            print(
                f"Result: Fail to reject the null hypothesis — no significant evidence that {metric} is associated with winner_a."
            )
        print("-" * 79)


if __name__ == "__main__":
    main()
