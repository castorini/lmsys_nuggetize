import json

from datasets import load_dataset
import pandas as pd
from scipy.stats import ttest_ind


def main():
    battle_data = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    battle_data = battle_data.to_pandas()

    variables = ["num_citations_a", "response_length_a"]

    for var in variables:
        winner_a = []
        var_a = []
        for i, row in battle_data.iterrows():
            win = 1 if row["winner"] == "model_a" else 0
            winner_a.append(win)
            var_a.append(row["conv_metadata"][var])

        df = pd.DataFrame({"Winner_a": winner_a, "Score_a": var_a})

        # Grouping based on winner binary
        group0 = df[df["Winner_a"] == 0]["Score_a"]
        group1 = df[df["Winner_a"] == 1]["Score_a"]

        with open(f"{var}_output.jsonl", "w") as f:
            f.write(json.dumps({"Score_a_0": group0.to_list()}) + "\n")
            f.write(json.dumps({"Score_a_1": group1.to_list()}) + "\n")

        alpha = 0.05
        t_stat, p_val = ttest_ind(group0, group1, equal_var=False)
        print(f"Test statistic: {t_stat:.4f}")
        print(f"P-value: {p_val:.10f}")

        if p_val < alpha:
            print(
                f"Result: Reject the null hypothesis — {var} is associated with Winner_a."
            )
        else:
            print(
                f"Result: Fail to reject the null hypothesis — no significant evidence that {var} is associated with Winner_a."
            )
        print("-" * 79)

if __name__ == "__main__":
    main()