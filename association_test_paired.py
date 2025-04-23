from datasets import load_dataset
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind


def main():
    battle_data = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    battle_data = battle_data.to_pandas()

    variables = [
        ("num_citations_a", "num_citations_b", "Num citations"),
        ("response_length_a", "response_length_b", "Response length"),
    ]

    for var in variables:
        print("-" * 79)
        print("-" * 79)
        print(f"Evaluating for {var[2]}")
        print("-" * 79)
        ######################## T test ########################
        print("T-test")
        winner_a = []
        var_a = []
        for i, row in battle_data.iterrows():
            win = 1 if row["winner"] == "model_a" else 0
            winner_a.append(win)
            higher_var = (
                1
                if row["conv_metadata"][var[0]] > row["conv_metadata"][var[1]]
                else 0
            )
            var_a.append(higher_var)

        df = pd.DataFrame({"Winner_a": winner_a, "Score_a_bin": var_a})

        # Grouping based on winner binary
        group0 = df[df["Winner_a"] == 0]["Score_a_bin"]
        group1 = df[df["Winner_a"] == 1]["Score_a_bin"]

        alpha = 0.05
        t_stat, p_val = ttest_ind(group0, group1, equal_var=False)
        print(f"Test statistic: {t_stat:.4f}")
        print(f"P-value: {p_val:.10f}")

        if p_val < alpha:
            print(
                f"Result: Reject the null hypothesis — {var[2]} is associated with Winner_a."
            )
        else:
            print(
                f"Result: Fail to reject the null hypothesis — no significant evidence that {var[2]} is associated with Winner_a."
            )

        print("-" * 79)
        ######################## Chi test ########################
        print("Chi-test")

        a = sum(1 for x, y in zip(group0, group1) if x == 1 and y == 1)
        b = sum(1 for x, y in zip(group0, group1) if x == 1 and y == 0)
        c = sum(1 for x, y in zip(group0, group1) if x == 0 and y == 1)
        d = sum(1 for x, y in zip(group0, group1) if x == 0 and y == 0)

        contingency_table = [[a, b], [c, d]]
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        print("Contingency table:", contingency_table)
        print("Chi-Square statistic:", chi2)
        print("Degrees of freedom:", dof)
        print("P-value:", p)
        print("Expected frequencies:\n", expected)

        alpha = 0.05
        if p < alpha:
            print("Reject the null hypothesis - the lists are dependent.")
        else:
            print("Fail to reject the null hypothesis - the lists are independent.")


if __name__ == "__main__":
    main()
