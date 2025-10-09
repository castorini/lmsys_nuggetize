import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from scipy.stats import kendalltau, spearmanr
from datasets import load_dataset


def load_test_dataset(cache_dir):
    """Load the test split of the Search Arena dataset."""
    return load_dataset(
        "castorini/search-arena-v1-nuggets-with-urls-5k",
        split="test",
        cache_dir=cache_dir
    )


def get_sample_indices(dataset, sample_size=1000, seed=42):
    """Return a random sample of valid (non-tie) indices."""
    valid_indices = [
        idx for idx, row in enumerate(dataset)
        if row.get("winner") in ("model_a", "model_b")
    ]
    random.seed(seed)
    return sorted(random.sample(valid_indices, sample_size))


def extract_query_and_doc(messages):
    """Extract query (user message) and document (assistant response)."""
    query = next((m["content"] for m in messages if m["role"] == "user"), None)
    doc = next((m["content"] for m in messages if m["role"] == "assistant"), None)
    return query, doc


def evaluate_response(openai_client, prompt_template, response, query):
    """Send evaluation request to OpenAI model and parse JSON output."""
    fluency_prompt = prompt_template.format(query=query, response=response)
    messages = [
        {
            "role": "system",
            "content": (
                "You are Expert system, an intelligent assistant that can analyse "
                "and judge a response based on the provided details."
            ),
        },
        {"role": "user", "content": fluency_prompt},
    ]

    completion = openai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
    )

    try:
        return json.loads(completion.choices[0].message.content)
    except Exception:
        return None


def compute_eval_agreement(comparison_data, metric_name):
    """Compute agreement metrics (accuracy, Kendall's tau, Spearman's rho)."""
    agree = []
    diffs = []
    votes = []

    for row in comparison_data:
        a = row[f"{metric_name}_a"]
        b = row[f"{metric_name}_b"]
        winner = row["human_vote"]
        
        if winner == "model_a":
            pred_winner = "model_a" if a >= b else "model_b"
            agree.append(pred_winner == winner)
        else:
            pred_winner = "model_a" if a > b else "model_b"
            agree.append(pred_winner == winner)

        # For rank correlation (kendall/spearman)
        diff = a - b
        label = 1 if winner == "model_a" else -1
        diffs.append(diff)
        votes.append(label)

    acc = np.mean(agree)
    kendall_corr = kendalltau(diffs, votes).correlation
    spearman_corr = spearmanr(diffs, votes).correlation

    print(f"\n== {metric_name.capitalize()} ==")
    print(f"Agreement with human vote: {acc:.3f}")
    print(f"Kendall's tau: {kendall_corr:.3f}")
    print(f"Spearman's rho: {spearman_corr:.3f}")


def main():
    # Configuration
    cache_dir = "/store2/scratch/sjupadhy/huggingface"
    prompt_file = "fluency_prompt.txt"
    output_csv = "model_vs_human_comparison.csv"

    # Load dataset and sample subset
    ds = load_test_dataset(cache_dir)
    sample_indices = get_sample_indices(ds)

    # Load evaluation prompt
    with open(prompt_file) as f:
        prompt_template = f.read().strip()

    # Initialize OpenAI client
    openai_client = OpenAI(api_key="")

    fluency_aggr = defaultdict(lambda: [0.0, 0.0, 0.0])  # [fluency, grammar, readability]
    comparison_data = []

    # Process sample
    for i, idx in tqdm(enumerate(sample_indices), total=len(sample_indices)):
        row = ds[idx]
        human_vote = row.get("winner")
        if human_vote == "tie":
            continue

        messages_a, messages_b = row.get("messages_a"), row.get("messages_b")
        if not messages_a or not messages_b:
            continue

        query_a, response_a = extract_query_and_doc(messages_a)
        query_b, response_b = extract_query_and_doc(messages_b)

        if query_a and response_a:
            res_a = evaluate_response(openai_client, prompt_template, response_a, query_a)
            if res_a:
                fluency_aggr[f"{idx}_a"] = [
                    res_a.get('Criterion 1: Fluency and Coherence', {}).get('score', 0) / 5,
                    res_a.get('Criterion 2: Grammar and Syntax', {}).get('score', 0) / 5,
                    res_a.get('Criterion 3: Readability and Presentation', {}).get('score', 0) / 5,
                ]

        if query_b and response_b:
            res_b = evaluate_response(openai_client, prompt_template, response_b, query_b)
            if res_b:
                fluency_aggr[f"{idx}_b"] = [
                    res_b.get('Criterion 1: Fluency and Coherence', {}).get('score', 0) / 5,
                    res_b.get('Criterion 2: Grammar and Syntax', {}).get('score', 0) / 5,
                    res_b.get('Criterion 3: Readability and Presentation', {}).get('score', 0) / 5,
                ]

        if f"{idx}_a" not in fluency_aggr or f"{idx}_b" not in fluency_aggr:
            continue

        fluency_a, grammar_a, read_a = fluency_aggr[f"{idx}_a"]
        fluency_b, grammar_b, read_b = fluency_aggr[f"{idx}_b"]

        comparison_data.append({
            "idx": idx,
            "human_vote": human_vote,
            "fluency_a": fluency_a,
            "fluency_b": fluency_b,
            "grammar_a": grammar_a,
            "grammar_b": grammar_b,
            "read_a": read_a,
            "read_b": read_b,
        })

    # Compute correlations
    compute_eval_agreement(comparison_data, "fluency")
    compute_eval_agreement(comparison_data, "grammar")
    compute_eval_agreement(comparison_data, "read")

    # Save results
    df = pd.DataFrame(comparison_data)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()
