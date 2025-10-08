import os
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------
# Global Configuration
# ----------------------------------
CACHE_DIR = "/store2/scratch/sjupadhy/hugging_face"
DATA_PATH = "/mnt/users/s8sharif/search_arena/gpt41_retrieved_urls_and_responses/nuggets/"
OUTPUT_JSON = Path("nugget_analysis_results.json")
JACCARD_PLOT = "similarity_analysis_summary_jaccard.pdf"
SEMANTIC_PLOT = "similarity_analysis_summary_semantic.pdf"


# ----------------------------------
# Helper Functions
# ----------------------------------
def load_model():
    """Load SentenceTransformer model once."""
    print("Loading sentence embedding model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder=CACHE_DIR)
    model.to("cuda")
    device = "cuda" if model._target_device.type == "cuda" else "cpu"
    return model, device


def load_nuggets(path):
    """Load all nugget files from a directory."""
    nugg_paths = os.listdir(path)
    all_nuggets = []
    for filename in tqdm(nugg_paths, desc="Loading nuggets"):
        file_path = os.path.join(path, filename)
        with open(file_path, "r") as f:
            data = json.load(f)
            nuggets = data.get("scored_nuggets", [])
            if nuggets:
                all_nuggets.append(nuggets)
    return all_nuggets


def unigram_jaccard_similarity(a, b):
    """Compute Jaccard similarity over unigrams."""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union != 0 else 0.0


def analyze_nuggets(nuggets, model, device, embed_batch_size=32):
    """Compute lexical and semantic similarity for one nugget list."""
    texts = [n["text"] for n in nuggets]

    # Lexical overlap
    n = len(texts)
    jaccard_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score = unigram_jaccard_similarity(texts[i], texts[j])
            jaccard_matrix[i, j] = jaccard_matrix[j, i] = score

    # Semantic similarity
    embeddings = model.encode(
        texts,
        batch_size=embed_batch_size,
        device=device,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    semantic_matrix = cosine_similarity(embeddings)

    return {
        "jaccard_matrix": jaccard_matrix,
        "semantic_matrix": semantic_matrix,
    }


def analyze_all_nugget_sets(nugget_sets, model, device):
    """Run similarity analysis across all nugget sets."""
    results = {}
    for i, nugget_list in tqdm(enumerate(nugget_sets), desc="Analyzing"):
        if nugget_list:
            results[i] = analyze_nuggets(nugget_list, model, device)
    return results


def serialize_result(result):
    """Convert numpy arrays to serializable lists."""
    return {
        "jaccard_matrix": result["jaccard_matrix"].tolist(),
        "semantic_matrix": result["semantic_matrix"].tolist(),
    }


def extract_pairwise_scores(matrices):
    """Extract average pairwise upper-triangle similarity scores."""
    all_scores = []
    for mat in matrices:
        arr = np.array(mat)
        if arr.shape[0] < 2:
            continue
        triu_vals = arr[np.triu_indices_from(arr, k=1)]
        all_scores.append(np.mean(triu_vals))
    return np.array(all_scores)


def plot_similarity(scores, xlabel, filename):
    """Plot histogram + KDE of similarity scores."""
    plt.figure(figsize=(8, 5))
    sns.histplot(scores, bins=30, kde=True, color="skyblue")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def print_summary_stats(name, scores):
    """Print descriptive statistics for similarity scores."""
    print(f"\n{name} Similarity:")
    print(
        f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}, "
        f"Min: {scores.min():.3f}, Max: {scores.max():.3f}"
    )


# ----------------------------------
# Main Pipeline
# ----------------------------------
def main():
    # Load model
    model, device = load_model()

    # Load nuggets
    all_nuggets = load_nuggets(DATA_PATH)

    # Analyze all sets
    results = analyze_all_nugget_sets(all_nuggets, model, device)

    # Serialize and save results
    for i in results:
        results[i] = serialize_result(results[i])

    with OUTPUT_JSON.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved analysis results to: {OUTPUT_JSON.resolve()}")

    # Extract matrices
    jaccard_matrices = [results[r]["jaccard_matrix"] for r in results]
    semantic_matrices = [results[r]["semantic_matrix"] for r in results]

    # Compute mean pairwise similarities
    jaccard_scores = extract_pairwise_scores(jaccard_matrices)
    semantic_scores = extract_pairwise_scores(semantic_matrices)

    # Plot and save
    plot_similarity(jaccard_scores, "Jaccard Score", JACCARD_PLOT)
    plot_similarity(semantic_scores, "Cosine Score", SEMANTIC_PLOT)

    # Print stats
    print_summary_stats("Jaccard", jaccard_scores)
    print_summary_stats("Semantic", semantic_scores)


if __name__ == "__main__":
    main()
