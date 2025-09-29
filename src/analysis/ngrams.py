import spacy
from typing import List, Tuple, Dict, Any
from collections import Counter, defaultdict
from tqdm import tqdm

from datasets import load_dataset

# Optional: semantic-ish tie-breaker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: robust token set similarity
try:
    from rapidfuzz.fuzz import token_set_ratio
except Exception:
    token_set_ratio = None

############################
# Text preprocessing utils #
############################

# Initialize spaCy and tokenizer globally
nlp = spacy.load("xx_sent_ud_sm")
nlp.max_length = 2_000_000_000


def sentence_split(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def tokenize(text: str) -> List[str]:
    return [t.text for t in nlp.tokenizer(text)]


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def jaccard(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values())
    union = sum((a | b).values()) or 1
    return inter / union


def containment(a: Counter, b: Counter) -> float:
    # How much of a is covered by b (len-wise, multiset version)
    inter = sum((a & b).values())
    denom = sum(a.values()) or 1
    return inter / denom


######################
# ROUGE-L (LCS-based)#
######################


def _lcs_len(x: List[str], y: List[str]) -> int:
    # O(len(x)*len(y)) DP
    dp = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        prev = 0
        for j in range(1, len(y) + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(
    ref_tokens: List[str], hyp_tokens: List[str], beta: float = 1.2
) -> float:
    # ref=hyp naming not critical; we want an F1-like overlap score
    lcs = _lcs_len(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    prec = lcs / (len(hyp_tokens) or 1)
    rec = lcs / (len(ref_tokens) or 1)
    if prec == 0 and rec == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * (prec * rec) / (rec + beta2 * prec)


###################################
# Candidate retrieval (n-gram idx)#
###################################


class NgramIndex:
    """Inverted index from n-gram to doc sentence IDs to prune candidates."""

    def __init__(self, doc_sentences: List[str], n: int = 3):
        self.n = n
        self.doc_sents = doc_sentences
        self.doc_tok = [tokenize(s) for s in doc_sentences]
        self.index = defaultdict(set)
        for sid, toks in enumerate(self.doc_tok):
            grams = set(ngrams(toks, n)) if len(toks) >= n else set()
            for g in grams:
                self.index[g].add(sid)

    def candidates(
        self, query_tokens: List[str], backoff: bool = True, k: int = 200
    ) -> List[int]:
        n = self.n
        grams = list(ngrams(query_tokens, n))
        cand = set()
        if grams:
            for g in grams:
                cand |= self.index.get(g, set())
        # Backoff to bigrams if no hits
        if backoff and not cand and n > 1:
            grams2 = list(ngrams(query_tokens, n - 1))
            for g in grams2:
                cand |= self.index.get(g, set())
        # If still none, consider all (degenerate)
        if not cand:
            cand = set(range(len(self.doc_sents)))
        # return up to k
        return list(cand)[:k]


############################################
# Scoring one gen sentence vs doc sentence #
############################################


def score_pair(gen_toks: List[str], doc_toks: List[str]) -> Dict[str, float]:
    uni_g, uni_d = Counter(gen_toks), Counter(doc_toks)
    bi_g, bi_d = Counter(ngrams(gen_toks, 2)), Counter(ngrams(doc_toks, 2))
    tri_g, tri_d = Counter(ngrams(gen_toks, 3)), Counter(ngrams(doc_toks, 3))

    scores = {
        # "jaccard_uni": jaccard(uni_g, uni_d),
        # "jaccard_bi":  jaccard(bi_g, bi_d) if bi_g and bi_d else 0.0,
        # "contain_uni": containment(uni_g, uni_d),
        # "contain_bi":  containment(bi_g, bi_d) if bi_g else 0.0,
        # "contain_tri": containment(tri_g, tri_d) if tri_g else 0.0,
        "rougeL": rouge_l_f1(doc_toks, gen_toks),
    }
    # if token_set_ratio is not None:
    #     # Scaled 0..1
    #     scores["token_set_ratio"] = token_set_ratio(" ".join(gen_toks), " ".join(doc_toks)) / 100.0
    return scores


def composite_score(scores: Dict[str, float]) -> float:
    # Weighted blend; tune weights to your taste or ablate them.
    w = {
        "rougeL": 1,
        # "contain_tri": 0.20,
        # "contain_bi": 0.15,
        # "contain_uni": 0.10,
        # "jaccard_bi": 0.5,
        # "jaccard_uni": 0.05,
        # "token_set_ratio": 0.05 if "token_set_ratio" in scores else 0.0
    }
    return sum(scores[k] * w.get(k, 0.0) for k in scores)


#############################
# Evidence highlighting     #
#############################


def highlight_supported(gen_toks: List[str], doc_toks: List[str]) -> str:
    # Simple scheme: highlight tokens in gen that appear in doc (multi-set aware).
    doc_counter = Counter(doc_toks)
    out = []
    for tok in gen_toks:
        if doc_counter[tok] > 0:
            out.append(f"[{tok}]")  # mark supported tokens
            doc_counter[tok] -= 1
        else:
            out.append(tok)
    return " ".join(out)


#############################################
# Main: evaluate one generation vs documents#
#############################################


def get_completion(row, key):
    message = row[f"messages_{key}"][1]
    assert message["role"] == "assistant"
    return message["content"]


def evaluate_groundedness(
    generation: str,
    doc_chunks: List[str],
    k_candidates: int = 200,
    coverage_threshold: float = 0.5,
    use_tfidf_tiebreak: bool = True,
) -> Dict[str, Any]:
    # 1) Prepare document sentences and index
    doc_sents = [s for chunk in doc_chunks for s in sentence_split(chunk)]
    index = NgramIndex(doc_sents, n=3)

    # Optional TF-IDF matrix for tie-breaking
    tfidf = None
    doc_vecs = None
    if use_tfidf_tiebreak:
        tfidf = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), min_df=1
        )
        doc_vecs = tfidf.fit_transform(doc_sents)

    # 2) Split generation into sentences
    gen_sents = sentence_split(generation)
    results = []
    supported = 0

    for i, gs in enumerate(gen_sents):
        gtoks = tokenize(gs)
        if not gtoks:
            results.append(
                {
                    "gen_sent_id": i,
                    "gen_sentence": gs,
                    "best_doc_sent": "",
                    "best_doc_id": None,
                    "scores": {},
                    "composite": 0.0,
                    "coverage_hit": False,
                    "highlighted": "",
                }
            )
            continue

        cand_ids = index.candidates(gtoks, backoff=True, k=k_candidates)
        best = None
        best_score = -1.0
        best_scores = None

        # Rank candidates
        for sid in cand_ids:
            dtoks = index.doc_tok[sid]
            scores = score_pair(gtoks, dtoks)
            s = composite_score(scores)

            if s > best_score + 1e-9:
                best_score = s
                best = sid
                best_scores = scores
            elif abs(s - best_score) <= 1e-9 and use_tfidf_tiebreak:
                # Tie-break with TF-IDF cosine
                qv = tfidf.transform([" ".join(gtoks)])
                cos = cosine_similarity(qv, doc_vecs[best])[0, 0]
                cos2 = cosine_similarity(qv, doc_vecs[sid])[0, 0]
                if cos2 > cos:
                    best = sid
                    best_scores = scores
                    best_score = s

        matched = best is not None
        doc_sent = doc_sents[best] if matched else ""
        cov_hit = best_score >= coverage_threshold
        if cov_hit:
            supported += 1

        highlighted = highlight_supported(gtoks, tokenize(doc_sent)) if matched else ""

        results.append(
            {
                "gen_sent_id": i,
                "gen_sentence": gs,
                "best_doc_id": best,
                "best_doc_sent": doc_sent,
                "scores": best_scores or {},
                "composite": round(best_score, 4),
                "coverage_hit": cov_hit,
                "highlighted": highlighted,
            }
        )

    coverage = supported / (len(gen_sents) or 1)

    # Aggregate averages (ignoring empty-score rows)
    agg = defaultdict(list)
    for r in results:
        for k, v in r["scores"].items():
            agg[k].append(v)
    avg_scores = {k: (sum(vs) / len(vs) if vs else 0.0) for k, vs in agg.items()}
    return {
        "coverage@tau": coverage,
        "tau": coverage_threshold,
        "avg_scores": avg_scores,
        "num_gen_sents": len(gen_sents),
        "sentence_matches": results,
    }


######################################
# Compare two generations end-to-end #
######################################


def compare_two_generations(
    gen_a: str, gen_b: str, doc_chunks: List[str], coverage_threshold: float = 0.5
) -> Dict[str, Any]:
    A = evaluate_groundedness(gen_a, doc_chunks, coverage_threshold=coverage_threshold)
    B = evaluate_groundedness(gen_b, doc_chunks, coverage_threshold=coverage_threshold)
    # Simple side-by-side summary
    summary = {
        "coverage@tau": {
            "A": A["coverage@tau"],
            "B": B["coverage@tau"],
            "tau": coverage_threshold,
        },
        "avg_scores": {"A": A["avg_scores"], "B": B["avg_scores"]},
        "num_gen_sents": {"A": A["num_gen_sents"], "B": B["num_gen_sents"]},
    }
    return {"A": A, "B": B, "summary": summary}


#####################
# Example (pseudo)  #
#####################


import json

if __name__ == "__main__":
    DATASET = "castorini/search-arena-v1-nuggets-with-urls-5k"
    data = load_dataset(DATASET, split="test")
    data_df = data.to_pandas()
    count = 0
    sum_rouge = 0
    with open("ngrams.jsonl", "w") as f:
        for index, row in tqdm(data_df.iterrows()):
            gen_a = get_completion(row, "a")
            gen_b = get_completion(row, "b")
            doc_chunks = [doc["segment"] for doc in row["nugget_documents"][2:]]
            if not doc_chunks:
                continue
            report = compare_two_generations(
                gen_a, gen_b, doc_chunks, coverage_threshold=0.5
            )
            s = json.dumps(
                {
                    "winner": row["winner"],
                    "avg_scores": report["summary"]["avg_scores"],
                },
                ensure_ascii=False,
            )
            count += 2
            sum_rouge += report["summary"]["avg_scores"]["A"]["rougeL"]
            sum_rouge += report["summary"]["avg_scores"]["B"]["rougeL"]
            f.write(s)
            f.write("\n")
    print(sum_rouge / count)
