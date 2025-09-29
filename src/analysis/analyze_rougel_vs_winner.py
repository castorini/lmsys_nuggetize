#!/usr/bin/env python3
import argparse, json, sys, math, csv
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    fh = sys.stdin if path == "-" else open(path, "r", encoding="utf-8")
    rows = []
    with fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"[warn] Skipping line {ln}: JSON decode error: {e}",
                    file=sys.stderr,
                )
    return rows


def parse_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        try:
            a = float(r["avg_scores"]["A"]["rougeL"])
            b = float(r["avg_scores"]["B"]["rougeL"])
            winner = str(r["winner"]).strip().lower()
        except Exception as e:
            # Skip malformed
            continue
        delta = b - a
        if b > a:
            argmax = "model_b"
        elif a > b:
            argmax = "model_a"
        else:
            argmax = "tie"
        out.append(
            {
                "winner": winner,  # e.g., "model_a", "model_b", "tie", "tie (bothbad)"
                "rougeL_A": a,
                "rougeL_B": b,
                "delta_B_minus_A": delta,
                "argmax_by_rouge": argmax,
            }
        )
    return out


def encode_winner_for_corr(label: str, mode: str) -> Optional[float]:
    """
    mode='ab01' -> model_b=1, model_a=0, others=None (drop ties)
    mode='abpm' -> model_b=+1, model_a=-1, others=None (drop ties)
    mode='ab0'  -> model_b=+1, model_a=-1, tie or tie (bothbad)=0 (keep ties)
    """
    lab = label.lower()
    if mode == "ab01":
        if lab == "model_b":
            return 1.0
        if lab == "model_a":
            return 0.0
        return None
    if mode == "abpm":
        if lab == "model_b":
            return 1.0
        if lab == "model_a":
            return -1.0
        return None
    if lab == "model_b":
        return 1.0
    if lab == "model_a":
        return -1.0
    # treat "tie" and "tie (bothbad)" as 0
    return 0.0


def rankdata(x: List[float]) -> List[float]:
    # average ranks for ties; 1..n
    idx = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        # average rank from i..j (1-indexed)
        r = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[idx[k]] = r
        i = j + 1
    return ranks


def pearson_r(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    deny = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def spearman_corr(x: List[float], y: List[float]) -> Tuple[float, float]:
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def pointbiserial_corr(bin_y: List[int], x: List[float]) -> Tuple[float, float]:
    # Bin y is 0/1 for (model_a/model_b). Returns r, p (p if SciPy else nan).
    r, p = stats.pointbiserialr(bin_y, x)
    return float(r), float(p)


def analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # strict A/B only
    strict = [r for r in rows if r["winner"] in ("model_a", "model_b")]
    strict_n = len(strict)

    # accuracy of argmax(ROUGE_L) predicting winner
    strict_hits = sum(1 for r in strict if r["argmax_by_rouge"] == r["winner"])
    strict_acc = strict_hits / strict_n if strict_n else float("nan")

    # confusion matrix (winner -> argmax)
    conf = Counter((r["winner"], r["argmax_by_rouge"]) for r in strict)

    # correlations on strict
    y_ab01 = [encode_winner_for_corr(r["winner"], "ab01") for r in strict]
    x_delta = [r["delta_B_minus_A"] for r in strict]
    r_pb, p_pb = (
        pointbiserial_corr([int(v) for v in y_ab01], x_delta)
        if strict_n
        else (float("nan"), float("nan"))
    )

    y_abpm = [encode_winner_for_corr(r["winner"], "abpm") for r in strict]
    rho_strict, p_s_strict = (
        spearman_corr(y_abpm, x_delta) if strict_n else (float("nan"), float("nan"))
    )

    # overall Spearman including ties (winner ∈ {-1,0,+1} vs sign(delta) ∈ {-1,0,+1})
    y_ab0 = [encode_winner_for_corr(r["winner"], "ab0") for r in rows]
    x_sign = [
        float((r["delta_B_minus_A"] > 0) - (r["delta_B_minus_A"] < 0)) for r in rows
    ]
    rho_all, p_s_all = spearman_corr(y_ab0, x_sign)

    return {
        "n_total": len(rows),
        "n_strict": strict_n,
        "strict_accuracy_argmax_rouge_predicts_winner": strict_acc,
        "confusion_matrix_strict": dict(conf),
        "pointbiserial_corr_winner_vs_delta": {"r": r_pb, "p_value": p_pb},
        "spearman_corr_winner_vs_delta_strict": {
            "rho": rho_strict,
            "p_value": p_s_strict,
        },
        "overall_spearman_corr_including_ties": {"rho": rho_all, "p_value": p_s_all},
    }


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    fields = ["winner", "rougeL_A", "rougeL_B", "delta_B_minus_A", "argmax_by_rouge"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def main():
    ap = argparse.ArgumentParser(
        description="Analyze whether higher average ROUGE-L predicts winner."
    )
    ap.add_argument("--input", "-i", default="-", help="JSONL path (or '-' for stdin)")
    ap.add_argument("--csv", help="Optional path to write per-row CSV")
    args = ap.parse_args()

    records = read_jsonl(args.input)
    rows = parse_rows(records)

    if not rows:
        print("No valid rows parsed.", file=sys.stderr)
        sys.exit(1)

    summary = analyze(rows)

    # Pretty print summary
    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        else:
            print(f"{k}: {v}")

    # Human-friendly confusion matrix
    print("\nConfusion matrix (winner -> argmax_by_rouge) [strict A/B]:")
    labels = ["model_a", "model_b", "tie"]
    # Collect only strict columns actually present
    cols = sorted({c for (_, c) in summary["confusion_matrix_strict"].keys()})
    for w in ["model_a", "model_b"]:
        row = []
        for c in cols:
            row.append(summary["confusion_matrix_strict"].get((w, c), 0))
        print(f"{w:9s} -> {dict(zip(cols, row))}")

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nWrote per-row CSV to: {args.csv}")


if __name__ == "__main__":
    main()
