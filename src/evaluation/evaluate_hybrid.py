from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# File loading helpers

# Load a specified JSONL file into a list of dictionaries.
# Every populated line is parsed as a single JSON object.
def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            # If any line fails to parse, display an error message with the line number for simple debugging.
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL at {path}:{line_number}: {e}")
    return rows


# Load query relevance labels and change them into a mapping from query ID to a set of relevant resource IDs.
def load_relevance_labels(path: Path) -> Dict[str, Set[str]]:
    qrels_raw = json.loads(path.read_text(encoding="utf-8"))
    qrels: Dict[str, Set[str]] = {}

    if isinstance(qrels_raw, list):
        for entry in qrels_raw:
            if not isinstance(entry, dict):
                continue
            qid = str(entry.get("qid", "")).strip()
            rel = entry.get("relevant_ids", [])
            if not qid:
                continue
            if not isinstance(rel, list):
                raise ValueError(f"qrels for {qid} must be a list of resource IDs")
            qrels[qid] = {str(x).strip() for x in rel if str(x).strip()}

    # Alternatively, qrels can be displayed as a dict mapping qid to relevant_ids.
    elif isinstance(qrels_raw, dict):
        for qid, rel in qrels_raw.items():
            qid = str(qid).strip()
            if not qid:
                continue
            if not isinstance(rel, list):
                raise ValueError(f"qrels for {qid} must be a list of resource IDs")
            qrels[qid] = {str(x).strip() for x in rel if str(x).strip()}

    else:
        raise ValueError(
            "qrels.json must be either: "
            "(a) a list of objects with keys {qid, relevant_ids}, or "
            "(b) a dict mapping qid -> [relevant_ids]"
        )

    return qrels


# Retrieval metrics

# Calculate Evaluation metrics (Precision, Recall, F1) for a single ranked list of resource IDs.
def precision_recall_f1_at_k(ranking: List[str], relevant: Set[str], k: int) -> Tuple[float, float, float]:
    top_k_ids = ranking[:k]
    if not top_k_ids:
        return (0.0, 0.0, 0.0)

    hits = sum(1 for rid in top_k_ids if rid in relevant)
    prec = hits / len(top_k_ids) if top_k_ids else 0.0
    rec = hits / len(relevant) if relevant else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return (prec, rec, f1)


# Calculate Average Precision at k for a single ranked list.
def average_precision_at_k(ranking: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0

    top_k_ids = ranking[:k]
    num_hits = 0
    precisions: List[float] = []

    for i, rid in enumerate(top_k_ids, start=1):
        if rid in relevant:
            num_hits += 1
            precisions.append(num_hits / i)

    if not precisions:
        return 0.0

    return sum(precisions) / len(relevant)


# Calculate Discounted Cumulative Gain at k for each ranked list using binary relevance.
def dcg_at_k(ranking: List[str], relevant: Set[str], k: int) -> float:
    top_k_ids = ranking[:k]
    dcg = 0.0
    for i, rid in enumerate(top_k_ids, start=1):
        rel_i = 1.0 if rid in relevant else 0.0
        if rel_i > 0:
            dcg += rel_i / math.log2(i + 1)
    return dcg


# Calculate normalised DCG at k by comparing the ranking to an ideal ranking.
def ndcg_at_k(ranking: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0

    actual_dcg = dcg_at_k(ranking, relevant, k)
    ideal_ranking = list(relevant)[:k]
    ideal = dcg_at_k(ideal_ranking, relevant, k)
    if ideal == 0:
        return 0.0
    return actual_dcg / ideal


# Compute the average mean in the list of values.
def list_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# Hybrid run evaluation
# Assess the hybrid run retrieval build agains qrels and return both per-query and macro-average metrics.
def evaluate_hybrid_outputs(run_rows: List[Dict[str, Any]], qrels: Dict[str, Set[str]], k: int) -> Dict[str, Any]:
    per_query: Dict[str, Any] = {}

    # Lists to store metric output values so macro averages can be calculated at the end.
    p_list: List[float] = []
    r_list: List[float] = []
    f_list: List[float] = []
    ap_list: List[float] = []
    ndcg_list: List[float] = []

    used_queries = 0

    # Take every judged query in isolation and evaluate it, then store the overall and query-specific outputs.
    for row in run_rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue

        relevant = qrels.get(qid, set())
        if len(relevant) == 0:
            continue

        # Take the ranked reource IDs from the hybrid execution output.
        ranking_items = row.get("results", []) or []
        ranking = [str(item.get("rid")) for item in ranking_items if "rid" in item]

        # Calculate all evaluation metrics for every query at the specified threshld cutoff at k.
        p, r, f1 = precision_recall_f1_at_k(ranking, relevant, k)
        ap = average_precision_at_k(ranking, relevant, k)
        ndcg = ndcg_at_k(ranking, relevant, k)

        per_query[qid] = {
            "query": row.get("query", ""),
            f"P@{k}": p,
            f"R@{k}": r,
            f"F1@{k}": f1,
            f"AP@{k}": ap,
            f"nDCG@{k}": ndcg,
            "retrieved@k": len(ranking[:k]),
            "relevant_total": len(relevant),
            "hits@k": sum(1 for rid in ranking[:k] if rid in relevant),
        }

        p_list.append(p)
        r_list.append(r)
        f_list.append(f1)
        ap_list.append(ap)
        ndcg_list.append(ndcg)
        used_queries += 1

    # Output macro average and per-query metric build results.
    return {
        "k": k,
        "macro_avg": {
            f"P@{k}": list_mean(p_list),
            f"R@{k}": list_mean(r_list),
            f"F1@{k}": list_mean(f_list),
            f"MAP@{k}": list_mean(ap_list),
            f"nDCG@{k}": list_mean(ndcg_list),
            "num_queries_used": used_queries,
        },
        "per_query": per_query,
    }


# Console output helper

# Print a full summary table of all the macro-average metrics in terminal for comparison and evaluation.
def show_metrics_summary(k: int, macro: Dict[str, float]) -> None:
    def fmt(x: float) -> str:
        return f"{x:.3f}"

    print("\nHybrid evaluation (macro-average over judged queries)")
    print("-" * 76)
    print(f"{'P@'+str(k):>10} {'R@'+str(k):>10} {'F1@'+str(k):>10} {'MAP@'+str(k):>10} {'nDCG@'+str(k):>10}")
    print("-" * 76)
    print(
        f"{fmt(macro['P@'+str(k)]):>10} "
        f"{fmt(macro['R@'+str(k)]):>10} "
        f"{fmt(macro['F1@'+str(k)]):>10} "
        f"{fmt(macro['MAP@'+str(k)]):>10} "
        f"{fmt(macro['nDCG@'+str(k)]):>10}"
    )
    print("-" * 76)


# Command-line argument structure

# Execute the command-line parser for hybrid retrieval judgement.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.evaluation.evaluate_hybrid",
        description="Evaluate a hybrid run JSONL against qrels.",
    )
    p.add_argument("--qrels", type=Path, required=True, help="Path to qrels JSON")
    p.add_argument("--run", type=Path, required=True, help="Path to hybrid run JSONL")
    p.add_argument("--out", type=Path, required=True, help="Output metrics JSON path")
    p.add_argument("--k", type=int, default=20, help="Cutoff k")
    return p


# Main pipeline

# Load all inputs, analyse the hybrid run against the qrels, save output artefacts, and print a summary of the results.
def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if not args.qrels.exists():
        raise FileNotFoundError(f"Missing qrels file: {args.qrels}")
    if not args.run.exists():
        raise FileNotFoundError(f"Missing hybrid run file: {args.run}")

    # Load the qrels and hybrid ranking outputs.
    qrels = load_relevance_labels(args.qrels)
    run_rows = load_jsonl_rows(args.run)
    metrics = evaluate_hybrid_outputs(run_rows, qrels, args.k)

    # Record the evaluation results and print a brief summary for quick assessment.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    show_metrics_summary(args.k, metrics["macro_avg"])
    print(f"\nSaved metrics to: {args.out}")


if __name__ == "__main__":
    main()