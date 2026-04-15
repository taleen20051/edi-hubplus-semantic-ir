#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ----------------------------
# Loading helpers
# ----------------------------

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_qrels(qrels_obj: Any) -> Dict[str, Dict[str, float]]:
    """Normalize qrels into: { qid: { rid: rel_score(float) } }

    Supports common shapes:
      1) {"q1": ["13","19"], "q2": []}
      2) {"q1": {"13": 1, "19": 2}, "q2": {}}
      3) [{"qid":"q1","rid":"13","rel":1}, ...]
      4) {"qrels": [{"qid":..., "rid":..., "rel":...}, ...]}
    """
    # Case 4
    if isinstance(qrels_obj, dict) and "qrels" in qrels_obj and isinstance(qrels_obj["qrels"], list):
        out: Dict[str, Dict[str, float]] = defaultdict(dict)
        for it in qrels_obj["qrels"]:
            qid = str(it["qid"])
            rid = str(it["rid"])
            rel = float(it.get("rel", it.get("relevance", 1)))
            out[qid][rid] = rel
        return dict(out)

    # Case 1 or 2
    if isinstance(qrels_obj, dict):
        out: Dict[str, Dict[str, float]] = {}
        for qid, val in qrels_obj.items():
            qid = str(qid)
            if isinstance(val, list):
                # binary relevance
                out[qid] = {str(rid): 1.0 for rid in val}
            elif isinstance(val, dict):
                out[qid] = {str(rid): float(rel) for rid, rel in val.items()}
            else:
                raise ValueError(f"Unsupported qrels value type for {qid}: {type(val)}")
        return out

    # Case 3
    if isinstance(qrels_obj, list):
        out = defaultdict(dict)
        for it in qrels_obj:
            qid = str(it["qid"])
            rid = str(it["rid"])
            rel = float(it.get("rel", it.get("relevance", 1)))
            out[qid][rid] = rel
        return dict(out)

    raise ValueError("Unsupported qrels JSON format.")


def load_run_jsonl(path: str | Path) -> Dict[str, List[str]]:
    """Load run file into: { qid: [rid1, rid2, ...] }

    Supports JSONL lines like:
      A) {"qid": "q1", "ranking": [{"rid": "44", "score": 1.23}, ...]}
      B) {"qid": "q1", "results": [{"rid": "44", "rank": 1, ...}, ...]}

    Notes:
      - If 'rank' is present, we sort by rank.
      - Otherwise we keep the list order as-is.
    """
    out: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for obj in iter_jsonl(path):
        qid = str(obj["qid"])
        ranking = obj.get("ranking")
        if ranking is None:
            ranking = obj.get("results")
        if ranking is None:
            raise ValueError(f"Run line for {qid} has no 'ranking' or 'results' field")

        # If any items have rank, sort; else keep insertion order
        has_rank = any(isinstance(item, dict) and ("rank" in item) for item in ranking)
        if has_rank:
            for item in ranking:
                rid = str(item["rid"])
                rank = int(item.get("rank", 10**9))
                out[qid].append((rank, rid))
        else:
            for i, item in enumerate(ranking, start=1):
                rid = str(item["rid"])
                out[qid].append((i, rid))

    final: Dict[str, List[str]] = {}
    for qid, pairs in out.items():
        pairs.sort(key=lambda x: x[0])
        final[qid] = [rid for _, rid in pairs]
    return final


# ----------------------------
# Metrics
# ----------------------------

def precision_at_k(ranked: List[str], rel: Dict[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for rid in top if rel.get(rid, 0.0) > 0.0)
    return hits / float(k)


def recall_at_k(ranked: List[str], rel: Dict[str, float], k: int) -> float:
    total_rel = sum(1 for _, v in rel.items() if v > 0.0)
    if total_rel == 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for rid in top if rel.get(rid, 0.0) > 0.0)
    return hits / float(total_rel)


def f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision_at_k(ranked: List[str], rel: Dict[str, float], k: int) -> float:
    """AP@k (binary relevance: rel>0).

    Normalized by total number of relevant docs for the query (standard MAP).
    """
    top = ranked[:k]
    total_rel = sum(1 for _, v in rel.items() if v > 0.0)
    if total_rel == 0:
        return 0.0

    num_rel_seen = 0
    sum_prec = 0.0
    for i, rid in enumerate(top, start=1):
        if rel.get(rid, 0.0) > 0.0:
            num_rel_seen += 1
            sum_prec += num_rel_seen / float(i)

    return sum_prec / float(total_rel)


def dcg_at_k(ranked: List[str], rel: Dict[str, float], k: int) -> float:
    """DCG@k (graded relevance).

    gain = 2^rel - 1
    discount = log2(i+1)
    """
    top = ranked[:k]
    score = 0.0
    for i, rid in enumerate(top, start=1):
        r = float(rel.get(rid, 0.0))
        gain = (2.0 ** r) - 1.0
        discount = math.log2(i + 1.0)
        score += gain / discount
    return score


def ndcg_at_k(ranked: List[str], rel: Dict[str, float], k: int) -> float:
    dcg = dcg_at_k(ranked, rel, k)
    # Ideal ordering by rel desc
    ideal_ranked = [rid for rid, _ in sorted(rel.items(), key=lambda x: x[1], reverse=True)]
    idcg = dcg_at_k(ideal_ranked, rel, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to run JSONL (bm25 / ontology_only / hybrid)")
    ap.add_argument("--qrels", required=True, help="Path to qrels JSON")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--include_empty_qrels",
        action="store_true",
        help="If set, include queries with empty relevant sets in macro-average (usually BAD).",
    )
    args = ap.parse_args()

    qrels_raw = load_json(args.qrels)
    qrels = normalize_qrels(qrels_raw)
    run = load_run_jsonl(args.run)
    k = int(args.k)

    used_qids: List[str] = []
    p_list: List[float] = []
    r_list: List[float] = []
    f_list: List[float] = []
    ap_list: List[float] = []
    ndcg_list: List[float] = []

    for qid, rel_map in qrels.items():
        total_rel = sum(1 for _, v in rel_map.items() if v > 0.0)
        if (not args.include_empty_qrels) and total_rel == 0:
            continue

        ranked = run.get(qid, [])
        p = precision_at_k(ranked, rel_map, k)
        r = recall_at_k(ranked, rel_map, k)
        f = f1(p, r)
        ap_k = average_precision_at_k(ranked, rel_map, k)
        ndcg_k = ndcg_at_k(ranked, rel_map, k)

        used_qids.append(qid)
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        ap_list.append(ap_k)
        ndcg_list.append(ndcg_k)

    if not used_qids:
        print("No queries evaluated (did you skip empties and all qrels were empty?)")
        return 2

    macro = {
        f"P@{k}": sum(p_list) / len(p_list),
        f"R@{k}": sum(r_list) / len(r_list),
        f"F1@{k}": sum(f_list) / len(f_list),
        f"MAP@{k}": sum(ap_list) / len(ap_list),
        f"nDCG@{k}": sum(ndcg_list) / len(ndcg_list),
        "num_queries_used": len(used_qids),
    }

    header = "Macro average (over judged queries)" if not args.include_empty_qrels else "Macro average (including empty qrels)"
    print(header)
    print(json.dumps(macro, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())