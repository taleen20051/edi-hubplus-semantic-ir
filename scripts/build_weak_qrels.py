import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


# ----------------------------
# Defaults (kept for backwards compatibility)
# ----------------------------
DEFAULT_RESOURCES_CSV = Path("data/processed/resources_included_only.csv")
DEFAULT_QUERIES_JSONL = Path("data/phase7_evaluation/queries_v2_iter01_expanded.jsonl")
DEFAULT_OUT_QRELS = Path("data/phase8_iterations/iter01/qrels_weak_iter01.jsonl")

TAG_COL_CANDIDATES = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
    "Tags",
    "tag",
    "tags",
    "Theme",
    "Category",
    "Subcategory",
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def tokens(text: str) -> List[str]:
    # Keep 4+ chars to avoid over-triggering on short stopwords.
    return [t for t in re.findall(r"[a-z0-9\+\-]+", norm(text)) if len(t) >= 4]


def load_resources_blob(resources_csv: Path) -> Dict[str, str]:
    """Load resources CSV and build a lightweight tag/title blob per rid."""
    resources: Dict[str, str] = {}
    with resources_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        tag_cols = [c for c in TAG_COL_CANDIDATES if c in cols] or cols
        rid_col = "id" if "id" in cols else ("rid" if "rid" in cols else ("ID" if "ID" in cols else cols[0]))

        for row in reader:
            rid = str(row.get(rid_col, "")).strip()
            if not rid:
                continue

            blob = " | ".join(norm(row.get(c, "")) for c in tag_cols)
            title_blob = (norm(row.get("Title", "")) + " " + norm(row.get("title", ""))).strip()
            resources[rid] = (blob + " " + title_blob).strip()

    return resources


def load_queries(queries_jsonl: Path) -> List[Tuple[str, str]]:
    """Load queries JSONL as (qid, query_text)."""
    queries: List[Tuple[str, str]] = []
    with queries_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qid = str(q["qid"]).strip()
            qtext = q.get("expanded_query") or q.get("query") or ""
            queries.append((qid, str(qtext)))
    return queries


def iter_pooled_candidates_from_run_jsonl(run_path: Path, top_k: int) -> Dict[str, Set[str]]:
    """Parse a run JSONL and return pooled candidates per qid.

    Supports two common shapes:
    1) One line per query with a `ranking` list:
       {"qid":"q1", "ranking":[{"rid":"44","score":...}, ...]}

    2) One line per hit:
       {"qid":"q1", "rid":"44", "score":...}
       or {"qid":"q1", "docid":"44", "score":...}

    In (2) we take top_k by score per qid.
    """
    pooled: Dict[str, Set[str]] = defaultdict(set)

    # First pass: detect format by peeking at first non-empty line
    first_obj = None
    with run_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first_obj = json.loads(line)
            break

    if first_obj is None:
        return pooled

    # Format (1): per-query ranking list
    if isinstance(first_obj, dict) and "ranking" in first_obj and isinstance(first_obj.get("ranking"), list):
        with run_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = str(obj.get("qid", "")).strip()
                if not qid:
                    continue
                ranking = obj.get("ranking") or []
                for item in ranking[:top_k]:
                    if not isinstance(item, dict):
                        continue
                    rid = str(item.get("rid") or item.get("docid") or item.get("id") or "").strip()
                    if rid:
                        pooled[qid].add(rid)
        return pooled

    # Format (2): per-hit lines -> collect and top-k by score per qid
    hits_by_qid: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    def _score(x) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    def _consume_obj(obj: dict):
        qid = str(obj.get("qid", "")).strip()
        if not qid:
            return
        rid = str(obj.get("rid") or obj.get("docid") or obj.get("id") or "").strip()
        if not rid:
            return
        score = _score(obj.get("score", 0.0))
        hits_by_qid[qid].append((rid, score))

    _consume_obj(first_obj)
    with run_path.open("r", encoding="utf-8") as f:
        # skip the first line already processed
        skipped_first = False
        for line in f:
            if not skipped_first:
                # consume the actual first line again only once; easiest is to skip it
                skipped_first = True
                continue
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                _consume_obj(obj)

    for qid, hits in hits_by_qid.items():
        hits.sort(key=lambda x: x[1], reverse=True)
        for rid, _ in hits[:top_k]:
            pooled[qid].add(rid)

    return pooled


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build weak qrels JSONL. Optionally pools candidates from multiple run files (top-k per run per query).\n"
            "Weak relevance rule: rel=1 if >=min_hits query tokens (len>=min_token_len) appear in resource tag/title blob."
        )
    )
    p.add_argument("--resources-csv", type=Path, default=DEFAULT_RESOURCES_CSV,
                   help=f"Resources CSV used to build tag/title blobs (default: {DEFAULT_RESOURCES_CSV})")
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_JSONL,
                   help=f"Queries JSONL (default: {DEFAULT_QUERIES_JSONL})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_QRELS,
                   help=f"Output weak qrels JSONL (default: {DEFAULT_OUT_QRELS})")
    p.add_argument("--runs", type=Path, nargs="*", default=[],
                   help="Optional: one or more run JSONL files to pool candidate (qid,rid) pairs from.")
    p.add_argument("--top-k", type=int, default=20,
                   help="Top-k per run per query to include in the pooled candidate set (default: 20)")
    p.add_argument("--min-token-len", type=int, default=4,
                   help="Minimum query token length to consider (default: 4)")
    p.add_argument("--min-hits", type=int, default=2,
                   help="Weak relevance threshold: minimum number of token hits to mark rel=1 (default: 2)")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.resources_csv.exists():
        raise FileNotFoundError(f"Missing resources CSV: {args.resources_csv}")
    if not args.queries.exists():
        raise FileNotFoundError(f"Missing queries JSONL: {args.queries}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    resources = load_resources_blob(args.resources_csv)
    queries = load_queries(args.queries)

    # Pool candidates if run files are provided; otherwise fall back to dense matrix (old behaviour).
    pooled_candidates: Dict[str, Set[str]] = defaultdict(set)
    if args.runs:
        for run_path in args.runs:
            if not run_path.exists():
                raise FileNotFoundError(f"Missing run file: {run_path}")
            per_run_pool = iter_pooled_candidates_from_run_jsonl(run_path, top_k=args.top_k)
            for qid, rids in per_run_pool.items():
                pooled_candidates[qid].update(rids)

    def _tokens_custom(text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9\+\-]+", norm(text)) if len(t) >= int(args.min_token_len)]

    total_pairs = 0
    with args.out.open("w", encoding="utf-8") as out:
        for qid, qtext in queries:
            qtok_set = set(_tokens_custom(qtext))

            if pooled_candidates:
                # pooled mode
                cand_rids = sorted(pooled_candidates.get(qid, set()))
            else:
                # dense mode (legacy)
                cand_rids = list(resources.keys())

            for rid in cand_rids:
                tagtext = resources.get(rid, "")
                hits = sum(1 for t in qtok_set if t and t in tagtext)
                rel = 1 if hits >= int(args.min_hits) else 0
                out.write(json.dumps({"qid": qid, "rid": rid, "rel": rel}) + "\n")
                total_pairs += 1

    mode = "pooled" if pooled_candidates else "dense"
    print(
        f"Wrote weak qrels to: {args.out}  (mode={mode}, queries={len(queries)}, resources={len(resources)}, pairs={total_pairs})"
    )

    if pooled_candidates:
        # Helpful diagnostic: how many candidates per query on average
        sizes = [len(pooled_candidates.get(qid, set())) for qid, _ in queries]
        if sizes:
            avg = sum(sizes) / len(sizes)
            print(f"Pooled candidates per query: min={min(sizes)}, max={max(sizes)}, avg={avg:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())