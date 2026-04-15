from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.retrieval.bm25 import BM25Index


# ----------------------------
# Defaults (match your repo)
# ----------------------------
DEFAULT_ONTOLOGY_JSON = Path("ontology/edi_ontology_v1.json")
DEFAULT_ONTOLOGY_NAME = "v1"
DEFAULT_ONTOLOGY_VERSIONS_DIR = Path("ontology/versions")
DEFAULT_TAXONOMY_CLEAN_CSV = Path("data/processed/taxonomy_clean.csv")

DEFAULT_RESOURCE_CSV_CANDIDATES = [
    Path("data/processed/resources_included_only.csv"),
    Path("data/processed/taxonomy_raw.csv"),
    Path("data/raw/resources_full_data_full.csv"),
]

DEFAULT_RESOURCE_TEXT_JSONL_CANDIDATES = [
    Path("data/processed/resources_text.jsonl"),
    Path("data/processed/resources_unified.jsonl"),
]

DEFAULT_RESOURCE_TEXT_CSV_CANDIDATES = [
    Path("data/processed/resource_text.csv"),
    Path("data/processed/resources_text.csv"),
]

DEFAULT_QUERIES_JSON = Path("data/evaluation/queries.json")
DEFAULT_OUT_DIR = Path("data/baselines")


# ----------------------------
# Helpers
# ----------------------------
def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def tokenize(s: str) -> List[str]:
    s = norm(s)
    if not s:
        return []
    return re.findall(r"[a-z0-9\+\-]+", s)


def split_title_body(text: str) -> Tuple[str, str]:
    text = text or ""
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


def split_tags(cell: Any) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def load_ontology_label_map(ontology_path: Path) -> Dict[str, str]:
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    label_to_concept: Dict[str, str] = {}
    for c in data.get("concepts", []):
        cid = c["id"]
        pref = c.get("pref_label", "")
        if pref:
            label_to_concept[norm(pref)] = cid
        for a in c.get("alt_labels", []) or []:
            if a:
                label_to_concept[norm(str(a))] = cid
    return label_to_concept


def build_concept_to_labels(label_to_concept: Dict[str, str]) -> Dict[str, Set[str]]:
    concept_to_labels: Dict[str, Set[str]] = defaultdict(set)
    for label, cid in label_to_concept.items():
        if label:
            concept_to_labels[cid].add(label)
    return concept_to_labels


def text_contains_label(text: str, label: str) -> bool:
    tn = norm(text)
    lab = norm(label)
    if not tn or not lab:
        return False
    pattern = r"(^|\s)" + re.escape(lab) + r"(\s|$)"
    return re.search(pattern, tn) is not None


def token_overlap_score(text: str, label: str) -> float:
    """
    Fallback soft match for multi-word ontology labels.
    Returns a score in [0,1] based on token coverage.
    """
    text_tokens = set(tokenize(text))
    label_tokens = tokenize(label)

    if not text_tokens or not label_tokens:
        return 0.0

    # single-token labels should remain strict to avoid noise
    if len(label_tokens) == 1:
        return 1.0 if label_tokens[0] in text_tokens else 0.0

    overlap = sum(1 for tok in label_tokens if tok in text_tokens)
    return overlap / float(len(label_tokens))


def concept_match_strength(title: str, body: str, labels: Set[str]) -> Tuple[float, List[str]]:
    """
    Returns:
      best score for this concept in this document
      evidence labels that matched
    """
    best = 0.0
    evidence: List[str] = []

    for label in labels:
        label_n = norm(label)
        if not label_n:
            continue

        # 1) Exact phrase in title -> strongest
        if text_contains_label(title, label_n):
            score = 3.0
        # 2) Exact phrase in body
        elif text_contains_label(body, label_n):
            score = 2.0
        else:
            # 3) Soft token-overlap fallback
            title_overlap = token_overlap_score(title, label_n)
            body_overlap = token_overlap_score(body, label_n)

            if title_overlap >= 0.6:
                score = 1.5 + title_overlap
            elif body_overlap >= 0.6:
                score = 1.0 + body_overlap
            else:
                score = 0.0

        if score > 0.0:
            evidence.append(label)
        if score > best:
            best = score

    return best, evidence


def find_resource_csv(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find a resource metadata CSV. Tried:\n  - "
        + "\n  - ".join(str(p) for p in candidates)
        + "\n\nFix: pass --resource-csv explicitly, or update candidate paths."
    )


def load_resources_title(resource_csv: Path) -> Dict[str, str]:
    with resource_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        if not cols:
            raise ValueError(f"{resource_csv} has no header row.")

        id_col = "ID" if "ID" in cols else ("id" if "id" in cols else cols[0])

        title_candidates = ["Title", "title", "Resource Title", "resource_title", "Name", "name"]
        title_col = next((c for c in title_candidates if c in cols), None)
        if title_col is None:
            title_col = cols[1] if len(cols) > 1 else id_col

        out: Dict[str, str] = {}
        for r in reader:
            rid = str(r.get(id_col, "")).strip()
            if not rid:
                continue
            out[rid] = str(r.get(title_col, "")).strip()
        return out


def load_resource_text(
    jsonl_candidates: List[Path],
    csv_candidates: List[Path],
) -> Dict[str, str]:
    for p in jsonl_candidates:
        if not p.exists():
            continue

        out: Dict[str, str] = {}
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSONL at {p}:{line_no}: {e}")

                rid = str(obj.get("id") or obj.get("ID") or "").strip()
                if not rid:
                    continue

                text = obj.get("extracted_text")
                if text is None:
                    text = obj.get("text")
                if text is None:
                    text = ""

                out[rid] = str(text)

        if out:
            return out

    for p in csv_candidates:
        if not p.exists():
            continue

        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if "ID" not in cols or "text" not in cols:
                raise ValueError(f"{p} must contain columns: ID, text")

            out: Dict[str, str] = {}
            for r in reader:
                rid = str(r["ID"]).strip()
                out[rid] = str(r["text"] or "")

            if out:
                return out

    return {}


def load_taxonomy_clean(taxonomy_csv: Path) -> Dict[str, Set[str]]:
    with taxonomy_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "ID" not in cols:
            raise ValueError("taxonomy_clean.csv must include an ID column named 'ID'.")

        taxonomy_cols = [
            "Individual Characteristics",
            "Career Pathway",
            "Research Funding Process",
            "Organisational Culture",
        ]
        missing = [c for c in taxonomy_cols if c not in cols]
        if missing:
            raise ValueError(f"taxonomy_clean.csv missing required taxonomy columns: {missing}")

        out: Dict[str, Set[str]] = defaultdict(set)
        for r in reader:
            rid = str(r["ID"]).strip()
            if not rid:
                continue
            for col in taxonomy_cols:
                for t in split_tags(r.get(col, "")):
                    out[rid].add(t)
        return out


def load_queries_json(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("queries.json must be a list of {qid, query} objects.")
    for q in data:
        if "qid" not in q or "query" not in q:
            raise ValueError("Each query must have keys: qid, query")
    return data


def load_queries_jsonl(path: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse queries JSONL at {path}:{line_no}: {e}")
            if "qid" not in obj or "query" not in obj:
                raise ValueError(f"Each JSONL line must contain keys qid, query ({path}:{line_no}).")
            out.append({"qid": str(obj["qid"]), "query": str(obj["query"])})
    return out


def available_ontology_names() -> List[str]:
    if not DEFAULT_ONTOLOGY_VERSIONS_DIR.exists():
        return []
    names: List[str] = []
    for p in sorted(DEFAULT_ONTOLOGY_VERSIONS_DIR.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            names.append(p.name)
    return names


def resolve_ontology_path(ontology_json: Path | None, ontology_name: str | None) -> Path:
    if ontology_json is not None:
        if ontology_json.exists():
            return ontology_json

    if ontology_name:
        if ontology_name == "v1":
            if DEFAULT_ONTOLOGY_JSON.exists():
                return DEFAULT_ONTOLOGY_JSON
        else:
            candidate = DEFAULT_ONTOLOGY_VERSIONS_DIR / ontology_name / f"edi_ontology_{ontology_name}.json"
            if candidate.exists():
                return candidate

    if ontology_json is not None:
        base = ontology_json.name
        if DEFAULT_ONTOLOGY_VERSIONS_DIR.exists():
            matches = list(DEFAULT_ONTOLOGY_VERSIONS_DIR.rglob(base))
            matches = [m for m in matches if m.is_file()]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Ontology file not found at: {ontology_json}\n"
                    f"But multiple matches were found under {DEFAULT_ONTOLOGY_VERSIONS_DIR}:\n  - "
                    + "\n  - ".join(str(m) for m in matches)
                    + "\n\nFix: pass --ontology-json with the exact path you want, or use --ontology-name."
                )

    avail = available_ontology_names()
    hint = ""
    if avail:
        hint = "\nAvailable ontology names under ontology/versions: " + ", ".join(avail)
        hint += "\nExample: --ontology-name v2_iter01"

    if ontology_name:
        raise FileNotFoundError(f"Could not resolve ontology for name: {ontology_name}." + hint)

    if ontology_json is None:
        raise FileNotFoundError(f"Missing ontology file: {DEFAULT_ONTOLOGY_JSON}." + hint)

    raise FileNotFoundError(f"Missing ontology file: {ontology_json}." + hint)


# ----------------------------
# Baselines
# ----------------------------
def run_keyword_bm25(
    resources_text: Dict[str, str],
    queries: List[Dict[str, str]],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    docs = [(rid, text) for rid, text in resources_text.items()]
    index = BM25Index.build(docs)

    results: List[Dict[str, Any]] = []
    for q in queries:
        scored = index.score(q["query"])
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [{"rid": rid, "score": float(score)} for rid, score in scored[:top_k]]
        results.append({"qid": q["qid"], "query": q["query"], "ranking": ranked})
    return results


def match_query_to_concepts(query: str, label_to_concept: Dict[str, str]) -> Set[str]:
    qn = norm(query)
    matched: Set[str] = set()

    labels_sorted = sorted(label_to_concept.keys(), key=len, reverse=True)

    for lab in labels_sorted:
        if not lab:
            continue
        pattern = r"(^|\s)" + re.escape(lab) + r"(\s|$)"
        if re.search(pattern, qn):
            matched.add(label_to_concept[lab])

    return matched


def run_ontology_only(
    queries: List[Dict[str, str]],
    taxonomy_tags: Dict[str, Set[str]],
    label_to_concept: Dict[str, str],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for q in queries:
        matched_concepts = match_query_to_concepts(q["query"], label_to_concept)

        scores: List[Tuple[str, float]] = []
        for rid in taxonomy_tags.keys():
            rid_concepts: Set[str] = set()
            for t in taxonomy_tags[rid]:
                cid = label_to_concept.get(norm(t))
                if cid:
                    rid_concepts.add(cid)

            overlap = matched_concepts.intersection(rid_concepts)
            if overlap:
                scores.append((rid, float(len(overlap))))

        scores.sort(key=lambda x: x[1], reverse=True)
        ranked = [{"rid": rid, "score": score} for rid, score in scores[:top_k]]

        results.append(
            {
                "qid": q["qid"],
                "query": q["query"],
                "matched_concepts": sorted(list(matched_concepts)),
                "ranking": ranked,
            }
        )
    return results


def run_ontology_text(
    queries: List[Dict[str, str]],
    resources_text: Dict[str, str],
    label_to_concept: Dict[str, str],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Improved ontology-text baseline:
    - exact phrase matches
    - token-overlap fallback
    - title bonus
    - concept-level scoring
    """
    concept_to_labels = build_concept_to_labels(label_to_concept)
    results: List[Dict[str, Any]] = []

    for q in queries:
        matched_concepts = match_query_to_concepts(q["query"], label_to_concept)
        scores: List[Tuple[str, float]] = []

        for rid, text in resources_text.items():
            title, body = split_title_body(text)

            concept_scores: List[float] = []
            evidence_by_concept: Dict[str, List[str]] = {}

            for cid in matched_concepts:
                labels = concept_to_labels.get(cid, set())
                best_score, evidence = concept_match_strength(title, body, labels)

                if best_score > 0.0:
                    concept_scores.append(best_score)
                    evidence_by_concept[cid] = evidence

            if concept_scores:
                matched_concept_count = len(concept_scores)
                total_strength = sum(concept_scores)

                # Main ranking logic:
                # prioritize number of matched concepts, then match strength
                score = (2.5 * matched_concept_count) + total_strength
                scores.append((rid, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        ranked = [{"rid": rid, "score": score} for rid, score in scores[:top_k]]

        results.append(
            {
                "qid": q["qid"],
                "query": q["query"],
                "matched_concepts": sorted(list(matched_concepts)),
                "ranking": ranked,
            }
        )

    return results


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.retrieval.run_baselines",
        description=(
            "Run baseline retrieval methods (Keyword BM25, Ontology-tag, Ontology-text) "
            "and write rankings to JSONL."
        ),
    )
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--ontology-json",
        type=Path,
        default=None,
        help=(
            "Ontology JSON path. If omitted, uses --ontology-name (default: v1). "
            "If you pass a path that doesn't exist, we will try to find it under ontology/versions/."
        ),
    )
    g.add_argument(
        "--ontology-name",
        type=str,
        default=DEFAULT_ONTOLOGY_NAME,
        help=(
            "Ontology version name. 'v1' uses ontology/edi_ontology_v1.json. "
            "Other names resolve to ontology/versions/<name>/edi_ontology_<name>.json "
            "(e.g., v2_iter01)."
        ),
    )

    p.add_argument(
        "--taxonomy-csv",
        type=Path,
        default=DEFAULT_TAXONOMY_CLEAN_CSV,
        help=f"taxonomy_clean.csv path (default: {DEFAULT_TAXONOMY_CLEAN_CSV})",
    )
    p.add_argument(
        "--resource-csv",
        type=Path,
        default=None,
        help="Resource metadata CSV path. If omitted, tries common candidates.",
    )
    p.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_JSON,
        help=f"Queries file (.json list OR .jsonl). Default: {DEFAULT_QUERIES_JSON}",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K results per query (default: 20)",
    )

    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir_explicit = argv is not None and "--out-dir" in argv

    ontology_path = resolve_ontology_path(args.ontology_json, getattr(args, "ontology_name", None))
    if not args.taxonomy_csv.exists():
        raise FileNotFoundError(f"Missing taxonomy file: {args.taxonomy_csv}")
    if not args.queries.exists():
        raise FileNotFoundError(f"Missing queries file: {args.queries}")

    resource_csv = args.resource_csv
    if resource_csv is None:
        resource_csv = find_resource_csv(DEFAULT_RESOURCE_CSV_CANDIDATES)
    if not resource_csv.exists():
        raise FileNotFoundError(f"Missing resource CSV: {resource_csv}")

    titles = load_resources_title(resource_csv)
    texts = load_resource_text(
        DEFAULT_RESOURCE_TEXT_JSONL_CANDIDATES,
        DEFAULT_RESOURCE_TEXT_CSV_CANDIDATES,
    )

    resources_text: Dict[str, str] = {}
    all_rids = set(titles.keys()) | set(texts.keys())
    for rid in all_rids:
        title = titles.get(rid, "")
        body = texts.get(rid, "")
        resources_text[rid] = (title + "\n\n" + body).strip()

    taxonomy_tags = load_taxonomy_clean(args.taxonomy_csv)

    if args.queries.suffix.lower() == ".jsonl":
        queries = load_queries_jsonl(args.queries)
    else:
        queries = load_queries_json(args.queries)

    label_to_concept = load_ontology_label_map(ontology_path)

    out_dir = args.out_dir
    if not out_dir_explicit:
        name = getattr(args, "ontology_name", None)
        if name and name != "v1":
            out_dir = out_dir / name
        elif ontology_path.name != DEFAULT_ONTOLOGY_JSON.name:
            out_dir = out_dir / ontology_path.stem

    out_dir.mkdir(parents=True, exist_ok=True)
    out_keyword = out_dir / "keyword_bm25.jsonl"
    out_ontology_tag = out_dir / "ontology_only.jsonl"
    out_ontology_text = out_dir / "ontology_text.jsonl"

    keyword_results = run_keyword_bm25(resources_text, queries, top_k=args.top_k)
    ontology_tag_results = run_ontology_only(queries, taxonomy_tags, label_to_concept, top_k=args.top_k)
    ontology_text_results = run_ontology_text(queries, resources_text, label_to_concept, top_k=args.top_k)

    write_jsonl(out_keyword, keyword_results)
    write_jsonl(out_ontology_tag, ontology_tag_results)
    write_jsonl(out_ontology_text, ontology_text_results)

    print("Baselines complete")
    print(f"Keyword BM25:        {out_keyword}")
    print(f"Ontology-tag:        {out_ontology_tag}")
    print(f"Ontology-text:       {out_ontology_text}")
    print(f"Resource CSV used:   {resource_csv}")
    print(f"Ontology JSON used:  {ontology_path}")
    if getattr(args, "ontology_name", None):
        print(f"Ontology name:       {args.ontology_name}")
    print(f"Taxonomy CSV used:   {args.taxonomy_csv}")
    if not texts:
        print("No extracted text found (JSONL/CSV); keyword baseline used title-only.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())