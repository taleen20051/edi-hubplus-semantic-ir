from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# Small data containers and helper functions for the hybrid retrieval pipeline.
@dataclass
class HybridHit:
    """One ranked retrieval result with score components kept for explanation."""
    rid: str
    score: float
    semantic_score: float
    concept_boost: float
    matched_concepts: List[Tuple[str, str]]
    matched_predicted_tags: List[Dict[str, Any]]



# Lightweight model cache to avoid reloading the embedding model for every query in a run.
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def _get_model(model_id: str) -> SentenceTransformer:
    """Load the embedding model once per process and reuse it."""
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = SentenceTransformer(model_id)
    return _MODEL_CACHE[model_id]


# JSON helpers
def _read_json(path: Path) -> dict:
    """Read a JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))



def _read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file, skipping blank lines."""
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def _norm(s: str) -> str:
    """Normalise text for simple lexical matching."""
    return " ".join((s or "").strip().lower().split())


# Loading ontology + matching concepts
def load_ontology(ontology_path: Path) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    Load ontology concepts and build a surface-label lookup.

    Returns:
      - concepts_by_id: concept_id -> concept object
      - label_to_id: normalised label -> concept_id (pref_label + alt_labels)
    """
    ont = _read_json(ontology_path)
    concepts_by_id: Dict[str, dict] = {}
    label_to_id: Dict[str, str] = {}

    for c in ont.get("concepts", []):
        cid = str(c["id"])
        concepts_by_id[cid] = c

        pref = _norm(c.get("pref_label", ""))
        if pref:
            label_to_id[pref] = cid

        for alt in (c.get("alt_labels") or []):
            alt_n = _norm(str(alt))
            if alt_n:
                label_to_id.setdefault(alt_n, cid)

    return concepts_by_id, label_to_id



def find_matching_concepts(query: str, concepts_by_id: Dict[str, dict]) -> List[Tuple[str, str]]:
    """
    Match query text against ontology preferred labels and alternative labels.

    Policy:
    - multi-word phrases must appear directly in the query
    - single-word matches are allowed only for a small allowlist of high-signal terms
    """
    q_norm = _norm(query)
    if not q_norm:
        return []

    q_tokens = set(q_norm.split())

    # Restrict single-token matching to avoid broad tags causing false positives.
    single_token_allowlist = {
        "ageism",
        "microaggressions",
        "neurodivergent",
        "transgender",
        "lgbtq+",
        "lgbtqia+",
    }

    matches: List[Tuple[str, str]] = []
    seen: set[str] = set()

    for cid, concept in concepts_by_id.items():
        surfaces: List[str] = []

        pref = str(concept.get("pref_label") or "").strip()
        if pref:
            surfaces.append(pref)

        alt_labels = concept.get("alt_labels") or []
        if isinstance(alt_labels, list):
            surfaces.extend(str(label).strip() for label in alt_labels if str(label).strip())

        for surface in surfaces:
            surface_norm = _norm(surface)
            if not surface_norm:
                continue

            # Multi-word labels are matched as full phrases.
            if " " in surface_norm:
                if surface_norm in q_norm:
                    if cid not in seen:
                        matches.append((cid, surface))
                        seen.add(cid)
                    break

            # Single-word labels are matched only when they are high-signal terms.
            elif surface_norm in single_token_allowlist and surface_norm in q_tokens:
                if cid not in seen:
                    matches.append((cid, surface))
                    seen.add(cid)
                break

    matches.sort(key=lambda x: x[1].lower())
    return matches



def build_expanded_query_text(
    query: str,
    matched_concepts: List[Tuple[str, str]],
    concepts_by_id: Dict[str, dict],
    include_alt_labels: bool = True,
) -> str:
    """
    Build the query text that will be embedded.

    The expanded query contains the original query plus matched ontology surfaces,
    and optionally the alternative labels belonging to matched concepts.
    """
    parts: List[str] = [query.strip()]

    for cid, matched_surface in matched_concepts:
        if matched_surface:
            parts.append(matched_surface)

        if include_alt_labels:
            concept = concepts_by_id.get(cid, {})
            for alt in (concept.get("alt_labels") or []):
                alt_text = str(alt).strip()
                if alt_text:
                    parts.append(alt_text)

    # De-duplicate expansion terms while preserving their original order.
    seen = set()
    deduped: List[str] = []
    for part in parts:
        key = _norm(part)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(part.strip())

    return " ".join(deduped).strip()


# Loading resource embeddings and predicted tags
def load_resource_embeddings_npz(npz_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load precomputed resource embeddings and their aligned resource IDs.

    Expects `embeddings` and `resource_ids` arrays in the NPZ file.
    """
    z = np.load(npz_path, allow_pickle=True)

    if "embeddings" not in z or "resource_ids" not in z:
        raise ValueError(
            f"Expected keys 'embeddings' and 'resource_ids' in {npz_path}. "
            f"Found: {list(z.keys())}"
        )

    embeddings = np.asarray(z["embeddings"], dtype=np.float32)
    resource_ids = [str(x).strip() for x in z["resource_ids"].tolist()]

    # Validate that the embedding matrix has one row per resource.
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings array must be 2D, got shape {embeddings.shape} from {npz_path}")

    if len(resource_ids) != embeddings.shape[0]:
        raise ValueError(
            "Mismatch between number of resource IDs and embedding rows: "
            f"{len(resource_ids)} IDs vs {embeddings.shape[0]} vectors"
        )

    return embeddings, resource_ids



def load_semantic_tags_jsonl(tags_path: Path) -> Dict[str, List[dict]]:
    """
    Load predicted semantic concept tags for each resource.

    Returns:
      rid -> [{"concept_id": "...", "score": float}, ...]
    """
    rows = _read_jsonl(tags_path)
    out: Dict[str, List[dict]] = {}
    for row in rows:
        rid = str(row.get("rid", "")).strip()
        if not rid:
            continue
        top_k = row.get("top_k", []) or []
        out[rid] = top_k if isinstance(top_k, list) else []
    return out


# Core hybrid search
def _ensure_unit(vec: np.ndarray) -> np.ndarray:
    """Normalise a vector so dot product behaves as cosine similarity."""
    denom = float(np.linalg.norm(vec) + 1e-12)
    return vec / denom



def hybrid_search(
    query: str,
    *,
    model_id: str,
    concepts_by_id: Dict[str, dict],
    resource_embeddings: np.ndarray,
    resource_ids: List[str],
    predicted_tags_by_rid: Dict[str, List[dict]],
    top_n: int = 15,
    alpha_semantic: float = 0.85,
    beta_concept_boost: float = 0.15,
    boost_per_overlap: float = 1.0,
    include_alt_labels_in_expansion: bool = True,
) -> Tuple[List[HybridHit], List[Tuple[str, str]], str]:
    """
    Run hybrid retrieval for one query.

    Returns:
      - hits: ranked HybridHit list
      - matched_concepts: (concept_id, matched_surface) list
      - expanded_query: query text actually embedded
    """
    query = query.strip()
    if not query:
        return [], [], ""

    if resource_embeddings.shape[0] != len(resource_ids):
        raise ValueError(
            "resource_embeddings and resource_ids must have the same number of rows/items. "
            f"Got {resource_embeddings.shape[0]} embeddings and {len(resource_ids)} IDs."
        )

    # 1) Match ontology concepts for explainability and optional query expansion.
    matched_concepts = find_matching_concepts(query, concepts_by_id)
    matched_cids = {cid for cid, _ in matched_concepts}

    # 2) Build the query text used by the embedding model.
    expanded_query = build_expanded_query_text(
        query=query,
        matched_concepts=matched_concepts,
        concepts_by_id=concepts_by_id,
        include_alt_labels=include_alt_labels_in_expansion,
    )

    # 3) Embed the expanded query in the same vector space as the resources.
    model = _get_model(model_id)
    q_vec = model.encode([expanded_query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec[0], dtype=np.float32)
    q_vec = _ensure_unit(q_vec)

    # 4) Compute semantic similarity against all resources.
    # Resource embeddings are normalised, so cosine similarity equals dot product.
    sims = resource_embeddings @ q_vec
    sims = np.asarray(sims, dtype=np.float32)

    # 5) Add ontology/tag overlap as a transparent concept-based boost.
    hits: List[HybridHit] = []
    for i, rid in enumerate(resource_ids):
        semantic_score = float(sims[i])

        pred_items = predicted_tags_by_rid.get(str(rid), []) or []
        overlap_items: List[Dict[str, Any]] = []
        overlap_count = 0

        # Count overlap between query-matched concepts and resource predicted tags.
        if matched_cids:
            for item in pred_items:
                cid = item.get("concept_id")
                if cid and cid in matched_cids:
                    overlap_items.append(item)
                    overlap_count += 1

        concept_boost = 0.0
        if overlap_count > 0:
            concept_boost = boost_per_overlap * float(overlap_count)

        # Weighted hybrid score used for final ranking.
        score = alpha_semantic * semantic_score + beta_concept_boost * concept_boost

        hits.append(
            HybridHit(
                rid=str(rid),
                score=float(score),
                semantic_score=float(semantic_score),
                concept_boost=float(concept_boost),
                matched_concepts=matched_concepts,
                matched_predicted_tags=overlap_items,
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_n], matched_concepts, expanded_query


# CLI: generate a hybrid run file (JSONL)
def _load_queries_jsonl(path: Path) -> List[dict]:
    """Read queries in JSONL format: {"qid": "q1", "query": "..."}."""
    return _read_jsonl(path)



def _write_run_jsonl(path: Path, rows: List[dict]) -> None:
    """Write retrieval run rows to JSONL for later evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def run_hybrid_over_queries(
    *,
    ontology_path: Path,
    embeddings_npz: Path,
    semantic_tags_jsonl: Path,
    queries_jsonl: Path,
    out_run_jsonl: Path,
    model_id: str,
    top_n: int,
    alpha: float,
    beta: float,
    include_alt_labels: bool,
) -> None:
    """Run hybrid retrieval over a fixed query set and save the ranked outputs."""
    # Load all reusable artefacts required by the hybrid retrieval run.
    concepts_by_id, _ = load_ontology(ontology_path)
    resource_embeddings, resource_ids = load_resource_embeddings_npz(embeddings_npz)
    predicted_tags_by_rid = load_semantic_tags_jsonl(semantic_tags_jsonl)
    queries = _load_queries_jsonl(queries_jsonl)

    out_rows: List[dict] = []

    for query_row in queries:
        qid = str(query_row.get("qid") or query_row.get("id") or "").strip()
        qtext = str(query_row.get("query") or "").strip()
        if not qid or not qtext:
            continue

        hits, matched_concepts, expanded_query = hybrid_search(
            qtext,
            model_id=model_id,
            concepts_by_id=concepts_by_id,
            resource_embeddings=resource_embeddings,
            resource_ids=resource_ids,
            predicted_tags_by_rid=predicted_tags_by_rid,
            top_n=top_n,
            alpha_semantic=alpha,
            beta_concept_boost=beta,
            include_alt_labels_in_expansion=include_alt_labels,
        )

        # Store enough detail to evaluate the ranking and inspect explanations.
        out_rows.append(
            {
                "qid": qid,
                "query": qtext,
                "expanded_query": expanded_query,
                "matched_concepts": [
                    {"concept_id": cid, "pref_label": surface} for cid, surface in matched_concepts
                ],
                "results": [
                    {
                        "rid": hit.rid,
                        "rank": rank + 1,
                        "score": hit.score,
                        "semantic_score": hit.semantic_score,
                        "concept_boost": hit.concept_boost,
                        "matched_predicted_tags": hit.matched_predicted_tags,
                    }
                    for rank, hit in enumerate(hits)
                ],
            }
        )

    _write_run_jsonl(out_run_jsonl, out_rows)
    print("Hybrid run written:", out_run_jsonl)



def _build_arg_parser():
    """Create the command-line interface for reproducible hybrid runs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a hybrid retrieval run (JSONL) for a set of queries."
    )
    parser.add_argument("--ontology", required=True, help="Path to ontology JSON (v1/v2 iteration)")
    parser.add_argument("--embeddings", required=True, help="Path to resource_embeddings_*.npz")
    parser.add_argument("--semantic_tags", required=True, help="Path to semantic_tags_*.jsonl")
    parser.add_argument("--queries", required=True, help="Path to queries JSONL")
    parser.add_argument("--out_run", required=True, help="Output run JSONL path")
    parser.add_argument(
        "--model_id",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model id used for query embedding",
    )
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--include_alt_labels", action="store_true")
    return parser



def main() -> None:
    """Parse CLI arguments and run the hybrid retrieval experiment."""
    args = _build_arg_parser().parse_args()

    run_hybrid_over_queries(
        ontology_path=Path(args.ontology),
        embeddings_npz=Path(args.embeddings),
        semantic_tags_jsonl=Path(args.semantic_tags),
        queries_jsonl=Path(args.queries),
        out_run_jsonl=Path(args.out_run),
        model_id=args.model_id,
        top_n=args.top_n,
        alpha=args.alpha,
        beta=args.beta,
        include_alt_labels=args.include_alt_labels,
    )


if __name__ == "__main__":
    main()