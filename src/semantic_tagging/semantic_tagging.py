from typing import Dict, List, Tuple
import numpy as np


# Helper to reduce many surface-label scores to one concept score.
def max_sim_per_concept(
    sim_matrix: np.ndarray,
    surface_meta: List[dict],
) -> Dict[str, float]:
    """
    For one resource, each ontology concept may have multiple labels
    (preferred label + alternative labels).

    We keep the highest similarity score found for each concept.
    """
    best: Dict[str, float] = {}

    for i, meta in enumerate(surface_meta):
        cid = meta["concept_id"]
        score = float(sim_matrix[i])

        # Keep the strongest matching surface form for this concept.
        if cid not in best or score > best[cid]:
            best[cid] = score

    return best


# Get top semantic tags for one resource.
def top_k_concepts_for_resource(
    resource_vec: np.ndarray,
    surface_embeddings: np.ndarray,
    surface_meta: List[dict],
    k: int,
    min_score: float = 0.30,
) -> List[Dict]:
    """
    Compare one resource embedding against all concept-label embeddings.
    Because vectors were normalised earlier, dot product equals cosine similarity.
    """
    # Similarity between this resource and every concept surface label.
    sims = surface_embeddings @ resource_vec

    # Collapse multiple labels into one score per concept.
    best = max_sim_per_concept(sims, surface_meta)

    # Rank concepts from strongest to weakest match.
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)

    out = []
    for cid, score in ranked[:k]:
        # Ignore weak matches below threshold.
        if score < min_score:
            continue

        out.append({"concept_id": cid, "score": float(score)})

    return out


# Run semantic tagging across all resources.
def run_semantic_tagging(
    resource_embeddings: np.ndarray,
    resource_ids: List[str],
    surface_embeddings: np.ndarray,
    surface_meta: List[dict],
    top_k: int = 8,
    min_score: float = 0.30,
) -> List[Dict]:
    """
    Assign top ontology concepts to every resource in the corpus.
    """
    results: List[Dict] = []

    for i, rid in enumerate(resource_ids):
        vec = resource_embeddings[i]

        top = top_k_concepts_for_resource(
            resource_vec=vec,
            surface_embeddings=surface_embeddings,
            surface_meta=surface_meta,
            k=top_k,
            min_score=min_score,
        )

        results.append({"rid": rid, "top_k": top})

    return results