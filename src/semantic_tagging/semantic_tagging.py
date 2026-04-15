from typing import Dict, List, Tuple

import numpy as np


def max_sim_per_concept(
    sim_matrix: np.ndarray,
    surface_meta: List[dict],
) -> Dict[str, float]:
    """
    sim_matrix: shape [num_concepts_surface_forms] for ONE resource
    We collapse multiple surface forms into one concept score using MAX.
    """
    best: Dict[str, float] = {}
    for i, m in enumerate(surface_meta):
        cid = m["concept_id"]
        s = float(sim_matrix[i])
        if cid not in best or s > best[cid]:
            best[cid] = s
    return best


def top_k_concepts_for_resource(
    resource_vec: np.ndarray,
    surface_embeddings: np.ndarray,
    surface_meta: List[dict],
    k: int,
    min_score: float = 0.30,
) -> List[Dict]:
    """
    Because we normalized embeddings, cosine similarity = dot product.
    """
    sims = surface_embeddings @ resource_vec  # shape [num_surface]
    best = max_sim_per_concept(sims, surface_meta)

    # Sort by score
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    out = []
    for cid, score in ranked[:k]:
        if score < min_score:
            continue
        out.append({"concept_id": cid, "score": float(score)})
    return out


def run_semantic_tagging(
    resource_embeddings: np.ndarray,
    resource_ids: List[str],
    surface_embeddings: np.ndarray,
    surface_meta: List[dict],
    top_k: int = 8,
    min_score: float = 0.30,
) -> List[Dict]:
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