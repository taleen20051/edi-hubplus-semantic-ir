import time
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .io_utils import read_json, write_json
from .text_utils import normalise_text


def build_concept_texts(ontology: dict) -> Tuple[List[str], List[dict]]:
    """
    We represent each concept by multiple surface forms:
    - pref_label
    - each alt_label
    Then we embed all surface forms and keep a mapping back to concept_id.
    """
    surface_texts: List[str] = []
    surface_meta: List[dict] = []

    for c in ontology.get("concepts", []):
        cid = c["id"]
        cat = c.get("category_id")
        pref = normalise_text(c.get("pref_label", ""))

        if pref:
            surface_texts.append(pref)
            surface_meta.append({"concept_id": cid, "category_id": cat, "label_type": "pref", "label": pref})

        for alt in c.get("alt_labels", []) or []:
            alt = normalise_text(alt)
            if alt and alt.lower() != pref.lower():
                surface_texts.append(alt)
                surface_meta.append({"concept_id": cid, "category_id": cat, "label_type": "alt", "label": alt})

    return surface_texts, surface_meta


def build_and_save_concept_index(
    ontology_json_path,
    model_name: str,
    model_id: str,
    out_index_json_path,
    out_embeddings_npz_path,
    batch_size: int = 64,
) -> Dict:
    t0 = time.time()
    ontology = read_json(ontology_json_path)

    surface_texts, surface_meta = build_concept_texts(ontology)

    model = SentenceTransformer(model_id)
    # normalize_embeddings=True makes cosine similarity = dot product
    emb = model.encode(surface_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    # Build fast lookup: concept_id -> list of surface indices
    concept_to_surface_idxs: Dict[str, List[int]] = {}
    for i, m in enumerate(surface_meta):
        concept_to_surface_idxs.setdefault(m["concept_id"], []).append(i)

    index_obj = {
        "model_name": model_name,
        "model_id": model_id,
        "created_unix": int(time.time()),
        "counts": {
            "concepts": len(ontology.get("concepts", [])),
            "surface_forms": len(surface_texts),
        },
        "surface_meta": surface_meta,
        "concept_to_surface_idxs": concept_to_surface_idxs,
    }

    out_index_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_embeddings_npz_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(out_index_json_path, index_obj)
    np.savez_compressed(out_embeddings_npz_path, embeddings=emb)

    dt = time.time() - t0
    index_obj["runtime_seconds"] = dt
    write_json(out_index_json_path, index_obj)  # rewrite with runtime
    return index_obj