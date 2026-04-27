import time
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .io_utils import read_json, write_json
from .text_utils import normalise_text


def build_concept_texts(ontology: dict) -> Tuple[List[str], List[dict]]:
    """
    Convert ontology concepts into text labels ready for embedding.

    Each concept may contribute:
    - pref_label  : main preferred label
    - alt_labels  : synonyms / alternative labels

    Returns:
        surface_texts : list of text strings to embed
        surface_meta  : aligned metadata for each text string
    """
    surface_texts: List[str] = []
    surface_meta: List[dict] = []

    # Read every concept in the ontology
    for c in ontology.get("concepts", []):
        cid = c["id"]
        cat = c.get("category_id")

        # Preferred concept label
        pref = normalise_text(c.get("pref_label", ""))

        # Add preferred label if present
        if pref:
            surface_texts.append(pref)
            surface_meta.append(
                {
                    "concept_id": cid,
                    "category_id": cat,
                    "label_type": "pref",
                    "label": pref,
                }
            )

        # Add alternative labels if they are not duplicates
        for alt in c.get("alt_labels", []) or []:
            alt = normalise_text(alt)
            if alt and alt.lower() != pref.lower():
                surface_texts.append(alt)
                surface_meta.append(
                    {
                        "concept_id": cid,
                        "category_id": cat,
                        "label_type": "alt",
                        "label": alt,
                    }
                )

    return surface_texts, surface_meta



def build_and_save_concept_index(
    ontology_json_path,
    model_name: str,
    model_id: str,
    out_index_json_path,
    out_embeddings_npz_path,
    batch_size: int = 64,
) -> Dict:
    """
    Build semantic embeddings for ontology labels and save index files.

    Outputs:
    - JSON metadata index
    - NPZ embedding matrix
    """
    # Start timer for runtime reporting
    t0 = time.time()

    # Load ontology JSON file
    ontology = read_json(ontology_json_path)

    # Convert concepts into embed-ready text labels
    surface_texts, surface_meta = build_concept_texts(ontology)

    # Load sentence-transformer model
    model = SentenceTransformer(model_id)

    # Encode labels into dense vectors
    # normalize_embeddings=True means cosine similarity = dot product
    emb = model.encode(
        surface_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    # Build lookup: concept_id -> indices of its labels in embedding matrix
    concept_to_surface_idxs: Dict[str, List[int]] = {}
    for i, m in enumerate(surface_meta):
        concept_to_surface_idxs.setdefault(m["concept_id"], []).append(i)

    # Metadata object describing the saved index
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

    # Create output folders if needed
    out_index_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_embeddings_npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata JSON and compressed embeddings file
    write_json(out_index_json_path, index_obj)
    np.savez_compressed(out_embeddings_npz_path, embeddings=emb)

    # Add runtime statistics and rewrite JSON file
    dt = time.time() - t0
    index_obj["runtime_seconds"] = dt
    write_json(out_index_json_path, index_obj)

    return index_obj