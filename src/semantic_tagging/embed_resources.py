import time
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .io_utils import read_jsonl, write_json
from .text_utils import make_resource_text


def load_resources(resources_unified_jsonl_path) -> List[dict]:
    """
    Load the unified resource corpus from JSONL.

    Each line in the input file represents one resource with metadata and
    extracted text. This function returns the resources as a list of dicts.
    """
    return list(read_jsonl(resources_unified_jsonl_path))


def embed_resources(
    resources: List[dict],
    model_id: str,
    batch_size: int = 16,
    max_chars: int = 8000,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Embed each resource using its title and extracted text.

    Returns:
    - embeddings matrix
    - aligned resource IDs
    - metadata describing the embedding run
    """
    # Start timer so the embedding runtime can be recorded in metadata.
    t0 = time.time()

    # Load the sentence-transformer model used to encode resource text.
    model = SentenceTransformer(model_id)

    texts: List[str] = []
    ids: List[str] = []

    # Prepare one text string per resource and keep IDs aligned with the vectors.
    for idx, r in enumerate(resources):
        rid_raw = r.get("ID")
        if rid_raw is None:
            raise KeyError(
                f"Missing 'ID' field in resources_unified.jsonl at row index {idx}. "
                f"Available keys: {list(r.keys())}"
            )

        rid = str(rid_raw).strip()
        if rid == "" or rid.lower() == "none":
            raise ValueError(f"Invalid resource ID at row index {idx}: {rid_raw}")

        title = r.get("title", "") or ""
        txt = r.get("extracted_text", "") or ""

        # Combine title and extracted text, truncating very long resources.
        texts.append(make_resource_text(title, txt, max_chars=max_chars))
        ids.append(rid)

    # Encode all resources into dense vectors.
    # normalize_embeddings=True allows cosine similarity to be computed as a dot product.
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    # Save metadata so the embedding file is reproducible and inspectable.
    meta = {
        "model_id": model_id,
        "resource_count": len(ids),
        "max_chars": max_chars,
        "runtime_seconds": time.time() - t0,
    }
    return emb, ids, meta


def save_resource_embeddings(
    out_npz_path,
    out_meta_json_path,
    embeddings: np.ndarray,
    resource_ids: List[str],
    meta: Dict,
):
    """
    Save resource embeddings and metadata to disk.

    The NPZ file stores the vector matrix and aligned resource IDs.
    The JSON file stores run metadata such as model name and corpus size.
    """
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz_path, embeddings=embeddings, resource_ids=np.array(resource_ids))
    write_json(out_meta_json_path, meta)