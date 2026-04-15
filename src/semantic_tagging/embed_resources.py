import time
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .io_utils import read_jsonl, write_json
from .text_utils import make_resource_text


def load_resources(resources_unified_jsonl_path) -> List[dict]:
    return list(read_jsonl(resources_unified_jsonl_path))


def embed_resources(
    resources: List[dict],
    model_id: str,
    batch_size: int = 16,
    max_chars: int = 8000,
) -> Tuple[np.ndarray, List[str], Dict]:
    
    """ Embeds each resource using (title + extracted_text)."""
    
    t0 = time.time()
    model = SentenceTransformer(model_id)

    texts: List[str] = []
    ids: List[str] = []

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

        texts.append(make_resource_text(title, txt, max_chars=max_chars))
        ids.append(rid)

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine sim == dot product
    )
    emb = np.asarray(emb, dtype=np.float32)

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
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz_path, embeddings=embeddings, resource_ids=np.array(resource_ids))
    write_json(out_meta_json_path, meta)