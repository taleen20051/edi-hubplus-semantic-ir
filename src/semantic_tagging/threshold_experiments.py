import time
import numpy as np
from pathlib import Path

from .paths import (
    ONTOLOGY_V1_JSON,
    RESOURCES_UNIFIED_JSONL,
    SEMANTIC_DIR,
)
from .models import MODEL_SPECS
from .io_utils import read_json, read_jsonl, write_json
from .embed_resources import embed_resources
from .build_concept_index import build_and_save_concept_index
from .semantic_tagging import run_semantic_tagging
from .evaluate_semantic_tags import evaluate_semantic_tagging


def load_npz_embeddings(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    return z["embeddings"]


THRESHOLD_CONFIGS = [
    {"top_k": 5, "min_score": 0.25},
    {"top_k": 5, "min_score": 0.30},
    {"top_k": 8, "min_score": 0.30},
    {"top_k": 8, "min_score": 0.40},
    {"top_k": 10, "min_score": 0.30},
]


def main():
    ontology = read_json(ONTOLOGY_V1_JSON)
    resources = list(read_jsonl(RESOURCES_UNIFIED_JSONL))

    summary = []

    # We'll test only MiniLM for threshold experiments
    model_name = "minilm"
    model_id = MODEL_SPECS[model_name]

    print(f"Running threshold experiments using {model_name}")

    # Load concept embeddings
    concept_emb_npz = SEMANTIC_DIR / f"concept_surface_embeddings_{model_name}.npz"
    concept_index_json = SEMANTIC_DIR / f"concept_index_{model_name}.json"

    concept_index = read_json(concept_index_json)
    surface_meta = concept_index["surface_meta"]
    surface_embeddings = load_npz_embeddings(concept_emb_npz)

    # Load resource embeddings
    res_emb_npz = SEMANTIC_DIR / f"resource_embeddings_{model_name}.npz"
    z = np.load(res_emb_npz, allow_pickle=True)
    resource_embeddings = z["embeddings"]
    resource_ids = z["resource_ids"]

    for cfg in THRESHOLD_CONFIGS:
        print(f"\nTesting top_k={cfg['top_k']} min_score={cfg['min_score']}")

        tags = run_semantic_tagging(
            resource_embeddings=resource_embeddings,
            resource_ids=resource_ids,
            surface_embeddings=surface_embeddings,
            surface_meta=surface_meta,
            top_k=cfg["top_k"],
            min_score=cfg["min_score"],
        )

        report = evaluate_semantic_tagging(
            resources,
            ontology,
            tags,
            SEMANTIC_DIR / f"threshold_eval_{cfg['top_k']}_{cfg['min_score']}.json"
        )

        row = {
            "top_k": cfg["top_k"],
            "min_score": cfg["min_score"],
            "macro_precision": report["macro_avg"]["precision"],
            "macro_recall": report["macro_avg"]["recall"],
            "macro_f1": report["macro_avg"]["f1"],
        }

        summary.append(row)

    write_json(SEMANTIC_DIR / "threshold_summary.json", summary)
    print("\nThreshold experiments complete.")


if __name__ == "__main__":
    main()