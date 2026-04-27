import numpy as np

from .paths import (
    ONTOLOGY_V1_JSON,
    RESOURCES_UNIFIED_JSONL,
    SEMANTIC_DIR,
)
from .models import MODEL_SPECS
from .io_utils import read_json, read_jsonl, write_json
from .semantic_tagging import run_semantic_tagging
from .evaluate_semantic_tags import evaluate_semantic_tagging


# Load an embedding matrix from a compressed NPZ file.
def load_npz_embeddings(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    return z["embeddings"]


# Threshold settings tested during semantic tagging evaluation.
THRESHOLD_CONFIGS = [
    {"top_k": 5, "min_score": 0.25},
    {"top_k": 5, "min_score": 0.30},
    {"top_k": 8, "min_score": 0.30},
    {"top_k": 8, "min_score": 0.40},
    {"top_k": 10, "min_score": 0.30},
]


# Evaluate semantic tagging under different top_k and min_score settings.
def main():
    
    # Load fixed project inputs.
    ontology = read_json(ONTOLOGY_V1_JSON)
    resources = list(read_jsonl(RESOURCES_UNIFIED_JSONL))

    summary = []

    # Threshold experiments use MiniLM only to keep comparison focused and efficient.
    model_name = "minilm"
    model_id = MODEL_SPECS[model_name]

    print(f"Running threshold experiments using {model_name}")

    # Load precomputed concept-surface embeddings and their metadata.
    concept_emb_npz = SEMANTIC_DIR / f"concept_surface_embeddings_{model_name}.npz"
    concept_index_json = SEMANTIC_DIR / f"concept_index_{model_name}.json"

    concept_index = read_json(concept_index_json)
    surface_meta = concept_index["surface_meta"]
    surface_embeddings = load_npz_embeddings(concept_emb_npz)

    # Load precomputed resource embeddings for the same model.
    res_emb_npz = SEMANTIC_DIR / f"resource_embeddings_{model_name}.npz"
    z = np.load(res_emb_npz, allow_pickle=True)
    resource_embeddings = z["embeddings"]
    resource_ids = z["resource_ids"]

    # Run semantic tagging and evaluation for each threshold configuration.
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

        # Evaluate predicted tags against manual taxonomy labels.
        report = evaluate_semantic_tagging(
            resources,
            ontology,
            tags,
            SEMANTIC_DIR / f"threshold_eval_{cfg['top_k']}_{cfg['min_score']}.json",
        )

        # Keep only the summary metrics needed for threshold comparison.
        row = {
            "top_k": cfg["top_k"],
            "min_score": cfg["min_score"],
            "macro_precision": report["macro_avg"]["precision"],
            "macro_recall": report["macro_avg"]["recall"],
            "macro_f1": report["macro_avg"]["f1"],
        }

        summary.append(row)

    # Save one compact summary file comparing all tested configurations.
    write_json(SEMANTIC_DIR / "threshold_summary.json", summary)
    print("\nThreshold experiments complete.")


if __name__ == "__main__":
    main()