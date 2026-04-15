import time
import numpy as np

from .paths import (
    ONTOLOGY_V1_JSON,
    RESOURCES_UNIFIED_JSONL,
    SEMANTIC_DIR,
)
from .models import MODEL_SPECS
from .io_utils import read_json, read_jsonl, write_jsonl, read_json, write_json
from .build_concept_index import build_and_save_concept_index
from .embed_resources import load_resources, embed_resources, save_resource_embeddings
from .semantic_tagging import run_semantic_tagging
from .evaluate_semantic_tags import evaluate_semantic_tagging


def load_npz_embeddings(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    return z["embeddings"]


def main():
    ontology = read_json(ONTOLOGY_V1_JSON)
    resources = list(read_jsonl(RESOURCES_UNIFIED_JSONL))

    summary = {"runs": []}

    for short_name, model_id in MODEL_SPECS.items():
        print(f"\n=== Semantic tagging with model: {short_name} ({model_id}) ===")
        t0 = time.time()

        concept_index_json = SEMANTIC_DIR / f"concept_index_{short_name}.json"
        concept_emb_npz = SEMANTIC_DIR / f"concept_surface_embeddings_{short_name}.npz"

        # 1) Concept embeddings
        concept_index = build_and_save_concept_index(
            ontology_json_path=ONTOLOGY_V1_JSON,
            model_name=short_name,
            model_id=model_id,
            out_index_json_path=concept_index_json,
            out_embeddings_npz_path=concept_emb_npz,
            batch_size=64,
        )

        surface_meta = concept_index["surface_meta"]
        surface_embeddings = load_npz_embeddings(concept_emb_npz)

        # 2) Resource embeddings (cache)
        res_emb_npz = SEMANTIC_DIR / f"resource_embeddings_{short_name}.npz"
        res_meta_json = SEMANTIC_DIR / f"resource_embeddings_{short_name}_meta.json"

        emb, ids, meta = embed_resources(resources, model_id=model_id, batch_size=16, max_chars=8000)
        save_resource_embeddings(res_emb_npz, res_meta_json, emb, ids, meta)

        # 3) Similarity tagging
        tags = run_semantic_tagging(
            resource_embeddings=emb,
            resource_ids=ids,
            surface_embeddings=surface_embeddings,
            surface_meta=surface_meta,
            top_k=8,
            min_score=0.30,
        )

        out_tags = SEMANTIC_DIR / f"semantic_tags_{short_name}.jsonl"
        write_jsonl(out_tags, tags)

        # 4) Evaluate vs manual tags in resources_unified
        out_eval = SEMANTIC_DIR / f"semantic_tagging_eval_{short_name}.json"
        report = evaluate_semantic_tagging(resources, ontology, tags, out_eval)

        dt = time.time() - t0
        run_row = {
            "model": short_name,
            "model_id": model_id,
            "runtime_seconds_total": dt,
            "outputs": {
                "concept_index": str(concept_index_json),
                "concept_surface_embeddings": str(concept_emb_npz),
                "resource_embeddings": str(res_emb_npz),
                "semantic_tags": str(out_tags),
                "eval": str(out_eval),
            },
            "eval_macro_avg": report["macro_avg"],
        }
        summary["runs"].append(run_row)

        print(f"Saved tags: {out_tags}")
        print(f"Saved eval: {out_eval}")
        print(f"Done in {dt:.1f}s")

    # Overall summary
    summary_path = SEMANTIC_DIR / "semantic_tagging_summary.json"
    write_json(summary_path, summary)
    print(f"\nAll done. Summary: {summary_path}")


if __name__ == "__main__":
    main()