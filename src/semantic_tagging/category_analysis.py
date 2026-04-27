from collections import defaultdict
import json

from .paths import ONTOLOGY_V1_JSON, RESOURCES_UNIFIED_JSONL, SEMANTIC_DIR
from .io_utils import read_json, read_jsonl


def main():
    """
    Analyse how often predicted semantic tags fall into each ontology category.

    Output:
    data/semantic/category_frequency.json
    """
    # Load ontology structure and resource corpus
    ontology = read_json(ONTOLOGY_V1_JSON)
    resources = list(read_jsonl(RESOURCES_UNIFIED_JSONL))

    # File containing predicted semantic tags for each resource
    tags_path = SEMANTIC_DIR / "semantic_tags_minilm.jsonl"

    # Read tag predictions into memory: resource_id -> predicted tags
    tags = {}
    with tags_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tags[str(row["rid"])] = row["top_k"]

    # Map ontology concept IDs to their parent category
    concept_to_category = {
        c["id"]: c["category_id"]
        for c in ontology["concepts"]
    }

    # Counter for how many times each category appears
    category_counts = defaultdict(int)

    # Count predicted tags by category across all resources
    for rid, preds in tags.items():
        for p in preds:
            cid = p["concept_id"]
            cat = concept_to_category.get(cid)
            if cat:
                category_counts[cat] += 1

    # Convert defaultdict to normal dictionary for JSON export
    out = dict(category_counts)

    # Save category frequency analysis
    with (SEMANTIC_DIR / "category_frequency.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Category frequency saved.")


if __name__ == "__main__":
    main()