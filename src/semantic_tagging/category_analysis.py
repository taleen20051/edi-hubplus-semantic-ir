from collections import defaultdict
from pathlib import Path
import json

from .paths import ONTOLOGY_V1_JSON, RESOURCES_UNIFIED_JSONL, SEMANTIC_DIR
from .io_utils import read_json, read_jsonl


def main():
    ontology = read_json(ONTOLOGY_V1_JSON)
    resources = list(read_jsonl(RESOURCES_UNIFIED_JSONL))

    tags_path = SEMANTIC_DIR / "semantic_tags_minilm.jsonl"

    tags = {}
    with tags_path.open() as f:
        for line in f:
            row = json.loads(line)
            tags[str(row["rid"])] = row["top_k"]

    concept_to_category = {
        c["id"]: c["category_id"]
        for c in ontology["concepts"]
    }

    category_counts = defaultdict(int)

    for rid, preds in tags.items():
        for p in preds:
            cid = p["concept_id"]
            cat = concept_to_category.get(cid)
            if cat:
                category_counts[cat] += 1

    out = dict(category_counts)

    with (SEMANTIC_DIR / "category_frequency.json").open("w") as f:
        json.dump(out, f, indent=2)

    print("Category frequency saved.")


if __name__ == "__main__":
    main()