from collections import defaultdict
import json
from pathlib import Path

from .paths import SEMANTIC_DIR


def main():
    tags_path = SEMANTIC_DIR / "semantic_tags_minilm.jsonl"

    concept_freq = defaultdict(int)

    with tags_path.open() as f:
        for line in f:
            row = json.loads(line)
            for item in row["top_k"]:
                concept_freq[item["concept_id"]] += 1

    sorted_freq = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)

    with (SEMANTIC_DIR / "concept_frequency.json").open("w") as f:
        json.dump(sorted_freq, f, indent=2)

    print("Concept frequency saved.")


if __name__ == "__main__":
    main()