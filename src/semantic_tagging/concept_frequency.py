from collections import defaultdict
import json

from .paths import SEMANTIC_DIR


def main():
    # Input file containing semantic tagging predictions.
    # Each line represents one resource and its top matched concepts.
    tags_path = SEMANTIC_DIR / "semantic_tags_minilm.jsonl"

    # Used to count how many times each concept appears.
    concept_freq = defaultdict(int)

    # Read the JSONL file line-by-line.
    with tags_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            # top_k contains the highest scoring ontology concepts predicted for this resource.
            for item in row["top_k"]:
                concept_id = item["concept_id"]
                concept_freq[concept_id] += 1

    # Sort concepts from most common to least common.
    sorted_freq = sorted(
        concept_freq.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Save ranked concept frequencies as JSON output.
    out_path = SEMANTIC_DIR / "concept_frequency.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_freq, f, indent=2)

    print("Concept frequency saved.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()