import json
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

# -----------------------------
# Config (edit if your paths differ)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

V1_PATH = PROJECT_ROOT / "ontology" / "edi_ontology_v1.json"
V2_PATH = PROJECT_ROOT / "ontology" / "edi_ontology_v2.json"

OUT_DIR = PROJECT_ROOT / "data"
OUT_STATS = OUT_DIR / "v2_iteration1_stats.json"

OUT_ADDED_CSV = OUT_DIR / "v2_iteration1_terms_added.csv"
OUT_FULL_V1_CSV = OUT_DIR / "v2_iteration1_terms_full_v1.csv"
OUT_FULL_V2_CSV = OUT_DIR / "v2_iteration1_terms_full_v2.csv"

# Column order exactly as you requested:
COL_ORDER = [
    "Organisational Culture",
    "Career Pathway",
    "Research Funding Process",
    "Individual Characteristics",
]

# category_id -> pretty label (matches ontology JSON categories)
CATEGORY_ID_TO_LABEL = {
    "organisational_culture": "Organisational Culture",
    "career_pathway": "Career Pathway",
    "research_funding_process": "Research Funding Process",
    "individual_characteristics": "Individual Characteristics",
}


# -----------------------------
# Helpers
# -----------------------------
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def collect_terms_by_category(ont: dict) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    """
    Returns:
      terms_by_label: "Career Pathway" -> set of unique terms (pref + alt labels)
      concept_count_by_label: "Career Pathway" -> number of concepts in that category
    """
    terms_by_label: Dict[str, Set[str]] = {lab: set() for lab in CATEGORY_ID_TO_LABEL.values()}
    concept_count_by_label: Dict[str, int] = {lab: 0 for lab in CATEGORY_ID_TO_LABEL.values()}

    for c in ont.get("concepts", []):
        cat_id = c.get("category_id") or c.get("category")  # v2 showed one concept used "category"
        cat_label = CATEGORY_ID_TO_LABEL.get(cat_id)
        if not cat_label:
            continue

        concept_count_by_label[cat_label] += 1

        pref = c.get("pref_label")
        if pref and norm(pref):
            terms_by_label[cat_label].add(pref.strip())

        for alt in (c.get("alt_labels") or []):
            if alt and norm(str(alt)):
                terms_by_label[cat_label].add(str(alt).strip())

    return terms_by_label, concept_count_by_label


def write_terms_csv(path: Path, terms_by_category: Dict[str, List[str]]) -> None:
    """
    Writes a 4-column CSV. Each column is a category. Rows contain terms.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # ensure all 4 columns exist
    for col in COL_ORDER:
        terms_by_category.setdefault(col, [])

    max_len = max(len(terms_by_category[c]) for c in COL_ORDER) if COL_ORDER else 0

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COL_ORDER)

        for i in range(max_len):
            row = []
            for col in COL_ORDER:
                row.append(terms_by_category[col][i] if i < len(terms_by_category[col]) else "")
            w.writerow(row)


# -----------------------------
# Main
# -----------------------------
def main():
    v1 = read_json(V1_PATH)
    v2 = read_json(V2_PATH)

    v1_terms, v1_concepts = collect_terms_by_category(v1)
    v2_terms, v2_concepts = collect_terms_by_category(v2)

    # Compute "added terms" = terms present in v2 but not v1, per category
    added_terms: Dict[str, List[str]] = {}
    full_v1_terms: Dict[str, List[str]] = {}
    full_v2_terms: Dict[str, List[str]] = {}

    for cat in CATEGORY_ID_TO_LABEL.values():
        v1_set = v1_terms.get(cat, set())
        v2_set = v2_terms.get(cat, set())

        added = sorted(v2_set - v1_set, key=lambda x: x.lower())
        full_v1 = sorted(v1_set, key=lambda x: x.lower())
        full_v2 = sorted(v2_set, key=lambda x: x.lower())

        added_terms[cat] = added
        full_v1_terms[cat] = full_v1
        full_v2_terms[cat] = full_v2

    # Stats object similar in spirit to taxonomy_stats.json
    stats = {
        "inputs": {
            "v1": str(V1_PATH.relative_to(PROJECT_ROOT)),
            "v2": str(V2_PATH.relative_to(PROJECT_ROOT)),
        },
        "counts": {
            "concept_count_per_category_v1": v1_concepts,
            "concept_count_per_category_v2": v2_concepts,
            "unique_terms_per_category_v1": {k: len(v) for k, v in v1_terms.items()},
            "unique_terms_per_category_v2": {k: len(v) for k, v in v2_terms.items()},
            "added_unique_terms_per_category_v2_minus_v1": {k: len(v) for k, v in added_terms.items()},
        },
        "notes": {
            "concept_count": "Number of concept nodes (one per ontology concept) in that top-level category.",
            "unique_terms": "Unique labels per category = pref_label + all alt_labels (deduplicated). This reflects 'expansion'.",
            "added_terms": "Terms present in v2 but not v1 (new alt_labels and/or new concept pref_labels).",
            "csv_outputs": {
                "added_terms_csv": str(OUT_ADDED_CSV.relative_to(PROJECT_ROOT)),
                "full_v1_terms_csv": str(OUT_FULL_V1_CSV.relative_to(PROJECT_ROOT)),
                "full_v2_terms_csv": str(OUT_FULL_V2_CSV.relative_to(PROJECT_ROOT)),
            },
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write CSVs
    write_terms_csv(OUT_ADDED_CSV, added_terms)
    write_terms_csv(OUT_FULL_V1_CSV, full_v1_terms)
    write_terms_csv(OUT_FULL_V2_CSV, full_v2_terms)

    print("Wrote:", OUT_STATS)
    print("Wrote:", OUT_ADDED_CSV)
    print("Wrote:", OUT_FULL_V1_CSV)
    print("Wrote:", OUT_FULL_V2_CSV)


if __name__ == "__main__":
    main()