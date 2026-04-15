# src/ontology/build_ontology_v1.py
"""
Build Ontology v1 (spreadsheet-fidelity) from cleaned taxonomy outputs.

Inputs (Phase 1 outputs):
- data/processed/taxonomy_clean.csv
- data/processed/normalisation_log.csv

Output (Deliverable D2):
- data/processed/edi_ontology_v1.json

Design policy (v1):
- Each unique cleaned tag becomes a concept (no semantic merging)
- Parent category is one of the 4 high-level taxonomy dimensions
- alt_labels are ONLY variants observed in the spreadsheet cleaning log
- No external synonym expansion
- No inferred hierarchy beyond the 4 top-level categories
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple

import pandas as pd


IN_CLEAN = Path("data/processed/taxonomy_clean.csv")
IN_LOG = Path("data/processed/normalisation_log.csv")

OUT_ONTOLOGY = Path("ontology/edi_ontology_v1.json")
OUT_STATS = Path("data/processed/edi_ontology_v1_stats.json")

ID_COL = "ID"
TAXONOMY_COLS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]

CATEGORY_MAP = {
    "Individual Characteristics": ("individual_characteristics", "Individual Characteristics"),
    "Career Pathway": ("career_pathway", "Career Pathway"),
    "Research Funding Process": ("research_funding_process", "Research Funding Process"),
    "Organisational Culture": ("organisational_culture", "Organisational Culture"),
}

# change human language to machine-friendly slug language
# Example:"Visibility & Recognition" -> "visibility_recognition", "Work-Life Balance" -> "work_life_balance"
def slugify(label: str) -> str:
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "unnamed"


# Convert one taxonomy cell into a list of cleaned tags
# we split because the spreadsheet taxonomy is a drop down one resource can have multiple tags
def split_clean_cell(cell: Any) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    text = str(cell).strip()
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]

# (column_name, cleaned_label) -> set(alt_labels)
# We want the ontology to remember the observed variants so that later the query expansion can match both forms
def build_alt_labels_from_log(log_df: pd.DataFrame) -> Dict[Tuple[str, str], Set[str]]:
    alt: Dict[Tuple[str, str], Set[str]] = {}

    if log_df.empty:
        return alt

    required = {"column", "original", "cleaned"}
    missing = required - set(log_df.columns)
    if missing:
        raise ValueError(f"normalisation_log.csv missing columns: {sorted(list(missing))}")

    for _, r in log_df.iterrows():
        col = str(r["column"])
        original = str(r["original"]) if not pd.isna(r["original"]) else ""
        cleaned = str(r["cleaned"]) if not pd.isna(r["cleaned"]) else ""

        original = original.strip()
        cleaned = cleaned.strip()

        if not cleaned:
            continue
        if not original:
            continue

        # If original is cleaned, it's not a variant so we skip it
        if original == cleaned:
            continue

        key = (col, cleaned)
        alt.setdefault(key, set()).add(original)

    return alt


    """ Main pipeline for building Ontology v1.
    Steps:
    1) Load Phase 1 cleaned taxonomy (taxonomy_clean.csv)
    2) Load Phase 1 normalisation log (optional, for alt_labels)
    3) For each taxonomy column:
         - collect unique cleaned tags across all resources
         - create one ontology concept per unique tag
         - attach alt_labels (observed variants) where available
    4) Write ontology JSON + stats JSON for reporting
    """
def main() -> None:
    if not IN_CLEAN.exists():
        raise FileNotFoundError(f"Missing input: {IN_CLEAN} (run Phase 1 cleaning first)")
    if not IN_LOG.exists():
        # Still allow ontology build without alt_labels, but be explicit in stats/policy
        log_df = pd.DataFrame()
    else:
        log_df = pd.read_csv(IN_LOG, dtype=object)

    df = pd.read_csv(IN_CLEAN, dtype=object)

    # Validate columns
    missing = [c for c in [ID_COL, *TAXONOMY_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"taxonomy_clean.csv missing required columns: {missing}")

    alt_map = build_alt_labels_from_log(log_df)

    categories = [{"id": CATEGORY_MAP[c][0], "label": CATEGORY_MAP[c][1]} for c in TAXONOMY_COLS]

    concepts: List[Dict[str, Any]] = []
    seen_concept_ids: Set[str] = set()

    per_category_counts: Dict[str, int] = {}
    per_category_unique_labels: Dict[str, Set[str]] = {CATEGORY_MAP[c][0]: set() for c in TAXONOMY_COLS}

    for col in TAXONOMY_COLS:
        cat_id, _ = CATEGORY_MAP[col]

        unique_labels: Set[str] = set()
        for cell in df[col].tolist():
            for tag in split_clean_cell(cell):
                unique_labels.add(tag)

        for label in sorted(unique_labels):
            concept_id = f"{cat_id}__{slugify(label)}"

            # Ensure uniqueness even if slug collisions occur (rare, but safe)
            if concept_id in seen_concept_ids:
                concept_id = f"{concept_id}__{abs(hash(label)) % 10**8}"

            seen_concept_ids.add(concept_id)
            per_category_unique_labels[cat_id].add(label)

            alts = sorted(list(alt_map.get((col, label), set())))

            concepts.append(
                {
                    "id": concept_id,
                    "pref_label": label,
                    "alt_labels": alts,
                    "category_id": cat_id,
                    "source_column": col,
                }
            )

        per_category_counts[cat_id] = len(unique_labels)

    ontology = {
        "ontology_id": "edi-hubplus-taxonomy",
        "version": "v1",
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "description": (
            "Spreadsheet-fidelity ontology derived directly from the cleaned EDI Hub+ taxonomy columns. "
            "Each distinct cleaned tag value is represented as a concept under one of four top-level categories. "
            "Alternative labels are limited to variants observed during cleaning (normalisation log)."
        ),
        "provenance": {
            "source_files": {
                "taxonomy_clean_csv": str(IN_CLEAN),
                "normalisation_log_csv": str(IN_LOG),
            },
            "taxonomy_columns_included": TAXONOMY_COLS,
            "design_policy": {
                "spreadsheet_fidelity": True,
                "no_external_synonyms": True,
                "no_inferred_hierarchy": True,
                "alt_labels_from_observed_variants_only": True,
            },
        },
        "categories": categories,
        "concepts": concepts,
    }

    OUT_ONTOLOGY.parent.mkdir(parents=True, exist_ok=True)
    OUT_ONTOLOGY.write_text(json.dumps(ontology, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = {
        "created_utc": ontology["created_utc"],
        "n_concepts_total": len(concepts),
        "n_categories": len(categories),
        "concepts_per_category": per_category_counts,
        "has_normalisation_log": bool(IN_LOG.exists()),
        "n_concepts_with_alt_labels": sum(1 for c in concepts if c["alt_labels"]),
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Ontology v1 build complete")
    print(f"  Ontology: {OUT_ONTOLOGY}")
    print(f"  Stats: {OUT_STATS}")
    print(f"  Total concepts: {len(concepts)}")


if __name__ == "__main__":
    main()