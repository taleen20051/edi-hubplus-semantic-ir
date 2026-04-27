"""
Build Ontology v1 from the cleaned EDI Hub+ taxonomy outputs.

Inputs:
- data/processed/taxonomy_clean.csv
- data/processed/normalisation_log.csv

Outputs:
- ontology/edi_ontology_v1.json
- data/processed/edi_ontology_v1_stats.json

Design policy for v1:
- each unique cleaned taxonomy tag becomes one concept
- each concept belongs to one of the four EDI taxonomy dimensions
- alternative labels come only from observed cleaning-log variants
- no external synonym expansion is added
- no inferred hierarchy is introduced beyond the four top-level categories
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple

import pandas as pd


# Cleaned taxonomy and normalisation log produced by the earlier data pipeline.
IN_CLEAN = Path("data/processed/taxonomy_clean.csv")
IN_LOG = Path("data/processed/normalisation_log.csv")

# Ontology JSON and summary statistics written by this script.
OUT_ONTOLOGY = Path("ontology/edi_ontology_v1.json")
OUT_STATS = Path("data/processed/edi_ontology_v1_stats.json")

ID_COL = "ID"

# The four taxonomy dimensions used as the ontology's top-level categories.
TAXONOMY_COLS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]

# Mapping from spreadsheet column names to stable category identifiers and labels.
CATEGORY_MAP = {
    "Individual Characteristics": ("individual_characteristics", "Individual Characteristics"),
    "Career Pathway": ("career_pathway", "Career Pathway"),
    "Research Funding Process": ("research_funding_process", "Research Funding Process"),
    "Organisational Culture": ("organisational_culture", "Organisational Culture"),
}


def slugify(label: str) -> str:
    """
    Convert a human-readable taxonomy label into a machine-friendly identifier.

    Example:
    "Visibility & Recognition" - "visibility_recognition"
    """
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "unnamed"


def split_clean_cell(cell: Any) -> List[str]:
    """
    Convert one cleaned taxonomy cell into a list of individual tags.

    Taxonomy cells can contain multiple comma-separated values because one
    resource may belong to more than one EDI concept within the same dimension.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    text = str(cell).strip()
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]


def build_alt_labels_from_log(log_df: pd.DataFrame) -> Dict[Tuple[str, str], Set[str]]:
    """
    Build alternative labels from the normalisation log.

    This lets the ontology preserve observed spreadsheet variants while keeping
    the preferred label as the cleaned canonical form.
    """
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

        # If the original value already equals the cleaned value, it is not a variant.
        if original == cleaned:
            continue

        key = (col, cleaned)
        alt.setdefault(key, set()).add(original)

    return alt


def main() -> None:
    # The cleaned taxonomy is required because it defines the canonical labels.
    if not IN_CLEAN.exists():
        raise FileNotFoundError(f"Missing input: {IN_CLEAN} (run Phase 1 cleaning first)")

    # The ontology can still be built without a log, but it will have no observed alt labels.
    if not IN_LOG.exists():
        log_df = pd.DataFrame()
    else:
        log_df = pd.read_csv(IN_LOG, dtype=object)

    # Read as object/string values to avoid unwanted type conversion of IDs or labels.
    df = pd.read_csv(IN_CLEAN, dtype=object)

    # Ensure the cleaned taxonomy contains the required identifier and taxonomy fields.
    missing = [c for c in [ID_COL, *TAXONOMY_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"taxonomy_clean.csv missing required columns: {missing}")

    alt_map = build_alt_labels_from_log(log_df)

    # Create the top-level category records used by the ontology JSON.
    categories = [{"id": CATEGORY_MAP[c][0], "label": CATEGORY_MAP[c][1]} for c in TAXONOMY_COLS]

    concepts: List[Dict[str, Any]] = []
    seen_concept_ids: Set[str] = set()

    per_category_counts: Dict[str, int] = {}
    per_category_unique_labels: Dict[str, Set[str]] = {CATEGORY_MAP[c][0]: set() for c in TAXONOMY_COLS}

    for col in TAXONOMY_COLS:
        cat_id, _ = CATEGORY_MAP[col]

        # Collect all unique cleaned labels used in this taxonomy column.
        unique_labels: Set[str] = set()
        for cell in df[col].tolist():
            for tag in split_clean_cell(cell):
                unique_labels.add(tag)

        for label in sorted(unique_labels):
            concept_id = f"{cat_id}__{slugify(label)}"

            # Ensure uniqueness even if two labels produce the same slug.
            if concept_id in seen_concept_ids:
                concept_id = f"{concept_id}__{abs(hash(label)) % 10**8}"

            seen_concept_ids.add(concept_id)
            per_category_unique_labels[cat_id].add(label)

            # Attach only observed variants from the cleaning log.
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

    # Main ontology artefact used by retrieval and evaluation.
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

    # Compact statistics file for reporting and validation checks.
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