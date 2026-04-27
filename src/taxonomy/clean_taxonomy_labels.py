"""
Clean and normalise the four taxonomy columns extracted from taxonomy_raw.csv.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


IN_RAW = Path("data/processed/taxonomy_raw.csv")

OUT_CLEAN = Path("data/processed/taxonomy_clean.csv")
OUT_LOG = Path("data/processed/normalisation_log.csv")
OUT_STATS = Path("data/processed/taxonomy_stats.json")

# Required identifier and taxonomy columns from taxonomy_raw.csv.
ID_COL = "ID"
TAXONOMY_COLS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]

# Values that should not become real ontology labels.
PLACEHOLDERS = {"not specified", "n/a", "na", "none", ""}

# Conservative typo whitelist.
TYPO_FIXES = {
    "microagression": "microaggression",
    "accessiblity": "accessibility",
    "eligiblity": "eligibility",
}


# Logging structure
@dataclass
class NormLogRow:
    """One recorded taxonomy-normalisation change."""

    resource_id: str
    column: str
    stage: str
    original: str
    cleaned: str
    reason: str


# Normalisation helpers
def collapse_spaces(s: str) -> str:
    """Trim text and replace repeated whitespace with a single space."""
    return re.sub(r"\s+", " ", s).strip()


# Standardise label casing while preserving meaningful tokens.
def title_case_preserve_acronyms(s: str) -> str:
    tokens = s.split(" ")
    out = []

    for token in tokens:
        if not token:
            continue

        if token.isupper():
            out.append(token)
        elif any(ch.isdigit() for ch in token):
            out.append(token)
        elif "+" in token or "-" in token or "&" in token or "/" in token:
            # Preserve symbolic labels, but capitalise the first letter if needed.
            out.append(token[:1].upper() + token[1:] if token[:1].islower() else token)
        else:
            out.append(token[:1].upper() + token[1:].lower())

    return " ".join(out)


# Apply minimal typo fixes using an explicit whitelist only.
def apply_typo_fixes(s: str) -> str:
    key = s.lower()
    if key in TYPO_FIXES:
        fixed = TYPO_FIXES[key]
        return fixed[:1].upper() + fixed[1:] if s[:1].isupper() else fixed
    return s


def is_placeholder(s: str) -> bool:
    """Return True if the label is a placeholder rather than a real tag."""
    return collapse_spaces(s).lower() in PLACEHOLDERS


# Split a spreadsheet multi-select cell into raw tag tokens.
def split_multiselect_cell(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    raw = str(value).strip()
    if raw == "":
        return []

    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p.strip() != ""]


def clean_cell_tags(
    resource_id: str,
    column: str,
    cell_value: Any,
    logs: List[NormLogRow],
) -> List[str]:
    """
    Clean one taxonomy cell into a list of final labels.
    """
    raw_parts = split_multiselect_cell(cell_value)

    # Record when a multi-select cell has been split into separate labels.
    if cell_value is not None and isinstance(cell_value, str) and "," in cell_value:
        logs.append(
            NormLogRow(
                resource_id=resource_id,
                column=column,
                stage="split",
                original=str(cell_value),
                cleaned=" | ".join(raw_parts),
                reason="Split comma-separated multi-select cell into individual tag tokens.",
            )
        )

    cleaned: List[str] = []
    seen_lower = set()

    for p in raw_parts:
        orig = p

        # 1) Remove placeholder labels.
        if is_placeholder(p):
            logs.append(
                NormLogRow(
                    resource_id=resource_id,
                    column=column,
                    stage="placeholder_remove",
                    original=orig,
                    cleaned="",
                    reason="Removed placeholder tag (e.g., Not Specified / NA / empty).",
                )
            )
            continue

        # 2) Collapse whitespace.
        p2 = collapse_spaces(p)
        if p2 != p:
            logs.append(
                NormLogRow(
                    resource_id=resource_id,
                    column=column,
                    stage="canonicalise",
                    original=orig,
                    cleaned=p2,
                    reason="Collapsed repeated spaces / trimmed whitespace.",
                )
            )
        p = p2

        # 3) Apply conservative typo fixes.
        p3 = apply_typo_fixes(p)
        if p3 != p:
            logs.append(
                NormLogRow(
                    resource_id=resource_id,
                    column=column,
                    stage="typo_fix",
                    original=p,
                    cleaned=p3,
                    reason="Applied minimal, pre-defined typo correction (whitelist).",
                )
            )
        p = p3

        # 4) Standardise casing.
        p4 = title_case_preserve_acronyms(p)
        if p4 != p:
            logs.append(
                NormLogRow(
                    resource_id=resource_id,
                    column=column,
                    stage="canonicalise",
                    original=p,
                    cleaned=p4,
                    reason="Standardised casing to Title Case while preserving acronyms/symbol tokens.",
                )
            )
        p = p4

        # 5) Remove duplicate labels within the same cell.
        key = p.lower()
        if key in seen_lower:
            logs.append(
                NormLogRow(
                    resource_id=resource_id,
                    column=column,
                    stage="dedupe",
                    original=p,
                    cleaned="",
                    reason="Removed duplicate tag within the same cell (case-insensitive).",
                )
            )
            continue

        seen_lower.add(key)
        cleaned.append(p)

    return cleaned


# Main pipeline
def main() -> None:
    """Run taxonomy cleaning and write clean CSV, log CSV, and stats JSON."""
    if not IN_RAW.exists():
        raise FileNotFoundError(
            f"Missing input: {IN_RAW}\n"
            "Run your extraction step that created taxonomy_raw.csv first."
        )

    OUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_RAW, dtype=object)

    # Confirm that all required columns are present before processing.
    missing = [c for c in [ID_COL, *TAXONOMY_COLS] if c not in df.columns]
    if missing:
        raise ValueError(
            f"taxonomy_raw.csv is missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    logs: List[NormLogRow] = []
    clean_rows: List[Dict[str, Any]] = []

    # Clean each taxonomy column for each resource row.
    for _, row in df.iterrows():
        rid = str(row[ID_COL]).strip()
        out_row: Dict[str, Any] = {ID_COL: rid}

        for col in TAXONOMY_COLS:
            tags = clean_cell_tags(
                resource_id=rid,
                column=col,
                cell_value=row.get(col, None),
                logs=logs,
            )

            # Store cleaned labels as a readable comma-separated string.
            out_row[col] = ", ".join(tags)

        clean_rows.append(out_row)

    # Write cleaned taxonomy dataset.
    clean_df = pd.DataFrame(clean_rows)
    clean_df.to_csv(OUT_CLEAN, index=False, encoding="utf-8")

    # Write a full log of all normalisation actions.
    log_df = pd.DataFrame([vars(x) for x in logs])
    log_df.to_csv(OUT_LOG, index=False, encoding="utf-8")

    def unique_tags_from_col(series: pd.Series) -> int:
        """Count unique cleaned labels in one taxonomy column."""
        tags = set()
        for cell in series.fillna("").astype(str).tolist():
            cell = cell.strip()
            if not cell:
                continue
            for t in [x.strip() for x in cell.split(",")]:
                if t:
                    tags.add(t.lower())
        return len(tags)

    # Write summary statistics for methodology and reporting.
    stats = {
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "input": str(IN_RAW),
        "output_clean_csv": str(OUT_CLEAN),
        "output_log_csv": str(OUT_LOG),
        "n_resources": int(len(clean_df)),
        "taxonomy_columns": TAXONOMY_COLS,
        "unique_tags_per_category": {
            col: unique_tags_from_col(clean_df[col]) for col in TAXONOMY_COLS
        },
        "log_rows": int(len(log_df)),
        "log_stage_counts": log_df["stage"].value_counts().to_dict() if len(log_df) else {},
        "policy_notes": {
            "placeholders_removed": sorted(list(PLACEHOLDERS)),
            "typo_fixes_whitelist": TYPO_FIXES,
            "dedupe_scope": "within-cell only (case-insensitive)",
            "no_semantic_enrichment": True,
        },
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Taxonomy cleaning complete")
    print(f" Cleaned taxonomy: {OUT_CLEAN}")
    print(f" Normalisation log: {OUT_LOG}")
    print(f" Stats: {OUT_STATS}")
    print(f" Rows processed: {len(clean_df)}")


if __name__ == "__main__":
    main()