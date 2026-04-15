from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# Input and output paths for the full normalised dataset.
CLEAN_TAXONOMY_PATH = Path("data/processed/taxonomy_clean.csv")
TEXT_JSONL = Path("data/processed/resources_text.jsonl")

OUTPUT_UNIFIED_CSV = Path("data/processed/resources_unified.csv")
OUT_UNIFIED_JSONL = Path("data/processed/resources_unified.jsonl")
JOIN_REPORT_PATH = Path("data/processed/unified_join_report.json")

# Define the four taxonomy main categories that should be shown in the final version 1 CSV.
TAX_COLS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]

# The taxonomy CSV and extracted text JSONL use different ID field names.
# The CSV and JSONL file both store IDs as strings but different column names. 
ID_COL_TAX = "ID"
ID_COL_TEXT = "id"


# Helper functions
# Create parent directories for output files that don't already exist, to avoid errors when saving results.
def create_output_dirs(*paths: Path) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


# Parse taxonomy labels into a normalised list format, handling various input formats and edge cases.
# The output will store tags/labels as JSON strings like ["Age", "Career stage"] in the CSV, and as Python lists in the JSONL.
def convert_tax_cell_to_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    # Attempt to read the cell as a JSON list at first.
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(v).strip() for v in obj if str(v).strip()]
    except Exception:
        pass
    # If it's not a JSON list, treat it as a single string tag/label.
    return [s]


# Load the full extracted JSONL file into a pandas file, which can then be merged with the taxonomy DataSet.
def read_text_records(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df



# Main pipeline implementation

# Execute the final dataset file by merging cleaned taxonomy labels with extracted text.
# Save the result in both CSV and JSONL formats for downstream use.
def main() -> None:
    if not CLEAN_TAXONOMY_PATH.exists():
        raise FileNotFoundError(f"Missing taxonomy file: {CLEAN_TAXONOMY_PATH}")
    if not TEXT_JSONL.exists():
        raise FileNotFoundError(f"Missing text file: {TEXT_JSONL}")

    create_output_dirs(OUTPUT_UNIFIED_CSV, OUT_UNIFIED_JSONL, JOIN_REPORT_PATH)

    # 1/ Load the normalised taxonomy file.
    tax = pd.read_csv(CLEAN_TAXONOMY_PATH, dtype=str).copy()
    if ID_COL_TAX not in tax.columns:
        raise ValueError(f"Expected '{ID_COL_TAX}' column in {CLEAN_TAXONOMY_PATH}. Found: {list(tax.columns)}")

    # Ensure that all expected category columns are present.
    missing_tax_cols = [c for c in TAX_COLS if c not in tax.columns]
    if missing_tax_cols:
        raise ValueError(
            f"Missing taxonomy columns in taxonomy_clean.csv: {missing_tax_cols}\n"
            f"Available: {list(tax.columns)}"
        )

    # Change taxonomy cells into structured Python lists so they can be merged and visualised easily.
    for c in TAX_COLS:
        tax[c] = tax[c].apply(convert_tax_cell_to_list)

    tax[ID_COL_TAX] = tax[ID_COL_TAX].astype(str).str.strip()

    # 2/ Load the extracted JSONL text file.
    resource_text_df = read_text_records(TEXT_JSONL).copy()
    if ID_COL_TEXT not in resource_text_df.columns:
        raise ValueError(f"Expected '{ID_COL_TEXT}' field in JSONL records. Found: {list(resource_text_df.columns)}")

    resource_text_df[ID_COL_TEXT] = resource_text_df[ID_COL_TEXT].astype(str).str.strip()

    # 3/ Redefine the text ID column so both tables use the same join key.
    resource_text_df = resource_text_df.rename(columns={ID_COL_TEXT: ID_COL_TAX})

    # Accept only resources that include both finalised taxonomy labels and extracted text.
    unified = tax.merge(resource_text_df, on=ID_COL_TAX, how="inner")

    # 4/ Document how many IDs matched and which IDs are not seen on either side of the join.
    tax_ids = set(tax[ID_COL_TAX].dropna().tolist())
    text_ids = set(resource_text_df[ID_COL_TAX].dropna().tolist())

    missing_text = sorted(tax_ids - text_ids, key=lambda x: int(x) if x.isdigit() else x)
    extra_text = sorted(text_ids - tax_ids, key=lambda x: int(x) if x.isdigit() else x)

    # Keep a saved short join report so the final data file can be audited later.
    report = {
        "taxonomy_rows": int(len(tax)),
        "text_rows": int(len(resource_text_df)),
        "unified_rows_inner_join": int(len(unified)),
        "missing_text_ids": missing_text,
        "extra_text_ids": extra_text,
        "notes": {
            "join_policy": "INNER JOIN on ID: only resources with BOTH cleaned taxonomy labels and extracted text are included.",
            "expected_behavior": "Some Included/validated resources may fail extraction; these appear in missing_text_ids.",
        },
    }
    JOIN_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 5/ Export the resulting normalised dataset in both CSv and JSONL formats.
    # The CSV reserves taxonomy registers as JSON strings to ensure they can be read in a single cell.
    unified_csv = unified.copy()
    for c in TAX_COLS:
        unified_csv[c] = unified_csv[c].apply(lambda lst: json.dumps(lst, ensure_ascii=False))

    unified_csv.to_csv(OUTPUT_UNIFIED_CSV, index=False, encoding="utf-8")

    # The JSONL file keeps taxonomy labels as columns with list entries for downstream scripts.
    with open(OUT_UNIFIED_JSONL, "w", encoding="utf-8") as out_f:
        for _, row in unified.iterrows():
            # Change every merged row entry into a JSON-readable record.
            rec = row.to_dict()
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 6/ Print the summary of join outputs for quick future auditing and verification.
    print("\n Unified dataset created")
    print(f"Taxonomy rows: {len(tax)}")
    print(f"Text rows: {len(resource_text_df)}")
    print(f"Unified rows (INNER JOIN): {len(unified)}")
    print(f"\nSaved CSV:   {OUTPUT_UNIFIED_CSV}")
    print(f"Saved JSONL: {OUT_UNIFIED_JSONL}")
    print(f"Saved report:{JOIN_REPORT_PATH}")

    if missing_text:
        print(f"\n Missing text for {len(missing_text)} taxonomy IDs (kept out of unified dataset):")
        print(missing_text)
    if extra_text:
        print(f"\n Text exists for {len(extra_text)} IDs not found in taxonomy (unexpected):")
        print(extra_text)


if __name__ == "__main__":
    main()