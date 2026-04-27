from __future__ import annotations

from pathlib import Path
import pandas as pd

# Input file created after filtering the spreadsheet to included resources only.
INPUT_CSV_FILE = Path("data/processed/resources_included_only.csv")
# Output file containing only the taxonomy columns needed for ontology construction.
OUTPUT_CSV_FILE = Path("data/processed/taxonomy_raw.csv")

# Core metadata fields kept so taxonomy rows can be joined back to resources later.
CORE_COLUMNS = ["ID", "Title", "Link"]

# The four EDI taxonomy dimensions used as the conceptual basis of the project.
TAXONOMY_HEADER_COLUMNS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]


def main() -> None:
    # This script depends on the included-only dataset produced in the previous stage.
    if not INPUT_CSV_FILE.exists():
        raise FileNotFoundError(
            f"Input CSV not detected: {INPUT_CSV_FILE}\n"
            "Run filter_included_resources.py first."
        )

    # Ensure the processed data directory exists before saving the output.
    OUTPUT_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Read all columns as objects/strings to avoid pandas changing IDs or empty cells.
    source_df = pd.read_csv(INPUT_CSV_FILE, dtype=object)

    # Check that the spreadsheet export still contains the expected schema.
    required_cols = CORE_COLUMNS + TAXONOMY_HEADER_COLUMNS
    missing = [c for c in required_cols if c not in source_df.columns]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Available columns: {list(source_df.columns)}"
        )

    # Keep only identifiers and taxonomy fields for the next cleaning stage.
    taxonomy_df = source_df[required_cols].copy()

    # Save the raw taxonomy extract before later normalisation/cleaning.
    taxonomy_df.to_csv(OUTPUT_CSV_FILE, index=False)

    print(" Taxonomy raw dataset generated (included-only)")
    print(f"Rows: {len(taxonomy_df)}")
    print(f"Saved to: {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    main()