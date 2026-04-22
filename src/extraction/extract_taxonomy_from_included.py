from __future__ import annotations

from pathlib import Path
import pandas as pd

# Input file including only resources marked as included in the spreadsheet
INPUT_CSV_FILE = Path("data/processed/resources_included_only.csv")
# Output file for full extracted taxonomy labels
OUTPUT_CSV_FILE = Path("data/processed/taxonomy_raw.csv")

# Base metadata columns taken for joining and identification
CORE_COLUMNS = ["ID", "Title", "Link"]

# Main EDI column headers derived from the spreadsheet
TAXONOMY_HEADER_COLUMNS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]

# Generate a normalised CSV that includes only core metadata and taxonomy fields
def main() -> None:
    if not INPUT_CSV_FILE.exists():
        raise FileNotFoundError(
            f"Input CSV not detected: {INPUT_CSV_FILE}\n"
            "Run filter_included_resources.py first."
        )

    # Build the output file if it does not already exist
    OUTPUT_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Read all values as text to maintain original spreadsheet content
    source_df = pd.read_csv(INPUT_CSV_FILE, dtype=object)

    # Ensure that all expected columns are detected before extraction
    required_cols = CORE_COLUMNS + TAXONOMY_HEADER_COLUMNS
    missing = [c for c in required_cols if c not in source_df.columns]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Available columns: {list(source_df.columns)}"
        )

    # Keep only the selected columns in a new dataframe
    taxonomy_df = source_df[required_cols].copy()

    # Save the final extracted taxonomy dataset to be used for processing and analysis
    taxonomy_df.to_csv(OUTPUT_CSV_FILE, index=False)

    print(" Taxonomy raw dataset generated (included-only)")
    print(f"Rows: {len(taxonomy_df)}")
    print(f"Saved to: {OUTPUT_CSV_FILE}")

if __name__ == "__main__":
    main()