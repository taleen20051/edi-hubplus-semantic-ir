from __future__ import annotations

from pathlib import Path
import pandas as pd

# Source CSV exported from the original spreadsheet.
IN_CSV = Path("data/raw/resources_full_data_full.csv")
# Output CSV containing only rows approved for inclusion.
OUT_CSV = Path("data/processed/resources_included_only.csv")

# Spreadsheet column used to decide whether a resource is kept.
INCLUDED_COLUMN = "Included_Excluded"
# Rows matching this value are retained.
INCLUDED_VALUE = "Included"


def main() -> None:
    # Ensure the raw extracted dataset exists first.
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {IN_CSV}\n"
            "Run extract_sheet5_to_csv.py first."
        )

    # Create output directory if it does not already exist.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Read all values as text to preserve IDs and spreadsheet formatting.
    df = pd.read_csv(IN_CSV, dtype=object)

    # Confirm that the inclusion flag column is present.
    if INCLUDED_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{INCLUDED_COLUMN}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Keep only resources explicitly marked as Included.
    included_df = df[df[INCLUDED_COLUMN].str.strip() == INCLUDED_VALUE].copy()

    # Save filtered dataset for later validation and extraction stages.
    included_df.to_csv(OUT_CSV, index=False)

    print(" Included-only dataset created")
    print(f"Total rows (raw): {len(df)}")
    print(f"Included rows: {len(included_df)}")
    print(f"Saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()