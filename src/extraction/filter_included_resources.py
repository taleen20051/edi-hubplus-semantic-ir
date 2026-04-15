from __future__ import annotations

from pathlib import Path
import pandas as pd

IN_CSV = Path("data/raw/resources_full_data_full.csv")
OUT_CSV = Path("data/processed/resources_included_only.csv")

INCLUDED_COLUMN = "Included_Excluded"
INCLUDED_VALUE = "Included"

def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {IN_CSV}\n"
            "Run extract_sheet5_to_csv.py first."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV, dtype=object)

    if INCLUDED_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{INCLUDED_COLUMN}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Filter included resources only
    included_df = df[df[INCLUDED_COLUMN].str.strip() == INCLUDED_VALUE].copy()

    included_df.to_csv(OUT_CSV, index=False)

    print(" Included-only dataset created")
    print(f"Total rows (raw): {len(df)}")
    print(f"Included rows: {len(included_df)}")
    print(f"Saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()