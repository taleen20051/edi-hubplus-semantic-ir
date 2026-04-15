from __future__ import annotations

from pathlib import Path
import pandas as pd

IN_CSV = Path("data/processed/resources_included_only.csv")
OUT_CSV = Path("data/processed/taxonomy_raw.csv")

JOIN_COLUMNS = ["ID", "Title", "Link"]

TAXONOMY_COLUMNS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]

def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {IN_CSV}\n"
            "Run filter_included_resources.py first."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV, dtype=object)

    required_cols = JOIN_COLUMNS + TAXONOMY_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    taxonomy_df = df[required_cols].copy()

    taxonomy_df.to_csv(OUT_CSV, index=False)

    print(" Taxonomy raw dataset created (included-only)")
    print(f"Rows: {len(taxonomy_df)}")
    print(f"Saved to: {OUT_CSV}")

if __name__ == "__main__":
    main()