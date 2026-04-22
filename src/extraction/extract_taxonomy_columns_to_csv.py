from __future__ import annotations

from pathlib import Path
import pandas as pd

IN_CSV = Path("data/raw/resources_full_data_full.csv")
OUT_CSV = Path("data/processed/taxonomy_raw.csv")

TAXONOMY_COLUMNS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]

JOIN_KEYS = ["ID", "Title", "Link"]

def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {IN_CSV}\n"
            "Run extract_sheet5_to_csv.py first."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV, dtype=object)

    missing = [c for c in JOIN_KEYS + TAXONOMY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df[JOIN_KEYS + TAXONOMY_COLUMNS].copy()
    out.to_csv(OUT_CSV, index=False)

    print(" Taxonomy-only CSV created")
    print(f"Rows: {len(out)}")
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()