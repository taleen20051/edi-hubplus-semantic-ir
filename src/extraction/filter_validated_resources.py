from __future__ import annotations

from pathlib import Path
import pandas as pd

# Manifest produced after link validation.
IN_MANIFEST = Path("data/processed/resource_manifest.csv")
# Output file containing only resources approved for text extraction.
OUT_VALIDATED = Path("data/processed/resources_validated_for_extraction.csv")


def main() -> None:
    # Ensure the validation stage has already been completed.
    if not IN_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {IN_MANIFEST}\n"
            "Run validate_links.py first."
        )

    # Create output directory if required.
    OUT_VALIDATED.parent.mkdir(parents=True, exist_ok=True)

    # Read manifest values as text for consistency.
    df = pd.read_csv(IN_MANIFEST, dtype=object)

    # Confirm that key validation columns are available.
    required = ["ID", "Title", "Link", "url_status", "detected_type", "final_decision"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in manifest: {missing}")

    # Keep only accessible resources approved for extraction in supported formats.
    validated = df[
        (df["final_decision"] == "extract")
        & (df["url_status"] == "ok")
        & (df["detected_type"].isin(["pdf", "html"]))
    ].copy()

    # Save filtered dataset for the extraction pipeline.
    validated.to_csv(OUT_VALIDATED, index=False)

    print(" Validated extraction set created")
    print(f"Rows in manifest: {len(df)}")
    print(f"Rows validated for extraction: {len(validated)}")
    print(f"Saved to: {OUT_VALIDATED}")
    print("\nBreakdown by detected_type:")
    print(validated["detected_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()