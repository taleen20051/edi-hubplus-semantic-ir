#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def label_cell(pref: str, alts: List[str], show_alts: bool) -> str:
    pref = (pref or "").strip()
    alts = [a.strip() for a in (alts or []) if a and a.strip()]
    if not show_alts or not alts:
        return pref
    # Excel-friendly: use semicolons inside the cell (no commas)
    return f"{pref} | " + "; ".join(alts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True, help="Path to ontology JSON")
    ap.add_argument("--out_csv", required=True, help="Path to write 4-column CSV")
    ap.add_argument("--show_alt_labels", action="store_true", help="If set, include alt_labels in each cell")
    ap.add_argument("--sort", action="store_true", help="If set, sort concepts alphabetically within each category")
    args = ap.parse_args()

    onto = load_json(Path(args.ontology))
    concepts = onto.get("concepts", [])

    # Map category_id -> list of cell strings
    cols: Dict[str, List[str]] = {
        "individual_characteristics": [],
        "career_pathway": [],
        "research_funding_process": [],
        "organisational_culture": [],
    }

    for c in concepts:
        cat = c.get("category_id", "")
        if cat not in cols:
            continue
        cell = label_cell(
            pref=c.get("pref_label", ""),
            alts=c.get("alt_labels", []) or [],
            show_alts=args.show_alt_labels,
        )
        cols[cat].append(cell)

    if args.sort:
        for k in cols:
            cols[k] = sorted(cols[k], key=lambda s: s.lower())

    # Align rows by max length (blank cells are normal)
    max_len = max(len(v) for v in cols.values()) if cols else 0

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Individual Characteristics", "Career Pathway", "Research Funding Process", "Organisational Culture"])
        for i in range(max_len):
            w.writerow([
                cols["individual_characteristics"][i] if i < len(cols["individual_characteristics"]) else "",
                cols["career_pathway"][i] if i < len(cols["career_pathway"]) else "",
                cols["research_funding_process"][i] if i < len(cols["research_funding_process"]) else "",
                cols["organisational_culture"][i] if i < len(cols["organisational_culture"]) else "",
            ])

    print(f"Wrote 4-column CSV: {out_path}")
    print(f"show_alt_labels={args.show_alt_labels}, sort={args.sort}")


if __name__ == "__main__":
    main()