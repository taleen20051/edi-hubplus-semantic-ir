import argparse
import json
import csv
from typing import Dict, List


CAT_ORDER = [
    ("individual_characteristics", "Individual Characteristics"),
    ("career_pathway", "Career Pathway"),
    ("research_funding_process", "Research Funding Process"),
    ("organisational_culture", "Organisational Culture"),
]


def load_ontology(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_cell(pref_label: str, alt_labels: List[str]) -> str:
    # Keep it Excel-friendly:
    # - No commas required
    # - Use " | " as an internal separator
    items = [pref_label.strip()] + [a.strip() for a in alt_labels if a and a.strip()]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for x in items:
        key = x.lower()
        if key not in seen:
            out.append(x)
            seen.add(key)
    return " | ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True, help="Path to ontology JSON")
    ap.add_argument("--out", required=True, help="Output CSV path (4-column table)")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    onto = load_ontology(args.ontology)

    # Group concepts by category_id
    by_cat: Dict[str, List[dict]] = {cid: [] for cid, _ in CAT_ORDER}
    for c in onto.get("concepts", []):
        cid = c.get("category_id")
        if cid in by_cat:
            by_cat[cid].append(c)

    # Sort concepts in each category by pref_label for a stable table
    for cid in by_cat:
        by_cat[cid].sort(key=lambda x: (x.get("pref_label") or "").lower())

    # Build rows: max length across the four lists
    max_rows = max(len(by_cat[cid]) for cid, _ in CAT_ORDER) if by_cat else 0

    headers = [label for _, label in CAT_ORDER]

    rows = []
    for i in range(max_rows):
        row = []
        for cid, _label in CAT_ORDER:
            if i < len(by_cat[cid]):
                c = by_cat[cid][i]
                pref = c.get("pref_label", "")
                alts = c.get("alt_labels") or []
                row.append(format_cell(pref, alts))
            else:
                row.append("")
        rows.append(row)

    # Write CSV robustly (proper quoting)
    with open(args.out, "w", encoding=args.encoding, newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(headers)
        w.writerows(rows)

    print(f"Wrote 4-column ontology table: {args.out}")
    print(f"Ontology version: {onto.get('version')}")
    print(f"Concepts: {len(onto.get('concepts', []))}")


if __name__ == "__main__":
    main()