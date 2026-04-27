#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Any


def iter_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_run_jsonl(path: str | Path) -> Dict[str, List[str]]:
    """
    Supports JSONL lines like:
      A) {"qid": "q1", "ranking": [{"rid": "44", ...}, ...]}
      B) {"qid": "q1", "results": [{"rid": "44", ...}, ...]}
    Returns: {qid: [rid1, rid2, ...]} in the order provided.
    """
    out: Dict[str, List[str]] = {}
    for obj in iter_jsonl(path):
        qid = str(obj["qid"])
        ranking = obj.get("ranking") or obj.get("results") or []
        out[qid] = [str(item["rid"]) for item in ranking]
    return out


def load_rid_to_themes(taxonomy_csv: str | Path) -> Dict[str, Set[str]]:
    """
    Builds rid -> set(themes) using taxonomy_clean.csv (or similar).

    We treat the 4 top-level columns as 'themes':
      - Individual Characteristics
      - Career Pathway
      - Research Funding Process
      - Organisational Culture
    """
    THEMES = [
        "Individual Characteristics",
        "Career Pathway",
        "Research Funding Process",
        "Organisational Culture",
    ]

    rid_to_themes: Dict[str, Set[str]] = {}
    with open(taxonomy_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rid_to_themes

        # try common id columns
        id_col = None
        for c in ["id", "rid", "ID", "resource_id", "ResourceID"]:
            if c in reader.fieldnames:
                id_col = c
                break
        if id_col is None:
            id_col = reader.fieldnames[0]

        for row in reader:
            rid = str(row.get(id_col, "")).strip()
            if not rid:
                continue

            themes: Set[str] = set()
            for t in THEMES:
                val = str(row.get(t, "")).strip()
                if val:
                    themes.add(t)

            rid_to_themes[rid] = themes

    return rid_to_themes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run JSONL path")
    ap.add_argument("--taxonomy_csv", default="data/processed/taxonomy_clean.csv", help="taxonomy_clean.csv path")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    run = load_run_jsonl(args.run)
    rid_to_themes = load_rid_to_themes(args.taxonomy_csv)

    THEMES = [
        "Individual Characteristics",
        "Career Pathway",
        "Research Funding Process",
        "Organisational Culture",
    ]
    num_themes_total = len(THEMES)

    per_query_subtopic_recall: Dict[str, float] = {}
    per_query_theme_counts: Dict[str, int] = {}
    missing_theme_info = 0

    for qid, ranked in run.items():
        topk = ranked[: args.k]
        found: Set[str] = set()
        for rid in topk:
            if rid in rid_to_themes:
                found |= rid_to_themes[rid]
            else:
                missing_theme_info += 1

        per_query_theme_counts[qid] = len(found)
        per_query_subtopic_recall[qid] = (len(found) / num_themes_total) if num_themes_total else 0.0

    # macro averages across queries present in the run
    if not per_query_subtopic_recall:
        print("No queries found in run file.")
        return 2

    macro_subtopic_recall = sum(per_query_subtopic_recall.values()) / len(per_query_subtopic_recall)
    macro_theme_count = sum(per_query_theme_counts.values()) / len(per_query_theme_counts)

    out = {
        f"SubtopicRecall@{args.k}": macro_subtopic_recall,
        f"AvgUniqueThemes@{args.k}": macro_theme_count,
        "themes_total": num_themes_total,
        "num_queries_in_run": len(per_query_subtopic_recall),
        "missing_rid_theme_rows_seen_in_topk": missing_theme_info,
    }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())