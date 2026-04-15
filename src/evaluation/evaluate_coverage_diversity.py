from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import math


# Taxonomy category title setup
CATEGORY_COLUMNS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]

# Link every column in the spreadsheet taxonomy file to a specific theme ID used in the outputs.
THEME_ID = {
    "Individual Characteristics": "individual_characteristics",
    "Career Pathway": "career_pathway",
    "Research Funding Process": "research_funding_process",
    "Organisational Culture": "organisational_culture",
}

# Define the list of all themes to utilise when calculation coverage and diversity metrics.
FULL_THEME_LIST = [
    "individual_characteristics",
    "career_pathway",
    "research_funding_process",
    "organisational_culture",
]


# File input/output helpers
# Read a JSON file and return it as a Python dictionary.
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# Read a JSONL file and return all populated rows as a list of dictionaries.
def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# Write a Python object to disk as JSON, creating parent folders if required.
def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")



# Resource loading helpers
# Load the full finalised resource dataset and index it by ID for simpler lookup during evaluation.
def load_resource_index(resources_jsonl: Path) -> Dict[str, dict]:
    rows = load_jsonl(resources_jsonl)
    out: Dict[str, dict] = {}
    for r in rows:
        rid = str(r.get("ID", "")).strip()
        if rid:
            out[rid] = r
    return out


# Extract the theme IDs for one resource based on which taxonomy columns have tags/labels.
def get_resource_themes(resource: dict) -> List[str]:
    themes = []
    for col in CATEGORY_COLUMNS:
        vals = resource.get(col)
        if isinstance(vals, list) and len(vals) > 0:
            themes.append(THEME_ID[col])
        elif isinstance(vals, str) and vals.strip():
            # Accept populated string values in case the data was exported that way.
            themes.append(THEME_ID[col])
    return themes



# Coverage and diversity metrics
# Detect how many different themes are present in the top-k results, relative to the total number of themes in the dataset.
def coverage_at_k(rids: List[str], resources_by_id: Dict[str, dict], k: int) -> float:
    topk = rids[:k]
    present = set()

    for rid in topk:
        r = resources_by_id.get(str(rid))
        if not r:
            continue
        for t in get_resource_themes(r):
            present.add(t)

    return len(present) / float(len(FULL_THEME_LIST))


# Compute the top-k theme distribution as proportions. 
# If a resource belongs to different themes, it affects all of them.
def theme_distribution_at_k(rids: List[str], resources_by_id: Dict[str, dict], k: int) -> Dict[str, float]:
    topk = rids[:k]
    counts = {t: 0 for t in FULL_THEME_LIST}

    total_assignments = 0
    for rid in topk:
        r = resources_by_id.get(str(rid))
        if not r:
            continue
        themes = get_resource_themes(r)
        if not themes:
            continue
        for t in themes:
            counts[t] += 1
            total_assignments += 1

    if total_assignments == 0:
        return {t: 0.0 for t in FULL_THEME_LIST}

    return {t: counts[t] / float(total_assignments) for t in FULL_THEME_LIST}


# Convert a theme arrangement into a diversity ranking score between 0 and 1, 
# where 0 explains that all results belong to the same theme and 1 means the themes are exactly balanced.
def normalised_entropy(dist: Dict[str, float]) -> float:
    probs = [dist[t] for t in FULL_THEME_LIST if dist[t] > 0.0]
    if not probs:
        return 0.0

    H = 0.0
    for p in probs:
        H += -p * math.log(p)

    # Entropy is highest when all four themes are perfectly balanced.
    Hmax = math.log(len(FULL_THEME_LIST))
    return float(H / Hmax) if Hmax > 0 else 0.0



# Ranking loader
# Loads ranked outputs from a valid retrieval run format.
# Function converts both baseline-style and hybrid-style files into the same internal structure for evaluation.
def load_ranked_results(results_jsonl: Path) -> List[dict]:
    rows = load_jsonl(results_jsonl)
    out = []

    for r in rows:
        qid = str(r.get("qid", "")).strip()
        query = str(r.get("query", "")).strip()

        ranking_items = None
        if isinstance(r.get("ranking"), list):
            ranking_items = r["ranking"]
        elif isinstance(r.get("results"), list):
            ranking_items = r["results"]
        else:
            continue

        if not qid or not query:
            continue

        rids = [
            str(item.get("rid", "")).strip()
            for item in ranking_items
            if isinstance(item, dict) and str(item.get("rid", "")).strip()
        ]

        if rids:
            out.append({"qid": qid, "query": query, "rids": rids})

    if not out:
        raise ValueError(
            f"No valid rows found in rankings file: {results_jsonl}. "
            "Expected either a 'ranking' field or a 'results' field."
        )

    return out


# Main evaluation pipeline

# Calculate coverage and diversity metric scores for a single ranking file and record the results.
def main() -> None:
    # Read command-line inputs for the ranking file, the resource file, and the output location.
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True, help="Ranking results .jsonl (bm25/ontology/hybrid)")
    ap.add_argument("--resources", type=Path, required=True, help="resources_unified.jsonl")
    ap.add_argument("--k", type=int, default=10, help="cutoff k for @k metrics")
    ap.add_argument("--out", type=Path, required=True, help="output json path")
    args = ap.parse_args()

    # Load the resource dataset information and the ranked outputs that will be analysed.
    resources_by_id = load_resource_index(args.resources)
    rankings = load_ranked_results(args.results)

    # Empty containers to store per-query outputs and macro averages for all search queries.
    per_query = {}
    covs = []
    ents = []

    # Analyse each query individually to view how different methods perform before computing combines averages.
    for row in rankings:
        qid = row["qid"]
        query = row["query"]
        rids = row["rids"]

        # Calculate both theme coverage and balance for a query's top-k ranking and store its results.
        cov = coverage_at_k(rids, resources_by_id, args.k)
        dist = theme_distribution_at_k(rids, resources_by_id, args.k)
        ent = normalised_entropy(dist)

        per_query[qid] = {
            "query": query,
            "coverage@k": cov,
            "theme_distribution@k": dist,
            "diversity_entropy@k": ent,
        }

        covs.append(cov)
        ents.append(ent)

    # Build the final output object with the macro averages, per-query specifics, and metadata.
    out_obj = {
        "k": args.k,
        "query_count": len(per_query),
        "macro_avg": {
            "coverage@k": sum(covs) / max(1, len(covs)),
            "diversity_entropy@k": sum(ents) / max(1, len(ents)),
        },
        "per_query": per_query,
        "notes": {
            "coverage": "coverage@k = (#unique themes present in top-k)/4 using manual spreadsheet tags.",
            "diversity": "diversity_entropy@k is normalised entropy over the theme distribution in top-k (0=single theme, 1=balanced).",
            "theme_assignment": "a resource may contribute to multiple themes if it has tags in multiple taxonomy columns.",
        },
        "inputs": {
            "results": str(args.results),
            "resources": str(args.resources),
        },
    }

    # Record the evaluation summary list and show confirmation of completion.
    write_json(args.out, out_obj)
    print("Saved:", args.out)
    print("Macro avg:", out_obj["macro_avg"])


if __name__ == "__main__":
    main()