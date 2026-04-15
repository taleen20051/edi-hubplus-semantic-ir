# src/ontology/validate_ontology_v1.py
"""
Validate Ontology v1 (spreadsheet-fidelity) for structural integrity.

Checks:
- No duplicate concept.id
- Every category_id is one of the 4 allowed
- pref_label is non-empty
- alt_labels does not contain pref_label
- No placeholder labels like "Not Specified" were included as concepts
- Counts match edi_ontology_v1_stats.json (total + per category)
- Optional: writes a validation report JSON

Inputs:
- data/processed/edi_ontology_v1.json
- data/processed/edi_ontology_v1_stats.json

Output:
- Console PASS/FAIL with details
- Optional JSON report: data/processed/edi_ontology_v1_validation.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


IN_ONTOLOGY = Path("ontology/edi_ontology_v1.json")
IN_STATS = Path("data/processed/edi_ontology_v1_stats.json")
OUT_REPORT = Path("data/processed/edi_ontology_v1_validation.json")

ALLOWED_CATEGORY_IDS = {
    "individual_characteristics",
    "career_pathway",
    "research_funding_process",
    "organisational_culture",
}

# Things we should never allow to become concepts in v1
FORBIDDEN_LABELS_CASEFOLD = {
    "not specified",
    "n/a",
    "na",
    "none",
    "null",
    "",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _casefold(s: str) -> str:
    return (s or "").strip().casefold()


def _norm_space(s: str) -> str:
    """Normalise whitespace for exact-string comparisons."""
    return " ".join((s or "").strip().split())


def main() -> None:
    ontology = _load_json(IN_ONTOLOGY)
    stats = _load_json(IN_STATS)

    errors: List[str] = []
    warnings: List[str] = []

    # Basic shape checks
    if "concepts" not in ontology or not isinstance(ontology["concepts"], list):
        errors.append("Ontology missing 'concepts' list.")
        concepts: List[Dict[str, Any]] = []
    else:
        concepts = ontology["concepts"]

    if "categories" not in ontology or not isinstance(ontology["categories"], list):
        warnings.append("Ontology missing 'categories' list (not fatal, but unexpected).")

    # Check: no duplicate ids
    ids = [c.get("id") for c in concepts]
    id_counts = Counter(ids)
    dup_ids = [cid for cid, ct in id_counts.items() if cid is not None and ct > 1]
    if dup_ids:
        errors.append(f"Duplicate concept.id found: {dup_ids[:10]}{'...' if len(dup_ids) > 10 else ''}")

    # Per-category counts (actual)
    actual_counts = Counter()
    n_empty_pref = 0
    n_bad_category = 0
    n_forbidden_labels = 0
    n_alt_contains_pref = 0

    bad_category_examples: List[Tuple[str, Any]] = []
    forbidden_label_examples: List[str] = []
    alt_contains_pref_examples: List[str] = []
    empty_pref_examples: List[str] = []

    # Check each concept fields
    for c in concepts:
        cid = c.get("id", "<missing id>")
        pref = c.get("pref_label")
        category_id = c.get("category_id")
        alt_labels = c.get("alt_labels", [])

        # pref_label non-empty
        if pref is None or not str(pref).strip():
            n_empty_pref += 1
            empty_pref_examples.append(str(cid))
        else:
            pref_cf = _casefold(str(pref))
            # Forbidden label check
            if pref_cf in FORBIDDEN_LABELS_CASEFOLD:
                n_forbidden_labels += 1
                forbidden_label_examples.append(f"{cid} -> '{pref}'")

        # category_id allowed
        if category_id not in ALLOWED_CATEGORY_IDS:
            n_bad_category += 1
            bad_category_examples.append((str(cid), category_id))
        else:
            actual_counts[category_id] += 1

        # alt_labels sanity
        if alt_labels is None:
            alt_labels = []
        if not isinstance(alt_labels, list):
            errors.append(f"{cid}: alt_labels is not a list (found {type(alt_labels).__name__}).")
            alt_labels = []

        # alt_labels must not include pref_label as an exact duplicate.
        # NOTE: In Ontology v1 we *allow* casing-only variants (e.g., "Pregnancy and Maternity")
        # because they are observed spreadsheet variants. We only reject true duplicates.
        if pref is not None and str(pref).strip():
            pref_norm = _norm_space(str(pref))
            for a in alt_labels:
                if _norm_space(str(a)) == pref_norm:
                    n_alt_contains_pref += 1
                    alt_contains_pref_examples.append(f"{cid} pref='{pref}' alt='{a}'")
                    break

    if n_empty_pref > 0:
        errors.append(
            f"{n_empty_pref} concepts have empty pref_label. Examples: {empty_pref_examples[:5]}"
        )

    if n_forbidden_labels > 0:
        errors.append(
            f"{n_forbidden_labels} concepts have forbidden placeholder labels (e.g., 'Not Specified'). "
            f"Examples: {forbidden_label_examples[:5]}"
        )

    if n_bad_category > 0:
        errors.append(
            f"{n_bad_category} concepts have invalid category_id. Examples: {bad_category_examples[:5]}"
        )

    if n_alt_contains_pref > 0:
        errors.append(
            f"{n_alt_contains_pref} concepts have alt_labels containing pref_label (case-insensitive). "
            f"Examples: {alt_contains_pref_examples[:5]}"
        )

    # Check: counts match stats file
    expected_total = stats.get("n_concepts_total")
    if isinstance(expected_total, int):
        if len(concepts) != expected_total:
            errors.append(
                f"Total concept count mismatch: ontology has {len(concepts)} but stats expects {expected_total}"
            )
    else:
        warnings.append("Stats file missing integer 'n_concepts_total'.")

    expected_per_category = stats.get("concepts_per_category")
    if isinstance(expected_per_category, dict):
        # Compare each allowed category id
        for cat in sorted(ALLOWED_CATEGORY_IDS):
            expected = expected_per_category.get(cat)
            actual = actual_counts.get(cat, 0)
            if isinstance(expected, int):
                if actual != expected:
                    errors.append(
                        f"Per-category count mismatch for '{cat}': ontology has {actual} but stats expects {expected}"
                    )
            else:
                warnings.append(f"Stats missing integer per-category count for '{cat}'.")
    else:
        warnings.append("Stats file missing dict 'concepts_per_category'.")

    # Summary report
    passed = len(errors) == 0
    report = {
        "validated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "inputs": {"ontology": str(IN_ONTOLOGY), "stats": str(IN_STATS)},
        "passed": passed,
        "n_concepts_seen": len(concepts),
        "actual_counts_per_category": dict(actual_counts),
        "checks": {
            "duplicate_ids": len(dup_ids) == 0,
            "allowed_category_ids": n_bad_category == 0,
            "pref_label_non_empty": n_empty_pref == 0,
            "no_placeholder_labels": n_forbidden_labels == 0,
            "alt_labels_exclude_pref_label": n_alt_contains_pref == 0,
            "counts_match_stats": (
                (isinstance(expected_total, int) and len(concepts) == expected_total)
                and isinstance(expected_per_category, dict)
                and all(
                    isinstance(expected_per_category.get(cat), int)
                    and actual_counts.get(cat, 0) == expected_per_category.get(cat)
                    for cat in ALLOWED_CATEGORY_IDS
                )
            ),
        },
        "errors": errors,
        "warnings": warnings,
    }

    # Write report JSON (always, so you can cite it in the dissertation)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console output
    if passed:
        print(" Ontology v1 validation: PASS")
        print(f"Report written to: {OUT_REPORT}")
        print(f"Total concepts: {len(concepts)}")
        for cat in sorted(ALLOWED_CATEGORY_IDS):
            print(f"  {cat}: {actual_counts.get(cat, 0)}")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        sys.exit(0)
    else:
        print(" Ontology v1 validation: FAIL")
        print(f"Report written to: {OUT_REPORT}\n")
        print("Errors:")
        for e in errors:
            print(f"  - {e}")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        sys.exit(1)


if __name__ == "__main__":
    main()