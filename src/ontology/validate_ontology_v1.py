from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Ontology and stats file produced by build_ontology_v1.py.
IN_ONTOLOGY = Path("ontology/edi_ontology_v1.json")
IN_STATS = Path("data/processed/edi_ontology_v1_stats.json")

# Validation report written for reproducibility and dissertation evidence.
OUT_REPORT = Path("data/processed/edi_ontology_v1_validation.json")

# The only valid top-level category IDs in the four-column EDI taxonomy.
ALLOWED_CATEGORY_IDS = {
    "individual_characteristics",
    "career_pathway",
    "research_funding_process",
    "organisational_culture",
}

# Placeholder values should not become ontology concepts.
FORBIDDEN_LABELS_CASEFOLD = {
    "not specified",
    "n/a",
    "na",
    "none",
    "null",
    "",
}


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a required JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _casefold(s: str) -> str:
    """Normalise text for case-insensitive placeholder checks."""
    return (s or "").strip().casefold()


def _norm_space(s: str) -> str:
    """Normalise whitespace for exact-string comparisons."""
    return " ".join((s or "").strip().split())


def main() -> None:
    """Run ontology validation checks and write a JSON report."""
    ontology = _load_json(IN_ONTOLOGY)
    stats = _load_json(IN_STATS)

    errors: List[str] = []
    warnings: List[str] = []

    # Check the expected top-level ontology structure.
    if "concepts" not in ontology or not isinstance(ontology["concepts"], list):
        errors.append("Ontology missing 'concepts' list.")
        concepts: List[Dict[str, Any]] = []
    else:
        concepts = ontology["concepts"]

    if "categories" not in ontology or not isinstance(ontology["categories"], list):
        warnings.append("Ontology missing 'categories' list (not fatal, but unexpected).")

    # Concept identifiers must be unique.
    ids = [c.get("id") for c in concepts]
    id_counts = Counter(ids)
    dup_ids = [cid for cid, ct in id_counts.items() if cid is not None and ct > 1]
    if dup_ids:
        errors.append(f"Duplicate concept.id found: {dup_ids[:10]}{'...' if len(dup_ids) > 10 else ''}")

    # Counters used for validation summaries and final report output.
    actual_counts = Counter()
    n_empty_pref = 0
    n_bad_category = 0
    n_forbidden_labels = 0
    n_alt_contains_pref = 0

    bad_category_examples: List[Tuple[str, Any]] = []
    forbidden_label_examples: List[str] = []
    alt_contains_pref_examples: List[str] = []
    empty_pref_examples: List[str] = []

    # Validate each concept object.
    for c in concepts:
        cid = c.get("id", "<missing id>")
        pref = c.get("pref_label")
        category_id = c.get("category_id")
        alt_labels = c.get("alt_labels", [])

        # Preferred labels must be present and non-empty.
        if pref is None or not str(pref).strip():
            n_empty_pref += 1
            empty_pref_examples.append(str(cid))
        else:
            pref_cf = _casefold(str(pref))
            if pref_cf in FORBIDDEN_LABELS_CASEFOLD:
                n_forbidden_labels += 1
                forbidden_label_examples.append(f"{cid} -> '{pref}'")

        # Category IDs must belong to the four accepted taxonomy dimensions.
        if category_id not in ALLOWED_CATEGORY_IDS:
            n_bad_category += 1
            bad_category_examples.append((str(cid), category_id))
        else:
            actual_counts[category_id] += 1

        # Alternative labels should be stored as a list.
        if alt_labels is None:
            alt_labels = []
        if not isinstance(alt_labels, list):
            errors.append(f"{cid}: alt_labels is not a list (found {type(alt_labels).__name__}).")
            alt_labels = []

        # alt_labels must not include the preferred label as an exact duplicate.
        # Casing-only variants are allowed because they may be observed spreadsheet variants.
        if pref is not None and str(pref).strip():
            pref_norm = _norm_space(str(pref))
            for a in alt_labels:
                if _norm_space(str(a)) == pref_norm:
                    n_alt_contains_pref += 1
                    alt_contains_pref_examples.append(f"{cid} pref='{pref}' alt='{a}'")
                    break

    # Convert accumulated validation failures into readable error messages.
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

    # Check that ontology counts match the statistics file produced during construction.
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

    # Write a structured validation report whether validation passes or fails.
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

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary for quick terminal inspection.
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