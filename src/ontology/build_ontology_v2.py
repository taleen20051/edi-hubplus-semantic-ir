"""
Build an ontology v2 iteration by applying a controlled patch to ontology v1.

This script supports reproducible ontology refinement. Instead of editing the
baseline ontology directly, a patch file describes the exact changes to apply.
The script then writes both the new ontology version and a change log.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Resolve paths from the repository root so the script works from different terminals.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V1_PATH = PROJECT_ROOT / "ontology" / "edi_ontology_v1.json"


# Helper functions for file I/O, normalisation, and concept indexing
def read_json(path: Path) -> dict:
    """Read a JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    """Write a dictionary as formatted JSON, creating parent folders if required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def norm(s: str) -> str:
    """Normalise labels for duplicate checking and comparison."""
    return " ".join((s or "").strip().lower().split())


def now_utc_iso() -> str:
    """Return the current UTC timestamp in ISO-like format."""
    return datetime.utcnow().isoformat() + "Z"


def ensure_list(x: Any) -> List[str]:
    """Convert a patch value into a clean list of strings."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    return [str(x)]


def index_concepts_by_id(ontology: dict) -> Dict[str, dict]:
    """Create a lookup from concept id to concept object."""
    concepts = ontology.get("concepts", []) or []
    return {c["id"]: c for c in concepts if "id" in c}


# Patch functionality
def add_alt_labels(concept: dict, labels: List[str]) -> Dict[str, Any]:
    """Add new alternative labels to a concept without duplicating existing labels."""
    before = ensure_list(concept.get("alt_labels"))
    before_norm = {norm(x) for x in before}

    added = []
    for lab in labels:
        lab = str(lab).strip()
        if not lab:
            continue
        if norm(lab) not in before_norm:
            before.append(lab)
            before_norm.add(norm(lab))
            added.append(lab)

    concept["alt_labels"] = before
    return {"added": added}


def remove_alt_labels(concept: dict, labels: List[str]) -> Dict[str, Any]:
    """Remove selected alternative labels from a concept."""
    before = ensure_list(concept.get("alt_labels"))
    remove_set = {norm(x) for x in labels if str(x).strip()}

    kept = []
    removed = []
    for lab in before:
        if norm(lab) in remove_set:
            removed.append(lab)
        else:
            kept.append(lab)

    concept["alt_labels"] = kept
    return {"removed": removed}


def set_broader(concept: dict, broader: str) -> Dict[str, Any]:
    """Set or replace an optional broader-concept relation."""
    prev = concept.get("broader")
    concept["broader"] = broader
    return {"previous": prev, "new": broader}


def add_new_concept(ontology: dict, new_c: dict) -> None:
    """Append a new concept object to the ontology."""
    ontology.setdefault("concepts", [])
    ontology["concepts"].append(new_c)


# Core patch application
def apply_patch(v1: dict, patch: dict) -> Tuple[dict, dict]:
    """
    Apply controlled patch changes to ontology v1.

    Returns:
      - v2 ontology dictionary
      - change log dictionary recording all applied changes and warnings
    """
    log: Dict[str, Any] = {
        "timestamp_utc": now_utc_iso(),
        "base_ontology_id": v1.get("ontology_id"),
        "base_version": v1.get("version"),
        "patch_version": patch.get("version"),
        "iteration_id": patch.get("iteration_id"),
        "changes_applied": {
            "add_alt_labels": [],
            "remove_alt_labels": [],
            "set_broader": [],
            "add_new_concepts": []
        },
        "warnings": [],
        "notes": "This log records exactly what was changed to construct v2 from v1."
    }

    # Deep copy v1 so the original dictionary is not modified in memory.
    out = json.loads(json.dumps(v1))

    # Update version metadata for the derived ontology.
    out["version"] = patch.get("version", "v2")
    out["created_utc"] = now_utc_iso()
    out["description"] = (
        (out.get("description") or "").strip()
        + f"\n\nOntology {out['version']} constructed from v1 using controlled patch iteration {patch.get('iteration_id')}."
    ).strip()

    idx = index_concepts_by_id(out)
    changes = patch.get("changes", {}) or {}

    # 1) Add alternative labels to existing concepts.
    for row in changes.get("add_alt_labels", []) or []:
        cid = str(row.get("concept_id", "")).strip()
        labels = ensure_list(row.get("alt_labels_to_add", []))
        if cid not in idx:
            log["warnings"].append(f"add_alt_labels: concept_id not found: {cid}")
            continue
        res = add_alt_labels(idx[cid], labels)
        log["changes_applied"]["add_alt_labels"].append({"concept_id": cid, **res})

    # 2) Remove alternative labels from existing concepts.
    for row in changes.get("remove_alt_labels", []) or []:
        cid = str(row.get("concept_id", "")).strip()
        labels = ensure_list(row.get("alt_labels_to_remove", []))
        if cid not in idx:
            log["warnings"].append(f"remove_alt_labels: concept_id not found: {cid}")
            continue
        res = remove_alt_labels(idx[cid], labels)
        log["changes_applied"]["remove_alt_labels"].append({"concept_id": cid, **res})

    # 3) Add optional broader relations where required by a refinement iteration.
    for row in changes.get("set_broader", []) or []:
        cid = str(row.get("concept_id", "")).strip()
        broader = str(row.get("broader", "")).strip()
        if cid not in idx:
            log["warnings"].append(f"set_broader: concept_id not found: {cid}")
            continue
        if not broader:
            log["warnings"].append(f"set_broader: empty broader for {cid}")
            continue
        res = set_broader(idx[cid], broader)
        log["changes_applied"]["set_broader"].append({"concept_id": cid, **res})

    # 4) Add new concepts if a patch explicitly introduces them.
    for new_c in changes.get("add_new_concepts", []) or []:
        new_id = str(new_c.get("id", "")).strip()
        if not new_id:
            log["warnings"].append("add_new_concepts: missing id on one new concept")
            continue
        if new_id in idx:
            log["warnings"].append(f"add_new_concepts: concept already exists: {new_id}")
            continue

        concept_obj = {
            "id": new_id,
            "pref_label": str(new_c.get("pref_label", "")).strip(),
            "alt_labels": ensure_list(new_c.get("alt_labels")),
            "category_id": str(new_c.get("category_id", "")).strip(),
            "source_column": str(new_c.get("source_column", "MANUAL_V2")).strip()
        }
        if new_c.get("broader"):
            concept_obj["broader"] = str(new_c["broader"]).strip()

        # Record schema issues as warnings while still writing a traceable output.
        if not concept_obj["pref_label"]:
            log["warnings"].append(f"add_new_concepts: missing pref_label for {new_id}")
        if not concept_obj["category_id"]:
            log["warnings"].append(f"add_new_concepts: missing category_id for {new_id}")

        add_new_concept(out, concept_obj)
        log["changes_applied"]["add_new_concepts"].append({"concept_id": new_id, "created": concept_obj})
        idx[new_id] = concept_obj

    return out, log


def parse_args() -> argparse.Namespace:
    """Define command-line arguments for building an ontology iteration."""
    p = argparse.ArgumentParser(description="Build ontology v2 iteration by applying a controlled patch to v1.")
    p.add_argument("--v1", type=str, default=str(DEFAULT_V1_PATH), help="Path to ontology v1 JSON.")
    p.add_argument("--iter", type=str, required=True, help="Iteration id, e.g. v2_iter01")
    p.add_argument("--patch", type=str, required=True, help="Path to patch JSON for this iteration.")
    return p.parse_args()


def main() -> None:
    """Load v1 and patch files, apply the patch, and write versioned outputs."""
    args = parse_args()
    v1_path = Path(args.v1)
    patch_path = Path(args.patch)
    iteration_id = args.iter.strip()

    if not v1_path.exists():
        raise FileNotFoundError(f"Missing v1 ontology: {v1_path}")
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing patch file: {patch_path}")

    v1 = read_json(v1_path)
    patch = read_json(patch_path)

    v2, change_log = apply_patch(v1, patch)

    # Store outputs in versioned ontology and change-log folders.
    out_dir = PROJECT_ROOT / "ontology" / "versions" / iteration_id
    log_dir = PROJECT_ROOT / "ontology" / "change_logs"

    v2_out_path = out_dir / f"edi_ontology_{iteration_id}.json"
    change_log_path = log_dir / f"{iteration_id}_change_log.json"

    write_json(v2_out_path, v2)
    write_json(change_log_path, change_log)

    print("Wrote:", v2_out_path)
    print("Wrote:", change_log_path)

    if change_log.get("warnings"):
        print("\nWARNINGS:")
        for w in change_log["warnings"]:
            print("-", w)


if __name__ == "__main__":
    main()