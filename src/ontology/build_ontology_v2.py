from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------------
# Project root + defaults
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V1_PATH = PROJECT_ROOT / "ontology" / "edi_ontology_v1.json"


# -----------------------------
# Helpers
# -----------------------------
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def now_utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    return [str(x)]


def index_concepts_by_id(ontology: dict) -> Dict[str, dict]:
    concepts = ontology.get("concepts", []) or []
    return {c["id"]: c for c in concepts if "id" in c}


# -----------------------------
# Patch operations
# -----------------------------
def add_alt_labels(concept: dict, labels: List[str]) -> Dict[str, Any]:
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
    prev = concept.get("broader")
    concept["broader"] = broader
    return {"previous": prev, "new": broader}


def add_new_concept(ontology: dict, new_c: dict) -> None:
    ontology.setdefault("concepts", [])
    ontology["concepts"].append(new_c)


# -----------------------------
# Core patch application
# -----------------------------
def apply_patch(v1: dict, patch: dict) -> Tuple[dict, dict]:
    """
    Applies controlled patch changes and returns (v2_ontology, change_log).
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

    # Deep copy v1
    out = json.loads(json.dumps(v1))

    # Update metadata
    out["version"] = patch.get("version", "v2")
    out["created_utc"] = now_utc_iso()
    out["description"] = (
        (out.get("description") or "").strip()
        + f"\n\nOntology {out['version']} constructed from v1 using controlled patch iteration {patch.get('iteration_id')}."
    ).strip()

    idx = index_concepts_by_id(out)
    changes = patch.get("changes", {}) or {}

    # 1) Add alt labels
    for row in changes.get("add_alt_labels", []) or []:
        cid = str(row.get("concept_id", "")).strip()
        labels = ensure_list(row.get("alt_labels_to_add", []))
        if cid not in idx:
            log["warnings"].append(f"add_alt_labels: concept_id not found: {cid}")
            continue
        res = add_alt_labels(idx[cid], labels)
        log["changes_applied"]["add_alt_labels"].append({"concept_id": cid, **res})

    # 2) Remove alt labels
    for row in changes.get("remove_alt_labels", []) or []:
        cid = str(row.get("concept_id", "")).strip()
        labels = ensure_list(row.get("alt_labels_to_remove", []))
        if cid not in idx:
            log["warnings"].append(f"remove_alt_labels: concept_id not found: {cid}")
            continue
        res = remove_alt_labels(idx[cid], labels)
        log["changes_applied"]["remove_alt_labels"].append({"concept_id": cid, **res})

    # 3) Set broader relations (optional in iter01; more useful iter03)
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

    # 4) Add new concepts (optional; if used must match schema: category_id)
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

        if not concept_obj["pref_label"]:
            log["warnings"].append(f"add_new_concepts: missing pref_label for {new_id}")
        if not concept_obj["category_id"]:
            log["warnings"].append(f"add_new_concepts: missing category_id for {new_id}")

        add_new_concept(out, concept_obj)
        log["changes_applied"]["add_new_concepts"].append({"concept_id": new_id, "created": concept_obj})
        idx[new_id] = concept_obj

    return out, log


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ontology v2 iteration by applying a controlled patch to v1.")
    p.add_argument("--v1", type=str, default=str(DEFAULT_V1_PATH), help="Path to ontology v1 JSON.")
    p.add_argument("--iter", type=str, required=True, help="Iteration id, e.g. v2_iter01")
    p.add_argument("--patch", type=str, required=True, help="Path to patch JSON for this iteration.")
    return p.parse_args()


def main() -> None:
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

    # Iteration-scoped output paths
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