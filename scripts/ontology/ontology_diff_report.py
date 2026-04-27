#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_by_id(ont: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for c in ont.get("concepts", []):
        cid = c.get("id", "")
        if cid:
            out[cid] = c
    return out


def stats(ont: dict) -> dict:
    concepts = ont.get("concepts", [])
    n_concepts = len(concepts)
    alt_counts = [len(c.get("alt_labels", []) or []) for c in concepts]
    n_with_alts = sum(1 for x in alt_counts if x > 0)
    n_alt_total = sum(alt_counts)
    return {
        "version": ont.get("version", ""),
        "n_concepts": n_concepts,
        "n_concepts_with_alt_labels": n_with_alts,
        "n_alt_labels_total": n_alt_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="Path to baseline ontology (e.g., v1)")
    ap.add_argument("--v2", required=True, help="Path to comparison ontology (e.g., v2_iter04)")
    ap.add_argument("--out", required=True, help="Where to write JSON diff report")
    args = ap.parse_args()

    v1_path = Path(args.v1)
    v2_path = Path(args.v2)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ont1 = load_json(v1_path)
    ont2 = load_json(v2_path)

    idx1 = index_by_id(ont1)
    idx2 = index_by_id(ont2)

    concept_ids: Set[str] = set(idx1.keys()) | set(idx2.keys())

    changes: List[dict] = []
    n_added_total = 0
    n_removed_total = 0

    for cid in sorted(concept_ids):
        c1 = idx1.get(cid)
        c2 = idx2.get(cid)

        if c1 is None:
            changes.append({
                "concept_id": cid,
                "change_type": "concept_added",
                "pref_label_v2": (c2 or {}).get("pref_label", ""),
            })
            continue
        if c2 is None:
            changes.append({
                "concept_id": cid,
                "change_type": "concept_removed",
                "pref_label_v1": (c1 or {}).get("pref_label", ""),
            })
            continue

        pref1 = (c1.get("pref_label", "") or "").strip()
        pref2 = (c2.get("pref_label", "") or "").strip()

        alts1 = [a for a in (c1.get("alt_labels", []) or []) if a and str(a).strip()]
        alts2 = [a for a in (c2.get("alt_labels", []) or []) if a and str(a).strip()]

        s1 = {_norm(a) for a in alts1}
        s2 = {_norm(a) for a in alts2}

        added_norm = sorted(s2 - s1)
        removed_norm = sorted(s1 - s2)

        norm_to_orig2 = {_norm(a): a for a in alts2}
        norm_to_orig1 = {_norm(a): a for a in alts1}

        added = [norm_to_orig2[n] for n in added_norm if n in norm_to_orig2]
        removed = [norm_to_orig1[n] for n in removed_norm if n in norm_to_orig1]

        pref_changed = pref1 != pref2
        alt_changed = bool(added or removed)

        if pref_changed or alt_changed:
            n_added_total += len(added)
            n_removed_total += len(removed)
            changes.append({
                "concept_id": cid,
                "category_id": c2.get("category_id", "") or c1.get("category_id", ""),
                "pref_label_v1": pref1,
                "pref_label_v2": pref2,
                "pref_label_changed": pref_changed,
                "alt_labels_added": added,
                "alt_labels_removed": removed,
            })

    report = {
        "input": {"v1": str(v1_path), "v2": str(v2_path)},
        "stats_v1": stats(ont1),
        "stats_v2": stats(ont2),
        "delta": {
            "alt_labels_added_total": n_added_total,
            "alt_labels_removed_total": n_removed_total,
            "concepts_changed": len(changes),
        },
        "changes": changes,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote diff report: {out_path}")
    print(f"Concepts changed: {len(changes)}")
    print(f"Alt labels added: {n_added_total} | removed: {n_removed_total}")


if __name__ == "__main__":
    main()