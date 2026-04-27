from __future__ import annotations

import re
from typing import Dict, List, Set

from .io_utils import write_json


# Taxonomy columns used as gold labels from the unified dataset.
TAXONOMY_KEYS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]


# Normalisation helpers
def _concept_category(concept_id: str) -> str:
    """Extract the top-level category prefix from a concept ID."""
    return (concept_id or "").split("__", 1)[0].strip()


def _norm_label(x: str) -> str:
    """Lowercase, strip, and collapse whitespace for robust label matching."""
    s = (x or "").strip().lower()
    s = " ".join(s.split())
    return s


# Some spreadsheet cells contain multiple labels in one string, such as:
# "Inclusive Language, Allyship, Cultural Competency".
# This regex splits those cells into individual labels.
_SPLIT_RE = re.compile(r"\s*[,;]\s*")


def _split_multi_label(s: str) -> List[str]:
    """Split a taxonomy cell into individual labels."""
    s = (s or "").strip()
    if not s:
        return []
    parts = _SPLIT_RE.split(s)
    return [p.strip() for p in parts if p and p.strip()]


# Ontology and gold-label mapping
def build_label_to_concept_id(ontology: dict) -> Dict[str, str]:
    """
    Build a lookup from ontology labels to concept IDs.

    Both preferred labels and alternative labels are included so that manual
    taxonomy labels can be matched consistently to ontology concepts.
    """
    m: Dict[str, str] = {}

    for c in ontology.get("concepts", []):
        cid = c["id"]

        # Preferred label for the concept.
        pref = _norm_label(c.get("pref_label") or "")
        if pref:
            m[pref] = cid

        # Alternative labels are added only if the label is not already mapped.
        for alt in c.get("alt_labels") or []:
            alt_norm = _norm_label(alt or "")
            if alt_norm:
                m.setdefault(alt_norm, cid)

    return m


def resource_gold_concepts_by_category(
    resource: dict,
    label_to_cid: Dict[str, str],
) -> Dict[str, Set[str]]:
    """
    Convert a resource's manual taxonomy labels into ontology concept IDs.
    """
    gold_by_cat: Dict[str, Set[str]] = {
        "individual_characteristics": set(),
        "career_pathway": set(),
        "research_funding_process": set(),
        "organisational_culture": set(),
    }

    for k in TAXONOMY_KEYS:
        raw_vals = resource.get(k)

        # Taxonomy values may be missing, a list, or a single string.
        if raw_vals is None:
            vals: List[str] = []
        elif isinstance(raw_vals, list):
            vals = [str(v) for v in raw_vals if v is not None]
        else:
            vals = [str(raw_vals)]

        # Split multi-label cells and map each label to an ontology concept ID.
        for v in vals:
            for atom in _split_multi_label(v):
                key = _norm_label(atom)
                if not key:
                    continue

                cid = label_to_cid.get(key)
                if cid:
                    cat = _concept_category(cid)
                    if cat in gold_by_cat:
                        gold_by_cat[cat].add(cid)

    return gold_by_cat


def preds_by_category(pred_items: List[dict]) -> Dict[str, Set[str]]:
    """
    Group predicted concept IDs by top-level ontology category.
    The input is the top_k prediction list produced by semantic tagging.
    """
    pred_by_cat: Dict[str, Set[str]] = {
        "individual_characteristics": set(),
        "career_pathway": set(),
        "research_funding_process": set(),
        "organisational_culture": set(),
    }

    for it in pred_items or []:
        cid = it.get("concept_id")
        if not cid:
            continue

        cat = _concept_category(cid)
        if cat in pred_by_cat:
            pred_by_cat[cat].add(cid)

    return pred_by_cat


# Metric helpers
def precision_recall_f1(pred: Set[str], gold: Set[str]) -> Dict[str, float]:
    """Calculate set-based precision, recall, and F1 for concept IDs."""
    if not pred and not gold:
        # If both are empty, the prediction is treated as correct for that case.
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positive = len(pred & gold)
    precision = true_positive / max(1, len(pred))
    recall = true_positive / max(1, len(gold))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def _macro_avg(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Average precision, recall, and F1 across resources or categories."""
    if not metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "precision": sum(m["precision"] for m in metrics) / len(metrics),
        "recall": sum(m["recall"] for m in metrics) / len(metrics),
        "f1": sum(m["f1"] for m in metrics) / len(metrics),
    }


# Main evaluation function
def evaluate_semantic_tagging(
    resources: List[dict],
    ontology: dict,
    semantic_tags: List[dict],
    out_path,
) -> Dict:
    """
    Evaluate predicted semantic tags against manual taxonomy labels.

    Gold labels are taken from resources_unified.jsonl taxonomy columns and
    mapped to ontology concept IDs. Predicted labels are the semantic tagging
    outputs. The function writes both summary metrics and per-resource results.
    """
    # Map ontology labels to concept IDs so spreadsheet labels can be compared
    # with predicted semantic tag concept IDs.
    label_to_cid = build_label_to_concept_id(ontology)

    # Convert semantic tag rows into a resource_id -> predictions lookup.
    tags_by_rid = {str(r["rid"]).strip(): r.get("top_k", []) for r in semantic_tags}

    # Accumulators for overall macro averages.
    overall_metrics: List[Dict[str, float]] = []

    # Accumulators for category-level macro averages.
    per_cat_metrics: Dict[str, List[Dict[str, float]]] = {
        "individual_characteristics": [],
        "career_pathway": [],
        "research_funding_process": [],
        "organisational_culture": [],
    }

    per_resource = {}

    for idx, r in enumerate(resources):
        rid_raw = r.get("ID")
        if rid_raw is None:
            raise KeyError(
                f"Missing 'ID' field in resources_unified.jsonl at row index {idx}. "
                f"Available keys: {list(r.keys())}"
            )

        rid = str(rid_raw).strip()

        # Build gold and predicted concept sets for the current resource.
        gold_by_cat = resource_gold_concepts_by_category(r, label_to_cid)
        pred_items = tags_by_rid.get(rid, [])
        pred_by_cat = preds_by_category(pred_items)

        # Overall sets are the union of all four category-specific sets.
        gold_all: Set[str] = set().union(*gold_by_cat.values())
        pred_all: Set[str] = set().union(*pred_by_cat.values())

        # Overall resource-level metrics.
        m_all = precision_recall_f1(pred_all, gold_all)
        overall_metrics.append(m_all)

        # Category-level metrics for this resource.
        per_cat_row = {}
        for cat in per_cat_metrics.keys():
            m_cat = precision_recall_f1(pred_by_cat[cat], gold_by_cat[cat])
            per_cat_metrics[cat].append(m_cat)
            per_cat_row[cat] = {
                "precision": m_cat["precision"],
                "recall": m_cat["recall"],
                "f1": m_cat["f1"],
                "gold_count": len(gold_by_cat[cat]),
                "pred_count": len(pred_by_cat[cat]),
            }

        # Store detailed breakdown for later inspection.
        per_resource[rid] = {
            "overall": {
                "precision": m_all["precision"],
                "recall": m_all["recall"],
                "f1": m_all["f1"],
                "gold_count": len(gold_all),
                "pred_count": len(pred_all),
            },
            "by_category": per_cat_row,
        }

    # Summary report used in dissertation evaluation/analysis.
    report = {
        "resource_count": len(resources),
        "macro_avg": _macro_avg(overall_metrics),
        "macro_avg_by_category": {
            cat: _macro_avg(rows) for cat, rows in per_cat_metrics.items()
        },
        "note": (
            "Gold labels derived from resources_unified taxonomy columns; "
            "mapped to concept_ids via ontology pref/alt labels "
            "(case-insensitive, whitespace-normalised, and split on commas/semicolons)."
        ),
    }

    write_json(out_path, {"report": report, "per_resource": per_resource})
    return report