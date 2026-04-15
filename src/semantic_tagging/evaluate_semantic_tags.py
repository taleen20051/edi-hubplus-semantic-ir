from __future__ import annotations

from typing import Dict, List, Set, Tuple
import re

from .io_utils import write_json


# IMPORTANT:
# Your resources_unified.jsonl uses these exact taxonomy column names (case + spaces):
# - "Individual Characteristics"
# - "Career Pathway"
# - "Research Funding Process"
# - "Organisational Culture"
#
# We evaluate by mapping these manual labels to concept_ids using ontology pref/alt labels,
# then comparing to the predicted concept_ids from semantic_tags_*.jsonl.
TAXONOMY_KEYS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]


# Helper: concept_id prefix -> category_id
# e.g. "organisational_culture__inclusive_language" -> "organisational_culture"
def _concept_category(concept_id: str) -> str:
    return (concept_id or "").split("__", 1)[0].strip()


def _norm_label(x: str) -> str:
    """Lowercase + strip + collapse whitespace for robust matching."""
    s = (x or "").strip().lower()
    s = " ".join(s.split())
    return s


# Many spreadsheet cells are exported as a list of strings, but one element can still contain:
# "Inclusive Language, Allyship, Cultural Competency"
# We split on commas and semicolons. (You can extend this if needed.)
_SPLIT_RE = re.compile(r"\s*[,;]\s*")


def _split_multi_label(s: str) -> List[str]:
    """
    Split a single string that may contain multiple labels separated by commas/semicolons.
    Returns list of atomic labels.
    """
    s = (s or "").strip()
    if not s:
        return []
    parts = _SPLIT_RE.split(s)
    return [p.strip() for p in parts if p and p.strip()]


def build_label_to_concept_id(ontology: dict) -> Dict[str, str]:
    """
    Build mapping: surface label (pref/alt) -> concept_id.
    Matching is case-insensitive and whitespace-normalised.
    """
    m: Dict[str, str] = {}
    for c in ontology.get("concepts", []):
        cid = c["id"]

        pref = _norm_label(c.get("pref_label") or "")
        if pref:
            m[pref] = cid

        for alt in c.get("alt_labels") or []:
            alt_norm = _norm_label(alt or "")
            if alt_norm:
                # keep first mapping if duplicates exist
                m.setdefault(alt_norm, cid)

    return m


def resource_gold_concepts_by_category(
    resource: dict,
    label_to_cid: Dict[str, str],
) -> Dict[str, Set[str]]:
    """
    Convert the resource's manual taxonomy labels into concept_ids,
    grouped by category_id.

    Returns:
      {
        "individual_characteristics": {...},
        "career_pathway": {...},
        "research_funding_process": {...},
        "organisational_culture": {...}
      }
    """
    gold_by_cat: Dict[str, Set[str]] = {
        "individual_characteristics": set(),
        "career_pathway": set(),
        "research_funding_process": set(),
        "organisational_culture": set(),
    }

    for k in TAXONOMY_KEYS:
        raw_vals = resource.get(k)

        # Defensive: in your export it's normally a list, but handle other cases.
        if raw_vals is None:
            vals: List[str] = []
        elif isinstance(raw_vals, list):
            vals = [str(v) for v in raw_vals if v is not None]
        else:
            vals = [str(raw_vals)]

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
    Take predicted items [{"concept_id": ..., "score": ...}, ...]
    and group concept_ids by category_id.
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


def precision_recall_f1(pred: Set[str], gold: Set[str]) -> Dict[str, float]:
    """
    Standard set-based precision/recall/F1 on concept_id sets.
    """
    if not pred and not gold:
        # No gold + no predictions => "perfect" for that case
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred & gold)
    p = tp / max(1, len(pred))
    r = tp / max(1, len(gold))
    f1 = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)
    return {"precision": p, "recall": r, "f1": f1}


def _macro_avg(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": sum(m["precision"] for m in metrics) / len(metrics),
        "recall": sum(m["recall"] for m in metrics) / len(metrics),
        "f1": sum(m["f1"] for m in metrics) / len(metrics),
    }


def evaluate_semantic_tagging(
    resources: List[dict],
    ontology: dict,
    semantic_tags: List[dict],
    out_path,
) -> Dict:
    """
    Evaluate predicted semantic tags (concept_ids) against gold manual tags
    stored in resources_unified.jsonl taxonomy columns.

    Outputs:
      - overall macro precision/recall/F1
      - per-category macro precision/recall/F1
      - per-resource breakdown (overall + per-category)
    """
    label_to_cid = build_label_to_concept_id(ontology)
    tags_by_rid = {str(r["rid"]).strip(): r.get("top_k", []) for r in semantic_tags}

    # Accumulators for macro averages
    overall_metrics: List[Dict[str, float]] = []

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

        gold_by_cat = resource_gold_concepts_by_category(r, label_to_cid)
        pred_items = tags_by_rid.get(rid, [])
        pred_by_cat = preds_by_category(pred_items)

        # Overall sets = union across categories
        gold_all: Set[str] = set().union(*gold_by_cat.values())
        pred_all: Set[str] = set().union(*pred_by_cat.values())

        m_all = precision_recall_f1(pred_all, gold_all)
        overall_metrics.append(m_all)

        # Per-category
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