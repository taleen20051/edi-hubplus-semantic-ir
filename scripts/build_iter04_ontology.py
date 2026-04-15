#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple


# -----------------------------
# Controlled synonym policy (Iter04)
# -----------------------------
GENERIC_SINGLE_WORD_BLOCKLIST = {
    "policy", "review", "support", "guidance", "training",
    "inclusion", "diversity", "workplace", "staff", "university",
    "process", "approach", "strategy", "strategies", "scheme", "schemes",
}

# Single-token allowlist (only if unambiguous & EDI-specific)
EDI_SINGLE_WORD_ALLOWLIST = {
    "ageism", "microaggressions", "neurodivergent", "transgender",
    "lgbtq+", "lgbtqia+",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalise_label(s: str) -> str:
    return " ".join((s or "").strip().split())


def is_single_token(s: str) -> bool:
    return len(s.split()) == 1


def should_keep_label(label: str) -> bool:
    """
    Iter04 rules:
    - Prefer multi-word phrases (precision-leaning)
    - Block generic single words
    - Allow tiny set of EDI-specific single tokens (ageism, LGBTQ+, etc.)
    """
    lab = normalise_label(label)
    if not lab:
        return False

    if is_single_token(lab):
        low = lab.lower()
        if low in GENERIC_SINGLE_WORD_BLOCKLIST:
            return False
        if low in EDI_SINGLE_WORD_ALLOWLIST:
            return True
        return False  # default: reject unknown single tokens

    return True  # multi-word phrases OK


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter04_alt_label_plan() -> Dict[str, List[str]]:
    """
    concept_id -> proposed alt_labels

    Controlled design:
    - Keep v1 structure (same concept IDs)
    - Do NOT rename pref_labels
    - Only add alt_labels that reflect likely user phrasing in your 32-query set
    - Avoid generic single word triggers
    """
    return {
        # --- Individual Characteristics ---
        "individual_characteristics__age": [
            "age discrimination at work",
            "age-inclusive workplace",
            "age equality",
            "age diversity",
            "ageism",
            "older workers inclusion",
        ],
        "individual_characteristics__disability": [
            "reasonable adjustments",
            "reasonable adjustments at work",
            "workplace adjustments",
            "adjustments for disabled staff",
            "access needs",
            "accessibility needs",
        ],
        "individual_characteristics__neurodiversity": [
            "neurodivergent",
            "support for autistic staff",
            "support for ADHD staff",
            "neurodiversity workplace adjustments",
            "neuroinclusion at work",
        ],
        "individual_characteristics__pregnancy_and_maternity": [
            "return to work after maternity",
            "return-to-work support after maternity",
            "pregnancy discrimination",
            "pregnancy discrimination at work",
            "maternity leave policy",
            "parental leave policy",
        ],
        "individual_characteristics__religion_or_belief": [
            "religion and belief",
            "religious accommodation at work",
            "faith accommodation in workplace",
        ],
        "individual_characteristics__sexual_orientation": [
            "LGBTQ+",
            "LGBTQIA+",
            "LGBTQ+ inclusion",
            "sexual orientation inclusion",
            "queer inclusion at work",
        ],
        "individual_characteristics__trans_identity": [
            "gender identity",
            "gender transition support",
            "support during gender transition",
            "trans inclusion at work",
            "gender reassignment support",
            "transgender inclusion",
        ],
        "individual_characteristics__socio_economic_background": [
            "socioeconomic background",
            "socio-economic disadvantage",
            "widening participation",
            "first generation academics",
            "social mobility",
            "working-class background",
        ],
        "individual_characteristics__temporary_impairment": [
            "temporary impairment adjustments",
            "short-term condition workplace adjustments",
        ],
        "individual_characteristics__race": [
            "race and ethnicity",
            "ethnic minority inclusion",
            "racial equality",
        ],

        # --- Career Pathway ---
        "career_pathway__mentorship_sponsorship": [
            "mentorship programmes",
            "mentoring programmes",
            "career mentoring",
            "career sponsorship schemes",
            "sponsorship programmes",
        ],
        "career_pathway__career_progression": [
            "promotion equity",
            "fair promotion processes",
            "equitable promotion criteria",
            "transparent promotion criteria",
            "career progression in universities",
        ],
        "career_pathway__career_retention": [
            "staff retention",
            "academic staff retention",
            "retaining diverse talent",
            "attrition reduction",
            "retention strategy",
        ],
        "career_pathway__leadership_development": [
            "inclusive leadership development",
            "leadership pipeline development",
            "leadership development programme",
            "inclusive leadership training",
        ],
        "career_pathway__recruitment_strategies": [
            "inclusive recruitment",
            "inclusive hiring practices",
            "bias-free hiring process",
            "fair recruitment process",
            "diverse recruitment",
        ],

        # --- Research Funding Process ---
        "research_funding_process__bias_mitigation": [
            "bias reduction strategies",
            "bias mitigation in funding",
            "grant application bias reduction",
        ],
        "research_funding_process__peer_review": [
            "peer review bias",
            "reviewer bias",
            "equitable peer review",
            "fair peer review",
            "inclusive peer review training",
        ],
        "research_funding_process__panel_diversity": [
            "review panel diversity",
            "committee diversity",
            "diverse funding panels",
            "panel composition",
        ],
        "research_funding_process__application_support": [
            "grant writing guidance",
            "grant application guidance",
            "proposal writing support",
            "funding application guidance",
        ],
        "research_funding_process__transparency_and_accessiblity_of_process": [
            "transparent funding allocation processes",
            "transparency of funding process",
            "transparent funding process",
            "accessible funding process",
        ],

        # --- Organisational Culture ---
        "organisational_culture__inclusive_language": [
            "inclusive language policy",
            "inclusive communications guidance",
            "inclusive wording",
            "inclusive terminology",
            "gender-inclusive language",
        ],
        "organisational_culture__work_life_balance": [
            "work life balance academic staff support",
            "flexible working policy",
            "hybrid working policy",
            "family friendly policies",
        ],
        "organisational_culture__bullying_and_microagression_prevention": [
            "microaggressions",
            "microaggressions at work",
            "bullying prevention",
            "anti-bullying policy",
            "civility at work",
        ],
        "organisational_culture__bias": [
            "implicit bias",
            "unconscious bias",
            "bias awareness",
        ],
        "organisational_culture__data_collection_monitoring": [
            "data monitoring diversity metrics",
            "workforce diversity metrics",
            "EDI metrics monitoring",
        ],
        "organisational_culture__policy_review_reform": [
            "inclusive policy reform",
            "institutional change",
            "strategy and policy review",
        ],
        "organisational_culture__inclusive_leadership": [
            "inclusive management",
            "inclusive leadership behaviours",
            "inclusive management training",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ontology", required=True, help="Path to v1 ontology JSON")
    ap.add_argument("--out_ontology", required=True, help="Path to write v2_iter04 ontology JSON")
    ap.add_argument("--out_candidate_csv", required=True, help="Path to write candidate table CSV")
    ap.add_argument("--out_change_json", required=True, help="Path to write change report JSON")
    ap.add_argument("--version", default="v2_iter04")
    args = ap.parse_args()

    in_path = Path(args.in_ontology)
    out_path = Path(args.out_ontology)
    cand_csv = Path(args.out_candidate_csv)
    change_json = Path(args.out_change_json)

    onto = load_json(in_path)
    if "concepts" not in onto or "categories" not in onto:
        raise SystemExit(f"Input ontology missing required fields: {in_path}")

    plan = iter04_alt_label_plan()

    concepts = onto["concepts"]
    by_id = {c["id"]: c for c in concepts}

    # --- Stats before ---
    v1_total_alts = sum(len(c.get("alt_labels", []) or []) for c in concepts)
    v1_concepts_with_alts = sum(1 for c in concepts if (c.get("alt_labels") or []))

    # --- Apply plan + build reports ---
    candidate_rows = []
    added_by_concept: Dict[str, List[str]] = {}
    rejected_by_concept: Dict[str, List[str]] = {}

    n_concepts_changed = 0
    n_added_total = 0

    for cid, proposed in plan.items():
        c = by_id.get(cid)
        if c is None:
            raise SystemExit(f"Plan refers to missing concept id in v1 (iter04 must not add concepts): {cid}")

        v1_alts_raw = c.get("alt_labels", []) or []
        v1_alts = [normalise_label(x) for x in v1_alts_raw if normalise_label(x)]
        v1_set: Set[str] = set(v1_alts)

        kept, rejected = [], []
        for lab in proposed:
            labn = normalise_label(lab)
            if not should_keep_label(labn):
                rejected.append(labn)
                continue
            kept.append(labn)

        # de-dup vs v1
        to_add = [x for x in kept if x and x not in v1_set]

        if to_add:
            n_concepts_changed += 1
            n_added_total += len(to_add)
            c["alt_labels"] = v1_alts + to_add
            added_by_concept[cid] = to_add

        if rejected:
            rejected_by_concept[cid] = rejected

        candidate_rows.append({
            "concept_id": cid,
            "category_id": c.get("category_id", ""),
            "pref_label": c.get("pref_label", ""),
            "alt_labels_v1": "; ".join(v1_alts),
            "proposed_alt_labels_v2_iter04": "; ".join(kept),
            "added_alt_labels_v2_iter04": "; ".join(to_add),
        })

    # --- Update metadata ---
    onto["version"] = args.version
    onto["created_utc"] = now_utc_iso()

    base_desc = onto.get("description") or ""
    if "v2_iter04" not in base_desc:
        onto["description"] = (
            base_desc.rstrip()
            + "\n\n"
            + "Ontology v2_iter04: controlled synonym enrichment derived from v1 (same concepts and pref_labels). "
              "Only precision-leaning alt_labels were added to improve user-query coverage while avoiding generic triggers."
        )

    prov = onto.get("provenance", {})
    pol = prov.get("design_policy", {})
    pol["alt_labels_from_observed_variants_only"] = False
    pol["curated_user_facing_synonyms"] = True
    pol["no_inferred_hierarchy"] = True
    prov["design_policy"] = pol
    onto["provenance"] = prov

    # --- Stats after ---
    v2_total_alts = sum(len(c.get("alt_labels", []) or []) for c in concepts)
    v2_concepts_with_alts = sum(1 for c in concepts if (c.get("alt_labels") or []))

    change_report = {
        "iteration": args.version,
        "created_utc": onto["created_utc"],
        "input_ontology": str(in_path),
        "output_ontology": str(out_path),
        "policy": {
            "precision_leaning": True,
            "single_word_blocklist": sorted(GENERIC_SINGLE_WORD_BLOCKLIST),
            "single_word_allowlist": sorted(EDI_SINGLE_WORD_ALLOWLIST),
            "no_pref_label_renames": True,
            "no_new_concepts": True,
        },
        "stats_before": {
            "n_concepts_total": len(concepts),
            "n_concepts_with_alt_labels": v1_concepts_with_alts,
            "n_alt_labels_total": v1_total_alts,
        },
        "stats_after": {
            "n_concepts_total": len(concepts),
            "n_concepts_with_alt_labels": v2_concepts_with_alts,
            "n_alt_labels_total": v2_total_alts,
        },
        "delta": {
            "concepts_changed": n_concepts_changed,
            "alt_labels_added_total": n_added_total,
        },
        "added_alt_labels_by_concept": added_by_concept,
        "rejected_labels_by_concept": rejected_by_concept,
    }

    # --- Write outputs ---
    write_json(out_path, onto)
    write_json(change_json, change_report)

    cand_csv.parent.mkdir(parents=True, exist_ok=True)
    with cand_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "concept_id",
                "category_id",
                "pref_label",
                "alt_labels_v1",
                "proposed_alt_labels_v2_iter04",
                "added_alt_labels_v2_iter04",
            ],
        )
        w.writeheader()
        for r in candidate_rows:
            w.writerow(r)

    print(f"Wrote ontology: {out_path}")
    print(f"Wrote change report JSON: {change_json}")
    print(f"Wrote candidate table CSV: {cand_csv}")
    print(f"Concepts changed: {n_concepts_changed}")
    print(f"Alt labels added total: {n_added_total}")


if __name__ == "__main__":
    main()