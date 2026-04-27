#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def load_ontology(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def concepts_to_rows(ont: dict) -> List[dict]:
    rows = []
    for c in ont.get("concepts", []):
        rows.append(
            {
                "category_id": c.get("category_id", ""),
                "source_column": c.get("source_column", ""),
                "concept_id": c.get("id", ""),
                "pref_label_v1": c.get("pref_label", ""),
                "alt_labels_v1": c.get("alt_labels", []) or [],
            }
        )
    return rows


def build_safe_iter04_additions() -> Dict[str, List[str]]:
    """
    Controlled alt_label additions for v2_iter04.

    Rules:
    - Prefer multi-word phrases (precision-leaning)
    - Allow a few highly EDI-specific acronyms (EDI/DEI/LGBTQ+) when clearly beneficial
    - Avoid generic triggers like: support, policy, review, guidance, training (alone), workplace (alone), staff (alone)
    - Avoid duplicating near-identical triggers across overlapping concepts
    """

    return {
        # -------------------------
        # Individual Characteristics
        # -------------------------
        "individual_characteristics__age": [
            "age discrimination",
            "ageism",
            "age inclusive",
            "older workers",
        ],
        "individual_characteristics__disability": [
            "reasonable adjustments",
            "workplace adjustments",
            "access needs",
            "disability inclusion",
        ],
        "individual_characteristics__neurodiversity": [
            "neurodivergent",
            "autism at work",
            "ADHD at work",
            "neurodiversity awareness",
        ],
        "individual_characteristics__pregnancy_and_maternity": [
            "maternity leave",
            "parental leave",
            "pregnancy discrimination",
            "return to work after maternity",
        ],
        "individual_characteristics__religion_or_belief": [
            "religious observance",
            "faith inclusion",
        ],
        "individual_characteristics__sexual_orientation": [
            "LGBTQ+",
            "LGBTQIA+",
            "sexual orientation inclusion",
        ],
        "individual_characteristics__socio_economic_background": [
            "social mobility",
            "working class background",
            "widening participation",
        ],
        "individual_characteristics__trans_identity": [
            "gender identity",
            "gender reassignment",
            "trans inclusion",
        ],

        # -------------------------
        # Career Pathway
        # -------------------------
        "career_pathway__career_progression": [
            "promotion criteria",
            "fair promotion",
            "promotion equity",
            "transparent promotion criteria",
        ],
        # Keep retention terms on Career Retention, keep strategy terms on Retention Strategies
        "career_pathway__career_retention": [
            "staff retention",
            "retaining talent",
            "reducing attrition",
        ],
        "career_pathway__leadership_development": [
            "leadership pipeline",
            "leadership development programme",
            "inclusive leadership development",
        ],
        "career_pathway__mentorship_sponsorship": [
            "mentoring scheme",
            "mentorship programme",
            "sponsorship programme",
        ],
        "career_pathway__recruitment_strategies": [
            "inclusive recruitment",
            "inclusive hiring",
            "bias free hiring",
            "fair recruitment process",
        ],
        "career_pathway__retention_strategies": [
            "retention strategy",
            "retention interventions",
        ],

        # -------------------------
        # Research Funding Process
        # -------------------------
        # Keep “application support” concept label; add high-signal phrases
        "research_funding_process__application_support": [
            "grant writing",
            "proposal writing",
            "application guidance",
            "funding application support",
        ],
        "research_funding_process__eligiblity_criteria": [
            "eligibility criteria",
            "eligibility requirements",
        ],
        # Peer Review: keep review terms here (not across multiple places)
        "research_funding_process__peer_review": [
            "peer review bias",
            "reviewer bias",
            "fair peer review",
        ],
        "research_funding_process__panel_diversity": [
            "diverse panels",
            "panel composition",
        ],
        "research_funding_process__transparency_and_accessiblity_of_process": [
            "transparent funding process",
            "accessible funding process",
            "process transparency",
        ],

        # -------------------------
        # Organisational Culture
        # -------------------------
        "organisational_culture__awareness_training": [
            "EDI training",
            "DEI training",
            "bias awareness training",
            "inclusion training",
        ],
        "organisational_culture__bias": [
            "implicit bias",
            "unconscious bias",
        ],
        "organisational_culture__bullying_and_microagression_prevention": [
            "microaggressions at work",
            "bullying prevention",
            "microaggression prevention",
        ],
        "organisational_culture__data_collection_monitoring": [
            "diversity data",
            "EDI metrics",
            "workforce diversity metrics",
        ],
        "organisational_culture__employee_resource_groups": [
            "staff networks",
            "employee networks",
            "affinity groups",
        ],
        "organisational_culture__harassment": [
            "workplace harassment",
            "sexual harassment",
            "harassment reporting",
        ],
        "organisational_culture__inclusive_language": [
            "gender inclusive language",
            "inclusive terminology",
            "inclusive communications",
        ],
        "organisational_culture__inclusive_leadership": [
            "inclusive management",
            "inclusive leadership behaviours",
        ],
        "organisational_culture__work_life_balance": [
            "flexible working",
            "hybrid working",
            "family friendly policies",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ontology", required=True, help="Path to v1 ontology JSON")
    ap.add_argument("--out_csv", required=True, help="Where to write the candidate CSV table")
    args = ap.parse_args()

    in_path = Path(args.in_ontology)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ont = load_ontology(in_path)
    rows = concepts_to_rows(ont)

    additions = build_safe_iter04_additions()

    # Precompute a set of all added labels to spot duplication risk
    added_norm_to_concepts: Dict[str, List[str]] = {}
    for cid, labels in additions.items():
        for lab in labels:
            key = _norm(lab)
            added_norm_to_concepts.setdefault(key, []).append(cid)

    # Build CSV rows
    out_rows = []
    for r in rows:
        cid = r["concept_id"]
        v1_alts = r["alt_labels_v1"]
        v1_norm = {_norm(x) for x in v1_alts}

        proposed = additions.get(cid, [])
        proposed_dedup = []
        for lab in proposed:
            if _norm(lab) in v1_norm:
                continue
            proposed_dedup.append(lab)

        # duplication risk: a label appears in >1 concept additions
        dup_risk_labels = [lab for lab in proposed_dedup if len(added_norm_to_concepts.get(_norm(lab), [])) > 1]

        out_rows.append(
            {
                "category_id": r["category_id"],
                "source_column": r["source_column"],
                "concept_id": cid,
                "pref_label_v1": r["pref_label_v1"],
                "alt_labels_v1": "; ".join(v1_alts),
                "proposed_alt_labels_v2_iter04": "; ".join(proposed_dedup),
                "duplication_risk_labels": "; ".join(dup_risk_labels),
                "curation_notes": "",
            }
        )

    fieldnames = [
        "category_id",
        "source_column",
        "concept_id",
        "pref_label_v1",
        "alt_labels_v1",
        "proposed_alt_labels_v2_iter04",
        "duplication_risk_labels",
        "curation_notes",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"Wrote candidate table: {out_csv}")
    print("Next: open the CSV, review proposed_alt_labels_v2_iter04, and edit curation_notes where needed.")


if __name__ == "__main__":
    main()