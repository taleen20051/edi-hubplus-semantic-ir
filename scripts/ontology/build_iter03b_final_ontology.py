#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

SRC = Path("ontology/versions/v2_iter03/edi_ontology_v2_iter03.json")
OUT = Path("ontology/versions/v2_iter03b/edi_ontology_v2_iter03b.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        k = norm(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out

# Guardrail: remove super-generic single words unless explicitly whitelisted
GENERIC_SINGLE_WORDS = {
    "policy","review","support","guidance","training","inclusion","diversity",
    "workplace","staff","university","promotion","retention","mentoring",
    "sponsorship","recruitment","leadership","networking","process"
}

SAFE_SINGLE_WORDS = {
    # domain-specific enough to keep
    "ageism","neurodivergent","microaggressions","transgender","allyship"
}

def prune_alt_labels(labels: list[str]) -> list[str]:
    kept = []
    for lab in labels:
        t = norm(lab)
        if not t:
            continue
        # single token?
        if " " not in t:
            if t in SAFE_SINGLE_WORDS:
                kept.append(lab)
            elif t in GENERIC_SINGLE_WORDS:
                continue
            else:
                # keep only if it contains + or is an acronym-ish pattern
                if "+" in t or re.fullmatch(r"[A-Z0-9]{2,}", lab.strip()):
                    kept.append(lab)
                else:
                    continue
        else:
            kept.append(lab)
    return dedupe_keep_order(kept)

# Supervisor-aligned structural renames (pref_label changes)
PREF_RENAMES_BY_ID: dict[str, str] = {
    # Individual Characteristics
    "individual_characteristics__race": "Race and Ethnicity",
    "individual_characteristics__religion_or_belief": "Religion and Belief",
    "individual_characteristics__socio_economic_background": "Socio-Economic Disadvantage",
    # Career Pathway (make wording more like supervisor)
    "career_pathway__recruitment_strategies": "Recruitment",
    "career_pathway__career_retention": "Retention",
    "career_pathway__mentorship_sponsorship": "Mentoring",
    "career_pathway__leadership_development": "Leadership",
    "career_pathway__networking_collaboration": "Networking and Collaboration",
    # Organisational Culture
    "organisational_culture__bullying_and_microagression_prevention": "Harassment, Bullying and Microagression",
    "organisational_culture__policy_review_reform": "Strategy or Policy",
    # Research Funding Process
    "research_funding_process__advertising_approach": "Advertising",
    "research_funding_process__application_processes": "Application Process",
    "research_funding_process__peer_review": "Assessment Process",
    "research_funding_process__application_support": "Funding Guidance",
    "research_funding_process__transparency_and_accessiblity_of_process": "University Selection",
}

# Relatable, phrase-based alt-label expansions (big but still precision-leaning)
ALT_ADDITIONS_BY_ID: dict[str, list[str]] = {
    # Individual characteristics
    "individual_characteristics__age": [
        "age discrimination at work",
        "age-inclusive workplace",
        "ageism awareness training",
        "older workers inclusion"
    ],
    "individual_characteristics__disability": [
        "reasonable adjustments at work",
        "disability adjustments policy",
        "workplace accessibility adjustments",
        "adjustments for disabled staff",
        "accessibility needs"
    ],
    "individual_characteristics__neurodiversity": [
        "support for autistic staff",
        "support for ADHD staff",
        "neurodiversity workplace adjustments",
        "neuroinclusion at work",
        "neurodiversity awareness training"
    ],
    "individual_characteristics__pregnancy_and_maternity": [
        "maternity leave policy",
        "parental leave policy",
        "return to work after maternity",
        "return-to-work support after maternity",
        "pregnancy discrimination at work",
        "family friendly return to work"
    ],
    "individual_characteristics__sexual_orientation": [
        "LGBTQ+ inclusion",
        "LGBTQIA+ inclusion",
        "sexual orientation inclusion",
        "queer inclusion at work"
    ],
    "individual_characteristics__trans_identity": [
        "gender reassignment support",
        "trans inclusion at work",
        "support during gender transition"
    ],
    "individual_characteristics__temporary_impairment": [
        "temporary impairment adjustments",
        "short-term condition workplace support"
    ],

    # Career pathway
    "career_pathway__career_progression": [
        "equitable promotion criteria",
        "fair promotion processes",
        "transparent promotion criteria",
        "promotion equity in academia"
    ],
    "career_pathway__recruitment_strategies": [
        "inclusive recruitment practices",
        "inclusive hiring practices",
        "bias-free hiring process",
        "fair recruitment process"
    ],
    "career_pathway__career_retention": [
        "staff retention",
        "retaining diverse talent",
        "academic staff retention",
        "retention strategy"
    ],
    "career_pathway__mentorship_sponsorship": [
        "mentoring programmes",
        "mentoring scheme",
        "sponsorship programmes",
        "career sponsorship schemes"
    ],
    "career_pathway__leadership_development": [
        "inclusive leadership training",
        "leadership pipeline development",
        "leadership development programme"
    ],

    # Organisational culture
    "organisational_culture__inclusive_language": [
        "inclusive language policy",
        "inclusive communications guidance",
        "gender-inclusive language guidance"
    ],
    "organisational_culture__inclusive_leadership": [
        "inclusive leadership behaviours",
        "inclusive management training"
    ],
    "organisational_culture__harassment": [
        "anti-harassment policy",
        "harassment reporting procedure",
        "sexual harassment prevention"
    ],
    "organisational_culture__bullying_and_microagression_prevention": [
        "microaggressions at work",
        "anti-bullying policy",
        "bullying reporting procedure",
        "civility at work policy"
    ],
    "organisational_culture__work_life_balance": [
        "flexible working policy",
        "hybrid working policy",
        "work-life balance support"
    ],
    "organisational_culture__data_collection_monitoring": [
        "diversity data monitoring",
        "EDI metrics monitoring",
        "workforce diversity metrics"
    ],
    "organisational_culture__policy_review_reform": [
        "inclusive policy reform",
        "institutional policy reform",
        "strategy and policy review"
    ],

    # Research funding process
    "research_funding_process__panel_diversity": [
        "review panel diversity",
        "committee diversity",
        "diverse funding panels"
    ],
    "research_funding_process__peer_review": [
        "equitable assessment process",
        "reviewer bias mitigation",
        "peer review bias",
        "fair assessment process"
    ],
    "research_funding_process__application_support": [
        "grant application guidance",
        "grant writing guidance",
        "proposal writing support",
        "funding application guidance"
    ],
    "research_funding_process__eligiblity_criteria": [
        "inclusive eligibility criteria",
        "eligibility requirements"
    ],
    "research_funding_process__transparency_and_accessiblity_of_process": [
        "transparent funding process",
        "accessible funding process",
        "transparency of funding process"
    ],
}

def main():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    out = deepcopy(src)

    out["version"] = "v2_iter03b"
    out["created_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    out["description"] = (
        (out.get("description") or "").strip()
        + "\n\nFinal iteration v2_iter03b: supervisor-aligned taxonomy naming + expanded user-facing alt_labels "
          "(precision-leaning phrases; generic single-word triggers pruned)."
    )

    # If you want to be explicit that this version uses extra (curated) synonyms:
    if "provenance" in out and "design_policy" in out["provenance"]:
        out["provenance"]["design_policy"]["alt_labels_from_observed_variants_only"] = False
        out["provenance"]["design_policy"]["curated_user_facing_synonyms"] = True

    # Index concepts by id
    by_id = {c["id"]: c for c in out.get("concepts", [])}

    # Apply pref_label renames
    renamed = []
    for cid, new_pref in PREF_RENAMES_BY_ID.items():
        if cid in by_id:
            old = by_id[cid].get("pref_label")
            by_id[cid]["pref_label"] = new_pref
            renamed.append((cid, old, new_pref))

    # Expand alt_labels + prune noise
    expanded = 0
    for cid, adds in ALT_ADDITIONS_BY_ID.items():
        if cid not in by_id:
            continue
        current = by_id[cid].get("alt_labels") or []
        merged = current + adds
        merged = dedupe_keep_order(merged)
        merged = prune_alt_labels(merged)
        by_id[cid]["alt_labels"] = merged
        expanded += 1

    # Global prune across all concepts to keep final clean
    for c in out.get("concepts", []):
        c["alt_labels"] = prune_alt_labels(c.get("alt_labels") or [])

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total_alt = sum(len(c.get("alt_labels") or []) for c in out.get("concepts", []))
    nonempty = sum(1 for c in out.get("concepts", []) if c.get("alt_labels"))
    print(f"Wrote: {OUT}")
    print(f"Renamed pref_labels: {len(renamed)}")
    print(f"Expanded concepts (added/pruned alt_labels): {expanded}")
    print(f"Concepts: {len(out.get('concepts', []))}")
    print(f"Concepts with alt_labels: {nonempty}")
    print(f"Total alt_labels: {total_alt}")

if __name__ == "__main__":
    main()