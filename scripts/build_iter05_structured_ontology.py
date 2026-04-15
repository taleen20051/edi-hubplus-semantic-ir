#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


OUT_DIR = Path("ontology/versions/v3_iter05_structured")
OUT_JSON = OUT_DIR / "edi_ontology_v3_iter05_structured.json"
OUT_TTL = OUT_DIR / "edi_ontology_v3_iter05_structured.ttl"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def slug(s: str) -> str:
    s = s.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "&", "/", "+"}:
            out.append("_")
    text = "".join(out)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


SCHEME = {
    "id": "edi-hubplus-taxonomy-v3-iter05-structured",
    "uri": "https://example.org/edi-hubplus/v3_iter05_structured/scheme",
    "title": "EDI Hub+ Structured Taxonomy Ontology (v3_iter05_structured)",
    "description": (
        "Structured SKOS concept scheme for EDI Hub+ retrieval. "
        "This iteration introduces explicit category/subcategory hierarchy, "
        "preferred labels, alternative labels, and hidden search labels."
    ),
    "version": "v3_iter05_structured",
    "created_utc": now_utc_iso(),
}

# ----------------------------
# Structured ontology source
# ----------------------------
STRUCTURE = [
    {
        "category": "Individual Characteristics",
        "subcategories": [
            {
                "name": "Age and Life Stage",
                "concepts": [
                    {
                        "pref_label": "Age",
                        "alt_labels": [
                            "Age discrimination at work",
                            "Age-inclusive workplace",
                            "Age equality",
                            "Age diversity",
                            "Ageism",
                            "Older workers inclusion",
                        ],
                        "hidden_labels": ["older worker support"],
                    }
                ],
            },
            {
                "name": "Disability and Adjustments",
                "concepts": [
                    {
                        "pref_label": "Disability",
                        "alt_labels": [
                            "Reasonable adjustments",
                            "Reasonable adjustments at work",
                            "Workplace adjustments",
                            "Adjustments for disabled staff",
                            "Access needs",
                            "Accessibility needs",
                        ],
                        "hidden_labels": ["reasonable workplace adjustments"],
                    },
                    {
                        "pref_label": "Temporary Impairment",
                        "alt_labels": [
                            "Temporary impairment adjustments",
                        ],
                        "hidden_labels": ["short-term impairment", "temporary condition support"],
                    },
                ],
            },
            {
                "name": "Neurodiversity",
                "concepts": [
                    {
                        "pref_label": "Neurodiversity",
                        "alt_labels": [
                            "Neurodivergent",
                            "Support for autistic staff",
                            "Support for ADHD staff",
                            "Neurodiversity workplace adjustments",
                            "Neuroinclusion at work",
                        ],
                        "hidden_labels": ["autism support at work", "adhd support at work"],
                    }
                ],
            },
            {
                "name": "Pregnancy, Maternity and Parental Leave",
                "concepts": [
                    {
                        "pref_label": "Pregnancy and Maternity",
                        "alt_labels": [
                            "Return to work after maternity",
                            "Pregnancy discrimination",
                            "Maternity leave policy",
                            "Parental leave policy",
                        ],
                        "hidden_labels": ["maternity return to work", "parental leave support"],
                    }
                ],
            },
            {
                "name": "Religion and Belief",
                "concepts": [
                    {
                        "pref_label": "Religion or Belief",
                        "alt_labels": [
                            "Religious accommodation at work",
                        ],
                        "hidden_labels": ["faith accommodation", "belief accommodation workplace"],
                    }
                ],
            },
            {
                "name": "Sexual Orientation and Gender Identity",
                "concepts": [
                    {
                        "pref_label": "Sexual Orientation",
                        "alt_labels": [
                            "LGBTQ+",
                            "LGBTQIA+",
                            "Sexual orientation inclusion",
                        ],
                        "hidden_labels": ["lgbtq", "lgbt+", "queer inclusion"],
                    },
                    {
                        "pref_label": "Trans Identity",
                        "alt_labels": [
                            "Gender identity",
                            "Gender reassignment support",
                            "Transgender inclusion",
                        ],
                        "hidden_labels": ["trans inclusion", "gender transition support"],
                    },
                ],
            },
            {
                "name": "Socio-economic Background",
                "concepts": [
                    {
                        "pref_label": "Socio-Economic Background",
                        "alt_labels": [
                            "Socioeconomic background",
                            "Widening participation",
                            "First generation academics",
                            "Social mobility",
                            "Working-class background",
                        ],
                        "hidden_labels": ["first-gen academics"],
                    }
                ],
            },
            {
                "name": "Race and Ethnicity",
                "concepts": [
                    {
                        "pref_label": "Race",
                        "alt_labels": [
                            "Race and ethnicity",
                            "Racial equality",
                        ],
                        "hidden_labels": ["ethnic equality"],
                    }
                ],
            },
        ],
    },
    {
        "category": "Career Pathway",
        "subcategories": [
            {
                "name": "Progression and Promotion",
                "concepts": [
                    {
                        "pref_label": "Career Progression",
                        "alt_labels": [
                            "Promotion equity",
                            "Equitable promotion criteria",
                            "Transparent promotion criteria",
                            "Career progression in universities",
                        ],
                        "hidden_labels": ["promotion fairness"],
                    }
                ],
            },
            {
                "name": "Retention",
                "concepts": [
                    {
                        "pref_label": "Career Retention",
                        "alt_labels": [
                            "Staff retention",
                            "Academic staff retention",
                            "Retaining diverse talent",
                            "Attrition reduction",
                            "Retention strategy",
                        ],
                        "hidden_labels": ["retention strategies"],
                    }
                ],
            },
            {
                "name": "Mentorship and Sponsorship",
                "concepts": [
                    {
                        "pref_label": "Mentorship and Sponsorship",
                        "alt_labels": [
                            "Mentorship programmes",
                            "Career sponsorship schemes",
                            "Sponsorship programmes",
                        ],
                        "hidden_labels": ["mentoring programmes", "academic sponsorship"],
                    }
                ],
            },
            {
                "name": "Leadership Development",
                "concepts": [
                    {
                        "pref_label": "Leadership Development",
                        "alt_labels": [
                            "Inclusive leadership development",
                            "Leadership pipeline development",
                            "Leadership development programme",
                        ],
                        "hidden_labels": ["leadership training", "future leaders pipeline"],
                    }
                ],
            },
            {
                "name": "Recruitment",
                "concepts": [
                    {
                        "pref_label": "Recruitment Strategies",
                        "alt_labels": [
                            "Inclusive recruitment",
                            "Inclusive hiring practices",
                            "Bias-free hiring process",
                        ],
                        "hidden_labels": ["fair recruitment", "inclusive hiring"],
                    }
                ],
            },
        ],
    },
    {
        "category": "Research Funding Process",
        "subcategories": [
            {
                "name": "Bias Mitigation",
                "concepts": [
                    {
                        "pref_label": "Bias Mitigation",
                        "alt_labels": [
                            "Bias reduction strategies",
                            "Bias mitigation in funding",
                            "Grant application bias reduction",
                        ],
                        "hidden_labels": ["reduce bias in funding"],
                    }
                ],
            },
            {
                "name": "Peer Review",
                "concepts": [
                    {
                        "pref_label": "Peer Review",
                        "alt_labels": [
                            "Peer review bias",
                            "Reviewer bias",
                            "Equitable peer review",
                        ],
                        "hidden_labels": ["inclusive peer review", "fair peer review"],
                    }
                ],
            },
            {
                "name": "Panel Diversity",
                "concepts": [
                    {
                        "pref_label": "Panel Diversity",
                        "alt_labels": [
                            "Review panel diversity",
                            "Diverse funding panels",
                        ],
                        "hidden_labels": ["committee diversity"],
                    }
                ],
            },
            {
                "name": "Application Support",
                "concepts": [
                    {
                        "pref_label": "Application Support",
                        "alt_labels": [
                            "Grant writing guidance",
                            "Proposal writing support",
                        ],
                        "hidden_labels": ["funding application support"],
                    }
                ],
            },
            {
                "name": "Transparency and Accessibility",
                "concepts": [
                    {
                        "pref_label": "Transparency and Accessibility of Process",
                        "alt_labels": [
                            "Transparent funding allocation processes",
                            "Accessible funding process",
                        ],
                        "hidden_labels": ["transparent funding process"],
                    }
                ],
            },
        ],
    },
    {
        "category": "Organisational Culture",
        "subcategories": [
            {
                "name": "Respect, Bullying and Microaggressions",
                "concepts": [
                    {
                        "pref_label": "Bullying and Microaggression Prevention",
                        "alt_labels": [
                            "Microaggressions",
                            "Microaggressions at work",
                            "Bullying prevention",
                            "Civility at work",
                        ],
                        "hidden_labels": ["anti-bullying policy"],
                    }
                ],
            },
            {
                "name": "Inclusive Language",
                "concepts": [
                    {
                        "pref_label": "Inclusive Language",
                        "alt_labels": [
                            "Inclusive language policy",
                            "Inclusive terminology",
                            "Gender-inclusive language",
                        ],
                        "hidden_labels": ["inclusive wording"],
                    }
                ],
            },
            {
                "name": "Work-Life Balance and Flexibility",
                "concepts": [
                    {
                        "pref_label": "Work-Life Balance",
                        "alt_labels": [
                            "Work life balance academic staff support",
                            "Flexible working policy",
                            "Hybrid working policy",
                        ],
                        "hidden_labels": ["family friendly policies"],
                    }
                ],
            },
            {
                "name": "Policy Review and Reform",
                "concepts": [
                    {
                        "pref_label": "Policy Review and Reform",
                        "alt_labels": [
                            "Inclusive policy reform",
                            "Institutional change",
                            "Strategy and policy review",
                        ],
                        "hidden_labels": ["policy reform"],
                    }
                ],
            },
            {
                "name": "Data Collection and Monitoring",
                "concepts": [
                    {
                        "pref_label": "Data Collection and Monitoring",
                        "alt_labels": [
                            "Data monitoring diversity metrics",
                            "Workforce diversity metrics",
                            "EDI metrics monitoring",
                        ],
                        "hidden_labels": ["diversity data monitoring"],
                    }
                ],
            },
            {
                "name": "Inclusive Leadership",
                "concepts": [
                    {
                        "pref_label": "Inclusive Leadership",
                        "alt_labels": [
                            "Inclusive management",
                            "Inclusive leadership behaviours",
                        ],
                        "hidden_labels": ["inclusive management training"],
                    }
                ],
            },
        ],
    },
]


def make_runtime_json() -> Dict:
    categories: List[Dict] = []
    subcategories: List[Dict] = []
    concepts: List[Dict] = []

    for cat in STRUCTURE:
        cat_label = cat["category"]
        cat_id = f"cat__{slug(cat_label)}"
        categories.append(
            {
                "id": cat_id,
                "pref_label": cat_label,
                "scheme_id": SCHEME["id"],
            }
        )

        for sub in cat["subcategories"]:
            sub_label = sub["name"]
            sub_id = f"subcat__{slug(cat_label)}__{slug(sub_label)}"
            subcategories.append(
                {
                    "id": sub_id,
                    "pref_label": sub_label,
                    "category_id": cat_id,
                    "broader_id": cat_id,
                    "scheme_id": SCHEME["id"],
                }
            )

            for c in sub["concepts"]:
                pref = c["pref_label"]
                concept_id = f"concept__{slug(cat_label)}__{slug(pref)}"
                concepts.append(
                    {
                        "id": concept_id,
                        "pref_label": pref,
                        "alt_labels": c.get("alt_labels", []),
                        "hidden_labels": c.get("hidden_labels", []),
                        "category_id": cat_id,
                        "subcategory_id": sub_id,
                        "broader_id": sub_id,
                        "scheme_id": SCHEME["id"],
                    }
                )

    return {
        "ontology_id": SCHEME["id"],
        "version": SCHEME["version"],
        "created_utc": SCHEME["created_utc"],
        "description": SCHEME["description"],
        "scheme": SCHEME,
        "categories": categories,
        "subcategories": subcategories,
        "concepts": concepts,
        "provenance": {
            "base_source": "v2_iter04 ontology terms",
            "design_policy": {
                "model": "SKOS-inspired structured concept scheme",
                "preferred_labels": True,
                "alternative_labels": True,
                "hidden_labels": True,
                "explicit_hierarchy": True,
                "leaf_concepts_used_for_retrieval": True,
            },
        },
    }


def ttl_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def write_ttl(runtime: Dict) -> str:
    lines: List[str] = []
    lines.append('@prefix ex: <https://example.org/edi-hubplus/v3_iter05_structured/> .')
    lines.append('@prefix skos: <http://www.w3.org/2004/02/skos/core#> .')
    lines.append('@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .')
    lines.append('@prefix owl: <http://www.w3.org/2002/07/owl#> .')
    lines.append('@prefix dcterms: <http://purl.org/dc/terms/> .')
    lines.append("")

    lines.append("ex:scheme a skos:ConceptScheme, owl:Ontology ;")
    lines.append(f'  dcterms:title "{ttl_escape(runtime["scheme"]["title"])}"@en ;')
    lines.append(f'  dcterms:description "{ttl_escape(runtime["scheme"]["description"])}"@en ;')
    lines.append(f'  dcterms:created "{runtime["created_utc"]}" ;')
    top_concepts = [f"ex:{cat['id']}" for cat in runtime["categories"]]
    if top_concepts:
        lines.append("  skos:hasTopConcept " + ", ".join(top_concepts) + " ;")
    lines[-1] = lines[-1].rstrip(" ;")
    lines.append(".")
    lines.append("")

    for cat in runtime["categories"]:
        lines.append(f"ex:{cat['id']} a skos:Concept ;")
        lines.append(f'  skos:prefLabel "{ttl_escape(cat["pref_label"])}"@en ;')
        lines.append("  skos:topConceptOf ex:scheme ;")
        lines.append("  skos:inScheme ex:scheme .")
        lines.append("")

    for sub in runtime["subcategories"]:
        lines.append(f"ex:{sub['id']} a skos:Concept ;")
        lines.append(f'  skos:prefLabel "{ttl_escape(sub["pref_label"])}"@en ;')
        lines.append(f"  skos:broader ex:{sub['broader_id']} ;")
        lines.append("  skos:inScheme ex:scheme .")
        lines.append("")

    for c in runtime["concepts"]:
        lines.append(f"ex:{c['id']} a skos:Concept ;")
        lines.append(f'  skos:prefLabel "{ttl_escape(c["pref_label"])}"@en ;')
        for alt in c.get("alt_labels", []):
            lines.append(f'  skos:altLabel "{ttl_escape(alt)}"@en ;')
        for hidden in c.get("hidden_labels", []):
            lines.append(f'  skos:hiddenLabel "{ttl_escape(hidden)}"@en ;')
        lines.append(f"  skos:broader ex:{c['broader_id']} ;")
        lines.append("  skos:inScheme ex:scheme .")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = make_runtime_json()
    OUT_JSON.write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_TTL.write_text(write_ttl(runtime), encoding="utf-8")

    print(f"Wrote JSON: {OUT_JSON}")
    print(f"Wrote TTL:  {OUT_TTL}")


if __name__ == "__main__":
    main()