# src/ontology/export_ontology_v1_to_ttl.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

IN_JSON = Path("ontology/edi_ontology_v1.json")  # <-- matches your VS Code file path
OUT_TTL = Path("data/processed/edi_ontology_v1.ttl")

# Choose a base IRI for your ontology namespace (doesn't need to resolve yet).
BASE = "https://example.org/edi-hubplus/ontology/v1#"

def esc(s: str) -> str:
    # Turtle string escape (minimal)
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

def main() -> None:
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))

    ontology_id = data.get("ontology_id", "edi-hubplus-taxonomy")
    version = data.get("version", "v1")
    created_utc = data.get("created_utc", "")
    description = data.get("description", "")

    categories = data.get("categories", [])
    concepts = data.get("concepts", [])

    lines = []

    # Prefixes
    lines += [
        "@prefix : <" + BASE + "> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "",
    ]

    # Concept scheme (the ontology as a SKOS scheme)
    scheme_iri = ":scheme"
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    lines += [
        f"{scheme_iri} a skos:ConceptScheme ;",
        f'  dcterms:identifier "{esc(ontology_id)}" ;',
        f'  dcterms:hasVersion "{esc(version)}" ;',
        f'  dcterms:created "{esc(created_utc or now_utc)}" ;',
        f'  dcterms:modified "{esc(now_utc)}" ;',
        f'  dcterms:description "{esc(description)}" .',
        "",
    ]

    # Categories as top concepts
    # We make each category a skos:Concept and link it to the scheme.
    for cat in categories:
        cat_id = cat["id"]
        cat_label = cat.get("label", cat_id)

        lines += [
            f":{cat_id} a skos:Concept ;",
            f'  skos:prefLabel "{esc(cat_label)}"@en ;',
            f"  skos:topConceptOf {scheme_iri} ;",
            f"  skos:inScheme {scheme_iri} .",
            "",
        ]

    # Concepts
    for c in concepts:
        cid = c["id"]
        pref = c.get("pref_label", "").strip()
        alts = c.get("alt_labels", []) or []
        cat_id = c.get("category_id", "").strip()
        source_col = c.get("source_column", "").strip()

        lines.append(f":{cid} a skos:Concept ;")
        if pref:
            lines.append(f'  skos:prefLabel "{esc(pref)}"@en ;')

        # alt labels (variants observed)
        for a in alts:
            a = str(a).strip()
            if a:
                lines.append(f'  skos:altLabel "{esc(a)}"@en ;')

        # link into scheme + category
        lines.append(f"  skos:inScheme {scheme_iri} ;")
        if cat_id:
            lines.append(f"  skos:broader :{cat_id} ;")

        # provenance-like note (optional but useful)
        if source_col:
            lines.append(f'  dcterms:source "{esc(source_col)}" ;')

        # end statement (replace last ; with .)
        if lines[-1].endswith(";"):
            lines[-1] = lines[-1][:-1] + " ."
        else:
            lines.append(".")

        lines.append("")

    OUT_TTL.parent.mkdir(parents=True, exist_ok=True)
    OUT_TTL.write_text("\n".join(lines), encoding="utf-8")

    print(" Export complete")
    print(f"Input:  {IN_JSON}")
    print(f"Output: {OUT_TTL}")

if __name__ == "__main__":
    main()