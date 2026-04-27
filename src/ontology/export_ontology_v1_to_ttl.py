"""
Export ontology v1 from JSON to Turtle (TTL) format.

The JSON ontology is the main format used by the Python retrieval pipeline.
This script creates a SKOS-style Turtle representation so the ontology can also
be inspected as a lightweight web artefact.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


# Input ontology JSON built from the cleaned taxonomy.
IN_JSON = Path("ontology/edi_ontology_v1.json")
# Output TTL file used as an RDF/SKOS-compatible ontology export.
OUT_TTL = Path("data/processed/edi_ontology_v1.ttl")

# Base IRI for the ontology namespace. It is used to create local concept IRIs.
BASE = "https://example.org/edi-hubplus/ontology/v1#"


def esc(s: str) -> str:
    """Escape text so it can be safely written inside Turtle string literals."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def main() -> None:
    """Read ontology JSON and write an equivalent SKOS-style Turtle file."""
    data = json.loads(IN_JSON.read_text(encoding="utf-8"))

    ontology_id = data.get("ontology_id", "edi-hubplus-taxonomy")
    version = data.get("version", "v1")
    created_utc = data.get("created_utc", "")
    description = data.get("description", "")

    categories = data.get("categories", [])
    concepts = data.get("concepts", [])

    lines = []

    # Standard RDF/SKOS prefixes used in the TTL export.
    lines += [
        "@prefix : <" + BASE + "> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "",
    ]

    # Represent the ontology as one SKOS concept scheme.
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

    # Export the four taxonomy dimensions as top-level SKOS concepts.
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

    # Export each ontology concept with preferred labels, alt labels, and category link.
    for c in concepts:
        cid = c["id"]
        pref = c.get("pref_label", "").strip()
        alts = c.get("alt_labels", []) or []
        cat_id = c.get("category_id", "").strip()
        source_col = c.get("source_column", "").strip()

        lines.append(f":{cid} a skos:Concept ;")
        if pref:
            lines.append(f'  skos:prefLabel "{esc(pref)}"@en ;')

        # Alternative labels preserve observed variants from the cleaning process.
        for a in alts:
            a = str(a).strip()
            if a:
                lines.append(f'  skos:altLabel "{esc(a)}"@en ;')

        # Link each concept to the scheme and, where available, its top-level category.
        lines.append(f"  skos:inScheme {scheme_iri} ;")
        if cat_id:
            lines.append(f"  skos:broader :{cat_id} ;")

        # Keep the source taxonomy column as lightweight provenance.
        if source_col:
            lines.append(f'  dcterms:source "{esc(source_col)}" ;')

        # End the Turtle statement by replacing the final semicolon with a full stop.
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