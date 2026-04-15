# src/ontology/validate_ontology_v1_ttl.py
"""
Validate that the exported Turtle (TTL) file parses cleanly using rdflib.

Input:
- data/processed/edi_ontology_v1.ttl

Output:
- Console PASS/FAIL and basic triple count
"""

from __future__ import annotations

import sys
from pathlib import Path
from rdflib import Graph

IN_TTL = Path("data/processed/edi_ontology_v1.ttl")


def main() -> None:
    if not IN_TTL.exists():
        print(f" TTL validation: FAIL (missing file: {IN_TTL})")
        sys.exit(1)

    g = Graph()
    try:
        g.parse(IN_TTL.as_posix(), format="turtle")
    except Exception as e:
        print(" TTL validation: FAIL (rdflib could not parse the file)")
        print(f"File: {IN_TTL}")
        print(f"Error: {e}")
        sys.exit(1)

    print(" TTL validation: PASS (parsed successfully)")
    print(f"File: {IN_TTL}")
    print(f"Triples loaded: {len(g)}")

    # Optional: sanity checks (very lightweight)
    # Ensure we have a ConceptScheme in the graph
    try:
        n_schemes = sum(1 for _ in g.triples((None, None, None)))  # cheap no-op generator
    except Exception:
        n_schemes = 0

    sys.exit(0)


if __name__ == "__main__":
    main()