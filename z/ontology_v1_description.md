# Ontology v1 Description (Spreadsheet-Fidelity Baseline)

## Overview

Ontology v1 is the initial taxonomy ontology for the EDI Hub+ Semantic IR system. It was constructed to faithfully represent the taxonomy labels already used by EDI Hub+ contributors in the project spreadsheet. As a result, v1 is intentionally conservative: it preserves contributor vocabulary without introducing new interpretation, inferred structure, or externally sourced synonym expansion.

This version is treated as the immutable baseline ontology used for downstream retrieval baselines and for comparison against later ontology refinements (v2).

---

## Inputs and Provenance

Ontology v1 was generated directly from the cleaned taxonomy outputs produced in Phase 1:

- `data/processed/taxonomy_clean.csv`Cleaned taxonomy labels for each resource across the four taxonomy columns.
- `data/processed/normalisation_log.csv`
  An audit log recording normalisation decisions (e.g., casing changes, spacing fixes, minor corrections). This log is used to populate `alt_labels` with *observed* variants only.

The generated ontology artefact is stored as:

- `data/processed/edi_ontology_v1.json`
- `data/processed/edi_ontology_v1_stats.json`

---

## Scope

Ontology v1 includes concepts from exactly four top-level EDI dimensions, corresponding to the four taxonomy columns used in the spreadsheet:

1. Individual Characteristics
2. Career Pathway
3. Research Funding Process
4. Organisational Culture

No additional categories are introduced.

---

## Modelling Approach

Each distinct cleaned tag value from a taxonomy column is represented as a concept. Each concept includes:

- `id`: a stable identifier derived deterministically from the category and label
- `pref_label`: the canonical (cleaned) label string
- `alt_labels`: optional observed variants recorded during cleaning (e.g., casing variants)
- `category_id`: one of the four allowed category identifiers
- `source_column`: the original spreadsheet column name for traceability

Ontology v1 intentionally contains no relations beyond category membership (i.e., no broader/narrower hierarchies, equivalence mappings, or external links).

---

## Design Policy (Spreadsheet Fidelity)

Ontology v1 follows a strict spreadsheet-fidelity policy:

- **No inferred hierarchies** beyond the four main categories
- **No external synonym expansion** (e.g., no WordNet or embedding-driven synonym addition)
- **No semantic merging** of concepts (each distinct cleaned label becomes a separate concept)
- **Alternative labels are restricted to observed variants** captured in the normalisation log
- **Placeholder values are excluded**, such as “Not Specified”

This policy ensures that v1 functions as a neutral baseline reflecting current contributor practice.

---

## Statistics

Ontology v1 contains **71** concepts in total, distributed across the four categories as follows:

- Individual Characteristics: 16
- Career Pathway: 16
- Research Funding Process: 13
- Organisational Culture: 26

These totals are recorded in `data/processed/edi_ontology_v1_stats.json`.

---

## Validation

A structural validation script was executed to ensure ontology integrity before downstream use:

- `src/ontology/validate_ontology_v1.py`

The validation checks confirm:

- concept identifiers are unique
- every concept belongs to one of the four allowed categories
- preferred labels are non-empty
- placeholder labels (e.g., “Not Specified”) are absent
- alternative labels do not contain exact duplicates of the preferred label
- total and per-category concept counts match the expected statistics

A validation report is written to:

- `data/processed/edi_ontology_v1_validation.json`

---

## Intended Use in the System

Ontology v1 is used as:

1. **A baseline ontology** for ontology-only retrieval (Phase 3)
2. **The controlled concept inventory** for semantic tagging experiments (Phase 4–6)
3. **The reference point** for measuring the impact of ontology evolution in v2 (Phase 8)

---

## Known Limitations (Motivation for v2)

Because Ontology v1 is fidelity-based, it inherits limitations from inconsistent contributor vocabulary:

- concepts are flat (no hierarchy beyond category membership)
- synonyms not present in the spreadsheet are not captured
- minor spelling inconsistencies may still exist if present in the cleaned labels
- recall in ontology-only retrieval may be limited when users phrase queries differently from taxonomy labels

These limitations motivate Ontology v2, which introduces controlled refinement based on retrieval failures and human feedback.

---
