# Ontology Iteration Notes — v2_iter03b (Final: Supervisor-aligned + User-facing Synonyms)

- Source ontology: v2_iter03
- Target ontology: v2_iter03b
- Change type: (A) structural renaming + (B) curated synonym enrichment

## A) Structural renaming (taxonomy alignment)

Updated preferred labels to align with supervisor terminology, e.g.:

- Race → Race and Ethnicity
- Religion Or Belief → Religion and Belief
- Socio-Economic Background → Socio-Economic Disadvantage
- Recruitment Strategies → Recruitment
- Career Retention → Retention
- Mentorship & Sponsorship → Mentoring
- Leadership Development → Leadership
- Policy Review & Reform → Strategy or Policy
- Peer Review → Assessment Process
- Application Support → Funding Guidance

## B) Synonym enrichment (user-facing query coverage)

- Added relatable, phrase-based alternative labels (e.g., “inclusive language policy”, “reasonable adjustments at work”).
- Avoided generic single-word triggers where possible; preference given to multi-word phrases to reduce lexical overmatching.

## Expected effect

- Improved alignment between real user phrasing and ontology concept vocabulary.
- Improved retrieval robustness for ontology-driven and hybrid retrieval.
- Potential trade-off: broader vocabulary may increase matches; phrase-based guardrails used to manage precision.
