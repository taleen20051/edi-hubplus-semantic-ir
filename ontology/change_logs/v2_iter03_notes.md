# Ontology Iteration Notes — v2_iter03 (Boundary Tightening)

- Source ontology: v2_iter01
- Target ontology: v2_iter03
- Change type: boundary tightening (alt_label pruning)

## Rules applied
- Removed single-word alt_labels unless strongly EDI-specific and unambiguous
- Removed generic alt_labels that could trigger irrelevant matches (e.g., broad terms like “promotion”, “retention”, “mentoring”, “sponsorship” when used alone)
- Retained multi-word phrases (e.g., “staff retention”, “gender identity”, “return to work after maternity”) for higher precision

## Expected effect
- Fewer false positives from lexical overmatching
- Improved precision and hybrid ranking stability
- Possible small recall reduction (trade-off)
