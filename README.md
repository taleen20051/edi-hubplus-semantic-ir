# EDI Hub+ Semantic IR

Ontology-driven semantic tagging and retrieval system for Equality, Diversity and Inclusion (EDI) resources.

This project builds a searchable EDI resource collection from the EDI Hub+ spreadsheet, validates and extracts text from linked resources, cleans taxonomy labels, constructs a unified dataset, generates semantic embeddings and tags, runs retrieval baselines and hybrid retrieval experiments, and provides a final Streamlit web interface for interactive search and browsing.

---

## Repository Overview

The project has five main stages:

1. Spreadsheet and taxonomy preparation
2. Link validation and text extraction
3. Unified dataset construction
4. Semantic tagging and retrieval experiments
5. Interactive Streamlit application

The final application uses the **final unified corpus of 64 valid resources**.

---

## Tested Environment

This project was developed and tested in a Python virtual environment on macOS.

Recommended setup:

- Python 3.11+
- pip
- virtual environment (venv)

---

## Clone the Repository

```bash
git clone https://github.com/taleen20051/edi-hubplus-semantic-ir.git
cd edi-hubplus-semantic-ir
```

If downloaded as a ZIP file, extract it and open the project folder in Terminal.

---

## Create and Activate Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

If any package is missing:

```bash
pip install openpyxl trafilatura streamlit watchdog
```

### Dependency Notes

- `openpyxl` = Excel reading
- `trafilatura` = HTML extraction
- `streamlit` = Web app
- `watchdog` = optional Streamlit speed improvement

---

## Important Notes Before Running

### Semantic Tagging Scripts Must Be Run As Modules

Correct:

```bash
python -m src.semantic_tagging.build_concept_index
```

Incorrect:

```bash
python src/semantic_tagging/build_concept_index.py
```

### First Embedding Run

The first run may download Hugging Face models:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`

This is normal.

### Harmless Warnings

You may see:

- openpyxl warnings
- `UNEXPECTED embeddings.position_ids`
- `datetime.utcnow()` warnings

These do not prevent execution.

---

# FULL PIPELINE

---

## Input File Required

```text
data/raw/Resource centre taxonomy and resources_wd_macros.xlsm
```

---

# STEP 1 — Extract Spreadsheet To CSV

**Script:** `src/extraction/extract_sheet5_to_csv.py`

### Purpose

- Reads workbook
- Extracts sheet `5_Resources_full_data`
- Saves CSV

### Run

```bash
python src/extraction/extract_sheet5_to_csv.py
```

### Output

```text
data/raw/resources_full_data_full.csv
```

### Example Result

- Rows exported: 104

---

# STEP 2 — Filter Included Resources

**Script:** `src/extraction/filter_included_resources.py`

### Purpose

- Removes excluded rows
- Keeps included resources

### Run

```bash
python src/extraction/filter_included_resources.py
```

### Output

```text
data/processed/resources_included_only.csv
```

### Example Result

- Raw rows: 104
- Included rows: 74

---

# STEP 3 — Extract Taxonomy Dataset

**Script:** `src/extraction/extract_taxonomy_from_included.py`

### Run

```bash
python src/extraction/extract_taxonomy_from_included.py
```

### Output

```text
data/processed/taxonomy_raw.csv
```

---

# STEP 4 — Clean Taxonomy Labels

**Script:** `src/taxonomy/clean_taxonomy_labels.py`

### Run

```bash
python src/taxonomy/clean_taxonomy_labels.py
```

### Outputs

```text
data/processed/taxonomy_clean.csv
data/processed/normalisation_log.csv
data/processed/taxonomy_stats.json
```

---

# STEP 5 — Validate Links

**Script:** `src/extraction/validate_links.py`

### Run

```bash
python src/extraction/validate_links.py
```

### Outputs

```text
data/processed/resource_manifest.csv
data/processed/link_validation_log.csv
```

---

# STEP 6 — Filter Validated Extraction Set

**Script:** `src/extraction/filter_validated_resources.py`

### Run

```bash
python src/extraction/filter_validated_resources.py
```

### Output

```text
data/processed/resources_validated_for_extraction.csv
```

---

# STEP 7 — Extract Text From Resources

**Script:** `src/extraction/extract_text_from_validated.py`

### Run

```bash
python src/extraction/extract_text_from_validated.py
```

### Outputs

```text
data/processed/resources_text.jsonl
data/processed/extraction_manifest.csv
data/processed/text_by_id/
data/cache/
```

### Example Result

- Successful extractions: 64

---

# STEP 8 — Build Final Unified Dataset

**Script:** `src/dataset/build_unified_dataset.py`

### Run

```bash
python src/dataset/build_unified_dataset.py
```

### Outputs

```text
data/processed/resources_unified.csv
data/processed/resources_unified.jsonl
data/processed/unified_join_report.json
```

### Final Corpus Size

**64 resources**

---

# STEP 9 — Validate Ontology v1

```bash
python src/ontology/validate_ontology_v1.py
```

### Output

```text
data/processed/edi_ontology_v1_validation.json
```

---

# STEP 10 — Export Ontology To Turtle

```bash
python src/ontology/export_ontology_v1_to_ttl.py
```

### Output

```text
data/processed/edi_ontology_v1.ttl
```

---

# STEP 11 — Build Concept Index

```bash
python -m src.semantic_tagging.build_concept_index
```

---

# STEP 12 — Build Resource Embeddings

```bash
python -m src.semantic_tagging.embed_resources
```

### Outputs

```text
data/semantic/resource_embeddings_minilm.npz
data/semantic/resource_embeddings_mpnet.npz
```

---

# STEP 13 — Run Semantic Tagging

```bash
python -m src.semantic_tagging.run_semantic_tagging
```

### Outputs

```text
data/semantic/semantic_tags_minilm.jsonl
data/semantic/semantic_tags_mpnet.jsonl
```

---

# STEP 14 — Evaluate Semantic Tags

```bash
python -m src.semantic_tagging.evaluate_semantic_tags
```

---

# STEP 15 — Threshold Experiments

```bash
python -m src.semantic_tagging.threshold_experiments
```

---

# STEP 16 — Category Analysis

```bash
python -m src.semantic_tagging.category_analysis
```

---

# STEP 17 — Concept Frequency Analysis

```bash
python -m src.semantic_tagging.concept_frequency
```

---

# STEP 18 — Run Final Baselines

```bash
python -m src.retrieval.run_baselines \
--ontology-name v2_iter04 \
--taxonomy-csv data/processed/taxonomy_clean.csv \
--queries data/phase8_iterations/iter04/inputs/queries_v2_iter04_32.jsonl \
--out-dir data/phase8_iterations/iter04/runs/v2_iter04/edi_ontology_v2_iter04 \
--top-k 20
```

### Produces

- BM25 baseline
- Ontology-only baseline
- Ontology-text baseline

---

# STEP 19 — Run Final Hybrid Retrieval

```bash
python -m src.hybrid_search.hybrid_retrieval \
--ontology ontology/versions/v2_iter04/edi_ontology_v2_iter04.json \
--embeddings data/semantic/resource_embeddings_mpnet.npz \
--semantic_tags data/semantic/semantic_tags_mpnet.jsonl \
--queries data/phase8_iterations/iter04/inputs/queries_v2_iter04_32.jsonl \
--out_run data/phase8_iterations/iter04/runs/v2_iter04/hybrid_mpnet_a085_b015.jsonl \
--model_id sentence-transformers/all-mpnet-base-v2 \
--top_n 20 \
--alpha 0.85 \
--beta 0.15
```

---

# STEP 20 — Evaluate Hybrid Metrics

```bash
python -m src.evaluation.evaluate_hybrid \
--qrels data/phase8_iterations/iter04/qrels/qrels_iter04.json \
--run data/phase8_iterations/iter04/runs/v2_iter04/hybrid_mpnet_a085_b015.jsonl \
--out data/phase8_iterations/iter04/metrics_final/hybrid_metrics_iter04_recomputed.json \
--k 20
```

### Example Metrics

- P@20 = 0.154
- R@20 = 0.625
- F1@20 = 0.218
- MAP@20 = 0.293
- nDCG@20 = 0.469

---

# STEP 21 — Evaluate Coverage & Diversity

```bash
python -m src.evaluation.evaluate_coverage_diversity \
--results data/phase8_iterations/iter04/runs/v2_iter04/hybrid_mpnet_a085_b015.jsonl \
--resources data/processed/resources_unified.jsonl \
--k 20 \
--out data/phase8_iterations/iter04/metrics_final/hybrid_covdiv_iter04.json
```

### Example Metrics

- coverage@20 = 0.9921875
- diversity_entropy@20 = 0.8760

---

# STEP 22 — Run Final Web Application

```bash
streamlit run web/app.py
```

### Opens At

```text
http://localhost:8501
```

### Uses Final Assets

```text
ontology/versions/v2_iter04/edi_ontology_v2_iter04.json
data/processed/resources_unified.jsonl
data/semantic/resource_embeddings_minilm.npz
```

---

# Quick Start For Assessors

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd edi-hubplus-semantic-ir
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install openpyxl trafilatura streamlit watchdog
streamlit run web/app.py
```

---

# Final Dataset Summary

- Raw spreadsheet rows: 104
- Included resources: 74
- Validated for extraction: 70
- Successful text extractions: 64
- Final searchable corpus: **64**

---

# Troubleshooting

## Relative Import Error

Use:

```bash
python -m src.semantic_tagging.build_concept_index
```

not direct file execution.

## Missing Packages

```bash
pip install openpyxl trafilatura streamlit watchdog
```

## JSONDecodeError in Coverage Script

Use:

```bash
--resources data/processed/resources_unified.jsonl
```

not CSV.

## Streamlit Performance

```bash
pip install watchdog
```

---

# Notes For Assessors

- Final UI: `web/app.py`
- Final ontology: `v2_iter04`
- Final corpus: 64 resources
- Final hybrid weights: alpha=0.85 beta=0.15

---

# Author

Taleen Abubaker
Final Year Individual Project
University of Leeds
