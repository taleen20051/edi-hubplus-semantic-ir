# EDI Hub+ Semantic IR

### Ontology-Driven Semantic Tagging and Search for Equality, Diversity and Inclusion Resources

Final Year Individual Project – University of Leeds
Author: Taleen Abubaker

---

## Project Overview

This project develops an **ontology-guided semantic information retrieval system** for Equality, Diversity and Inclusion (EDI) resources.

Traditional keyword search often fails when users use different wording from the indexed resource vocabulary.

Examples:

- `religious support` vs `faith accommodation`
- `bias in grants` vs `funding discrimination`
- `neurodivergent staff support` vs `reasonable adjustments`

To address this, the system combines:

1. **Ontology-based retrieval** – structured EDI concepts and controlled vocabulary
2. **Semantic retrieval** – sentence embeddings for meaning-based matching
3. **Hybrid ranking** – combines symbolic and semantic evidence
4. **Explainable UI** – shows why results were returned

---

## Main Features

### Search Modes

#### Manual Semantic Search

- Natural language queries
- Semantic similarity ranking
- Ontology query expansion

#### Controlled Browse Search

- Browse by EDI category
- Browse by ontology labels
- Filtered ranked results

### Explainability

Each result can display:

- Matched ontology concepts
- Keyword evidence
- Semantic relevance score
- Why the result was retrieved

### Ontology Iteration Framework

Supports evaluation across ontology versions:

- v1 baseline
- v2 iterative refinements
- Controlled comparison experiments

---

## System Architecture

```text
User Query
   ↓
Query Processing
   ↓
Ontology Matching + Expansion
   ↓
Embedding Similarity Search
   ↓
Hybrid Ranking
   ↓
Explainable Results UI
```

## **Repository Structure**

edi-hubplus-semantic-ir/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── semantic/
│   └── phase8_iterations/
│
├── ontology/
│   ├── versions/
│   └── change_logs/
│
├── src/
│   ├── extraction/
│   ├── retrieval/
│   ├── evaluation/
│   ├── hybrid_search/
│   └── ontology/
│
├── web/
│   ├── app.py
│   └── styles.css
│
├── requirements.txt
└── README.md


## **Installation**

### **1. Clone Repository**

```
git clone https://github.com/taleen20051/edi-hubplus-semantic-ir.git
cd edi-hubplus-semantic-ir
```

### **2. Create Virtual Environment**

#### **macOS / Linux**

```
python3 -m venv .venv
source .venv/bin/activate
```

#### **Windows**

```
python -m venv .venv
.venv\Scripts\activate
```

### **3. Install Dependencies**

```
pip install -r requirements.txt
```

## **Run Web Application**

```
streamlit run web/app.py
```

Then open: http://localhost:8501

## **Core Technologies**

| **Component** | **Tool**              |
| ------------------- | --------------------------- |
| Language            | Python                      |
| UI                  | Streamlit                   |
| Semantic Embeddings | SentenceTransformers        |
| Parsing             | BeautifulSoup / Trafilatura |
| PDF Extraction      | PyMuPDF / pdfminer          |
| Data Handling       | Pandas / NumPy              |
| Evaluation          | Custom IR Metrics           |

## **Retrieval Models Implemented**

### **1. BM25 Baseline**

Traditional lexical ranking using exact term overlap.

### **2. Ontology-Only Retrieval**

Uses structured EDI concepts and label matching.

### **3. Hybrid Retrieval**

Combines: Final Score =
α × Semantic Similarity + β × Ontology Match Score

## **Evaluation Metrics**

The system was evaluated using:

* Precision@k
* Recall@k
* F1@k
* MAP
* nDCG
* Coverage@k
* Diversity Entropy

## **Final Dissertation Findings**

The final controlled experiments found:

* **Ontology-only retrieval achieved strongest top-ranked relevance**
* **Hybrid retrieval improved thematic breadth**
* Structured vocabularies were highly effective in domain-specific EDI search

This demonstrates that more complex AI models do not always outperform carefully designed symbolic systems.

## **Example Queries**

* trans identity support
* religion accommodations
* bullying microaggressions
* women leadership pipeline
* grant bias reduction
* inclusive policy reform

## **Data Source**

The project uses a curated institutional EDI Hub+ spreadsheet of publicly accessible resources including:

* PDFs
* HTML webpages
* Guidance documents
* Toolkits
* Policies
* Academic materials

## **Reproducibility**

All experiments use fixed:

* Corpus
* Query set
* Embedding model
* Evaluation metrics

This ensures controlled comparison between ontology versions and retrieval methods.

## **Notes for Assessors / Markers**

To run the final prototype:

`pip install -r requirements.txt
streamlit run web/app.py`

The interface demonstrates:

* Semantic search
* Ontology browsing
* Explainability
* Final refined ontology system

## **Future Improvements**

* Learned hybrid ranking
* Larger corpus
* User studies
* Adaptive weighting
* Expanded ontology coverage
* Deployment as live web platform

## **Academic Context**

This repository was developed as a Final Year Individual Project in Computer Science at the University of Leeds.

## **Author**

**Taleen Abubaker**

## **License**

Academic / Educational Use Only


## Full Experimental Reproduction Pipeline (Optional)

These commands reproduce the full dissertation pipeline from raw data to final evaluation outputs.

### Build Unified Dataset

```bash
python src/extraction/extract_sheet5_to_csv.py
python src/extraction/filter_included_resources.py
python src/extraction/validate_links.py
python src/extraction/extract_text_from_validated.py
python src/dataset/build_unified_dataset.py
```

### **Generate Embeddings**

`python -m src.semantic_tagging.embed_resources`

### **Run Baselines**

`python -m src.retrieval.run_baselines`

### **Run Hybrid Retrieval**

`python -m src.hybrid_search.hybrid_retrieval`

### **Evaluate Results**

`python -m src.evaluation.evaluate_hybrid
python -m src.evaluation.evaluate_coverage_diversity`
