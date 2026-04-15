from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SEMANTIC_DIR = DATA_DIR / "semantic"

# Ontology is at project root /ontology
ONTOLOGY_V1_JSON = PROJECT_ROOT / "ontology" / "edi_ontology_v1.json"

RESOURCES_UNIFIED_JSONL = PROCESSED_DIR / "resources_unified.jsonl"

# Outputs
SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)