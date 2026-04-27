from pathlib import Path

# Resolve the project root folder dynamically.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main data folders used across the semantic tagging pipeline.
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SEMANTIC_DIR = DATA_DIR / "semantic"

# Location of the baseline ontology file.
# Later ontology iterations are stored separately under ontology/versions/.
ONTOLOGY_V1_JSON = PROJECT_ROOT / "ontology" / "edi_ontology_v1.json"

# Final cleaned resource corpus used for semantic tagging.
RESOURCES_UNIFIED_JSONL = PROCESSED_DIR / "resources_unified.jsonl"

# Ensure the semantic output folder exists before saving files.
SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)