import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import streamlit as st

# Define the paths and project root for loading the application's fixed resource files.
BASE_DIR = Path(__file__).resolve().parents[1]

ONTOLOGY_FINAL_PATH = BASE_DIR / "ontology" / "versions" / "v2_iter04" / "edi_ontology_v2_iter04.json"
JOINED_RESOURCES_PATH = BASE_DIR / "data" / "processed" / "resources_unified.jsonl"
EMBED_PATH = BASE_DIR / "data" / "semantic" / "resource_embeddings_minilm.npz"

# The sentence transformer minilm embedding model was used for all lexical encoding.
# A semantic cutoff was chosen to normalise scores across queries to improve the interpretability of the application behaviour.
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.20

CATEGORY_BROWSE_OPTIONS = [
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
]

BROWSING_OPTIONS = [
    "Semantic score",
    "Resource ID",
    "Title (A–Z)",
]

UI_LOG_REPO = BASE_DIR / "data" / "ui_logs"
UI_LOG_REPO.mkdir(parents=True, exist_ok=True)
FLAG_FEEDBACK_PATH = UI_LOG_REPO / "reviewer_flags.jsonl"

# This collection of constants was set so the retrieval remains transparent with no hidden stopword lists or embedding-only thresholds. 
COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "such", "that", "the", "their",
    "then", "there", "these", "this", "those", "to", "was", "were", "will", "with",
    "after", "before", "between", "during", "over", "under", "within", "without",
}


# Simple helper functions for text canonicalisation, tokenisation, file I/O handling, and resource metadata formatting. 
# These improve retrieval flow and ensure consistent processing across retrieval and explainability components.
def normalise_text(s: str) -> str:
    """Handle text normalisation for matching by trimming, lowercasing, and ignoring whitespace."""
    return " ".join((s or "").strip().lower().split())

# Tokenise text using small punctuation deletion and stopword filtering.
def tokenize(s: str, *, min_len: int = 3) -> List[str]:
    s = normalise_text(s)
    if not s:
        return []

    for ch in [",", ".", ";", ":", "(", ")", "[", "]", "{", "}", "\"", "'", "/", "\\", "|", "!", "?", "-", "_"]:
        s = s.replace(ch, " ")

    return [t for t in s.split() if len(t) >= min_len and t not in COMMON_STOPWORDS]


# Function to verify whether query and document text share enough relevant tokens.
# This is used as a lightweight query filter to ensure a minimum level of keyword relevance even when semantic similarity is also being used.
def token_overlap_detect(query: str, text_blob: str, *, min_token_len: int = 4, min_hits: int = 2) -> bool:
    q_toks = tokenize(query, min_len=min_token_len)
    if not q_toks:
        return False

    blob_toks = set(tokenize(text_blob, min_len=min_token_len))
    hits = sum(1 for t in q_toks if t in blob_toks)
    required = 1 if len(q_toks) == 1 else min_hits
    return hits >= required


# Function to read the JSON files previously prepared for word retrieval in the UI.
def read_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# Function to read JSONL files previously executed, returning a list of related tags for each non-empty line in the search query.
def read_jsonl_file(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Function to insert a new line into the JSONL log file for human reviewer feedback.
def append_jsonl_file(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# Function used to check all resourse URLs are valid by including a HTTPS scheme when necessary.
def safe_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return "https://" + u


# Function to transform the type of resource metadata into a user facing label (PDF or Web page) based on link.
def resource_type_label(r: dict) -> str:
    t = (r.get("detected_type") or r.get("resource_type") or "").strip().lower()
    if "pdf" in t:
        return "PDF"
    if "html" in t or "web" in t:
        return "Web page"
    return t.upper() if t else "Resource"


# Give users the option to sort the retrieved results by resource ID in ascending order.
def rid_sort_key(x: str) -> int:
    try:
        return int(x)
    except Exception:
        return 10**9


# Function to help users view a text blob for each resource that joins title, organisation, and extracted text summary.
def build_search_blob(r: dict) -> str:
    title = r.get("title") or ""
    text = r.get("extracted_text") or ""
    org = r.get("organisation") or r.get("organization") or ""
    return normalise_text(f"{title} {org} {text}")


# -- Explainability helpers to support in-depth understanding of resource ranking and filtering behaviour. --

# Summarise the lexical evidence for a query-resource pair by checking for distinct phrase matches and token-level overlaps.
def lexical_match_details(query: str, text_blob: str) -> Dict[str, Any]:
    qn = normalise_text(query)
    if not qn:
        return {
            "phrase_hit": False,
            "overlap_hit": False,
            "matched_tokens": [],
            "query_tokens": [],
        }

    # Check for whole word or exact phrase matches from embedding/ontology to give an indication of direct relevance.
    def whole_word_or_phrase_hit(blob_text: str, query_norm: str) -> bool:
        if not query_norm:
            return False
        return re.search(rf"\b{re.escape(query_norm)}\b", blob_text) is not None

    q_tokens = tokenize(query, min_len=4)
    blob_tokens = set(tokenize(text_blob, min_len=4))
    matched_tokens = [t for t in q_tokens if t in blob_tokens]

    return {
        "phrase_hit": whole_word_or_phrase_hit(text_blob, qn),
        "overlap_hit": token_overlap_detect(query, text_blob, min_token_len=4, min_hits=2),
        "matched_tokens": matched_tokens,
        "query_tokens": q_tokens,
    }


# Function to improve user-facing features by explaining why resource was retrived based on tag matches, lexical evidence, and word similarity
def result_reason_summary(
    query: str,
    resource_blob: str,
    semantic_score: float,
    concept_matches: List[dict],
    *,
    semantic_cutoff: float,
) -> Tuple[List[str], List[str]]:
    """Display general explanation phrases and detailed lines for each result card."""
    lexical = lexical_match_details(query, resource_blob)
    reasons: List[str] = []
    detail_lines: List[str] = []

    if lexical["phrase_hit"]:
        reasons.append("Exact phrase / keyword match")
        detail_lines.append("The query appears directly in the title or extracted text.")
    elif lexical["overlap_hit"]:
        reasons.append("Keyword overlap")
        if lexical["matched_tokens"]:
            detail_lines.append("Shared informative words: " + ", ".join(lexical["matched_tokens"][:6]))
        else:
            detail_lines.append("The query shares multiple informative words with this resource.")

    if concept_matches:
        reasons.append("Ontology-assisted semantic search")
        detail_lines.append(
            "Matched ontology tags expanded the query before semantic retrieval: "
            + ", ".join(m["display_label"] for m in concept_matches[:6])
        )

    # Similarity score is shown for all resource results, but only affects retrieval if above the defined cutoff. 
    # This way users can see when similarity is detected but not strong enough to retrieve on its own.
    if semantic_score >= semantic_cutoff:
        reasons.append("Semantic similarity")
        detail_lines.append(f"The query embedding is similar to this resource embedding (score={semantic_score:.3f}).")

    if not reasons:
        reasons.append("Filtered match")
        detail_lines.append("This resource passed the retrieval filter.")

    seen = set()
    reasons = [r for r in reasons if not (r in seen or seen.add(r))]
    return reasons, detail_lines

# Display a message above the resoyrce list to help users understand why results were shown.
def summarise_query_state(query: str, concept_matches: List[dict], result_count: int) -> str:
    if not query.strip():
        return ""

    if concept_matches:
        labels = ", ".join(m["display_label"] for m in concept_matches[:6])
        if len(concept_matches) > 6:
            labels += ", …"
        return (
            f"The query matched ontology tags ({labels}). These tags were used to expand the query, "
            f"and results were then filtered and ranked using lexical evidence plus semantic similarity."
        )

    if result_count > 0:
        return (
            "No ontology tags matched this query specificity. Results are still shown because the system also uses "
            "semantic similarity between the query and document embeddings, as well as direct keyword evidence in the text."
        )

    return (
        "No ontology tags matched this query, and no resources passed the lexical or semantic filter. "
        "Try broader wording or related concepts."
    )


# Function to truncate and format related ontology terms for display as tags under each query output.
def format_matched_tags(concept_matches: List[dict], max_tags: int = 12) -> str:
    labels = [m["display_label"] for m in concept_matches if m.get("display_label")]
    labels = list(dict.fromkeys(labels))
    if not labels:
        return ""
    if len(labels) > max_tags:
        labels = labels[:max_tags] + ["…"]
    return ", ".join(labels)


# Cached loaders

# Streamlit interface caching is utilised to ensure accurate and fast response from system.
# 1. Decorator for load_ontology_concepts
@st.cache_data(show_spinner=False)
# Detect and load the final ontology JSON file and index terms by concept ID from specified path.
def load_ontology_concepts(ontology_path: str) -> Dict[str, dict]:
    ont = read_json_file(Path(ontology_path))
    return {c["id"]: c for c in ont.get("concepts", [])}


# 2. Decorator for load_resources
@st.cache_data(show_spinner=False)
# Load the final merged dataset and terms by resource ID from path.
def load_resources() -> Dict[str, dict]:
    rows = read_jsonl_file(JOINED_RESOURCES_PATH)
    out: Dict[str, dict] = {}
    for r in rows:
        rid = str(r.get("ID") or r.get("id") or r.get("rid") or "").strip()
        if rid:
            out[rid] = r
    return out


# 3. Decorator for load_resource_blobs
@st.cache_data(show_spinner=False)
# Preload search blobs for efficiency in retrieval for all resources computed in the dataset..
def load_resource_blobs(resources_by_id: Dict[str, dict]) -> Dict[str, str]:
    return {rid: build_search_blob(r) for rid, r in resources_by_id.items()}


# 4. Decorator for load_embeddings
@st.cache_data(show_spinner=False)
# Load the precomputed embeddings for all resources with their associated resouce IDs.
def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    z = np.load(EMBED_PATH, allow_pickle=True)
    emb = np.asarray(z["embeddings"], dtype=np.float32)
    ids = [str(x) for x in z["resource_ids"].tolist()]
    return emb, ids


# 5. Decorator for load_embed_model
@st.cache_resource(show_spinner=False)
# Load the sentence-transformer embedding model to ensure it is only loaded once and shared across all retrieval calls for efficiency.
# Store the model in Streamlit's dataset cache.
def load_embed_model() -> Any:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_ID)

# 6. Add warm_up_search_stack helper after load_embed_model

def warm_up_search_stack() -> None:
    """Preload the semantic search model once so manual search feels immediate."""
    load_embed_model()


# Ontology helper functions

# Function to relate resources to user search queries through ontology-defined tags (preferred and alternative labels).
def find_matching_concepts(query: str, concepts_by_id: Dict[str, dict]) -> List[dict]:
    q_raw = (query or "").strip()
    q = normalise_text(q_raw)
    if not q:
        return []

    q_tokens = tokenize(q_raw, min_len=4)
    matches: List[dict] = []

    for cid, c in concepts_by_id.items():
        pref = str(c.get("pref_label", "") or "").strip()
        if not pref:
            continue

        candidates: List[Tuple[str, str]] = [("pref", pref)]
        for alt in (c.get("alt_labels") or []):
            alt_s = str(alt or "").strip()
            if alt_s:
                candidates.append(("alt", alt_s))

        hit_type: Optional[str] = None
        hit_label: Optional[str] = None

        # Initially check for distinct term matches between search query and set ontology labels.
        for typ, lab in candidates:
            lab_n = normalise_text(lab)
            if lab_n and lab_n in q:
                hit_type = typ
                hit_label = lab
                break

        # If there are no exact term matches for retrieval, check for subtle partial matches.
        if not hit_type:
            for typ, lab in candidates:
                lab_n = normalise_text(lab)
                if lab_n and q in lab_n:
                    hit_type = "query_in_label"
                    hit_label = q_raw
                    break

        # If neither whole nor partial term matches are detected, check for token-level overlap to give any sign of relevance.
        if not hit_type and q_tokens:
            for _, lab in candidates:
                lab_tokens = set(tokenize(lab, min_len=4))
                hits = [t for t in q_tokens if t in lab_tokens]
                required = 1 if len(q_tokens) == 1 else 2
                if len(hits) >= required:
                    hit_type = "token_overlap"
                    hit_label = ", ".join(hits)
                    break

        # Only retrieve resources for concepts with one of the above conditions to ensure a non-excessive recall result set.
        if hit_type and hit_label:
            matches.append(
                {
                    "concept_id": cid,
                    "display_label": pref,
                    "matched_label": hit_label,
                    "match_type": hit_type,
                }
            )

    uniq: Dict[str, dict] = {}
    for m in matches:
        uniq[m["concept_id"]] = m

    out = list(uniq.values())
    out.sort(key=lambda x: (x["display_label"] or "").lower())
    return out


# Function to improve recall by expanding the query without overly diluting it with irrelevant terms.
# This is executed before encoding to check if the query can be improved by adding related ontology terms that may not be present in the original query.
def build_expanded_query(query: str, concept_matches: List[dict], concepts_by_id: Dict[str, dict]) -> str:
    parts: List[str] = [query.strip()]

    for m in concept_matches:
        cid = m["concept_id"]
        c = concepts_by_id.get(cid, {})
        pref = str(c.get("pref_label", "") or "").strip()
        if pref:
            parts.append(pref)
        for alt in (c.get("alt_labels") or []):
            alt_s = str(alt or "").strip()
            if alt_s:
                parts.append(alt_s)

    seen = set()
    deduped: List[str] = []
    for p in parts:
        k = normalise_text(p)
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(p.strip())

    return " ".join(deduped).strip()


# Check for precise or partial matches to group ontology defined pref labels under the four main categories.
def get_category_to_labels(concepts_by_id: Dict[str, dict]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {cat: [] for cat in CATEGORY_BROWSE_OPTIONS}
    for c in concepts_by_id.values():
        category_id = str(c.get("category_id", "") or "").strip()
        pref = str(c.get("pref_label", "") or "").strip()
        if not pref:
            continue

        # Define the main categories for mapping.
        if category_id == "individual_characteristics":
            out["Individual Characteristics"].append(pref)
        elif category_id == "career_pathway":
            out["Career Pathway"].append(pref)
        elif category_id == "research_funding_process":
            out["Research Funding Process"].append(pref)
        elif category_id == "organisational_culture":
            out["Organisational Culture"].append(pref)

    for cat in out:
        out[cat] = sorted(list(dict.fromkeys(out[cat])), key=lambda x: x.lower())
    return out



# --- Enhanced label surface matching for ontology-driven filtering ---

def concept_surface_labels(selected_label: str, concepts_by_id: Dict[str, dict]) -> List[str]:
    """Return the preferred and alternative labels for the selected ontology label."""
    selected_norm = normalise_text(selected_label)
    if not selected_norm or selected_label == "All labels":
        return []

    surfaces: List[str] = [selected_label]
    for concept in concepts_by_id.values():
        labels = [str(concept.get("pref_label", "") or "")]
        labels.extend(str(x or "") for x in (concept.get("alt_labels") or []))

        if any(normalise_text(label) == selected_norm for label in labels):
            for label in labels:
                label = label.strip()
                if label:
                    surfaces.append(label)

    return list(dict.fromkeys(surfaces))


def resource_matches_selected_label(
    resource: dict,
    selected_category: str,
    selected_label: str,
    concepts_by_id: Dict[str, dict],
) -> bool:
    """Check whether a resource belongs to the selected browse label.

    The browse dropdown uses ontology labels, but the resource file stores the
    original spreadsheet tags. To avoid false zero-result filters, this checks
    the selected label together with its ontology preferred/alternative labels.
    """
    vals = resource.get(selected_category, [])
    if isinstance(vals, str):
        vals = [v.strip() for v in vals.split(",") if v.strip()]
    if not isinstance(vals, list):
        return False

    accepted = {normalise_text(label) for label in concept_surface_labels(selected_label, concepts_by_id)}
    accepted = {label for label in accepted if label}
    return any(normalise_text(str(v)) in accepted for v in vals)


# New function: resource_matches_browse_label
def resource_matches_browse_label(
    resource: dict,
    resource_blob: str,
    selected_category: str,
    selected_label: str,
    concepts_by_id: Dict[str, dict],
    semantic_score: float,
    semantic_cutoff: float,
) -> bool:
    """Apply a practical Browse label filter.

    Browse labels come from the ontology, while resource tags come from the
    original spreadsheet. Therefore, the label should not be treated only as an
    exact spreadsheet-tag filter. A resource is kept if it matches the selected
    label through its stored tags, direct text evidence, or semantic similarity.
    """
    if selected_label == "All labels":
        return True

    if resource_matches_selected_label(resource, selected_category, selected_label, concepts_by_id):
        return True

    label_norm = normalise_text(selected_label)
    if label_norm and re.search(rf"\b{re.escape(label_norm)}\b", resource_blob):
        return True

    if token_overlap_detect(selected_label, resource_blob, min_token_len=4, min_hits=1):
        return True

    return semantic_score >= semantic_cutoff


# Introduce a browse option for each category and label for users unfamiliar with the field terms.
def get_browse_semantic_query(selected_category: str, selected_label: str) -> str:
    if selected_label != "All labels":
        return selected_label.strip()
    if selected_category != "All categories":
        return selected_category.strip()
    return ""


# Transform labels into filters for retrieval so browsing behaves similarly to manual search option.
def get_browse_filters(selected_category: str, selected_label: str) -> Tuple[str, str]:
    """
    Browse mode applies both selected controls as filters. The category narrows
    the collection first, then the selected label narrows it further. The label
    is also used as the semantic query so the remaining resources can still be
    ranked by similarity.
    """
    return selected_category, selected_label


# Retrieval function

# Retrieval joins ontology-driven label ranking with a lightweight lexical fallback.
def rank_and_filter_resources(
    query: str,
    concepts_by_id: Dict[str, dict],
    resources_by_id: Dict[str, dict],
    resource_blobs: Dict[str, str],
    resource_embeddings: np.ndarray,
    resource_ids: List[str],
    *,
    semantic_cutoff: float = 0.25,
    filter_category: str = "All categories",
    filter_label: str = "All labels",
) -> Tuple[List[str], List[dict], Dict[str, float]]:
    # Retrieve, filter, and rank resources for both Search and Browse modes.
    q = (query or "").strip()

    # Phase 1: compute semantic scores if a query is initiated.
    concept_matches: List[dict] = []
    if q:
        concept_matches = find_matching_concepts(q, concepts_by_id)
        expanded_query = build_expanded_query(q, concept_matches, concepts_by_id)

        model = load_embed_model()
        q_vec = model.encode([expanded_query], normalize_embeddings=True)[0]
        q_vec = np.asarray(q_vec, dtype=np.float32)
        sims = np.asarray(resource_embeddings @ q_vec, dtype=np.float32)
    else:
        sims = np.zeros(len(resource_ids), dtype=np.float32)

    # Phase 2: align similarity scores back to resource IDs for display options.
    semantic_scores_by_rid: Dict[str, float] = {}
    for i, rid in enumerate(resource_ids):
        rid_s = str(rid)
        if rid_s in resources_by_id:
            semantic_scores_by_rid[rid_s] = float(sims[i])

    qn = normalise_text(q)
    dyn_cutoff = semantic_cutoff
    if q:
        q_words = [w for w in qn.split() if w]
        if len(q_words) <= 1:
            dyn_cutoff = max(dyn_cutoff, 0.35)

    # Helper function to check for matching from embeddings and ontology expansion.
    def whole_word_or_phrase_hit(blob_text: str, query_norm: str) -> bool:
        if not query_norm:
            return False
        return re.search(rf"\b{re.escape(query_norm)}\b", blob_text) is not None

    # Phase 3: collect retrieval terms using lexical evidence and/or semantic similarity.
    browse_filter_active = filter_category != "All categories" or filter_label != "All labels"

    candidate_rids: List[str] = []
    for rid, r in resources_by_id.items():
        if not q or browse_filter_active:
            candidate_rids.append(rid)
            continue

        blob = resource_blobs.get(rid, "")
        lexical_hit = whole_word_or_phrase_hit(blob, qn) or token_overlap_detect(q, blob, min_token_len=4, min_hits=2)
        sem = semantic_scores_by_rid.get(rid, 0.0)

        if lexical_hit or sem >= dyn_cutoff:
            candidate_rids.append(rid)

    # Phase 4: implement controlled Browse filters after retrieval so Search remains broad but contained.
    filtered: List[str] = []
    for rid in candidate_rids:
        r = resources_by_id[rid]

        if filter_category != "All categories":
            vals = r.get(filter_category, [])
            has_category_value = False
            if isinstance(vals, list):
                has_category_value = len(vals) > 0
            elif isinstance(vals, str):
                has_category_value = bool(vals.strip())
            if not has_category_value:
                continue

        if filter_label != "All labels":
            blob = resource_blobs.get(rid, "")
            sem = semantic_scores_by_rid.get(rid, 0.0)

            if filter_category == "All categories":
                found_any = False
                for category_name in CATEGORY_BROWSE_OPTIONS:
                    if resource_matches_browse_label(
                        r,
                        blob,
                        category_name,
                        filter_label,
                        concepts_by_id,
                        sem,
                        dyn_cutoff,
                    ):
                        found_any = True
                        break
                if not found_any:
                    continue
            else:
                if not resource_matches_browse_label(
                    r,
                    blob,
                    filter_category,
                    filter_label,
                    concepts_by_id,
                    sem,
                    dyn_cutoff,
                ):
                    continue

        filtered.append(rid)

    # Phase 5: sort by semantic score when submitting a query, otherwise keep a stable browse order.
    if q:
        filtered.sort(key=lambda rid: (-semantic_scores_by_rid.get(rid, 0.0), rid_sort_key(rid)))
    else:
        filtered.sort(key=rid_sort_key)

    return filtered, concept_matches, semantic_scores_by_rid


# Rendering

# Rendering functions keep presentation separate from retrieval logic so the app flow is easier to follow.
def construct_resource_card(
    r: dict,
    *,
    query: str = "",
    resource_blob: str = "",
    semantic_score: Optional[float] = None,
    concept_matches: Optional[List[dict]] = None,
    semantic_cutoff: float = 0.20,
    show_reasoning: bool = True,
) -> None:
    """Render one resource entry with metadata, score, snippet, and optional explanation."""
    title = (r.get("title") or "(no title)").strip()
    url = safe_url(r.get("url") or "")
    typ = resource_type_label(r)

    org = (r.get("organisation") or r.get("organization") or "").strip()
    date = (r.get("date") or r.get("published") or r.get("published_date") or "").strip()

    st.markdown(f"### {title}")

    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        st.markdown(f"**Type:** {typ}")
    with c2:
        if semantic_score is not None:
            st.markdown(f"**Semantic score:** {semantic_score:.3f}")
    with c3:
        if url:
            st.link_button("Open resource", url)

    meta_parts = []
    if org:
        meta_parts.append(org)
    if date:
        meta_parts.append(date)
    if meta_parts:
        st.caption(" • ".join(meta_parts))

    if show_reasoning and query.strip():
        reasons, detail_lines = result_reason_summary(
            query=query,
            resource_blob=resource_blob,
            semantic_score=float(semantic_score or 0.0),
            concept_matches=concept_matches or [],
            semantic_cutoff=semantic_cutoff,
        )
        st.markdown("**Why this result appeared:** " + " | ".join(reasons))
        with st.expander("Show explanation"):
            for line in detail_lines:
                st.write("- " + line)

    text = (r.get("extracted_text") or "").strip()
    if text:
        snippet = text[:350] + ("…" if len(text) > 350 else "")
        st.write(snippet)

    st.divider()


def render_resource_list(
    *,
    title: str,
    subtitle: str,
    resources_by_id: Dict[str, dict],
    resource_blobs: Dict[str, str],
    resource_ids_to_show: List[str],
    semantic_scores: Dict[str, float],
    query: str = "",
    concept_matches: Optional[List[dict]] = None,
    semantic_cutoff: float = 0.20,
    show_reasoning: bool = False,
    show_scores: bool = False,
) -> None:
    """Render a complete list of resource cards for the current mode and state."""
    st.subheader(title)
    st.caption(subtitle)

    if not resource_ids_to_show:
        st.warning("No matching resources were found.")
        return

    for rid in resource_ids_to_show:
        r = resources_by_id.get(rid)
        if r:
            construct_resource_card(
                r,
                query=query,
                resource_blob=resource_blobs.get(rid, ""),
                semantic_score=semantic_scores.get(rid) if show_scores else None,
                concept_matches=concept_matches or [],
                semantic_cutoff=semantic_cutoff,
                show_reasoning=show_reasoning,
            )


def apply_browse_sort(
    filtered_rids: List[str],
    resources_by_id: Dict[str, dict],
    semantic_scores: Dict[str, float],
    browse_sort: str,
    effective_query: str,
) -> List[str]:
    """Apply the Browse-mode sort option to an already filtered result list."""
    out = list(filtered_rids)

    if browse_sort == "Title (A–Z)":
        out.sort(key=lambda rid: (resources_by_id[rid].get("title") or "").strip().lower())
    elif browse_sort == "Resource ID":
        out.sort(key=rid_sort_key)
    else:
        if effective_query:
            out.sort(key=lambda rid: (-semantic_scores.get(rid, 0.0), rid_sort_key(rid)))
        else:
            out.sort(key=rid_sort_key)

    return out


# UI using Streamlit

# The interface introduces two complementary workflows: broad semantic search and controlled ontology browsing.

st.set_page_config(page_title="EDI Hub+ Resources", layout="wide")
# Load the custom Streamlit styling from a separate stylesheet so the app logic
# remains focused on retrieval and interaction behaviour.
STYLE_PATH = BASE_DIR / "web" / "styles.css"
if STYLE_PATH.exists():
    st.markdown(
        f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# 7. App startup: replace block with warm_up_search_stack call after variables
concepts_v2 = load_ontology_concepts(str(ONTOLOGY_FINAL_PATH))
resources_by_id = load_resources()
resource_blobs = load_resource_blobs(resources_by_id)
resource_embeddings, resource_ids = load_embeddings()
category_to_labels = get_category_to_labels(concepts_v2)
semantic_cutoff = SIMILARITY_THRESHOLD

# Preload the embedding model once at app startup so the first search does not feel delayed.
warm_up_search_stack()

# Session stability used to keep search query interface consistent across user interactions and mode switches.
if "active_search_query" not in st.session_state:
    st.session_state["active_search_query"] = ""
if "search_box_version" not in st.session_state:
    st.session_state["search_box_version"] = 0

st.title("EDI Hub+ Resources")
st.caption(
    "Explore the final ontology-driven resource collection through two clear modes: "
    "manual semantic search and controlled ontology browsing."
)

ui_mode = st.radio(
    "Mode",
    options=["Search", "Browse"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("")

if ui_mode == "Search":
    # Search mode: expanded retrieval over the full collection using a free-text query.
    st.markdown("### Search")
    st.caption(
        "Use the manual search box to look for related results across the full collection. "
        "This mode enables open-ended search and returns resources based on semantic similarity and keyword evidence, "
        "even if they are not specifically tagged under a defined main category."
    )

    current_search_box_key = f"search_box_text_{st.session_state['search_box_version']}"

    with st.form("search_form", clear_on_submit=False):
        manual_input = st.text_input(
            "Manual search",
            key=current_search_box_key,
            placeholder="e.g., neurodivergent staff support, return to work after maternity, religious accommodation",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            search_submitted = st.form_submit_button("Search", use_container_width=True)
        with c2:
            reset_search = st.form_submit_button("Clear search", use_container_width=True)

    if reset_search:
        st.session_state["active_search_query"] = ""
        st.session_state["search_box_version"] += 1
        st.rerun()
    elif search_submitted:
        st.session_state["active_search_query"] = manual_input.strip()

    ongoing_query = st.session_state["active_search_query"]

    # If no query is made, display the full resource list before filtering.
    if not ongoing_query:
        all_rids = sorted(resources_by_id.keys(), key=rid_sort_key)
        render_resource_list(
            title="All resources",
            subtitle=f"Showing all {len(all_rids)} EDI related resources.",
            resources_by_id=resources_by_id,
            resource_blobs=resource_blobs,
            resource_ids_to_show=all_rids,
            semantic_scores={},
            query="",
            concept_matches=[],
            semantic_cutoff=semantic_cutoff,
            show_reasoning=False,
            show_scores=False,
        )
    else:
        # Build the retrieval output set by checking through all hybrid filters and ranks.
        filtered_rids, concept_matches, semantic_scores = rank_and_filter_resources(
            query=ongoing_query,
            concepts_by_id=concepts_v2,
            resources_by_id=resources_by_id,
            resource_blobs=resource_blobs,
            resource_embeddings=resource_embeddings,
            resource_ids=resource_ids,
            semantic_cutoff=semantic_cutoff,
            filter_category="All categories",
            filter_label="All labels",
        )

        # Display ontology tags related to user search query for retrieval mapping transparency.
        matched_tags_str = format_matched_tags(concept_matches)
        if matched_tags_str:
            st.markdown(f"**Matched ontology tags:** {matched_tags_str}")
        else:
            st.markdown("**Matched ontology tags:** _none_")

        state_msg = summarise_query_state(ongoing_query, concept_matches, len(filtered_rids))
        if state_msg:
            st.info(state_msg)

        # Show all results from unified resourse set that passed all filters with distinct score and textual reasoning.
        render_resource_list(
            title="Search results",
            subtitle=f"{len(filtered_rids)} matching resources for “{ongoing_query}”.",
            resources_by_id=resources_by_id,
            resource_blobs=resource_blobs,
            resource_ids_to_show=filtered_rids,
            semantic_scores=semantic_scores,
            query=ongoing_query,
            concept_matches=concept_matches,
            semantic_cutoff=semantic_cutoff,
            show_reasoning=True,
            show_scores=True,
        )

else:
    # Browse mode: general exploration where category filters the collection and labels guide semantic scoring.
    st.markdown("### Browse")
    st.caption(
        "Use ontology-based labels to explore the collection. "
        "Selecting a main category restricts results to that tag group, "
        "while selecting a label narrows the results using tag evidence, text evidence, and semantic similarity."
    )

    c1, c2, c3 = st.columns([1, 1, 1])

    # Define all label options and category-label matches for search and browse modes.
    with c1:
        selected_category = st.selectbox(
            "Main category",
            options=["All categories"] + CATEGORY_BROWSE_OPTIONS,
            index=0,
        )

    if selected_category == "All categories":
        label_options = ["All labels"]
    else:
        label_options = ["All labels"] + category_to_labels.get(selected_category, [])

    with c2:
        selected_label = st.selectbox(
            "Label",
            options=label_options,
            index=0,
        )

    with c3:
        browse_sort = st.selectbox(
            "Browse sort",
            options=BROWSING_OPTIONS,
            index=0,
        )

    effective_browse_query = get_browse_semantic_query(selected_category, selected_label)
    browse_filter_category, browse_filter_label = get_browse_filters(selected_category, selected_label)

    # Run the final retrieval for browse mode based on the selected category and label filters, 
    # then apply the chosen sort option to the filtered results.
    filtered_rids, concept_matches, semantic_scores = rank_and_filter_resources(
        query=effective_browse_query,
        concepts_by_id=concepts_v2,
        resources_by_id=resources_by_id,
        resource_blobs=resource_blobs,
        resource_embeddings=resource_embeddings,
        resource_ids=resource_ids,
        semantic_cutoff=semantic_cutoff,
        filter_category=browse_filter_category,
        filter_label=browse_filter_label,
    )

    filtered_rids = apply_browse_sort(
        filtered_rids=filtered_rids,
        resources_by_id=resources_by_id,
        semantic_scores=semantic_scores,
        browse_sort=browse_sort,
        effective_query=effective_browse_query,
    )

    # Display user facing text for enhanced transparency.
    active_filters = []
    if selected_category != "All categories":
        active_filters.append(f"Category filter: {selected_category}")
    if selected_label != "All labels":
        active_filters.append(f"Related label: {selected_label}")

    if not active_filters:
        subtitle = f"Showing all {len(filtered_rids)} included resources."
    else:
        subtitle = f"{len(filtered_rids)} resources shown with " + " | ".join(active_filters) + "."

    if selected_category != "All categories":
        st.info(
            "Browse mode applies a category filter, so recall may be weaker than in manual search. "
            "Manual search explores the full collection, while browsing focuses on specific tag subsets."
        )

    # Enable enhanced explanation features in browse mode.
    if effective_browse_query:
        matched_tags_str = format_matched_tags(concept_matches)
        st.markdown(f"**Browse ranking query:** {effective_browse_query}")
        st.caption(
            "In Browse mode, the selected category filters the collection first. "
            "The selected label then narrows the results using stored tags, text evidence, and semantic similarity."
        )
        if matched_tags_str:
            st.markdown(f"**Matched ontology tags:** {matched_tags_str}")

    # Show all outputs from unified resource dataset that passed all filters with similarity score and textual reasoning.
    render_resource_list(
        title="Browse results",
        subtitle=subtitle,
        resources_by_id=resources_by_id,
        resource_blobs=resource_blobs,
        resource_ids_to_show=filtered_rids,
        semantic_scores=semantic_scores,
        query=effective_browse_query,
        concept_matches=concept_matches,
        semantic_cutoff=semantic_cutoff,
        show_reasoning=bool(effective_browse_query),
        show_scores=bool(effective_browse_query),
    )

 # Optional reviewer logging function to support minimal human feedback during system testing and stakeholder demonstrations.
with st.expander("Reviewer feedback (optional)"):
    flag_rid = st.text_input("Resource ID to flag", placeholder="e.g., 23")
    note = st.text_area("Note", placeholder="e.g., irrelevant result, missing tag, extraction issue...")
    if st.button("Save flag"):
        rid = (flag_rid or "").strip()
        if rid and rid in resources_by_id:
            log_entry = {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "mode": ui_mode,
                "query": st.session_state.get("active_search_query", ""),
                "resource_id": rid,
                "note": (note or "").strip(),
            }
            append_jsonl_file(FLAG_FEEDBACK_PATH, log_entry)
            st.success("Saved flag.")
        else:
            st.error("Please enter a valid Resource ID from the results list.")